"""Convert Mecka "mono" episodes (per-episode JSON pose files) to EgoVerse zarr.

Unlike the RL2 format read by ``mecka_to_zarr.py`` (hands.csv + egomotion.txt,
hand keypoints in the camera frame), a mono episode ships:

  video_1      1080p30 mp4
  hand_pose    JSON, per frame: both hands' 21 world-frame landmarks and the
               camera pose ``camera_transform`` (wTc, same axis convention as
               RL2 egomotion); ``landmarks_2d`` is unusable and ignored
  intrinsics   JSON {w, h, fl_x, fl_y, cx, cy, k1, k2, p1, p2}
  body_pose / pelvis_pose   unused

The output zarr has the same keys, embodiment and conventions (post-fix left
wrist) as the RL2 converter, so the existing human keypoint keymaps apply.

The master JSON and manifest are private and live outside the repo; this script
only takes their paths.

  python -m egomimic.scripts.mecka_process.mono.mecka_mono_to_zarr \
      --master <master.json> --manifest <manifest.csv> \
      --raw-dir <raw> --out-dir <zarr> --workers 48 [--shard i/n]
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment.embodiment import EMBODIMENT
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.mecka_process.mecka_to_zarr import (
    MeckaExtractor,
    compute_hand_pose_xyzquat,
    download_with_retry,
)

logger = logging.getLogger(__name__)

IMG_W, IMG_H = 640, 360
FPS = 30
RAW_FILES = {
    "video_1": "video.mp4",
    "hand_pose": "hand_pose.json",
    "intrinsics": "intrinsics.json",
}
# Per-frame camera motion above these is a capture glitch (pose and video jump together).
GLITCH_STEP_M = 0.05
GLITCH_ROT_DEG = 10.0


def load_hand_pose(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """World-frame keypoints (T, 2, 21, 3) [left, right], RL2-style egomotion
    rows (T, 11) holding the camera position at 1:4 and quat xyzw at 7:11, and
    the video frame index of row 0."""
    h = json.load(open(path))
    # Frame keys are video frame indices; some episodes start tracking a few
    # frames in.
    idx = sorted(int(k) for k in h["frames"])
    first, T = idx[0], len(idx)
    if idx[-1] - first + 1 != T:
        raise ValueError(f"hand_pose frames {first}..{idx[-1]} have gaps ({T} present)")
    kp = np.zeros((T, 2, 21, 3))
    ego = np.zeros((T, 11))
    for i in range(T):
        f = h["frames"][str(first + i)]
        c = f["camera_transform"]
        ego[i, 1:4] = [c["position"][a] for a in "xyz"]
        ego[i, 7:11] = [c["orientation"][a] for a in "xyzw"]
        for hand in f["hands"]:
            if not hand.get("landmarks"):
                continue
            side = 0 if hand["hand_type"] == "LEFT" else 1
            pts = sorted(hand["landmarks"], key=lambda p: p["landmark_index"])
            kp[i, side] = [[p["x"], p["y"], p["z"]] for p in pts]
    return kp, ego, first


def hand_poses(kp_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 14) hand and wrist poses [left7, right7], xyz + quat wxyz.

    The hand frame is built from keypoint differences and cross products, so
    computing it on world keypoints equals the RL2 path (camera frame, then wTc).
    """
    T = len(kp_world)
    hand = np.zeros((T, 14))
    wrist = np.zeros((T, 14))
    for i in range(T):
        for side in (0, 1):
            h, w = compute_hand_pose_xyzquat(kp_world[i, side], side)
            hand[i, side * 7 : side * 7 + 7] = h
            wrist[i, side * 7 : side * 7 + 7] = w
    return hand, wrist


def read_video(path: Path, max_frames: int, skip: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    for _ in range(skip):
        if not cap.grab():
            break
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def scaled_intrinsics(k: dict) -> list[list[float]]:
    sx, sy = IMG_W / k["w"], IMG_H / k["h"]
    return [
        [k["fl_x"] * sx, 0.0, k["cx"] * sx, 0.0],
        [0.0, k["fl_y"] * sy, k["cy"] * sy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]


def motion_qa(ego: np.ndarray, kp_world: np.ndarray) -> dict:
    t = ego[:, 1:4]
    q = ego[:, 7:11]
    step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    rot = np.degrees(
        (Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()).magnitude()
    )
    wrist_step = np.linalg.norm(np.diff(kp_world[:, :, 0], axis=0), axis=-1)
    glitch = (step > GLITCH_STEP_M) | (rot > GLITCH_ROT_DEG)
    return {
        "glitch_frames": int(glitch.sum()),
        "max_cam_step_m": float(step.max(initial=0)),
        "max_cam_rot_deg": float(rot.max(initial=0)),
        "frozen_cam_frames": int(((step < 1e-7) & (rot < 1e-5)).sum()),
        "max_wrist_step_m": float(wrist_step.max(initial=0)),
        "zero_hand_frames": int(np.all(kp_world == 0, axis=(2, 3)).any(1).sum()),
    }


def download(ep: dict, raw: Path) -> None:
    """Idempotent; each file lands via rename so a concurrent or interrupted
    downloader never leaves a truncated file under its final name."""
    if (raw / ".complete").exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    for key, name in RAW_FILES.items():
        if (raw / name).exists():
            continue
        part = raw / f"{name}.part{os.getpid()}"
        download_with_retry(ep[key], part)
        os.replace(part, raw / name)
    (raw / ".complete").touch()


def convert(ep: dict, row: dict, raw: Path, out_dir: Path) -> dict:
    kp, ego, first = load_hand_pose(raw / "hand_pose.json")
    images = read_video(raw / "video.mp4", len(kp), skip=first)
    T = min(len(images), len(kp))
    if T < FPS:
        raise ValueError(f"only {T} usable frames")
    kp, ego, images = kp[:T], ego[:T], images[:T]

    hand, wrist = hand_poses(kp)
    numeric = {
        "left.obs_ee_pose": hand[:, :7],
        "right.obs_ee_pose": hand[:, 7:],
        "left.obs_wrist_pose": wrist[:, :7],
        "right.obs_wrist_pose": wrist[:, 7:],
        "left.obs_keypoints": kp[:, 0].reshape(T, 63),
        "right.obs_keypoints": kp[:, 1].reshape(T, 63),
        "obs_head_pose": MeckaExtractor._extract_head_poses(ego),
    }
    qa = motion_qa(ego, kp)
    intr = json.load(open(raw / "intrinsics.json"))
    meta = {
        "episode_id": ep["episode_id"],
        "user_id": ep["user_id"],
        "duration": ep["duration"],
        "environment_id": ep["environment_id"],
        "scene_id": ep["scene_id"],
        "task_id": ep["task_id"],
        "task": row["category"],
        "lab": "mecka",
        "source": "mecka_mono",
        "raw_intrinsics": intr,
        "first_video_frame": first,
        "qa": qa,
    }
    final = out_dir / f"{ep['episode_id']}.zarr"
    # Outside out_dir: LocalEpisodeResolver treats every directory there as an episode.
    tmp = out_dir.parent / f"{out_dir.name}_tmp" / f"{ep['episode_id']}.zarr"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    ZarrWriter.create_and_write(
        episode_path=tmp,
        numeric_data=numeric,
        image_data={"images.front_1": images},
        embodiment=EMBODIMENT.HUMAN_BIMANUAL.name.lower(),
        fps=FPS,
        task_name=row["category"],
        task_description=ep["task_description"],
        annotations=[(ep["task_description"], 0, T)],
        intrinsics={"front_1": scaled_intrinsics(intr)},
        metadata_override=meta,
    )
    tmp.rename(final)
    return {"frames": T, **qa}


def work(args: tuple) -> dict:
    ep, row, raw_root, out_dir, keep_raw, download_only = args
    cv2.setNumThreads(2)
    eid = ep["episode_id"]
    t0 = time.time()
    try:
        raw = raw_root / eid
        download(ep, raw)
        if download_only:
            return {"episode_id": eid, "ok": True, "sec": time.time() - t0}
        res = convert(ep, row, raw, out_dir)
        if not keep_raw:
            (raw / "video.mp4").unlink(missing_ok=True)
        return {"episode_id": eid, "ok": True, "sec": time.time() - t0, **res}
    except Exception as e:
        return {
            "episode_id": eid,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=3),
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--shard", default="0/1", help="i/n: this process's slice")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include-off-task", action="store_true")
    p.add_argument("--keep-raw-video", action="store_true")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    rows = {r["episode_id"]: r for r in csv.DictReader(open(args.manifest))}
    episodes = [
        e
        for e in json.load(open(args.master))["episodes"]
        if e["episode_id"] in rows
        and (args.include_off_task or rows[e["episode_id"]]["off_task"] == "0")
    ]
    i, n = map(int, args.shard.split("/"))
    episodes = sorted(episodes, key=lambda e: e["episode_id"])[i::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        e
        for e in episodes
        if not (args.out_dir / f"{e['episode_id']}.zarr").exists()
        and not (
            args.download_only
            and (args.raw_dir / e["episode_id"] / ".complete").exists()
        )
    ][: args.limit]
    print(
        f"shard {args.shard}: {len(episodes)} episodes, {len(todo)} to convert",
        flush=True,
    )

    stem = "download_log" if args.download_only else "convert_log"
    log_dir = args.out_dir.parent / f"{args.out_dir.name}_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{stem}.shard{i}of{n}.jsonl"
    jobs = [
        (
            e,
            rows[e["episode_id"]],
            args.raw_dir,
            args.out_dir,
            args.keep_raw_video,
            args.download_only,
        )
        for e in todo
    ]
    ok = fail = 0
    with mp.get_context("spawn").Pool(args.workers) as pool, open(log_path, "a") as log:
        for res in pool.imap_unordered(work, jobs):
            log.write(json.dumps(res) + "\n")
            log.flush()
            ok += res["ok"]
            fail += not res["ok"]
            if (ok + fail) % 50 == 0 or not res["ok"]:
                print(
                    f"{ok + fail}/{len(jobs)} ok={ok} fail={fail}"
                    + ("" if res["ok"] else f"  {res['episode_id']}: {res['error']}"),
                    flush=True,
                )
    print(f"done: ok={ok} fail={fail}; log {log_path}", flush=True)


if __name__ == "__main__":
    main()
