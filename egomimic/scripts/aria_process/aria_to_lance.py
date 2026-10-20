"""Convert one raw Aria recording (VRS + MPS) into a delivery: per-frame parquet plus MP4s.

Unlike aria_to_zarr, every RGB frame is kept; untracked hands, missing poses and timing
gaps become validity flags instead of deleted rows. The committer
(egomimic/scripts/lance_pilot/commit.py) appends deliveries to the Lance tables.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps.utils import get_nearest_hand_tracking_result
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

from egomimic.rldb.embodiment.embodiment import END_EFFECTOR
from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
from egomimic.rldb.lance.schema import HUMAN_STEPS, recording_id
from egomimic.rldb.lance.timeline import build_timeline, nearest_dt, pose_outliers
from egomimic.scripts.aria_process.aria_utils import (
    AriaVRSExtractor,
    convert_aria_keypoints_to_mano,
    mano_keypoints_to_cartesian,
    slam_to_rgb,
)

FPS = 30.0
RGB = StreamId("214-1")
SENTINEL = 1e8
# A pose sample further than this from the frame time is treated as missing.
MATCH_TOL_NS = 50_000_000
ARM_OF = {
    "human_bimanual": "both",
    "human_left_arm": "left",
    "human_right_arm": "right",
}
VIDEOS = {"front_1": (360, 480), "front_1_480": (480, 640)}
IDENTITY_POSE = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)


def make_undistort(provider, height=480, width=640):
    """aria_utils.undistort_to_linear with the calibration built once."""
    label = provider.get_label_from_stream_id(RGB)
    calib = provider.get_device_calibration().get_camera_calib(label)
    focal = 133.25430222 * (height / 240)
    linear = calibration.get_linear_camera_calibration(
        height, width, focal, label, calib.get_transform_device_camera()
    )

    def f(raw):
        return np.ascontiguousarray(
            np.rot90(calibration.distort_by_calibration(raw, linear, calib), k=3)
        )

    return f


def open_encoder(path: Path, h: int, w: int, ffmpeg: str):
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(int(FPS)), "-i", "-",
        "-c:v", "libsvtav1", "-g", "2", "-crf", "30", "-preset", "12", "-pix_fmt", "yuv420p",
        # Index at the front, so a reader over HTTP range requests can seek without fetching the tail.
        "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def hand_arrays(kp_rows: np.ndarray, side_offset: int):
    """Split get_hand_keypoints rows (wrist pose 7 + 63 per hand) for one hand."""
    blk = kp_rows[:, side_offset : side_offset + 70]
    return blk[:, :7], blk[:, 7:70]


def convert(args) -> dict:
    t0 = time.time()
    vrs = args.raw_dir / f"{args.vendor_recording_id}.vrs"
    mps_dir = args.raw_dir / f"mps_{args.vendor_recording_id}_vrs"
    out = args.out_dir / args.vendor_recording_id
    out.mkdir(parents=True, exist_ok=True)
    arm = ARM_OF[args.embodiment]
    use = {"left": arm in ("left", "both"), "right": arm in ("right", "both")}

    provider = data_provider.create_vrs_data_provider(str(vrs))
    hands = mps.hand_tracking.read_hand_tracking_results(
        str(mps_dir / "hand_tracking" / "hand_tracking_results.csv")
    )
    traj = mps.read_closed_loop_trajectory(
        str(mps_dir / "slam" / "closed_loop_trajectory.csv")
    )
    gaze_csv = mps_dir / "eye_gaze" / "general_eye_gaze.csv"
    gaze = mps.read_eyegaze(str(gaze_csv)) if gaze_csv.exists() else None

    rgb_ts = np.asarray(
        provider.get_timestamps_ns(RGB, TimeDomain.DEVICE_TIME), dtype=np.int64
    )
    med_dt = float(np.median(np.diff(rgb_ts)))
    nominal_fps = 1e9 / med_dt
    if abs(nominal_fps - FPS) > 0.1 * FPS:
        raise ValueError(f"RGB stream is {nominal_fps:.1f} fps, expected {FPS}")
    stream_ts = {"rgb": list(rgb_ts)}

    # Same per-frame pose math as aria_to_zarr, on every RGB frame (no clean_data).
    kp = AriaVRSExtractor.get_hand_keypoints(traj, stream_ts, hands, arm="both")
    mps_ee = AriaVRSExtractor.get_ee_pose(traj, stream_ts, hands, arm="both")
    head = AriaVRSExtractor.get_head_pose(
        traj, slam_to_rgb(provider).inverse(), stream_ts
    )
    eye = (
        AriaVRSExtractor.get_eye_gaze(gaze, stream_ts)
        if gaze is not None
        else np.full((len(rgb_ts), 3), SENTINEL * 10)
    )

    traj_ts = np.array(
        [int(p.tracking_timestamp.total_seconds() * 1e9) for p in traj], dtype=np.int64
    )
    traj_graph = np.array([p.graph_uid for p in traj])
    hand_ts = np.array(
        [int(h.tracking_timestamp.total_seconds() * 1e9) for h in hands], dtype=np.int64
    )
    pose_close = nearest_dt(rgb_ts, traj_ts) <= MATCH_TOL_NS
    hand_close = nearest_dt(rgb_ts, hand_ts) <= MATCH_TOL_NS
    graph = traj_graph[np.clip(np.searchsorted(traj_ts, rgb_ts), 0, len(traj_ts) - 1)]
    graph_change = np.r_[False, graph[1:] != graph[:-1]]

    per_hand = {}
    for side, off, ee_off, rhand in (("left", 0, 0, False), ("right", 70, 7, True)):
        wrist, aria_kp = hand_arrays(kp, off)
        tracked = use[side] & (wrist < SENTINEL).all(1) & hand_close & pose_close
        conf = np.full(len(rgb_ts), -1.0, dtype=np.float32)
        for i, t in enumerate(rgb_ts):
            h = get_nearest_hand_tracking_result(hands, int(t))
            one = getattr(h, f"{side}_hand", None) if h is not None else None
            if one is not None:
                conf[i] = one.confidence
        if args.mano and tracked.any():
            mano_kp = np.full(aria_kp.shape, np.nan)
            mano_kp[tracked] = convert_aria_keypoints_to_mano(
                aria_kp[tracked], is_rhand=rhand, n_iters=args.mano_iters
            )
            ee = mano_keypoints_to_cartesian(mano_kp, is_rhand=rhand)
            keypoints = mano_kp
        else:
            ee = mps_ee[:, ee_off : ee_off + 7]
            keypoints = aria_kp
        ok = tracked & np.isfinite(keypoints).all(1) & (np.abs(ee) < SENTINEL).all(1)
        ok &= ~pose_outliers(wrist[:, :3], ok)
        per_hand[side] = dict(
            wrist=wrist, ee=ee, kp=keypoints, aria_kp=aria_kp, conf=conf, ok=ok
        )

    head_ok = (head < SENTINEL).all(1) & pose_close
    head_ok &= ~pose_outliers(head[:, :3], head_ok)

    ts, src, filler, after_gap = build_timeline(rgb_ts)
    brk = after_gap | (graph_change[src] & ~filler)
    head_valid = head_ok[src] & ~filler & ~brk
    n = len(ts)

    def col(x, valid, fill):
        x = np.asarray(x, dtype=np.float32)[src]
        x[~valid] = fill
        return pa.FixedSizeListArray.from_arrays(
            pa.array(x.reshape(-1)), x.shape[1]
        ).cast(pa.list_(pa.float32(), x.shape[1]))

    cols = {
        "recording_id": pa.array([args.recording_id] * n),
        "timestamp_ns": pa.array(ts),
        "head_pose": col(head, head_valid, IDENTITY_POSE),
    }
    valid = {}
    for side, h in per_hand.items():
        v = h["ok"][src] & ~filler & ~brk
        valid[side] = v
        cols[side] = pa.StructArray.from_arrays(
            [
                col(h["wrist"], v, IDENTITY_POSE),
                col(h["ee"], v, IDENTITY_POSE),
                col(h["kp"], v, 0.0),
                col(h["aria_kp"], v, 0.0),
                pa.array(np.where(filler, -1.0, h["conf"][src]).astype(np.float32)),
            ],
            fields=list(HUMAN_STEPS.field(side).type),
        )
    cols["left_valid"] = pa.array(valid["left"])
    cols["right_valid"] = pa.array(valid["right"])
    cols["head_pose_valid"] = pa.array(head_valid)
    eye_ok = (np.abs(eye) < SENTINEL).all(1)[src] & ~filler
    cols["eye_gaze"] = col(eye, eye_ok, 0.0)
    table = pa.table(cols, schema=HUMAN_STEPS)
    pq.write_table(table, out / "steps.parquet", compression="zstd")
    t_numeric = time.time() - t0

    undistort = make_undistort(provider)
    enc = {
        k: open_encoder(out / f"{k}.mp4", h, w, args.ffmpeg)
        for k, (h, w) in VIDEOS.items()
    }
    last_src, frame = -1, None
    for s in src:
        if s != last_src:
            frame = undistort(
                provider.get_image_data_by_index(RGB, int(s))[0].to_numpy_array()
            )
            assert frame.shape == (*VIDEOS["front_1_480"], 3), frame.shape
            small = cv2.resize(
                frame,
                (VIDEOS["front_1"][1], VIDEOS["front_1"][0]),
                interpolation=cv2.INTER_AREA,
            )
            last_src = s
        enc["front_1_480"].stdin.write(frame.tobytes())
        enc["front_1"].stdin.write(small.tobytes())
    for p in enc.values():
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg failed")

    k = np.asarray(ARIA_INTRINSICS, dtype=np.float64)[:3, :3]
    scale = VIDEOS["front_1"][0] / VIDEOS["front_1_480"][0]
    k360 = k.copy()
    k360[:2] *= scale
    meta = {
        "vendor_recording_id": args.vendor_recording_id,
        "recording_id": args.recording_id,
        "embodiment": args.embodiment,
        "end_effector": END_EFFECTOR.HUMAN_HAND.value,
        "fps": FPS,
        "rgb_fps_measured": nominal_fps,
        "num_frames": n,
        "rgb_frames": int(len(rgb_ts)),
        "filler_frames": int(filler.sum()),
        "gaps": int(after_gap.sum()),
        "graph_changes": int(graph_change.sum()),
        "t_start_ns": int(ts[0]),
        # Half-open like every Lance interval: the last frame is inside its task and alias.
        "t_end_ns": int(ts[-1]) + round(1e9 / FPS),
        "mano": bool(args.mano),
        "intrinsics": {
            "front_1": k360.reshape(-1).tolist(),
            "front_1_480": k.reshape(-1).tolist(),
        },
        # head_pose is the front_1 camera pose, so the camera sits at the head frame origin.
        "extrinsics": {"front_1": np.eye(4).reshape(-1).tolist()},
        "valid_frac": {
            "left": float(valid["left"].mean()),
            "right": float(valid["right"].mean()),
            "head": float(head_valid.mean()),
        },
        "pose_unmatched": int((~pose_close).sum()),
        "hand_unmatched": int((~hand_close).sum()),
        "seconds_numeric": round(t_numeric, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--vendor-recording-id", required=True, help="epoch ms, the VRS file stem"
    )
    ap.add_argument("--embodiment", required=True, choices=sorted(ARM_OF))
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mano-iters", type=int, default=2400)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()
    args.recording_id = recording_id("aria", args.vendor_recording_id)
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(convert(args), indent=1))


if __name__ == "__main__":
    main()
