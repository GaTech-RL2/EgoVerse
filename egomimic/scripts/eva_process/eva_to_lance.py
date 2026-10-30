"""Convert one raw EVA demo (HDF5 + metadata JSON) into a delivery: per-frame parquet plus MP4s.

Numerics go through the same EvaHD5Extractor and per-arm split as eva_to_zarr, so steps
columns equal the zarr keys. Frames where an arm's pose reads all zeros stay forward-filled
as in zarr, and are also flagged `<arm>_valid = False`. The HDF5 carries no per-frame
times, so frame i is at start + i / --fps. With --stage-to the delivery is staged for the
committer (egomimic/scripts/lance_commit.py), which appends it to the Lance tables.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egomimic.rldb.embodiment.end_effector import END_EFFECTOR
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.lance import commit
from egomimic.rldb.lance.schema import ARM, EVA_STEPS, recording_id, robot_steps_table
from egomimic.rldb.zarr.required_metadata import required_metadata_error
from egomimic.scripts.eva_process.eva_to_zarr import (
    _separate_numeric_and_image,
    _split_per_arm,
)
from egomimic.scripts.eva_process.eva_utils import EvaHD5Extractor
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.video_utils import open_av1_encoder

log = logging.getLogger("eva_to_lance")

ARM_OF = {"eva_bimanual": "both", "eva_left_arm": "left", "eva_right_arm": "right"}
# HDF5 camera -> delivery camera; 640x480 sources, encoded at short side 360.
CAMERAS = {
    "front_img_1": "front_1",
    "left_wrist_img": "left_wrist",
    "right_wrist_img": "right_wrist",
}
SIZE = (360, 480)


def numeric_columns(raw: Path, arm: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """({zarr key: (T, d) array} of the robot's arms, (T, 2) [left, right] dropout),
    with eva_to_zarr's per-arm math.

    Raw arrays are (T, 14) with the left arm in columns 0-6 and the right in 7-13 even
    on a one-arm robot; eva_to_zarr reads a lone right arm from columns 0-6 (zeros), so
    this splits both slots and keeps the robot's arms.
    """
    feats = EvaHD5Extractor.process_episode(raw, arm=arm, with_images=False)
    numeric, _ = _separate_numeric_and_image(feats)
    widths = {k: v.shape[1] for k, v in numeric.items()}
    if set(widths.values()) != {14}:
        raise ValueError(f"expected (T, 14) arm arrays, got widths {widths}")
    keep = ("left", "right") if arm == "both" else (arm,)
    cols = {
        k: v
        for k, v in _split_per_arm(numeric, "both").items()
        if k.split(".")[0] in keep
    }
    return cols, feats["metadata.pose_dropout"]


def steps_table(rid: str, ts: np.ndarray, cols: dict, valid: dict) -> pa.Table:
    """EVA_STEPS rows; an arm the robot lacks is zeros."""
    n = len(ts)
    zeros = {
        f"{side}.{f.name}": np.zeros((n, getattr(f.type, "list_size", 1)))
        for side in ("left", "right")
        for f in ARM
    }
    return robot_steps_table(EVA_STEPS, rid, ts, {**zeros, **cols}, valid)


def encode_videos(raw: Path, out: Path, n: int, fps: float, ffmpeg: str) -> list[str]:
    h, w = SIZE
    done = []
    with h5py.File(raw, "r") as f:
        imgs = f["observations/images"]
        for src, cam in CAMERAS.items():
            if src not in imgs:
                continue
            if len(imgs[src]) != n:
                raise ValueError(f"{src} has {len(imgs[src])} frames, numerics {n}")
            enc = open_av1_encoder(out / f"{cam}.mp4", h, w, fps, ffmpeg)
            for a in range(0, n, 64):
                for frame in imgs[src][a : a + 64]:
                    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    enc.stdin.write(np.ascontiguousarray(small).tobytes())
            enc.stdin.close()
            if enc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed on {cam}")
            done.append(cam)
    return done


def required_fields(md: dict) -> dict[str, str]:
    """lab, task_name and operator from the demo metadata; raises if any is absent,
    empty or a placeholder (only scene and objects may be missing)."""
    fields = {
        "lab": md.get("lab"),
        "task_name": md.get("task"),
        "operator": md.get("operator"),
    }
    errors = [e for f, v in fields.items() if (e := required_metadata_error(f, v))]
    if errors:
        raise ValueError("; ".join(errors))
    return fields


def convert(args) -> dict:
    t0 = time.time()
    raw = args.raw_dir / f"{args.vendor_recording_id}.hdf5"
    md = json.loads(
        (args.raw_dir / f"{args.vendor_recording_id}_metadata.json").read_text()
    )
    required = required_fields(md)
    embodiment = md["robot_name"]
    arm = ARM_OF[embodiment]
    out = args.out_dir / args.vendor_recording_id
    out.mkdir(parents=True, exist_ok=True)
    rid = recording_id("eva", args.vendor_recording_id)

    cols, dropout = numeric_columns(raw, arm)
    n = len(dropout)
    valid = {
        side: ~dropout[:, slot] & (arm in (side, "both"))
        for slot, side in enumerate(("left", "right"))
    }
    start_ns = int(args.vendor_recording_id) * 1_000_000
    ts = start_ns + np.round(np.arange(n) * 1e9 / args.fps).astype(np.int64)
    pq.write_table(
        steps_table(rid, ts, cols, valid), out / "steps.parquet", compression="zstd"
    )
    t_numeric = time.time() - t0

    cams = encode_videos(raw, out, n, args.fps, args.ffmpeg)
    k = np.asarray(Eva.INTRINSICS, np.float64)[:3, :3].copy()
    k[:2] *= SIZE[0] / 480
    meta = {
        "vendor_recording_id": args.vendor_recording_id,
        "recording_id": rid,
        "episode_hash": timestamp_ms_to_episode_hash(args.vendor_recording_id),
        "embodiment": embodiment,
        "end_effector": END_EFFECTOR.PARALLEL_JAW_GRIPPER.value,
        "lab": required["lab"],
        "operator": required["operator"],
        "scene": md.get("scene"),
        "task_name": required["task_name"],
        "task_description": md.get("task_description"),
        "objects": md.get("objects") or [],
        "fps": args.fps,
        "num_frames": n,
        "videos": cams,
        "t_start_ns": int(ts[0]),
        "t_end_ns": int(ts[-1]) + round(1e9 / args.fps),
        "intrinsics": {"front_1": k.reshape(-1).tolist()},
        # Same per-arm base-to-camera transforms eva_to_zarr stores.
        "extrinsics": {a: m.reshape(-1).tolist() for a, m in Eva.EXTRINSICS.items()},
        "valid_frac": {s: float(v.mean()) for s, v in valid.items()},
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
        "--vendor-recording-id", required=True, help="epoch ms, the HDF5 file stem"
    )
    ap.add_argument(
        "--fps",
        type=float,
        required=True,
        help="the demo's frame rate; the HDF5 has no per-frame times",
    )
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument(
        "--stage-to", help="staging root (s3:// or local) for the committer"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(convert(args), indent=1))
    if args.stage_to:
        commit.stage(
            args.out_dir / args.vendor_recording_id,
            commit.join(args.stage_to, "eva", args.vendor_recording_id),
        )


if __name__ == "__main__":
    main()
