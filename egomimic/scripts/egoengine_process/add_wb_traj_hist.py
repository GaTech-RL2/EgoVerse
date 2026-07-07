"""Add whole-body temporal features to a LeRobot dataset for the V2 policy.

Adds two per-frame columns (both derived from columns already present, so NO reconversion):
  obs.robot0_joint_pos_hist : last-10-step causal window of obs.robot0_joint_pos_no_wheel (22),
                              flattened oldest->newest -> 220-D  (proprio history)
  obs.base_traj             : last-100-step causal window of the integrated base pose in the
                              FIRST-FRAME world frame = cumsum(actions.joint_base_torso_head_arm_hand[:, 0:3])
                              (dx,dy,dyaw per step -> x,y,yaw), flattened oldest->newest -> 300-D

Causal left-pad with the first frame (repeat frame 0) so every frame yields a fixed-length window;
for t >= window-1 it is the exact sliding window (pop earliest, append current). The current
pose/proprio is always the LAST block of the window.

Mirrors egoengine_lerobot_extract_arm_hand.py: per-episode add_column, rewrite per-episode parquet,
copy meta + update info["features"] and stats.json, copy videos/. The input dataset is untouched.

Usage:
  python egomimic/scripts/egoengine_process/add_wb_traj_hist.py \
      datasets/aria_egoposer_firm --output-path datasets/aria_egoposer_firm_v2
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset

JOINT_HIST_STEPS = 10
BASE_TRAJ_STEPS = 100
JOINT_DIM = 22          # obs.robot0_joint_pos_no_wheel
BASE_POSE_DIM = 3       # x, y, yaw (integrated base pose)

SRC_JOINT = "obs.robot0_joint_pos_no_wheel"
SRC_ACTION = "actions.joint_base_torso_head_arm_hand"   # base deltas = [:, 0:3]
OUT_JOINT_HIST = "obs.robot0_joint_pos_hist"
OUT_BASE_TRAJ = "obs.base_traj"

FEATURE_SPECS = {
    OUT_JOINT_HIST: {"dtype": "float32", "shape": [JOINT_DIM * JOINT_HIST_STEPS], "names": [OUT_JOINT_HIST]},
    OUT_BASE_TRAJ:  {"dtype": "float32", "shape": [BASE_POSE_DIM * BASE_TRAJ_STEPS], "names": [OUT_BASE_TRAJ]},
}


def _ep_scalar(v) -> int:
    return int(v[0]) if hasattr(v, "__len__") and len(v) and not isinstance(v, str) else int(v)


def _causal_window(seq: np.ndarray, t: int, k: int) -> np.ndarray:
    """Length-k window seq[t-k+1 .. t] left-padded by repeating seq[0]; flattened oldest->newest."""
    lo = t - k + 1
    if lo >= 0:
        win = seq[lo:t + 1]
    else:
        pad = np.repeat(seq[0:1], -lo, axis=0)
        win = np.concatenate([pad, seq[0:t + 1]], axis=0)
    return win.reshape(-1)


def main():
    ap = argparse.ArgumentParser(description="Add proprio-history + base-trajectory columns to a LeRobot dataset.")
    ap.add_argument("dataset_path")
    ap.add_argument("--output-path", required=True)
    args = ap.parse_args()

    src = Path(args.dataset_path).resolve()
    out = Path(args.output_path).resolve()
    if (src / "meta" / "info.json").is_file():
        root = src
    else:  # accept a parent with exactly one dataset inside (mirror the extractor)
        cands = sorted(p for p in src.iterdir() if (p / "meta" / "info.json").is_file())
        if len(cands) != 1:
            raise FileNotFoundError(f"No single LeRobot root under {src}")
        root = cands[0]
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = info.get("chunks_size", 1000)
    ds = load_dataset("parquet", data_dir=str(root / "data"), split="train")

    # group rows by episode, ordered by frame_index (defensive; cumsum needs frame order)
    has_frame_idx = "frame_index" in ds.column_names
    frame_idx = ds["frame_index"] if has_frame_idx else None
    ep_to_indices: dict[int, list[int]] = {}
    for i, ev in enumerate(ds["episode_index"]):
        ep_to_indices.setdefault(_ep_scalar(ev), []).append(i)
    if has_frame_idx:
        for ep in ep_to_indices:
            ep_to_indices[ep].sort(key=lambda i: _ep_scalar(frame_idx[i]))

    N = len(ds)
    joint_hist: list = [None] * N
    base_traj: list = [None] * N
    for ep, idxs in ep_to_indices.items():
        sub = ds.select(idxs)
        joint = np.stack([np.asarray(x, np.float32) for x in sub[SRC_JOINT]], axis=0)   # (T, 22)
        act = np.stack([np.asarray(x, np.float32) for x in sub[SRC_ACTION]], axis=0)     # (T, 49)
        base_pose = np.cumsum(act[:, 0:3], axis=0).astype(np.float32)                    # (T, 3) first-frame world frame
        for j, row in enumerate(idxs):
            joint_hist[row] = _causal_window(joint, j, JOINT_HIST_STEPS).tolist()
            base_traj[row] = _causal_window(base_pose, j, BASE_TRAJ_STEPS).tolist()

    if any(v is None for v in joint_hist) or any(v is None for v in base_traj):
        raise RuntimeError("Failed to fill all rows")
    ds = ds.add_column(OUT_JOINT_HIST, joint_hist)
    ds = ds.add_column(OUT_BASE_TRAJ, base_traj)

    # rewrite per-episode parquet in original LeRobot layout
    data_out = out / "data"
    for ep in sorted(ep_to_indices):
        chunk_dir = data_out / f"chunk-{ep // chunks_size:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ds.select(ep_to_indices[ep]).to_parquet(str(chunk_dir / f"episode_{ep:06d}.parquet"))

    # meta: copy, add feature specs + stats
    shutil.copytree(root / "meta", out / "meta", dirs_exist_ok=True)
    info = json.loads((out / "meta" / "info.json").read_text())
    info["features"].update(FEATURE_SPECS)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats_path = out / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for key in (OUT_JOINT_HIST, OUT_BASE_TRAJ):
        data = np.stack([np.asarray(x, np.float32) for x in ds[key]], axis=0)
        stats[key] = {"mean": data.mean(0).tolist(), "std": data.std(0).tolist(),
                      "min": data.min(0).tolist(), "max": data.max(0).tolist()}
    stats_path.write_text(json.dumps(stats, indent=4))

    if (root / "videos").exists():
        shutil.copytree(root / "videos", out / "videos", dirs_exist_ok=True)

    print(f"Saved V2 dataset -> {out}")
    print(f"  + {OUT_JOINT_HIST}: {JOINT_DIM * JOINT_HIST_STEPS}-D  ({JOINT_HIST_STEPS} steps x {JOINT_DIM})")
    print(f"  + {OUT_BASE_TRAJ}:  {BASE_POSE_DIM * BASE_TRAJ_STEPS}-D ({BASE_TRAJ_STEPS} steps x {BASE_POSE_DIM})")


if __name__ == "__main__":
    main()
