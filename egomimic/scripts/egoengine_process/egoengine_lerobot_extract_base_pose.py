"""
Add `obs.base_pose_integrated` to an RBY1 LeRobot dataset.

The integrated base pose is the world-frame (x, y, theta) the robot occupies at
each frame, obtained by integrating the per-step base-frame deltas
(actions.joint_base_torso_head_arm_hand[..., 0:3] = dx_base, dy_base, dyaw)
forward from (0, 0, 0) at episode start. This gives the policy a dead-reckoning
spatial anchor that survives april-tag occlusion.

Reference frame note (BASE frame deltas → WORLD frame pose):
    pose[0]                = (0, 0, 0)
    pose[t]                = pose[t-1] + apply(action[t-1].base, pose[t-1])

with
    apply(d, p):
        dx_world = cos(p.theta) * d.dx_base - sin(p.theta) * d.dy_base
        dy_world = sin(p.theta) * d.dx_base + cos(p.theta) * d.dy_base
        x'      = p.x + dx_world
        y'      = p.y + dy_world
        theta'  = p.theta + d.dyaw

This is purely additive on top of an existing LeRobot dataset produced by
`egoengine_lerobot_extract_arm_hand.py` — the input dataset is not modified;
a new dataset is written to `--output-path` (default: `<input>_basepose`).
Videos are symlinked by default to save disk.

Usage:
  python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_base_pose.py \
      datasets/0524_ye_push_cart_new_ours \
      --output-path datasets/0524_ye_push_cart_new_ours_basepose

  # With explicit video copying instead of symlinks:
  python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_base_pose.py \
      datasets/0524_ye_push_cart_new_ours --copy-videos
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset


# -----------------------------------------------------------------------------
# Layout: actions.joint_base_torso_head_arm_hand (49-D), produced by
# egoengine_lerobot_extract_arm_hand.py. Base block lives at [0:3].
# (See egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py
#  and CLAUDE.md "Action Vector Layout".)
# -----------------------------------------------------------------------------
SOURCE_ACTION_KEY = "actions.joint_base_torso_head_arm_hand"
BASE_DELTA_START = 0
BASE_DELTA_END = 3  # [dx_base, dy_base, dyaw]

# Fallback key (raw SEW 49-D layout) — base is at [22:25] in actions.joint.
FALLBACK_ACTION_KEY = "actions.joint"
FALLBACK_BASE_START = 22
FALLBACK_BASE_END = 25

NEW_OBS_KEY = "obs.base_pose_integrated"
NEW_OBS_DIM = 3

FEATURE_SPEC = {
    NEW_OBS_KEY: {
        "dtype": "float32",
        "shape": [NEW_OBS_DIM],
        "names": ["x_world", "y_world", "theta_world"],
    }
}


def integrate_base_deltas_base_frame(deltas: np.ndarray) -> np.ndarray:
    """Integrate (dx_base, dy_base, dyaw) into world-frame (x, y, theta).

    Convention: at frame t the action is the delta from pose[t] to pose[t+1].
    So pose[0] = (0, 0, 0) and pose[t] is the cumulative effect of
    deltas[0..t-1]. The pose at the LAST frame is the result of applying
    deltas[0..T-2] (the final delta would land outside the episode and is unused).

    Parameters
    ----------
    deltas : np.ndarray, shape (T, 3), dtype float
        Per-step base-frame deltas: columns are [dx_base, dy_base, dyaw].

    Returns
    -------
    pose : np.ndarray, shape (T, 3), dtype float32
        World-frame poses: columns are [x_world, y_world, theta_world].
    """
    T = deltas.shape[0]
    pose = np.zeros((T, 3), dtype=np.float32)
    for t in range(1, T):
        x_prev, y_prev, th_prev = pose[t - 1]
        dx_base, dy_base, dyaw = deltas[t - 1]
        c, s = float(np.cos(th_prev)), float(np.sin(th_prev))
        pose[t, 0] = x_prev + c * dx_base - s * dy_base
        pose[t, 1] = y_prev + s * dx_base + c * dy_base
        pose[t, 2] = th_prev + dyaw
    return pose


def _episode_index_scalar(v) -> int:
    """Extract scalar episode_index (handles int or shape [1] array)."""
    if hasattr(v, "__len__") and len(v) and not isinstance(v, str):
        return int(v[0])
    return int(v)


def _resolve_lerobot_root(path: Path) -> Path:
    """LeRobot root must contain meta/info.json and data/. Handles single-child parents."""
    if (path / "meta" / "info.json").is_file():
        return path
    if not path.is_dir():
        return path
    candidates = sorted(
        p for p in path.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No LeRobot dataset root at {path}: missing meta/info.json."
        )
    names = ", ".join(c.name for c in candidates)
    raise ValueError(
        f"Multiple LeRobot roots under {path}: {names}. Pass the specific folder."
    )


def _pick_action_key_and_slice(info: dict):
    """Pick which action key to read deltas from, based on what's in info.json."""
    features = info.get("features", {})
    if SOURCE_ACTION_KEY in features:
        return SOURCE_ACTION_KEY, BASE_DELTA_START, BASE_DELTA_END
    if FALLBACK_ACTION_KEY in features:
        return FALLBACK_ACTION_KEY, FALLBACK_BASE_START, FALLBACK_BASE_END
    raise KeyError(
        f"Neither {SOURCE_ACTION_KEY!r} nor {FALLBACK_ACTION_KEY!r} found in "
        f"meta/info.json features. Available: {sorted(features.keys())}"
    )


def _stats_for_key(arrays: list) -> dict:
    """mean/std/min/max for a 1D vector feature, per dim."""
    data = np.stack([np.asarray(x, dtype=np.float32) for x in arrays], axis=0)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
    }


def _link_or_copy_videos(src: Path, dst: Path, symlink: bool):
    """Mirror src/videos to dst/videos. Symlink by default to save disk."""
    if not src.exists():
        return
    if symlink:
        # Create one symlink at dst/videos -> src/videos (cheap, atomic).
        # If dst/videos already exists, leave it alone (idempotent for re-runs).
        if (dst / "videos").exists() or (dst / "videos").is_symlink():
            return
        dst.mkdir(parents=True, exist_ok=True)
        os.symlink(src.resolve(), dst / "videos", target_is_directory=True)
        print(f"Symlinked videos: {dst / 'videos'} -> {src.resolve()}")
    else:
        shutil.copytree(src, dst / "videos", dirs_exist_ok=True)
        print(f"Copied videos to {dst / 'videos'}")


def main():
    parser = argparse.ArgumentParser(
        description="Add obs.base_pose_integrated to an RBY1 LeRobot dataset "
                    "by integrating base-frame action deltas in world frame."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to LeRobot dataset root (folder with meta/info.json and data/).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output dataset path. Default: <dataset_path>_basepose",
    )
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos/ folder instead of symlinking (slower, more disk).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If output exists, wipe and re-run instead of erroring.",
    )
    args = parser.parse_args()

    in_path = Path(args.dataset_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {in_path}")
    in_path = _resolve_lerobot_root(in_path)

    if args.output_path:
        out_path = Path(args.output_path).resolve()
    else:
        out_path = in_path.parent / f"{in_path.name}_basepose"

    if out_path.exists():
        if args.overwrite:
            print(f"Overwriting existing output: {out_path}")
            shutil.rmtree(out_path)
        else:
            # Idempotency: if output already has the new feature, just exit.
            try:
                with open(out_path / "meta" / "info.json") as f:
                    out_info = json.load(f)
                if NEW_OBS_KEY in out_info.get("features", {}):
                    print(f"Output {out_path} already contains {NEW_OBS_KEY}; skipping. "
                          f"(Use --overwrite to redo.)")
                    return
            except FileNotFoundError:
                pass
            raise FileExistsError(
                f"Output exists and is incomplete: {out_path}. "
                f"Pass --overwrite to redo."
            )
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load meta + dataset
    # ------------------------------------------------------------------
    info_path = in_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", 1000)

    action_key, base_start, base_end = _pick_action_key_and_slice(info)
    print(f"Reading base deltas from '{action_key}'[{base_start}:{base_end}]")

    data_dir = str(in_path / "data")
    hf_dataset = load_dataset("parquet", data_dir=data_dir, split="train")
    print(f"Loaded {len(hf_dataset)} rows total")

    # ------------------------------------------------------------------
    # Group by episode, integrate per episode
    # ------------------------------------------------------------------
    ep_to_indices: dict[int, list[int]] = {}
    for i, ep_val in enumerate(hf_dataset["episode_index"]):
        ep_idx = _episode_index_scalar(ep_val)
        ep_to_indices.setdefault(ep_idx, []).append(i)

    total_rows = len(hf_dataset)
    poses: list = [None] * total_rows
    for ep_idx in sorted(ep_to_indices):
        indices = ep_to_indices[ep_idx]
        subset = hf_dataset.select(indices)

        # Sort by frame_index within episode (defensive — parquet usually preserves order)
        frame_idx = np.array(
            [_episode_index_scalar(x) for x in subset["frame_index"]]
        )
        order = np.argsort(frame_idx)

        actions = np.stack(
            [np.asarray(x, dtype=np.float32) for x in subset[action_key]],
            axis=0,
        )
        actions = actions[order]
        deltas = actions[:, base_start:base_end]
        pose = integrate_base_deltas_base_frame(deltas)  # (T, 3)

        global_sorted = [indices[i] for i in order]
        for row_idx, value in zip(global_sorted, pose, strict=True):
            poses[row_idx] = value.tolist()

    if any(p is None for p in poses):
        missing = sum(1 for p in poses if p is None)
        raise RuntimeError(f"Failed to fill {missing}/{total_rows} rows with base pose")

    hf_dataset = hf_dataset.add_column(NEW_OBS_KEY, poses)

    # ------------------------------------------------------------------
    # Write per-episode parquet matching LeRobot layout
    # ------------------------------------------------------------------
    data_out = out_path / "data"
    for ep_idx in sorted(ep_to_indices):
        ep_chunk = ep_idx // chunks_size
        chunk_dir = data_out / f"chunk-{ep_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"

        # Preserve original row order so frame_index → row mapping is exact.
        ep_subset = hf_dataset.select(ep_to_indices[ep_idx])
        ep_subset.to_parquet(str(ep_path))
    print(f"Wrote {len(ep_to_indices)} episode parquets to {data_out}")

    # ------------------------------------------------------------------
    # Copy meta + update info.json + stats.json
    # ------------------------------------------------------------------
    meta_src = in_path / "meta"
    meta_dst = out_path / "meta"
    shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)

    with open(meta_dst / "info.json") as f:
        new_info = json.load(f)
    new_info["features"].update(FEATURE_SPEC)
    with open(meta_dst / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    print(f"Updated {meta_dst / 'info.json'} with {NEW_OBS_KEY}")

    stats_path = meta_dst / "stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    stats[NEW_OBS_KEY] = _stats_for_key(hf_dataset[NEW_OBS_KEY])
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Updated {stats_path} with {NEW_OBS_KEY} stats: "
          f"min={stats[NEW_OBS_KEY]['min']}, max={stats[NEW_OBS_KEY]['max']}")

    # ------------------------------------------------------------------
    # Videos (symlink or copy)
    # ------------------------------------------------------------------
    _link_or_copy_videos(in_path / "videos", out_path, symlink=not args.copy_videos)

    print(f"\nDone. New dataset at: {out_path}")


if __name__ == "__main__":
    main()
