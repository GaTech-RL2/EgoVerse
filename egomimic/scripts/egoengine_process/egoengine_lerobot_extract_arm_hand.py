"""
Extract right arm + hand from RBY1 LeRobot dataset for egoengine.

Creates (see module constants for index layouts):
- obs.right_arm / obs.left_arm from obs.robot0_joint_pos (26-D: base, torso, right arm, left arm, head)
- actions.joint_* from actions.joint only (full **49-D** SEW command: arms, torso, head, deltas, hands)

Output preserves original LeRobot structure: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
Processes all episodes (train + valid splits).

Example usage:
- python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py datasets/0309/RBY1_0309 --output-path datasets/0309/RBY1_0309_right_arm
- python egomimic/scripts/egoengine_process/egoengine_lerobot_extract_arm_hand.py datasets/0309/RBY1_0309 --black-image
"""

# TODO: obs: get the robot joint (remove wheel), action: add arm torso joint base (base from joint)

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Default black image used when obs.aria_image is missing.
DEFAULT_ARIA_IMAGE_HEIGHT = 128
DEFAULT_ARIA_IMAGE_WIDTH = 128
DEFAULT_ARIA_IMAGE_CHANNELS = 3

# -----------------------------------------------------------------------------
# RBY1 layouts (egoengine / SEW-style HDF5 → LeRobot)
#
# 1) obs.robot0_joint_pos — fixed 26-D (state on the robot; includes mobile base):
#    [0:4)   base (4)
#    [4:10)  torso (6)
#    [10:17) right arm (7)
#    [17:24) left arm (7)
#    [24:26) head (2)
#
# 2) actions.joint — SEW data_collection command layout (NOT parallel to robot0_joint_pos).
#    Full vector is 49-D:
#        actions_joint = np.hstack([
#            q_goal_left,          # (N, 7)   [0:7)
#            q_goal_right,         # (N, 7)   [7:14)
#            q_goal_torso,         # (N, 6)   [14:20)
#            q_goal_head,          # (N, 2)   [20:22)
#            delta_x, delta_y, delta_yaw,  # (N, 1) each → [22:25)
#            hand_left_cmd_qpos,   # (N, 12)  [25:37)
#            hand_right_cmd_qpos,  # (N, 12)  [37:49)
#        ])
#    Torso for derived keys is [14:20], not [4:10]. This script expects the full 49-D
#    vector so arms, torso, and hands are read from one tensor (no actions.joint_arm_hand).
# -----------------------------------------------------------------------------
LEFT_ARM, RIGHT_ARM, LEFT_HAND, RIGHT_HAND = 7, 7, 12, 12

# --- obs.robot0_joint_pos only ---
OBS_BASE_END = 4
OBS_TORSO_START = 4
OBS_TORSO_END = 10
OBS_RIGHT_ARM_START = 10
OBS_RIGHT_ARM_END = OBS_RIGHT_ARM_START + RIGHT_ARM
OBS_LEFT_ARM_START = OBS_RIGHT_ARM_END
OBS_LEFT_ARM_END = OBS_LEFT_ARM_START + LEFT_ARM
OBS_HEAD_START = 24
OBS_HEAD_END = 26
ROBOT0_DIM = OBS_HEAD_END

# --- actions.joint (SEW); torso slice [14:20] ---
ACTJ_LEFT_ARM_START = 0
ACTJ_LEFT_ARM_END = LEFT_ARM
ACTJ_RIGHT_ARM_START = LEFT_ARM
ACTJ_RIGHT_ARM_END = LEFT_ARM + RIGHT_ARM
ACTJ_TORSO_START = ACTJ_RIGHT_ARM_END
ACTJ_TORSO_END = ACTJ_TORSO_START + 6
# After head (2) and deltas (3): hand command blocks match joint_arm_hand ordering.
ACTJ_HEAD_START = 20
ACTJ_HEAD_END = ACTJ_HEAD_START + 2
ACTJ_BASE_START = 22
ACTJ_BASE_END = ACTJ_BASE_START + 3 # delta_x, delta_y, delta_yaw
ACTJ_LEFT_HAND_START = 25
ACTJ_LEFT_HAND_END = ACTJ_LEFT_HAND_START + LEFT_HAND
ACTJ_RIGHT_HAND_START = ACTJ_LEFT_HAND_END
ACTJ_RIGHT_HAND_END = ACTJ_RIGHT_HAND_START + RIGHT_HAND
ACTJ_FULL_DIM = ACTJ_RIGHT_HAND_END  # 49

TORSO_DIM = ACTJ_TORSO_END - ACTJ_TORSO_START  # 6; same size as OBS torso, different indices

FEATURE_SPECS = {
    "obs.right_arm": {
        "dtype": "float32",
        "shape": [RIGHT_ARM],
        "names": ["right_arm"],
    },
    "obs.left_arm": {
        "dtype": "float32",
        "shape": [LEFT_ARM],
        "names": ["left_arm"],
    },
    "actions.joint_right_arm_hand": {
        "dtype": "float32",
        "shape": [RIGHT_ARM + RIGHT_HAND],
        "names": ["joint_right_arm_hand"],
    },
    "actions.joint_left_arm_hand": {
        "dtype": "float32",
        "shape": [LEFT_ARM + LEFT_HAND],
        "names": ["joint_left_arm_hand"],
    },
    "actions.joint_right_arm": {
        "dtype": "float32",
        "shape": [RIGHT_ARM],
        "names": ["joint_right_arm"],
    },
    "actions.joint_left_arm": {
        "dtype": "float32",
        "shape": [LEFT_ARM],
        "names": ["joint_left_arm"],
    },
    "actions.joint_right_arm_hand_torso": {
        "dtype": "float32",
        "shape": [RIGHT_ARM + RIGHT_HAND + TORSO_DIM],
        "names": ["joint_right_arm_hand_torso"],
    },
    "actions.joint_arm_hand_torso": {
        "dtype": "float32",
        "shape": [LEFT_ARM + RIGHT_ARM + LEFT_HAND + RIGHT_HAND + TORSO_DIM],
        "names": ["joint_arm_hand_torso"],
    },
    "actions.joint_base_torso": {
        "dtype": "float32",
        "shape": [TORSO_DIM + 3],
        "names": ["joint_base_torso"],
    },
    "actions.joint_base_torso_head_arm_hand": {
        "dtype": "float32",
        "shape": [3 + TORSO_DIM + 2 + LEFT_ARM + RIGHT_ARM + LEFT_HAND + RIGHT_HAND],
        "names": ["joint_base_torso_head_arm_hand"],
    },
    "actions.joint_hands": {
        "dtype": "float32",
        "shape": [LEFT_HAND + RIGHT_HAND],
        "names": ["joint_hands"],
    },
    "actions.joint_arm_head": {
        "dtype": "float32",
        "shape": [LEFT_ARM + RIGHT_ARM + 2],
        "names": ["joint_arm_head"],
    },
    "obs.robot0_joint_pos_no_wheel": {
        "dtype": "float32",
        "shape": [ROBOT0_DIM - OBS_BASE_END],
        "names": ["obs.robot0_joint_pos_no_wheel"],
    },
}

DELTA_ACTION_SOURCES = {
    "actions.delta_joint_right_arm": "actions.joint_right_arm",
    "actions.delta_joint_left_arm": "actions.joint_left_arm",
    "actions.delta_joint_right_arm_hand": "actions.joint_right_arm_hand",
    "actions.delta_joint_left_arm_hand": "actions.joint_left_arm_hand",
    "actions.delta_joint_arm_hand_torso": "actions.joint_arm_hand_torso",
    "actions.delta_joint_base_torso_head_arm_hand": "actions.joint_base_torso_head_arm_hand",
}

# For mixed mobile action vectors, the base commands are already per-step deltas
# in the raw SEW action layout. Preserve them instead of finite-differencing them.
DELTA_PRESERVE_CURRENT_SLICES = {
    "actions.delta_joint_base_torso_head_arm_hand": [(0, 3)],
}

DELTA_FEATURE_SPECS = {
    delta_key: {
        "dtype": "float32",
        "shape": FEATURE_SPECS[source_key]["shape"],
        "names": [delta_key.removeprefix("actions.")],
    }
    for delta_key, source_key in DELTA_ACTION_SOURCES.items()
}

BASE_STATS_KEYS = [
    "obs.right_arm",
    "obs.left_arm",
    "actions.joint_right_arm_hand",
    "actions.joint_left_arm_hand",
    "actions.joint_right_arm",
    "actions.joint_left_arm",
    "actions.joint_right_arm_hand_torso",
    "actions.joint_arm_hand_torso",
]


def slice_actions_joint(joint: np.ndarray):
    """
    Slice the full SEW actions.joint (49-D): arms, torso, left_hand, right_hand.

    Hands live at [25:37) and [37:49), same layout as the 38-D joint_arm_hand tensor
    would concatenate (left_hand then right_hand after the fixed prefix).
    """
    j = np.asarray(joint).reshape(-1)
    n = j.shape[0]
    if n != ACTJ_FULL_DIM:
        raise ValueError(
            f"Expected actions.joint length {ACTJ_FULL_DIM} (full SEW command), got {n}"
        )
    left_arm = j[ACTJ_LEFT_ARM_START:ACTJ_LEFT_ARM_END]
    right_arm = j[ACTJ_RIGHT_ARM_START:ACTJ_RIGHT_ARM_END]
    torso = j[ACTJ_TORSO_START:ACTJ_TORSO_END]
    left_hand = j[ACTJ_LEFT_HAND_START:ACTJ_LEFT_HAND_END]
    right_hand = j[ACTJ_RIGHT_HAND_START:ACTJ_RIGHT_HAND_END]
    base = j[ACTJ_BASE_START:ACTJ_BASE_END]
    head = j[ACTJ_HEAD_START:ACTJ_HEAD_END]
    return left_arm, right_arm, torso, left_hand, right_hand, base, head


def slice_obs_robot0_joint_pos(robot0: np.ndarray):
    """Right and left arm slices from obs.robot0_joint_pos (26-D)."""
    r = np.asarray(robot0).reshape(-1)
    if r.shape[0] < ROBOT0_DIM:
        raise ValueError(
            f"obs.robot0_joint_pos must have length >={ROBOT0_DIM}, got {r.shape[0]}"
        )
    right_arm = r[OBS_RIGHT_ARM_START:OBS_RIGHT_ARM_END]
    left_arm = r[OBS_LEFT_ARM_START:OBS_LEFT_ARM_END]
    return right_arm, left_arm


def add_arm_hand(example: dict, black_image: bool = False) -> dict:
    """Add obs.right_arm / obs.left_arm and derived action keys from RBY1 data."""
    joint = np.asarray(example["actions.joint"])
    left_arm, right_arm, torso, left_hand, right_hand, base, head = slice_actions_joint(joint)

    example["actions.joint_right_arm_hand"] = np.concatenate([right_arm, right_hand])
    example["actions.joint_left_arm_hand"] = np.concatenate([left_arm, left_hand])
    example["actions.joint_right_arm"] = right_arm
    example["actions.joint_left_arm"] = left_arm
    example["actions.joint_right_arm_hand_torso"] = np.concatenate([right_arm, right_hand, torso])
    example["actions.joint_arm_hand_torso"] = np.concatenate(
        [left_arm, right_arm, left_hand, right_hand, torso]
    )
    # added for mobile manipulation, we predict it in a hierarchical manner
    example["actions.joint_hands"] = np.concatenate([left_hand, right_hand])
    example["actions.joint_arm_head"] = np.concatenate([left_arm, right_arm, head])
    example["actions.joint_base_torso"] = np.concatenate([torso, base])
    example["actions.joint_base_torso_head_arm_hand"] = np.concatenate([base, torso, head, left_arm, right_arm, left_hand, right_hand])

    robot0_joint_pos = np.asarray(example["obs.robot0_joint_pos"])
    obs_right, obs_left = slice_obs_robot0_joint_pos(robot0_joint_pos)
    example["obs.right_arm"] = obs_right
    example["obs.left_arm"] = obs_left
    example["obs.robot0_joint_pos_no_wheel"] = robot0_joint_pos[OBS_BASE_END:]

    if black_image:
        if "obs.aria_image" in example:
            img = example["obs.aria_image"]
            if hasattr(img, "shape"):  # array-like (skip VideoFrame dicts)
                img = np.asarray(img)
                example["obs.aria_image"] = np.zeros_like(img)
        else:
            example["obs.aria_image"] = np.zeros(
                (
                    DEFAULT_ARIA_IMAGE_CHANNELS,
                    DEFAULT_ARIA_IMAGE_HEIGHT,
                    DEFAULT_ARIA_IMAGE_WIDTH,
                ),
                dtype=np.float16,
            )
    return example


def _episode_index_scalar(v) -> int:
    """Extract scalar episode_index (handles int or shape [1] array)."""
    return int(v[0]) if hasattr(v, "__len__") and len(v) and not isinstance(v, str) else int(v)


def _add_stepwise_delta_columns(dataset, ep_to_indices: dict[int, list[int]]):
    """Add per-episode joint deltas while preserving already-delta action slices."""
    total_rows = len(dataset)
    for delta_key, source_key in DELTA_ACTION_SOURCES.items():
        deltas = [None] * total_rows
        for indices in ep_to_indices.values():
            source = np.stack(
                [np.asarray(x, dtype=np.float32) for x in dataset.select(indices)[source_key]],
                axis=0,
            )
            delta = np.zeros_like(source, dtype=np.float32)
            if source.shape[0] > 1:
                delta[:-1] = source[1:] - source[:-1]
            for start, end in DELTA_PRESERVE_CURRENT_SLICES.get(delta_key, []):
                delta[:, start:end] = source[:, start:end]
            for row_idx, value in zip(indices, delta, strict=True):
                deltas[row_idx] = value.tolist()

        if any(value is None for value in deltas):
            raise RuntimeError(f"Failed to fill all rows for {delta_key}")
        dataset = dataset.add_column(delta_key, deltas)
    return dataset


def _resolve_lerobot_root(path: Path) -> Path:
    """
    LeRobot root must contain meta/info.json and data/ (parquet).
    If the user passes a parent folder (e.g. .../0417_pickupbox_no_mobile_2) with one
    child dataset inside, use that child automatically.
    """
    if (path / "meta" / "info.json").is_file():
        return path
    if not path.is_dir():
        return path
    candidates = sorted(
        p
        for p in path.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No LeRobot dataset root at {path}: missing meta/info.json. "
            "Pass the directory that contains meta/ and data/ "
            "(e.g. .../0417_pickupbox_no_mobile_2/RBY1_0416_pickupbox_no_mobile)."
        )
    names = ", ".join(c.name for c in candidates)
    raise ValueError(
        f"Multiple LeRobot roots under {path}: {names}. "
        "Pass the specific dataset folder that has meta/info.json."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract right arm+hand obs/actions from RBY1 LeRobot dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help=(
            "Path to LeRobot dataset root (folder with meta/info.json and data/). "
            "If you pass a parent with exactly one such subfolder, it is used automatically."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output dataset path. Default: <dataset_path>_right_arm or _right_arm_black_image if --black-image",
    )
    parser.add_argument(
        "--black-image",
        action="store_true",
        help=(
            "Overwrite obs.aria_image with same-size black images. "
            "If obs.aria_image is missing, create a default 224x224x3 black image. "
            "Output path gets _black_image appendix."
        ),
    )
    parser.add_argument(
        "--add-delta-actions",
        action="store_true",
        help=(
            "Add per-episode stepwise joint-delta action columns. Absolute joint "
            "target slices use delta[t] = target[t+1] - target[t] with zero final "
            "deltas; already-delta base command slices are preserved."
        ),
    )
    args = parser.parse_args()

    path = Path(args.dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    path = _resolve_lerobot_root(path)

    suffix = "_black_image" if args.black_image else "_right_arm"
    default_output = path.parent / f"{path.name}{suffix}"
    if args.output_path:
        output_path = Path(str(args.output_path)).resolve()
    else:
        output_path = Path(str(default_output)).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load meta for chunks_size and data_path template
    info_path = path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", 1000)

    # Load all parquet data (all episodes, train + valid)
    data_dir = str(path / "data")
    hf_dataset = load_dataset("parquet", data_dir=data_dir, split="train")
    hf_dataset = hf_dataset.map(
        lambda ex: add_arm_hand(ex, black_image=args.black_image),
        desc="Extract arm+hand",
    )

    def _compute_stats_for_key(dataset, key: str):
        """Compute mean/std/min/max for 1D vector features."""
        arrays = [np.asarray(x) for x in dataset[key]]
        data = np.stack(arrays, axis=0)
        return {
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
        }

    # Group row indices by episode (single pass via column)
    ep_to_indices = {}
    for i, ep_val in enumerate(hf_dataset["episode_index"]):
        ep_idx = _episode_index_scalar(ep_val)
        ep_to_indices.setdefault(ep_idx, []).append(i)

    if args.add_delta_actions:
        hf_dataset = _add_stepwise_delta_columns(hf_dataset, ep_to_indices)

    # Write per-episode parquet in original LeRobot structure
    data_out = output_path / "data"
    for ep_idx in sorted(ep_to_indices):
        ep_chunk = ep_idx // chunks_size
        chunk_dir = data_out / f"chunk-{ep_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"

        indices = ep_to_indices[ep_idx]
        ep_subset = hf_dataset.select(indices)
        ep_subset.to_parquet(str(ep_path))
        print(f"Wrote {ep_path.relative_to(output_path)}")

    # Copy meta and add new features to info.json
    meta_src = path / "meta"
    meta_dst = output_path / "meta"
    if meta_src.exists():
        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
        with open(meta_dst / "info.json") as f:
            info = json.load(f)
        info["features"].update(FEATURE_SPECS)
        if args.add_delta_actions:
            info["features"].update(DELTA_FEATURE_SPECS)
        if args.black_image and "obs.aria_image" not in info["features"]:
            info["features"]["obs.aria_image"] = {
                "dtype": "image",
                "shape": [DEFAULT_ARIA_IMAGE_CHANNELS, DEFAULT_ARIA_IMAGE_HEIGHT, DEFAULT_ARIA_IMAGE_WIDTH],
                "names": ["channel", "height", "width"],
            }
        with open(meta_dst / "info.json", "w") as f:
            json.dump(info, f, indent=4)
        print(f"Updated {meta_dst / 'info.json'} with new features")

        # Update stats.json with normalization statistics for new features
        stats_path = meta_dst / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            stats = {}

        new_keys = list(BASE_STATS_KEYS)
        if args.add_delta_actions:
            new_keys.extend(DELTA_ACTION_SOURCES)
        for key in new_keys:
            if key not in stats:
                stats[key] = _compute_stats_for_key(hf_dataset, key)

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Updated {stats_path} with normalization statistics for new features")

    # Copy videos if present
    videos_src = path / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, output_path / "videos", dirs_exist_ok=True)
        print("Copied videos/")

    print(f"Saved to {output_path} (LeRobot format: data/chunk-*/episode_*.parquet)")


if __name__ == "__main__":
    main()
