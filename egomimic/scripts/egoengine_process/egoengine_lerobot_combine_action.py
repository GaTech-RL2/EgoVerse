"""
Concatenate egoengine LeRobot actions ``actions.eef`` and ``actions.hand_right_cmd_qpos``
into a single column ``actions.right_arm_eef_hand`` (order: eef, then hand).

Expects datasets produced from ``RBY1_egoengine_HDF5_config.json`` (see ``robomimic_hd5.py``).

Output preserves LeRobot layout: ``data/chunk-*/episode_*.parquet``; copies ``meta/`` and
updates ``info.json`` / ``stats.json``. Processes all rows (train + valid episodes).

Example:
  python egomimic/scripts/egoengine_process/egoengine_lerobot_combine_action.py \\
    datasets/RBY1_egoengine_mustard_cropped \\
    --output-path datasets/RBY1_egoengine_mustard_cropped_right_arm_eef_hand
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset

KEY_EEF = "actions.eef"
KEY_HAND = "actions.hand_right_cmd_qpos"
KEY_OUT = "actions.right_arm_eef_hand"
KEY_EEF_PAD = "actions.eef_is_pad"
KEY_HAND_PAD = "actions.hand_right_cmd_qpos_is_pad"
KEY_OUT_PAD = "actions.right_arm_eef_hand_is_pad"


def _episode_index_scalar(v) -> int:
    return int(v[0]) if hasattr(v, "__len__") and len(v) and not isinstance(v, str) else int(v)


def _as_vec(x, expected: int | None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if expected is not None and a.shape[0] != expected:
        raise ValueError(f"Expected length {expected}, got {a.shape[0]}")
    return a


def combine_row(example: dict) -> dict:
    eef = _as_vec(example[KEY_EEF], None)
    hand = _as_vec(example[KEY_HAND], None)
    example[KEY_OUT] = np.concatenate([eef, hand], axis=0).astype(np.float32)

    if KEY_EEF_PAD in example and KEY_HAND_PAD in example:
        pe = _as_vec(example[KEY_EEF_PAD], eef.shape[0])
        ph = _as_vec(example[KEY_HAND_PAD], hand.shape[0])
        example[KEY_OUT_PAD] = np.concatenate([pe, ph], axis=0).astype(np.float32)
    return example


def _compute_stats_for_key(dataset, key: str):
    arrays = [np.asarray(x, dtype=np.float32) for x in dataset[key]]
    data = np.stack(arrays, axis=0)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Concat actions.eef + actions.hand_right_cmd_qpos → actions.right_arm_eef_hand."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to LeRobot dataset root (contains data/ and meta/info.json).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output dataset path. Default: <dataset_path>_right_arm_eef_hand",
    )
    parser.add_argument(
        "--drop-sources",
        action="store_true",
        help="Remove actions.eef and actions.hand_right_cmd_qpos (and their *_is_pad) from output parquet.",
    )
    args = parser.parse_args()

    path = Path(args.dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    default_output = path.parent / f"{path.name}_right_arm_eef_hand"
    output_path = Path(args.output_path).resolve() if args.output_path else default_output
    output_path.mkdir(parents=True, exist_ok=True)

    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")

    with open(info_path) as f:
        info = json.load(f)
    feats = info.get("features", {})
    if KEY_EEF not in feats or KEY_HAND not in feats:
        raise KeyError(
            f"Dataset must contain {KEY_EEF!r} and {KEY_HAND!r}; found keys: {list(feats)[:20]}..."
        )

    eef_dim = int(np.prod(feats[KEY_EEF]["shape"]))
    hand_dim = int(np.prod(feats[KEY_HAND]["shape"]))
    out_dim = eef_dim + hand_dim

    chunks_size = info.get("chunks_size", 1000)

    data_dir = str(path / "data")
    hf_dataset = load_dataset("parquet", data_dir=data_dir, split="train")

    def _map_fn(ex):
        out = combine_row(dict(ex))
        if args.drop_sources:
            for k in (
                KEY_EEF,
                KEY_HAND,
                KEY_EEF_PAD,
                KEY_HAND_PAD,
            ):
                if k in out:
                    del out[k]
        return out

    hf_dataset = hf_dataset.map(_map_fn, desc="Concat eef+hand")

    ep_to_indices: dict[int, list[int]] = {}
    for i, ep_val in enumerate(hf_dataset["episode_index"]):
        ep_idx = _episode_index_scalar(ep_val)
        ep_to_indices.setdefault(ep_idx, []).append(i)

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

    meta_src = path / "meta"
    meta_dst = output_path / "meta"
    if meta_src.exists():
        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
        with open(meta_dst / "info.json") as f:
            info_out = json.load(f)

        info_out["features"][KEY_OUT] = {
            "dtype": "float32",
            "shape": [out_dim],
            "names": ["right_arm_eef_hand"],
        }
        col_names = getattr(hf_dataset, "column_names", list(hf_dataset.features))
        if KEY_OUT_PAD in col_names:
            info_out["features"][KEY_OUT_PAD] = {
                "dtype": "float32",
                "shape": [out_dim],
                "names": ["right_arm_eef_hand_is_pad"],
            }

        if args.drop_sources:
            for k in (KEY_EEF, KEY_HAND, KEY_EEF_PAD, KEY_HAND_PAD):
                info_out["features"].pop(k, None)

        with open(meta_dst / "info.json", "w") as f:
            json.dump(info_out, f, indent=4)
        print(f"Updated {meta_dst / 'info.json'} ({KEY_OUT} shape [{out_dim}])")

        stats_path = meta_dst / "stats.json"
        stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        stats[KEY_OUT] = _compute_stats_for_key(hf_dataset, KEY_OUT)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"Updated {stats_path} for {KEY_OUT}")

    videos_src = path / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, output_path / "videos", dirs_exist_ok=True)
        print("Copied videos/")

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
