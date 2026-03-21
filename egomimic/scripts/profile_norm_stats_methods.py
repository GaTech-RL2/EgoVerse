import argparse
import bisect
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.utils import DataSchematic


KEYS = [
    "observations.state.ee_pose",
    "observations.state.joint_positions",
    "actions_joints",
    "actions_cartesian",
]
STAT_NAMES = ["mean", "std", "min", "max", "median", "quantile_1", "quantile_99"]


class _EpisodeDataset:
    def __init__(self, episode_path: Path):
        self.episode_path = episode_path


class _DonutDataset:
    """Dataset shim for both infer_norm_from_episodes and infer_norm_from_dataset."""

    def __init__(self, episode_paths: list[Path]):
        self.datasets = {p.stem: _EpisodeDataset(p) for p in episode_paths}
        self._groups = []
        self._lengths = []
        for p in episode_paths:
            group = zarr.open_group(str(p), mode="r")
            n_frames = int(dict(group.attrs).get("total_frames", group[KEYS[0]].shape[0]))
            self._groups.append(group)
            self._lengths.append(n_frames)

        self._cum_lengths = []
        running = 0
        for n in self._lengths:
            running += n
            self._cum_lengths.append(running)

    def __len__(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def __getitem__(self, idx: int):
        ep_idx = bisect.bisect_right(self._cum_lengths, idx)
        prev = 0 if ep_idx == 0 else self._cum_lengths[ep_idx - 1]
        local_idx = idx - prev
        group = self._groups[ep_idx]
        return {k: np.asarray(group[k][local_idx]) for k in KEYS}


def _build_schematic(embodiment_name: str) -> DataSchematic:
    schematic_dict = {
        embodiment_name: {
            KEYS[0]: {"key_type": "proprio_keys", "zarr_key": KEYS[0]},
            KEYS[1]: {"key_type": "proprio_keys", "zarr_key": KEYS[1]},
            KEYS[2]: {"key_type": "action_keys", "zarr_key": KEYS[2]},
            KEYS[3]: {"key_type": "action_keys", "zarr_key": KEYS[3]},
        }
    }
    viz_img_key = {embodiment_name: "observations.images.base_0_rgb"}
    return DataSchematic(schematic_dict, viz_img_key)


def main():
    parser = argparse.ArgumentParser(
        description="Profile infer_norm_from_episodes vs infer_norm_from_dataset on zarr episodes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/opt/dlami/nvme/aseem/robot_zarr_dataset/donuts_flat"),
    )
    parser.add_argument("--max-episodes", type=int, default=300)
    parser.add_argument("--embodiment", type=str, default="scale_aloha_bimanual")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write profile report JSON",
    )
    args = parser.parse_args()

    episode_paths = sorted(args.dataset_dir.glob("*.zarr"))
    if not episode_paths:
        raise RuntimeError(f"No .zarr episodes found in {args.dataset_dir}")
    if args.max_episodes > 0:
        episode_paths = episode_paths[: args.max_episodes]

    print(f"[profile] dataset_dir={args.dataset_dir}", flush=True)
    print(f"[profile] selected_episodes={len(episode_paths)}", flush=True)
    print(f"[profile] embodiment={args.embodiment}", flush=True)
    print(
        f"[profile] dataloader batch_size={args.batch_size} num_workers={args.num_workers}",
        flush=True,
    )

    dataset = _DonutDataset(episode_paths)
    print(f"[profile] total_frames={len(dataset)}", flush=True)

    # Method A: bulk episode reads
    print("[profile] running infer_norm_from_episodes...", flush=True)
    bulk_schematic = _build_schematic(args.embodiment)
    bulk_t0 = time.perf_counter()
    bulk_schematic.infer_norm_from_episodes(dataset, args.embodiment)
    bulk_t1 = time.perf_counter()

    # Method B: existing DataLoader path
    print("[profile] running infer_norm_from_dataset...", flush=True)
    dl_schematic = _build_schematic(args.embodiment)
    dl_t0 = time.perf_counter()
    dl_schematic.infer_norm_from_dataset(
        dataset,
        args.embodiment,
        sample_frac=1.0,
        seed=42,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dl_t1 = time.perf_counter()

    emb_id = get_embodiment_id(args.embodiment)
    diffs = {}
    for key in KEYS:
        diffs[key] = {}
        for stat in STAT_NAMES:
            bulk = bulk_schematic.norm_stats[emb_id][key][stat].float()
            dl = dl_schematic.norm_stats[emb_id][key][stat].float()
            delta = (bulk - dl).abs()
            diffs[key][stat] = {
                "max_abs_diff": float(delta.max().item()),
                "mean_abs_diff": float(delta.mean().item()),
            }

    bulk_seconds = bulk_t1 - bulk_t0
    dl_seconds = dl_t1 - dl_t0
    report = {
        "dataset_dir": str(args.dataset_dir),
        "episodes": len(episode_paths),
        "frames": len(dataset),
        "embodiment": args.embodiment,
        "bulk_seconds": round(bulk_seconds, 4),
        "dataloader_seconds": round(dl_seconds, 4),
        "speedup_x": round(dl_seconds / bulk_seconds, 4) if bulk_seconds > 0 else None,
        "diffs": diffs,
    }

    print("[profile] done", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"[profile] wrote report to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
