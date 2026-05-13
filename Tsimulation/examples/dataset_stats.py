"""Aggregate stats over a directory of PushShapes episode_*.zarr stores.

Reports episode count, length stats, final / mean coverage, action ranges,
and per-config breakdown (object × pusher × obstacle_level). Useful for
sanity-checking a collected dataset at a glance.

Usage::

    python -m Tsimulation.examples.dataset_stats --dataset data/pushshapes_demos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.collect.zarr_writer import ACTION_KEY, REWARD_KEY

_EPISODE_RE = re.compile(r"^episode_(\d+)\.zarr$")


def _list_episodes(dataset: Path) -> list[Path]:
    eps = []
    for entry in sorted(dataset.iterdir()):
        if entry.is_dir() and _EPISODE_RE.match(entry.name):
            eps.append(entry)
    return eps


def _summarize(dataset: Path) -> dict:
    ep_paths = _list_episodes(dataset)
    if not ep_paths:
        raise SystemExit(f"no episode_*.zarr stores in {dataset}")

    lengths: list[int] = []
    final_cov: list[float] = []
    mean_cov: list[float] = []
    actions_min: list[np.ndarray] = []
    actions_max: list[np.ndarray] = []
    configs: Counter = Counter()
    per_config_lengths: dict[tuple, list[int]] = defaultdict(list)
    per_config_final_cov: dict[tuple, list[float]] = defaultdict(list)
    fps_seen: set[int] = set()

    for ep_path in ep_paths:
        store = zarr.open_group(str(ep_path), mode="r")
        attrs = dict(store.attrs)
        total = int(attrs.get("total_frames", 0))
        if total <= 0:
            continue
        actions = np.asarray(store[ACTION_KEY][:total])
        rewards = np.asarray(store[REWARD_KEY][:total]).reshape(-1)

        lengths.append(total)
        final_cov.append(float(rewards[-1]))
        mean_cov.append(float(rewards.mean()))
        actions_min.append(actions.min(axis=0))
        actions_max.append(actions.max(axis=0))
        fps_seen.add(int(attrs.get("fps", 0)))

        try:
            env_args = json.loads(attrs.get("task_description", "{}")).get(
                "env_args", {}
            )
        except json.JSONDecodeError:
            env_args = {}
        cfg = (
            env_args.get("object_shape", "?"),
            env_args.get("pusher_shape", "?"),
            env_args.get("obstacle_level", "?"),
        )
        configs[cfg] += 1
        per_config_lengths[cfg].append(total)
        per_config_final_cov[cfg].append(float(rewards[-1]))

    lengths_arr = np.asarray(lengths)
    final_cov_arr = np.asarray(final_cov)
    mean_cov_arr = np.asarray(mean_cov)
    actions_lo = np.stack(actions_min).min(axis=0) if actions_min else np.zeros(2)
    actions_hi = np.stack(actions_max).max(axis=0) if actions_max else np.zeros(2)

    return {
        "dataset": dataset,
        "n_episodes": len(ep_paths),
        "n_with_data": int(lengths_arr.size),
        "total_frames": int(lengths_arr.sum()),
        "length_min": int(lengths_arr.min()),
        "length_max": int(lengths_arr.max()),
        "length_mean": float(lengths_arr.mean()),
        "length_median": float(np.median(lengths_arr)),
        "final_cov_min": float(final_cov_arr.min()),
        "final_cov_max": float(final_cov_arr.max()),
        "final_cov_mean": float(final_cov_arr.mean()),
        "mean_cov_mean": float(mean_cov_arr.mean()),
        "action_lo": actions_lo.tolist(),
        "action_hi": actions_hi.tolist(),
        "fps_seen": sorted(fps_seen),
        "configs": configs,
        "per_config_lengths": per_config_lengths,
        "per_config_final_cov": per_config_final_cov,
    }


def _print(stats: dict, success_threshold: float) -> None:
    print(f"dataset: {stats['dataset']}")
    print(f"  episodes: {stats['n_episodes']}  (with data: {stats['n_with_data']})")
    print(f"  total frames: {stats['total_frames']}")
    print(
        f"  length: min={stats['length_min']}  median={stats['length_median']:.0f}  "
        f"max={stats['length_max']}  mean={stats['length_mean']:.1f}"
    )
    print(
        f"  final coverage: min={stats['final_cov_min']:.3f}  "
        f"mean={stats['final_cov_mean']:.3f}  max={stats['final_cov_max']:.3f}"
    )
    print(f"  mean coverage (per-ep average over frames): {stats['mean_cov_mean']:.3f}")
    print(
        f"  action range:  x in [{stats['action_lo'][0]:.1f}, "
        f"{stats['action_hi'][0]:.1f}]   "
        f"y in [{stats['action_lo'][1]:.1f}, {stats['action_hi'][1]:.1f}]"
    )
    print(f"  fps values: {stats['fps_seen']}")

    print("  per-config breakdown  (object, pusher, obstacle_level):")
    for cfg, count in sorted(stats["configs"].items()):
        lengths = stats["per_config_lengths"][cfg]
        finals = np.asarray(stats["per_config_final_cov"][cfg])
        successes = int((finals >= success_threshold).sum())
        print(
            f"    {cfg}: n={count:4d}  "
            f"len mean={np.mean(lengths):5.1f}  "
            f"final_cov mean={finals.mean():.3f}  "
            f"success(@{success_threshold:.2f})={successes}/{count}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="directory of episode_*.zarr")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.6,
        help="threshold on final coverage used to count successes (default: 0.6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = _summarize(Path(args.dataset))
    _print(stats, success_threshold=args.success_threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
