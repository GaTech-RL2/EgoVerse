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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.collect.zarr_writer import ACTION_KEY, REWARD_KEY

_EPISODE_NEW_RE = re.compile(r"^episode_[A-Za-z0-9]+_[A-Za-z0-9]+_obs\d+_\d+\.zarr$")
_EPISODE_OLD_RE = re.compile(r"^episode_\d+\.zarr$")

# (object_shape, pusher_shape, obstacle_level)
Config = tuple[str, str, object]


# ---------------------------------------------------------------------- #
# Types
# ---------------------------------------------------------------------- #


@dataclass
class _ConfigBucket:
    """Per-(object, pusher, obstacles) accumulator filled while iterating."""

    lengths: list[int] = field(default_factory=list)
    final_cov: list[float] = field(default_factory=list)

    def add(self, length: int, final_coverage: float) -> None:
        self.lengths.append(length)
        self.final_cov.append(final_coverage)

    @property
    def count(self) -> int:
        return len(self.lengths)


@dataclass
class DatasetStats:
    """Aggregate stats for a directory of episode_*.zarr stores."""

    dataset: Path
    n_episodes: int
    n_with_data: int
    total_frames: int
    length_min: int
    length_max: int
    length_mean: float
    length_median: float
    final_cov_min: float
    final_cov_max: float
    final_cov_mean: float
    mean_cov_mean: float
    action_lo: list[float]
    action_hi: list[float]
    fps_seen: list[int]
    per_config: dict[Config, _ConfigBucket]


# ---------------------------------------------------------------------- #
# Episode iteration
# ---------------------------------------------------------------------- #


def _list_episodes(dataset: Path) -> list[Path]:
    """Episode subdirectories under `dataset` — accepts both the new
    `episode_<obj>_<pusher>_obs<N>_<idx>.zarr` and legacy
    `episode_NNNNNN.zarr` naming."""
    return [
        entry
        for entry in sorted(dataset.iterdir())
        if entry.is_dir()
        and (_EPISODE_NEW_RE.match(entry.name) or _EPISODE_OLD_RE.match(entry.name))
    ]


def _parse_env_args(metadata: dict) -> dict:
    try:
        return json.loads(metadata.get("task_description", "{}")).get("env_args", {})
    except json.JSONDecodeError:
        return {}


def _config_of(env_args: dict) -> Config:
    return (
        env_args.get("object_shape", "?"),
        env_args.get("pusher_shape", "?"),
        env_args.get("obstacle_level", "?"),
    )


@dataclass
class _EpisodeRecord:
    """One episode's flattened stats — folded into the global summary."""

    length: int
    final_coverage: float
    mean_coverage: float
    action_min: np.ndarray
    action_max: np.ndarray
    fps: int
    config: Config


def _load_one(ep_path: Path) -> _EpisodeRecord | None:
    """Return None for empty / unreadable episodes (skipped in the summary)."""
    store = zarr.open_group(str(ep_path), mode="r")
    attrs = dict(store.attrs)
    total = int(attrs.get("total_frames", 0))
    if total <= 0:
        return None
    actions = np.asarray(store[ACTION_KEY][:total])
    rewards = np.asarray(store[REWARD_KEY][:total]).reshape(-1)
    return _EpisodeRecord(
        length=total,
        final_coverage=float(rewards[-1]),
        mean_coverage=float(rewards.mean()),
        action_min=actions.min(axis=0),
        action_max=actions.max(axis=0),
        fps=int(attrs.get("fps", 0)),
        config=_config_of(_parse_env_args(attrs)),
    )


# ---------------------------------------------------------------------- #
# Summarize
# ---------------------------------------------------------------------- #


def _summarize(dataset: Path) -> DatasetStats:
    ep_paths = _list_episodes(dataset)
    if not ep_paths:
        raise SystemExit(f"no episode_*.zarr stores in {dataset}")

    records = [r for r in (_load_one(p) for p in ep_paths) if r is not None]
    if not records:
        raise SystemExit(f"all episodes in {dataset} are empty")

    lengths = np.asarray([r.length for r in records])
    final_cov = np.asarray([r.final_coverage for r in records])
    mean_cov = np.asarray([r.mean_coverage for r in records])
    action_lo = np.stack([r.action_min for r in records]).min(axis=0)
    action_hi = np.stack([r.action_max for r in records]).max(axis=0)

    per_config: dict[Config, _ConfigBucket] = {}
    for r in records:
        per_config.setdefault(r.config, _ConfigBucket()).add(r.length, r.final_coverage)

    return DatasetStats(
        dataset=dataset,
        n_episodes=len(ep_paths),
        n_with_data=len(records),
        total_frames=int(lengths.sum()),
        length_min=int(lengths.min()),
        length_max=int(lengths.max()),
        length_mean=float(lengths.mean()),
        length_median=float(np.median(lengths)),
        final_cov_min=float(final_cov.min()),
        final_cov_max=float(final_cov.max()),
        final_cov_mean=float(final_cov.mean()),
        mean_cov_mean=float(mean_cov.mean()),
        action_lo=action_lo.tolist(),
        action_hi=action_hi.tolist(),
        fps_seen=sorted({r.fps for r in records}),
        per_config=per_config,
    )


def _print(stats: DatasetStats, success_threshold: float) -> None:
    print(f"dataset: {stats.dataset}")
    print(f"  episodes: {stats.n_episodes}  (with data: {stats.n_with_data})")
    print(f"  total frames: {stats.total_frames}")
    print(
        f"  length: min={stats.length_min}  median={stats.length_median:.0f}  "
        f"max={stats.length_max}  mean={stats.length_mean:.1f}"
    )
    print(
        f"  final coverage: min={stats.final_cov_min:.3f}  "
        f"mean={stats.final_cov_mean:.3f}  max={stats.final_cov_max:.3f}"
    )
    print(f"  mean coverage (per-ep average over frames): {stats.mean_cov_mean:.3f}")
    print(
        f"  action range:  x in [{stats.action_lo[0]:.1f}, {stats.action_hi[0]:.1f}]   "
        f"y in [{stats.action_lo[1]:.1f}, {stats.action_hi[1]:.1f}]"
    )
    print(f"  fps values: {stats.fps_seen}")
    print("  per-config breakdown  (object, pusher, obstacle_level):")
    for cfg, bucket in sorted(stats.per_config.items()):
        finals = np.asarray(bucket.final_cov)
        successes = int((finals >= success_threshold).sum())
        print(
            f"    {cfg}: n={bucket.count:4d}  "
            f"len mean={float(np.mean(bucket.lengths)):5.1f}  "
            f"final_cov mean={finals.mean():.3f}  "
            f"success(@{success_threshold:.2f})={successes}/{bucket.count}"
        )


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #


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
