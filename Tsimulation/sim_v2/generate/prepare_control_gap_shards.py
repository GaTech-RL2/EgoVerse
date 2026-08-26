"""Freeze current control-gap progress and create collision-free shard manifests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import zarr

from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS, VALID_PUSHERS


def episode_dirs(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("*.zarr") if item.is_dir())


def is_committed_episode(path: Path) -> bool:
    """A completed writer store has metadata and all arrays at full length."""
    try:
        group = zarr.open_group(str(path), mode="r")
        frames = int(group.attrs["total_frames"])
        features = dict(group.attrs["features"])
        # ``annotations`` is a valid sparse side channel and is intentionally
        # empty for generated PushShapes episodes. It is not a frame stream.
        frame_features = [key for key in features if key != "annotations"]
        return frames > 0 and all(
            key in group and int(group[key].shape[0]) >= frames
            for key in frame_features
        )
    except Exception:
        return False


def quarantine_incomplete(path: Path, quarantine: Path) -> list[Path]:
    moved: list[Path] = []
    for episode in episode_dirs(path):
        if is_committed_episode(episode):
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        destination = quarantine / episode.name
        suffix = 1
        while destination.exists():
            destination = quarantine / f"{episode.name}.partial{suffix}"
            suffix += 1
        shutil.move(str(episode), str(destination))
        moved.append(destination)
    return moved


def split_quota(total: int, shards: int) -> list[int]:
    base, remainder = divmod(total, shards)
    return [base + (index < remainder) for index in range(shards)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.target <= 0 or args.shards <= 0:
        parser.error("--target and --shards must be positive")

    shard_rows: list[str] = []
    cell_rows: list[str] = []
    quarantined = 0
    cell_index = 0
    shard_task = 0
    for mode in CONTROL_GAPS:
        for agent in VALID_PUSHERS:
            source_dir = args.seeds / mode / agent / "T"
            destination = args.data / mode / agent / "T"
            source_count = len(episode_dirs(source_dir))
            if source_count <= 0:
                raise RuntimeError(f"no sources for {mode}/{agent}")
            moved = quarantine_incomplete(
                destination,
                args.quarantine / mode / agent / "T",
            )
            quarantined += len(moved)
            complete = [ep for ep in episode_dirs(destination)
                        if is_committed_episode(ep)]
            base_count = len(complete)
            if not source_count <= base_count <= args.target:
                raise RuntimeError(
                    f"invalid count for {mode}/{agent}: sources={source_count} "
                    f"complete={base_count} target={args.target}"
                )
            remaining = args.target - base_count
            quotas = split_quota(remaining, args.shards)
            cell_rows.append(
                f"{cell_index}\t{mode}\t{agent}\t{source_count}\t"
                f"{base_count}\t{remaining}\t{args.shards}"
            )
            for shard, quota in enumerate(quotas):
                if quota <= 0:
                    continue
                shard_rows.append(
                    f"{shard_task}\t{cell_index}\t{mode}\t{agent}\t{shard}\t"
                    f"{quota}\t{source_count}\t{base_count}"
                )
                shard_task += 1
            cell_index += 1

    args.manifests.mkdir(parents=True, exist_ok=True)
    (args.manifests / "cells.tsv").write_text(
        "\n".join(cell_rows) + "\n", encoding="utf-8"
    )
    (args.manifests / "shards.tsv").write_text(
        "\n".join(shard_rows) + "\n", encoding="utf-8"
    )
    print(
        f"cells={len(cell_rows)} shard_tasks={len(shard_rows)} "
        f"quarantined_partial={quarantined} target={args.target}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
