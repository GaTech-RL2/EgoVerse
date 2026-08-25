"""Merge one completed multi-CPU generation cell into its final directory."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from Tsimulation.sim_v2.generate.prepare_control_gap_shards import (
    episode_dirs,
    is_committed_episode,
    quarantine_incomplete,
)

_INDEX = re.compile(r"^(.*_)(\d{6})(\.zarr)$")


def indexed_name(name: str, index: int) -> str:
    match = _INDEX.match(name)
    if match is None:
        raise ValueError(f"unrecognized episode name: {name}")
    return f"{match.group(1)}{index:06d}{match.group(3)}"


def manifest_row(path: Path, task: int) -> list[str]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line]
    matches = [row for row in rows if int(row[0]) == task]
    if len(matches) != 1:
        raise RuntimeError(f"task {task} has {len(matches)} rows in {path}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1000)
    args = parser.parse_args()

    (_cell, mode, agent, source_count_s, base_count_s, remaining_s,
     shard_count_s) = manifest_row(args.cells, args.task)
    source_count = int(source_count_s)
    base_count = int(base_count_s)
    expected_generated = int(remaining_s)
    shard_count = int(shard_count_s)
    source_names = {
        episode.name for episode in episode_dirs(args.seeds / mode / agent / "T")
    }
    if len(source_names) != source_count:
        raise RuntimeError(
            f"source drift for {mode}/{agent}: {len(source_names)} != {source_count}"
        )

    destination = args.data / mode / agent / "T"
    quarantine_incomplete(
        destination,
        args.quarantine / "merge" / mode / agent / "T",
    )
    existing = [ep for ep in episode_dirs(destination) if is_committed_episode(ep)]
    if not base_count <= len(existing) <= args.target:
        raise RuntimeError(
            f"invalid destination count for {mode}/{agent}: "
            f"base={base_count} current={len(existing)} target={args.target}"
        )
    candidates: list[Path] = []
    for shard in range(shard_count):
        shard_dir = args.shards / mode / agent / f"shard{shard}" / agent / "T"
        incomplete = [ep for ep in episode_dirs(shard_dir)
                      if not is_committed_episode(ep)]
        if incomplete:
            raise RuntimeError(
                f"incomplete shard stores for {mode}/{agent}/shard{shard}: "
                f"{[ep.name for ep in incomplete]}"
            )
        candidates.extend(
            ep for ep in episode_dirs(shard_dir) if ep.name not in source_names
        )
    if len(candidates) != expected_generated:
        raise RuntimeError(
            f"shard count mismatch for {mode}/{agent}: "
            f"{len(candidates)} != {expected_generated}"
        )

    indices = []
    for episode in existing:
        match = _INDEX.match(episode.name)
        if match is None:
            raise ValueError(f"unrecognized destination episode: {episode.name}")
        indices.append(int(match.group(2)))
    next_index = max(indices, default=-1) + 1
    needed = args.target - len(existing)
    for episode in candidates[:needed]:
        destination_path = destination / indexed_name(episode.name, next_index)
        if destination_path.exists():
            raise FileExistsError(destination_path)
        shutil.move(str(episode), str(destination_path))
        next_index += 1

    final = [ep for ep in episode_dirs(destination) if is_committed_episode(ep)]
    if len(final) != args.target:
        raise RuntimeError(
            f"final count for {mode}/{agent}: {len(final)} != {args.target}"
        )
    print(
        f"MERGED mode={mode} agent={agent} base={base_count} "
        f"generated={needed} surplus_unmerged={len(candidates) - needed} "
        f"total={len(final)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
