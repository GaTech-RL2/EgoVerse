"""Move completed supplement outputs into one original primary shard."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from Tsimulation.sim_v2.generate.merge_control_gap_shards import manifest_row
from Tsimulation.sim_v2.generate.prepare_control_gap_shards import (
    episode_dirs,
    is_committed_episode,
)

_INDEX = re.compile(r"^(.*_)(\d{6})(\.zarr)$")


def indexed_name(name: str, index: int) -> str:
    match = _INDEX.match(name)
    if match is None:
        raise ValueError(f"unrecognized episode name: {name}")
    return f"{match.group(1)}{index:06d}{match.group(3)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--unfinished", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--supplements", type=Path, required=True)
    args = parser.parse_args()

    (_merge_task, original_task_s, _cell, mode, agent, original_shard_s,
     source_count_s, generated_s, quota_s, remaining_s, parts_s) = manifest_row(
        args.unfinished, args.task
    )
    original_task = int(original_task_s)
    original_shard = int(original_shard_s)
    source_count = int(source_count_s)
    generated_at_freeze = int(generated_s)
    quota = int(quota_s)
    remaining = int(remaining_s)
    parts = int(parts_s)
    source_names = {
        episode.name
        for episode in episode_dirs(args.seeds / mode / agent / "T")
    }
    if len(source_names) != source_count:
        raise RuntimeError(f"source drift for {mode}/{agent}")

    destination = (
        args.shards / mode / agent / f"shard{original_shard}" / agent / "T"
    )
    existing = [episode for episode in episode_dirs(destination)
                if is_committed_episode(episode)]
    current_generated = sum(ep.name not in source_names for ep in existing)
    if not generated_at_freeze <= current_generated <= quota:
        raise RuntimeError(
            f"invalid primary shard count for task {original_task}: "
            f"freeze={generated_at_freeze} current={current_generated} "
            f"quota={quota}"
        )

    candidates: list[Path] = []
    for part in range(parts):
        part_dir = (
            args.supplements / f"task{original_task}" / f"part{part}" / agent / "T"
        )
        incomplete = [ep for ep in episode_dirs(part_dir)
                      if not is_committed_episode(ep)]
        if incomplete:
            raise RuntimeError(f"incomplete supplement stores: {incomplete}")
        candidates.extend(
            ep for ep in episode_dirs(part_dir) if ep.name not in source_names
        )
    needed = quota - current_generated
    if len(candidates) < needed:
        raise RuntimeError(
            f"supplement count for task {original_task}: "
            f"available={len(candidates)} needed={needed} "
            f"freeze_remaining={remaining}"
        )

    indices = []
    # Include incomplete/reserved names when selecting the next index. A
    # stopped writer may have left an uncommitted directory after the freeze;
    # it must not collide with a valid supplement being moved here.
    for episode in episode_dirs(destination):
        match = _INDEX.match(episode.name)
        if match is None:
            raise ValueError(f"unrecognized episode name: {episode.name}")
        indices.append(int(match.group(2)))
    next_index = max(indices, default=-1) + 1
    for episode in candidates[:needed]:
        target = destination / indexed_name(episode.name, next_index)
        while target.exists():
            next_index += 1
            target = destination / indexed_name(episode.name, next_index)
        shutil.move(str(episode), str(target))
        next_index += 1

    final_generated = sum(
        ep.name not in source_names
        for ep in episode_dirs(destination)
        if is_committed_episode(ep)
    )
    if final_generated != quota:
        raise RuntimeError(
            f"final primary quota for task {original_task}: "
            f"{final_generated} != {quota}"
        )
    print(
        f"CONSOLIDATED original_task={original_task} mode={mode} agent={agent} "
        f"added={needed} surplus_unmerged={len(candidates) - needed} "
        f"generated={final_generated}/{quota}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
