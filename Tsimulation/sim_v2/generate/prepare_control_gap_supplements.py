"""Freeze unfinished primary shards and split their remaining quotas again."""

from __future__ import annotations

import argparse
from pathlib import Path

from Tsimulation.sim_v2.generate.prepare_control_gap_shards import (
    episode_dirs,
    is_committed_episode,
    quarantine_incomplete,
    split_quota,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-manifest", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()
    if args.parts <= 0:
        parser.error("--parts must be positive")

    supplement_rows: list[str] = []
    unfinished_rows: list[str] = []
    supplement_task = 0
    consolidate_task = 0
    quarantined = 0
    for line in args.shard_manifest.read_text().splitlines():
        if not line:
            continue
        (original_task_s, cell_s, mode, agent, original_shard_s, quota_s,
         source_count_s, _base_count_s) = line.split("\t")
        original_task = int(original_task_s)
        log_path = args.logs / f"ps_gap4x_3719195_{original_task}.out"
        if log_path.exists() and "\nCOMPLETE " in "\n" + log_path.read_text(
            errors="replace"
        ):
            # The primary job committed and reported its exact target. Avoid a
            # costly metadata rescan of shards that do not need supplements.
            continue
        quota = int(quota_s)
        source_count = int(source_count_s)
        original_shard = int(original_shard_s)
        source_dir = args.seeds / mode / agent / "T"
        source_names = {episode.name for episode in episode_dirs(source_dir)}
        if len(source_names) != source_count:
            raise RuntimeError(f"source drift for {mode}/{agent}")
        destination = (
            args.shards / mode / agent / f"shard{original_shard}" / agent / "T"
        )
        moved = quarantine_incomplete(
            destination,
            args.quarantine / f"task{original_task}" / agent / "T",
        )
        quarantined += len(moved)
        generated = sum(
            episode.name not in source_names
            for episode in episode_dirs(destination)
            if is_committed_episode(episode)
        )
        if generated > quota:
            raise RuntimeError(
                f"task {original_task} generated {generated} above quota {quota}"
            )
        remaining = quota - generated
        if remaining == 0:
            continue
        parts = min(args.parts, remaining)
        unfinished_rows.append(
            f"{consolidate_task}\t{original_task}\t{cell_s}\t{mode}\t{agent}\t"
            f"{original_shard}\t{source_count}\t{generated}\t{quota}\t"
            f"{remaining}\t{parts}"
        )
        for part, part_quota in enumerate(split_quota(remaining, parts)):
            supplement_rows.append(
                f"{supplement_task}\t{original_task}\t{cell_s}\t{mode}\t"
                f"{agent}\t{original_shard}\t{part}\t{part_quota}\t"
                f"{source_count}\t{generated}\t{quota}"
            )
            supplement_task += 1
        consolidate_task += 1

    args.manifests.mkdir(parents=True, exist_ok=True)
    (args.manifests / "supplements.tsv").write_text(
        "\n".join(supplement_rows) + ("\n" if supplement_rows else ""),
        encoding="utf-8",
    )
    (args.manifests / "unfinished.tsv").write_text(
        "\n".join(unfinished_rows) + ("\n" if unfinished_rows else ""),
        encoding="utf-8",
    )
    print(
        f"unfinished_primary_shards={len(unfinished_rows)} "
        f"supplement_tasks={len(supplement_rows)} "
        f"quarantined_partial={quarantined}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
