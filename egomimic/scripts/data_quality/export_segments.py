"""Export episode segments + annotations to JSON for LLM relabeling.

Read-only consumer of the episode registry and zarr stores. For every episode
matching ``--lab``/``--task`` (the same SQL filter as the
``mecka_pi_fold_clothes_freeform`` data config), dump its annotation spans
("segments") so a Claude Code workflow can distill the verbose freeform text into
a small set of canonical task labels. The relabel map produced from this JSON is
fed back into training via ``annotation_override_path`` on the data config (no
zarr rewrite). See ``relabel_utils.py`` and the relabel workflow.

This never writes to the SQL table or to any zarr store.

----------------------------------------------------------------------------
Usage  (login node, cheap — reads only the tiny per-episode ``annotations`` array)
----------------------------------------------------------------------------
    source emimic/bin/activate

    # all mecka folding_clothes episodes (matches the freeform config):
    python -m egomimic.scripts.data_quality.export_segments \
        --lab mecka --task folding_clothes --out segments.json

    # quick dry run on 5 episodes:
    python -m egomimic.scripts.data_quality.export_segments --limit 5 --out /tmp/seg5.json

    # restrict to the 179 human-accepted episodes (combined quality gate):
    python -m egomimic.scripts.data_quality.export_segments --accepted-only --out segments.json

Output ``segments.json`` is a list of:
    {"episode_hash": str, "lab": str, "task": str, "total_frames": int,
     "segments": [{"start_idx": int, "end_idx": int, "text": str}, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode
from egomimic.scripts.data_quality.review_episodes import (
    DEFAULT_DATASET_DIR,
    DecisionLog,
    discover_episodes,
    read_annotations,
    resolve_episode_dir,
)
from egomimic.utils.aws.aws_data_utils import load_env


def _accepted_hashes(decisions_path: Path) -> set[str]:
    """Hashes marked ``accept`` (latest decision wins) in a review JSONL."""
    log = DecisionLog(decisions_path)
    return {h for h, d in log.decisions.items() if d == "accept"}


def export_segments(
    lab: str,
    task: str,
    out_path: Path,
    dataset_dir: Path,
    cache_dir: Path,
    extra_filters: list[str] | None = None,
    limit: int | None = None,
    accepted_only: bool = False,
    decisions_path: Path | None = None,
    debug: int | None = None,
) -> list[dict]:
    """Resolve matched episodes, read their annotation spans, write ``out_path``."""
    load_env()
    ordered, hash_to_s3, _hash_to_task = discover_episodes(
        lab, task, extra_filters or [], debug
    )
    print(f"Matched {len(ordered)} episodes for lab={lab!r} task={task!r}")

    if accepted_only:
        decisions_path = Path(
            decisions_path
            or (Path("./data_quality_review") / f"{lab}_{task}_decisions.jsonl")
        )
        accepted = _accepted_hashes(decisions_path)
        ordered = [h for h in ordered if h in accepted]
        print(f"Restricted to {len(ordered)} accepted episodes (from {decisions_path})")

    if limit is not None:
        ordered = ordered[:limit]
        print(f"Limited to first {len(ordered)} episodes")

    if not ordered:
        raise SystemExit("No episodes to export.")

    cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    n_segments = 0
    skipped: list[str] = []
    for i, h in enumerate(ordered):
        try:
            zarr_dir = resolve_episode_dir(dataset_dir, cache_dir, h, hash_to_s3[h])
            total_frames = int(len(ZarrEpisode(zarr_dir)))
            segments = read_annotations(zarr_dir)
        except Exception as exc:  # noqa: BLE001 — skip unreadable episodes, keep going
            print(f"[{i + 1}/{len(ordered)}] SKIP {h}: {exc}")
            skipped.append(h)
            continue
        records.append(
            {
                "episode_hash": h,
                "lab": lab,
                "task": task,
                "total_frames": total_frames,
                "segments": [
                    {
                        "start_idx": s["start_idx"],
                        "end_idx": s["end_idx"],
                        "text": s["text"],
                    }
                    for s in segments
                ],
            }
        )
        n_segments += len(segments)
        if (i + 1) % 25 == 0:
            print(f"  ...exported {i + 1}/{len(ordered)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)

    empties = sum(1 for r in records if not r["segments"])
    print(
        f"Wrote {len(records)} episodes ({n_segments} segments; "
        f"{empties} with no annotations; {len(skipped)} skipped) -> {out_path}"
    )
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lab", default="mecka")
    p.add_argument("--task", default="folding_clothes")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./data_quality_review/segments.json"),
        help="Output JSON path (default ./data_quality_review/segments.json).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Existing (read-only) shared store of downloaded episodes "
        "(= paths.dataset_dir). Episodes found here are never re-downloaded.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./qc_cache"),
        help="Writable fallback cache for episodes NOT in --dataset-dir.",
    )
    p.add_argument(
        "--extra-filter",
        action="append",
        default=[],
        help="Extra filter lambda string (repeatable), e.g. "
        "\"lambda row: row['operator'] == '<hash>'\".",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Export only the first N episodes."
    )
    p.add_argument(
        "--accepted-only",
        action="store_true",
        help="Restrict to hashes marked 'accept' in the review decisions JSONL.",
    )
    p.add_argument(
        "--decisions-file",
        type=Path,
        default=None,
        help="Decisions JSONL for --accepted-only "
        "(default ./data_quality_review/<lab>_<task>_decisions.jsonl).",
    )
    p.add_argument(
        "--debug", type=int, default=None, help="Limit SQL match to first N episodes."
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    export_segments(
        lab=a.lab,
        task=a.task,
        out_path=a.out.resolve(),
        dataset_dir=a.dataset_dir.resolve(),
        cache_dir=a.cache_dir.resolve(),
        extra_filters=a.extra_filter,
        limit=a.limit,
        accepted_only=a.accepted_only,
        decisions_path=a.decisions_file,
        debug=a.debug,
    )


if __name__ == "__main__":
    main()
