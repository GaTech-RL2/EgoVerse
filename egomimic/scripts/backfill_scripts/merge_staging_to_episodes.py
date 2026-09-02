#!/usr/bin/env python3
"""Merge a staging table (built by stage_processed_folder.py) into app.episodes,
with guards. DRY-RUN by default — nothing is written to app.episodes unless you
pass --apply.

A staging row is eligible to merge only if BOTH:
  1. sufficient columns are filled  — every column in --required-columns is
     NON-NULL (default: episode_hash, embodiment, task, num_frames,
     zarr_processed_path). Rows missing any are reported as INSUFFICIENT and skipped.
  2. no conflicting hash            — its episode_hash is not already in
     app.episodes. Conflicting rows are reported and skipped (never overwritten).

Merged rows carry the staged metadata (episode_hash, embodiment, task,
task_description, num_frames, zarr_processed_path) and PRESERVE the staged
``created_at`` (the recording time parsed from the hash). ``updated_at`` gets the
merge time (DEFAULT now()); the eval/flag columns get the standard defaults
(is_deleted=false, is_eval=false, eval_score=-1, eval_success=true), matching a
normally-ingested row. ``lab`` is set from the staged ``source_folder`` and ``zarr_mp4_path`` is derived
from the zarr path (the sibling ``<episode_hash>.mp4`` preview); operator/scene/objects/
license/segments are left NULL (the zarr metadata does not provide them). ON CONFLICT (episode_hash) DO NOTHING
is applied as a final safety net.

Usage:
    # preview (safe):
    python -m egomimic.scripts.backfill_scripts.merge_staging_to_episodes \
        --staging-table app.staging_microagi
    # actually merge:
    python -m egomimic.scripts.backfill_scripts.merge_staging_to_episodes \
        --staging-table app.staging_microagi --apply
"""

from __future__ import annotations

import argparse
import re

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

DEFAULT_REQUIRED = ["episode_hash", "embodiment", "task", "num_frames", "zarr_processed_path"]

# staged column -> app.episodes column (identity here, but explicit so the
# mapping is auditable). Only these are copied; everything else uses DB defaults.
_MERGE_COLUMNS = [
    "episode_hash", "embodiment", "task", "task_description",
    "num_frames", "zarr_processed_path", "created_at",
]

# app.episodes column -> SQL expression over the staging row. These are NOT
# identity copies, so they are listed separately to keep the mapping auditable:
#   lab            <- source_folder (the processed_v3/<folder> the episode came
#                     from, i.e. the contributing lab/vendor). `lab` is populated
#                     for ~all existing rows and is the field split-builders
#                     filter on, so leaving it NULL would make merged episodes
#                     unfilterable by source.
#   zarr_mp4_path  <- the sibling preview video required by CONTRIBUTING_DATA
#                     (`<episode_hash>.mp4` next to `<episode_hash>.zarr`).
#                     NULL when the stored path is not a .zarr, so a malformed
#                     path never yields a bogus preview path.
_DERIVED_COLUMNS = [
    ("lab", "s.source_folder"),
    (
        "zarr_mp4_path",
        "CASE WHEN rtrim(s.zarr_processed_path, '/') LIKE '%.zarr' "
        "THEN regexp_replace(rtrim(s.zarr_processed_path, '/'), '\\.zarr$', '.mp4') "
        "ELSE NULL END",
    ),
]


def _ident_ok(name):
    # guard against SQL injection via --staging-table / --required-columns
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging-table", required=True,
                    help="Fully-qualified staging table, e.g. app.staging_microagi.")
    ap.add_argument("--required-columns", default=",".join(DEFAULT_REQUIRED),
                    help="Comma-separated columns that must be NON-NULL to merge a row.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually INSERT into app.episodes (default is a dry-run preview).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Merge at most N eligible rows (with --apply).")
    args = ap.parse_args()

    staging = args.staging_table
    required = [c.strip() for c in args.required_columns.split(",") if c.strip()]
    if not _ident_ok(staging) or not all(_ident_ok(c) for c in required):
        raise SystemExit("Invalid identifier in --staging-table / --required-columns.")

    engine = create_default_engine()

    # sufficiency predicate + not-already-present predicate
    suff = " AND ".join(f"s.{c} IS NOT NULL" for c in required)
    not_present = ("NOT EXISTS (SELECT 1 FROM app.episodes e "
                   "WHERE e.episode_hash = s.episode_hash)")

    with engine.connect() as conn:
        # sanity: staging table + episode_hash column exist
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = split_part(:t,'.',1) "
            "AND table_name = split_part(:t,'.',2)"), {"t": staging}).all()}
        if not cols:
            raise SystemExit(f"Staging table {staging} not found.")
        missing = [c for c in required if c not in cols]
        if missing:
            raise SystemExit(f"{staging} is missing required columns: {missing}")

        total = conn.execute(text(f"SELECT COUNT(*) FROM {staging}")).scalar()
        insufficient = conn.execute(text(
            f"SELECT COUNT(*) FROM {staging} s WHERE NOT ({suff})")).scalar()
        conflicts = conn.execute(text(
            f"SELECT COUNT(*) FROM {staging} s WHERE ({suff}) AND NOT ({not_present})")).scalar()
        eligible = conn.execute(text(
            f"SELECT COUNT(*) FROM {staging} s WHERE ({suff}) AND {not_present}")).scalar()

    print(f"Merge preview: {staging} -> app.episodes")
    print(f"  required (must be non-null): {required}")
    print(f"  total staged rows      : {total}")
    print(f"  INSUFFICIENT (skipped) : {insufficient}")
    print(f"  CONFLICT already exists: {conflicts}  (skipped, never overwritten)")
    print(f"  ELIGIBLE to merge      : {eligible}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to merge the eligible rows.")
        engine.dispose()
        return 0

    select_cols = ", ".join(
        [f"s.{c}" for c in _MERGE_COLUMNS] + [expr for _, expr in _DERIVED_COLUMNS]
    )
    insert_cols = ", ".join(
        _MERGE_COLUMNS
        + [col for col, _ in _DERIVED_COLUMNS]
        + ["is_deleted", "is_eval", "eval_score", "eval_success"]
    )
    limit_sql = f"LIMIT {int(args.limit)}" if args.limit else ""
    insert_sql = text(f"""
        INSERT INTO app.episodes ({insert_cols})
        SELECT {select_cols}, FALSE, FALSE, -1, TRUE
        FROM {staging} s
        WHERE ({suff}) AND {not_present}
        {limit_sql}
        ON CONFLICT (episode_hash) DO NOTHING
    """)
    with engine.begin() as conn:
        result = conn.execute(insert_sql)
    print(f"\nAPPLIED: inserted {result.rowcount} rows into app.episodes.")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
