"""Dump subtask annotations (`segments`) from the legacy DB into a bundled
cache file, so the deployed app can attach subtask annotations without a live
legacy-DB connection (e.g. on Modal, where AWS Secrets Manager creds aren't
available).

Run locally (where ~/.egoverse_env_old + AWS creds work), then commit/redeploy:

    cd ego-rating
    python scripts/dump_segments.py                 # all folding_clothes episodes
    python scripts/dump_segments.py --task dishwashing --task folding_clothes

Writes backend/segments_cache.json: {episode_hash: [{label, start_seconds,
end_seconds}, ...]}. `db._enrich_segments` consults the live legacy DB first,
then this file. Regenerate whenever the pool's task set changes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import legacy_db  # noqa: E402

CACHE_PATH = Path(__file__).resolve().parent.parent / "backend" / "segments_cache.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        action="append",
        default=None,
        help="task name(s) to dump (repeatable; default: folding_clothes)",
    )
    args = ap.parse_args()
    tasks = args.task or ["folding_clothes"]

    eng = legacy_db._get_engine()
    if eng is None:
        raise SystemExit(
            "legacy DB not reachable — check ~/.egoverse_env_old + AWS creds"
        )

    from sqlalchemy import text

    out: dict[str, list] = {}
    with eng.connect() as c:
        rows = c.execute(
            text(
                "SELECT episode_hash, segments FROM app.episodes "
                "WHERE task = ANY(:t) AND segments IS NOT NULL"
            ),
            {"t": tasks},
        ).fetchall()
        for r in rows:
            segs = legacy_db._normalize_segments(r[1])
            if segs:
                out[str(r[0])] = segs

    CACHE_PATH.write_text(json.dumps(out))
    print(f"wrote {len(out)} episodes' segments for tasks {tasks} -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
