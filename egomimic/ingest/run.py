"""The ingest box's loop: a watcher pass, then the committer, every few minutes.

    python -m egomimic.ingest.run --lance s3://rldb/lance --work /var/lib/ingest

The pass converts and hands off deliveries (egomimic/ingest/watch.py); the committer
(egomimic/scripts/lance_commit.py, the only Lance writer) then commits whatever is staged,
one source at a time and annotation changes last. A STOP file in --work ends the loop after
the current round; --once runs a single round.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from egomimic.ingest.watch import Busy, add_pass_args, pass_from_args


def commit_round(lance_staging: str, lance_root: str, work) -> dict:
    """lance_commit over every source with staged work, annotations last."""
    from egomimic.rldb.lance import commit
    from egomimic.scripts.lance_commit import ANNOTATIONS, commit_staged

    sources = sorted(commit.list_dirs(lance_staging), key=lambda s: s == ANNOTATIONS)
    out = {}
    for source in sources:
        report = commit_staged(lance_staging, source, lance_root, work / "commit")
        out[source] = {
            "tags": [b["tag"] for b in report["batches"]],
            "rejected": sorted(report["rejected"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    from egomimic.ingest.convert_job import store_from_args

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_pass_args(ap)
    ap.add_argument("--interval", type=float, default=300, help="seconds per round")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    store = store_from_args(args)
    while True:
        started = time.monotonic()
        try:
            summary = pass_from_args(args, store).__dict__
        except Busy as e:
            summary = {"busy": str(e)}
        except Exception as e:  # keep the loop alive; the next round retries
            summary = {"error": f"{type(e).__name__}: {e}"}
        try:
            committed = commit_round(args.lance_staging, args.lance, args.work)
        except Exception as e:
            committed = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps({"pass": summary, "commit": committed}), flush=True)
        if args.once or (args.work / "STOP").exists():
            return 0
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    sys.exit(main())
