"""Staff review from a terminal: list deliveries, read a stage's diff, approve or decline.

    python -m egomimic.ingest.review list [--vendor acme] [--state ready]
    python -m egomimic.ingest.review show acme/episode_0001
    python -m egomimic.ingest.review approve acme/episode_0001 ... --who simar
    python -m egomimic.ingest.review approve --all-ready --vendor acme --who simar
    python -m egomimic.ingest.review decline acme/episode_0001 --who simar --comment "..."

A decision is bound to the stage shown in the delivery's status, as the dashboard's are:
the watcher acts on it only while that stage is current, so a delivery re-uploaded after
it was reviewed goes back for review.
"""

from __future__ import annotations

import argparse
import json
import sys

from egomimic.ingest.convert_job import add_store_args, store_from_args
from egomimic.ingest.store import Ref, Store


def _ref(name: str) -> Ref:
    vendor, _, delivery = name.partition("/")
    return Ref(vendor, delivery)


def rows(store: Store, vendor: str | None, state: str | None) -> list[dict]:
    out = []
    for ref, st in sorted(store.statuses(vendor).items()):
        if state and st.get("state") != state:
            continue
        report = st.get("report") or {}
        out.append(
            {
                "delivery": f"{ref.vendor}/{ref.delivery_id}",
                "state": st.get("state", ""),
                "frames": report.get("num_frames", ""),
                "labels": report.get("labels", ""),
                "head_valid": (report.get("valid_frac") or {}).get("head", ""),
                "action": (st.get("diff") or {}).get("action", ""),
            }
        )
    return out


def decide(
    store: Store, refs: list[Ref], decision: str, who: str, comment: str = ""
) -> dict[str, str]:
    """Record `decision` for each ref's current stage; returns ref -> outcome."""
    out = {}
    for ref in refs:
        st = store.get_status(ref) or {}
        name = f"{ref.vendor}/{ref.delivery_id}"
        if st.get("state") != "ready" or not st.get("stage"):
            out[name] = f"skipped: {st.get('state', 'unknown')}, nothing to review"
            continue
        store.add_decision(ref, decision, who, comment, stage=st["stage"])
        out[name] = decision
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_store_args(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--vendor")
    p.add_argument("--state")
    p = sub.add_parser("show")
    p.add_argument("delivery")
    for name in ("approve", "decline"):
        p = sub.add_parser(name)
        p.add_argument("deliveries", nargs="*")
        p.add_argument("--all-ready", action="store_true")
        p.add_argument("--vendor")
        p.add_argument("--who", required=True)
        p.add_argument("--comment", default="", required=name == "decline")
    args = ap.parse_args(argv)
    store = store_from_args(args)

    if args.cmd == "list":
        for r in rows(store, args.vendor, args.state):
            print("\t".join(str(r[k]) for k in r))
        return 0
    if args.cmd == "show":
        st = store.get_status(_ref(args.delivery))
        if st is None:
            print(f"no status for {args.delivery}")
            return 1
        keys = ("state", "problems", "error", "report", "diff", "decision", "history")
        print(json.dumps({k: st.get(k) for k in keys}, indent=1, default=str))
        return 0
    refs = [_ref(d) for d in args.deliveries]
    if args.all_ready:
        refs += [
            ref
            for ref, st in store.statuses(args.vendor).items()
            if st.get("state") == "ready"
        ]
    if not refs:
        ap.error("name deliveries or pass --all-ready")
    out = decide(store, sorted(set(refs)), args.cmd, args.who, args.comment)
    for name, outcome in out.items():
        print(f"{name}\t{outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
