"""Group action segments by a set of categories and rank groups by total frames.

Reads ``episode_lists/action_segments.csv`` (one row per ``[start, end)`` segment;
see ``build_action_table.py``). You pick the categories to group on with ``--by``
(e.g. ``object,hand,grip``); rows that match on **every** one of those columns form
a group. Groups are ranked by number of segments and the top ``--top`` are printed.
Optional ``--where col=value`` filters narrow the pool before grouping.

A ``-`` in a category means "field doesn't apply / unknown" (e.g. ``grip`` on a
place, ``object`` on a return). By default such a segment is treated as a **wildcard**
on that column: it is counted in *every* concrete group whose other (non-``-``) fields
it matches (so one ``-`` segment can land in several groups; its frames/episodes are
counted in each). A segment whose ``-`` matches no concrete group stays in its own
literal ``-`` group, so nothing is lost. Pass ``--no-expand-dash`` to instead treat
``-`` as a plain literal value (one ``-`` group per column).

    # top 10 (object, hand, grip) combos by segment count
    python -m egomimic.ricl.scripts.group_segments --by object,hand,grip

    # only pick segments, grouped by object + grasp location, top 20
    python -m egomimic.ricl.scripts.group_segments \
        --by object,grasp_loc --where verb=pick --top 20
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd

DASH = "-"

# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "episode_lists")
DEFAULT_CSV = os.path.join(LISTS, "action_segments.csv")
DEFAULT_BY = "object,hand,grip"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path to action_segments.csv")
    ap.add_argument(
        "--by", default=DEFAULT_BY, help="comma list of categories to group on"
    )
    ap.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COL=VALUE",
        help="filter rows before grouping (repeatable); exact string match",
    )
    ap.add_argument("--top", type=int, default=30, help="number of groups to print")
    ap.add_argument(
        "--expand-dash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="treat '-' as a wildcard, folding the segment into every matching "
        "concrete group (default); --no-expand-dash treats '-' as a literal value",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str).fillna("-")
    df["frames"] = pd.to_numeric(df["frames"], errors="coerce").fillna(0).astype(int)

    by = [c.strip() for c in args.by.split(",") if c.strip()]
    cols = set(df.columns)
    bad = [c for c in by if c not in cols]
    if bad:
        raise SystemExit(f"unknown category(ies) {bad}; choose from {sorted(cols)}")

    for w in args.where:
        if "=" not in w:
            raise SystemExit(f"--where expects COL=VALUE, got {w!r}")
        col, val = w.split("=", 1)
        col = col.strip()
        if col not in cols:
            raise SystemExit(
                f"unknown --where column {col!r}; choose from {sorted(cols)}"
            )
        df = df[df[col] == val.strip()]

    if df.empty:
        raise SystemExit("no segments match the given --where filters")

    # universe of concrete group keys = rows with no '-' in any --by column. A '-'
    # segment fans out across the concrete keys it matches; with --no-expand-dash the
    # universe is empty, so every row (dash or not) keys on its literal values.
    by_vals = df[by].to_numpy()
    frames = df["frames"].to_numpy()
    eps = df["episode"].to_numpy()
    if args.expand_dash:
        concrete = df.loc[(df[by] != DASH).all(axis=1), by].drop_duplicates()
        universe = [tuple(k) for k in concrete.to_numpy()]
    else:
        universe = []

    acc: dict[tuple, dict] = defaultdict(lambda: {"segs": 0, "frames": 0, "eps": set()})
    for rowvals, fr, ep in zip(by_vals, frames, eps):
        fixed = [(j, v) for j, v in enumerate(rowvals) if v != DASH]
        if len(fixed) == len(by):  # fully concrete -> only its own key
            keys = [tuple(rowvals)]
        else:  # has '-' -> every concrete key matching the fixed fields
            keys = [k for k in universe if all(k[j] == v for j, v in fixed)]
            if not keys:  # matched nothing -> keep the literal '-' group
                keys = [tuple(rowvals)]
        for k in keys:
            a = acc[k]
            a["segs"] += 1
            a["frames"] += int(fr)
            a["eps"].add(ep)

    ranked = sorted(acc.items(), key=lambda kv: -kv[1]["segs"])

    sig_w = max([len("|".join(by))] + [len(" | ".join(map(str, k))) for k in acc])
    print(f"csv={args.csv}")
    print(
        f"by={by}  where={args.where or '-'}  expand_dash={args.expand_dash}  "
        f"segments={len(df)}  groups={len(acc)}\n"
    )
    print(f"  {'|'.join(by):{sig_w}s} {'segs':>5} {'frames':>8} {'eps':>4}")
    for k, a in ranked[: args.top]:
        sig = " | ".join(str(v) for v in k)
        print(f"  {sig:{sig_w}s} {a['segs']:>5} {a['frames']:>8} {len(a['eps']):>4}")


if __name__ == "__main__":
    main()
