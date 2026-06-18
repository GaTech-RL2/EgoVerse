"""Flatten every annotation action-segment across all eva episodes into one table.

One row per parsed pick/place/dump/return segment, with all structured fields
(see extract_action_groups for definitions) plus episode/scene/frame bounds and
the raw text. No grouping -- this is the queryable substrate. Load with pandas
and count episodes/segments by any field combination, e.g.

    import pandas as pd
    df = pd.read_csv("egomimic/ricl/episode_lists/action_segments.csv")
    # how many EPISODES have a blue-marker side-grip pick?
    q = df[(df.object=="blue marker") & (df.verb=="pick") & (df.grip=="side")]
    q.episode.nunique(), len(q)                       # episodes, segments
    # objects by episode count:
    df.groupby("object").episode.nunique().sort_values(ascending=False)
    # crosstab object x grasp_loc (segment counts):
    pd.crosstab(df.object, df.grasp_loc)

    python -m egomimic.ricl.scripts.build_action_table [--scene SCENE] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import zarr

from egomimic.ricl.scripts.extract_action_groups import (
    ALL_FIELDS,
    episode_segments,
    parse_segment,
)

# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "episode_lists")
MASTER = os.path.join(LISTS, "eva_episode_objects.json")
COLUMNS = [
    "episode",
    "scene",
    "task_description",
    "start",
    "end",
    "frames",
    *ALL_FIELDS,
    "text",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--zarr-root", default="/storage/project/r-dxu345-0/agao81/pick_place/{hash}"
    )
    ap.add_argument("--master", default=MASTER)
    ap.add_argument("--scene", help="restrict to one scene; default = all eva episodes")
    ap.add_argument("--out", default=os.path.join(LISTS, "action_segments.csv"))
    ap.add_argument(
        "--corrections",
        default=os.path.join(LISTS, "action_corrections.csv"),
        help="LLM-annotator overlay applied after parsing; '' to skip",
    )
    args = ap.parse_args()

    master = json.load(open(args.master))["episodes"]
    hashes = [
        h for h, v in master.items() if args.scene is None or v["scene"] == args.scene
    ]

    rows, n_unparsed, n_noann = [], 0, 0
    for h in hashes:
        store = zarr.open(args.zarr_root.format(hash=h), mode="r")
        if "annotations" not in store or store["annotations"].shape[0] == 0:
            n_noann += 1
            continue
        sc, td = master[h]["scene"], master[h]["task_description"]
        for s, e, txt in episode_segments(store):
            rec = parse_segment(txt)
            if rec is None:
                n_unparsed += 1
                continue
            rows.append(
                {
                    "episode": h,
                    "scene": sc,
                    "task_description": td,
                    "start": s,
                    "end": e,
                    "frames": e - s,
                    **rec,
                }
            )

    # LLM-annotator overlay: apply hand-verified corrections the regex can't infer
    # (e.g. source-only placements with no destination relation). Keeps rebuilds gold.
    n_corr = 0
    if args.corrections and os.path.isfile(args.corrections):
        index = {(r["episode"], r["start"], r["end"]): r for r in rows}
        with open(args.corrections, newline="") as f:
            for c in csv.DictReader(f):
                if c["new_value"] in ("UNSURE", ""):
                    continue
                row = index.get((c["episode"], int(c["start"]), int(c["end"])))
                if row is not None and row.get(c["field"]) != c["new_value"]:
                    row[c["field"]] = c["new_value"]
                    n_corr += 1

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    eps = {r["episode"] for r in rows}
    verbs = {}
    for r in rows:
        verbs[r["verb"]] = verbs.get(r["verb"], 0) + 1
    objs = len({r["object"] for r in rows if r["object"] != "-"})
    print(f"wrote {args.out}")
    print(
        f"  {len(rows)} action segments | {len(eps)} episodes | {n_noann} episodes had no annotations"
    )
    print(f"  distinct objects: {objs} | by verb: {verbs}")
    print(f"  LLM corrections overlaid: {n_corr}")
    if n_unparsed:
        print(f"  {n_unparsed} segments could not be parsed (skipped)")
    print("\nquery it:\n  import pandas as pd; df = pd.read_csv('" + args.out + "')")
    print("  df.groupby('object').episode.nunique().sort_values(ascending=False)")


if __name__ == "__main__":
    main()
