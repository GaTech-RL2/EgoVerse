"""Extract per-object pick-place *segments* (frame ranges) from the annotations.

Most eva episodes are multi-object: ~10 pick-places over ~5 objects each. For a
"same scene + same object" experiment we want only the frames where one target
object is being manipulated. The zarr ``annotations`` already carry segment
boundaries (``start_idx`` / ``end_idx``), so a "unit" for object O is:

    a "Pick up the O ..." segment, extended through the immediately following
    "Put/Place/Dump the O ..." segment (the carry-to-target frames).

Object-agnostic "Return to home" segments are excluded. Multiple O-picks in one
episode yield multiple units (more samples). Output is a manifest the retrieval
builder and dataset can filter on -- nothing is re-written to zarr, so the SQL/S3
resolver and the per-(hash, frame_idx) retrieval keying are untouched.

    python -m egomimic.ricl.scripts.extract_object_segments --object 'blue marker' \
        [--scene pick_place_diverse] \
        [--zarr-root '/storage/project/r-dxu345-0/agao81/pick_place/{hash}'] \
        [--train-list ...] [--valid-list ...]   # else 80/20 by episode

Writes egomimic/ricl/episode_lists/segments_<object_slug>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import zarr

from egomimic.ricl.scripts.validate_object_labels import PICK, canon

# place verbs carry the object to its target; same terminator set as PICK + targets
PLACE = re.compile(
    r"(?:[Pp]ut|[Pp]lace|[Dd]ump) the (.+?)"
    r"(?: from| with| using| at | by | to | on | in | into| onto| behind| between| next to| inside|$)"
)
# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "episode_lists")
MASTER = os.path.join(LISTS, "eva_episode_objects.json")


def segment_object(text: str) -> str | None:
    """Canonical object a segment manipulates, or None (e.g. 'Return to home')."""
    t = text.strip()
    m = PICK.match(t) or PLACE.match(t)
    return canon(m.group(1)) if m else None


def units_for_object(store, obj: str) -> list[tuple[int, int]]:
    """Frame ranges [start, end) where ``obj`` is being manipulated."""
    if "annotations" not in store or store["annotations"].shape[0] == 0:
        return []
    # canonical_segments returns one text per (start,end); recover bounds in order
    by = defaultdict(list)
    for a in store["annotations"][:]:
        d = json.loads(a.decode() if isinstance(a, (bytes, bytearray)) else a)
        by[(d["start_idx"], d["end_idx"])].append(d["text"])
    segs = []
    for s, e in sorted(by):
        txts = by[(s, e)]
        rep = next(
            (t for t in txts if t.lstrip().lower().startswith("pick up the")), txts[0]
        )
        segs.append((s, e, rep))

    units = []
    for i, (s, e, txt) in enumerate(segs):
        if not txt.strip().lower().startswith("pick up the"):
            continue
        if segment_object(txt) != obj:
            continue
        end = e
        # extend through the next segment if it places the same object
        if i + 1 < len(segs):
            ns, ne, ntxt = segs[i + 1]
            if segment_object(ntxt) == obj and re.match(
                r"\s*(put|place|dump)\b", ntxt, re.I
            ):
                end = ne
        units.append((int(s), int(end)))
    return units


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--object", required=True)
    ap.add_argument("--scene", default="pick_place_diverse")
    ap.add_argument(
        "--zarr-root", default="/storage/project/r-dxu345-0/agao81/pick_place/{hash}"
    )
    ap.add_argument("--master", default=MASTER)
    ap.add_argument(
        "--train-list",
        help="episode-hash file; else 80/20 by episode (every 5th -> valid)",
    )
    ap.add_argument("--valid-list")
    ap.add_argument("--valid-every", type=int, default=5)
    args = ap.parse_args()

    master = json.load(open(args.master))["episodes"]
    pool = sorted(
        h
        for h, v in master.items()
        if v["scene"] == args.scene and args.object in v["objects"]
    )
    if not pool:
        raise SystemExit(
            f"no episodes with object {args.object!r} in scene {args.scene!r}"
        )

    if args.train_list and args.valid_list:
        valid = set(open(args.valid_list).read().split())
        role = {h: ("valid" if h in valid else "train") for h in pool}
    else:
        role = {
            h: ("valid" if i % args.valid_every == 0 else "train")
            for i, h in enumerate(pool)
        }

    episodes, tot = (
        {},
        {"train": [0, 0, 0], "valid": [0, 0, 0]},
    )  # [episodes, units, frames]
    for h in pool:
        units = units_for_object(
            zarr.open(args.zarr_root.format(hash=h), mode="r"), args.object
        )
        nframes = sum(e - s for s, e in units)
        episodes[h] = {
            "role": role[h],
            "n_units": len(units),
            "n_frames": nframes,
            "units": units,
        }
        t = tot[role[h]]
        t[0] += 1
        t[1] += len(units)
        t[2] += nframes

    slug = re.sub(r"\W+", "_", args.object.strip())
    out = os.path.join(LISTS, f"segments_{slug}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "object": args.object,
                "scene": args.scene,
                "split": (
                    {"train_list": args.train_list, "valid_list": args.valid_list}
                    if args.train_list
                    else {"valid_every": args.valid_every}
                ),
                "totals": {
                    k: dict(zip(["episodes", "units", "frames"], v))
                    for k, v in tot.items()
                },
                "episodes": episodes,
            },
            f,
            indent=2,
        )

    print(f"object={args.object!r} scene={args.scene!r}  pool={len(pool)} episodes")
    for k in ("train", "valid"):
        ep, un, fr = tot[k]
        print(
            f"  {k:5s}: {ep:3d} episodes | {un:4d} units | {fr:6d} frames"
            f" | {fr/max(un,1):.0f} frames/unit"
        )
    print(f"wrote {out}")
    z = [h for h, v in episodes.items() if v["n_units"] == 0]
    if z:
        print(
            f"  NOTE: {len(z)} episodes contributed 0 units (object present only as place target)"
        )


if __name__ == "__main__":
    main()
