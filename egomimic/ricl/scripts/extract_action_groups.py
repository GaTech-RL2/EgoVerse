"""Group annotation *segments* by the exact action, with a configurable key.

Each segment of the zarr ``annotations`` is parsed into structured fields; groups
are formed by a chosen subset of them (``--key``). Fields:

  verb       pick / place / dump / return
  object     canonical manipulated object
  hand       left / right / both / ?
  grip       (pick) grasp style: plain / side / lip / handle
  grasp_loc  (pick) where on the object: middle / entire / top / bottom / left / right / handle
  angle      (pick) approach: diagonal / top / side / straight
  source     (pick) picked off: table / <object>
  relation   (place/dump) raw placement direction: front / behind / left / right /
             inside / on_top / between / center / table / corner / next_to / other
  ref_object (place/dump) the object used as a location reference, or None
  ref_direction (place/dump) direction relative to the reference object
             (= relation), or "none" when placed with no object reference
  orientation (place/dump) object heading from "facing X": forwards / backwards /
             left / right / top_left / top_right / bottom_left / bottom_right / none
  flip       (place/dump) up (right side up) / down (upside down) / none
  sub_position (place/dump) fine table region: top / bottom / top_left / ... / none

Fields that don't apply to a verb are "-". Member = (episode, start, end, ...) so
this plugs into the per-(hash, frame_idx) retrieval keying. The LLM half reads the
emitted groups to confirm each is one action and catch parser mistakes.

    python -m egomimic.ricl.scripts.extract_action_groups --object 'blue marker' \
        [--key verb,object,grip,grasp_loc,angle,hand,relation,has_ref] [--min-size 2]

Writes egomimic/ricl/episode_lists/actions_<object_slug>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import zarr

from egomimic.ricl.scripts.validate_object_labels import canon

# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "episode_lists")
MASTER = os.path.join(LISTS, "eva_episode_objects.json")
DEFAULT_KEY = "verb,object,grip,grasp_loc,angle,hand,ref_direction"
ALL_FIELDS = [
    "verb",
    "object",
    "hand",
    "grip",
    "grasp_loc",
    "angle",
    "source",
    "relation",
    "ref_object",
    "ref_direction",
    "orientation",
    "flip",
    "sub_position",
]

_OBJ = (
    r"(.+?)(?: from| with| using| at | by | to | on | in | into| onto| behind| between| next to"
    r"| inside| right side up| upside down| below| and |$)"
)
PICK = re.compile(r"[Pp]ick up the " + _OBJ)
PLACE = re.compile(r"(?:[Pp]ut|[Pp]lace|[Dd]ump) the " + _OBJ)

# place relation -> direction (first match wins; specific phrases before bare left/right;
# 'right side(?! up)' so the "right side up" flip phrase isn't read as a right placement)
PLACE_DIR = [
    (r"\b(inside|into)\b", "inside"),
    (r"\bon top of\b", "on_top"),
    (r"\b(in front|to the front of)\b", "front"),
    (r"\b(behind|at the back of)\b", "behind"),
    (r"\bin the middle of .+ and \b", "between"),
    (r"\b(in between|between)\b", "between"),
    (r"\bnext to\b", "next_to"),
    (r"\bto the (top|bottom|front|rear|back) (left|right)\b", "corner"),
    (r"\bon the (top|bottom) (left|right)\b", "corner"),
    (r"\bside of the table\b", "table"),  # before the top/bottom-side rules below
    (r"\b(to|on) the top of\b", "above"),
    (r"\btop side of\b", "on_top"),
    (r"\b(to|on) the bottom of\b", "below"),
    (r"\bbottom side of\b", "below"),
    (r"\bbelow\b", "below"),
    (r"\b(to the left|left side|on the left)\b", "left"),
    (r"\b(to the right|right side(?! up)|on the right)\b", "right"),
    (r"\b(center|middle of the table)\b", "center"),
    (r"\bin the .+?(bowl|cup|mug)\b", "inside"),
    (r"\bon the .+?(plate|tray)\b", "on_top"),
    (r"\b(on the table|side of the table)\b", "table"),
]
# capture the reference object after each relation (shared stop suffix; 'on its'/'upside'
# stop trailing pose phrases from being swallowed into the ref)
_STOP = r"(?: with| using| facing| on the| on its| upside| right side| relative| and | to the |$)"
REF_PATS = [
    r"\b(?:inside|into) the (.+?)" + _STOP,
    r"\bin the (.+?(?:bowl|cup|mug))" + _STOP,
    r"\bon top of the (.+?)" + _STOP,
    r"\b(?:below|above|(?:to|on) the (?:top|bottom)(?: side)? of) the (.+?)"
    + _STOP,  # vertical/side, before 'on the X plate'
    r"\bon the (?:(?:top|bottom|left|right)(?: (?:left|right))? side of the )?(.+?(?:plate|tray))"
    + _STOP,
    r"\b(?:in front of|to the front of) the (.+?)" + _STOP,
    r"\b(?:behind|at the back of) the (?:(?:left|right|top|bottom) side of the )?(.+?)"
    + _STOP,
    r"\b(?:between|in the middle of) the (.+? and (?:the )?.+?)" + _STOP,
    r"\b(?:to|on|onto) the (?:top |bottom |front |rear |back )?(?:left|right)(?: side)? of the (.+?)"
    + _STOP,
    r"\bon the (?:top|bottom) (?:left|right) of the (.+?)" + _STOP,
    r"\bnext to the (.+?)" + _STOP,
]


def _hand(low):
    return (
        "both"
        if "both" in low
        else "left"
        if re.search(r"\bleft (hand|arm)\b", low)
        else "right"
        if re.search(r"\bright (hand|arm)\b", low)
        else "?"
    )


# object heading at placement (anchor varies: tip / handle / front / object)
_FACING = re.compile(
    r"facing (?:to the |the )?(top left|top right|bottom left|bottom right|"
    r"forwards?|backwards?|back|front|left|right)"
)
_SUBPOS = re.compile(r"on (?:its |the )?(top|bottom)(?: (left|right))? side")


def _place_extras(low):
    m = _FACING.search(low)
    if m:
        v = {
            "forward": "forwards",
            "backward": "backwards",
            "back": "backwards",
            "front": "forwards",
        }.get(m.group(1), m.group(1))
        orientation = v.replace(" ", "_")
    else:
        orientation = "none"
    flip = (
        "down" if "upside down" in low else "up" if "right side up" in low else "none"
    )
    sp = _SUBPOS.search(low)
    sub_position = (
        (sp.group(1) + ("_" + sp.group(2) if sp.group(2) else "")) if sp else "none"
    )
    return orientation, flip, sub_position


def _pick_fields(low):
    grip = (
        "handle"
        if re.search(r"\bhandle\b", low)
        else "lip"
        if re.search(r"\blip (grip|grasp)\b", low)
        else "side"
        if re.search(r"\bside (grip|grasp)\b", low)
        else "plain"
    )
    # check 'middle' before the top/bottom fallbacks so "from the top at the middle"
    # (approach=top, grasp=middle) isn't mis-read as a top grasp
    if re.search(r"\bentire object\b", low):
        loc = "entire"
    elif re.search(r"\b(the middle|grasping its middle|gripping the middle)\b", low):
        loc = "middle"
    elif re.search(r"\b(right side|on the right|(?:at|from|by) the right)\b", low):
        loc = "right"
    elif re.search(r"\b(left side|on the left|(?:at|from|by) the left)\b", low):
        loc = "left"
    elif re.search(
        r"\btop of the object\b|\bat the top\b|\b(?:from|on|by) the top\b(?! of)"
        r"|\bgrasping the top\b|\btop(?: left| right)? lip (?:grasp|grip)\b",
        low,
    ):
        loc = "top"  # '(?! of)' so a "from the top OF the plate" SOURCE doesn't set grasp_loc
    elif re.search(r"\b(at the bottom|(?:from|on|by) the bottom)\b", low):
        loc = "bottom"
    elif grip == "handle":
        loc = "handle"
    else:
        loc = "none"  # null token: no grasp location stated (was "other")
    angle = (
        "diagonal"
        if "diagonal" in low
        else "top"
        if re.search(
            r"\bfrom the top\b(?! of)|\btop pick\b|\bfrom the top of the object\b", low
        )
        else "side"
        if re.search(r"\bfrom the side\b", low)
        else "straight"
    )
    m = re.search(
        r"\bfrom the (?:top of the )?(table|[a-z ]*?(?:tray|plate|bowl|cup|mug))\b", low
    )
    source = "table" if not m or m.group(1) == "table" else canon(m.group(1))
    return grip, loc, angle, source


def _place_fields(low):
    relation = next((d for pat, d in PLACE_DIR if re.search(pat, low)), "other")
    ref = None
    for pat in REF_PATS:
        m = re.search(pat, low)
        if m:
            ref = canon(m.group(1))
            break
    has_ref = bool(ref) and "table" not in (ref or "")
    ref_direction = relation if has_ref else "none"
    return relation, ref, ref_direction


def parse_segment(text: str) -> dict | None:
    t = text.strip()
    low = t.lower()
    if low.startswith("return"):
        return {
            **{f: "-" for f in ALL_FIELDS},
            "verb": "return",
            "hand": _hand(low),
            "text": t,
        }
    if low.startswith("pick up the"):
        verb, rx = "pick", PICK
    elif low.startswith("dump the"):
        verb, rx = "dump", PLACE
    elif low.startswith(("put the", "place the")):
        verb, rx = "place", PLACE
    else:
        return None
    m = rx.match(t)
    if not m:
        return None
    rec = {f: "-" for f in ALL_FIELDS}
    rec.update(verb=verb, object=canon(m.group(1)), hand=_hand(low), text=t)
    if verb == "pick":
        rec["grip"], rec["grasp_loc"], rec["angle"], rec["source"] = _pick_fields(low)
    else:
        rec["relation"], rec["ref_object"], rec["ref_direction"] = _place_fields(low)
        rec["orientation"], rec["flip"], rec["sub_position"] = _place_extras(low)
    return rec


def episode_segments(store):
    by = defaultdict(list)
    for a in store["annotations"][:]:
        d = json.loads(a.decode() if isinstance(a, (bytes, bytearray)) else a)
        by[(d["start_idx"], d["end_idx"])].append(d["text"])
    return [
        (
            int(s),
            int(e),
            next(
                (t for t in by[(s, e)] if t.lstrip().lower().startswith("pick up the")),
                by[(s, e)][0],
            ),
        )
        for (s, e) in sorted(by)
    ]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--object")
    ap.add_argument("--scene", default="pick_place_diverse")
    ap.add_argument(
        "--zarr-root", default="/storage/project/r-dxu345-0/agao81/pick_place/{hash}"
    )
    ap.add_argument("--master", default=MASTER)
    ap.add_argument(
        "--key", default=DEFAULT_KEY, help=f"comma list from: {','.join(ALL_FIELDS)}"
    )
    ap.add_argument("--min-size", type=int, default=2)
    args = ap.parse_args()

    key_fields = [k.strip() for k in args.key.split(",")]
    bad = [k for k in key_fields if k not in ALL_FIELDS]
    if bad:
        raise SystemExit(f"unknown key field(s) {bad}; choose from {ALL_FIELDS}")

    master = json.load(open(args.master))["episodes"]
    pool = sorted(
        h
        for h, v in master.items()
        if v["scene"] == args.scene
        and (args.object is None or args.object in v["objects"])
    )

    groups = defaultdict(list)
    for h in pool:
        for s, e, txt in episode_segments(
            zarr.open(args.zarr_root.format(hash=h), mode="r")
        ):
            rec = parse_segment(txt)
            if rec is None or (args.object and rec["object"] != args.object):
                continue
            sig = "|".join(str(rec[k]) for k in key_fields)
            groups[sig].append({"episode": h, "start": s, "end": e, **rec})

    kept = dict(
        sorted(
            ((s, m) for s, m in groups.items() if len(m) >= args.min_size),
            key=lambda kv: -len(kv[1]),
        )
    )
    slug = re.sub(r"\W+", "_", args.object.strip()) if args.object else "all"
    out = os.path.join(LISTS, f"actions_{slug}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "object": args.object,
                "scene": args.scene,
                "key": key_fields,
                "min_size": args.min_size,
                "groups": {
                    s: {
                        "n": len(m),
                        "n_episodes": len({x["episode"] for x in m}),
                        "members": m,
                    }
                    for s, m in kept.items()
                },
            },
            f,
            indent=2,
        )

    print(
        f"object={args.object!r} scene={args.scene!r}  pool={len(pool)} eps  key={key_fields}"
    )
    print(
        f"{len(kept)} groups (>= {args.min_size}); "
        f"{sum(1 for m in groups.values() if len(m) < args.min_size)} smaller groups dropped\n"
    )
    print(f"  {'signature':62s} {'n':>3} {'eps':>3}")
    for s, m in kept.items():
        print(f"  {s:62s} {len(m):>3} {len({x['episode'] for x in m}):>3}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
