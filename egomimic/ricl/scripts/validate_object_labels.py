"""Validate the object labels we extract from episode annotations.

We pull the manipulated-object label out of every "Pick up the <obj> ..." segment
in each episode's zarr ``annotations`` array, canonicalize it, and then run a set
of heuristics that flag labels worth a human eyeballing:

  * ``parse_failure`` -- the label still contains a relational phrase ("on top of",
    "in front of", ...), meaning the pick-target regex over-captured. These are
    extraction bugs, not real objects. (ERROR -> non-zero exit.)
  * ``typo``          -- a rare label one or two edits from a common one
    (``doritosl`` -> ``doritos``, ``screw driver`` -> ``screwdriver``).
  * ``truncation``    -- a rare label that is a prefix of a common one
    (``blue stuffed`` -> ``blue stuffed toy``).
  * ``ambiguous_bare``-- a single-word label that is the head of several colored
    variants (``bowl`` while ``green/beige/silver bowl`` also exist).
  * ``singleton``     -- appears in exactly one episode; not wrong, just review-worthy.

Each finding prints example (episode, raw annotation) evidence so you can confirm
and, if needed, extend ``canon()`` below.

Usage:
    python -m egomimic.ricl.scripts.validate_object_labels \
        --zarr-root '/storage/project/r-dxu345-0/agao81/pick_place/{hash}' \
        [--episodes-file egomimic/ricl/episode_lists/d0_eva_pool_all.txt] \
        [--min-common 5] [--strict]

With no ``--episodes-file`` it validates every episode key in
``eva_episode_objects.json`` (in the parent ricl/episode_lists/).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import zarr

PICK = re.compile(
    r"[Pp]ick up the (.+?)"
    r"(?: from| with| using| at | by | to | on | in | behind| between| next to| inside|$)"
)
# annotator typos / truncations folded to their canonical object
TYPO = {
    "doritosl": "doritos",
    "screw driver": "screwdriver",
    "pink bow": "pink bowl",
    "blue stuffed": "blue stuffed toy",
    "pink stuffed": "pink stuffed toy",
    "brown stuff toy": "brown stuffed toy",
    "pink stuff animal": "pink stuffed toy",
}
# relational phrases that should never appear inside a clean object label
RELATIONAL = re.compile(
    r"\b(on top of|in front of|behind|next to|between|inside|onto|above|below|"
    r"underneath|on the|to the|on its|in the|of the)\b"
)
# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MASTER = os.path.join(HERE, "episode_lists", "eva_episode_objects.json")


def canon(o: str) -> str:
    """Mirror the canonicalizer used to build eva_episode_objects.json."""
    o = o.strip().lower().rstrip(".")
    o = re.sub(r"\bstuffed animal\b", "stuffed toy", o)
    o = re.sub(r"\b(can good|canned good)\b", "canned goods", o)
    o = re.sub(r"\bcarrots\b", "carrot", o)
    o = re.sub(r"^(left|right) ", "", o)
    o = re.sub(r"\s+", " ", o).strip()
    return TYPO.get(o, o)


def canonical_segments(store) -> list[str]:
    """One representative text per (start, end) segment, preferring a
    paraphrase that starts with 'Pick up the' so picks are detectable."""
    by = defaultdict(list)
    for a in store["annotations"][:]:
        d = json.loads(a.decode() if isinstance(a, (bytes, bytearray)) else a)
        by[(d["start_idx"], d["end_idx"])].append(d["text"])
    out = []
    for k in sorted(by):
        txts = by[k]
        out.append(
            next(
                (t for t in txts if t.lstrip().lower().startswith("pick up the")),
                txts[0],
            )
        )
    return out


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract(hashes, zarr_root):
    """label -> {count, episodes:set, examples:[(hash, raw, text)]}"""
    info = defaultdict(lambda: {"count": 0, "episodes": set(), "examples": []})
    missing = []
    for h in hashes:
        path = zarr_root.format(hash=h)
        if not os.path.isdir(path):
            missing.append(h)
            continue
        s = zarr.open(path, mode="r")
        if "annotations" not in s or s["annotations"].shape[0] == 0:
            continue
        for seg in canonical_segments(s):
            m = PICK.match(seg.strip())
            if not m:
                continue
            raw = m.group(1).strip()
            label = canon(raw)
            rec = info[label]
            if h not in rec["episodes"]:
                rec["episodes"].add(h)
                rec["count"] += 1
            if len(rec["examples"]) < 3:
                rec["examples"].append((h, raw, seg.strip()))
    return info, missing


def validate(info, min_common):
    common = {lab for lab, r in info.items() if r["count"] >= min_common}
    common_heads = Counter()
    for lab in info:
        toks = lab.split()
        if len(toks) > 1:
            common_heads[toks[-1]] += 1

    findings = defaultdict(list)  # category -> [(label, detail)]
    for lab, r in sorted(info.items(), key=lambda kv: kv[1]["count"]):
        n = r["count"]
        if RELATIONAL.search(lab):
            findings["parse_failure"].append((lab, n, "contains a relational phrase"))
            continue
        if n >= min_common:
            continue
        # rare labels: look for a near / containing common label
        sugg = None
        for c in common:
            if lab != c and (
                levenshtein(lab, c) <= 2 or lab == c[: len(lab)] or c == lab[: len(c)]
            ):
                sugg = c
                break
        if sugg and (lab == sugg[: len(lab)] or sugg == lab[: len(sugg)]):
            findings["truncation"].append((lab, n, f"prefix of common '{sugg}'"))
        elif sugg:
            findings["typo"].append(
                (lab, n, f"~ '{sugg}' (edit dist {levenshtein(lab, sugg)})")
            )
        elif len(lab.split()) == 1 and common_heads.get(lab, 0) >= 3:
            findings["ambiguous_bare"].append(
                (lab, n, f"bare head of {common_heads[lab]} multiword labels")
            )
        elif n == 1:
            findings["singleton"].append((lab, n, "appears in 1 episode"))
    return findings


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--zarr-root",
        default="/storage/project/r-dxu345-0/agao81/pick_place/{hash}",
        help="path template with {hash}",
    )
    ap.add_argument(
        "--episodes-file",
        help="newline-delimited hashes; default = keys of the master json",
    )
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument(
        "--min-common",
        type=int,
        default=5,
        help="count at/above which a label is trusted as 'common'",
    )
    ap.add_argument(
        "--strict", action="store_true", help="exit 1 if any parse_failure is found"
    )
    args = ap.parse_args()

    if args.episodes_file:
        hashes = open(args.episodes_file).read().split()
    else:
        hashes = list(json.load(open(args.master))["episodes"].keys())

    info, missing = extract(hashes, args.zarr_root)
    print(
        f"episodes: {len(hashes)} ({len(missing)} missing on disk) | distinct labels: {len(info)}"
    )

    findings = validate(info, args.min_common)
    order = ["parse_failure", "typo", "truncation", "ambiguous_bare", "singleton"]
    total = sum(len(findings[c]) for c in order)
    if not total:
        print("no issues flagged.")
    for cat in order:
        rows = findings.get(cat, [])
        if not rows:
            continue
        print(f"\n## {cat}  ({len(rows)})")
        for lab, n, detail in rows:
            ex_h, ex_raw, ex_txt = info[lab]["examples"][0]
            print(f"  [{n:>2}x] {lab!r}: {detail}")
            print(f"         e.g. {ex_h}: raw={ex_raw!r}")
            print(f'              "{ex_txt[:110]}"')

    print(
        f"\n{total} labels flagged for review "
        f"({len(findings.get('parse_failure', []))} parse_failure)."
    )
    if args.strict and findings.get("parse_failure"):
        sys.exit(1)


if __name__ == "__main__":
    main()
