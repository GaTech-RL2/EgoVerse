#!/usr/bin/env python3
"""
Pair human (aria) and robot (eva) pick_place episodes for side-by-side eval.

Produces two tiers of human<->robot pairs and writes them to JSON (+ CSV):

  1. TRUE pairs - the "alignment data set" co-located captures: the same scenes
     recorded by both the robot and a human, grouped into set 1 / set 2. Derived
     from the app.episodes DB (task_description ~ "alignment data set N"). These
     are the only genuinely same-scene human<->robot episodes in the dataset.

  2. SIMILAR-TASK pairs - matched via the dense-language annotations. Each
     episode's manipulated-object set is parsed from its annotation file
     (s3://rldb/processed_annotations/<hash>_annotations.json), normalized with a
     hand-built synonym map (see `canon`), and each aria scene is matched to eva
     demos by object-set CONTAINMENT (is the short robot demo's object set
     covered by the long human play scene?). These are "same objects / similar
     task", NOT the same physical scene -- the bulk human and robot episodes were
     collected independently from a shared ~50-object vocabulary, so they overlap
     partially (best matches share ~4-5 objects) but never exactly.

Why language and not the DB `scene`/`objects` columns: `objects` is literal
"{None}" and `scene` has 2 coarse values, so neither supports fine matching.

Requires R2 + Postgres creds in ~/.egoverse_env (run setup_secret.sh).

Usage:
    python -m egomimic.scripts.pair_episodes_by_language \
        --out-json egomimic/scripts/human_robot_pairs.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client, load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

BUCKET = "rldb"
ANN_PREFIX = "processed_annotations/"
HASH_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6})_annotations\.json")
MIN_SHARED = 3  # min normalized objects in common to count as a similar-task pair
TOPK = 3  # eva matches kept per aria scene

# --- object-name normalization (scene-level granularity) ---------------------
# Built by reading the corpus: annotators describe the same physical object many
# ways across episodes and embodiments (aria says "stuffed toy", eva says
# "stuffed animal"; "white coffee cup" / "white mug" / "white cup"; etc.).
BARE = {  # color-less / ambiguous tokens dropped after normalization
    "cup",
    "plush",
    "bowl",
    "plate",
    "can",
    "chips",
    "juice",
    "marker",
    "pen",
    "screwdriver",
    "canister",
    "coffee",
    "bread",
    "yogurt",
    "tomato",
    "tong",
    "tongs",
    "cannedgoods",
}


def canon(o: str) -> str:
    o = o.lower().strip().rstrip(".")
    o = re.sub(r"\s+right[\s-]?side[\s-]?up$", "", o).strip()
    o = re.sub(r"^(small|big|large|left|right)\s+", "", o)
    o = re.sub(r"\bscrew driver\b", "screwdriver", o)
    o = re.sub(r"\bcarrots\b", "carrot", o)
    o = re.sub(r"\bgreen cabbage\b", "cabbage", o)
    o = re.sub(r"\bstainless\b", "silver", o)
    o = re.sub(r"\b(neon|mint|light|dark)\s+green\b", "green", o)
    if "coffee pod" in o:  # distinct object; do not fold into the cup family
        return "coffee pod"
    o = re.sub(r"\b(canned goods|canned good|can goods|canned)\b", "cannedgoods", o)
    o = re.sub(r"\b(coffee cup|mug)\b", "cup", o)  # vessel family, keyed by color
    o = re.sub(
        r"\b(stuffed animal|stuffed toy|stuff animal|teddy bear|duck toy|stuffed)\b",
        "plush",
        o,
    )
    o = re.sub(r"\b([a-z]+) toy\b", r"\1 plush", o)
    o = re.sub(
        r"\b(bag of chips|pack of chips|potato chips|zapps potato chips)\b", "chips", o
    )
    o = re.sub(
        r"\b(juice pouch|juice pack|juice bag|bag of juice|snack pouch)\b", "juice", o
    )
    o = re.sub(r"\b(pepsi can|coke can)\b", "soda can", o)
    return re.sub(r"\s+", " ", o).strip()


# manipulated-object phrase = first determiner-led noun phrase before a
# preposition / spatial-relation boundary (so "put X behind the Y" yields X).
_OBJ_RE = re.compile(
    r"\b(?:the|a|an)\s+([a-z][a-z \-]*?)\s+"
    r"(?:from|to|onto|on|into|in|with|at|by|near|next|off|using|over|and|that|,"
    r"|behind|inside|beside|above|below|underneath|under|between|around|atop|aside)\b"
)
_STOP = {"entire", "whole", "full"}


def object_set(records: list[dict]) -> set[str]:
    """Normalized set of manipulated objects in an episode. Uses the base (first)
    paraphrase of each annotation span and keeps objects touched >=2x to drop
    one-off mis-parses."""
    spans: dict[tuple, str] = {}
    for r in records:
        spans.setdefault((r["start_idx"], r["end_idx"]), r["text"])
    raw = Counter()
    for text in spans.values():
        m = _OBJ_RE.search(text.lower().rstrip("."))
        if m:
            o = " ".join(w for w in m.group(1).split() if w not in _STOP).strip()
            if o:
                raw[o] += 1
    kept = [o for o, c in raw.items() if c >= 2] or list(raw)
    return {c for c in map(canon, kept) if c and c not in BARE}


# --- R2 annotation fetch -----------------------------------------------------
def list_annotation_hashes(s3) -> list[str]:
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=ANN_PREFIX
    ):
        for obj in page.get("Contents", []):
            m = HASH_RE.search(obj["Key"])
            if m:
                out.append(m.group(1))
    return sorted(set(out))


def download_all(s3, hashes: list[str], cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    def _get(h):
        dst = os.path.join(cache_dir, f"{h}_annotations.json")
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            s3.download_file(BUCKET, f"{ANN_PREFIX}{h}_annotations.json", dst)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_get, hashes))


# --- pairing -----------------------------------------------------------------
def alignment_pairs(df) -> dict:
    """The co-located 'alignment data set N' captures, grouped by set."""
    pp = df[df["task"].astype(str).str.strip() == "pick_place"].copy()
    pp["td"] = pp["task_description"].astype(str).str.strip().str.lower()
    al = pp[pp["td"].str.contains("alignment")]
    out = {}
    for s in ("1", "2"):
        sub = al[al["td"].str.contains(f"set {s}")]
        grp = {}
        for emb in ("eva_bimanual", "aria_bimanual"):
            rows = sub[sub["embodiment"] == emb].sort_values("episode_hash")
            grp[emb] = [
                {"episode_hash": r["episode_hash"], "num_frames": int(r["num_frames"])}
                for _, r in rows.iterrows()
            ]
        out[f"alignment_set_{s}"] = grp
    out["note"] = (
        "Same scenes recorded by robot and human. Set 1 was an interleaved "
        "same-session capture on 2026-04-14 (pair eva<->aria by capture time); "
        "set-2 aria was recollected 2026-04-26. aria-alignment episodes have no "
        "language annotations."
    )
    return out


def language_pairs(aria_c: dict, eva_c: dict) -> tuple[list, Counter]:
    rows, best_overlap = [], Counter()
    for ah, (adesc, aset) in sorted(aria_c.items()):
        cands = []
        for eh, (edesc, eset) in eva_c.items():
            inter = aset & eset
            if len(inter) < MIN_SHARED:
                continue
            cands.append(
                (
                    len(inter),
                    len(inter) / len(eset),
                    len(inter) / len(aset | eset),
                    eh,
                    edesc,
                    inter,
                    eset,
                )
            )
        cands.sort(key=lambda x: (x[0], round(x[1], 3), x[2]), reverse=True)
        best_overlap[cands[0][0] if cands else 0] += 1
        rows.append(
            {
                "aria_hash": ah,
                "aria_desc": adesc,
                "aria_scene": sorted(aset),
                "matches": [
                    {
                        "eva_hash": eh,
                        "eva_desc": edesc,
                        "n_shared": ni,
                        "containment": round(contain, 3),
                        "jaccard": round(jac, 3),
                        "shared_objects": sorted(inter),
                        "eva_demo": sorted(eset),
                    }
                    for ni, contain, jac, eh, edesc, inter, eset in cands[:TOPK]
                ],
            }
        )
    return rows, best_overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-dir", default=os.path.expanduser("~/.cache/egoverse_annotations")
    )
    ap.add_argument("--out-json", default="human_robot_pairs.json")
    ap.add_argument("--out-csv", default="human_robot_pairs.csv")
    args = ap.parse_args()

    load_env()
    s3 = get_boto3_s3_client()
    hashes = list_annotation_hashes(s3)
    print(f"{len(hashes)} annotation files; syncing to {args.cache_dir} ...")
    download_all(s3, hashes, args.cache_dir)

    df = episode_table_to_df(create_default_engine())
    df = df[~df["is_deleted"].fillna(False).astype(bool)]
    meta = {
        r["episode_hash"]: (
            str(r["embodiment"]).strip(),
            str(r["task_description"]).strip(),
            str(r.get("task", "")).strip(),
        )
        for _, r in df.iterrows()
    }

    eva_c, aria_c = {}, {}
    for h in hashes:
        if h not in meta or meta[h][2] != "pick_place":
            continue
        emb, desc, _ = meta[h]
        objs = object_set(
            json.load(open(os.path.join(args.cache_dir, f"{h}_annotations.json")))
        )
        if emb == "eva_bimanual" and len(objs) >= 3:
            eva_c[h] = (desc, objs)
        elif emb == "aria_bimanual":
            aria_c[h] = (desc, objs)

    lang_rows, best_overlap = language_pairs(aria_c, eva_c)
    align = alignment_pairs(df)

    result = {
        "task": "pick_place",
        "method": "human(aria)<->robot(eva) pairing for side-by-side eval",
        "tiers": {
            "true_pairs_alignment": align,
            "similar_task_language_pairs": lang_rows,
        },
        "summary": {
            "eva_episodes_matched_on": len(eva_c),
            "aria_episodes": len(aria_c),
            "best_match_object_overlap_histogram": dict(
                sorted(best_overlap.items(), reverse=True)
            ),
            "min_shared_objects": MIN_SHARED,
            "topk_per_aria": TOPK,
        },
        "notes": (
            "Object-set CONTAINMENT matching on normalized dense-language objects. "
            "Similar-task pairs share objects, NOT the same physical scene. "
            "For true same-scene pairs use tiers.true_pairs_alignment."
        ),
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "aria_hash",
                "aria_desc",
                "rank",
                "eva_hash",
                "eva_desc",
                "n_shared",
                "containment",
                "jaccard",
                "shared_objects",
            ]
        )
        for row in lang_rows:
            for rank, m in enumerate(row["matches"], 1):
                w.writerow(
                    [
                        row["aria_hash"],
                        row["aria_desc"],
                        rank,
                        m["eva_hash"],
                        m["eva_desc"],
                        m["n_shared"],
                        m["containment"],
                        m["jaccard"],
                        " / ".join(m["shared_objects"]),
                    ]
                )

    print(f"wrote {args.out_json} and {args.out_csv}")
    print(
        f"  aria scenes: {len(aria_c)}; best-overlap histogram: "
        f"{dict(sorted(best_overlap.items(), reverse=True))}"
    )


if __name__ == "__main__":
    main()
