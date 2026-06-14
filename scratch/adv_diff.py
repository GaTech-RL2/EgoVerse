"""Diff my independent re-dump against the phase-2 baseline (and report file-set deltas)."""
import json
import os
import sys

REPO = "/coc/flash7/paphiwetsa3/projects/EgoVerse2"
MINE = os.path.join(REPO, "scratch/adv_verify/resolved")
BASE = os.path.join(REPO, "scratch/config_phase2_baseline/resolved")


def walk_json(root):
    out = {}
    for r, _d, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            rel = os.path.relpath(os.path.join(r, fn), root)
            out[rel] = os.path.join(r, fn)
    return out


def load(p):
    with open(p) as f:
        return f.read()


def main():
    mine = walk_json(MINE)
    base = walk_json(BASE)
    # baseline has methods.json artifact; ignore it for set comparison
    base.pop("methods.json", None)
    mine.pop("methods.json", None)

    mine_only = sorted(set(mine) - set(base))
    base_only = sorted(set(base) - set(mine))
    common = sorted(set(mine) & set(base))

    empty, nonempty = [], []
    for rel in common:
        a = load(mine[rel])
        b = load(base[rel])
        if a == b:
            empty.append(rel)
        else:
            nonempty.append(rel)

    print(f"COMMON={len(common)}  EMPTY_DIFF={len(empty)}  NONEMPTY_DIFF={len(nonempty)}")
    print(f"MINE_ONLY({len(mine_only)}): " + ", ".join(mine_only))
    print(f"BASE_ONLY({len(base_only)}): " + ", ".join(base_only))
    if nonempty:
        print("=== NON-EMPTY DIFFS ===")
        for rel in nonempty:
            print("  DIFF " + rel)
    else:
        print("ALL COMMON FILES IDENTITY-CLEAN")


if __name__ == "__main__":
    main()
