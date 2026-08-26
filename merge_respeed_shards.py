"""Merge the sharded 0.25x conversion into one dataset directory.

Shards each index their episodes from 0, so names collide across shards. This
builds the final dataset as a SYMLINK VIEW with globally unique names,
renumbering per (object, pusher, obstacle_level) family. Real data stays in the
shard directories -- consistent with the project rule that a merged folder is a
view, never where data lives.

  python merge_respeed_shards.py <shards_dir> <dest_dir> [--dry-run]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NAME = re.compile(r"^episode_(?P<obj>.+?)_(?P<push>.+?)_obs(?P<lvl>\d+)_"
                  r"(?P<tag>[A-Za-z0-9]+)_(?P<idx>\d+)\.zarr$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    shard_dirs = sorted(d for d in a.shards.iterdir() if d.is_dir())
    eps = []
    for sd in shard_dirs:
        for p in sorted(sd.iterdir()):
            if p.name.endswith(".zarr") and p.is_dir():
                eps.append(p)
    print("shards: %d   episodes found: %d" % (len(shard_dirs), len(eps)))
    if not eps:
        print("nothing to merge")
        return 1

    counters, by_lvl, bad = defaultdict(int), Counter(), 0
    plan = []
    for p in eps:
        m = NAME.match(p.name)
        if not m:
            bad += 1
            continue
        key = (m["obj"], m["push"], m["lvl"], m["tag"])
        idx = counters[key]
        counters[key] += 1
        by_lvl[int(m["lvl"])] += 1
        plan.append((p, "episode_%s_%s_obs%s_%s_%06d.zarr"
                     % (m["obj"], m["push"], m["lvl"], m["tag"], idx)))
    if bad:
        print("WARNING: %d episodes did not match the expected name pattern" % bad)

    print("levels: %d distinct, obs0=%d, obstacle=%d"
          % (len(by_lvl), by_lvl.get(0, 0), sum(v for k, v in by_lvl.items() if k)))

    if a.dry_run:
        for src, name in plan[:5]:
            print("  %s -> %s" % (src.name, name))
        print("  ... %d total" % len(plan))
        return 0

    a.dest.mkdir(parents=True, exist_ok=True)
    made = 0
    for src, name in plan:
        link = a.dest / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(src.resolve())
        made += 1
    print("created %d symlinks in %s" % (made, a.dest))

    dangling = [p.name for p in a.dest.iterdir() if not p.resolve().exists()]
    print("dangling symlinks: %d" % len(dangling))
    (a.dest / "_MERGE_MANIFEST.json").write_text(json.dumps(
        {"shards_dir": str(a.shards), "n_shards": len(shard_dirs),
         "episodes": len(plan), "by_level": {str(k): v for k, v in sorted(by_lvl.items())},
         "note": "symlink view; real data lives in the shard dirs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
