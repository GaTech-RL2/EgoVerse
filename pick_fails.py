"""Worst 0.25x episodes, from the shard result JSONs."""
import glob
import json
import re

rows = []
for f in sorted(glob.glob("/coc/flash7/scratch/paphiwetsa3/respeed/full025/shard*.json")):
    for e in json.load(open(f))["episodes"]:
        m = re.search(r"_obs(\d+)_", e["src"])
        rows.append((e["peak_coverage"], int(m.group(1)) if m else -1,
                     e["src"], e["src_frames"], e["steps"]))
rows.sort()

print("total %d episodes\n" % len(rows))
print("=== 5 worst OBSTACLE ===")
for c, l, n, sf, st in [r for r in rows if r[1] > 0][:5]:
    print("  %-44s obs%-3d peak=%.3f  %d->%d" % (n, l, c, sf, st))
print("=== 5 worst IN-DOMAIN ===")
for c, l, n, sf, st in [r for r in rows if r[1] == 0][:5]:
    print("  %-44s obs%-3d peak=%.3f  %d->%d" % (n, l, c, sf, st))
