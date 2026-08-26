#!/bin/bash
# Wait for the 0.25x array to drain, then merge the shards into a symlink view
# and report coverage overall / in-domain / obstacle / per level band.
JOB=3679116
S=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/_shards_pusher0.25x
DST=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle4500gen_v2_pusher0.25x
R=/coc/flash7/paphiwetsa3/projects/_wt_stack
PY=/coc/flash7/paphiwetsa3/projects/EgoVerse7/.venv/bin/python
LOG=/coc/flash7/scratch/paphiwetsa3/respeed/full025

while squeue -h -j "$JOB" -t R,PD -o "%i" 2>/dev/null | grep -q .; do sleep 60; done

echo "=== array final states ==="
sacct -j "$JOB" --format=JobID,State,ExitCode -n 2>/dev/null \
  | grep -vE "\.(batch|extern|[0-9]+) " | awk '{print $2}' | sort | uniq -c

echo "=== episodes produced ==="
find "$S" -maxdepth 2 -name '*.zarr' -type d 2>/dev/null | wc -l

echo "=== merge ==="
"$PY" "$R/merge_respeed_shards.py" "$S" "$DST"

echo "=== coverage ==="
"$PY" - <<PY
import glob, json, re
from collections import defaultdict
peaks, lvl = [], []
for f in sorted(glob.glob("$LOG/shard*.json")):
    d = json.load(open(f))
    for e in d["episodes"]:
        peaks.append(e["peak_coverage"])
        m = re.search(r"_obs(\d+)_", e["src"])
        lvl.append(int(m.group(1)) if m else -1)
def rep(name, vals):
    if not vals: return
    n=len(vals)
    print("  %-22s n=%-5d mean %.4f   >=.80 %4d   >=.90 %4d   >=.95 %4d"
          % (name, n, sum(vals)/n,
             sum(1 for x in vals if x>=.80),
             sum(1 for x in vals if x>=.90),
             sum(1 for x in vals if x>=.95)))
rep("ALL", peaks)
rep("in-domain (obs0)", [p for p,l in zip(peaks,lvl) if l==0])
rep("obstacle (1-30)", [p for p,l in zip(peaks,lvl) if l>0])
band=defaultdict(list)
for p,l in zip(peaks,lvl):
    if l>0: band[(l-1)//6*6+1].append(p)
for b in sorted(band):
    rep("  levels %2d-%2d"%(b,b+5), band[b])
PY
