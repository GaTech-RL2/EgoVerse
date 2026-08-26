"""Composition of circle_4500_plus_gen_v2 by obstacle level, from episode_init."""
import json
import re
from collections import Counter
from pathlib import Path

import zarr

SRC = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2")
eps = sorted(p for p in SRC.iterdir() if p.name.endswith(".zarr"))
print("total episodes:", len(eps))

# fast path: the level is in the filename (obs<N>)
byname = Counter()
for p in eps:
    m = re.search(r"_obs(\d+)_", p.name)
    byname[int(m.group(1)) if m else -1] += 1
print("\nby filename obs tag:")
for k in sorted(byname):
    print("   obs%-3s %5d" % (k if k >= 0 else "?", byname[k]))

# verify the tag against the actual env_args on a sample of non-zero levels
nz = [p for p in eps if not re.search(r"_obs0_", p.name)]
print("\nnon-obs0 episodes: %d" % len(nz))
if nz:
    print("first few:", [p.name for p in nz[:4]])
    chk = Counter()
    for p in nz[:: max(1, len(nz) // 40)][:40]:
        g = zarr.open_group(str(p), mode="r")
        lvl = json.loads(dict(g.attrs)["task_description"])["env_args"].get(
            "obstacle_level", 0)
        tag = int(re.search(r"_obs(\d+)_", p.name).group(1))
        chk["match" if lvl == tag else "MISMATCH"] += 1
    print("filename-tag vs env_args on 40 sampled:", dict(chk))
