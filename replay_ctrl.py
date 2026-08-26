"""Control: replay determinism for the CIRCLE embodiment, two passes in one process."""
import sys, json, os, glob
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")
from pathlib import Path
from Tsimulation.examples.replay_zarr import replay_one

D = sys.argv[1]; N = int(sys.argv[2])
paths = sorted(glob.glob(os.path.join(D, "episode_*.zarr")))[:N]
print(f"dataset={D}  episodes={len(paths)}")
res = {}
for pas in (1, 2):
    out = []
    for p in paths:
        try:
            r = replay_one(Path(p), tol=0.05)
            out.append((os.path.basename(p), round(r["drift_max"], 6), round(r["replay_cov"], 4), bool(r["ok"])))
        except Exception as e:
            out.append((os.path.basename(p), None, None, False))
    res[pas] = out
    nf = sum(1 for x in out if not x[3])
    print(f"  pass{pas}: fail={nf}/{len(out)}")
a, b = res[1], res[2]
ident = sum(1 for x, y in zip(a, b) if x[1] == y[1])
print(f"  identical drift_max across the 2 passes: {ident}/{len(a)} ({ident/len(a):.1%})")
nz = sum(1 for x in a if x[1] and x[1] > 0)
print(f"  episodes with ANY nonzero drift (pass1): {nz}/{len(a)}")
bad = [(x, y) for x, y in zip(a, b) if x[1] != y[1]]
for x, y in bad[:10]:
    print(f"    DIFFERS {x[0]}: p1_drift={x[1]} p2_drift={y[1]}")
