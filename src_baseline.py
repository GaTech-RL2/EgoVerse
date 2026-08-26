"""Baseline: stored peak coverage of the SAME first-100 source episodes.

Reads the recorded reward (= coverage) directly -- no rendering -- so the
0.5x / 1.5x results can be compared against what the source actually achieved.
"""
import json
from pathlib import Path

import numpy as np
import zarr

SRC = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2")
eps = sorted(p for p in SRC.iterdir() if p.name.endswith(".zarr"))[:100]

peaks = []
for p in eps:
    g = zarr.open_group(str(p), mode="r")
    md = dict(g.attrs)
    T = int(md.get("total_frames", 0)) or None
    r = np.asarray(g["reward"][:T]).squeeze()
    peaks.append(float(r.max()) if r.size else 0.0)

peaks = np.array(peaks)
for thr in (0.80, 0.90, 0.95):
    print("source SR@%.2f = %.3f  (%d/%d)"
          % (thr, (peaks >= thr).mean(), int((peaks >= thr).sum()), len(peaks)))
print("source mean peak = %.4f" % peaks.mean())
print("source min/max   = %.4f / %.4f" % (peaks.min(), peaks.max()))
below = [(eps[i].name, round(float(peaks[i]), 4))
         for i in np.argsort(peaks)[:5]]
print("lowest 5:", below)
Path("/coc/flash7/scratch/paphiwetsa3/respeed/source_baseline100.json").write_text(
    json.dumps({"n": len(peaks), "mean_peak": float(peaks.mean()),
                "sr80": float((peaks >= 0.80).mean()),
                "peaks": [float(x) for x in peaks]}, indent=2))
