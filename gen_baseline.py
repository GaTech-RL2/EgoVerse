"""Source baseline for the SAME 100 obstacle-gen episodes the converter used."""
import json
import re
from pathlib import Path

import numpy as np
import zarr

SRC = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2")
rx = re.compile(r"_obs([1-9]|[12][0-9]|30)_")
eps = [p for p in sorted(SRC.iterdir()) if p.name.endswith(".zarr") and rx.search(p.name)]
idx = np.linspace(0, len(eps) - 1, 100).round().astype(int)
sel = [eps[i] for i in sorted(set(idx.tolist()))]
print("obstacle episodes total %d, sampled %d" % (len(eps), len(sel)))

peaks, lvls = [], []
for p in sel:
    g = zarr.open_group(str(p), mode="r")
    T = int(dict(g.attrs).get("total_frames", 0)) or None
    r = np.asarray(g["reward"][:T]).squeeze()
    peaks.append(float(r.max()) if r.size else 0.0)
    lvls.append(int(re.search(r"_obs(\d+)_", p.name).group(1)))

peaks = np.array(peaks)
print("levels covered: %d..%d  (%d distinct)"
      % (min(lvls), max(lvls), len(set(lvls))))
for t in (0.80, 0.90, 0.95):
    print("source-gen SR@%.2f = %.3f (%d/%d)"
          % (t, (peaks >= t).mean(), int((peaks >= t).sum()), len(peaks)))
print("source-gen mean peak = %.4f   min %.4f" % (peaks.mean(), peaks.min()))
Path("/coc/flash7/scratch/paphiwetsa3/respeed/source_gen100.json").write_text(
    json.dumps({"peaks": [float(x) for x in peaks], "levels": lvls,
                "names": [p.name for p in sel]}, indent=2))
