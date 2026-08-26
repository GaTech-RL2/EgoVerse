#!/usr/bin/env python3
# Aggregate the 40-seed / 1800-step genlvl sweep: per cell, mean of the per-obstacle-level
# emb15 coverages across levels 0-29. Also recompute SR@0.95 and SR@0.8 from per-episode peaks.
import glob, re, os
BASE = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl40"
CELLS = ["hptC8_nogen","hptC8_gen","txC8_nogen","txC8_gen","hnetC8_nogen","hnetC8_gen"]
PEAK_RE = re.compile(r"(?:final_cov=|\bcov=)([0-9]+\.[0-9]+)")

cov, sr95, sr80, lvls = {}, {}, {}, {}
for cell in CELLS:
    perlvl = {}
    allpeaks = []
    for f in glob.glob(f"{BASE}/genlvl2_{cell}_*.out"):
        m = re.search(r"_(\d+)\.out$", f)
        if not m:
            continue
        txt = open(f, errors="ignore").read()
        c = re.search(r"emb15_sim_coverage\D+([0-9]+\.[0-9]+(?:[eE][+-]?[0-9]+)?)", txt)
        if c:
            perlvl[int(m.group(1))] = float(c.group(1))
        allpeaks += [float(x) for x in PEAK_RE.findall(txt)]
    cov[cell] = perlvl
    lvls[cell] = len(perlvl)
    n = len(allpeaks)
    sr95[cell] = (sum(p >= 0.95 for p in allpeaks)/n) if n else None
    sr80[cell] = (sum(p >= 0.80 for p in allpeaks)/n) if n else None
    mean = (sum(perlvl.values())/len(perlvl)) if perlvl else None
    print(f"{cell:16s} mean_cov={('%.3f'%mean) if mean is not None else 'NONE':>6s}  "
          f"levels={len(perlvl)}/30  SR95={('%.3f'%sr95[cell]) if sr95[cell] is not None else 'NA'}  "
          f"SR80={('%.3f'%sr80[cell]) if sr80[cell] is not None else 'NA'}  npeaks={n}")

print("--- deltas (+gen minus -gen), mean coverage ---")
for fam, ng, g in [("HPT-flow c8","hptC8_nogen","hptC8_gen"),
                   ("TransformerAR c8","txC8_nogen","txC8_gen"),
                   ("H-Net C8","hnetC8_nogen","hnetC8_gen")]:
    a, b = cov.get(ng, {}), cov.get(g, {})
    if a and b:
        ma, mb = sum(a.values())/len(a), sum(b.values())/len(b)
        print(f"{fam:18s} -gen={ma:.3f}  +gen={mb:.3f}  delta={mb-ma:+.3f}  (lvls {len(a)}/{len(b)})")
