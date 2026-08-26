#!/usr/bin/env python3
# Aggregate genlvl2 (1-min) eval: per cell, mean of the 30 per-obstacle-level emb15 coverages.
import glob, re
BASE = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl"
CELLS = ["hptC8_nogen","hptC8_gen","txC8_nogen","txC8_gen",
         "hnetC8_nogen","hnetC8_gen","hnetC4_nogen","hnetC4_gen"]
cov = {}
for cell in CELLS:
    perlvl = {}
    for f in glob.glob(f"{BASE}/genlvl2_{cell}_*.out"):
        m = re.search(r"_(\d+)\.out$", f)
        if not m:
            continue
        c = re.search(r"emb15_sim_coverage\D+([0-9]+\.[0-9]+)", open(f, errors="ignore").read())
        if c:
            perlvl[int(m.group(1))] = float(c.group(1))
    cov[cell] = perlvl
    mean = (sum(perlvl.values())/len(perlvl)) if perlvl else None
    print(f"{cell}: {('mean=%.3f'%mean) if mean is not None else 'NONE'}  levels={len(perlvl)}/30")
print("--- deltas (+gen minus -gen) ---")
for fam, ng, g in [("HPT-flow c8","hptC8_nogen","hptC8_gen"),
                   ("TransformerAR c8","txC8_nogen","txC8_gen"),
                   ("H-Net C8","hnetC8_nogen","hnetC8_gen"),
                   ("H-Net C4","hnetC4_nogen","hnetC4_gen")]:
    a, b = cov.get(ng, {}), cov.get(g, {})
    if a and b:
        ma, mb = sum(a.values())/len(a), sum(b.values())/len(b)
        print(f"{fam}: -gen={ma:.3f}  +gen={mb:.3f}  delta={mb-ma:+.3f}  (lvls {len(a)}/{len(b)})")
