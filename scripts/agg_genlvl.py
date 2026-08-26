#!/usr/bin/env python3
# Aggregate genlvl eval: per cell, mean of the 30 per-obstacle-level emb15 coverages.
import glob, re, os
BASE = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/logs/genlvl"
CELLS = ["hptC8_nogen", "hptC8_gen", "txC8_nogen", "txC8_gen"]
cov = {}
for cell in CELLS:
    perlvl = {}
    for f in glob.glob(f"{BASE}/genlvl_{cell}_*.out"):
        m = re.search(r"_(\d+)\.out$", f)
        if not m:
            continue
        lvl = int(m.group(1))
        txt = open(f, errors="ignore").read()
        c = re.search(r"emb15_sim_coverage\D+([0-9]+\.[0-9]+)", txt)
        if c:
            perlvl[lvl] = float(c.group(1))
    cov[cell] = perlvl
    if perlvl:
        mean = sum(perlvl.values()) / len(perlvl)
        print(f"{cell}: mean={mean:.3f}  levels_done={len(perlvl)}/30")
    else:
        print(f"{cell}: NONE yet")
# Δ if both halves present
for fam, ng, g in [("HPT-flow c8", "hptC8_nogen", "hptC8_gen"), ("TransformerAR c8", "txC8_nogen", "txC8_gen")]:
    a, b = cov.get(ng, {}), cov.get(g, {})
    if a and b:
        ma, mb = sum(a.values())/len(a), sum(b.values())/len(b)
        print(f"{fam}:  -gen={ma:.3f}  +gen={mb:.3f}  Δ={mb-ma:+.3f}  (levels {len(a)}/{len(b)})")
