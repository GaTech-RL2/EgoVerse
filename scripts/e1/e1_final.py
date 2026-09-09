#!/usr/bin/env python3
"""Final read: arc vs Time on the fair (gt-span) arc-matched MSE.

Cell-by-cell (spread x seed), plus the M1 slope -- how each row moves from the
narrow training-tempo spread to the full one. Reads the eval JSONs directly so
nothing is transcribed by hand.
"""
import glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/scratch/runs")
TERT = ("test_low", "test_mid", "test_high")


def val(run, tag, key):
    xs = []
    for ts in TERT:
        try:
            r = json.load(open(f"{run}/eval_{tag}_{ts}/eval_metrics.json"))["results"][0]
        except Exception:
            return None
        v = next((v for kk, v in r.items() if kk.startswith("Valid/E1/" + key + "/")), None)
        if v is None:
            return None
        xs.append(v)
    return sum(xs) / 3


def tag_of(run):
    p = f"{run}/best.json"
    if os.path.exists(p):
        bj = json.load(open(p))
        for k in ("rescore_arcmatch_gtspan_paired_mse", "rescore_arcmatch_paired_mse", "rescore"):
            if isinstance(bj.get(k), dict) and bj[k].get("selected"):
                return bj[k]["selected"]
    return "best"


def collect(fams, rows):
    out = defaultdict(dict)
    for fam in fams:
        for run in glob.glob(f"{ROOT}/{fam}/*/"):
            run = run.rstrip("/"); name = os.path.basename(run)
            m = re.match(r"(.+)_(narrow|medium|full)_s(\d+)$", name)
            if not m or m.group(1) not in rows:
                continue
            v = val(run, tag_of(run), "arcmatch_gtspan_paired_mse")
            if v is not None:
                out[m.group(1)][(m.group(2), int(m.group(3)))] = v
    return out


def report(title, data, base, arcs):
    print(f"\n===== {title} =====")
    cells = [(sp, sd) for sp in ("narrow", "medium", "full") for sd in (42, 43)]
    hdr = "{:12}" + " {:>9}" * len(cells) + " {:>9}"
    print(hdr.format("row", *[f"{sp[:3]}_s{sd}" for sp, sd in cells], "mean"))
    for row in [base] + arcs:
        d = data.get(row, {})
        vs = [d.get(c) for c in cells]
        if any(v is None for v in vs):
            print(f"{row:12} incomplete ({sum(v is not None for v in vs)}/6)"); continue
        print(hdr.format(row, *[f"{v:.5f}" for v in vs], f"{sum(vs)/6:.5f}"))
    b = data.get(base, {})
    if not b or any(b.get(c) is None for c in cells):
        return
    bm = sum(b[c] for c in cells) / 6
    print(f"\nvs {base} (negative = arc better):")
    for row in arcs:
        d = data.get(row, {})
        if any(d.get(c) is None for c in cells):
            continue
        wins = sum(1 for c in cells if d[c] < b[c])
        m = sum(d[c] for c in cells) / 6
        print(f"  {row:12} {100*(m-bm)/bm:+6.1f} %   beats {base} in {wins}/6 cells")
    print("\nM1 slope, narrow -> full (negative = improves with wider tempo spread):")
    for row in [base] + arcs:
        d = data.get(row, {})
        if any(d.get(c) is None for c in cells):
            continue
        n = (d[("narrow", 42)] + d[("narrow", 43)]) / 2
        f = (d[("full", 42)] + d[("full", 43)]) / 2
        print(f"  {row:12} {n:.5f} -> {f:.5f}   {100*(f-n)/n:+6.1f} %")


report("ROBOT - ABC skirts, 30k, gt span 0.370 m",
       collect(["e1_abc", "e1_abc_30k"], {"time", "arcdur", "arclogdur", "arcmean", "arcvel"}),
       "time", ["arcdur", "arclogdur", "arcmean", "arcvel"])
report("HUMAN - mecka freeform fold, 30k, gt span 0.381 m",
       collect(["e1_fold"], {"time", "arcmean", "arcvel"}),
       "time", ["arcmean", "arcvel"])
