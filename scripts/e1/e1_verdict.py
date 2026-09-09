#!/usr/bin/env python3
"""One table answering "does the arc tokenizer beat the time row, and on which metric".

Per run and tertile it puts the three reads side by side:

  arcmatch      the lab's arc-matched paired MSE -- span = min(travel(pred), travel(gt)),
                so a row that under-travels is scored over less trajectory
  gtspan        same construction but span = min(travel(gt), D): identical piece of
                ground-truth path for every row, under-travel paid for
  xyz (time)    time-indexed xyz MSE over all actions, the frame-aligned read

usage: e1_verdict.py [family ...]
"""
import glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/scratch/runs")
FAMS = sys.argv[1:] or ["e1_abc", "e1_abc_30k", "e1_abc_D75", "e1_fold"]
TERTILES = ("test_low", "test_mid", "test_high")


def read(p):
    try:
        r = json.load(open(p))["results"][0]
    except Exception:
        return None
    g = lambda k: next((v for kk, v in r.items() if kk.startswith("Valid/E1/" + k + "/")), None)
    xyz = g("xyz_mse") if not any("xyz_mse_full" in k for k in r) else g("xyz_mse_full")
    return dict(am=g("arcmatch_paired_mse"), amg=g("arcmatch_gtspan_paired_mse"),
                span=g("arcmatch_span_m"), gspan=g("arcmatch_gtspan_span_m"), xyz=xyz)


def family(fam, name):
    v = name.split("_")[0]
    if v == "time":
        return "Time"
    tag = "D75" if "_D75_" in name else "D40"
    return f"{v} {tag}"


rows = defaultdict(list)
for fam in FAMS:
    for run in sorted(glob.glob(f"{ROOT}/{fam}/*/")):
        run = run.rstrip("/"); name = os.path.basename(run)
        for ts in TERTILES:
            for tag in ("best", "final"):
                r = read(f"{run}/eval_{tag}_{ts}/eval_metrics.json")
                if r and r["am"] is not None:
                    rows[(fam, family(fam, name))].append(r)

f = lambda v: "-" if v is None else f"{v:.5f}"
def agg(vals, k):
    xs = [v[k] for v in vals if v.get(k) is not None]
    return sum(xs) / len(xs) if xs else None

print("{:14} {:12} {:>5} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
    "family", "run-family", "n", "span_m", "arcmatch", "gt_span_m", "gtspan", "xyz(time)"))
for (fam, fm), vals in sorted(rows.items()):
    print("{:14} {:12} {:>5} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
        fam, fm, len(vals), f(agg(vals, "span")), f(agg(vals, "am")),
        f(agg(vals, "gspan")), f(agg(vals, "amg")), f(agg(vals, "xyz"))))

have = [k for k, v in rows.items() if agg(v, "amg") is not None]
if not have:
    print("\n(no gt-span evals have landed yet)")
