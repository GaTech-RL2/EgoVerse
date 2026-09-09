#!/usr/bin/env python3
"""Like-for-like verdict: does the arc tokenizer beat the time row, and on which metric.

Three reads of the same checkpoints, side by side:

  arcmatch   the lab's arc-matched paired MSE. Span = min(travel(pred), travel(gt)),
             so a row that under-travels is scored over less trajectory.
  gtspan     same construction, span = min(travel(gt), arcmatch_gt_span_m). Identical
             piece of gt path for every row; under-travel is paid for, not divided out.
  xyz        time-indexed xyz MSE over all actions, frame-aligned.

Rules that make it like-for-like, unlike a raw harvest:
  * ONE tag per run -- the one best.json recorded as selected, else `best`.
  * only runs with ALL THREE tertiles scored; each run contributes its mean over them.
  * gt spans are checked: comparing rows whose gt span differs is meaningless, and it
    happens by default because arcmatch_gt_span_m follows e1.D (0.40 vs 0.75).

usage: e1_verdict.py [--metric METRIC] [family ...]
"""
import glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.expanduser("~/scratch/runs")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
SELMETRIC = "arcmatch_gtspan_paired_mse"
FAMS = args or ["e1_abc", "e1_abc_30k", "e1_abc_D75", "e1_fold"]
TERTILES = ("test_low", "test_mid", "test_high")


def read(p):
    try:
        r = json.load(open(p))["results"][0]
    except Exception:
        return None
    g = lambda k: next((v for kk, v in r.items() if kk.startswith("Valid/E1/" + k + "/")), None)
    xyz = g("xyz_mse_full") if any("xyz_mse_full" in k for k in r) else g("xyz_mse")
    return dict(am=g("arcmatch_paired_mse"), amg=g("arcmatch_gtspan_paired_mse"),
                span=g("arcmatch_span_m"), gspan=g("arcmatch_gtspan_span_m"), xyz=xyz)


def famname(name):
    v = name.split("_")[0]
    return "Time" if v == "time" else f"{v} {'D75' if '_D75_' in name else 'D40'}"


per_fam = defaultdict(list)
for fam in FAMS:
    for run in sorted(glob.glob(f"{ROOT}/{fam}/*/")):
        run = run.rstrip("/"); name = os.path.basename(run)
        bj_p = f"{run}/best.json"
        tag = "best"
        if os.path.exists(bj_p):
            bj = json.load(open(bj_p))
            for k in (f"rescore_{SELMETRIC}", "rescore_arcmatch_paired_mse", "rescore"):
                if isinstance(bj.get(k), dict) and bj[k].get("selected"):
                    tag = bj[k]["selected"]; break
        vals = [read(f"{run}/eval_{tag}_{ts}/eval_metrics.json") for ts in TERTILES]
        if any(v is None for v in vals):
            continue
        rec = {}
        for k in ("am", "amg", "span", "gspan", "xyz"):
            xs = [v[k] for v in vals if v.get(k) is not None]
            rec[k] = sum(xs) / len(xs) if len(xs) == 3 else None
        per_fam[(fam, famname(name))].append((name, tag, rec))

f = lambda v: "-" if v is None else f"{v:.5f}"
print("{:13} {:14} {:>4} {:>8} {:>9} {:>8} {:>16} {:>9}".format(
    "family", "row", "runs", "span_m", "arcmatch", "gt_span", "gtspan (n)", "xyz(time)"))
gs_seen = {}
for (fam, fm), runs in sorted(per_fam.items()):
    a = lambda k: [r[2][k] for r in runs if r[2][k] is not None]
    m = lambda k: (sum(a(k)) / len(a(k))) if a(k) else None
    # the gtspan column fills in as re-scores land, so state how many runs it covers
    # rather than letting a partial mean hide behind the full run count
    amg = f"{f(m('amg'))} ({len(a('amg'))}/{len(runs)})"
    print("{:13} {:14} {:>4} {:>8} {:>9} {:>8} {:>16} {:>9}".format(
        fam, fm, len(runs), f(m("span")), f(m("am")), f(m("gspan")), amg, f(m("xyz"))))
    if m("gspan") is not None:
        gs_seen[fm] = m("gspan")

if len(gs_seen) > 1:
    lo, hi = min(gs_seen.values()), max(gs_seen.values())
    if hi - lo > 0.05:
        bad = {k: round(v, 3) for k, v in gs_seen.items()}
        print(f"\n!! gt spans disagree by {hi-lo:.3f} m -- the gtspan column is NOT comparable "
              f"across these rows: {bad}\n   pin e1.arcmatch_gt_span_m to one value and re-score.")
    else:
        print(f"\ngt spans agree to {hi-lo:.3f} m -- gtspan column is comparable.")

print("\nper run (mean over the three tertiles):")
for (fam, fm), runs in sorted(per_fam.items()):
    for name, tag, rec in sorted(runs):
        print("  {:26} {:6} arcmatch {:>9} gtspan {:>9} xyz {:>9}".format(
            name, tag, f(rec["am"]), f(rec["amg"]), f(rec["xyz"])))
