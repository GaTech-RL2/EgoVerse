#!/usr/bin/env python3
"""E2 / M2: is the arc rows' error flat across TEST tempo?

M2: error-vs-test-tempo slope ~ 0 for arc, > 0 for Time.
Null: equal slopes, or a positive arc slope.

The old E2 read was on E_time and every row rose ~9 cm slow->fast, which the
run matrix attributes to scoring over a fixed TIME window: a faster episode
covers more path in the same 100 frames, so there is more to get wrong. The
gt-span metric scores over a fixed PROGRESS horizon instead, so it should
remove most of that artefact -- the realized span per tertile is printed to
check that it actually does.
"""
import glob, json, os, re
from collections import defaultdict

ROOT = os.path.expanduser("~/scratch/runs")
TERT = ("test_low", "test_mid", "test_high")


def tag_of(run):
    p = f"{run}/best.json"
    if os.path.exists(p):
        bj = json.load(open(p))
        for k in ("rescore_arcmatch_gtspan_paired_mse", "rescore_arcmatch_paired_mse", "rescore"):
            if isinstance(bj.get(k), dict) and bj[k].get("selected"):
                return bj[k]["selected"]
    return "best"


def get(run, tag, ts, key):
    try:
        r = json.load(open(f"{run}/eval_{tag}_{ts}/eval_metrics.json"))["results"][0]
    except Exception:
        return None
    return next((v for kk, v in r.items() if kk.startswith("Valid/E1/" + key + "/")), None)


def run_source(title, fams, rows, snap=None):
    per = defaultdict(lambda: defaultdict(list))
    span = defaultdict(lambda: defaultdict(list))
    et = defaultdict(lambda: defaultdict(list))
    for fam in fams:
        for run in glob.glob(f"{ROOT}/{fam}/*/"):
            run = run.rstrip("/"); name = os.path.basename(run)
            m = re.match(r"(.+)_(narrow|medium|full)_s(\d+)$", name)
            if not m or m.group(1) not in rows:
                continue
            t = tag_of(run)
            for ts in TERT:
                v = get(run, t, ts, "arcmatch_gtspan_paired_mse")
                s = get(run, t, ts, "arcmatch_gtspan_span_m")
                e = get(run, t, ts, "E_time")
                if v is not None:
                    per[m.group(1)][ts].append(v)
                if s is not None:
                    span[m.group(1)][ts].append(s)
                if e is not None:
                    et[m.group(1)][ts].append(e)
    if snap and os.path.exists(snap):
        d = json.load(open(snap))
        for ts in TERT:
            if ts in d.get("per_tertile", {}):
                per["time"][ts] = d["per_tertile"][ts]

    print(f"\n===== {title} =====")
    print("{:11} {:>5} {:>9} {:>9} {:>9} {:>12} {:>11}".format(
        "row", "n", "low", "mid", "high", "low->high", "E_time l->h"))
    for row in rows:
        p = per.get(row, {})
        if not all(p.get(ts) for ts in TERT):
            print(f"{row:11} incomplete"); continue
        mu = {ts: sum(p[ts]) / len(p[ts]) for ts in TERT}
        slope = 100 * (mu["test_high"] - mu["test_low"]) / mu["test_low"]
        e = et.get(row, {})
        es = ""
        if all(e.get(ts) for ts in TERT):
            el = sum(e["test_low"]) / len(e["test_low"]); eh = sum(e["test_high"]) / len(e["test_high"])
            es = f"{100*(eh-el):+.2f} cm"
        print("{:11} {:>5} {:>9.5f} {:>9.5f} {:>9.5f} {:>11.1f}% {:>11}".format(
            row, len(p["test_low"]), mu["test_low"], mu["test_mid"], mu["test_high"], slope, es))
    s = span.get(rows[0] if rows[0] in span else next(iter(span), ""), {})
    if all(s.get(ts) for ts in TERT):
        print("  realized gt span (m):  " + "  ".join(
            f"{ts.replace('test_','')} {sum(s[ts])/len(s[ts]):.4f}" for ts in TERT))


run_source("ROBOT - ABC skirts (test tertile medians 0.312 / 0.367 / 0.438 m/s)",
           ["e1_abc", "e1_abc_30k"], ["time", "arcdur", "arclogdur", "arcmean", "arcvel"],
           snap=os.path.expanduser("~/scratch/runs/time_baseline_snapshot.json"))
run_source("HUMAN - mecka fold (test tertile medians 0.218 / 0.281 / 0.350 m/s)",
           ["e1_fold"], ["time", "arcmean", "arcvel"])
