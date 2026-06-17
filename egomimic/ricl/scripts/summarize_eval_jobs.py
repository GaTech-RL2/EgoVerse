#!/usr/bin/env python3
"""Scan RoboTwin eval .out logs, print per-(job,model,task) success rates, and flag
any task where the policies diverge (a difference between RICL/xemb and plain)."""

import glob
import os
import re
import sys


def strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse(path):
    rows = []  # (model, task, suc, n, done)
    cur = None
    suc = n = 0
    cfg = "?"
    for ln in open(path, encoding="utf-8", errors="ignore"):
        s = strip(ln)
        mc = re.search(r"task_config=(\w+)", s)
        if mc:
            cfg = mc.group(1)
        m = re.search(r"EVAL model=(\w+) task=(\w+)", s)
        if m:
            if cur:
                rows.append((*cur, suc, n))
            cur = (m.group(1), m.group(2))
            suc = n = 0
        r = re.search(r"Success rate: (\d+)/(\d+)", s)
        if r:
            suc, n = int(r.group(1)), int(r.group(2))
    if cur:
        rows.append((*cur, suc, n))
    done = (
        "mt eval compare DONE" in open(path, encoding="utf-8", errors="ignore").read()
        or "xemb eval compare DONE"
        in open(path, encoding="utf-8", errors="ignore").read()
    )
    return cfg, rows, done


paths = sys.argv[1:] or sorted(
    glob.glob("egomimic/ricl/outputs/robotwin_eval/*compare_*.out")
)
# task -> {model: (suc,n)} across all jobs (keyed by config to avoid mixing)
agg = {}
for p in paths:
    cfg, rows, done = parse(p)
    jid = re.search(r"_(\d+)\.out", p)
    jid = jid.group(1) if jid else os.path.basename(p)
    print(f"\n=== {jid}  config={cfg}  {'DONE' if done else 'running'} ===")
    for model, task, suc, n in rows:
        print(f"  {model:6s} {task:18s} {suc}/{n}")
        agg.setdefault((cfg, task), {})[model] = (suc, n)

print("\n=== DIVERGENCE (RICL/xemb vs plain, same task+config, both with >=5 eps) ===")
found = False
for (cfg, task), d in sorted(agg.items()):
    if "plain" not in d:
        continue
    ps, pn = d["plain"]
    if pn < 5:
        continue
    for model in ("xemb", "same", "ricl"):
        if model not in d:
            continue
        ms, mn = d[model]
        if mn < 5:
            continue
        diff = ms / mn - ps / pn
        if abs(diff) >= 0.2:  # >=20 pp gap
            found = True
            print(
                f"  [{cfg}] {task}: {model} {ms}/{mn} vs plain {ps}/{pn}  (Δ={diff:+.0%})"
            )
if not found:
    print("  none yet (no >=20pp gap among completed cells)")
