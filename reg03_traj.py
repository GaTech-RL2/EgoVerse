import csv, glob, os
for arm in ("bf_prdec_reg03", "bf_prdec_reglow", "bf_prdec_reg3", "bf_prdec_reg001"):
    dirs = sorted(glob.glob(f"logs/indomain_c4/{arm}_2026-*"), key=os.path.getmtime)
    if not dirs: continue
    csvs = glob.glob(dirs[-1] + "/csv_logs/lightning_logs/version_*/metrics.csv")
    if not csvs: continue
    rows = list(csv.reader(open(max(csvs, key=os.path.getmtime))))
    h = rows[0]; idx = {n: i for i, n in enumerate(h)}
    ei = idx["epoch"]
    want = {n: idx[n] for n in h if "frac_indecisive" in n}
    by_ep = {}
    for r in rows[1:]:
        if ei < len(r) and r[ei]:
            vals = {n: r[i] for n, i in want.items() if i < len(r) and r[i]}
            if vals: by_ep[int(float(r[ei]))] = vals
    eps = sorted(by_ep)
    picks = [e for t in (1, 25, 50, 100, 150, 200, 250, 300) for e in [max((x for x in eps if x <= t), default=None)] if e is not None]
    seen = set(); out = []
    for e in picks:
        if e in seen: continue
        seen.add(e)
        v = by_ep[e]
        def g(k):
            x = [v[n] for n in sorted(v) if k in n]
            return "/".join(f"{float(y):.2f}" for y in x) if x else "-"
        out.append(f"    ep{e:4d}: L0seam={g('L0')} L1bottom={g('L1')}")
    print(arm, f"(latest ep{eps[-1]})")
    print("\n".join(out))
