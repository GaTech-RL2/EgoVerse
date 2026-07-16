import csv, glob, os
d = sorted(glob.glob("logs/indomain_c4/bf_cvae_zs_var4_2026-*"), key=os.path.getmtime)[-1]
rows = list(csv.reader(open(glob.glob(d + "/csv_logs/lightning_logs/version_*/metrics.csv")[0])))
h = rows[0]; idx = {n: i for i, n in enumerate(h)}
ei = idx["epoch"]
want = {n: idx[n] for n in h if "frac_indecisive" in n}
by_ep = {}
for r in rows[1:]:
    if ei < len(r) and r[ei]:
        vals = {n: r[i] for n, i in want.items() if i < len(r) and r[i]}
        if vals:
            by_ep[int(float(r[ei]))] = vals
for target in (59, 100, 184, 250, 350, 500, 670):
    best = max((e for e in by_ep if e <= target + 8), default=None)
    if best is None: continue
    v = by_ep[best]
    def g(k):
        x = [v[n] for n in sorted(v) if k in n]
        return "/".join(f"{float(y):.2f}" for y in x) if x else "-"
    print(f"ep{best:4d}: frcInd15={g('15_L0')},{g('15_L1')} frcInd17={g('17_L0')},{g('17_L1')}")
