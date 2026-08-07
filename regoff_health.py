import csv, glob, os
ARMS = ["bf_prdec_regoff499", "bf_prdec_regoff999", "bf_prdec_ste4_499",
        "bf_prdec_regoff_ste4_499", "bf_prdec_ste4", "bf_prdec_ste8",
        "bf_cvae_var4_pre", "bf_prdec_reglow499", "bf_prdec_reglow",
        "bf_prdec_reg03", "bf_prdec_reg3", "bf_prdec_reg001", "bf_symw1_var4"]
for arm in ARMS:
    dirs = sorted(glob.glob(f"logs/indomain_c4/{arm}_2026-*"), key=os.path.getmtime)
    if not dirs:
        print(f"{arm:26s} NO RUN"); continue
    csvs = glob.glob(dirs[-1] + "/csv_logs/lightning_logs/version_*/metrics.csv")
    if not csvs:
        print(f"{arm:26s} NO CSV"); continue
    rows = list(csv.reader(open(max(csvs, key=os.path.getmtime))))
    h = rows[0]; idx = {n: i for i, n in enumerate(h)}
    want = [n for n in h if "frac_indecisive" in n or "boundary_rate" in n]
    ei = idx.get("epoch"); last = {}; ep = "?"
    for r in rows[1:]:
        for n in want:
            i = idx[n]
            if i < len(r) and r[i]:
                last[n] = r[i]
                if ei is not None and ei < len(r) and r[ei]:
                    ep = r[ei]
    def g(k):
        v = [last[n] for n in sorted(last) if k in n]
        return "/".join(f"{float(x):.2f}" for x in v) if v else "-"
    print(f"{arm:26s} ep~{ep:>4s} frcInd15={g('15_L0_frac')},{g('15_L1_frac')} "
          f"frcInd17={g('17_L0_frac')},{g('17_L1_frac')} "
          f"rate15={g('15_L0_bound')},{g('15_L1_bound')} rate17={g('17_L0_bound')},{g('17_L1_bound')}")
