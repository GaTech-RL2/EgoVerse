import re, glob
pat = re.compile(r"ep_coverages:\s*(.+)")
num = re.compile(r"([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)")

def agg_id(tag):
    for emb in (15, 17):
        vals = []
        for f in glob.glob(f"eval_bf/{tag}_par_e{emb}_s*.log"):
            for line in open(f, errors="ignore"):
                m = pat.search(line)
                if m: vals += [float(x) for x in num.findall(m.group(1))]
        if vals:
            print(f"ID {tag} e{emb}: n={len(vals)} mean={sum(vals)/len(vals):.3f} SR={sum(v>=0.8 for v in vals)/len(vals):.3f}")

def agg_obs(tag):
    for emb in (15, 17):
        means, pool = [], []
        for L in range(30):
            vals = []
            for f in glob.glob(f"eval_bf/obst_{tag}/L{L}_e{emb}.log"):
                for line in open(f, errors="ignore"):
                    m = pat.search(line)
                    if m: vals = [float(x) for x in num.findall(m.group(1))]
            if vals: means.append(sum(vals)/len(vals)); pool += vals
        if means:
            print(f"OBS {tag} e{emb}: {len(means)}/30 mean={sum(means)/len(means):.3f} SR={sum(v>=0.8 for v in pool)/len(pool):.3f}")

for t in ("bf_symw1_gmm_ep1999", "bf_nopre_gmm_ep2999", "bf_prdec_hb_ep499", "bf_symw1_hb_ep499"):
    agg_id(t)
for t in ("symw1gmm999", "symw1gmm1999", "nopre999", "nopre2999"):
    agg_obs(t)
