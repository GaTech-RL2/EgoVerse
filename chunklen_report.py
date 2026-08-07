"""Per-episode chunk counts + avg chunk length from chunkviz npz (argmax structure)."""
import sys, numpy as np
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    tag = path.split("/")[-1].replace(".npz", "")
    for emb in (15, 17):
        if f"emb{emb}_n_episodes" not in z: continue
        nep = int(z[f"emb{emb}_n_episodes"])
        rows = []
        for ep in range(nep):
            cid = z[f"emb{emb}_ep{ep}_cid"]  # (n_stages, T)
            T = cid.shape[1]
            counts = [len(np.unique(cid[s])) for s in range(cid.shape[0])]
            rows.append((T, counts))
        for s in range(rows[0][1].__len__()):
            cnts = [r[1][s] for r in rows]
            avglen = [round(r[0]/max(1,r[1][s]), 1) for r in rows]
            print(f"{tag} emb{emb} Stage{s}: chunks/ep={cnts} avg_len/ep={avglen}")
