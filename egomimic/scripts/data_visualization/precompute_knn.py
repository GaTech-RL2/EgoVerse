"""Precompute CROSS-embodiment KNN for each *_img layer of a latent bank.

For every row, find the k nearest rows of the OTHER embodiment (the
inspector's precomputed-KNN fast path is defined as cross-embodiment;
views.py drops idx=-1/dist=inf padding). Approximate NN via cuML IVF-Flat
on GPU in raw key space (knn_space="raw").

Bundles results back into <layer>_keys.pt (v2 dict: keys + knn_indices
(N,k) int32 + knn_distances (N,k) float32 + knn_k + knn_space), and also
writes a slim laptop sidecar <layer>_keys_knn_only.pt (no keys tensor —
the fast path never reads it; the CSV carries the metadata). Rename the
slim file to <layer>_keys.pt on machines that only need the KNN.

Annotation filtering: none here — extract the bank with
`data.annotated_frames_only=true` (AnnotatedFramesDataset) so unannotated
frames never enter the bank in the first place. Legacy banks whose
frame_idx column is the per-run sample counter cannot be filtered
retroactively (rows can't be mapped to annotation spans).

Usage: python precompute_knn.py <epoch0_dir> [k]

(Producer for the sortpp_latent_knn fulldata sidecars; previously an
uncommitted script in the EgoVerse-6dstack checkout.)
"""

import glob
import os
import sys

import cupy as cp
import numpy as np
import pandas as pd
import torch
from cuml.neighbors import NearestNeighbors

SRC = sys.argv[1].rstrip("/")
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8

for kp in sorted(glob.glob(os.path.join(SRC, "*_img_keys.pt"))):
    layer = os.path.basename(kp)[: -len("_keys.pt")]
    csv_path = os.path.join(SRC, f"{layer}.csv")
    slim_path = os.path.join(SRC, f"{layer}_keys_knn_only.pt")
    if os.path.exists(slim_path):
        print(f"=== {layer}: knn sidecar exists, skipping", flush=True)
        continue
    print(f"=== {layer}", flush=True)
    df = pd.read_csv(csv_path, usecols=["embodiment"])
    embs = df["embodiment"].astype(str).values
    obj = torch.load(kp, map_location="cpu", weights_only=False)
    keys = obj["keys"] if isinstance(obj, dict) else obj
    keys = keys.to(torch.float32).numpy()
    N = keys.shape[0]
    assert N == len(embs), f"keys {N} != csv {len(embs)}"

    knn_idx = np.full((N, K), -1, dtype=np.int32)
    knn_dist = np.full((N, K), np.inf, dtype=np.float32)

    uniq = sorted(set(embs))
    for src_emb in uniq:
        q_mask = embs == src_emb
        b_mask = ~q_mask  # bank = every OTHER embodiment
        if b_mask.sum() == 0:
            continue
        bank_rows = np.nonzero(b_mask)[0]
        bank = cp.asarray(keys[b_mask])
        nn = NearestNeighbors(n_neighbors=K, algorithm="ivfflat", metric="euclidean")
        nn.fit(bank)
        q_rows = np.nonzero(q_mask)[0]
        CH = 2_000_000
        for s in range(0, len(q_rows), CH):
            rows = q_rows[s : s + CH]
            d, i = nn.kneighbors(cp.asarray(keys[rows]))
            knn_idx[rows] = cp.asnumpy(i).astype(np.int32)
            knn_dist[rows] = cp.asnumpy(d).astype(np.float32)
            print(f"  {src_emb}: {min(s + CH, len(q_rows))}/{len(q_rows)}", flush=True)
        # map bank-local neighbor ids -> global row ids
        knn_idx[q_rows] = bank_rows[knn_idx[q_rows]]
        del bank, nn
        cp.get_default_memory_pool().free_all_blocks()

    out = {
        "keys": obj["keys"] if isinstance(obj, dict) else obj,
        "knn_indices": torch.from_numpy(knn_idx),
        "knn_distances": torch.from_numpy(knn_dist),
        "knn_k": K,
        "knn_space": "raw",
        "embs": [],
    }
    torch.save(out, kp + ".tmp")
    os.replace(kp + ".tmp", kp)
    slim = dict(out)
    slim["keys"] = None
    torch.save(slim, slim_path + ".tmp")
    os.replace(slim_path + ".tmp", slim_path)
    print(f"  wrote knn into {os.path.basename(kp)} + slim sidecar", flush=True)
print("PRECOMPUTE_DONE", flush=True)
