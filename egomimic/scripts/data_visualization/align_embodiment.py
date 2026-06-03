"""Embodiment-alignment sweep over captured latents (action-expert layers).

Hypothesis under test: in the raw / PCA-50 latent space, a few directions
encode *embodiment* (aria vs eva) rather than task semantics. If we strip those
directions, the human and robot manifolds superimpose and ordinary KNN returns
cross-embodiment neighbors organically — no explicit cross-embodiment sampling.

For each action-expert layer (`expert_layer_NN`) this script:
  1. loads `<layer>_keys.pt` (N, D) + the row-aligned `<layer>.csv`
     (video_hash, embodiment, frame_idx, token_idx),
  2. attaches a frame-level *semantic* label = the language annotation spanning
     that (hash, frame) — pulled from the zarr the same way the inspector does,
  3. applies several embodiment-removal transforms,
  4. scores each transform on two axes:
        - embodiment removal  : logreg 5-fold CV acc (want -> chance),
        - semantic preservation: annotation kNN 5-fold CV acc (want stays high),
        - cross-emb kNN rate  : mean fraction of each point's k-NN that are the
                                 OTHER embodiment (want -> population mix),
  5. optionally writes an inspector-loadable UMAP CSV per (layer, method).

Methods
  none        standardized features, no removal (baseline)
  mean_center subtract each embodiment's own mean (rank-1 domain offset)
  zscore_emb  per-embodiment mean+std standardization (1st+2nd order offset)
  drop_pc:k   PCA-50, drop the top-k embodiment-discriminative PCs (Fisher rank)

Run (emimic venv, on a GPU node so cuML UMAP is available):
  python egomimic/scripts/data_visualization/align_embodiment.py \
    --latent-dir logs/objgen6d_latent/.../latents/epoch_0 \
    --zarr-root  /storage/.../datasets/pick_place_cotrain \
    --out-dir    logs/objgen6d_latent/.../latents/epoch_0/aligned \
    --umap   # also emit UMAP CSV/PNG per (layer, method)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np

# sklearn is numpy-2 compatible in recent versions; all linear algebra here is
# small (subsampled), so CPU sklearn is fine. cuML is used only for UMAP.
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from egomimic.scripts.data_visualization.inspector_lib.language import (
    annotation_intervals,
    lang_for_frame,
)

DROP_KS = (1, 2, 3, 5, 10)  # top-k embodiment PCs to drop for drop_pc sweep
PCA_DIM = 50
KNN_K = 10
CV = 5
RNG = np.random.RandomState(0)


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def list_expert_layers(latent_dir):
    csvs = sorted(glob.glob(os.path.join(latent_dir, "expert_layer_*.csv")))
    # drop any _img/_lang/_combined/_tsne derivatives — bare expert layers only
    return [
        c
        for c in csvs
        if os.path.basename(c)[: -len(".csv")].replace("expert_layer_", "").isdigit()
    ]


def load_layer(csv_path):
    import torch

    keys_path = csv_path[: -len(".csv")] + "_keys.pt"
    if not os.path.isfile(keys_path):
        return None
    X = torch.load(keys_path, map_location="cpu").to(torch.float32).numpy()
    hashes, embs, frames = [], [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            hashes.append(row["video_hash"])
            embs.append(row["embodiment"])
            frames.append(int(row.get("frame_idx", -1)))
    n = min(len(hashes), X.shape[0])
    return X[:n], np.array(hashes[:n]), np.array(embs[:n]), np.array(frames[:n])


def annotation_labels(zarr_root, hashes, frames):
    """Frame-level annotation string per row (cached per hash)."""
    cache = {}
    out = []
    for h, fr in zip(hashes, frames):
        if h not in cache:
            try:
                cache[h] = annotation_intervals(zarr_root, h)
            except Exception:
                cache[h] = tuple()
        out.append(lang_for_frame(cache[h], int(fr)) or "")
    return np.array(out)


def balanced_subsample(embs, max_rows):
    idx_by = {e: np.where(embs == e)[0] for e in np.unique(embs)}
    per = max(1, max_rows // max(1, len(idx_by)))
    keep = []
    for e, ix in idx_by.items():
        if len(ix) > per:
            ix = RNG.choice(ix, per, replace=False)
        keep.append(ix)
    keep = np.concatenate(keep)
    RNG.shuffle(keep)
    return keep


# ----------------------------------------------------------------------------
# Removal methods  ->  return cleaned feature matrix F (N, d)
# ----------------------------------------------------------------------------
def method_none(Xs, y):
    return Xs


def method_mean_center(Xs, y):
    F = Xs.copy()
    for e in np.unique(y):
        m = Xs[y == e].mean(0)
        F[y == e] -= m
    return F


def method_zscore_emb(Xs, y):
    F = Xs.copy()
    for e in np.unique(y):
        sl = y == e
        m, s = Xs[sl].mean(0), Xs[sl].std(0) + 1e-6
        F[sl] = (Xs[sl] - m) / s
    return F


def method_drop_pc(Xs, y, k, pca_scores=None, fisher_order=None):
    """PCA-50 then zero the top-k embodiment-discriminative PCs (by Fisher)."""
    order = fisher_order
    Z = pca_scores
    keep = order[k:]  # drop the k most embodiment-discriminative PCs
    return Z[:, keep]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def emb_decode_acc(F, y_emb):
    return float(
        cross_val_score(
            LogisticRegression(max_iter=2000), F, y_emb, cv=CV, n_jobs=-1
        ).mean()
    )


def cross_emb_knn_rate(F, y_emb, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(F)
    _, idx = nn.kneighbors(F)
    idx = idx[:, 1:]  # drop self
    other = (y_emb[idx] != y_emb[:, None]).mean()
    return float(other)


def semantic_knn_acc(F, y_sem, k=KNN_K):
    # keep classes with enough support for CV
    cnt = Counter(y_sem)
    good = {c for c, n in cnt.items() if n >= max(CV, k + 1) and c != ""}
    if len(good) < 2:
        return float("nan"), 0, 0
    m = np.array([s in good for s in y_sem])
    Fg, yg = F[m], y_sem[m]
    shared_note = len(good)
    acc = float(
        cross_val_score(
            KNeighborsClassifier(n_neighbors=k), Fg, yg, cv=CV, n_jobs=-1
        ).mean()
    )
    return acc, int(m.sum()), shared_note


def shared_annotation_classes(y_sem, y_emb):
    by = {e: set(y_sem[y_emb == e]) for e in np.unique(y_emb)}
    sets = [s - {""} for s in by.values()]
    return len(set.intersection(*sets)) if sets else 0


# ----------------------------------------------------------------------------
# UMAP (cuML) viz output for the inspector
# ----------------------------------------------------------------------------
def write_umap_csv(out_csv, F, hashes, embs, frames):
    try:
        from cuml.manifold import UMAP as cuUMAP

        xyz = cuUMAP(n_components=3, random_state=0).fit_transform(F)
        xyz = np.asarray(xyz)
    except Exception as e:  # noqa: BLE001
        print(f"    [umap] skipped ({type(e).__name__}: {e})")
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video_hash",
                "embodiment",
                "frame_idx",
                "token_idx",
                "umap_x",
                "umap_y",
                "umap_z",
            ]
        )
        for i in range(F.shape[0]):
            w.writerow(
                [
                    hashes[i],
                    embs[i],
                    int(frames[i]),
                    0,
                    float(xyz[i, 0]),
                    float(xyz[i, 1]),
                    float(xyz[i, 2]),
                ]
            )


# ----------------------------------------------------------------------------
def run_layer(csv_path, zarr_root, out_dir, max_rows, do_umap):
    name = os.path.basename(csv_path)[: -len(".csv")]
    loaded = load_layer(csv_path)
    if loaded is None:
        print(f"[{name}] no keys.pt — skip")
        return []
    X, hashes, embs, frames = loaded
    sub = balanced_subsample(embs, max_rows)
    X, hashes, embs, frames = X[sub], hashes[sub], embs[sub], frames[sub]
    y_sem = annotation_labels(zarr_root, hashes, frames)

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(PCA_DIM, Xs.shape[1])).fit(Xs)
    Z = pca.transform(Xs)
    # Fisher ratio of each PC between the two embodiments -> drop order
    classes = np.unique(embs)
    fisher = np.zeros(Z.shape[1])
    if len(classes) == 2:
        a, b = (embs == classes[0]), (embs == classes[1])
        num = (Z[a].mean(0) - Z[b].mean(0)) ** 2
        den = Z[a].var(0) + Z[b].var(0) + 1e-9
        fisher = num / den
    fisher_order = np.argsort(-fisher)  # most embodiment-discriminative first

    chance = max(Counter(embs).values()) / len(embs)
    pop_other = (
        float((embs != classes[0]).mean()) if len(classes) == 2 else float("nan")
    )
    shared = shared_annotation_classes(y_sem, embs)

    methods = {
        "none": method_none(Xs, embs),
        "mean_center": method_mean_center(Xs, embs),
        "zscore_emb": method_zscore_emb(Xs, embs),
    }
    for k in DROP_KS:
        methods[f"drop_pc:{k}"] = method_drop_pc(
            Xs, embs, k, pca_scores=Z, fisher_order=fisher_order
        )

    rows = []
    print(
        f"\n[{name}]  N={len(embs)}  emb-chance={chance:.2f}  "
        f"pop_other={pop_other:.2f}  annot_classes(shared 2-emb)={shared}  "
        f"top-PC-Fisher={fisher[fisher_order[0]]:.3f}"
    )
    print(f"  {'method':<14}{'emb_acc':>9}{'crossKNN':>10}{'sem_acc':>9}{'sem_n':>7}")
    for mname, F in methods.items():
        ea = emb_decode_acc(F, embs)
        ck = cross_emb_knn_rate(F, embs)
        sa, sn, _ = semantic_knn_acc(F, y_sem)
        print(
            f"  {mname:<14}{ea:>9.3f}{ck:>10.3f}"
            f"{(sa if sa==sa else float('nan')):>9.3f}{sn:>7d}"
        )
        rows.append(
            dict(
                layer=name,
                method=mname,
                n=len(embs),
                emb_chance=round(chance, 4),
                emb_acc=round(ea, 4),
                cross_emb_knn=round(ck, 4),
                pop_other_frac=round(pop_other, 4),
                sem_knn_acc=(round(sa, 4) if sa == sa else None),
                sem_n=sn,
                shared_annot_classes=shared,
            )
        )
        if do_umap and mname in ("none", "mean_center", "drop_pc:3", "drop_pc:10"):
            write_umap_csv(
                os.path.join(out_dir, name, mname.replace(":", "_"), f"{name}.csv"),
                F,
                hashes,
                embs,
                frames,
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--zarr-root", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument(
        "--layers",
        default=None,
        help="comma list of layer indices, e.g. 0,8,17 (default: all)",
    )
    ap.add_argument("--umap", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.latent_dir, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    layers = list_expert_layers(args.latent_dir)
    if args.layers:
        want = {f"{int(i):02d}" for i in args.layers.split(",")}
        layers = [c for c in layers if os.path.basename(c).split("_")[-1][:2] in want]
    if not layers:
        raise SystemExit(f"No expert_layer_*.csv in {args.latent_dir}")
    print(f"Sweeping {len(layers)} action-expert layers -> {out_dir}")

    all_rows = []
    for c in layers:
        all_rows += run_layer(c, args.zarr_root, out_dir, args.max_rows, args.umap)

    with open(os.path.join(out_dir, "alignment_metrics.json"), "w") as f:
        json.dump(all_rows, f, indent=2)
    # also a flat CSV for quick eyeballing
    if all_rows:
        with open(os.path.join(out_dir, "alignment_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nWrote {out_dir}/alignment_metrics.{{json,csv}}")


if __name__ == "__main__":
    main()
