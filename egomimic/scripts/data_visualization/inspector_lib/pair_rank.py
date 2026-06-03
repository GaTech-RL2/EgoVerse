"""Perfect-pair retrieval report: average rank ("place") of the paired action.

For each layer in a run dir, sample N rows; for every sampled row, rank all
opposite-embodiment ACTION instances (recording + annotation interval, the same
identity the KNN dedupe uses) by their min row distance to the query, and find
the place of the query's true paired action (1 = nearest). Reports avg/median
place and top-1 rate per layer.

Two report flavours share the same ranking core (`_places_for_coords`):

  * `pair_rank_report` — distances in a CSV-baked reduction space
    (umap / pca_umap / tsne2d), the same spaces the scatter view plots. Cheap,
    matches what the eye sees, no multi-GB `keys.pt` reads.

  * `pair_rank_sweep` — distances in the *embedding* spaces the inspector's
    KNN pane actually queries: full-D raw keys, PCA-50, and PCA-50 with the
    top-k aria-vs-eva Fisher dims removed (a removal sweep in steps of `drop`).
    Restricted to a handful of focus layers; reports mean place only. This is
    the apples-to-apples answer to "does dropping the embodiment-discriminative
    directions pull the true cross-embodiment twin to the front?".
"""

from __future__ import annotations

import glob
import logging
import os
import time

import numpy as np
import pandas as pd

from .caches import LayerStore
from .views import _find_pair_action, _row_action_ids

logger = logging.getLogger(__name__)

_SPACE_COLS = {
    "umap": ["umap_x", "umap_y", "umap_z"],
    "pca_umap": ["pca_umap_x", "pca_umap_y", "pca_umap_z"],
    "tsne2d": ["tsne2d_x", "tsne2d_y"],
}

_META_COLS = ["video_hash", "frame_idx", "token_idx", "embodiment"]

# The focus layers requested for the objgen6d pair-rank sweep: the expert's
# first / middle / last block, and PaliGemma's first + last block in both the
# combined (img+lang) and lang-only token slices.
DEFAULT_SWEEP_LAYERS = [
    "expert_layer_00",
    "expert_layer_12",
    "expert_layer_17",
    "paligemma_layer_17_combined",
    "paligemma_layer_00_combined",
    "paligemma_layer_17_lang",
    "paligemma_layer_00_lang",
]


def _build_pairing(zarr_root, data):
    """Space-independent half of the pair-rank computation.

    Returns (ids, paired_aid_per_row, eligible):
      * ids                 (N,)  action-instance id of every row
      * paired_aid_per_row  (N,)  the opposite-embodiment action id paired with
                                  each row's action (-1 when the row's action
                                  has no twin)
      * eligible            (E,)  row indices whose action has a valid pair

    This is the expensive part — it hits the zarr annotation store once per
    action via `_find_pair_action` — so the sweep computes it a single time per
    layer and reuses it across every feature space.
    """
    ids = _row_action_ids(zarr_root, data)
    embs = data["embs"]

    # action id -> representative row (first occurrence)
    _, first_pos = np.unique(ids, return_index=True)
    rep_row = {int(ids[i]): int(i) for i in first_pos}

    # action id -> paired opposite-embodiment action id (None when no twin)
    pair_of: dict[int, int | None] = {}
    for aid, ridx in rep_row.items():
        res = _find_pair_action(
            zarr_root,
            data,
            data["hashes"][ridx],
            int(data["frame_idx"][ridx]),
            str(embs[ridx]),
        )
        if res is None:
            pair_of[aid] = None
            continue
        twin_h, s2, e2, _prompt = res
        twin_rows = np.where(
            (data["hashes"] == twin_h)
            & (data["frame_idx"] >= s2)
            & (data["frame_idx"] < e2)
        )[0]
        pair_of[aid] = int(ids[twin_rows[0]]) if twin_rows.size else None

    paired_aid_per_row = np.array(
        [pair_of.get(int(a)) if pair_of.get(int(a)) is not None else -1 for a in ids],
        dtype=np.int64,
    )
    eligible = np.where(paired_aid_per_row >= 0)[0]
    return ids, paired_aid_per_row, eligible


def _places_for_coords(
    coords, ids, embs, paired_aid_per_row, eligible, n_samples, seed
):
    """Ranking core for one feature space.

    `coords` is the (N, D) feature matrix (reduction coords, raw keys, PCA
    features, ...). Samples `n_samples` eligible rows, and for each ranks every
    opposite-embodiment ACTION by its min row-to-query distance, returning the
    place of the true paired action. Returns (places list, avg #opp actions).
    """
    if eligible.size == 0:
        return [], 0.0
    coords = np.ascontiguousarray(coords, dtype=np.float32)

    rng = np.random.default_rng(seed)
    take = min(n_samples, eligible.size)
    qidx = rng.choice(eligible, size=take, replace=False)

    places: list[int] = []
    n_opp_actions_seen = []
    for emb in np.unique(embs[qidx]):
        q_e = qidx[embs[qidx] == emb]
        opp_rows = np.where(embs != emb)[0]
        if opp_rows.size == 0:
            continue
        # Sort opposite rows by action id so per-action mins reduce with
        # one reduceat per query instead of a python loop over actions.
        order = np.argsort(ids[opp_rows], kind="stable")
        opp_sorted = opp_rows[order]
        opp_ids_sorted = ids[opp_sorted]
        opp_action_ids, starts = np.unique(opp_ids_sorted, return_index=True)
        opp_coords = coords[opp_sorted]
        n_opp_actions_seen.append(len(opp_action_ids))

        paired_pos = np.searchsorted(opp_action_ids, paired_aid_per_row[q_e])
        # Guard: paired action must exist on the opposite side of this layer.
        valid = (paired_pos < len(opp_action_ids)) & (
            opp_action_ids[np.minimum(paired_pos, len(opp_action_ids) - 1)]
            == paired_aid_per_row[q_e]
        )
        q_e, paired_pos = q_e[valid], paired_pos[valid]

        # ||q-x||² = ||q||² + ||x||² − 2 q·x, chunked — the broadcasted
        # (chunk, M, D) diff tensor peaked at ~2 GB on the 2.7M-row
        # paligemma layers and got the report OOM-killed on login nodes.
        opp_sq = np.einsum("nd,nd->n", opp_coords, opp_coords)
        CHUNK = 32
        for c0 in range(0, len(q_e), CHUNK):
            qc = q_e[c0 : c0 + CHUNK]
            pc = paired_pos[c0 : c0 + CHUNK]
            qcoords = coords[qc]
            q_sq = np.einsum("qd,qd->q", qcoords, qcoords)
            d2 = q_sq[:, None] + opp_sq[None, :] - 2.0 * (qcoords @ opp_coords.T)
            mins = np.minimum.reduceat(d2, starts, axis=1)  # (c, n_opp_actions)
            paired_min = mins[np.arange(len(qc)), pc]
            place = 1 + (mins < paired_min[:, None]).sum(axis=1)
            places.extend(place.tolist())
    return places, (float(np.mean(n_opp_actions_seen)) if n_opp_actions_seen else 0.0)


def _layer_places(zarr_root, df, space_cols, n_samples, seed):
    """Per-layer place computation for the CSV-coord report (thin wrapper)."""
    data = {
        "hashes": df["video_hash"].to_numpy(str),
        "frame_idx": df["frame_idx"].to_numpy(np.int64),
        "embs": df["embodiment"].to_numpy(str),
    }
    coords = df[space_cols].to_numpy(np.float32)
    ids, paired_aid_per_row, eligible = _build_pairing(zarr_root, data)
    return _places_for_coords(
        coords, ids, data["embs"], paired_aid_per_row, eligible, n_samples, seed
    )


def pair_rank_report(
    run_dir: str,
    zarr_root: str,
    n_samples: int = 1000,
    space: str = "umap",
    seed: int = 0,
) -> None:
    """Print the per-layer average place of the perfect pair (CSV-coord space)."""
    if space not in _SPACE_COLS:
        raise ValueError(f"space must be one of {sorted(_SPACE_COLS)}, got {space!r}")
    space_cols = _SPACE_COLS[space]
    csvs = sorted(glob.glob(os.path.join(run_dir, "*.csv")))
    if not csvs:
        raise SystemExit(f"No layer CSVs in {run_dir}")

    print(
        f"\nPerfect-pair retrieval report — avg place of the paired action "
        f"(1 = nearest) among opposite-embodiment actions\n"
        f"run: {run_dir}\nspace: {space} | samples/layer: {n_samples} | seed: {seed}\n"
    )
    header = f"{'layer':38s} {'n_eval':>6s} {'avg':>7s} {'median':>7s} {'top1%':>6s} {'#actions':>8s}"
    print(header)
    print("-" * len(header))
    for path in csvs:
        layer = os.path.basename(path)[:-4]
        t0 = time.time()
        try:
            df = pd.read_csv(path, usecols=_META_COLS + space_cols)
        except ValueError:
            print(f"{layer:38s}   (missing {'/'.join(space_cols)} — skipped)")
            continue
        places, n_opp = _layer_places(zarr_root, df, space_cols, n_samples, seed)
        if not places:
            print(f"{layer:38s}   (no paired actions found — skipped)")
            continue
        arr = np.asarray(places)
        print(
            f"{layer:38s} {len(arr):>6d} {arr.mean():>7.2f} "
            f"{np.median(arr):>7.1f} {100.0 * (arr == 1).mean():>5.1f}% "
            f"{n_opp:>8.0f}  [{time.time() - t0:.1f}s]"
        )
    print(
        "\nplace = rank of the true paired action when all opposite-embodiment "
        "actions are sorted by min row distance to the sampled row.\n"
        "Chance level = (#actions + 1) / 2."
    )


def _sweep_configs(store, run_dir, n_components, drop_steps):
    """Ordered list of (label, feature-builder) for the embedding-space sweep.

    Each builder takes a layer name and returns its (N, D) feature matrix (or
    None when the raw keys / Fisher ranking aren't available):
      * raw                          full-D raw keys
      * pca{C}                       C-dim PCA features (no removal)
      * pca{C}_drop{k}  (k in steps) PCA features with the top-k aria-vs-eva
                                     Fisher dims deleted — the embodiment-
                                     neutral subspace.
    """
    cfgs: list[tuple[str, callable]] = [
        ("raw", lambda layer: store.load_keys(run_dir, layer)),
        (
            f"pca{n_components}",
            lambda layer: store.pca_features(run_dir, layer, n_components),
        ),
    ]
    for k in drop_steps:
        if k <= 0 or k >= n_components:
            continue  # drop >= n_components would empty the space
        cfgs.append(
            (
                f"pca{n_components}_drop{k}",
                lambda layer, k=k: store.pca_features_dropped(
                    run_dir, layer, n_components, k
                ),
            )
        )
    return cfgs


def pair_rank_sweep(
    run_dir: str,
    zarr_root: str,
    n_samples: int = 1000,
    seed: int = 0,
    layers: list[str] | None = None,
    n_components: int = 50,
    drop_steps: tuple[int, ...] = (10, 20, 30, 40),
) -> None:
    """Mean-place sweep over a few focus layers and the inspector's actual KNN
    embedding spaces (raw keys / PCA / PCA−Fisher-removed).

    For each layer the space-independent pairing is computed once, then every
    feature space is ranked against it. Prints mean place only (chance level is
    (#opp actions + 1) / 2) — lower is better, 1 = the true twin is the single
    nearest opposite-embodiment action.
    """
    layers = list(layers) if layers else list(DEFAULT_SWEEP_LAYERS)
    # Tight caps: the sweep walks layers strictly in order and never revisits,
    # so we only need ~1 layer of keys/PCA resident at a time.
    store = LayerStore(full_max=2, keys_max=1, pca_max=2, knn_max=1, drop_max=2)
    configs = _sweep_configs(store, run_dir, n_components, drop_steps)

    print(
        f"\nPerfect-pair retrieval sweep — MEAN place of the paired action "
        f"(1 = nearest) among opposite-embodiment actions\n"
        f"run: {run_dir}\n"
        f"samples/layer: {n_samples} | seed: {seed} | PCA components: {n_components}\n"
        f"spaces: {', '.join(label for label, _ in configs)}\n"
        f"(lower mean place is better; chance = (#opp actions + 1) / 2)\n"
    )

    for layer in layers:
        pt_path = os.path.join(run_dir, f"{layer}_keys.pt")
        if not os.path.isfile(pt_path):
            print(f"=== {layer} ===  (missing {layer}_keys.pt — skipped)\n")
            continue
        try:
            data = store.load(run_dir, layer)
        except Exception as e:  # pragma: no cover - defensive
            print(f"=== {layer} ===  (load error: {e})\n")
            continue
        n_rows = len(data["hashes"])

        t_pair = time.time()
        ids, paired_aid_per_row, eligible = _build_pairing(zarr_root, data)
        pair_dt = time.time() - t_pair
        if eligible.size == 0:
            print(f"=== {layer} ===  (no paired actions found — skipped)\n")
            continue

        results = []
        for label, build in configs:
            t0 = time.time()
            try:
                coords = build(layer)
            except Exception as e:
                results.append((label, None, 0, 0.0, time.time() - t0, str(e)))
                continue
            if coords is None:
                results.append(
                    (label, None, 0, 0.0, time.time() - t0, "features unavailable")
                )
                continue
            if coords.shape[0] != n_rows:
                results.append(
                    (
                        label,
                        None,
                        0,
                        0.0,
                        time.time() - t0,
                        f"rows {coords.shape[0]} != {n_rows}",
                    )
                )
                continue
            places, n_opp = _places_for_coords(
                coords, ids, data["embs"], paired_aid_per_row, eligible, n_samples, seed
            )
            if not places:
                results.append(
                    (label, None, 0, 0.0, time.time() - t0, "no paired actions")
                )
                continue
            arr = np.asarray(places)
            results.append(
                (label, float(arr.mean()), len(arr), n_opp, time.time() - t0, None)
            )

        ok = [r for r in results if r[1] is not None]
        n_eval = ok[0][2] if ok else 0
        n_opp = ok[0][3] if ok else 0.0
        chance = (n_opp + 1) / 2 if n_opp else 0.0
        print(
            f"=== {layer} ===  rows={n_rows}  "
            f"n_eval={n_eval}  #opp_actions≈{n_opp:.0f}  chance≈{chance:.1f}  "
            f"[pairing {pair_dt:.1f}s]"
        )
        for label, mean, _ne, _no, dt, err in results:
            if mean is None:
                print(f"  {label:18s} ({err})  [{dt:.1f}s]")
            else:
                print(f"  {label:18s} mean place = {mean:8.2f}   [{dt:.1f}s]")
        print()

    print(
        "place = rank of the true paired action when all opposite-embodiment "
        "actions are sorted by min row distance to the sampled row.\n"
        "raw = full-D keys | pcaC = C-dim PCA | pcaC_dropK = PCA with the top-K "
        "aria-vs-eva Fisher dims removed.\n"
        "Chance level = (#opp actions + 1) / 2 — a mean place well below chance "
        "means the true cross-embodiment twin is retrieved near the top."
    )
