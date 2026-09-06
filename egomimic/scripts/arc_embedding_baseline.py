"""Baseline classic time-based action-chunk embedder (no arc tokenizer).

Same tabbed plotly HTML viewer as ``arc_embedding_sweep.py``, same data
iteration + image caching + click-to-view state-image overlay. The only
difference: each sweep point is a FIXED-FRAME time window of length H, not
an arc-length-parameterized chunk. Feature = ``positions_delta`` over the
first H frames of ``actions_cartesian``.

Usage:
    python -m egomimic.scripts.arc_embedding_baseline \\
        data=human_mecka_eva_rl2_fold_clothes \\
        paths.dataset_dir=/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets
"""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.arc_length_tokenizer import BIMANUAL_CARTESIAN_DIM
from egomimic.scripts.arc_embedding_sweep import (
    _emit_tabbed_html,
    _hdbscan_cluster,
    _iterate_samples_once,
    _print_cluster_stats,
)
from egomimic.utils.aws.aws_data_utils import load_env

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _first_padded_idx(chunk_14d: np.ndarray) -> int:
    """Detect where a chunk stops being real frames and starts being
    ``_pad_sequences`` copies. Padding replicates the last real frame
    verbatim (exact bit-equal copy), so walking backwards from the end
    and finding the first t where ``chunk[t] != chunk[t-1]`` marks the
    boundary. Returns the count of real frames (T if none padded)."""
    T = chunk_14d.shape[0]
    if T < 2:
        return T
    for t in range(T - 1, 0, -1):
        if not np.array_equal(chunk_14d[t], chunk_14d[t - 1]):
            return t + 1
    return 1


def _time_window_feature(
    chunk_14d: np.ndarray, H: int
) -> tuple[np.ndarray, bool, int] | None:
    """Return positions_delta over the first H frames — the classic time-based
    action chunk feature. Returns ``(feat, reached_H, real_frames_in_window)``
    or None if the chunk is too short to fill H frames.

    ``feat`` layout: ``[(L_xyz_t - L_xyz_0), (R_xyz_t - R_xyz_0)]`` flattened
    over H frames -> length H*6, translation-invariant.
    """
    chunk_14d = np.asarray(chunk_14d, dtype=np.float64)
    if chunk_14d.ndim != 2 or chunk_14d.shape[1] != BIMANUAL_CARTESIAN_DIM:
        return None
    T = chunk_14d.shape[0]
    if T < 2:
        return None
    # Count real frames within the H-window (excluding _pad_sequences copies).
    real_frames = min(_first_padded_idx(chunk_14d), int(H))
    end = min(int(H), T)
    reached = end == int(H) and real_frames == int(H)
    seg = chunk_14d[:end]
    left_xyz = seg[:, 0:3] - seg[0:1, 0:3]
    right_xyz = seg[:, 7:10] - seg[0:1, 7:10]
    feat = np.concatenate([left_xyz, right_xyz], axis=-1).reshape(-1)
    # Pad to fixed length so t-SNE gets a rectangular feature matrix.
    target_len = int(H) * 6
    if feat.shape[0] < target_len:
        pad = np.zeros(target_len - feat.shape[0], dtype=np.float64)
        feat = np.concatenate([feat, pad])
    return feat, reached, real_frames


def _run_one_baseline(
    chunks: np.ndarray,
    meta_in: list[dict],
    horizon: int,
    include_zero_tokens: bool,
    tsne_params: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Compute the classic time-window feature for every cached chunk, then
    embed with t-SNE. Returns ``(embed (N, 2), meta list)``."""
    feats: list[np.ndarray] = []
    metas: list[dict] = []
    skipped_short = 0
    skipped_zero = 0
    reached = 0
    for i in range(len(chunks)):
        r = _time_window_feature(chunks[i], horizon)
        if r is None:
            skipped_short += 1
            continue
        feat, reached_H, real_frames = r
        # Zero-motion sample: both arms delta magnitudes all zero.
        is_zero = bool(np.all(np.abs(feat) < 1e-9))
        if is_zero and not include_zero_tokens:
            skipped_zero += 1
            continue
        if reached_H:
            reached += 1
        feats.append(feat)
        rec = dict(meta_in[i])
        rec["is_zero_token"] = is_zero
        rec["reached_target_D"] = reached_H  # reuse meta key so tab UI is uniform
        # End-of-episode partial: fewer than half of H frames are real
        # (the rest are ``_pad_sequences`` copies of the last real frame).
        rec["partial_traj"] = bool(real_frames < 0.5 * int(horizon))
        metas.append(rec)

    if not feats:
        print(f"  H={horizon}: no samples (short={skipped_short} zero={skipped_zero})")
        return np.zeros((0, 2)), [], None

    X = np.stack(feats, axis=0).astype(np.float32)
    print(
        f"  H={horizon}: kept={len(feats)} reached_H={reached}/{len(feats)} "
        f"(short={skipped_short} zero={skipped_zero}) — tsne on X{X.shape}"
    )
    from sklearn.manifold import TSNE

    embed = TSNE(n_components=2, **(tsne_params or {})).fit_transform(X)
    labels, cluster_stats = _hdbscan_cluster(embed)
    for i, rec in enumerate(metas):
        rec["cluster"] = int(labels[i])
    _print_cluster_stats(cluster_stats)
    return embed, metas, cluster_stats


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="arc_embedding_baseline.yaml",
)
def main(cfg: DictConfig) -> None:
    load_env()

    action_key = str(cfg.get("action_key", "actions_cartesian"))
    split = str(cfg.get("split", "train"))
    include_zero_tokens = bool(cfg.include_zero_tokens)
    max_samples = int(cfg.max_samples_per_dataset)
    shuffle = bool(cfg.shuffle)
    seed = int(cfg.seed)

    include_state_image = bool(cfg.include_state_image)
    image_key_cfg = OmegaConf.select(cfg, "image_key", default=None)
    image_key = str(image_key_cfg) if image_key_cfg else None
    image_max_side = int(cfg.image_max_side)
    jpeg_quality = int(cfg.jpeg_quality)

    tsne_params = (
        OmegaConf.to_container(cfg.tsne_params, resolve=True)
        if OmegaConf.select(cfg, "tsne_params", default=None) is not None
        else {}
    )

    splits: list[tuple[str, DictConfig]] = []
    if split in ("train", "both"):
        splits.append(("train", cfg.data.train_datasets))
    if split in ("valid", "both"):
        splits.append(("valid", cfg.data.valid_datasets))
    if not splits:
        raise ValueError(f"split must be train|valid|both, got {split!r}")

    # Baseline uses the EMBODIMENT CLASS DEFAULTS for the read window and
    # interpolation length — NO horizon widening. Concretely:
    #   * Eva: 45 raw frames per action key (Eva.get_keymap → horizon=45),
    #     stride=1, InterpolatePose → 100 output frames.
    #   * Human: 30 raw frames (Human.ACTION_HORIZON), stride=3 → 10
    #     subsampled samples, InterpolatePose → 100 output frames.
    # This is the classic time-window baseline against which the arc-tok
    # sweep gets compared. Widening these here would change what a "45-frame
    # window" means physically vs. what a trained model saw at training
    # time, defeating the point of the comparison.
    print(
        "[baseline] using class-default read windows: eva 45→100 (stride=1), "
        "human 30→100 (stride=3). No horizon widen."
    )
    print("[baseline] iterating datasets once (cache chunks + images)")
    chunks, meta, images, contexts = _iterate_samples_once(
        splits,
        action_key=action_key,
        include_state_image=include_state_image,
        image_key=image_key,
        image_max_side=image_max_side,
        jpeg_quality=jpeg_quality,
        max_samples=max_samples,
        shuffle=shuffle,
        seed=seed,
        # target_horizon omitted -> _iterate_samples_once skips the widen
    )
    print(f"[baseline] cached N={len(chunks)} samples, {len(images)} image URIs")

    if len(chunks) == 0:
        raise RuntimeError(
            "No samples cached — nothing to embed. Check dataset paths + filters."
        )

    horizons = list(cfg.sweep.horizons)
    print(f"[baseline] horizons: {horizons}")

    sweeps: list[tuple[str, np.ndarray, list[dict], dict | None]] = []
    tab_meta: list[dict] = []
    for H in horizons:
        print(f"\n[baseline] H={H}")
        embed, per_meta, cluster_stats = _run_one_baseline(
            chunks,
            meta,
            horizon=int(H),
            include_zero_tokens=include_zero_tokens,
            tsne_params=tsne_params,
        )
        key = f"H={int(H)} frames (classic time window)"
        sweeps.append((key, embed, per_meta, cluster_stats))
        tab_meta.append({"kind": "time", "H": int(H)})

    _emit_tabbed_html(
        sweeps,
        images_shared=images,
        title=str(cfg.get("title", "Baseline (classic time-window)")),
        output_html=str(cfg.get("output_html", "arc_embedding_baseline.html")),
        contexts_shared=contexts,
        tab_meta=tab_meta,
    )


if __name__ == "__main__":
    main()
