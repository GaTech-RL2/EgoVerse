"""Arc tokenizer parameter sweep with a single tabbed plotly HTML output.

Sweeps over ``(joint_distance_cm, waypoints)`` — where joint_distance_cm is the
combined arc length across both arms (per-arm min_distance_unit is set to
``joint_cm / 2 / 100`` m). Each sweep point runs a full t-SNE, and all results
land in one HTML file with a tab bar. Per-point state images (front camera
with green GT overlay via ``<Embodiment>.viz``, same call path as
``hydra_configs/evaluator/viz/{cartesian,cotrain_lang}.yaml``) are deduped
across tabs via a shared JS array so the HTML stays a manageable size.

Usage:
    python -m egomimic.scripts.arc_embedding_sweep \\
        data=aria_eva_fold_clothes_rl2 \\
        paths.dataset_dir=/storage/project/r-dxu345-0/shared/aria_fold \\
        output_html=./out/arc_embedding_sweep.html
"""

from __future__ import annotations

import html as _html
import json
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.arc_length_tokenizer import (
    BIMANUAL_CARTESIAN_DIM,
    INVALID_POSE_THRESHOLD,
)
from egomimic.scripts.arc_embedding_viz import (
    _DEFAULT_IMAGE_CANDIDATES,  # noqa: F401 (kept for parity/reference)
    _extract_overlay_ctx,
    _extract_state_image_uri,
    _log_overlay_error,  # noqa: F401
)
from egomimic.utils.aws.aws_data_utils import load_env

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------------------
# Sample cache (iterated once, reused by every sweep point)
# ---------------------------------------------------------------------------


def _pad_gripper_zeros(chunk: np.ndarray) -> np.ndarray:
    """(T, 12) -> (T, 14) by inserting a zero-gripper column at slots 6 and 13.
    Passes (T, 14) or other shapes through unchanged."""
    if chunk.ndim != 2 or chunk.shape[-1] != 12:
        return chunk
    z = np.zeros((chunk.shape[0], 1), dtype=chunk.dtype)
    return np.concatenate((chunk[:, :6], z, chunk[:, 6:], z), axis=-1)


def _extend_dataset_horizons(ds, target_horizon: int) -> None:
    """Mutate a MultiDataset's leaves so each sample reads ``target_horizon``
    raw frames per action key (instead of the class-default 45 for eva / 30
    for human) and produces a ``target_horizon``-length interpolated chunk.

    Why: the arc-length tokenizer already implements the exact
    "accumulate until joint arc length D or end-of-episode" behavior we want
    (via ``s_end = min(joint_D_m, joint_cum[-1])``). The only limitation is
    the input window it operates on — with the default 45-frame raw window
    (≈1.5 s of eva motion at 30 fps) most chunks never reach D≥60 cm. By
    widening the raw window at read time and letting ``_pad_sequences``
    handle the "past end-of-episode" tail (repeats the last frame, so
    per-frame Δxyz→0 there and ``joint_cum`` plateaus at the true final
    value), we effectively iterate frame-by-frame up to D or episode end
    without duplicating the coordinate-transform pipeline.

    Only cartesian transforms are safe to widen this way — keypoints-mode
    transforms bake ``chunk_length`` into ``Reshape(shape=(chunk_length,
    63))`` steps that would then reject the widened output. This helper
    checks for that and refuses to widen in that case.
    """
    from egomimic.rldb.zarr.action_chunk_transforms import (
        InterpolateLinear,
        InterpolatePose,
        Reshape,
    )

    if not hasattr(ds, "datasets"):
        return
    for leaf in ds.datasets.values():
        # Widen every "horizon" spec in the keymap. Only action-shaped keys
        # carry an explicit horizon; obs poses stay single-frame.
        km = getattr(leaf, "key_map", None)
        if isinstance(km, dict):
            for spec in km.values():
                if isinstance(spec, dict) and "horizon" in spec:
                    spec["horizon"] = int(target_horizon)
        # Also widen every InterpolatePose / InterpolateLinear so the
        # returned chunk has ``target_horizon`` frames (matching the raw
        # window instead of resampling it down to 100).
        tlist = getattr(leaf, "transform", None) or getattr(
            leaf, "transform_list", None
        )
        if tlist is None:
            continue
        for t in tlist:
            if isinstance(t, Reshape):
                # Any Reshape whose target shape starts with the old
                # chunk_length (100) would break when fed the new length.
                shape = getattr(t, "shape", None)
                if isinstance(shape, tuple) and len(shape) >= 1 and shape[0] == 100:
                    raise RuntimeError(
                        "arc_embedding_sweep: cannot widen chunk_length "
                        "safely — transform pipeline has a Reshape hardcoded "
                        f"to length {shape}. Use a non-keypoints mode."
                    )
            if isinstance(t, (InterpolatePose, InterpolateLinear)):
                t.new_chunk_length = int(target_horizon)


def _iterate_samples_once(
    datasets_by_split,
    action_key: str,
    include_state_image: bool,
    image_key: str | None,
    image_max_side: int,
    jpeg_quality: int,
    max_samples: int,
    shuffle: bool,
    seed: int,
    target_horizon: int = 0,
) -> tuple[np.ndarray, list[dict], list[str], list[dict]]:
    """Iterate every dataset once and cache per-sample data:
    chunks (N, T, 14) float64,
    meta list of length N with keys {embodiment, split, sample_idx, img_idx},
    images list of length N — RAW ``data:image/jpeg;base64`` URIs (no
      overlay drawn server-side; JS draws per-tab overlay on click),
    contexts list of length N — per-sample ``{actions, T, D, K, img_h,
      img_w}`` dicts consumed by the client-side overlay renderer.
    """
    chunks_list: list[np.ndarray] = []
    meta_list: list[dict] = []
    images: list[str] = []
    contexts: list[dict | None] = []

    for split_name, ds_cfgs in datasets_by_split:
        for emb_name, ds_cfg in ds_cfgs.items():
            print(f"\n[{split_name}] instantiating {emb_name}")
            ds = hydra.utils.instantiate(ds_cfg)
            if target_horizon and target_horizon > 0:
                try:
                    _extend_dataset_horizons(ds, int(target_horizon))
                    print(
                        f"  [{split_name}] {emb_name}: extended raw horizon "
                        f"→ {int(target_horizon)} frames (frame-by-frame arc "
                        "accumulation until D or end-of-episode)"
                    )
                except RuntimeError as e:
                    print(f"  [{split_name}] {emb_name}: horizon-extend skipped: {e}")
            n = len(ds)
            if n == 0:
                print(f"  {emb_name}: empty, skipping")
                continue
            idxs = list(range(n))
            if shuffle:
                random.Random(seed).shuffle(idxs)
            if max_samples > 0:
                idxs = idxs[:max_samples]

            kept = 0
            skipped_shape = 0
            skipped_invalid = 0
            skipped_error = 0
            # Per-episode raw embodiment string (mecka_bimanual, aria_bimanual,
            # scale_bimanual, ...) drives the intrinsics fallback so overlay
            # projection uses the right K per vendor camera. index_map is a
            # list of (episode_hash, local_idx); leaf .embodiment holds the
            # raw pre-collapse string.
            for idx in idxs:
                try:
                    sample = ds[idx]
                except Exception as e:
                    skipped_error += 1
                    if skipped_error <= 3:
                        print(
                            f"  [{split_name}] {emb_name} idx={idx} "
                            f"sample failed: {type(e).__name__}: {e}"
                        )
                    continue
                if action_key not in sample:
                    skipped_shape += 1
                    continue
                chunk = sample[action_key]
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.detach().cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float64)
                if chunk.ndim != 2:
                    skipped_shape += 1
                    continue
                chunk = _pad_gripper_zeros(chunk)
                if chunk.shape[-1] != BIMANUAL_CARTESIAN_DIM:
                    skipped_shape += 1
                    continue
                if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
                    skipped_invalid += 1
                    continue

                # Look up the leaf that produced this sample to grab the raw
                # (vendor-tagged) embodiment string.
                vendor_hint = None
                try:
                    ep_name, _ = ds.index_map[idx]
                    vendor_hint = getattr(ds.datasets[ep_name], "embodiment", None)
                except Exception:
                    pass

                # Original image dimensions (before JPEG resize) — used to
                # scale K so JS projections land in the shipped image's
                # pixel coordinates. We MUST measure the frame at index 0
                # here so H, W come from the same timestep as chunk[0]; a
                # 4D stack unhandled would leak T into H and produce a
                # wildly wrong scale factor.
                orig_hw = None
                for _cand_key in (
                    image_key or "",
                    "observations.images.front_img_1",
                    "observations.images.front_1",
                ):
                    if _cand_key and _cand_key in sample:
                        _im = sample[_cand_key]
                        _im = (
                            _im.detach().cpu().numpy()
                            if hasattr(_im, "detach")
                            else np.asarray(_im)
                        )
                        # 4D = horizoned image (T, ...); align with chunk[0]
                        # by taking the first frame before shape inspection.
                        if _im.ndim == 4:
                            _im = _im[0]
                        # Handle CHW / HWC
                        if _im.ndim == 3 and _im.shape[0] in (1, 3):
                            orig_hw = (int(_im.shape[1]), int(_im.shape[2]))
                        elif _im.ndim == 3:
                            orig_hw = (int(_im.shape[0]), int(_im.shape[1]))
                        break

                if include_state_image:
                    # RAW image — no server-side overlay (JS draws per-tab).
                    uri = (
                        _extract_state_image_uri(
                            sample,
                            chunk_14d=chunk,
                            embodiment_name=str(emb_name),
                            image_key=image_key,
                            image_max_side=image_max_side,
                            jpeg_quality=jpeg_quality,
                            overlay_gt_chunk=False,
                            vendor_hint=vendor_hint,
                        )
                        or ""
                    )
                    ctx = _extract_overlay_ctx(
                        sample,
                        chunk_14d=chunk,
                        embodiment_name=str(emb_name),
                        vendor_hint=vendor_hint,
                        image_max_side=image_max_side,
                        orig_img_hw=orig_hw,
                    )
                else:
                    uri = ""
                    ctx = None

                img_idx = len(images)
                images.append(uri)
                contexts.append(ctx)
                chunks_list.append(chunk)
                meta_list.append(
                    {
                        "embodiment": str(emb_name),
                        "split": split_name,
                        "sample_idx": int(idx),
                        "img_idx": img_idx,
                    }
                )
                kept += 1
            print(
                f"  [{split_name}] {emb_name}: kept={kept}/{len(idxs)} "
                f"(shape={skipped_shape} invalid={skipped_invalid} err={skipped_error})"
            )

    if not chunks_list:
        return np.zeros((0, 0, BIMANUAL_CARTESIAN_DIM)), [], [], []
    chunks_arr = np.stack(chunks_list, axis=0)
    return chunks_arr, meta_list, images, contexts


# ---------------------------------------------------------------------------
# Per-sweep tokenize + t-SNE
# ---------------------------------------------------------------------------


def _joint_distance_tokenize(
    chunk_14d: np.ndarray,
    joint_D_m: float,
    M: int,
    zero_eps: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """Sample M waypoints of (left_xyz, right_xyz) uniform in JOINT arc length
    across the first stretch of the chunk where left+right combined arc length
    reaches ``joint_D_m``.

    Steps
      1. Per-arm cumulative arc lengths ``cum_L(t) = Σ ||Δleft_xyz||``,
         ``cum_R(t) = Σ ||Δright_xyz||`` (both start at 0, evaluated over the
         time-indexed chunk).
      2. Joint cumulative ``J(t) = cum_L(t) + cum_R(t)``.
      3. Effective boundary ``s_end = min(joint_D_m, J[-1])``.
      4. Target joint arc-lengths ``s_k = linspace(0, s_end, M)``.
      5. Fractional timesteps ``t_k = interp(s_k, J, [0..T-1])`` — for each
         target joint distance, invert J to find the moment in the chunk when
         that much combined motion has been accumulated.
      6. Interpolate each arm's xyz linearly at those ``t_k`` values.

    Returns:
      ``(left_M (M, 3), right_M (M, 3), J_max, duration_frames)`` — or
      ``(None, None, 0, 0)`` if the chunk is too short or its joint arc
      length is below ``zero_eps``. ``duration_frames`` is the fractional
      timestep span from t_k[0] to t_k[-1]; multiply by ``dt`` (seconds/step)
      to get chunk duration in seconds — used for the MEAN_SCALAR velocity
      channel that matches BimanualArcLengthTokenizer's default output.
    """
    T = chunk_14d.shape[0]
    if T < 2:
        return None, None, 0.0, 0.0
    left_xyz = np.asarray(chunk_14d[:, 0:3], dtype=np.float64)
    right_xyz = np.asarray(chunk_14d[:, 7:10], dtype=np.float64)
    left_step = np.linalg.norm(np.diff(left_xyz, axis=0), axis=1)
    right_step = np.linalg.norm(np.diff(right_xyz, axis=0), axis=1)
    cum_L = np.concatenate([[0.0], np.cumsum(left_step)])  # (T,)
    cum_R = np.concatenate([[0.0], np.cumsum(right_step)])
    joint_cum = cum_L + cum_R  # (T,) monotonically non-decreasing
    J_max = float(joint_cum[-1])
    if J_max < zero_eps:
        return None, None, J_max, 0.0
    s_end = min(float(joint_D_m), J_max)
    targets = np.linspace(0.0, s_end, M)
    t_grid = np.arange(T, dtype=np.float64)
    # np.interp requires monotonically-increasing xp; joint_cum is non-decreasing
    # but can have flat spans (both arms stationary). np.interp handles that by
    # returning the left-most matching x — good enough.
    t_k = np.interp(targets, joint_cum, t_grid)
    left_M = np.stack(
        [np.interp(t_k, t_grid, left_xyz[:, d]) for d in range(3)], axis=-1
    )
    right_M = np.stack(
        [np.interp(t_k, t_grid, right_xyz[:, d]) for d in range(3)], axis=-1
    )
    duration_frames = float(t_k[-1] - t_k[0])
    return left_M, right_M, J_max, duration_frames


def _joint_positions_delta_feature(
    left_M: np.ndarray, right_M: np.ndarray
) -> np.ndarray:
    """Translation-invariant per-arm position feature. (M*6,) vector."""
    left = left_M - left_M[0:1]
    right = right_M - right_M[0:1]
    return np.concatenate([left, right], axis=-1).reshape(-1)


def _joint_positions_delta_vel_feature(
    left_M: np.ndarray,
    right_M: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Positions-delta + MEAN_SCALAR velocity per arm, laid out to match the
    base BimanualArcLengthTokenizer's default token structure — per arm block
    is ``[xyz_delta(3), vel(1)]`` broadcast across all M waypoints, then arms
    concatenated. Flatten to a 1-D feature.

    Per arm ``vel = ||xyz[-1] - xyz[0]|| / duration_s``. Broadcasting the
    scalar across all M rows mirrors ``_broadcast_velocity`` — makes the
    velocity contribute ``M`` copies of one number to the flat feature.

    Total feature length: ``M * 8`` (per waypoint: L xyz-delta + L vel +
    R xyz-delta + R vel).
    """
    dur = max(float(duration_s), 1e-8)
    left = left_M - left_M[0:1]
    right = right_M - right_M[0:1]
    left_vel = float(np.linalg.norm(left_M[-1] - left_M[0])) / dur
    right_vel = float(np.linalg.norm(right_M[-1] - right_M[0])) / dur
    M = left.shape[0]
    left_vel_col = np.full((M, 1), left_vel, dtype=np.float64)
    right_vel_col = np.full((M, 1), right_vel, dtype=np.float64)
    return np.concatenate([left, left_vel_col, right, right_vel_col], axis=-1).reshape(
        -1
    )


def _joint_positions_delta_vel_append_feature(
    left_M: np.ndarray,
    right_M: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Positions-delta with a mean per-dim velocity appended ONCE per arm at
    the end. Velocity is the mean (per-axis) rate over the whole chunk:
    ``(xyz[-1] - xyz[0]) / duration_s`` — 3 dims per arm, so a total of 6
    velocity values (``[Lx_vel, Ly_vel, Lz_vel, Rx_vel, Ry_vel, Rz_vel]``)
    tacked onto the end of the positions-delta block.

    Feature length: ``M * 6 + 6`` — 6 dims per waypoint (positions-delta L+R)
    followed by the 6-dim per-dim mean velocity.

    Rationale for per-dim over scalar magnitude: two arms that trace the
    same-length trajectories but in different directions (e.g. one arm
    reaches forward, the other reaches down) look identical under a scalar
    magnitude channel and different under per-dim. Per-dim preserves
    direction info at only 4 extra feature dims vs. the old scalar version.
    """
    dur = max(float(duration_s), 1e-8)
    left = left_M - left_M[0:1]
    right = right_M - right_M[0:1]
    left_vel = (left_M[-1] - left_M[0]) / dur  # (3,) per-axis mean vel
    right_vel = (right_M[-1] - right_M[0]) / dur  # (3,)
    pos = np.concatenate([left, right], axis=-1).reshape(-1)
    return np.concatenate([pos, left_vel, right_vel], axis=0)


def _run_one_sweep(
    chunks: np.ndarray,
    meta_in: list[dict],
    joint_distance_cm: float,
    waypoints: int,
    tokenizer_cfg: DictConfig,
    feature_mode: str,
    include_zero_tokens: bool,
    tsne_params: dict,
) -> tuple[np.ndarray, list[dict], "None"]:
    """Tokenize every cached chunk under (joint_distance_cm, waypoints) using
    the joint-distance semantics (chunk covers the first stretch where
    left+right combined arc length reaches D), then embed with t-SNE."""
    joint_D_m = float(joint_distance_cm) / 100.0
    M = int(waypoints)
    zero_eps = float(OmegaConf.select(tokenizer_cfg, "zero_dist_epsilon", default=1e-6))
    # dt = seconds per source-trajectory time step (control period). Needed
    # to translate the joint-tokenizer's fractional-timestep duration into
    # seconds for the MEAN_SCALAR velocity channel.
    dt = float(OmegaConf.select(tokenizer_cfg, "dt", default=1.0 / 30.0))

    feats: list[np.ndarray] = []
    metas: list[dict] = []
    skipped_short = 0
    skipped_zero = 0
    reached_D = 0
    for i in range(len(chunks)):
        chunk = chunks[i]
        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            continue
        left_M, right_M, J_max, duration_frames = _joint_distance_tokenize(
            chunk, joint_D_m, M, zero_eps
        )
        if left_M is None:
            skipped_short += 1
            continue
        # A stretch below the requested D is still a valid token (short-window
        # sample), but a completely stationary chunk is dropped unless
        # include_zero_tokens.
        is_zero = J_max < zero_eps
        if is_zero and not include_zero_tokens:
            skipped_zero += 1
            continue
        if J_max >= joint_D_m:
            reached_D += 1

        duration_s = duration_frames * dt
        if feature_mode == "positions_delta":
            # Default (velappend): positions_delta + per-dim mean velocity
            # appended ONCE per arm at the end (M*6 + 6). Velocity is a
            # 3-vector per arm (mean per-axis rate over the chunk), so it
            # contributes 6 extra dims instead of ``2*M`` broadcast copies —
            # keeps direction info without drowning positions in distance.
            feat = _joint_positions_delta_vel_append_feature(
                left_M, right_M, duration_s
            )
        elif feature_mode == "positions_delta_vel_broadcast":
            # Broadcast-per-waypoint layout matching the base tokenizer's
            # default arc-token shape (per-arm [xyz(3), vel(1)] × M).
            feat = _joint_positions_delta_vel_feature(left_M, right_M, duration_s)
        elif feature_mode == "positions_delta_only":
            feat = _joint_positions_delta_feature(left_M, right_M)
        elif feature_mode == "positions":
            feat = np.concatenate([left_M, right_M], axis=-1).reshape(-1)
        elif feature_mode == "flatten":
            feat = np.concatenate([left_M, right_M], axis=-1).reshape(-1)
        else:
            raise ValueError(
                f"Unknown feature '{feature_mode}' for joint-distance sweep "
                "(pick 'positions_delta', 'positions_delta_only', "
                "'positions_delta_vel_broadcast', or 'positions')"
            )
        feats.append(feat)
        rec = dict(meta_in[i])
        rec["is_zero_token"] = bool(is_zero)
        rec["joint_arc_max_cm"] = round(J_max * 100.0, 2)
        rec["reached_target_D"] = bool(J_max >= joint_D_m)
        # End-of-episode partial trajectory flag: episode ended before
        # accumulating half the requested joint arc length. These samples
        # cover so little of the target motion that they mostly cluster
        # by "where the trajectory got cut off" rather than by actual
        # motion pattern — worth hiding by default (see UI toggle).
        rec["partial_traj"] = bool(J_max < 0.5 * joint_D_m)
        metas.append(rec)

    if not feats:
        print(
            f"  D={joint_distance_cm}cm W={waypoints}: no samples "
            f"(short={skipped_short} zero={skipped_zero}) — skipping"
        )
        return np.zeros((0, 2)), [], None

    X = np.stack(feats, axis=0).astype(np.float32)
    print(
        f"  D={joint_distance_cm}cm W={waypoints}: kept={len(feats)} "
        f"reached_D={reached_D}/{len(feats)} "
        f"(short={skipped_short} zero={skipped_zero}) — tsne on X{X.shape}"
    )
    from sklearn.manifold import TSNE

    embed = TSNE(n_components=2, **(tsne_params or {})).fit_transform(X)
    # HDBSCAN on the 2-D embedding — attach cluster label + one-shot stats.
    labels, cluster_stats = _hdbscan_cluster(embed)
    for i, rec in enumerate(metas):
        rec["cluster"] = int(labels[i])
    _print_cluster_stats(cluster_stats)
    return embed, metas, cluster_stats


def _print_cluster_stats(stats: dict) -> None:
    """One-line-ish console dump of the HDBSCAN outputs — used by both the
    sweep and baseline pipelines."""
    n = stats.get("n_clusters", 0)
    noise = stats.get("sizes", {}).get(-1, 0)
    sizes = stats.get("sizes", {})
    pers = stats.get("persistence", {})
    knn = stats.get("median_knn", {})
    print(
        f"    hdbscan: {n} clusters, noise={noise}, k={stats.get('min_cluster_size', '?')}"
    )
    for cid in sorted([c for c in sizes if c >= 0]):
        p = pers.get(cid, float("nan"))
        k = knn.get(cid, float("nan"))
        print(f"      c{cid}: n={sizes[cid]}  persistence={p:.3f}  medKNN={k:.3f}")


# ---------------------------------------------------------------------------
# HDBSCAN clustering + dual color arrays (embodiment vs cluster)
# ---------------------------------------------------------------------------


_EMBODIMENT_PALETTE = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]


def _embodiment_color(embodiment_idx: int) -> str:
    return _EMBODIMENT_PALETTE[embodiment_idx % len(_EMBODIMENT_PALETTE)]


def _cluster_shade_color(
    cluster_id: int, embodiment_idx: int, n_embodiments: int
) -> str:
    """Distinct hue per cluster (golden-angle spacing so neighbouring cluster
    ids don't share visually-similar hues), then per-embodiment shade of that
    hue so you can still tell embodiments apart when coloring by cluster.
    Cluster -1 (HDBSCAN noise) renders as neutral grey.
    """
    import colorsys

    if cluster_id < 0:
        # Noise — still shade by embodiment so it's readable.
        gray_v = 0.65 - 0.20 * (embodiment_idx / max(n_embodiments - 1, 1))
        r, g, b = gray_v, gray_v, gray_v
    else:
        hue = (cluster_id * 0.6180339887498949) % 1.0  # golden-angle wheel
        if n_embodiments <= 1:
            sat, val = 0.75, 0.85
        else:
            f = embodiment_idx / max(n_embodiments - 1, 1)  # 0..1
            # 0 -> light/desaturated, 1 -> deep/saturated. Two embodiments
            # come out as pastel-vs-vivid same-hue pair.
            sat = 0.45 + 0.50 * f
            val = 0.92 - 0.22 * f
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _hdbscan_cluster(embed_2d: np.ndarray) -> tuple[np.ndarray, dict]:
    """Cluster the 2-D t-SNE embedding with HDBSCAN. Returns per-point cluster
    labels (int array of length N; -1 = noise) plus stats dict with:

      - ``sizes``      : {cluster_id: n_points}
      - ``persistence``: {cluster_id: float}
          From ``clusterer.cluster_persistence_`` — HDBSCAN's own density-
          persistence score for each cluster. Unitless in [0, ~1]; higher =
          the cluster survives more of the density hierarchy (more strongly
          separated from background noise).
      - ``median_knn`` : {cluster_id: float}
          Median distance from each point in the cluster to its k-th nearest
          intra-cluster neighbor, where k = min_cluster_size (the same
          parameter HDBSCAN used to form the cluster). Robust density proxy
          in t-SNE units — smaller = tighter cluster.
      - ``n_clusters`` : count of non-noise clusters
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "HDBSCAN clustering requires hdbscan. Install with "
            "`pip install hdbscan`."
        ) from e
    from sklearn.neighbors import NearestNeighbors

    N = len(embed_2d)
    if N < 5:
        return (
            np.full(N, -1, dtype=np.int32),
            {"n_clusters": 0, "sizes": {}, "persistence": {}, "median_knn": {}},
        )

    # min_cluster_size scaled with N so the sweep grid gives comparable
    # cluster counts across (D, W) points that have similar N.
    mcs = max(10, int(N * 0.01))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=None,
        prediction_data=False,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(embed_2d).astype(np.int32)

    # cluster_persistence_ is indexed by cluster label 0..K-1.
    persistence = np.asarray(
        getattr(clusterer, "cluster_persistence_", []), dtype=float
    )

    stats: dict = {
        "sizes": {},
        "persistence": {},
        "median_knn": {},
        "min_cluster_size": int(mcs),
    }
    for c in np.unique(labels):
        pts = embed_2d[labels == c]
        n = int(len(pts))
        cid = int(c)
        stats["sizes"][cid] = n
        if cid < 0:
            continue
        # Persistence from HDBSCAN (matches the algorithm's own density
        # hierarchy — best "how prominent is this cluster" signal available
        # without recomputing).
        if cid < len(persistence):
            stats["persistence"][cid] = round(float(persistence[cid]), 4)
        # Median k-NN distance within this cluster, k = min_cluster_size.
        # Skip if the cluster is smaller than k+1 (can't ask for k neighbors
        # excluding self).
        k = mcs
        if n > k:
            nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
            d, _ = nn.kneighbors(pts)
            # d[:, 0] is distance to self (0); d[:, k] is the k-th neighbor.
            stats["median_knn"][cid] = round(float(np.median(d[:, k])), 4)
    stats["n_clusters"] = int(np.sum(np.asarray(sorted(stats["sizes"].keys())) >= 0))
    return labels, stats


def _build_color_arrays(
    meta: list[dict],
    embodiment_to_idx: dict[str, int],
    n_embodiments: int,
) -> tuple[list[str], list[str]]:
    """Precompute both color arrays. by_embodiment: standard palette.
    by_cluster: cluster hue + embodiment shade."""
    emb_colors: list[str] = []
    cluster_colors: list[str] = []
    for m in meta:
        eidx = embodiment_to_idx.get(m["embodiment"], 0)
        cid = int(m.get("cluster", -1))
        emb_colors.append(_embodiment_color(eidx))
        cluster_colors.append(_cluster_shade_color(cid, eidx, n_embodiments))
    return emb_colors, cluster_colors


# ---------------------------------------------------------------------------
# HTML emission (tabbed)
# ---------------------------------------------------------------------------


def _sweep_tab_figure(
    sweep_key: str,
    embed: np.ndarray,
    meta: list[dict],
    title_prefix: str,
    cluster_stats: dict | None = None,
) -> tuple["object | None", int, list[str], list[str]]:
    """Build the plotly figure for one tab as TWO scattergl traces —
    trace 0 = full-D samples (``partial_traj != True``),
    trace 1 = partial-D samples (episode ended before covering >=50% of D).
    Splitting like this lets the "hide partial" UI toggle work as a single
    ``Plotly.restyle(..., {visible}, [1])`` call rather than mutating point
    arrays. Returns ``(fig, n_points, emb_colors, cluster_colors)`` — the
    two color arrays are per-original-sample (before the split) and are
    stored client-side so the color-mode radio can restyle without a
    re-render. Cluster stats are baked into the plot title.
    """
    import plotly.graph_objects as go

    if not meta or len(meta) == 0 or embed.shape[0] == 0:
        return None, 0, [], []

    # Stable embodiment -> index mapping (order = first-seen), used by both
    # color arrays so a given embodiment gets a consistent slot / palette
    # index across tabs.
    seen: list[str] = []
    for m in meta:
        if m["embodiment"] not in seen:
            seen.append(m["embodiment"])
    emb_to_idx = {name: i for i, name in enumerate(seen)}
    n_embs = max(1, len(seen))

    emb_colors, cluster_colors = _build_color_arrays(meta, emb_to_idx, n_embs)

    n = len(meta)
    xs_all = embed[:, 0].tolist()
    ys_all = embed[:, 1].tolist()
    custom_data_all = [
        [
            m["img_idx"],
            m["embodiment"],
            m["sample_idx"],
            m.get("split", ""),
            bool(m.get("is_zero_token", False)),
            int(m.get("cluster", -1)),
            bool(m.get("partial_traj", False)),
            float(m.get("joint_arc_max_cm", 0.0)),
        ]
        for m in meta
    ]

    # Split into full vs partial indices (into the original arrays).
    full_idx = [i for i, m in enumerate(meta) if not m.get("partial_traj", False)]
    partial_idx = [i for i, m in enumerate(meta) if m.get("partial_traj", False)]

    def _pick(arr, idxs):
        return [arr[i] for i in idxs]

    hover_template = (
        "%{customdata[1]} idx=%{customdata[2]}"
        "<br>split=%{customdata[3]}"
        "<br>cluster=%{customdata[5]}"
        "<br>zero=%{customdata[4]}"
        "<br>joint_arc=%{customdata[7]:.1f}cm"
        "<br>partial=%{customdata[6]}"
        "<extra></extra>"
    )

    # Cluster stats -> plot title suffix.
    if cluster_stats and cluster_stats.get("n_clusters"):
        sizes = cluster_stats.get("sizes", {})
        noise = int(sizes.get(-1, 0))
        sig_sizes = sorted(
            [(c, s) for c, s in sizes.items() if c >= 0], key=lambda p: -p[1]
        )
        top = ", ".join(f"c{c}:{s}" for c, s in sig_sizes[:6])
        if len(sig_sizes) > 6:
            top += f", +{len(sig_sizes) - 6} more"
        cluster_suffix = (
            f"  |  {cluster_stats['n_clusters']} clusters  noise={noise}  ({top})"
        )
    else:
        cluster_suffix = "  |  0 clusters"

    label = f"{title_prefix} — {sweep_key}  (N={n}){cluster_suffix}"

    fig = go.Figure(
        [
            go.Scattergl(
                x=_pick(xs_all, full_idx),
                y=_pick(ys_all, full_idx),
                mode="markers",
                marker=dict(
                    size=5,
                    color=_pick(emb_colors, full_idx),
                    opacity=0.75,
                ),
                customdata=_pick(custom_data_all, full_idx),
                hovertemplate=hover_template,
                showlegend=False,
                name="full",
            ),
            go.Scattergl(
                x=_pick(xs_all, partial_idx),
                y=_pick(ys_all, partial_idx),
                mode="markers",
                marker=dict(
                    size=5,
                    color=_pick(emb_colors, partial_idx),
                    opacity=0.55,
                    line=dict(width=0.6, color="#ff8"),
                ),
                customdata=_pick(custom_data_all, partial_idx),
                hovertemplate=hover_template,
                showlegend=False,
                name="partial",
            ),
        ]
    )
    # Legend is per-point colors so plotly's default legend isn't useful; we
    # provide a compact color legend via the tab-bar radio + a per-plot info
    # overlay (rendered by the HTML JS).
    fig.update_layout(
        title=label,
        autosize=True,
        margin=dict(l=40, r=20, t=48, b=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#eee"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#333", zerolinecolor="#333")
    fig.update_yaxes(gridcolor="#333", zerolinecolor="#333")
    # Split color arrays for the two traces (full=0, partial=1) so the
    # color-mode radio can restyle both traces in one call.
    full_emb = _pick(emb_colors, full_idx)
    full_cluster = _pick(cluster_colors, full_idx)
    partial_emb = _pick(emb_colors, partial_idx)
    partial_cluster = _pick(cluster_colors, partial_idx)
    return (
        fig,
        n,
        (full_emb, partial_emb),
        (full_cluster, partial_cluster),
    )


def _compute_knn_grid_for_tab(
    embed: np.ndarray,
    meta: list[dict],
    k: int = 4,
) -> list[dict]:
    """Per-tab cross-embodiment KNN grid.

    For each HDBSCAN cluster with points from BOTH embodiments, pick an
    anchor per embodiment (that embodiment's medoid within the cluster —
    the point closest to the embodiment-cluster centroid in TSNE space).
    For each anchor, find the ``k`` nearest points in the OPPOSITE
    embodiment across the whole dataset (not just this cluster), keyed by
    the same 2-D TSNE coordinates.

    Returns a list of cluster records ready to ship to JS:
      [
        {
          "cluster_id": int,
          "n_points": int,           # total points in this cluster
          "embodiments": [emb_a, emb_b],
          "anchors": {
            "<emb>": {
              "point_idx": int,      # index into meta / embed
              "img_idx": int,        # shared-images array index
              "sample_idx": int,
              "coords": [x, y],
            },
            ...
          },
          "knn": {
            "<emb_from>_to_<emb_to>": [
              {"point_idx": ..., "img_idx": ..., "sample_idx": ..., "dist": ...},
              ...k neighbors
            ],
            ...
          },
        },
        ...
      ]

    Skips: noise cluster (id=-1) and any cluster missing a whole
    embodiment (can't compute a cross-embodiment anchor pair).
    """
    if len(meta) == 0 or embed.size == 0:
        return []
    embs = []
    for m in meta:
        e = m.get("embodiment")
        if e not in embs:
            embs.append(e)
    embs.sort()
    if len(embs) != 2:
        # KNN cross-lookup only defined for exactly two embodiments.
        return []
    emb_a, emb_b = embs[0], embs[1]

    idx_by_emb: dict[str, list[int]] = {emb_a: [], emb_b: []}
    for i, m in enumerate(meta):
        e = m.get("embodiment")
        if e in idx_by_emb:
            idx_by_emb[e].append(i)
    idx_by_emb_arr = {e: np.asarray(v, dtype=np.int64) for e, v in idx_by_emb.items()}

    cluster_to_indices: dict[int, list[int]] = {}
    for i, m in enumerate(meta):
        cid = int(m.get("cluster", -1))
        if cid < 0:
            continue
        cluster_to_indices.setdefault(cid, []).append(i)

    grid_out: list[dict] = []
    for cid in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cid]
        anchors: dict[str, dict] = {}
        for e in (emb_a, emb_b):
            emb_in_cluster = [i for i in indices if meta[i].get("embodiment") == e]
            if not emb_in_cluster:
                continue
            coords = embed[emb_in_cluster]
            centroid = coords.mean(axis=0)
            dists_to_centroid = np.linalg.norm(coords - centroid, axis=1)
            anchor_i = int(emb_in_cluster[int(np.argmin(dists_to_centroid))])
            anchors[e] = {
                "point_idx": anchor_i,
                "img_idx": int(meta[anchor_i].get("img_idx", -1)),
                "sample_idx": int(meta[anchor_i].get("sample_idx", -1)),
                "coords": [float(embed[anchor_i, 0]), float(embed[anchor_i, 1])],
            }
        if len(anchors) < 2:
            continue  # need both embodiments represented for the cross lookup

        knn: dict[str, list[dict]] = {}
        for e_from, e_to in ((emb_a, emb_b), (emb_b, emb_a)):
            anchor_coord = np.asarray(anchors[e_from]["coords"], dtype=np.float64)
            other_idx = idx_by_emb_arr[e_to]
            other_coords = embed[other_idx]
            dists = np.linalg.norm(other_coords - anchor_coord, axis=1)
            top_k = np.argsort(dists)[: max(1, int(k))]
            neighbors: list[dict] = []
            for j in top_k:
                orig_i = int(other_idx[j])
                neighbors.append(
                    {
                        "point_idx": orig_i,
                        "img_idx": int(meta[orig_i].get("img_idx", -1)),
                        "sample_idx": int(meta[orig_i].get("sample_idx", -1)),
                        "dist": float(dists[j]),
                    }
                )
            knn[f"{e_from}_to_{e_to}"] = neighbors

        grid_out.append(
            {
                "cluster_id": int(cid),
                "n_points": len(indices),
                "embodiments": [emb_a, emb_b],
                "anchors": anchors,
                "knn": knn,
            }
        )

    return grid_out


def _emit_tabbed_html(
    sweeps: list[tuple[str, np.ndarray, list[dict]]],
    images_shared: list[str],
    title: str,
    output_html: str,
    contexts_shared: list[dict | None] | None = None,
    tab_meta: list[dict] | None = None,
    overlay_pre_interp: bool = False,
) -> None:
    """Assemble the final HTML: shared plotly.js from CDN, one plotly div per
    sweep, tab bar toggles visibility, shared image array + click handler.
    """
    import plotly.io as pio

    tabs = []
    panels = []
    plot_payloads: dict[str, str] = {}  # div_id -> JSON string of {data,layout}
    color_arrays: dict[str, tuple[list[str], list[str]]] = {}
    per_tab_cluster_stats: dict[str, dict] = {}
    per_tab_knn_grid: dict[str, list[dict]] = {}
    for i, sweep_entry in enumerate(sweeps):
        # Backwards-compatible unpack: sweep entries may be 3-tuples (key,
        # embed, meta) or 4-tuples (key, embed, meta, cluster_stats).
        if len(sweep_entry) == 4:
            key, embed, meta, cluster_stats = sweep_entry
        else:
            key, embed, meta = sweep_entry
            cluster_stats = None
        div_id = f"egoverse-plot-{i}"
        panel_id = f"egoverse-panel-{i}"
        active_cls = " active-panel" if i == 0 else ""
        tabs.append(
            f'<button class="egoverse-tab" data-target="{panel_id}" '
            f'data-index="{i}">{_html.escape(key)}</button>'
        )
        fig, n_pts, emb_colors, cluster_colors = _sweep_tab_figure(
            key, embed, meta, title, cluster_stats=cluster_stats
        )
        if fig is None:
            panels.append(
                f'<div id="{panel_id}" class="egoverse-panel{active_cls}">'
                f'<div style="padding:12px;color:#888;"><em>empty sweep '
                f"(no samples kept for this D/W)</em></div></div>"
            )
            continue
        # Panel has TWO views: the plotly TSNE plot (default) and a KNN
        # grid (populated lazily on first switch). A single top-level view
        # toggle flips display between them without touching plot state.
        knn_div_id = f"egoverse-knn-{i}"
        panels.append(
            f'<div id="{panel_id}" class="egoverse-panel{active_cls}">'
            f'<div id="{div_id}" class="plotly-graph-div egoverse-view-plot" '
            f'style="width:100%;height:100%;"></div>'
            f'<div id="{knn_div_id}" class="egoverse-view-knn egoverse-knn-grid" '
            f'style="display:none;"></div>'
            f"</div>"
        )
        plot_payloads[div_id] = pio.to_json(fig)
        color_arrays[div_id] = (emb_colors, cluster_colors)
        # Cross-embodiment KNN grid: for every HDBSCAN cluster with points
        # in both embodiments, ship one medoid per embodiment + its 4
        # nearest opposite-embodiment neighbors. Rendered client-side when
        # the user flips to KNN view.
        per_tab_knn_grid[div_id] = _compute_knn_grid_for_tab(embed, meta, k=4)
        # Stash cluster stats (int-keyed sizes/densities dicts) for the
        # left-side stats bar to consume. Convert dict keys to strings so
        # they survive JSON serialization.
        stats_for_js = None
        if cluster_stats:
            stats_for_js = {
                "n_clusters": int(cluster_stats.get("n_clusters", 0)),
                "sizes": {
                    str(k): int(v) for k, v in cluster_stats.get("sizes", {}).items()
                },
                "persistence": {
                    str(k): float(v)
                    for k, v in cluster_stats.get("persistence", {}).items()
                },
                "median_knn": {
                    str(k): float(v)
                    for k, v in cluster_stats.get("median_knn", {}).items()
                },
                "min_cluster_size": int(cluster_stats.get("min_cluster_size", 0)),
                "label": key,
            }
        per_tab_cluster_stats[div_id] = stats_for_js or {
            "n_clusters": 0,
            "sizes": {},
            "persistence": {},
            "median_knn": {},
            "min_cluster_size": 0,
            "label": key,
        }

    # Shared images. JSON-embed as a JS var so all tabs reference the same
    # blob (avoids repeating each URI once per tab).
    images_json = json.dumps(images_shared)
    # Plot payloads: assembled as an object literal by concatenating the raw
    # JSON strings (avoids re-encoding numpy arrays plotly already handled).
    if plot_payloads:
        payloads_js = (
            "{"
            + ",".join(
                f'"{div_id}":{payload_json}'
                for div_id, payload_json in plot_payloads.items()
            )
            + "}"
        )
    else:
        payloads_js = "{}"
    # Color arrays keyed by plot div_id — each entry now contains BOTH
    # trace 0 (full) and trace 1 (partial) color arrays for each mode, so
    # applyColorModeToActive can restyle both traces in a single call.
    colors_js = json.dumps(
        {
            did: {
                "emb_full": e[0],
                "emb_partial": e[1],
                "cluster_full": c[0],
                "cluster_partial": c[1],
            }
            for did, (e, c) in color_arrays.items()
        }
    )
    cluster_stats_js = json.dumps(per_tab_cluster_stats)
    contexts_js = json.dumps(contexts_shared or [])
    tab_meta_js = json.dumps(tab_meta or [])
    knn_grid_js = json.dumps(per_tab_knn_grid)
    overlay_pre_interp_js = "true" if overlay_pre_interp else "false"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #0e1117; color: #eee;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                overflow: hidden; }}
  /* Layout regions (fixed, non-overlapping):
       tabs bar  : top 0-44
       controls  : top 44-76 (color-mode radio, no more scrolling right)
       left bar  : top 76-bottom, left 0, width 220 (cluster stats — doesn't
                                                     occlude the plot)
       plot area : top 76-bottom, left 220-right
       img card  : floating bottom-right, closable, hidden until first click
  */
  #egoverse-tabs {{ position: fixed; top: 0; left: 0; right: 0; height: 44px;
                     background: #111; border-bottom: 1px solid #444; padding: 6px 8px;
                     overflow-x: auto; white-space: nowrap; z-index: 100;
                     box-sizing: border-box; }}
  .egoverse-tab {{ background: #222; color: #ddd; border: 1px solid #444;
                   padding: 4px 10px; margin-right: 4px; cursor: pointer;
                   border-radius: 4px; font-size: 12px; font-family: inherit; }}
  .egoverse-tab:hover {{ background: #333; }}
  .egoverse-tab.active {{ background: #4c78a8; color: #fff; border-color: #4c78a8; }}
  #egoverse-controls {{ position: fixed; top: 44px; left: 0; right: 0; height: 32px;
                         background: #141414; border-bottom: 1px solid #333;
                         padding: 5px 12px; box-sizing: border-box; z-index: 99;
                         display: flex; align-items: center; gap: 12px;
                         font-size: 12px; color: #ccc; }}
  #egoverse-color-mode label {{ margin: 0 4px; cursor: pointer; }}
  #egoverse-color-mode input {{ margin-right: 3px; vertical-align: middle; }}
  .egoverse-view-btn {{ background: #222; color: #ddd; border: 1px solid #444;
                         padding: 3px 8px; margin: 0 2px; cursor: pointer;
                         border-radius: 3px; font-size: 11px;
                         font-family: inherit; }}
  .egoverse-view-btn:hover {{ background: #333; }}
  .egoverse-view-btn.active {{ background: #4c78a8; color: #fff;
                                border-color: #4c78a8; }}
  #egoverse-loading {{ display: none; margin-left: auto; padding: 3px 8px;
                        background: #333; color: #ffd166; border-radius: 4px;
                        font-size: 11px; }}
  #egoverse-loading::before {{ content: "● "; animation: egoverse-pulse 1s infinite; }}
  @keyframes egoverse-pulse {{ 0%,100% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} }}
  #egoverse-cluster-bar {{ position: fixed; top: 76px; left: 0; bottom: 0; width: 260px;
                            background: #111; border-right: 1px solid #333;
                            padding: 10px 12px; overflow-y: auto; z-index: 90;
                            box-sizing: border-box; font-size: 11px; }}
  #egoverse-cluster-bar .title {{ font-size: 12px; font-weight: 600; color: #ffd166;
                                   margin-bottom: 8px; }}
  #egoverse-cluster-bar .sublabel {{ color: #888; font-size: 10px; margin-bottom: 8px;
                                       word-wrap: break-word; }}
  #egoverse-cluster-bar .row {{ display: flex; justify-content: space-between;
                                 padding: 3px 0; border-bottom: 1px dotted #333; }}
  #egoverse-cluster-bar .cid {{ font-family: ui-monospace, monospace; color: #eee; }}
  #egoverse-cluster-bar .stat {{ color: #aaa; font-family: ui-monospace, monospace; }}
  #egoverse-cluster-bar .noise {{ color: #888; }}
  #egoverse-cluster-bar .swatch {{ display: inline-block; width: 10px; height: 10px;
                                    border-radius: 2px; margin-right: 6px;
                                    vertical-align: middle; }}
  #egoverse-panel-container {{ position: fixed; top: 76px; left: 260px; right: 0; bottom: 0; }}
  /* Only one plot exists at any time (lazy Plotly.newPlot on tab switch,
     Plotly.purge on the previous), so browser WebGL context limits and
     SVG-DOM stalls both go away. Hidden panels still use opacity so the
     active panel is guaranteed nonzero dimensions before newPlot runs. */
  .egoverse-panel {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                     opacity: 0; pointer-events: none; }}
  .egoverse-panel.active-panel {{ opacity: 1; pointer-events: auto; z-index: 2; }}
  .egoverse-panel > .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
  #egoverse-img-panel {{ position: fixed; bottom: 12px; right: 12px; width: 340px;
                          background: #111; color: #eee; border: 1px solid #444;
                          padding: 8px; border-radius: 8px; z-index: 9999;
                          box-shadow: 0 2px 10px rgba(0,0,0,0.5);
                          display: none; }}
  #egoverse-img-panel.open {{ display: block; }}
  #egoverse-img-close {{ position: absolute; top: 4px; right: 6px;
                          background: transparent; color: #ccc; border: 0;
                          font-size: 18px; cursor: pointer; padding: 2px 6px;
                          line-height: 1; }}
  #egoverse-img-close:hover {{ color: #fff; }}
  #egoverse-img-panel .hint {{ font-size: 12px; opacity: 0.7; margin-bottom: 6px;
                                padding-right: 20px; }}
  #egoverse-img-body {{ min-height: 40px; }}

  /* KNN grid view (per-tab alternate to the plot). Overflow-y so long
     cluster lists scroll independently of the rest of the page. */
  .egoverse-knn-grid {{ position: absolute; top: 0; left: 0; right: 0;
                         bottom: 0; padding: 14px 18px; overflow-y: auto;
                         box-sizing: border-box; }}
  .egoverse-knn-cluster {{ margin-bottom: 22px; padding-bottom: 16px;
                            border-bottom: 1px solid #333; }}
  .egoverse-knn-cluster-title {{ color: #ffd166; font-weight: 600;
                                    font-size: 13px; margin-bottom: 8px; }}
  .egoverse-knn-row {{ display: flex; gap: 12px; margin-bottom: 10px;
                        align-items: flex-start; }}
  .egoverse-knn-anchor {{ flex: 0 0 auto; }}
  .egoverse-knn-anchor .egoverse-knn-img-wrap {{ width: 360px; }}
  .egoverse-knn-anchor img {{ width: 100%; display: block;
                                border: 2px solid #ffd166;
                                border-radius: 6px; background: #000; }}
  .egoverse-knn-neighbors {{ display: flex; gap: 8px; flex-wrap: nowrap;
                              flex: 1 1 auto; overflow-x: auto; }}
  .egoverse-knn-neighbor {{ flex: 0 0 auto; }}
  .egoverse-knn-neighbor .egoverse-knn-img-wrap {{ width: 210px; }}
  .egoverse-knn-neighbor img {{ width: 100%; display: block;
                                 border: 1px solid #555; border-radius: 4px;
                                 background: #000; }}
  .egoverse-knn-label {{ font-size: 10px; opacity: 0.7; margin-top: 3px;
                          color: #ccc; line-height: 1.2; }}
  .egoverse-knn-empty {{ padding: 24px; color: #888; }}
</style>
</head>
<body>
<div id="egoverse-tabs">
  {"".join(tabs)}
</div>
<div id="egoverse-controls">
  <span id="egoverse-color-mode">
    Color:
    <label><input type="radio" name="egoverse-cmode" value="emb" checked>embodiment</label>
    <label><input type="radio" name="egoverse-cmode" value="cluster">cluster</label>
  </span>
  <label id="egoverse-partial-toggle" title="End-of-episode samples that reached <50% of the target joint arc D. Their placement in TSNE tends to reflect where the episode was truncated more than the actual motion, so they're hidden by default.">
    <input type="checkbox" id="egoverse-partial-cb"> show partial-D (end-of-ep)
    <span id="egoverse-partial-count" style="color:#888;font-size:10px;margin-left:4px;"></span>
  </label>
  <span style="margin-left:12px; border-left:1px solid #333; padding-left:12px;" title="TSNE plot vs. per-cluster cross-embodiment KNN grid.">
    View:
    <button type="button" id="egoverse-view-plot-btn" class="egoverse-view-btn active" data-view="plot">TSNE</button>
    <button type="button" id="egoverse-view-knn-btn" class="egoverse-view-btn" data-view="knn">KNN grid</button>
  </span>
  <span id="egoverse-loading">loading…</span>
</div>
<div id="egoverse-cluster-bar">
  <div class="title">HDBSCAN clusters</div>
  <div class="sublabel" id="egoverse-cluster-sublabel">(select a tab)</div>
  <div id="egoverse-cluster-body"></div>
</div>
<div id="egoverse-panel-container">
{"".join(panels)}
</div>
<div id="egoverse-img-panel">
  <button id="egoverse-img-close" type="button" title="close" onclick="var p=document.getElementById('egoverse-img-panel'); if(p) p.classList.remove('open');">×</button>
  <div class="hint">state image (with green GT overlay)</div>
  <div id="egoverse-img-body"></div>
</div>
<script>
var EGOVERSE_IMAGES = {images_json};
var EGOVERSE_PLOTS = {payloads_js};
var EGOVERSE_COLORS = {colors_js};
var EGOVERSE_CLUSTER_STATS = {cluster_stats_js};
var EGOVERSE_SAMPLE_CTX = {contexts_js};
var EGOVERSE_TAB_META = {tab_meta_js};
// Per-tab (div_id) cross-embodiment KNN grid data. See Python
// ``_compute_knn_grid_for_tab`` for the schema.
var EGOVERSE_KNN_GRID = {knn_grid_js};
// Which view is showing inside the active tab: "plot" (TSNE plotly) or
// "knn" (per-cluster medoid + KNN grid). Toggled from the controls bar.
var EGOVERSE_VIEW_MODE = "plot";
// Which tabs have had their KNN grid populated already (so we don't
// re-render on every toggle).
var EGOVERSE_KNN_RENDERED = {{}};
// When true, arc-mode overlays draw every raw frame from t=0 up to the
// timestep where joint_cum first hits D_m (i.e. the pre-interp trajectory),
// instead of M uniform-in-joint-arc waypoints.
var EGOVERSE_OVERLAY_PRE_INTERP = {overlay_pre_interp_js};
var EGOVERSE_CURRENT_PLOT_ID = null;
var EGOVERSE_CURRENT_TAB_INDEX = 0;
var EGOVERSE_COLOR_MODE = "emb";

// -- Overlay drawing (canvas atop the raw JPEG) ----------------------------
// The overlay reflects the CURRENT tab's semantics:
//   arc:  first-D-joint-arc chunk (uniform-in-joint-arc M waypoints)
//   time: first-H-frame time-window slice
// K in the sample context is pre-scaled to the JPEG's shipped pixel space,
// so project() outputs pixel coords directly usable on the canvas.
function _project(K, xyz) {{
  var u = K[0]*xyz[0] + K[1]*xyz[1] + K[2]*xyz[2] + K[3];
  var v = K[4]*xyz[0] + K[5]*xyz[1] + K[6]*xyz[2] + K[7];
  var w = K[8]*xyz[0] + K[9]*xyz[1] + K[10]*xyz[2] + K[11];
  if (Math.abs(w) < 1e-9) return null;
  return [u / w, v / w];
}}
function _sampleXyzAt(actions, T, D, t_frac, xyz_off) {{
  // linear interp for xyz at fractional timestep
  var t0 = Math.max(0, Math.min(T - 1, Math.floor(t_frac)));
  var t1 = Math.max(0, Math.min(T - 1, t0 + 1));
  var a = t_frac - t0;
  var out = [];
  for (var d = 0; d < 3; d++) {{
    var v0 = actions[t0 * D + xyz_off + d];
    var v1 = actions[t1 * D + xyz_off + d];
    out.push(v0 * (1 - a) + v1 * a);
  }}
  return out;
}}
function _computeWaypoints(ctx, tab) {{
  if (!ctx || !tab) return null;
  var T = ctx.T, D = ctx.D, actions = ctx.actions;
  var right_off = (D === 14) ? 7 : 6;
  if (tab.kind === "time") {{
    // first H frames — one waypoint per frame
    var H = Math.min(Math.max(1, tab.H | 0), T);
    var L = [], R = [];
    for (var t = 0; t < H; t++) {{
      L.push([actions[t*D+0], actions[t*D+1], actions[t*D+2]]);
      R.push([actions[t*D+right_off], actions[t*D+right_off+1], actions[t*D+right_off+2]]);
    }}
    return {{L: L, R: R}};
  }}
  if (tab.kind === "arc") {{
    var D_m = tab.D_m, M = Math.max(2, tab.M | 0);
    // Cumulative per-arm arc lengths on the fly
    var cum_L = [0], cum_R = [0];
    for (var t = 1; t < T; t++) {{
      var lx = actions[t*D+0] - actions[(t-1)*D+0];
      var ly = actions[t*D+1] - actions[(t-1)*D+1];
      var lz = actions[t*D+2] - actions[(t-1)*D+2];
      var rx = actions[t*D+right_off]   - actions[(t-1)*D+right_off];
      var ry = actions[t*D+right_off+1] - actions[(t-1)*D+right_off+1];
      var rz = actions[t*D+right_off+2] - actions[(t-1)*D+right_off+2];
      cum_L.push(cum_L[t-1] + Math.hypot(lx, ly, lz));
      cum_R.push(cum_R[t-1] + Math.hypot(rx, ry, rz));
    }}
    var joint_cum = new Array(T);
    for (var t = 0; t < T; t++) joint_cum[t] = cum_L[t] + cum_R[t];
    var J_max = joint_cum[T-1];
    if (J_max < 1e-6) return null;
    var s_end = Math.min(D_m, J_max);
    // Pre-interp mode: draw every raw frame from t=0 up to the first
    // timestep where joint_cum(t) >= s_end. Shows the underlying trajectory
    // before uniform-in-joint-arc resampling to M waypoints.
    if (EGOVERSE_OVERLAY_PRE_INTERP) {{
      var t_end_idx = T - 1;
      for (var t = 0; t < T; t++) {{
        if (joint_cum[t] >= s_end - 1e-12) {{ t_end_idx = t; break; }}
      }}
      var L_raw = [], R_raw = [];
      for (var t = 0; t <= t_end_idx; t++) {{
        L_raw.push([actions[t*D+0], actions[t*D+1], actions[t*D+2]]);
        R_raw.push([actions[t*D+right_off], actions[t*D+right_off+1], actions[t*D+right_off+2]]);
      }}
      return {{L: L_raw, R: R_raw}};
    }}
    // Standard: invert joint_cum(t) at M uniformly-spaced targets in [0, s_end]
    var L = [], R = [];
    for (var k = 0; k < M; k++) {{
      var s_k = (M === 1) ? 0 : (k / (M - 1)) * s_end;
      var i = 0;
      while (i + 1 < T && joint_cum[i+1] < s_k) i++;
      var span = Math.max(joint_cum[i+1] - joint_cum[i], 1e-12);
      var alpha = (s_k - joint_cum[i]) / span;
      var t_frac = i + Math.max(0, Math.min(1, alpha));
      L.push(_sampleXyzAt(actions, T, D, t_frac, 0));
      R.push(_sampleXyzAt(actions, T, D, t_frac, right_off));
    }}
    // Also produce the raw pre-interp trajectory (t=0..t_end where
    // joint_cum crosses s_end) as ``L_line``/``R_line`` — used by the
    // renderer to draw the thin polyline covering the full chunk while
    // dots stay at the M waypoints.
    var t_end_idx = T - 1;
    for (var t = 0; t < T; t++) {{
      if (joint_cum[t] >= s_end - 1e-12) {{ t_end_idx = t; break; }}
    }}
    var L_line = [], R_line = [];
    for (var t = 0; t <= t_end_idx; t++) {{
      L_line.push([actions[t*D+0], actions[t*D+1], actions[t*D+2]]);
      R_line.push([actions[t*D+right_off], actions[t*D+right_off+1], actions[t*D+right_off+2]]);
    }}
    return {{L: L, R: R, L_line: L_line, R_line: R_line}};
  }}
  return null;
}}
// matplotlib 'Greens' colormap sampled at 9 stops — used as a light→dark
// gradient along each arm's trajectory so the drawn path encodes temporal
// direction (matches ``plt.get_cmap('Greens')(linspace(0,1,N))`` from
// ``draw_dot_on_frame`` in ``egomimicUtils.py``).
var _GREENS = [
  [247, 252, 245], [229, 245, 224], [199, 233, 192],
  [161, 217, 155], [116, 196, 118], [65, 171, 93],
  [35, 139, 69],   [0, 109, 44],    [0, 68, 27]
];
function _greensAt(t) {{
  // t in [0,1] — linearly interp within the stops
  var n = _GREENS.length - 1;
  var f = Math.max(0, Math.min(n, t * n));
  var i = Math.floor(f);
  var a = f - i;
  var c0 = _GREENS[i], c1 = _GREENS[Math.min(i + 1, n)];
  return 'rgb(' + Math.round(c0[0]*(1-a)+c1[0]*a) + ',' +
                  Math.round(c0[1]*(1-a)+c1[1]*a) + ',' +
                  Math.round(c0[2]*(1-a)+c1[2]*a) + ')';
}}

function _renderOverlayCanvas(ctx, tab, image_url) {{
  // Returns a Promise resolving to a data URL that embeds the raw JPEG with
  // green dots overlaid, matching the canonical `_viz_traj → draw_actions →
  // draw_dot_on_frame` output: light-to-dark 'Greens' gradient per arm and
  // radius-5 dots.
  return new Promise(function(resolve) {{
    var wp = _computeWaypoints(ctx, tab);
    if (!wp) {{ resolve(image_url); return; }}
    var K = ctx.K;
    var img = new Image();
    img.onload = function() {{
      var c = document.createElement('canvas');
      c.width = ctx.img_w; c.height = ctx.img_h;
      var g = c.getContext('2d');
      g.drawImage(img, 0, 0, c.width, c.height);
      function _projectPts(pts) {{
        var pxs = [];
        for (var i = 0; i < pts.length; i++) {{
          var p = _project(K, pts[i]);
          if (!p) {{ pxs.push(null); continue; }}
          if (p[0] < -20 || p[0] > c.width + 20 || p[1] < -20 || p[1] > c.height + 20) {{ pxs.push(null); continue; }}
          pxs.push(p);
        }}
        return pxs;
      }}
      function drawTrail(dotPts, linePts, dot_radius) {{
        // ``dotPts`` is the sequence used for waypoint dots (Greens gradient
        // by index). ``linePts`` is optional — only the arc post-interp mode
        // sets it (line traces the full raw chunk while dots stay at the M
        // waypoints). Time-based (baseline) and arc pre-interp modes leave
        // ``linePts`` undefined so we draw only dots — matches the canonical
        // ``draw_dot_on_frame`` in ``egomimicUtils.py`` used at cotrain/eval
        // time (dots-only Greens gradient, no polyline).
        var pxs_dots = _projectPts(dotPts);
        if (linePts) {{
          var pxs_line = _projectPts(linePts);
          g.strokeStyle = 'rgba(65, 171, 93, 0.75)';
          g.lineWidth = 1.0;
          g.beginPath();
          var moved = false;
          for (var i = 0; i < pxs_line.length; i++) {{
            if (!pxs_line[i]) {{ moved = false; continue; }}
            if (!moved) {{ g.moveTo(pxs_line[i][0], pxs_line[i][1]); moved = true; }}
            else g.lineTo(pxs_line[i][0], pxs_line[i][1]);
          }}
          g.stroke();
        }}
        // dots with per-index Greens gradient
        var n = pxs_dots.length;
        var r = (dot_radius !== undefined) ? dot_radius : 2.2;
        for (var i = 0; i < n; i++) {{
          if (!pxs_dots[i]) continue;
          var t = (n <= 1) ? 1.0 : (i / (n - 1));
          g.fillStyle = _greensAt(t);
          g.beginPath();
          g.arc(pxs_dots[i][0], pxs_dots[i][1], r, 0, Math.PI * 2);
          g.fill();
        }}
      }}
      // Radius scaling: canonical draw_dot_on_frame uses radius=5 on the
      // native image (typically 480×640). We ship images scaled to
      // max_side ~= 224, so pick radius so it looks the same size as the
      // cotrain viz relative to the image (5 * 224/640 ≈ 1.75).
      // Time-based tabs have H ~= 100 dots per arm at native 30fps —
      // dense, so we shrink further to keep them distinguishable.
      var _isTime = (tab && tab.kind === "time");
      var _r = _isTime ? 1.4 : 2.2;
      drawTrail(wp.L, wp.L_line, _r);
      drawTrail(wp.R, wp.R_line, _r);
      resolve(c.toDataURL('image/png'));
    }};
    img.onerror = function() {{ resolve(image_url); }};
    img.src = image_url;
  }});
}}

// -- Cluster sidebar rendering ---------------------------------------------
// Same golden-angle HSV palette used by _cluster_shade_color in Python — we
// mirror it in JS so the swatches next to each cluster row match the plot
// colors when Color: cluster is selected.
function _clusterSwatchColor(cid) {{
  if (cid < 0) return "#888";
  var hue = ((cid * 0.6180339887498949) % 1.0) * 360.0;
  return "hsl(" + hue.toFixed(1) + ", 70%, 60%)";
}}
function renderClusterBar(plot_id) {{
  var subEl = document.getElementById("egoverse-cluster-sublabel");
  var body = document.getElementById("egoverse-cluster-body");
  if (!body || !subEl) return;
  var stats = EGOVERSE_CLUSTER_STATS[plot_id];
  if (!stats) {{ body.innerHTML = ""; subEl.textContent = "(no data)"; return; }}
  var kNote = stats.min_cluster_size ? (" (k=" + stats.min_cluster_size + ")") : "";
  subEl.innerHTML = (stats.label || "") + '<br><span style="opacity:0.6">' +
                    'persistence (0-1, hdbscan)<br>' +
                    'medKNN' + kNote + ' — smaller = tighter</span>';
  var sizes = stats.sizes || {{}};
  var pers = stats.persistence || {{}};
  var knn = stats.median_knn || {{}};
  // Sort clusters by persistence desc; noise (-1) at the bottom.
  var rows = [];
  Object.keys(sizes).forEach(function(k) {{
    var cid = parseInt(k, 10);
    rows.push({{cid: cid, n: sizes[k], p: pers[k], k: knn[k]}});
  }});
  rows.sort(function(a, b) {{
    if ((a.cid < 0) !== (b.cid < 0)) return a.cid < 0 ? 1 : -1;
    var ap = (a.p === undefined) ? -1 : a.p;
    var bp = (b.p === undefined) ? -1 : b.p;
    if (bp !== ap) return bp - ap;
    return b.n - a.n;
  }});
  var total = rows.reduce(function(s, r) {{ return s + r.n; }}, 0);
  var top = '<div class="row"><span class="cid">' +
            (stats.n_clusters || 0) + ' clusters</span>' +
            '<span class="stat">N=' + total + '</span></div>';
  var html = top + rows.map(function(r) {{
    var isNoise = r.cid < 0;
    var swatch = '<span class="swatch" style="background:' +
                  _clusterSwatchColor(r.cid) + ';"></span>';
    var label = isNoise ? "noise" : ("c" + r.cid);
    var pTxt = (r.p !== undefined && !isNoise) ? ("p=" + r.p.toFixed(3)) : "";
    var kTxt = (r.k !== undefined && !isNoise) ? ("kNN=" + r.k.toFixed(3)) : "";
    var rightLine1 = "n=" + r.n;
    var rightLine2 = [pTxt, kTxt].filter(Boolean).join("  ");
    return '<div class="row' + (isNoise ? ' noise' : '') + '">' +
           '<span class="cid">' + swatch + label + '</span>' +
           '<span class="stat" style="text-align:right;">' +
           rightLine1 + (rightLine2 ? '<br>' + rightLine2 : '') +
           '</span>' +
           '</div>';
  }}).join("");
  body.innerHTML = html;
}}

function currentColorArrays(plot_id) {{
  // Returns [full_trace_colors, partial_trace_colors] for the active mode.
  var entry = EGOVERSE_COLORS[plot_id];
  if (!entry) return null;
  if (EGOVERSE_COLOR_MODE === "cluster") {{
    return [entry.cluster_full, entry.cluster_partial];
  }}
  return [entry.emb_full, entry.emb_partial];
}}

function applyColorModeToActive() {{
  if (!EGOVERSE_CURRENT_PLOT_ID || !window.Plotly) return;
  var gd = document.getElementById(EGOVERSE_CURRENT_PLOT_ID);
  var arrs = currentColorArrays(EGOVERSE_CURRENT_PLOT_ID);
  if (!gd || !arrs) return;
  // Restyle both traces at once — trace 0 = full-D, trace 1 = partial-D.
  try {{
    Plotly.restyle(gd, {{'marker.color': [arrs[0], arrs[1]]}}, [0, 1]);
  }} catch (e) {{}}
}}

// -- Hide/show partial-D (end-of-episode) samples --------------------------
// Trace 1 in every plot is the partial-D subset. Toggling its visible
// attribute makes those points fully non-plotted, non-hoverable, and
// non-clickable in one call. Default = hidden (they cluster by episode
// truncation more than by motion pattern, so they mostly add noise).
var EGOVERSE_SHOW_PARTIAL = false;
function applyPartialVisibilityToActive() {{
  if (!EGOVERSE_CURRENT_PLOT_ID || !window.Plotly) return;
  var gd = document.getElementById(EGOVERSE_CURRENT_PLOT_ID);
  if (!gd || !gd.data || gd.data.length < 2) return;
  try {{
    Plotly.restyle(gd, {{visible: EGOVERSE_SHOW_PARTIAL}}, [1]);
  }} catch (e) {{}}
}}

function attachClickHandlers() {{
  var divs = document.querySelectorAll('.plotly-graph-div');
  divs.forEach(function(gd) {{
    if (!gd.on || gd.dataset.egoverseAttached === "1") return;
    gd.dataset.egoverseAttached = "1";
    gd.on('plotly_click', function(evt) {{
      if (!evt || !evt.points || !evt.points.length) return;
      var pt = evt.points[0];
      var cd = pt.customdata || [];
      var idx = (cd[0] !== undefined) ? parseInt(cd[0], 10) : -1;
      var uri = (idx >= 0 && idx < EGOVERSE_IMAGES.length) ? EGOVERSE_IMAGES[idx] : '';
      var emb = cd[1] || '';
      var samp = (cd[2] !== undefined) ? cd[2] : '';
      var split = cd[3] || '';
      var zero = cd[4];
      var el = document.getElementById('egoverse-img-body');
      var panel = document.getElementById('egoverse-img-panel');
      if (panel) panel.classList.add('open');
      var label = emb + '  idx=' + samp + '  [' + split + ']' + (zero ? '  (zero)' : '');
      if (!uri) {{
        el.innerHTML = '<em>no image cached</em><div style="font-size:11px;margin-top:6px;opacity:0.7;">' + label + '</div>';
        return;
      }}
      // Dynamic overlay: use current tab's semantics + this sample's actions.
      var ctx = (idx >= 0 && idx < EGOVERSE_SAMPLE_CTX.length) ? EGOVERSE_SAMPLE_CTX[idx] : null;
      var tab = EGOVERSE_TAB_META[EGOVERSE_CURRENT_TAB_INDEX];
      var tabLabel = tab ? (tab.kind === 'arc'
        ? ('arc D=' + Math.round((tab.D_m || 0) * 100) + 'cm  M=' + tab.M)
        : ('time H=' + tab.H)) : '';
      // Draw immediately with raw image while overlay is computed.
      el.innerHTML = '<img id="egoverse-img-el" src="' + uri +
                     '" style="width:100%;background:#000;display:block;border-radius:4px;"/>' +
                     '<div style="font-size:11px;margin-top:6px;">' + label +
                     (tabLabel ? '<br><span style="opacity:0.6;">overlay: ' + tabLabel + '</span>' : '') +
                     '</div>';
      if (ctx && tab) {{
        _renderOverlayCanvas(ctx, tab, uri).then(function(overlayed) {{
          var imgEl = document.getElementById('egoverse-img-el');
          if (imgEl) imgEl.src = overlayed;
        }});
      }}
    }});
  }});
}}

var EGOVERSE_LOADING_TOKEN = 0;

function setLoading(on) {{
  var pill = document.getElementById('egoverse-loading');
  if (pill) pill.style.display = on ? 'inline-block' : 'none';
}}

function switchTab(index) {{
  var target_id = 'egoverse-plot-' + index;
  // Re-clicking the already-live tab is a no-op (avoid pointless purge+redraw
  // which briefly blanks the plot and drops any zoom/pan state).
  if (target_id === EGOVERSE_CURRENT_PLOT_ID) return;

  // Race-guard: if a tab click happens while a previous newPlot is still
  // in-flight, we don't want the older promise to run its .then() and stomp
  // over the newer tab's state. Bump the token; only the callback whose
  // token matches the current one proceeds.
  EGOVERSE_LOADING_TOKEN += 1;
  var my_token = EGOVERSE_LOADING_TOKEN;

  document.querySelectorAll('.egoverse-panel').forEach(function(p) {{
    p.classList.remove('active-panel');
  }});
  document.querySelectorAll('.egoverse-tab').forEach(function(b) {{
    b.classList.remove('active');
  }});
  var panel = document.getElementById('egoverse-panel-' + index);
  var btn = document.querySelector('.egoverse-tab[data-index="' + index + '"]');
  if (panel) panel.classList.add('active-panel');
  if (btn) btn.classList.add('active');

  // Track active tab index for the JS overlay renderer.
  EGOVERSE_CURRENT_TAB_INDEX = index;
  // Update the left-side cluster stats sidebar to match the target tab.
  renderClusterBar(target_id);

  // Lazy: purge previous, then Plotly.newPlot the target. Only one plot is
  // ever alive at a time — avoids browser WebGL context limit + SVG-DOM
  // stalls.
  if (EGOVERSE_CURRENT_PLOT_ID && EGOVERSE_CURRENT_PLOT_ID !== target_id) {{
    var prev = document.getElementById(EGOVERSE_CURRENT_PLOT_ID);
    if (prev && window.Plotly) {{
      try {{ Plotly.purge(prev); }} catch (e) {{}}
      // Purge clears the plot but NOT our HTML dataset attribute, so the
      // click-handler dedup flag would prevent us from re-attaching if the
      // user revisits this tab. Reset it explicitly.
      if (prev.dataset) prev.dataset.egoverseAttached = "";
    }}
  }}
  var gd = document.getElementById(target_id);
  // Reset the target's flag too, in case it was ever set previously.
  if (gd && gd.dataset) gd.dataset.egoverseAttached = "";

  var payload = EGOVERSE_PLOTS[target_id];
  if (gd && payload && window.Plotly) {{
    setLoading(true);
    Plotly.newPlot(gd, payload.data, payload.layout, {{responsive: true}}).then(function() {{
      // Stale callback (user clicked another tab while this was rendering) —
      // don't touch DOM further; the newer switchTab will handle setup.
      if (my_token !== EGOVERSE_LOADING_TOKEN) return;
      setLoading(false);
      attachClickHandlers();
      // Apply the currently-selected color mode; the figure was serialized
      // with emb colors so this is a no-op for mode==emb.
      if (EGOVERSE_COLOR_MODE !== "emb") applyColorModeToActive();
      // Apply the partial-D toggle to the freshly-rendered plot.
      applyPartialVisibilityToActive();
      updatePartialCountLabel();
      // Reapply view mode so the correct sub-view (plot vs. KNN grid) is
      // visible after a D/W tab switch — and lazy-populate the KNN grid
      // for this tab if the user was already in KNN view.
      applyViewMode();
      try {{ Plotly.Plots.resize(gd); }} catch (e) {{}}
    }}).catch(function(err) {{
      if (my_token !== EGOVERSE_LOADING_TOKEN) return;
      setLoading(false);
      console.error('Plotly.newPlot failed:', err);
    }});
    EGOVERSE_CURRENT_PLOT_ID = target_id;
  }} else {{
    attachClickHandlers();
  }}
}}

document.querySelectorAll('.egoverse-tab').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    switchTab(parseInt(btn.dataset.index, 10));
  }});
}});

// Color-mode radio change: restyle the active plot in place.
document.querySelectorAll('input[name="egoverse-cmode"]').forEach(function(r) {{
  r.addEventListener('change', function() {{
    if (r.checked) {{ EGOVERSE_COLOR_MODE = r.value; applyColorModeToActive(); }}
  }});
}});

// Partial-D checkbox: flip the partial-trace visibility on the active plot.
function updatePartialCountLabel() {{
  var lbl = document.getElementById('egoverse-partial-count');
  if (!lbl) return;
  var payload = EGOVERSE_PLOTS[EGOVERSE_CURRENT_PLOT_ID];
  if (!payload || !payload.data || payload.data.length < 2) {{
    lbl.textContent = "";
    return;
  }}
  var n_partial = (payload.data[1].x || []).length;
  var n_full = (payload.data[0].x || []).length;
  lbl.textContent = "(" + n_partial + " of " + (n_partial + n_full) + " hidden)";
  if (EGOVERSE_SHOW_PARTIAL) {{
    lbl.textContent = "(" + n_partial + " of " + (n_partial + n_full) + " shown)";
  }}
}}
var _egPartialCb = document.getElementById('egoverse-partial-cb');
if (_egPartialCb) {{
  _egPartialCb.checked = EGOVERSE_SHOW_PARTIAL;
  _egPartialCb.addEventListener('change', function() {{
    EGOVERSE_SHOW_PARTIAL = _egPartialCb.checked;
    applyPartialVisibilityToActive();
    updatePartialCountLabel();
  }});
}}

// -- Cross-embodiment KNN grid view ---------------------------------------
// A per-tab alternate view accessed via the TSNE/KNN toggle in the controls
// bar. Each cluster contributes two rows: one anchor per embodiment on the
// left, four opposite-embodiment neighbors on the right. Anchor images are
// rendered ~1.7× the normal size; neighbor images ~1× the normal size.
// Overlays reuse the existing ``_renderOverlayCanvas`` so the drawn line +
// dots follow the SAME semantics as clicking a point on the TSNE plot for
// the current tab.
function _knnPanelIdForTab(tab_index) {{
  return "egoverse-knn-" + tab_index;
}}
function _makeKnnImgCell(imgIdx, isAnchor, labelHtml) {{
  var uri = (imgIdx >= 0 && imgIdx < EGOVERSE_IMAGES.length)
              ? EGOVERSE_IMAGES[imgIdx] : "";
  var wrapCls = isAnchor ? "egoverse-knn-anchor" : "egoverse-knn-neighbor";
  var el = document.createElement("div");
  el.className = wrapCls;
  var wrap = document.createElement("div");
  wrap.className = "egoverse-knn-img-wrap";
  var img = document.createElement("img");
  img.dataset.imgIdx = String(imgIdx);
  img.src = uri;   // raw (no overlay) until _renderOverlayCanvas fills it
  wrap.appendChild(img);
  el.appendChild(wrap);
  var lbl = document.createElement("div");
  lbl.className = "egoverse-knn-label";
  lbl.innerHTML = labelHtml;
  el.appendChild(lbl);
  return {{el: el, img: img}};
}}
function _renderKnnCellOverlay(imgEl, imgIdx, tab) {{
  var ctx = (imgIdx >= 0 && imgIdx < EGOVERSE_SAMPLE_CTX.length)
              ? EGOVERSE_SAMPLE_CTX[imgIdx] : null;
  var uri = (imgIdx >= 0 && imgIdx < EGOVERSE_IMAGES.length)
              ? EGOVERSE_IMAGES[imgIdx] : "";
  if (!ctx || !tab || !uri) return;
  _renderOverlayCanvas(ctx, tab, uri).then(function(overlayed) {{
    imgEl.src = overlayed;
  }});
}}
function populateKnnGrid(div_id, tab_index) {{
  var knn_panel_id = _knnPanelIdForTab(tab_index);
  var panel = document.getElementById(knn_panel_id);
  if (!panel) return;
  if (EGOVERSE_KNN_RENDERED[knn_panel_id]) return;   // already built
  var grid = EGOVERSE_KNN_GRID[div_id];
  if (!grid || grid.length === 0) {{
    panel.innerHTML = '<div class="egoverse-knn-empty">' +
      'No cross-embodiment clusters found for this tab — either HDBSCAN ' +
      'returned all-noise or every cluster is single-embodiment.</div>';
    EGOVERSE_KNN_RENDERED[knn_panel_id] = true;
    return;
  }}
  var tab = EGOVERSE_TAB_META[tab_index];
  panel.innerHTML = "";
  grid.forEach(function(cluster) {{
    var box = document.createElement("div");
    box.className = "egoverse-knn-cluster";
    var title = document.createElement("div");
    title.className = "egoverse-knn-cluster-title";
    title.innerHTML =
      '<span class="swatch" style="display:inline-block;width:12px;' +
      'height:12px;border-radius:2px;margin-right:6px;vertical-align:middle;' +
      'background:' + _clusterSwatchColor(cluster.cluster_id) + '"></span>' +
      'cluster c' + cluster.cluster_id + '  (N=' + cluster.n_points +
      ' points, medoid anchors + 4-NN in opposite embodiment)';
    box.appendChild(title);

    // Two rows: one per embodiment side. Row layout is
    //   [anchor of emb_from]  |  [4 nearest neighbors in emb_to]
    var embs = cluster.embodiments;   // [emb_a, emb_b] sorted
    for (var i = 0; i < embs.length; i++) {{
      var e_from = embs[i];
      var e_to = embs[1 - i];
      var anchor = cluster.anchors[e_from];
      var neighbors = cluster.knn[e_from + "_to_" + e_to] || [];
      if (!anchor) continue;
      var row = document.createElement("div");
      row.className = "egoverse-knn-row";
      var anchorLbl = e_from + '<br>sample_idx=' + anchor.sample_idx +
                      '  (cluster c' + cluster.cluster_id + ' medoid)';
      var anchorCell = _makeKnnImgCell(anchor.img_idx, true, anchorLbl);
      row.appendChild(anchorCell.el);
      _renderKnnCellOverlay(anchorCell.img, anchor.img_idx, tab);

      var neighborsWrap = document.createElement("div");
      neighborsWrap.className = "egoverse-knn-neighbors";
      neighbors.forEach(function(n, ni) {{
        var lbl = e_to + '<br>sample_idx=' + n.sample_idx +
                  '  d=' + n.dist.toFixed(2);
        var cell = _makeKnnImgCell(n.img_idx, false, lbl);
        neighborsWrap.appendChild(cell.el);
        _renderKnnCellOverlay(cell.img, n.img_idx, tab);
      }});
      row.appendChild(neighborsWrap);
      box.appendChild(row);
    }}
    panel.appendChild(box);
  }});
  EGOVERSE_KNN_RENDERED[knn_panel_id] = true;
}}
function applyViewMode() {{
  // Show/hide the plot vs. KNN grid inside every panel. Only the active
  // panel is visible at all (via .active-panel opacity), so we don't need
  // to touch inactive panels — but we do it anyway to keep state coherent
  // if switchTab hits a stale flag.
  var showPlot = (EGOVERSE_VIEW_MODE === "plot");
  document.querySelectorAll('.egoverse-view-plot').forEach(function(el) {{
    el.style.display = showPlot ? "block" : "none";
  }});
  document.querySelectorAll('.egoverse-view-knn').forEach(function(el) {{
    el.style.display = showPlot ? "none" : "block";
  }});
  // Update button-active state.
  document.querySelectorAll('.egoverse-view-btn').forEach(function(btn) {{
    if (btn.dataset.view === EGOVERSE_VIEW_MODE) btn.classList.add("active");
    else btn.classList.remove("active");
  }});
  // Populate the active tab's KNN grid on demand.
  if (!showPlot && EGOVERSE_CURRENT_PLOT_ID) {{
    populateKnnGrid(EGOVERSE_CURRENT_PLOT_ID, EGOVERSE_CURRENT_TAB_INDEX);
  }} else if (showPlot && EGOVERSE_CURRENT_PLOT_ID && window.Plotly) {{
    // Plot was untouched by the hide, so a quick resize brings it back
    // sized correctly if the viewport changed while we were on the grid.
    var gd = document.getElementById(EGOVERSE_CURRENT_PLOT_ID);
    if (gd) {{ try {{ Plotly.Plots.resize(gd); }} catch (e) {{}} }}
  }}
}}
document.querySelectorAll('.egoverse-view-btn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    EGOVERSE_VIEW_MODE = btn.dataset.view;
    applyViewMode();
  }});
}});

// Image card close button: hide the panel (open again on next point click).
var _egImgClose = document.getElementById('egoverse-img-close');
if (_egImgClose) {{
  _egImgClose.addEventListener('click', function() {{
    var p = document.getElementById('egoverse-img-panel');
    if (p) p.classList.remove('open');
  }});
}}

// Wait for plotly.js to finish loading from CDN, then activate tab 0 (which
// triggers the first newPlot). Fall back to polling if the load event races.
function egoverseBootstrap() {{
  if (window.Plotly && Plotly.newPlot) {{
    switchTab(0);
  }} else {{
    setTimeout(egoverseBootstrap, 60);
  }}
}}
egoverseBootstrap();

// Keep the active plot sized to its container.
window.addEventListener('resize', function() {{
  if (!EGOVERSE_CURRENT_PLOT_ID || !window.Plotly) return;
  var gd = document.getElementById(EGOVERSE_CURRENT_PLOT_ID);
  if (gd) {{ try {{ Plotly.Plots.resize(gd); }} catch (e) {{}} }}
}});
</script>
</body>
</html>
"""
    out_dir = os.path.dirname(os.path.abspath(output_html))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(doc)
    print(
        f"\nWrote sweep HTML: {output_html}  "
        f"({len(sweeps)} tabs, {len(images_shared)} shared images, "
        f"{os.path.getsize(output_html) / 1e6:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="arc_embedding_sweep.yaml",
)
def main(cfg: DictConfig) -> None:
    load_env()

    action_key = str(cfg.get("action_key", "actions_cartesian"))
    split = str(cfg.get("split", "train"))
    feature = str(cfg.feature)
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

    # Widen the per-sample raw window so joint_cum can actually reach the
    # largest D in the sweep before the chunk ends. Default heuristic:
    # ceil(max_D_cm * 2.5) with a floor of 200 — for D=280cm this gives 700
    # frames (~23 s of eva motion at 30 fps), comfortably above the raw hand
    # motion needed to accumulate 280 cm of joint arc even at brisk speed.
    # Override via ``raw_horizon`` in the top-level config.
    _max_D_cm = max(float(d) for d in cfg.sweep.joint_distance_cm)
    target_horizon = int(
        OmegaConf.select(cfg, "raw_horizon", default=max(200, int(_max_D_cm * 2.5) + 1))
    )
    print(
        f"[sweep] raw_horizon={target_horizon} (max_D={int(_max_D_cm)}cm) — "
        "keymap horizons + InterpolatePose lengths widened accordingly"
    )
    print("[sweep] iterating datasets once (cache chunks + images)")
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
        target_horizon=target_horizon,
    )
    print(f"[sweep] cached N={len(chunks)} samples, {len(images)} image URIs")

    if len(chunks) == 0:
        raise RuntimeError(
            "No samples cached — nothing to sweep. Check dataset paths + filters."
        )

    joint_dists = list(cfg.sweep.joint_distance_cm)
    waypoints_grid = list(cfg.sweep.waypoints)
    print(
        f"[sweep] grid: joint_distance_cm={joint_dists}  waypoints={waypoints_grid} "
        f"→ {len(joint_dists) * len(waypoints_grid)} sweep points"
    )

    sweeps: list[tuple[str, np.ndarray, list[dict], dict | None]] = []
    tab_meta: list[dict] = []
    for D in joint_dists:
        for W in waypoints_grid:
            print(f"\n[sweep] D={D}cm  W={W}")
            embed, per_meta, cluster_stats = _run_one_sweep(
                chunks,
                meta,
                joint_distance_cm=float(D),
                waypoints=int(W),
                tokenizer_cfg=cfg.tokenizer,
                feature_mode=feature,
                include_zero_tokens=include_zero_tokens,
                tsne_params=tsne_params,
            )
            key = f"D={int(D)}cm (joint)  W={int(W)}"
            sweeps.append((key, embed, per_meta, cluster_stats))
            tab_meta.append({"kind": "arc", "D_m": float(D) / 100.0, "M": int(W)})

    _emit_tabbed_html(
        sweeps,
        images_shared=images,
        title=str(cfg.get("title", "Arc token sweep")),
        output_html=str(cfg.get("output_html", "arc_embedding_sweep.html")),
        contexts_shared=contexts,
        tab_meta=tab_meta,
        overlay_pre_interp=bool(
            OmegaConf.select(cfg, "overlay_pre_interp", default=False)
        ),
    )


if __name__ == "__main__":
    main()
