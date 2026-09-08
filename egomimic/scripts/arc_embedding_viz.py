"""Arc-token embedding visualization.

Given a hydra data config (e.g. cotrain_pi_base.yaml) and an embedder config
(e.g. embedder/pca_2d.yaml), this script:

  1. Instantiates every dataset in ``data.train_datasets`` (and/or
     ``data.valid_datasets`` depending on ``split``).
  2. Draws a bounded random sample from each and pulls the post-transform
     ``actions_cartesian`` (T, 14) bimanual cartesian chunk.
  3. Runs each chunk through ``BimanualArcLengthTokenizer`` with the config
     from ``embedder.tokenizer`` (min_distance_unit, resampled_vector_length,
     velocity mode, dt, ...), giving one (M, arc_dim) arc token per sample.
  4. Extracts a per-token feature vector (see ``embedder.feature``: flatten /
     positions / positions_delta / velocity).
  5. Fits the chosen dimensionality reducer (PCA / t-SNE / UMAP) to 2D or 3D.
  6. Writes an interactive plotly HTML file colored by embodiment.

Example:
  python -m egomimic.scripts.arc_embedding_viz \\
      data=cotrain_pi_base embedder=pca_3d \\
      paths.dataset_dir=/path/to/zarr output_html=./out/pca3d.html

Overrides also work per-key, e.g.:
  ... embedder.tokenizer.min_distance_unit=0.1 \\
      embedder.tokenizer.resampled_vector_length=32 \\
      embedder.max_samples_per_dataset=2000
"""

from __future__ import annotations

import base64
import io
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARM_DIM,
    BIMANUAL_CARTESIAN_DIM,
    INVALID_POSE_THRESHOLD,
    BimanualArcLengthConfig,
    BimanualArcLengthTokenizer,
)
from egomimic.utils.aws.aws_data_utils import load_env

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _build_tokenizer(tokenizer_cfg: DictConfig) -> BimanualArcLengthTokenizer:
    config = BimanualArcLengthConfig(
        min_distance_unit=float(tokenizer_cfg.min_distance_unit),
        resampled_vector_length=int(tokenizer_cfg.resampled_vector_length),
        mode=str(tokenizer_cfg.mode),
        dt=float(tokenizer_cfg.dt),
        zero_dist_epsilon=float(
            OmegaConf.select(tokenizer_cfg, "zero_dist_epsilon", default=1e-6)
        ),
        max_steps_per_chunk=int(
            OmegaConf.select(tokenizer_cfg, "max_steps_per_chunk", default=200)
        ),
    )
    return BimanualArcLengthTokenizer(config)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_feature(
    arc: np.ndarray, arm_dim_tok: int, velocity_dim: int, feature: str
) -> np.ndarray:
    """Reduce one (M, arc_dim) arc token to a 1D feature vector.

    Layout per arm inside ``arc``: [xyz(3) ypr(3) grip(1) trans_vel(vd)].
    Bimanual concat: [<left block> | <right block>], each block ``arm_dim_tok``
    columns wide.
    """
    if feature == "flatten":
        return arc.reshape(-1)

    left_xyz = arc[:, 0:3]
    right_xyz = arc[:, arm_dim_tok : arm_dim_tok + 3]

    if feature == "positions":
        return np.concatenate([left_xyz, right_xyz], axis=-1).reshape(-1)
    if feature == "positions_delta":
        return np.concatenate(
            [left_xyz - left_xyz[0:1], right_xyz - right_xyz[0:1]], axis=-1
        ).reshape(-1)
    if feature == "velocity":
        left_vel = arc[:, ARM_DIM : ARM_DIM + velocity_dim]
        right_vel = arc[:, arm_dim_tok + ARM_DIM : arm_dim_tok + ARM_DIM + velocity_dim]
        return np.concatenate([left_vel, right_vel], axis=-1).reshape(-1)
    raise ValueError(
        f"Unknown feature '{feature}' — pick one of "
        "'flatten', 'positions', 'positions_delta', 'velocity'."
    )


# ---------------------------------------------------------------------------
# State image extraction (embedded thumbnails for click-to-view)
# ---------------------------------------------------------------------------


_DEFAULT_IMAGE_CANDIDATES = (
    "observations.images.front_img_1",
    "observations.images.front_1",
    "observations.images.base_0_rgb",
    "base_0_rgb",
)


_EMBODIMENT_CLASSES: dict[str, "type"] = {}
_INTRINSICS_BY_VENDOR: dict[str, np.ndarray] = {}


# Hard fallback for eva episodes that have empty/NaN intrinsics or extrinsics
# in their zarr.attrs. Values copied verbatim from
#   /storage/project/r-dxu345-0/shared/arc_tests/2025-12-30-01-47-21-674000/zarr.json
# (a canonical eva_bimanual fold_clothes episode collected for arc-tok
# testing). Same 266.5 / 320 / 240 K as ARIA_INTRINSICS + the class-constant
# Eva.EXTRINSICS — recording them here explicitly so the viz overlay stays
# correct even if future refactors move ARIA_INTRINSICS or Eva.EXTRINSICS.
_EVA_FALLBACK_INTRINSICS = np.array(
    [
        [266.50860444, 0.0, 320.0, 0.0],
        [0.0, 266.50860444, 240.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_EVA_FALLBACK_EXTRINSICS = {
    "left": np.array(
        [
            [0.01329544, -0.71757193, 0.69635749, -0.04409191],
            [-0.99959782, -0.02698416, -0.00872107, -0.23221381],
            [0.02504862, -0.69596148, -0.7176421, 0.57323278],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "right": np.array(
        [
            [-0.04733948, -0.76631195, 0.64072222, -0.01998031],
            [-0.9983006, 0.05811952, -0.00424732, 0.32539554],
            [-0.0339837, -0.63983444, -0.76776103, 0.64809634],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}


def _is_eva(embodiment_name: str | None) -> bool:
    return bool(embodiment_name) and str(embodiment_name).lower().startswith("eva")


def _get_embodiment_class(embodiment_name: str):
    """Return Eva / Human / etc. for the embodiment name — cached, lazy import
    so this module still loads in envs without projectaria_tools."""
    global _EMBODIMENT_CLASSES
    if not _EMBODIMENT_CLASSES:
        try:
            from egomimic.rldb.embodiment.eva import Eva
            from egomimic.rldb.embodiment.human import Human

            _EMBODIMENT_CLASSES = {
                "eva_bimanual": Eva,
                "eva_right_arm": Eva,
                "eva_left_arm": Eva,
                "human_bimanual": Human,
                "human_right_arm": Human,
                "human_left_arm": Human,
            }
        except Exception:
            return None
    return _EMBODIMENT_CLASSES.get(str(embodiment_name).lower())


def _get_vendor_intrinsics(vendor_hint: str | None) -> np.ndarray | None:
    """Return the correct camera K constant for the vendor prefix of the raw
    (pre-collapse) embodiment string. mecka/aria/scale/lightwheel each have
    different physical front cameras so the K matrices differ; using the
    right one is the difference between the GT overlay being "attached to
    the wrist" vs floating around.
    """
    global _INTRINSICS_BY_VENDOR
    if not _INTRINSICS_BY_VENDOR:
        try:
            from egomimic.rldb.embodiment.human import (
                ARIA_INTRINSICS,
                LIGHTWHEEL_INTRINSICS,
                MECKA_INTRINSICS,
                SCALE_INTRINSICS,
            )

            _INTRINSICS_BY_VENDOR = {
                "aria": ARIA_INTRINSICS,
                "mecka": MECKA_INTRINSICS,
                "scale": SCALE_INTRINSICS,
                "lightwheel": LIGHTWHEEL_INTRINSICS,
            }
        except Exception:
            return None
    if not vendor_hint:
        return None
    hint = vendor_hint.lower()
    for prefix, K in _INTRINSICS_BY_VENDOR.items():
        if hint.startswith(prefix):
            return np.asarray(K, dtype=np.float64)
    return None


def _load_hwc_uint8_image(value) -> np.ndarray | None:
    """Normalize a torch/np image tensor to (H, W, 3) uint8. Returns None on
    unsupported shapes / no channels found.

    IMPORTANT alignment guarantee: when the input is a temporal stack (4D)
    we always pick index 0 so the returned frame corresponds to the same
    timestep as the first action in the sample's action chunk. Both are
    read from the zarr at the same start_idx (see
    ``ZarrDataset.__getitem__``: image reads use ``read_interval=(idx, ...)``
    and action reads use the same ``idx`` with an extra horizon). The only
    place this alignment can silently drift is when a stacked image is
    handed to us and we forget to slice — hence the explicit slice below
    with a comment.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 4:
        # Stack of frames from a horizoned image key. Frame 0 is at the
        # same zarr timestep as chunk[0] — anything else would misalign the
        # overlay against the projected first action.
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _first_present(sample: dict, keys) -> object | None:
    for k in keys:
        if k and k in sample:
            return sample[k]
    return None


def _encode_pil_data_uri(pil, max_side: int, jpeg_quality: int) -> str:
    from PIL import Image

    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        pil = pil.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
        )
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_overlay_ctx(
    sample: dict,
    chunk_14d: np.ndarray | None,
    embodiment_name: str,
    vendor_hint: str | None,
    image_max_side: int,
    orig_img_hw: tuple[int, int] | None = None,
) -> dict | None:
    """Build the per-sample context the client-side JS needs to draw an
    overlay of any (D,M) arc token or (H) time slice on this sample's raw
    front image:
      - ``actions``: (T, 12/14) float32 flat list — the full action chunk,
        pre-transformed into the FRONT-CAMERA frame so JS can project with
        the front-cam K directly. For human (aria/mecka) the pipeline already
        produces head/front-cam-frame poses, so actions pass through
        unchanged. For eva the pipeline produces per-arm wrist-camera-frame
        poses; here we multiply each arm's xyz by ``Eva.EXTRINSICS[arm]``
        (which under the repo's ``T_cam_base`` convention takes wrist-cam
        points into the base/front-cam frame — the eva zarr stores no other
        extrinsics so front-cam ≡ base).
      - ``K``: (3, 4) intrinsics scaled to the resized image dimensions so
              projection outputs pixel coords in the JPEG that ships to
              the browser (no client-side scaling required).
      - ``T``, ``D`` : action tensor dimensions.
      - ``img_h``, ``img_w``: dimensions of the JPEG shipped to the browser.
    Returns ``None`` if data is incomplete (no chunk, no intrinsics)."""
    if chunk_14d is None:
        return None
    K_raw = sample.get("intrinsics") if isinstance(sample, dict) else None
    K = None
    if K_raw is not None:
        K = (
            K_raw.detach().cpu().numpy()
            if isinstance(K_raw, torch.Tensor)
            else np.asarray(K_raw)
        )
        if np.any(np.isnan(K)):
            K = None
    if K is None:
        K = _get_vendor_intrinsics(vendor_hint)
    if K is None and _is_eva(embodiment_name):
        # Eva-only fallback: sourced from a canonical eva fold_clothes zarr,
        # so we always project with the exact same K the data was collected
        # against (even if the per-episode intrinsics field was empty).
        K = _EVA_FALLBACK_INTRINSICS.copy()
    if K is None:
        emb_cls = _get_embodiment_class(embodiment_name)
        if emb_cls is not None:
            fallback = getattr(emb_cls, "INTRINSICS", None)
            if fallback is not None:
                K = np.asarray(fallback, dtype=np.float64)
    if K is None:
        return None
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 4):
        return None

    # Compute resize factor: PIL keeps aspect ratio and downsizes so that
    # max(w, h) == image_max_side. If neither exceeds it, no resize.
    if orig_img_hw is None:
        return None
    orig_h, orig_w = orig_img_hw
    scale = 1.0
    if max(orig_h, orig_w) > image_max_side:
        scale = image_max_side / float(max(orig_h, orig_w))
    resized_w = max(1, int(orig_w * scale))
    resized_h = max(1, int(orig_h * scale))

    # Scale first two rows of K (fx*s, cx*s, fy*s, cy*s) so projected pixels
    # land in the resized image's coordinate system directly.
    K_scaled = K.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale

    # Actions are assumed to be in the FRONT-CAMERA frame at this point:
    #   - human_bimanual: transform pipeline maps into head=front-cam frame
    #     (target_world = obs_head_pose)
    #   - eva_bimanual with mode=cartesian_world: transform pipeline leaves
    #     poses in the raw base frame (which coincides with the front cam on
    #     eva — the zarr stores no separate front-cam extrinsic, and eva
    #     intrinsics/extrinsics live as global class constants on Eva)
    # Under both conditions JS projects xyz through the front K directly.
    chunk = np.asarray(chunk_14d, dtype=np.float64)

    return {
        "actions": chunk.astype(np.float32).reshape(-1).tolist(),
        "T": int(chunk.shape[0]),
        "D": int(chunk.shape[1]),
        "K": K_scaled.reshape(-1).tolist(),
        "img_h": resized_h,
        "img_w": resized_w,
    }


def _extract_state_image_uri(
    sample: dict,
    chunk_14d: np.ndarray | None,
    embodiment_name: str,
    image_key: str | None,
    image_max_side: int,
    jpeg_quality: int,
    overlay_gt_chunk: bool = True,
    vendor_hint: str | None = None,
) -> str | None:
    """Return a data:image/jpeg;base64,... URI of the front camera image with
    the GT action chunk overlaid in green.

    Uses the exact same call path as the ``viz_gt_preds`` overlay that the
    ``hydra_configs/evaluator/viz/{cartesian,cotrain_lang}.yaml`` configs
    invoke for both eva and aria:

        <EmbodimentCls>.viz(image, actions_cartesian, mode='traj',
                            color='Greens', intrinsics=K)

    The embodiment class dispatches to ``_viz_traj`` which calls
    ``draw_actions(type='xyz', ...)`` and projects the actions with the given
    K. No extrinsics un-transform — for eva that's the pattern the eval viz
    configs use across every task run in this repo.
    """
    from PIL import Image

    K_raw = sample.get("intrinsics") if isinstance(sample, dict) else None
    K = None
    if K_raw is not None:
        K = (
            K_raw.detach().cpu().numpy()
            if isinstance(K_raw, torch.Tensor)
            else np.asarray(K_raw)
        )
        if np.any(np.isnan(K)):
            K = None

    emb_cls = _get_embodiment_class(embodiment_name)
    # Legacy caches (mecka pre-collapse, some aria episodes) have empty
    # intrinsics in zarr.attrs, which surfaces as NaN in the sample. Fall
    # back per raw-vendor prefix (mecka/aria/scale/lightwheel) — each vendor
    # has its own physical camera and K matrix, so using ARIA_INTRINSICS for
    # a mecka episode gives the "trajectory floating off the wrist" bug the
    # user reported. Only if the vendor hint doesn't resolve do we fall back
    # to the embodiment class's default INTRINSICS.
    if K is None:
        K = _get_vendor_intrinsics(vendor_hint)
    if K is None and _is_eva(embodiment_name):
        K = _EVA_FALLBACK_INTRINSICS.copy()
    if K is None and emb_cls is not None:
        fallback = getattr(emb_cls, "INTRINSICS", None)
        if fallback is not None:
            K = np.asarray(fallback, dtype=np.float64)

    keys = [image_key] if image_key else list(_DEFAULT_IMAGE_CANDIDATES)
    front_raw = _first_present(sample, keys)
    img = _load_hwc_uint8_image(front_raw) if front_raw is not None else None
    if img is None:
        return None

    if overlay_gt_chunk and chunk_14d is not None and K is not None:
        # Draw the interpolated arc chunk (post joint-arc-length resampling)
        # instead of the raw 100-frame time-indexed window, so the overlay's
        # horizon corresponds to the size of the action chunk, not the whole
        # dataset window. Uses a fixed representative (D, M) so the same
        # overlay works across every sweep tab (per-sweep would require 24x
        # image storage).
        chunk_for_draw = _interpolated_arc_chunk_for_overlay(
            np.asarray(chunk_14d, dtype=np.float64),
            joint_D_m=_OVERLAY_JOINT_D_M,
            M=_OVERLAY_M,
        )
        if chunk_for_draw is None:
            # arc chunk is degenerate (zero motion) — fall back to raw so the
            # user still gets some overlay signal.
            chunk_for_draw = np.asarray(chunk_14d, dtype=np.float64)
        if emb_cls is not None:
            try:
                img = emb_cls.viz(
                    img,
                    chunk_for_draw,
                    mode="traj",
                    intrinsics=K,
                    color="Greens",
                    alpha=1.0,
                )
            except Exception as e:
                _log_overlay_error(e)

    return _encode_pil_data_uri(Image.fromarray(img), image_max_side, jpeg_quality)


# Fixed (D, M) for the state-image overlay across all sweep tabs. Storing 24
# different overlays per sample would balloon the HTML by 24x — instead we
# pick a middle-of-grid sweep point so the overlay is a reasonable
# approximation of an arc chunk in the visualized regime.
_OVERLAY_JOINT_D_M = 0.40  # 40cm joint arc length
_OVERLAY_M = 15


def _interpolated_arc_chunk_for_overlay(
    chunk_14d: np.ndarray, joint_D_m: float, M: int
) -> np.ndarray | None:
    """Resample chunk_14d to M waypoints uniform in JOINT arc length across the
    first stretch where cum_L(t) + cum_R(t) reaches ``joint_D_m`` (clamped to
    the full window if it never reaches). Preserves the 14-D layout with
    grippers/ypr sampled at the same fractional timesteps as xyz.

    This is the "post-interpolation, post-tokenization" arc chunk — what the
    tokenizer would emit at this (D, M), suitable to draw as a green trail
    representing the arc chunk instead of the raw 100-frame series.
    """
    chunk_14d = np.asarray(chunk_14d, dtype=np.float64)
    if chunk_14d.ndim != 2 or chunk_14d.shape[0] < 2 or chunk_14d.shape[1] != 14:
        return None
    T, D = chunk_14d.shape
    left_xyz = chunk_14d[:, 0:3]
    right_xyz = chunk_14d[:, 7:10]
    left_step = np.linalg.norm(np.diff(left_xyz, axis=0), axis=1)
    right_step = np.linalg.norm(np.diff(right_xyz, axis=0), axis=1)
    cum_L = np.concatenate([[0.0], np.cumsum(left_step)])
    cum_R = np.concatenate([[0.0], np.cumsum(right_step)])
    joint_cum = cum_L + cum_R
    J_max = float(joint_cum[-1])
    if J_max < 1e-6:
        return None
    s_end = min(float(joint_D_m), J_max)
    targets = np.linspace(0.0, s_end, int(M))
    t_grid = np.arange(T, dtype=np.float64)
    t_k = np.interp(targets, joint_cum, t_grid)
    # Interp each column at the fractional timesteps (linear works fine for
    # the drawing use case — SLERP-quality rotation not needed for xyz dots).
    out = np.stack([np.interp(t_k, t_grid, chunk_14d[:, d]) for d in range(D)], axis=-1)
    return out


_OVERLAY_ERRORS_REPORTED = 0


def _log_overlay_error(err: Exception) -> None:
    global _OVERLAY_ERRORS_REPORTED
    if _OVERLAY_ERRORS_REPORTED < 3:
        print(f"  [overlay] projection failed: {type(err).__name__}: {err}")
        _OVERLAY_ERRORS_REPORTED += 1


# ---------------------------------------------------------------------------
# Per-dataset collection
# ---------------------------------------------------------------------------


def _collect_tokens_from_dataset(
    dataset,
    embodiment_name: str,
    split_name: str,
    tokenizer: BimanualArcLengthTokenizer,
    action_key: str,
    feature: str,
    max_samples: int,
    shuffle: bool,
    include_zero_tokens: bool,
    seed: int,
    include_state_image: bool,
    image_key: str | None,
    image_max_side: int,
    jpeg_quality: int,
) -> tuple[list[np.ndarray], list[dict]]:
    n = len(dataset)
    if n == 0:
        print(f"  [{split_name}] {embodiment_name}: empty dataset, skipping")
        return [], []

    idxs = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idxs)
    if max_samples > 0:
        idxs = idxs[:max_samples]

    arm_dim_tok = tokenizer.arc_arm_dim
    velocity_dim = tokenizer.velocity_dim

    feats: list[np.ndarray] = []
    meta: list[dict] = []
    skipped_missing = 0
    skipped_shape = 0
    skipped_invalid = 0
    skipped_zero = 0
    skipped_error = 0

    for idx in idxs:
        try:
            sample = dataset[idx]
        except Exception as e:  # bad episode / IO / etc.
            skipped_error += 1
            if skipped_error <= 3:
                print(
                    f"  [{split_name}] {embodiment_name} idx={idx} sample "
                    f"failed: {type(e).__name__}: {e}"
                )
            continue

        if action_key not in sample:
            skipped_missing += 1
            continue

        chunk = sample[action_key]
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.detach().cpu().numpy()
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            skipped_shape += 1
            continue
        # Human / aria data lacks gripper columns and arrives as (T, 12) — pad
        # zero grippers at positions 6 and 13 to hit the canonical (T, 14)
        # bimanual cartesian layout the arc tokenizer expects.
        if chunk.shape[-1] == 12:
            zeros = np.zeros((chunk.shape[0], 1), dtype=chunk.dtype)
            chunk = np.concatenate((chunk[:, :6], zeros, chunk[:, 6:], zeros), axis=-1)
        if chunk.shape[-1] != BIMANUAL_CARTESIAN_DIM:
            skipped_shape += 1
            continue

        arc = tokenizer.tokenize(chunk)
        if np.any(np.abs(arc) >= INVALID_POSE_THRESHOLD):
            skipped_invalid += 1
            continue

        left_vel = arc[:, ARM_DIM : ARM_DIM + velocity_dim]
        right_vel = arc[:, arm_dim_tok + ARM_DIM : arm_dim_tok + ARM_DIM + velocity_dim]
        is_zero = not np.any(left_vel) and not np.any(right_vel)
        if is_zero and not include_zero_tokens:
            skipped_zero += 1
            continue

        f = _extract_feature(arc, arm_dim_tok, velocity_dim, feature)
        feats.append(f)
        rec = {
            "embodiment": embodiment_name,
            "split": split_name,
            "sample_idx": int(idx),
            "is_zero_token": bool(is_zero),
        }
        if include_state_image:
            # `chunk` at this point is the (T, 14) canonical layout — for aria
            # it's the 12D pipeline output already zero-gripper-padded above.
            rec["_img"] = (
                _extract_state_image_uri(
                    sample,
                    chunk_14d=chunk,
                    embodiment_name=embodiment_name,
                    image_key=image_key,
                    image_max_side=image_max_side,
                    jpeg_quality=jpeg_quality,
                    overlay_gt_chunk=True,
                )
                or ""
            )
        meta.append(rec)

    print(
        f"  [{split_name}] {embodiment_name}: kept={len(feats)}/{len(idxs)} "
        f"(missing={skipped_missing} shape={skipped_shape} "
        f"invalid={skipped_invalid} zero={skipped_zero} err={skipped_error})"
    )
    return feats, meta


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


def _fit_embedder(
    features: np.ndarray, method: str, n_components: int, params: dict
) -> np.ndarray:
    method = method.lower()
    params = dict(params or {})
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, **params).fit_transform(features)
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_components, **params).fit_transform(features)
    if method == "umap":
        try:
            import umap  # umap-learn
        except ImportError as e:
            raise ImportError(
                "method='umap' requires umap-learn. Install with "
                "`pip install umap-learn`."
            ) from e
        return umap.UMAP(n_components=n_components, **params).fit_transform(features)
    raise ValueError(
        f"Unknown embedder method '{method}' — pick 'pca', 'tsne', or 'umap'."
    )


# ---------------------------------------------------------------------------
# Plotly HTML
# ---------------------------------------------------------------------------


def _plot_html(
    embed: np.ndarray,
    meta: list[dict],
    n_components: int,
    title: str,
    method: str,
    output_html: str,
) -> None:
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError(
            "plotly is required to write the HTML output. Install with "
            "`pip install plotly`."
        ) from e
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for the plotly dataframe path. Install with "
            "`pip install pandas`."
        ) from e

    df = pd.DataFrame(meta)
    coord_cols = ["x", "y", "z"][:n_components]
    for i, c in enumerate(coord_cols):
        df[c] = embed[:, i]

    label = f"{title} — {method}, {n_components}D  (N={len(df)})"
    hover_cols = ["embodiment", "split", "sample_idx", "is_zero_token"]

    has_images = "_img" in df.columns and df["_img"].astype(bool).any()
    if has_images:
        custom_cols = ["_img", "embodiment", "sample_idx", "split", "is_zero_token"]
    else:
        custom_cols = None

    if n_components == 2:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="embodiment",
            title=label,
            hover_data=hover_cols,
            custom_data=custom_cols,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=5))
    elif n_components == 3:
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="embodiment",
            title=label,
            hover_data=hover_cols,
            custom_data=custom_cols,
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=3))
    else:
        raise ValueError(
            f"n_components must be 2 or 3 for HTML plotting, got {n_components}"
        )

    out_dir = os.path.dirname(os.path.abspath(output_html))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html)
    if has_images:
        _inject_click_image_panel(output_html)
    print(f"Wrote embedding HTML: {output_html}  (N={len(df)})")


def _inject_click_image_panel(output_html: str) -> None:
    """Inject a fixed panel + a plotly_click listener into the HTML so users can
    click any point and see that arc chunk's state image (customdata[0]).
    """
    panel_and_script = """
<div id="egoverse-img-panel" style="position:fixed;top:12px;right:12px;width:340px;background:#111;color:#eee;border:1px solid #444;padding:8px;border-radius:8px;z-index:9999;font-family:sans-serif;box-shadow:0 2px 10px rgba(0,0,0,0.5);">
  <div style="font-size:12px;opacity:0.7;margin-bottom:6px;">click a point to see its state image</div>
  <div id="egoverse-img-body" style="min-height:60px;"></div>
</div>
<script>
(function(){
  function attach(){
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd.on) { setTimeout(attach, 200); return; }
    gd.on('plotly_click', function(evt){
      if (!evt || !evt.points || !evt.points.length) return;
      var pt = evt.points[0];
      var cd = pt.customdata || [];
      var img = cd[0] || '';
      var emb = cd[1] || '';
      var idx = (cd[2] !== undefined) ? cd[2] : '';
      var split = cd[3] || '';
      var isz = cd[4];
      var body = document.getElementById('egoverse-img-body');
      var label = emb + '  idx=' + idx + '  [' + split + ']' + (isz ? '  (zero)' : '');
      if (img) {
        body.innerHTML = '<img src="' + img + '" style="width:100%;background:#000;display:block;border-radius:4px;"/>' +
                         '<div style="font-size:11px;margin-top:6px;">' + label + '</div>';
      } else {
        body.innerHTML = '<em>no image cached for this point</em>' +
                         '<div style="font-size:11px;margin-top:6px;opacity:0.7;">' + label + '</div>';
      }
    });
  }
  attach();
})();
</script>
""".strip()
    with open(output_html, "r") as f:
        html = f.read()
    if "</body>" in html:
        html = html.replace("</body>", panel_and_script + "\n</body>")
    else:
        html += panel_and_script
    with open(output_html, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path="../hydra_configs",
    config_name="arc_embedding.yaml",
)
def main(cfg: DictConfig) -> None:
    load_env()

    embedder_cfg = cfg.embedder
    action_key = str(cfg.get("action_key", "actions_cartesian"))
    split = str(cfg.get("split", "train"))
    feature = str(embedder_cfg.feature)
    max_samples = int(embedder_cfg.max_samples_per_dataset)
    shuffle = bool(embedder_cfg.shuffle)
    seed = int(embedder_cfg.seed)
    include_zero_tokens = bool(embedder_cfg.include_zero_tokens)
    include_state_image = bool(
        OmegaConf.select(embedder_cfg, "include_state_image", default=True)
    )
    image_key_cfg = OmegaConf.select(embedder_cfg, "image_key", default=None)
    image_key = str(image_key_cfg) if image_key_cfg else None
    image_max_side = int(OmegaConf.select(embedder_cfg, "image_max_side", default=224))
    jpeg_quality = int(OmegaConf.select(embedder_cfg, "jpeg_quality", default=60))

    tokenizer = _build_tokenizer(embedder_cfg.tokenizer)
    print(
        f"[tokenizer] min_distance_unit={tokenizer.config.min_distance_unit} "
        f"M={tokenizer.M} mode={tokenizer.config.mode} "
        f"velocity_dim={tokenizer.velocity_dim} arc_dim={tokenizer.arc_dim}"
    )

    splits: list[tuple[str, DictConfig]] = []
    if split in ("train", "both"):
        splits.append(("train", cfg.data.train_datasets))
    if split in ("valid", "both"):
        splits.append(("valid", cfg.data.valid_datasets))
    if not splits:
        raise ValueError(f"split must be 'train' | 'valid' | 'both', got {split!r}")

    all_feats: list[np.ndarray] = []
    all_meta: list[dict] = []

    for split_name, ds_cfgs in splits:
        for emb_name, ds_cfg in ds_cfgs.items():
            print(f"\n[{split_name}] instantiating {emb_name}")
            ds = hydra.utils.instantiate(ds_cfg)
            feats, meta = _collect_tokens_from_dataset(
                dataset=ds,
                embodiment_name=str(emb_name),
                split_name=split_name,
                tokenizer=tokenizer,
                action_key=action_key,
                feature=feature,
                max_samples=max_samples,
                shuffle=shuffle,
                include_zero_tokens=include_zero_tokens,
                seed=seed,
                include_state_image=include_state_image,
                image_key=image_key,
                image_max_side=image_max_side,
                jpeg_quality=jpeg_quality,
            )
            all_feats.extend(feats)
            all_meta.extend(meta)

    if not all_feats:
        raise RuntimeError(
            "No arc tokens collected. Check dataset paths, action_key, and "
            "filter settings (include_zero_tokens, invalid poses)."
        )

    X = np.stack(all_feats, axis=0).astype(np.float32)
    print(
        f"\nFitting {embedder_cfg.method} on X.shape={X.shape} -> "
        f"{int(embedder_cfg.n_components)}D"
    )
    params_cfg = OmegaConf.select(embedder_cfg, "params", default={})
    params = OmegaConf.to_container(params_cfg, resolve=True) if params_cfg else {}
    embed = _fit_embedder(
        X,
        method=str(embedder_cfg.method),
        n_components=int(embedder_cfg.n_components),
        params=params or {},
    )

    _plot_html(
        embed=embed,
        meta=all_meta,
        n_components=int(embedder_cfg.n_components),
        title=str(cfg.get("title", "Arc-token embedding")),
        method=str(embedder_cfg.method),
        output_html=str(cfg.get("output_html", "arc_embedding.html")),
    )


if __name__ == "__main__":
    main()
