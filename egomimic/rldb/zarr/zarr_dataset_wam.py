"""
WAM dataset: loads camera keys as a short frame *clip* for the world model.

The base ``ZarrDataset`` (used by VLA/Pi training) reads a single frame per
camera key. WAM needs a contiguous clip ``[t, t+1, ..., t+horizon-1]`` (the Wan
VAE compresses 4x temporally, so ~5 frames -> 2 latent frames: a conditioning
frame + a future frame). This subclass **extends** the base — it only overrides
``__getitem__`` to decode a *windowed* camera read (a camera key carrying a
``horizon``) into a clip ``(T, C, H, W)``; proprio/action-chunk windowing,
transforms, bounds and normalization are all inherited unchanged. The base
loader is left untouched.

Wire it via ``S3WamEpisodeResolver`` (``_dataset_class = ZarrWamDataset``) and a
keymap whose camera key has a ``horizon`` (see ``Human.get_keymap(cam_horizon=...)``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import simplejpeg
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    S3EpisodeResolver,
    ZarrDataset,
    ZarrEpisode,
    get_fallback_idx,
    logger,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

__all__ = [
    "ZarrWamDataset",
    "WamMultiDataset",
    "S3WamEpisodeResolver",
    "LocalWamEpisodeResolver",
    "ZarrEpisode",
]


class WamMultiDataset(MultiDataset):
    """MultiDataset for the WAM pipeline. Identical to the base except it skips
    quantile bounds rejection (WAM action/state are raw frame-aligned ee-poses
    with no bound calibration) — a one-method override, base left untouched."""

    def _check_bounds(self, *args, **kwargs):
        return None


class ZarrWamDataset(ZarrDataset):
    """ZarrDataset that decodes windowed camera reads into a clip (T, C, H, W)."""

    _QUAT_ZERO_EPS = 1e-6

    @staticmethod
    def _decode_one(buf) -> np.ndarray:
        d = simplejpeg.decode_jpeg(buf, colorspace="RGB")
        return np.transpose(d, (2, 0, 1)) / 255.0  # (C, H, W) in [0,1]

    @classmethod
    def _validate_arrays(cls, data: dict) -> str | None:
        """Scan loaded arrays for NaN/Inf and zero-quaternion rows.

        Why: ``WamMultiDataset._check_bounds`` is a no-op for WAM, so the
        MultiDataset-level NaN/quantile check is skipped. A single bad row —
        NaN in a state chunk, or an all-zero quaternion in ``obs_head_pose`` /
        ``{left,right}_extrinsics_pose`` — crashes ``ActionChunkCoordinateFrameTransform``
        because ``SE3.from_matrix(_xyzwxyz_to_matrix(...))`` on a degenerate quat
        yields a NaN matrix whose ``.inverse()`` propagates NaN into every
        subsequent chunk. In a DDP dataloader worker that failure can wedge the
        rank silently (worker stalls or spins in a retry loop), causing GPU util
        to drop to 0% on that rank while the other ranks time out on the next
        NCCL collective. Catching these here lets us log the defect and resample
        a fresh idx cleanly.

        Returns a one-line reason string on the first defect, or None if clean.
        """
        for k, v in data.items():
            if not isinstance(v, np.ndarray):
                continue
            if v.dtype.kind not in "fc":
                continue
            if not np.all(np.isfinite(v)):
                return f"NaN/Inf in key={k}"
            # xyzwxyz layout: last dim 7 means [x, y, z, qw, qx, qy, qz].
            # All last-dim-7 arrays in the WAM keymap are poses.
            if v.shape and v.shape[-1] == 7:
                quat = v[..., 3:7].reshape(-1, 4)
                norms = np.linalg.norm(quat, axis=-1)
                if np.any(norms < cls._QUAT_ZERO_EPS):
                    return (
                        f"Zero-quat in key={k} " f"(min |q|={float(norms.min()):.2e})"
                    )
        return None

    def __getitem__(self, idx, _fallback_origin=None, _attempts=None):
        origin = _fallback_origin if _fallback_origin is not None else idx
        attempts = _attempts

        def _next(reason: str, key: str = "") -> int:
            nonlocal attempts
            next_idx, attempts = get_fallback_idx(
                idx=idx,
                candidates=range(self.total_frames),
                _attempts=attempts,
                max_attempts=self.total_frames,
                exhausted_error=(
                    f"Entire episode bad (no valid indices): "
                    f"ep={Path(self.episode_path).name}"
                ),
            )
            logger.warning(
                f"{reason} ep={Path(self.episode_path).name} frame={idx}"
                + (f" key={key}" if key else "")
                + f" | attempt {attempts}, trying random idx {next_idx}"
            )
            return next_idx

        while True:
            data = {}
            retry = False
            for k in self.key_map:
                zarr_key = self.key_map[k]["zarr_key"]
                key_type = self.key_map[k].get("key_type", None)
                horizon = self.key_map[k].get("horizon", None)

                if key_type == "annotation_keys":
                    data[k] = self._annotation_text_for_frame(idx)
                    continue

                if horizon is not None:
                    end_idx = self._chunk_end_idx(idx, horizon, key_type)
                    read_interval = (idx, end_idx)
                else:
                    read_interval = (idx, None)
                raw_data = self.episode_reader.read({zarr_key: read_interval})
                self._pad_sequences(raw_data, horizon)
                data[k] = raw_data[zarr_key]

                if zarr_key in self._image_keys:
                    try:
                        if horizon is not None:
                            # windowed camera read -> clip (T, C, H, W)
                            data[k] = np.stack([self._decode_one(b) for b in data[k]])
                        else:
                            data[k] = self._decode_one(data[k])  # (C, H, W)
                    except Exception:
                        idx = _next("JPEG decode failed", key=k)
                        retry = True
                        break
                elif zarr_key in self._json_keys:
                    if isinstance(data[k], np.ndarray):
                        data[k] = [self._decode_json_entry(v) for v in data[k]]
                    else:
                        data[k] = self._decode_json_entry(data[k])
            if retry:
                continue

            # --- Per-episode camera calibration into the batch (BEFORE
            # transforms) so downstream steps that operate in camera frame —
            # e.g. ``Eva.get_wam_transform_list`` — can pull per-episode
            # ``{left,right}_extrinsics_pose`` instead of hardcoding
            # ``Eva.EXTRINSICS``. Intrinsics mirror the base ZarrDataset plumbing
            # so viz/projection picks up the zarr-stored K.
            extr = self.episode_reader.extrinsics
            if isinstance(extr, dict):
                for arm_key, se3 in extr.items():
                    arr = np.asarray(se3, dtype=np.float32)
                    if arr.shape == (4, 4):
                        xyzq = _matrix_to_xyzwxyz(arr[None, :])[0].astype(np.float32)
                        data[f"{arm_key}_extrinsics_pose"] = xyzq

            # Pre-transform validation: catch NaN/Inf and zero-quats before
            # they blow up the SE3 coord-frame transform (see _validate_arrays).
            bad = self._validate_arrays(data)
            if bad is not None:
                idx = _next(bad)
                continue

            if self.transform:
                try:
                    for transform in self.transform or []:
                        data = transform.transform(data)
                except Exception as e:
                    idx = _next(f"Transform failed ({type(e).__name__}: {e})")
                    continue

            # Post-transform validation: transforms (interpolation, SE3
            # inverse, unwrap) can introduce NaN on near-degenerate inputs
            # that pass the pre-check.
            bad = self._validate_arrays(data)
            if bad is not None:
                idx = _next(f"post-transform: {bad}")
                continue

            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(torch.float32)

            data["embodiment"] = get_embodiment_id(self.embodiment)
            ep_name = Path(self.episode_path).name
            data["episode_hash"] = (
                ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            )

            K = self.episode_reader.intrinsics
            if isinstance(K, dict):
                K = next(
                    (v for k, v in K.items() if "front" in str(k).lower()),
                    next(iter(K.values()), None) if K else None,
                )
            if K is not None:
                K = np.asarray(K, dtype=np.float32)
                if K.shape == (3, 3):
                    K = np.concatenate([K, np.zeros((3, 1), dtype=np.float32)], axis=1)
                if K.shape != (3, 4):
                    K = np.full((3, 4), np.nan, dtype=np.float32)
            else:
                K = np.full((3, 4), np.nan, dtype=np.float32)
            data["intrinsics"] = torch.from_numpy(np.ascontiguousarray(K))
            _ = origin
            return data


class S3WamEpisodeResolver(S3EpisodeResolver):
    """S3 resolver that builds ZarrWamDataset leaves (clip-loading camera keys)."""

    _dataset_class = ZarrWamDataset


class LocalWamEpisodeResolver(LocalEpisodeResolver):
    _dataset_class = ZarrWamDataset
