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

    @staticmethod
    def _decode_one(buf) -> np.ndarray:
        d = simplejpeg.decode_jpeg(buf, colorspace="RGB")
        return np.transpose(d, (2, 0, 1)) / 255.0  # (C, H, W) in [0,1]

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

            if self.transform:
                for transform in self.transform or []:
                    data = transform.transform(data)

            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(torch.float32)

            data["embodiment"] = get_embodiment_id(self.embodiment)
            ep_name = Path(self.episode_path).name
            data["episode_hash"] = (
                ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            )
            _ = origin
            return data


class S3WamEpisodeResolver(S3EpisodeResolver):
    """S3 resolver that builds ZarrWamDataset leaves (clip-loading camera keys)."""

    _dataset_class = ZarrWamDataset


class LocalWamEpisodeResolver(LocalEpisodeResolver):
    _dataset_class = ZarrWamDataset
