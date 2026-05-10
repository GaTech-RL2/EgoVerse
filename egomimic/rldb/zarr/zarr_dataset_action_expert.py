"""
Action-expert dataset with annotation-anchored timestep sampling.

Sampling scheme:
  1. Sample a timestep t uniformly over the full episode (v_t).
  2. Use span annotations to find the enclosing annotation's
     BOS (begin-of-segment) and EOS (end-of-segment) frame indices.
  3. Load BOS, t, and EOS observations (image + language).
  4. Load actions a_t : a_EOS as the action-expert supervision target.
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
)

__all__ = [
    "ZarrActionExpertDataset",
    "S3ActionExpertEpisodeResolver",
    "LocalActionExpertEpisodeResolver",
    "MultiDataset",
    "ZarrEpisode",
]

_OBS_KEY_TYPES = {"camera_keys", "proprio_keys"}


def _obs_key_triple(key: str) -> tuple[str, str, str]:
    """Return (bos_key, t_key, eos_key) for an observation key."""
    if key.endswith("_1"):
        base = key[:-2]
        return key, base + "_t", base + "_T"
    return key + "_1", key + "_t", key + "_T"


class ZarrActionExpertDataset(ZarrDataset):
    """ZarrDataset subclass for action-expert episodes."""

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
        fixed_horizon: int | None = None,
    ):
        self.annotation_map = {}
        self.fixed_horizon = fixed_horizon
        super().__init__(Episode_path, key_map, transform_list)

    # --- episode setup -----------------------------------------------------

    def _load_annotation_map(self) -> dict[int, int]:
        """Build {frame_idx -> annotation_index} for every frame inside an annotation span."""
        raw = self.episode_reader._store["annotations"][:]
        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        for i, ann in enumerate(self._annotations):
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            for idx in range(start_idx, end_idx + 1):
                self.annotation_map[idx] = i
        return self.annotation_map

    def init_episode(self):
        super().init_episode()
        # Save true episode length before overriding total_frames with annotated-only count.
        self._episode_total_frames = self.metadata["total_frames"]
        self.annotation_map = self._load_annotation_map()
        if not self.annotation_map:
            raise ValueError("Annotation map is required for ZarrActionExpertDataset")
        self._valid_frame_indices = sorted(self.annotation_map.keys())
        self.total_frames = len(self._valid_frame_indices)

    # --- per-key loaders ---------------------------------------------------

    def _load_obs_at(self, zarr_key: str, frame_idx: int):
        """Read a single obs key at one frame, decoding JPEG/JSON as needed."""
        val = self.episode_reader.read({zarr_key: (frame_idx, None)})[zarr_key]
        if zarr_key in self._image_keys:
            decoded = simplejpeg.decode_jpeg(val, colorspace="RGB")
            return np.transpose(decoded, (2, 0, 1)) / 255.0
        if zarr_key in self._json_keys:
            return self._decode_json_entry(val)
        return val

    def _load_actions(
        self, zarr_key: str, t: int, eos: int, horizon: int | None
    ) -> np.ndarray:
        """Load the t..EOS action chunk, padded to ``horizon``.

        If ``self.fixed_horizon`` is an int H, that overrides ``horizon``: the
        read is clamped to EOS and zero-padded out to length H so a sample
        never includes actions from the next segment.
        """
        episode_end = self._episode_total_frames

        if self.fixed_horizon is not None:
            H = self.fixed_horizon
            end = min(t + H, eos + 1, episode_end)
            arr = self.episode_reader.read({zarr_key: (t, end)})[zarr_key]
            if arr.shape[0] < H:
                pad = np.zeros((H - arr.shape[0], *arr.shape[1:]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            return arr

        end = min(eos + 1, episode_end)
        raw = self.episode_reader.read({zarr_key: (t, end)})
        self._pad_sequences(raw, horizon)
        return raw[zarr_key]

    def _resample(self, index, _fallback_origin, _attempts) -> dict:
        origin = _fallback_origin if _fallback_origin is not None else index
        next_idx, attempts = get_fallback_idx(
            idx=index,
            candidates=range(self.total_frames),
            _attempts=_attempts,
            max_attempts=self.total_frames,
            exhausted_error=(
                f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
            ),
        )
        return self.__getitem__(next_idx, _fallback_origin=origin, _attempts=attempts)

    # --- main entrypoint ---------------------------------------------------

    def __getitem__(self, index: int, _fallback_origin=None, _attempts=None) -> dict:
        """Returns a single action-expert sample.

        Output keys:
          - Obs (camera/proprio): <key>_1 (BOS), <key>_t (frame t), <key>_T (EOS)
          - Annotations:          <key>_1 (BOS), <key>_t (frame t), <key>_T (EOS)
          - Actions:              <key>  (t..EOS, padded to horizon)
        """
        t = self._valid_frame_indices[index]
        ann = self._annotations[self.annotation_map[t]]
        bos, eos = int(ann["start_idx"]), int(ann["end_idx"])

        try:
            data: dict = {}
            for k, spec in self.key_map.items():
                zarr_key = spec["zarr_key"]
                key_type = spec.get("key_type")
                bos_key, t_key, eos_key = _obs_key_triple(k)

                if key_type == "annotation_keys":
                    data[bos_key] = self._annotation_text_for_frame(bos)
                    data[t_key] = self._annotation_text_for_frame(t)
                    data[eos_key] = self._annotation_text_for_frame(eos)
                elif key_type == "action_keys":
                    data[k] = self._load_actions(zarr_key, t, eos, spec.get("horizon"))
                else:  # camera_keys / proprio_keys: sample BOS, t, and EOS.
                    data[bos_key] = self._load_obs_at(zarr_key, bos)
                    data[t_key] = self._load_obs_at(zarr_key, t)
                    data[eos_key] = self._load_obs_at(zarr_key, eos)

            # Transforms expect obs at the canonical key (not <key>_t), so
            # alias <key>_t -> <key> for the pass and drop afterward.
            aliases: list[str] = []
            if self.transform:
                for k, spec in self.key_map.items():
                    if spec.get("key_type") in _OBS_KEY_TYPES:
                        _, t_key, _ = _obs_key_triple(k)
                        if t_key in data:
                            data[k] = data[t_key]
                            aliases.append(k)
                for transform in self.transform:
                    data = transform.transform(data)
                for alias in aliases:
                    data.pop(alias, None)
        except Exception:
            # Bad JPEG or transform failure -> resample.
            return self._resample(index, _fallback_origin, _attempts)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        data["embodiment"] = get_embodiment_id(self.embodiment)
        return data


class S3ActionExpertEpisodeResolver(S3EpisodeResolver):
    """S3EpisodeResolver that loads ZarrActionExpertDataset instances."""

    _dataset_class = ZarrActionExpertDataset

    def __init__(self, *args, fixed_horizon: int | None = None, **kwargs):
        self._fixed_horizon = fixed_horizon
        super().__init__(*args, **kwargs)

    def _load_zarr_datasets(self, search_path, valid_folder_names):
        datasets = super()._load_zarr_datasets(search_path, valid_folder_names)
        for ds in datasets.values():
            ds.fixed_horizon = self._fixed_horizon
        return datasets


class LocalActionExpertEpisodeResolver(LocalEpisodeResolver):
    """LocalEpisodeResolver that loads ZarrActionExpertDataset instances."""

    _dataset_class = ZarrActionExpertDataset

    def __init__(self, *args, fixed_horizon: int | None = None, **kwargs):
        self._fixed_horizon = fixed_horizon
        super().__init__(*args, **kwargs)

    def _load_zarr_datasets(self, search_path, valid_folder_names):
        datasets = super()._load_zarr_datasets(search_path, valid_folder_names)
        for ds in datasets.values():
            ds.fixed_horizon = self._fixed_horizon
        return datasets
