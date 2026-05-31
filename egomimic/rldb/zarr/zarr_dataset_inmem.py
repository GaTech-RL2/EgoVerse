"""In-memory ZarrDataset variant for fast HPT-style per-frame training.

Motivation
----------
The base ``ZarrDataset.__getitem__`` decodes JPEG frames and reads zarr arrays
*per sample*. For the pushshapes_sim per-frame loader (horizon=32 obs+action
windows) this puts an irreducible ~3-5 ms of Python/JPEG work on every sample,
which caps DataLoader throughput well below what the GPU can consume.

``InMemoryZarrDataset`` decodes each episode **once** at construction time and
holds the frames in RAM as ``uint8`` CHW tensors (plus float32 state/action
arrays). ``__getitem__`` then becomes a pure array slice + a ``uint8 -> float32
/255`` cast — no JPEG decode, no zarr read. Because the decode happens in the
parent process *before* the DataLoader forks its workers, the large image
buffers are shared with every worker via copy-on-write (the workers only read,
so the pages are never duplicated).

Design constraints
------------------
* **Drop-in / byte-identical output.** ``__getitem__`` returns exactly the same
  dict (same keys, dtypes, shapes, value ranges, and pad-by-repeat-last
  semantics) as ``ZarrDataset.__getitem__``. Normalization, bounds-checking,
  collation, ``norm_stats`` and the model's ``process_batch_for_training`` are
  therefore completely unchanged — this class only swaps *how* a leaf produces
  its raw window.
* **Zero edits to existing files.** Wiring is via a ``LocalEpisodeResolver``
  subclass that sets ``_dataset_class`` (the same mechanism
  ``LocalAnnotationCutoffEpisodeResolver`` already uses) plus a new data config.
  Other users / configs are unaffected.

Memory
------
One frame decoded to uint8 CHW at 96x96 is ~27 KB. A 954-episode x 300-frame
circle dataset is ~7.7 GB held once in the parent and shared to workers. State
and action arrays are negligible.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import simplejpeg
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    ZarrDataset,
)

logger = logging.getLogger(__name__)

# How many threads to use for the per-episode JPEG decode at construction.
# simplejpeg (libjpeg-turbo) releases the GIL during decode, so threads give a
# real speedup. Overridable for nodes with few cores.
_DECODE_THREADS = int(os.environ.get("EGOVERSE_INMEM_DECODE_THREADS", "8"))


class InMemoryZarrDataset(ZarrDataset):
    """``ZarrDataset`` that preloads its whole episode into RAM at construction.

    All windowing / padding semantics mirror the base class exactly:
      * window is ``[idx : min(idx + horizon, total_frames))``
      * sequences shorter than ``horizon`` are padded by repeating the last
        real frame (``_pad_sequences`` repeat-last)
      * images are returned CHW float32 in ``[0, 1]``; state/action are returned
        raw float32 (normalization is applied later by ``MultiDataset``)
    """

    def __init__(self, Episode_path, key_map, transform_list=None):
        # Base init opens the zarr store, reads metadata, sets total_frames,
        # _image_keys / _json_keys, key_map, etc.
        super().__init__(Episode_path, key_map, transform_list=transform_list)
        # Preloaded per-key arrays:
        #   image keys   -> uint8 ndarray (T, 3, H, W)
        #   other keys   -> float32 ndarray (T, ...)
        self._mem: dict[str, np.ndarray] = {}
        self._preload()
        # Optional idle-tail / (0,0)-garbage filter (EGOVERSE_TRIM_IDLE=1).
        # The pushshapes demos record the cursor at (0,0) when it parks/leaves
        # the screen (mostly an idle tail at episode end). Those frames teach a
        # BC policy to "drive to the corner and stop". Drop them so the policy
        # only sees real pushing actions; this also cleans the quantile-norm
        # floor (computed from this same dataset).
        self._valid_indices = None
        if os.environ.get("EGOVERSE_TRIM_IDLE") == "1":
            self._build_valid_indices()

    # ------------------------------------------------------------------ #
    # Construction-time preload
    # ------------------------------------------------------------------ #
    def _preload(self) -> None:
        store = self.episode_reader._get_store()
        for k in self.key_map:
            spec = self.key_map[k]
            zarr_key = spec["zarr_key"]
            key_type = spec.get("key_type", None)

            # Non-windowed metadata/annotation keys are handled lazily in
            # __getitem__ (mirrors the base class); nothing to preload.
            if key_type in ("annotation_keys", "metadata_keys"):
                continue

            raw = store[zarr_key][:]

            if zarr_key in self._image_keys:
                self._mem[k] = self._decode_all_jpegs(raw)
            elif zarr_key in self._json_keys:
                # JSON keys aren't used by the pushshapes per-frame keymap; keep
                # them out of the in-RAM fast path and let __getitem__ raise if
                # one is ever requested, rather than silently mishandling it.
                raise NotImplementedError(
                    f"InMemoryZarrDataset does not support JSON key {zarr_key!r}"
                )
            else:
                self._mem[k] = np.asarray(raw, dtype=np.float32)

    def _build_valid_indices(self):
        """Drop (0,0)-garbage action frames and truncate the trailing idle tail.

        Sets self._valid_indices (sampled positions) and self.total_frames to
        the idle-tail start (so action-chunk windows never read tail garbage).
        """
        act = self._mem.get("actions")
        if act is None:
            return
        a = np.asarray(act, dtype=np.float32).reshape(len(act), -1)
        zero = (np.abs(a[:, 0]) < 1.0) & (np.abs(a[:, 1]) < 1.0)
        nz = np.where(~zero)[0]
        if len(nz) == 0:
            return  # degenerate episode: keep as-is
        tail_start = int(nz[-1]) + 1            # exclude trailing idle (0,0) run
        self.total_frames = tail_start          # clamp windows to real frames
        # sample only non-(0,0) frames before the tail (also drops scattered ones)
        self._valid_indices = [i for i in range(tail_start) if not zero[i]]
        if not self._valid_indices:             # safety: never empty
            self._valid_indices = list(range(tail_start))

    def __len__(self):
        if getattr(self, "_valid_indices", None) is not None:
            return len(self._valid_indices)
        return self.total_frames

    @staticmethod
    def _decode_one(buf) -> np.ndarray:
        # HWC uint8 -> CHW uint8 (defer the /255 float cast to __getitem__ so the
        # in-RAM buffer stays 4x smaller).
        img = simplejpeg.decode_jpeg(buf, colorspace="RGB")
        return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))

    def _decode_all_jpegs(self, raw) -> np.ndarray:
        n = len(raw)
        if n == 0:
            return np.empty((0, 3, 0, 0), dtype=np.uint8)
        if _DECODE_THREADS > 1 and n > 1:
            with ThreadPoolExecutor(max_workers=_DECODE_THREADS) as ex:
                frames = list(ex.map(self._decode_one, raw))
        else:
            frames = [self._decode_one(b) for b in raw]
        return np.stack(frames, axis=0)  # (T, 3, H, W) uint8

    # ------------------------------------------------------------------ #
    # Fast per-sample read
    # ------------------------------------------------------------------ #
    def _window(self, arr: np.ndarray, idx: int, horizon: int) -> np.ndarray:
        """Slice ``[idx:idx+horizon]`` with repeat-last padding to ``horizon``.

        Matches ``ZarrDataset._chunk_end_idx`` + ``_pad_sequences``.
        """
        end = min(idx + horizon, self.total_frames)
        window = arr[idx:end]
        seq_len = window.shape[0]
        if seq_len < horizon:
            pad = np.repeat(window[-1:], horizon - seq_len, axis=0)
            window = np.concatenate([window, pad], axis=0)
        return window

    def __getitem__(self, idx, _fallback_origin=None, _attempts=None):
        if getattr(self, "_valid_indices", None) is not None:
            idx = self._valid_indices[idx]
        data: dict = {}
        for k in self.key_map:
            spec = self.key_map[k]
            zarr_key = spec["zarr_key"]
            key_type = spec.get("key_type", None)
            horizon = spec.get("horizon", None)

            # Annotation/metadata keys fall back to the base implementation's
            # behaviour (the pushshapes per-frame keymap has none of these, but
            # keep parity for safety).
            if key_type == "annotation_keys":
                data[k] = self._annotation_text_for_frame(idx)
                continue
            if key_type == "metadata_keys":
                continue

            arr = self._mem[k]
            if horizon is not None:
                window = self._window(arr, idx, int(horizon))
            else:
                window = arr[idx]

            if zarr_key in self._image_keys:
                # uint8 CHW -> float32 [0, 1], identical to the base decode path.
                window = window.astype(np.float32) / 255.0

            data[k] = torch.from_numpy(np.ascontiguousarray(window)).to(torch.float32)

        if self.transform:
            for transform in self.transform or []:
                data = transform.transform(data)
            # A transform may reintroduce numpy arrays; mirror the base cast.
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(torch.float32)

        data["metadata.robot_name"] = get_embodiment_id(self.embodiment)
        data["embodiment"] = get_embodiment_id(self.embodiment)
        return data


class InMemoryLocalEpisodeResolver(LocalEpisodeResolver):
    """LocalEpisodeResolver that loads :class:`InMemoryZarrDataset` leaves."""

    _dataset_class = InMemoryZarrDataset
