"""Base class for one episode's samples, whatever the storage: keymap windowing
(chunks, history, lag), padding and masks, and the shared sample finishing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import HISTORY_MASK_SUFFIX, get_embodiment_id


class EpisodeLeaf(torch.utils.data.Dataset):
    """One episode under a ``MultiDataset``.

    Subclasses set ``episode_path``, ``total_frames``, ``embodiment``, ``metadata``
    (with ``fps``) and ``intrinsics``, read the windows ``_read_windows`` names, and
    hand the raw arrays to ``_finish``.

    A leaf with ``batched_fetch`` set has ``MultiDataset.__getitems__`` read a whole
    DataLoader batch at once: ``plan_sample`` per sample, one ``fetch_planned`` per
    leaf class, then ``assemble_sample`` per sample.
    """

    # False when the leaf runs threads that a forked DataLoader worker would
    # inherit mid-flight (see multi.worker_context).
    fork_safe = True
    batched_fetch = False

    def __init__(self, *, key_map: dict, transform_list: list | None = None):
        self.key_map = key_map
        self.transform = transform_list
        super().__init__()

    def cache_fingerprint(self) -> str | None:
        """Digest the norm cache keys this episode's samples on; None disables it."""
        from egomimic.rldb.norm.norm_cache import episode_fingerprint

        return episode_fingerprint(getattr(self, "episode_path", None))

    def plan_sample(self, i: int):
        raise NotImplementedError

    @staticmethod
    def fetch_planned(items: list) -> list:
        raise NotImplementedError

    def assemble_sample(self, plan, fetched) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.total_frames

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        """End index (exclusive) for a windowed read starting at ``start_idx``.

        Subclasses can override to add per-key-type clamping (e.g., annotation EOS).
        """
        return min(start_idx + horizon, self.total_frames)

    def _action_horizon(self) -> int | None:
        """Horizon of the keymap's ``action_keys`` entries, if it has any."""
        for spec in self.key_map.values():
            if (
                spec.get("key_type") == "action_keys"
                and spec.get("horizon") is not None
            ):
                return int(spec["horizon"])
        return None

    def _action_pad_mask(self, idx: int) -> np.ndarray | None:
        """1.0 per real action frame, 0.0 per repeat-last padded one, (horizon,).

        ``_pad_to_horizon`` pads a short action chunk by repeating the last
        frame, which in a wrist-relative action space supervises "stop moving"
        at every episode (or annotation) tail. This mask says which steps are
        real so ``loss_fn`` can drop the rest; ``None`` when the keymap has no
        horizoned action key.
        """
        horizon = self._action_horizon()
        if horizon is None:
            return None
        n_real = self._chunk_end_idx(idx, horizon, "action_keys") - idx
        mask = np.zeros(horizon, dtype=np.float32)
        mask[: max(n_real, 0)] = 1.0
        return mask

    def _hand_masks(self, idx: int) -> dict:
        """Per-hand validity keys for sample ``idx`` (``action_hand_mask``,
        ``proprio_hand_valid``); none when the store tracks no per-hand validity."""
        return {}

    def _proprio_history_spec(self) -> dict | None:
        """The first non-camera keymap entry with ``history`` > 1, else None."""
        for spec in self.key_map.values():
            if spec.get("key_type") == "camera_keys":
                continue
            if int(spec.get("history") or 1) > 1:
                return spec
        return None

    def _proprio_history_mask(self, idx: int) -> np.ndarray | None:
        """1.0 per real history frame, 0.0 per front-padded one, ``(K,)``;
        ``None`` when the keymap has no proprio history key."""
        spec = self._proprio_history_spec()
        if spec is None:
            return None
        return self._history_indices(idx, spec)[1]

    def _fps(self) -> float:
        """The episode's frame rate (zarr ``fps`` attr; 30 when absent)."""
        return float(self.metadata.get("fps") or 30)

    def _seconds_to_frames(self, seconds: float) -> int:
        """``seconds`` in this episode's fps, at least one frame."""
        return max(1, round(float(seconds) * self._fps()))

    def _history_floor(self, idx: int) -> int:
        """First frame history and lagged reads for ``idx`` may reach back to."""
        return 0

    def _lagged_idx(self, idx: int, lag_s: float) -> int:
        """Frame ``lag_s`` seconds before ``idx`` (at least one frame back),
        clamped at the history floor -- so the first samples read the floor
        frame, the same front padding a history window gets."""
        return max(self._history_floor(idx), idx - self._seconds_to_frames(lag_s))

    def _history_stride(self, spec: dict) -> int:
        """Frames between history steps: ``history_stride_s`` (seconds, so a
        window spans the same time at any fps) wins over ``history_stride``."""
        stride_s = spec.get("history_stride_s")
        if stride_s is not None:
            return self._seconds_to_frames(stride_s)
        return int(spec.get("history_stride", 1) or 1)

    def _history_indices(self, idx: int, spec: dict) -> tuple[np.ndarray, np.ndarray]:
        """``(K,)`` frame indices of a ``history`` entry, oldest first, and their
        ``(K,)`` float32 mask.

        The newest step is ``idx`` or, with ``lag_s``, that many seconds before
        it. Steps before the history floor (the episode start) are masked 0
        and repeat the oldest real step (the floor frame when none is real), so
        the newest step is always ``[-1]``.
        """
        history = int(spec["history"])
        newest = idx
        if spec.get("lag_s") is not None:
            newest = idx - self._seconds_to_frames(spec["lag_s"])
        steps = newest - self._history_stride(spec) * np.arange(history - 1, -1, -1)
        floor = self._history_floor(idx)
        real = steps >= floor
        fill = steps[real][0] if real.any() else floor
        return np.where(real, steps, fill), real.astype(np.float32)

    @staticmethod
    def _pad_to_horizon(array, horizon: int | None):
        """Repeat-last out to ``horizon`` a chunk the episode (or annotation)
        ended before filling."""
        if horizon is None or not isinstance(array, np.ndarray):
            return array
        missing = horizon - array.shape[0]
        if missing <= 0:
            return array
        return np.concatenate([array, np.repeat(array[-1:], missing, axis=0)], axis=0)

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        """``_pad_to_horizon`` over a read result, in place. Keys are zarr keys."""
        if horizon is None:
            return data
        for k in data:
            data[k] = self._pad_to_horizon(data[k], horizon)
        return data

    def _read_windows(
        self, idx: int
    ) -> dict[str, tuple[str, tuple[int, int | None] | np.ndarray]]:
        """``{keymap key: (zarr key, window)}`` for frame *idx*.

        ``window`` is ``(start, end)``, ``end`` None meaning the single frame
        at ``start``, or for a ``history`` entry the ``(K,)`` frame indices of
        ``_history_indices``. Annotation keys read nothing and are left out.
        """
        windows = {}
        for k, spec in self.key_map.items():
            if spec.get("key_type") == "annotation_keys":
                continue
            horizon = spec.get("horizon")
            history = spec.get("history")
            lag_s = spec.get("lag_s")
            if horizon is not None:
                window = (
                    idx,
                    self._chunk_end_idx(idx, horizon, spec.get("key_type")),
                )
            elif history is not None and (
                int(history) > 1 or spec.get("key_type") == "camera_keys"
            ):
                # A camera window keeps its (K, ...) axis even at K = 1. Point
                # reads, not a contiguous span: on a JPEG key (one frame per
                # chunk) a span would fetch every frame between the steps.
                # ZarrEpisode.plan_reads still merges them on numeric keys.
                window = self._history_indices(idx, spec)[0]
            elif lag_s is not None:
                # A single PAST frame (image history): the same zarr array as
                # the current-frame entry, read ``lag_s`` seconds back.
                window = (self._lagged_idx(idx, lag_s), None)
            else:
                window = (idx, None)
            windows[k] = (spec["zarr_key"], window)
        return windows

    def _finish(self, idx: int, data: dict) -> dict:
        """Masks, transforms and per-episode metadata on the raw arrays read for
        frame ``idx``; returns the sample as tensors."""
        pad_mask = self._action_pad_mask(idx)
        if pad_mask is not None:
            # Travels with the chunk through the interpolators
            # (InterpolatePadMask) to the model horizon.
            data["action_pad_mask"] = pad_mask

        for k, spec in self.key_map.items():
            if spec.get("key_type") == "camera_keys" and spec.get("history"):
                # Unlike proprio, a padded frame has to be masked out of
                # attention, so each image window carries its own mask.
                data[f"{k}{HISTORY_MASK_SUFFIX}"] = self._history_indices(idx, spec)[1]

        data.update(self._hand_masks(idx))

        history_mask = self._proprio_history_mask(idx)
        if history_mask is not None:
            # Informational: the front padding already looks exactly like a
            # history-dropout sample, so the stem needs nothing extra.
            data["proprio_history_mask"] = history_mask

        if self.transform:
            for transform in self.transform or []:
                data = transform.transform(data)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        data["embodiment"] = get_embodiment_id(self.embodiment)
        # Control frequency of this episode (RDT's ctrl-freq token).
        data["fps"] = torch.tensor(self._fps(), dtype=torch.float32)
        data["intrinsics"] = torch.from_numpy(self._intrinsics_3x4())
        data["episode_hash"] = self._episode_hash()
        return data

    def _intrinsics_3x4(self) -> np.ndarray:
        """``intrinsics`` as the (3, 4) K the batch carries."""
        # Per-episode camera intrinsics travel with the batch so a single
        # data-driven embodiment can project without a hardcoded class const.
        # Single-camera -> (3,4) K matrix; no/multi-camera -> NaN sentinel,
        # which _intrinsics_from_batch treats as "fall back to cls.INTRINSICS".
        K = self.intrinsics
        if isinstance(K, dict):
            # Multi-camera rig: project against the front camera (viz/projection
            # target the front image). Prefer a "front"-keyed entry, else the
            # first available camera.
            K = next(
                (v for k, v in K.items() if "front" in str(k).lower()),
                next(iter(K.values()), None) if K else None,
            )
        if K is not None:
            K = np.asarray(K, dtype=np.float32)
            if K.size == 9:
                # Lance stores K flat.
                K = K.reshape(3, 3)
            if K.shape == (3, 3):
                # Normalize a 3x3 K to the canonical 3x4 (zeros last column);
                # some contributors store 3x3 (e.g. microagi).
                K = np.concatenate([K, np.zeros((3, 1), dtype=np.float32)], axis=1)
            if K.shape != (3, 4):  # unexpected -> sentinel (viz falls back to const)
                K = np.full((3, 4), np.nan, dtype=np.float32)
        else:
            K = np.full((3, 4), np.nan, dtype=np.float32)
        return np.ascontiguousarray(K)

    def _episode_hash(self) -> str:
        name = Path(self.episode_path).name
        return name[:-5] if name.endswith(".zarr") else name
