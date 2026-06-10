"""
Packed Zarr datasets and a pack_collate for flattened, variable-length sample
batching.

Each sample is one full episode (or one chunk of a long episode) read at the
native per-frame resolution by ``ZarrDataset._read_span``. The collator
concatenates samples along the time axis and emits ``cu_seqlens`` so the
downstream model can recover per-sample boundaries (FlashAttention-style).

Currently implemented:
  - ``ZarrEpisodePackedDataset``: one sample per episode (or per chunk of a
    long episode), intended for verifying full-episode loading paths.
  - ``pack_collate``: tensor-side concatenation + cu_seqlens emission, plus
    pass-through of variable-length list-valued keys (e.g. annotations).

The base ``_read_span`` and the collator are deliberately written to be
reusable by a future annotation-level packed dataset; only the index-building
logic differs between the two flavours.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    LocalEpisodeResolver,
    ZarrDataset,
    get_fallback_idx,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


ChunkingMode = Literal["sequential", "none"]

TailMode = Literal["drop", "hold_pad"]

HoldoutSplit = Literal["train", "val"]

ActionPairing = Literal["current", "incoming_meanpad"]


class ZarrEpisodePackedDataset(Dataset):
    """One sample per episode (or per chunk of a long episode).

    Each ``__getitem__`` call returns the dict produced by
    ``ZarrDataset._read_span``, i.e. variable-length per-frame tensors plus
    metadata. Samples are intended to flow into :func:`pack_collate`.

    Args:
        datasets: dict of episode-key -> per-episode ``ZarrDataset``. Same
            shape returned by ``LocalEpisodeResolver.resolve`` /
            ``S3EpisodeResolver.resolve``.
        max_seq_len: hard cap on sample length (in post-stride frames). With
            ``chunking="sequential"`` an episode longer than this is split
            into consecutive chunks of length ``max_seq_len`` (last chunk may
            be shorter). ``None`` disables chunking.
        min_seq_len: drop chunks shorter than this (in post-stride frames).
            Useful for the trailing chunk of an episode. See ``tail_mode``
            for the pad-instead-of-drop alternative.
        chunking: ``"sequential"`` for consecutive non-overlapping chunks,
            ``"none"`` to emit each episode as a single sample (raises if
            ``max_seq_len`` is exceeded — caller should pre-filter).
        max_resample_attempts: per ``__getitem__`` cap on resampling after a
            read failure (bad JPEG, transform failure, etc.).
        temporal_stride: keep every ``temporal_stride``-th frame of each
            sample (keyframe subsampling, applied at read time via
            ``_read_span(..., stride=...)``). ``1`` (default) reads every
            frame — byte-identical to pre-knob behavior. Index spans stay in
            raw-frame coordinates; ``max_seq_len`` / ``min_seq_len`` are
            interpreted in post-stride frames.
        tail_mode: ``"drop"`` (default) silently drops samples shorter than
            ``min_seq_len`` at index-build time (current behavior).
            ``"hold_pad"`` keeps them and repeat-pads each short sample up to
            ``min_seq_len`` at read time by holding the final frame — final
            observation AND final action row repeat, mirroring the upstream
            pureDF repeat-pad convention.
        val_holdout_first_n: when ``> 0``, the FIRST n episodes in sorted-key
            order form a held-out validation pool. Which side this instance
            keeps is selected by ``holdout_split``. ``0`` (default) disables
            the split entirely (current behavior: all episodes).
        holdout_split: ``"train"`` keeps everything EXCEPT the held-out pool;
            ``"val"`` keeps ONLY the held-out pool. Ignored when
            ``val_holdout_first_n == 0``. ``episode_idx`` assignment stays
            anchored to the FULL sorted key list, so an episode keeps the
            same index on both sides of the split.
        action_chunk_k: K action-chunk stacking (D07, upstream pureDF
            ``pusht_dataset.py``). At ``K > 1`` the sample is emitted at
            keyframe rate — obs keys keep every K-th post-stride frame —
            and each keyframe's action row is the K post-stride actions of
            its chunk flattened TIME-MAJOR into ``K*action_dim`` values
            (``[a0_d0, a0_d1, a1_d0, ...]``). Composes multiplicatively
            with ``temporal_stride`` (keyframes every ``temporal_stride*K``
            raw frames; the K stacked actions are the ``temporal_stride``-d
            ones between keyframes); the parity recipe uses
            ``action_chunk_k`` alone (``temporal_stride=1``).
            ``min_seq_len`` / ``max_seq_len`` are interpreted in EMITTED
            keyframes. ``1`` (default) is byte-identical to prior behavior.
        chunk_phase_aug: D26 phase-shift augmentation. When True and
            ``action_chunk_k > 1``, each read drops a per-sample uniform
            random number of leading post-stride frames in ``[0, K)``
            (upstream ``drop_front``) before chunking. False (default)
            draws NO RNG; also a no-op (no RNG) at K=1, mirroring
            upstream's ``randint(K) if K > 1 else 0``.
        action_pairing: which action chunk a keyframe carries.
            ``"current"`` (default): keyframe t carries post-stride actions
            ``[t, t+K)`` starting AT the keyframe — the K=1-compatible
            generalization of today's frame-t ↔ ``actions[t]`` pairing; the
            trailing partial chunk repeat-pads the final action row.
            ``"incoming_meanpad"``: upstream-exact incoming convention
            (``pusht_dataset.py.__getitem__``): drop the LAST post-stride
            frame (upstream ``drop_back=1``), keyframe count floors to
            ``(L-1)//K``, keyframe i ≥ 1 carries actions ``[(i-1)K, iK)``
            (the chunk that LED INTO it), and keyframe 0's row is the
            per-dim mean of the sample's post-stride actions tiled across
            the K slots (upstream pads with the dataset action mean — the
            normalization midpoint, exactly 0 post-norm; the per-sample
            mean is the dataset-local analog, ≈0 after algo-side norm).
    """

    def __init__(
        self,
        datasets: dict[str, ZarrDataset],
        max_seq_len: int | None = None,
        min_seq_len: int = 1,
        chunking: ChunkingMode = "sequential",
        max_resample_attempts: int | None = None,
        temporal_stride: int = 1,
        tail_mode: TailMode = "drop",
        val_holdout_first_n: int = 0,
        holdout_split: HoldoutSplit = "train",
        action_chunk_k: int = 1,
        chunk_phase_aug: bool = False,
        action_pairing: ActionPairing = "current",
    ):
        if not datasets:
            raise ValueError("ZarrEpisodePackedDataset received an empty datasets dict")
        if min_seq_len < 1:
            raise ValueError(f"min_seq_len must be >= 1, got {min_seq_len}")
        if chunking not in ("sequential", "none"):
            raise ValueError(f"Unknown chunking mode: {chunking}")
        if temporal_stride < 1:
            raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
        if tail_mode not in ("drop", "hold_pad"):
            raise ValueError(f"Unknown tail_mode: {tail_mode}")
        if val_holdout_first_n < 0:
            raise ValueError(
                f"val_holdout_first_n must be >= 0, got {val_holdout_first_n}"
            )
        if holdout_split not in ("train", "val"):
            raise ValueError(f"Unknown holdout_split: {holdout_split}")
        if action_chunk_k < 1:
            raise ValueError(f"action_chunk_k must be >= 1, got {action_chunk_k}")
        if action_pairing not in ("current", "incoming_meanpad"):
            raise ValueError(f"Unknown action_pairing: {action_pairing}")

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.chunking = chunking
        self.max_resample_attempts = max_resample_attempts
        self.temporal_stride = temporal_stride
        self.tail_mode = tail_mode
        self.val_holdout_first_n = val_holdout_first_n
        self.holdout_split = holdout_split
        self.action_chunk_k = action_chunk_k
        self.chunk_phase_aug = chunk_phase_aug
        self.action_pairing = action_pairing
        # K=1 + "current" must take the EXACT pre-knob code path (no chunk
        # post-processing, no extra _read_span kwargs, no RNG). incoming
        # pairing is a behavior change even at K=1 (drop_back + mean-pad
        # keyframe 0), so it routes through the chunk path too.
        self._chunk_active = action_chunk_k > 1 or action_pairing != "current"

        # Stable episode-index assignment: sorted by key, over the FULL
        # episode set (pre-holdout) so episode_idx is split-invariant.
        all_keys: list[str] = sorted(datasets.keys())
        self._episode_idx_by_key: dict[str, int] = {
            key: i for i, key in enumerate(all_keys)
        }

        if val_holdout_first_n > 0:
            if holdout_split == "val":
                kept = all_keys[:val_holdout_first_n]
            else:
                kept = all_keys[val_holdout_first_n:]
            if not kept:
                raise ValueError(
                    f"val_holdout_first_n={val_holdout_first_n} with "
                    f"holdout_split={holdout_split!r} left no episodes "
                    f"(total={len(all_keys)})."
                )
            self.datasets = {key: datasets[key] for key in kept}
            self._episode_keys: list[str] = kept
        else:
            self.datasets = datasets
            self._episode_keys = all_keys

        self.index: list[tuple[str, int, int]] = self._build_index()
        if not self.index:
            raise ValueError(
                "ZarrEpisodePackedDataset built an empty index — every episode "
                f"was shorter than min_seq_len={min_seq_len}."
            )

        super().__init__()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _strided_len(self, n_raw_frames: int) -> int:
        """Post-stride frame count of a raw span of ``n_raw_frames`` frames."""
        return -(-n_raw_frames // self.temporal_stride)

    def _emitted_len_bounds(self, n_raw_frames: int) -> tuple[int, int]:
        """(min, max) EMITTED sample length (keyframes) of a raw span.

        With the chunk path inactive both bounds equal ``_strided_len``.
        Otherwise the length depends on the per-read phase p (0 when
        ``chunk_phase_aug`` is off): post-phase virtual length ``V - p``,
        then ``ceil(·/K)`` for "current" or ``(· - 1) // K`` (drop_back=1 +
        floor) for "incoming_meanpad". min is at p=K-1, max at p=0.
        """
        v_frames = self._strided_len(n_raw_frames)
        if not self._chunk_active:
            return v_frames, v_frames
        k = self.action_chunk_k
        phases = range(k) if (self.chunk_phase_aug and k > 1) else range(1)
        lens = []
        for p in phases:
            v = v_frames - p
            if self.action_pairing == "incoming_meanpad":
                lens.append(max((v - 1) // k, 0))
            else:
                lens.append(max(-(-v // k), 0))
        return min(lens), max(lens)

    def _build_index(self) -> list[tuple[str, int, int]]:
        """Index spans are (key, raw_start, raw_end); length checks are in
        EMITTED frames (post-stride frames; keyframes when chunking actions).
        ``tail_mode="hold_pad"`` keeps under-length samples (padded at read
        time) instead of dropping them — but spans that can emit ZERO
        keyframes under some phase are always dropped (nothing to pad).
        """
        index: list[tuple[str, int, int]] = []
        max_len = self.max_seq_len
        hold_pad = self.tail_mode == "hold_pad"
        for key in self._episode_keys:
            ds = self.datasets[key]
            episode_len = int(ds.total_frames)
            lo, hi = self._emitted_len_bounds(episode_len)
            if lo < 1 or (lo < self.min_seq_len and not hold_pad):
                continue

            if max_len is None or self.chunking == "none":
                if max_len is not None and hi > max_len:
                    raise ValueError(
                        f"Episode {key} has {episode_len} frames > max_seq_len="
                        f"{max_len} but chunking='none'. Either enable "
                        f"chunking or raise max_seq_len."
                    )
                index.append((key, 0, episode_len))
                continue

            # Sequential chunking: max_len emitted frames per chunk (raw
            # span scales by stride AND chunk K). NOTE: with
            # action_pairing="incoming_meanpad" every chunk re-introduces a
            # mean-pad first keyframe; the parity recipe uses
            # chunking="none" (whole episodes).
            raw_chunk_len = max_len * self.temporal_stride * self.action_chunk_k
            for start in range(0, episode_len, raw_chunk_len):
                end = min(start + raw_chunk_len, episode_len)
                lo, _hi = self._emitted_len_bounds(end - start)
                if lo < 1 or (lo < self.min_seq_len and not hold_pad):
                    continue
                index.append((key, start, end))

        return index

    @classmethod
    def from_resolver(
        cls,
        resolver: EpisodeResolver,
        *,
        max_seq_len: int | None = None,
        min_seq_len: int = 1,
        chunking: ChunkingMode = "sequential",
        max_resample_attempts: int | None = None,
        temporal_stride: int = 1,
        tail_mode: TailMode = "drop",
        val_holdout_first_n: int = 0,
        holdout_split: HoldoutSplit = "train",
        action_chunk_k: int = 1,
        chunk_phase_aug: bool = False,
        action_pairing: ActionPairing = "current",
        **resolve_kwargs,
    ) -> "ZarrEpisodePackedDataset":
        """Build from an ``EpisodeResolver`` (mirrors ``MultiDataset._from_resolver``)."""
        if isinstance(resolver, LocalEpisodeResolver):
            datasets = resolver.resolve(**resolve_kwargs)
        else:
            datasets = resolver.resolve(**resolve_kwargs)
        return cls(
            datasets=datasets,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            chunking=chunking,
            max_resample_attempts=max_resample_attempts,
            temporal_stride=temporal_stride,
            tail_mode=tail_mode,
            val_holdout_first_n=val_holdout_first_n,
            holdout_split=holdout_split,
            action_chunk_k=action_chunk_k,
            chunk_phase_aug=chunk_phase_aug,
            action_pairing=action_pairing,
        )

    @classmethod
    def from_local_folder(
        cls,
        folder_path: str | Path,
        *,
        key_map: dict,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        max_seq_len: int | None = None,
        min_seq_len: int = 1,
        chunking: ChunkingMode = "sequential",
        max_resample_attempts: int | None = None,
        temporal_stride: int = 1,
        tail_mode: TailMode = "drop",
        val_holdout_first_n: int = 0,
        holdout_split: HoldoutSplit = "train",
        action_chunk_k: int = 1,
        chunk_phase_aug: bool = False,
        action_pairing: ActionPairing = "current",
        filters=None,
    ) -> "ZarrEpisodePackedDataset":
        """Convenience factory that discovers all episodes under a local folder.

        Useful for smoke tests against a known local dataset (e.g. the pushT
        ``test_demos`` directory) without needing the SQL-backed resolver.
        """
        resolver = LocalEpisodeResolver(
            folder_path=Path(folder_path),
            key_map=key_map,
            transform_list=transform_list,
            debug=debug,
        )
        return cls.from_resolver(
            resolver,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            chunking=chunking,
            max_resample_attempts=max_resample_attempts,
            temporal_stride=temporal_stride,
            tail_mode=tail_mode,
            val_holdout_first_n=val_holdout_first_n,
            holdout_split=holdout_split,
            action_chunk_k=action_chunk_k,
            chunk_phase_aug=chunk_phase_aug,
            action_pairing=action_pairing,
            filters=filters,
        )

    # ------------------------------------------------------------------ #
    # Norm-stat plumbing (trainHydra parity with MultiDataset)
    # ------------------------------------------------------------------ #

    def set_norm_stats_from(self, source) -> None:
        """No-op compatibility hook.

        ``MultiDataset.set_norm_stats_from`` wires per-key normalization
        stats into the dataset so that ``__getitem__`` normalizes at read
        time. The packed dataset deliberately keeps reads raw — normalization
        for the algo's packed batches happens algo-side in
        ``HNet.process_batch_for_training`` via ``norm_stats.normalize(...)``.

        Implemented as a no-op so trainHydra's
        ``for ds in datasets: ds.set_norm_stats_from(norm_stats)`` loop runs
        uniformly across padded and packed datasets.
        """
        return None

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.index)

    def _hold_pad_to_min(self, data: dict) -> dict:
        """Repeat-pad a short sample up to ``min_seq_len`` by holding the
        final frame: every per-frame value (observations AND the action row)
        repeats its last entry, mirroring the upstream pureDF repeat-pad.
        Mutates and returns ``data``; updates ``"seq_len"``.
        """
        seq_len = int(data["seq_len"])
        pad_n = self.min_seq_len - seq_len
        if pad_n <= 0:
            return data
        for k, v in list(data.items()):
            if k == "seq_len" or not _is_per_frame_tensor(v, seq_len):
                continue
            if isinstance(v, torch.Tensor):
                last = v[-1:]
                data[k] = torch.cat(
                    [v, last.repeat(pad_n, *([1] * (v.ndim - 1)))], dim=0
                )
            else:  # np.ndarray (object-dtype values _read_span left raw)
                data[k] = np.concatenate(
                    [v, np.repeat(v[-1:], pad_n, axis=0)], axis=0
                )
        data["seq_len"] = self.min_seq_len
        return data

    def _read_chunked(self, ds: ZarrDataset, start: int, end: int, *, episode_idx: int) -> dict:
        """Read span ``[start, end)`` at keyframe rate and stack K actions
        per keyframe. Mirrors upstream pureDF ``pusht_dataset.py.__getitem__``
        operating on the post-``temporal_stride`` ("virtual") frame stream.
        """
        k, ts = self.action_chunk_k, self.temporal_stride
        # Upstream drop_front: per-sample phase in [0, K) virtual frames.
        # Single conditional RNG draw — none at K=1 or with the aug off.
        phase = int(np.random.randint(k)) if (self.chunk_phase_aug and k > 1) else 0
        raw_start = start + phase * ts
        if raw_start >= end:
            raise ValueError(
                f"chunk phase {phase} emptied span [{start}, {end}) at "
                f"temporal_stride={ts} — index gating should prevent this"
            )
        data = ds._read_span(
            raw_start,
            end,
            episode_idx=episode_idx,
            stride=ts * k,
            action_stride=ts,
        )
        data = self._stack_action_chunks(ds, data)
        # Provenance: first EMITTED raw frame (phase-shifted), not the index
        # span start. __getitem__ only fills chunk_offset when unset.
        data["chunk_offset"] = int(raw_start)
        return data

    def _stack_action_chunks(self, ds: ZarrDataset, data: dict) -> dict:
        """Stack the full-rate action keys of a keyframe-rate span read into
        ``(T_keyframes, K*action_dim)`` rows (time-major within the chunk:
        ``[a0_d0, a0_d1, a1_d0, ...]``, upstream channel order) and trim every
        other per-frame tensor to the emitted keyframe count. Mutates and
        returns ``data``; updates ``"seq_len"``.
        """
        k = self.action_chunk_k
        action_keys = [
            name
            for name, spec in ds.key_map.items()
            if spec.get("key_type") == "action_keys" and name in data
        ]
        if not action_keys:
            raise ValueError(
                "action_chunk_k / action_pairing need at least one "
                "key_type='action_keys' entry in key_map"
            )
        n_obs = int(data["seq_len"])  # keyframe-rate read length
        v_len = int(data[action_keys[0]].shape[0])  # virtual (post-stride) length

        if self.action_pairing == "incoming_meanpad":
            # Upstream: drop_back=1, n_stacked = floor(T'/K); keyframe i>=1
            # carries acts[(i-1)K : iK], keyframe 0 the mean pad.
            n_key = (v_len - 1) // k
            if n_key < 1:
                raise ValueError(
                    f"span too short for incoming_meanpad at K={k}: "
                    f"{v_len} post-stride frames"
                )
        else:  # "current": keyframe i carries acts[iK : iK+K).
            n_key = -(-v_len // k)

        for name in action_keys:
            acts = data[name]
            if acts.shape[0] != v_len:
                raise ValueError(
                    f"action key {name!r} length {acts.shape[0]} != {v_len}"
                )
            if self.action_pairing == "incoming_meanpad":
                trimmed = acts[: v_len - 1]  # drop_back=1 virtual frame
                chunks = trimmed[: (n_key - 1) * k].reshape(n_key - 1, -1)
                mean_row = trimmed.mean(dim=0).repeat(k).reshape(1, -1)
                data[name] = torch.cat([mean_row, chunks], dim=0)
            else:
                pad_n = n_key * k - v_len
                if pad_n:
                    # Trailing partial chunk: hold the final action row
                    # (upstream repeat-pad convention).
                    acts = torch.cat(
                        [acts, acts[-1:].repeat(pad_n, *([1] * (acts.ndim - 1)))],
                        dim=0,
                    )
                data[name] = acts.reshape(n_key, -1)

        if n_key != n_obs:
            # incoming_meanpad floors the keyframe count below the ceil-len
            # obs read — trim the remaining per-frame tensors to match.
            for name, value in list(data.items()):
                if name in action_keys or name == "seq_len":
                    continue
                if _is_per_frame_tensor(value, n_obs):
                    data[name] = value[:n_key]
        data["seq_len"] = n_key
        return data

    def __getitem__(self, i: int) -> dict:
        attempts = 0
        max_attempts = self.max_resample_attempts or len(self.index)
        idx = i
        # Only forward ``stride`` when active so duck-typed datasets without
        # the kwarg keep working at the default.
        span_kwargs = (
            {"stride": self.temporal_stride} if self.temporal_stride != 1 else {}
        )
        while True:
            key, start, end = self.index[idx]
            ds = self.datasets[key]
            try:
                if self._chunk_active:
                    data = self._read_chunked(
                        ds, start, end, episode_idx=self._episode_idx_by_key[key]
                    )
                else:
                    data = ds._read_span(
                        start,
                        end,
                        episode_idx=self._episode_idx_by_key[key],
                        **span_kwargs,
                    )
            except Exception as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"ZarrEpisodePackedDataset exhausted resampling after "
                        f"{attempts} attempts; last error on episode {key} "
                        f"[{start}, {end}): {type(e).__name__}: {e}"
                    ) from e
                logger.warning(
                    "Read failed on episode %s [%d, %d) — %s: %s | "
                    "attempt %d, resampling",
                    key,
                    start,
                    end,
                    type(e).__name__,
                    e,
                    attempts,
                )
                # Pick another index uniformly at random.
                next_idx, _ = get_fallback_idx(
                    idx=idx,
                    candidates=range(len(self.index)),
                    _attempts=attempts,
                    max_attempts=max_attempts + 1,
                    exhausted_error=(
                        "ZarrEpisodePackedDataset cannot find any valid index"
                    ),
                )
                idx = next_idx
                continue

            if self.tail_mode == "hold_pad":
                data = self._hold_pad_to_min(data)
            if "chunk_offset" not in data:
                data["chunk_offset"] = int(start)
            return data


# ---------------------------------------------------------------------- #
# Pack collator
# ---------------------------------------------------------------------- #


def _is_per_frame_tensor(value, seq_len: int) -> bool:
    """A tensor is "per-frame" if its first dim equals the sample's seq_len."""
    if isinstance(value, torch.Tensor):
        return value.ndim >= 1 and value.shape[0] == seq_len
    if isinstance(value, np.ndarray):
        return value.ndim >= 1 and value.shape[0] == seq_len
    return False


PACK_COLLATE_MAX_TOTAL_FRAMES = (
    int(os.environ.get("PACK_COLLATE_MAX_TOTAL_FRAMES", "0")) or None
)


def pack_collate(batch: list[dict]) -> dict:
    """Pack variable-length samples into a single flattened batch.

    Concatenates per-frame tensors along the time axis and emits
    ``cu_seqlens`` (cumulative sequence lengths) so the downstream model can
    recover per-sample boundaries — same convention as FlashAttention's
    variable-length API.

    Output keys:
      - per-frame tensors: shape ``(sum(seq_lens), ...)`` along the
        concatenated time axis.
      - ``"seq_lens"``: ``LongTensor`` of length ``B`` with each sample's
        length.
      - ``"cu_seqlens"``: ``LongTensor`` of length ``B + 1``, ``cu[0] == 0``
        and ``cu[i] == sum(seq_lens[:i])``.
      - ``"max_seq_len"``: int, ``max(seq_lens)``.
      - ``"batch_size"``: int, number of samples in the pack.
      - List-valued keys (e.g. annotations) → ``list[list[str]]`` (one entry
        per sample), preserving the existing ``annotation_collate``
        contract.
      - Scalar-per-sample keys (e.g. ``embodiment``, ``episode_idx``,
        ``chunk_offset``) → ``LongTensor`` of length ``B``.

    ``PACK_COLLATE_MAX_TOTAL_FRAMES`` env var (set in the launcher) caps the
    combined number of frames across all samples in the pack. When the
    incoming ``batch`` would exceed it, trailing samples are dropped until
    ``sum(seq_lens) <= cap``. Always keeps at least one sample so the loader
    can't produce an empty batch.
    """
    if not batch:
        raise ValueError("pack_collate received an empty batch")

    cap = PACK_COLLATE_MAX_TOTAL_FRAMES
    if cap is not None:
        acc = 0
        keep = 0
        for s in batch:
            sl = int(s["seq_len"])
            if keep > 0 and acc + sl > cap:
                break
            acc += sl
            keep += 1
        if keep < len(batch):
            batch = batch[:keep]

    seq_lens = [int(s["seq_len"]) for s in batch]
    cu_seqlens = [0]
    acc = 0
    for sl in seq_lens:
        acc += sl
        cu_seqlens.append(acc)

    # Discover per-frame tensor keys vs. list keys vs. scalar keys, using the
    # first sample as the schema. All samples must share keys.
    first = batch[0]
    per_frame_keys: list[str] = []
    list_keys: list[str] = []
    scalar_keys: list[str] = []

    for k, v in first.items():
        if k == "seq_len":
            continue
        if isinstance(v, list):
            list_keys.append(k)
        elif _is_per_frame_tensor(v, seq_lens[0]):
            per_frame_keys.append(k)
        else:
            scalar_keys.append(k)

    out: dict = {}

    # Per-frame tensors: torch.cat along dim 0.
    for k in per_frame_keys:
        parts = []
        for sample in batch:
            v = sample[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            parts.append(v)
        out[k] = torch.cat(parts, dim=0)

    # List-valued keys: pass through as list-of-lists (matches annotation_collate).
    for k in list_keys:
        out[k] = [sample[k] for sample in batch]

    # Scalar-per-sample keys: collate into a 1D tensor where possible.
    for k in scalar_keys:
        values = [sample[k] for sample in batch]
        try:
            out[k] = default_collate(values)
        except Exception:
            out[k] = values

    out["seq_lens"] = torch.as_tensor(seq_lens, dtype=torch.long)
    out["cu_seqlens"] = torch.as_tensor(cu_seqlens, dtype=torch.long)
    out["max_seq_len"] = int(max(seq_lens))
    out["batch_size"] = len(batch)
    return out


__all__ = [
    "ZarrEpisodePackedDataset",
    "pack_collate",
]
