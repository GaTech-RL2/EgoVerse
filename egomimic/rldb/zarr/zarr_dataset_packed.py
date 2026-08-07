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


class ZarrEpisodePackedDataset(Dataset):
    """One sample per episode (or per chunk of a long episode).

    Each ``__getitem__`` call returns the dict produced by
    ``ZarrDataset._read_span``, i.e. variable-length per-frame tensors plus
    metadata. Samples are intended to flow into :func:`pack_collate`.

    Args:
        datasets: dict of episode-key -> per-episode ``ZarrDataset``. Same
            shape returned by ``LocalEpisodeResolver.resolve`` /
            ``S3EpisodeResolver.resolve``.
        max_seq_len: hard cap on sample length. With ``chunking="sequential"``
            an episode longer than this is split into consecutive chunks of
            length ``max_seq_len`` (last chunk may be shorter). ``None``
            disables chunking.
        min_seq_len: drop chunks shorter than this. Useful for the trailing
            chunk of an episode.
        chunking: ``"sequential"`` for consecutive non-overlapping chunks,
            ``"none"`` to emit each episode as a single sample (raises if
            ``max_seq_len`` is exceeded — caller should pre-filter).
        max_resample_attempts: per ``__getitem__`` cap on resampling after a
            read failure (bad JPEG, transform failure, etc.).
    """

    def __init__(
        self,
        datasets: dict[str, ZarrDataset],
        max_seq_len: int | None = None,
        min_seq_len: int = 1,
        chunking: ChunkingMode = "sequential",
        max_resample_attempts: int | None = None,
    ):
        if not datasets:
            raise ValueError("ZarrEpisodePackedDataset received an empty datasets dict")
        if min_seq_len < 1:
            raise ValueError(f"min_seq_len must be >= 1, got {min_seq_len}")
        if chunking not in ("sequential", "none"):
            raise ValueError(f"Unknown chunking mode: {chunking}")

        self.datasets = datasets
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.chunking = chunking
        self.max_resample_attempts = max_resample_attempts

        # Stable episode-index assignment: sorted by key.
        self._episode_keys: list[str] = sorted(datasets.keys())
        self._episode_idx_by_key: dict[str, int] = {
            key: i for i, key in enumerate(self._episode_keys)
        }

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

    def _build_index(self) -> list[tuple[str, int, int]]:
        index: list[tuple[str, int, int]] = []
        max_len = self.max_seq_len
        for key in self._episode_keys:
            ds = self.datasets[key]
            episode_len = int(ds.total_frames)
            if episode_len < self.min_seq_len:
                continue

            if max_len is None or self.chunking == "none":
                if max_len is not None and episode_len > max_len:
                    raise ValueError(
                        f"Episode {key} has {episode_len} frames > max_seq_len="
                        f"{max_len} but chunking='none'. Either enable "
                        f"chunking or raise max_seq_len."
                    )
                index.append((key, 0, episode_len))
                continue

            # Sequential chunking.
            for start in range(0, episode_len, max_len):
                end = min(start + max_len, episode_len)
                if end - start < self.min_seq_len:
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

    def __getitem__(self, i: int) -> dict:
        attempts = 0
        max_attempts = self.max_resample_attempts or len(self.index)
        idx = i
        while True:
            key, start, end = self.index[idx]
            ds = self.datasets[key]
            try:
                data = ds._read_span(
                    start,
                    end,
                    episode_idx=self._episode_idx_by_key[key],
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

            data["chunk_offset"] = int(start)
            return data


class ZarrAnnotationSpanPackedDataset(ZarrEpisodePackedDataset):
    """One packed sample per annotation span ``{text, start_idx, end_idx}``
    (span-as-episode). Reads the hardcoded ``annotations`` zarr key via each
    episode dataset's ``_load_annotations`` and packs the [start, end) window at
    native per-frame resolution. Ported from EgoVerse-gmm-dualstream for the
    robot-human fold cotrain (paired with fold_span_transforms)."""

    def _build_index(self) -> list[tuple[str, int, int]]:
        index: list[tuple[str, int, int]] = []
        seen: set = set()
        for key in self._episode_keys:
            ds = self.datasets[key]
            try:
                anns = ds._load_annotations()
            except Exception as e:
                logger.warning(
                    "episode %s: annotations load failed (%s); skipping", key, e)
                continue
            total_frames = int(ds.total_frames)
            for ann in anns:
                start = max(int(ann.get("start_idx", -1)), 0)
                end = min(int(ann.get("end_idx", -1)), total_frames)
                if end <= start or (key, start, end) in seen:
                    continue
                seen.add((key, start, end))
                if end - start < self.min_seq_len:
                    continue
                if self.max_seq_len is not None and end - start > self.max_seq_len:
                    if self.chunking == "sequential":
                        for s in range(start, end, self.max_seq_len):
                            e = min(s + self.max_seq_len, end)
                            if e - s >= self.min_seq_len:
                                index.append((key, s, e))
                        continue
                    end = start + self.max_seq_len
                index.append((key, start, end))
        return index


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
    "ZarrAnnotationSpanPackedDataset",
    "pack_collate",
]
