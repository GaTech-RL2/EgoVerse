"""
Abstract base class for transforms that take a list of input zarr keys on a
single episode store and produce a list of output zarr keys.

Subclasses load their model once and reuse it across episodes via repeated
``process_episode()`` calls. Subclasses are responsible for batched compute
inside ``compute()``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import zarr

logger = logging.getLogger(__name__)


class ZarrKeyTransform(ABC):
    """
    Abstract transform: a list of input zarr keys -> a list of output zarr keys.

    A single instance loads its model once and is reused across episodes via
    repeated process_episode() calls. Subclasses implement compute() and are
    responsible for batching their own GPU inference.

    Output naming: when ``output_keys`` is omitted, defaults are derived from
    each input key by replacing the leading dotted segment with
    :attr:`OUTPUT_PREFIX`. Subclasses must set OUTPUT_PREFIX to a non-empty
    string to support this. Example: with ``OUTPUT_PREFIX = "dinov2"``,
    ``images.front_1`` -> ``dinov2.front_1``. Pass ``output_keys``
    explicitly to override.
    """

    OUTPUT_PREFIX: str = ""

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        batch_size: int = 32,
        overwrite: bool = False,
        device: str | torch.device = "cuda",
        chunk_timesteps: int = 100,
    ):
        if not input_keys:
            raise ValueError("input_keys must be non-empty")
        if output_keys is None:
            output_keys = [self.default_output_key(k) for k in input_keys]
        if not output_keys:
            raise ValueError("output_keys must be non-empty")
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.device = torch.device(device)
        self.chunk_timesteps = chunk_timesteps
        self._setup_done = False

    @classmethod
    def default_output_key(cls, input_key: str) -> str:
        """
        Derive a default output key from an input key by swapping the leading
        dotted segment for ``cls.OUTPUT_PREFIX``. ``images.front_1`` becomes
        ``dinov2.front_1`` for a transform whose ``OUTPUT_PREFIX`` is
        ``"dinov2"``. If the input key has no dot, the prefix is prepended.
        """
        if not cls.OUTPUT_PREFIX:
            raise ValueError(
                f"{cls.__name__} does not define OUTPUT_PREFIX; pass output_keys "
                "explicitly."
            )
        head, sep, tail = input_key.partition(".")
        remainder = tail if sep else head
        return f"{cls.OUTPUT_PREFIX}.{remainder}"

    @abstractmethod
    def setup(self) -> None:
        """Load model / tokenizer / processor. Called lazily before first episode."""

    @abstractmethod
    def compute(self, store: zarr.Group) -> dict[str, np.ndarray]:
        """
        Read self.input_keys from the store and return a dict mapping every
        entry in self.output_keys to its full per-frame ndarray (T, ...).

        Implementations are responsible for batching reads + GPU inference.
        """

    def process_episode(
        self,
        episode_path: Path | str,
        output_store_path: Path | str | None = None,
    ) -> None:
        """Compute outputs from ``episode_path`` and write them to a store.

        By default outputs are written back into the source store (in-place,
        ``mode="a"``). When ``output_store_path`` is given, the source store is
        read read-only and outputs are written to a *separate* per-episode store
        at that path (created if absent) — used to mirror embeddings into a
        writable folder when the canonical store is read-only. The mirror store
        carries ``total_frames`` so it stays self-describing for downstream
        readers that strip the chunk padding.
        """
        episode_path = Path(episode_path)
        if not self._setup_done:
            self.setup()
            self._setup_done = True

        in_place = output_store_path is None
        src = zarr.open_group(
            str(episode_path), mode=("a" if in_place else "r"), zarr_format=3
        )

        for k in self.input_keys:
            if k not in src:
                raise KeyError(f"Input key '{k}' missing from {episode_path}")

        out = (
            src
            if in_place
            else zarr.open_group(str(output_store_path), mode="a", zarr_format=3)
        )

        existing = [k for k in self.output_keys if k in out]
        if existing and not self.overwrite:
            logger.info(
                "Skipping %s — output keys already exist: %s",
                output_store_path or episode_path,
                existing,
            )
            return
        for k in existing:
            del out[k]

        outputs = self.compute(src)

        missing = set(self.output_keys) - set(outputs.keys())
        if missing:
            raise RuntimeError(f"compute() did not return expected outputs: {missing}")

        if not in_place and "total_frames" in src.attrs:
            out.attrs["total_frames"] = int(src.attrs["total_frames"])

        for key in self.output_keys:
            self._write_numeric(out, key, np.asarray(outputs[key]))

        n = len(next(iter(outputs.values())))
        logger.info(
            "Wrote %s for %s (frames=%d)",
            self.output_keys,
            output_store_path or episode_path,
            n,
        )

    def _write_numeric(self, store: zarr.Group, key: str, arr: np.ndarray) -> None:
        """Write a numeric (T, ...) array, padding to chunk_timesteps for shard alignment."""
        if arr.ndim < 1:
            raise ValueError(
                f"Output '{key}' must be at least 1D, got shape {arr.shape}"
            )

        n = arr.shape[0]
        frame_shape = arr.shape[1:]
        chunk_n = max(1, min(self.chunk_timesteps, n))
        padded_n = ((n + chunk_n - 1) // chunk_n) * chunk_n
        if padded_n > n:
            pad = np.zeros((padded_n - n,) + frame_shape, dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)

        chunk_shape = (chunk_n,) + frame_shape
        store.create_array(
            key,
            data=arr,
            chunks=chunk_shape,
            shards=arr.shape,
        )

        attrs = dict(store.attrs)
        features = dict(attrs.get("features", {}))
        features[key] = {
            "dtype": str(arr.dtype),
            "shape": list(frame_shape),
            "names": [f"dim_{i}" for i in range(len(frame_shape))],
        }
        attrs["features"] = features
        store.attrs.update(attrs)
