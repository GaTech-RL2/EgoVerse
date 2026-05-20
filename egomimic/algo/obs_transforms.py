"""Post-normalize observation transforms applied in
``HNet.process_batch_for_training`` (train-only).

Each transform is a callable that takes a dict-of-tensors (the per-embodiment
batch with already-normalized keys like ``state_agent_obj``, ``front_img_1``,
``actions``, plus packed metadata ``cu_seqlens``, ``seq_lens``, ...) and
returns the same dict with selected keys modified in-place / replaced. The
transform must preserve metadata keys (do NOT drop ``cu_seqlens`` etc.).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch


class ObsTransform:
    """Base class — subclass and override ``__call__``."""

    def __call__(self, batch: dict) -> dict:
        raise NotImplementedError


class GaussianObsNoise(ObsTransform):
    """Add Gaussian noise to the specified keys in the (already-normalized) batch.

    Args:
        keys: list of dict keys to apply noise to. Each must be a floating-point
            tensor in the batch.
        std: noise standard deviation (in the normalized space — typically a
            small fraction like 0.02 since proprio has unit-ish std after norm).
    """

    def __init__(self, keys: Iterable[str], std: float):
        self.keys = list(keys)
        self.std = float(std)
        if self.std < 0.0:
            raise ValueError(f"std must be >= 0, got {self.std}")

    def __call__(self, batch: dict) -> dict:
        if self.std == 0.0:
            return batch
        for k in self.keys:
            if k not in batch:
                continue
            v = batch[k]
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                batch[k] = v + torch.randn_like(v) * self.std
        return batch
