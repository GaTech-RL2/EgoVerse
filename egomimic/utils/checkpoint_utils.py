"""Checkpoint weight loading that cannot silently leave random weights behind."""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)


def load_checkpoint_weights(
    model: torch.nn.Module, ckpt_path: str | os.PathLike
) -> None:
    """Load a Lightning checkpoint's ``state_dict`` into ``model``, failing when
    the checkpoint lacks any key the model needs and warning about keys the model
    does not use.

    ``strict=True`` would also reject unexpected keys; those are harmless (every
    parameter the model has was still loaded), so they are logged and ignored.
    The file is memory-mapped (as ``MmapCheckpointIO`` does), so the optimizer
    state it also holds is never read into RAM.
    """
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=False, mmap=True
    )
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if result.unexpected_keys:
        log.warning(
            "%s: ignoring %d unexpected key(s) not in the model, e.g. %s",
            ckpt_path,
            len(result.unexpected_keys),
            result.unexpected_keys[:5],
        )
    if result.missing_keys:
        raise RuntimeError(
            f"{ckpt_path}: checkpoint is missing {len(result.missing_keys)} key(s) "
            f"the model needs, e.g. {result.missing_keys[:5]}; the model would run "
            "on random weights. Is this checkpoint from the same model config?"
        )
    log.info("Loaded weights from %s", ckpt_path)
