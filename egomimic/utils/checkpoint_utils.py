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


def init_weights_from_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | os.PathLike,
    *,
    min_fraction: float = 0.5,
) -> dict:
    """Fine-tune initialization: copy every tensor of a Lightning checkpoint's
    ``state_dict`` whose name and shape match ``model``; leave the rest at their
    fresh init. Unlike ``ckpt_path`` (a full resume: optimizer, scheduler,
    epoch) nothing but weights is read, so a new embodiment's stems / head and
    a different action width start fresh while the shared trunk transfers.

    Raises when fewer than ``min_fraction`` of the model's parameters (by
    element count) were loaded: a wrong or unrelated checkpoint would otherwise
    "fine-tune" from random init without a word. Returns a report dict.
    """
    checkpoint = torch.load(
        ckpt_path, map_location="cpu", weights_only=False, mmap=True
    )
    ckpt_sd = checkpoint["state_dict"]
    own_sd = model.state_dict()
    loaded = {
        k: v for k, v in ckpt_sd.items() if k in own_sd and own_sd[k].shape == v.shape
    }
    shape_mismatch = sorted(
        k for k, v in ckpt_sd.items() if k in own_sd and own_sd[k].shape != v.shape
    )
    unused = sorted(k for k in ckpt_sd if k not in own_sd)
    fresh = sorted(k for k in own_sd if k not in loaded)
    model.load_state_dict(loaded, strict=False)

    total = sum(v.numel() for v in own_sd.values())
    covered = sum(own_sd[k].numel() for k in loaded)
    fraction = covered / max(total, 1)
    report = {
        "loaded": len(loaded),
        "shape_mismatch": shape_mismatch,
        "unused": unused,
        "fresh": fresh,
        "fraction": fraction,
    }
    log.info(
        "init_weights_ckpt %s: loaded %d tensors (%.1f%% of the model's elements); "
        "%d shape-mismatched, %d unused, %d left at fresh init, e.g. %s",
        ckpt_path,
        len(loaded),
        100 * fraction,
        len(shape_mismatch),
        len(unused),
        len(fresh),
        fresh[:5],
    )
    if fraction < min_fraction:
        raise RuntimeError(
            f"init_weights_ckpt {ckpt_path}: only {100 * fraction:.1f}% of the "
            f"model's weights matched (need {100 * min_fraction:.0f}%); is it "
            f"from the same model family? e.g. fresh {fresh[:5]}, "
            f"mismatched {shape_mismatch[:5]}"
        )
    return report
