"""Shared tensor-batch and action-shape helpers."""

from __future__ import annotations

import torch


def clone_batch(batch):
    """Recursively clone tensors in nested mappings, preserving other values."""
    if isinstance(batch, dict):
        return {key: clone_batch(value) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.clone()
    return batch


def extract_xyz(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 6DoF actions into position and rotation components.

    Supported widths are 6/7 for one arm and 12/14 for two arms. The
    gripper component in widths 7 and 14 is intentionally ignored.
    """
    width = x.shape[-1]
    if width in (6, 7):
        return x[..., :3], x[..., 3:6]
    if width == 12:
        xyz = torch.cat([x[..., :3], x[..., 6:9]], dim=-1)
        rot = torch.cat([x[..., 3:6], x[..., 9:12]], dim=-1)
        return xyz, rot
    if width == 14:
        xyz = torch.cat([x[..., :3], x[..., 7:10]], dim=-1)
        rot = torch.cat([x[..., 3:6], x[..., 10:13]], dim=-1)
        return xyz, rot
    raise ValueError(f"Unexpected shape for 6DoF input: {x.shape}")
