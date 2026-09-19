"""Concatenate compatible model inputs along their existing batch dimension."""

from collections import defaultdict

import torch


def batch_signature(value):
    """Describe structure, non-batch shapes, dtypes and devices without copies."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            raise ValueError("Batched tensors must have a leading batch dimension")
        return (torch.Tensor, tuple(value.shape[1:]), value.dtype, value.device)
    if isinstance(value, dict):
        return tuple((key, batch_signature(value[key])) for key in sorted(value))
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return (list, str)
    if value is None:
        return None
    raise TypeError(f"Unsupported batch value: {type(value).__name__}")


def concatenate_batches(values):
    """Concatenate tensors, nested dictionaries and per-sample prompt lists."""
    first = values[0]
    if len(values) == 1:
        return first
    if torch.is_tensor(first):
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {
            key: concatenate_batches([value[key] for value in values]) for key in first
        }
    if isinstance(first, list):
        return [item for value in values for item in value]
    if first is None:
        return None
    raise TypeError(f"Unsupported batch value: {type(first).__name__}")


def compatible_groups(batches):
    """Yield keys of batches that can be concatenated without padding."""
    groups = defaultdict(list)
    for key, value in batches.items():
        groups[batch_signature(value)].append(key)
    return groups.values()


def sample_mean(losses, sizes):
    """Average per-dataset means without cancelling the sampling proportions."""
    if not losses:
        raise ValueError("Cannot reduce an empty batch")
    return sum(loss * size for loss, size in zip(losses, sizes)) / sum(sizes)
