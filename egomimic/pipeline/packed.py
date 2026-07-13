"""packed.py — THE single home for packed-sequence (episode) operations.

Every stage imports these; no module reimplements cu_seqlens bookkeeping.
(Pre-refactor the repo had >=5 independent implementations: GMMLoss
_chunked_targets_packed, _stride_packed, probe unpackers, CVAE head pads,
chunkviz splitters — each a past or future bug site.)

Conventions: packed tensors are (T_total, ...) with cu_seqlens (B+1,) episode
boundaries in the SAME token space as the tensor. B = episodes.
"""
from __future__ import annotations

from typing import Tuple

import torch


def lengths(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(B,) episode lengths."""
    return (cu_seqlens[1:] - cu_seqlens[:-1]).long()


def episode_ids(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(T_total,) episode index of every token."""
    lens = lengths(cu_seqlens)
    return torch.repeat_interleave(
        torch.arange(lens.numel(), device=cu_seqlens.device), lens
    )


def frame_idx(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(T_total,) within-episode index of every token (0-based)."""
    seg = episode_ids(cu_seqlens)
    T = int(cu_seqlens[-1])
    return torch.arange(T, device=cu_seqlens.device) - cu_seqlens[:-1].long()[seg]


def broadcast_per_episode(z: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(B, d) per-episode values -> (T_total, d) per-token."""
    return torch.repeat_interleave(z, lengths(cu_seqlens), dim=0)


def seg_mean(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(T_total,) or (T_total, d) token values -> (B,) / (B, d) episode means."""
    lens = lengths(cu_seqlens).to(x.dtype)
    seg = episode_ids(cu_seqlens)
    B = lens.numel()
    out = x.new_zeros((B,) + x.shape[1:])
    out.index_add_(0, seg, x)
    return out / lens.reshape((B,) + (1,) * (x.dim() - 1))


def pad_per_episode(
    x: torch.Tensor, cu_seqlens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(T_total, d) packed -> ((B, L, d) padded, (B, L) pad_mask True=PAD, lengths)."""
    lens = lengths(cu_seqlens)
    B, L = int(lens.numel()), int(lens.max())
    out = x.new_zeros(B, L, x.shape[-1])
    pad = torch.ones(B, L, dtype=torch.bool, device=x.device)
    for b in range(B):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        out[b, : e - s] = x[s:e]
        pad[b, : e - s] = False
    return out, pad, lens


def chunk_targets(
    actions: torch.Tensor, cu_seqlens: torch.Tensor, chunk_len: int
) -> torch.Tensor:
    """(T_total, D) per-frame actions -> (T_total, C, D) next-C targets,
    index-clamped within each episode (repeat-pad at episode end)."""
    T, C = actions.shape[0], int(chunk_len)
    dev = actions.device
    seg = episode_ids(cu_seqlens)
    ep_end = cu_seqlens[1:].long()[seg] - 1  # (T,) last valid index per token
    base = torch.arange(T, device=dev)[:, None] + torch.arange(C, device=dev)[None, :]
    idx = torch.minimum(base, ep_end[:, None])
    return actions[idx]


def strided_view(
    cu_seqlens: torch.Tensor, stride: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep every ``stride``-th frame per episode.

    Returns (kept_idx (T_kept,) original-frame indices, new_cu (B+1,) in the
    KEPT-token space). Targets should be built from the ORIGINAL actions at
    kept_idx via chunk_targets-style clamping — see TargetBuilder.
    """
    dev = cu_seqlens.device
    kept, new_cu = [], [0]
    for b in range(int(cu_seqlens.numel()) - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        fr = torch.arange(s, e, int(stride), device=dev)
        kept.append(fr)
        new_cu.append(new_cu[-1] + int(fr.numel()))
    return torch.cat(kept), torch.tensor(new_cu, device=dev, dtype=torch.long)
