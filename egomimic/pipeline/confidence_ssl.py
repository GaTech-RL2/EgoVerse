"""Confidence-side predictive supervision for H-Net chunker levels.

The online view is the dechunked A stream multiplied by the chunker's
straight-through selected confidence. Its forward value is unchanged, while
its backward pass gives the router a signed ``(2 * boundary - 1)`` signal.
Targets are detached so the auxiliary task cannot move its own goalposts.
"""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool_by_boundaries(
    values: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool packed frame values into the hard chunks in ``boundary_mask``."""
    if values.ndim != 2:
        raise ValueError(f"values must have shape (T, D), got {tuple(values.shape)}")
    boundary_mask = boundary_mask.reshape(-1).to(device=values.device, dtype=torch.bool)
    if boundary_mask.numel() != values.shape[0]:
        raise ValueError("boundary_mask and values must have the same token length")

    n_chunks = int(boundary_mask.sum())
    if n_chunks == 0:
        return values.new_empty((0, values.shape[-1]))

    chunk_ids = boundary_mask.long().cumsum(dim=0) - 1
    if bool((chunk_ids < 0).any()):
        raise ValueError("the first packed token must be a forced boundary")

    sums = values.new_zeros((n_chunks, values.shape[-1]))
    sums.index_add_(0, chunk_ids, values)
    counts = values.new_zeros((n_chunks,))
    counts.index_add_(0, chunk_ids, torch.ones_like(chunk_ids, dtype=values.dtype))
    return sums / counts.clamp_min(1).unsqueeze(-1)


def _chunk_episode_ids(chunk_cu_seqlens: torch.Tensor) -> torch.Tensor:
    lengths = (chunk_cu_seqlens[1:] - chunk_cu_seqlens[:-1]).long()
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device),
        lengths,
    )


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.LayerNorm(d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


class ConfidenceSSLHead(nn.Module):
    """JEPA-style next-chunk prediction on a chunker's confidence-side output."""

    def __init__(
        self,
        d_model: int,
        proj_dim: int = 256,
        hidden_dim: int | None = None,
        ema_tau: float = 0.99,
        temperature: float = 0.1,
        vic_target_std: float = 1.0,
        enable_pred: bool = True,
        enable_id: bool = True,
        enable_vic: bool = True,
    ):
        super().__init__()
        d_model = int(d_model)
        proj_dim = int(proj_dim)
        hidden_dim = int(hidden_dim or max(d_model, proj_dim))
        if not 0.0 <= ema_tau <= 1.0:
            raise ValueError(f"ema_tau must be in [0, 1], got {ema_tau}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.ema_tau = float(ema_tau)
        self.temperature = float(temperature)
        self.vic_target_std = float(vic_target_std)
        self.enable_pred = bool(enable_pred)
        self.enable_id = bool(enable_id)
        self.enable_vic = bool(enable_vic)
        if not (self.enable_pred or self.enable_id or self.enable_vic):
            raise ValueError("at least one confidence SSL term must be enabled")

        self.online_projector = _mlp(d_model, hidden_dim, proj_dim)
        self.predictor = (
            _mlp(proj_dim, hidden_dim, proj_dim)
            if self.enable_pred or self.enable_id
            else None
        )
        self.identity_projector = (
            _mlp(d_model, hidden_dim, proj_dim)
            if self.enable_id
            else None
        )
        self.target_projector = (
            copy.deepcopy(self.online_projector)
            if self.enable_pred
            else None
        )
        if self.target_projector is not None:
            self.target_projector.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA-update the stop-gradient target from the online projector."""
        if self.target_projector is None:
            return
        tau = self.ema_tau
        for target, online in zip(
            self.target_projector.parameters(),
            self.online_projector.parameters(),
        ):
            target.mul_(tau).add_(online, alpha=1.0 - tau)
        for target, online in zip(
            self.target_projector.buffers(),
            self.online_projector.buffers(),
        ):
            target.copy_(online)

    def forward(
        self,
        confidence_frames: torch.Tensor,
        latent_frames: torch.Tensor,
        raw_frames: torch.Tensor,
        boundary_mask: torch.Tensor,
        chunk_cu_seqlens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return unweighted prediction, identity, and variance losses.

        ``confidence_frames`` is gradient-live through the selected confidence.
        The latent and raw targets are detached inside this method.
        """
        if self.training and self.target_projector is not None:
            self.update_target()

        chunk_cu = chunk_cu_seqlens.to(
            device=confidence_frames.device,
            dtype=torch.long,
        )
        zero = confidence_frames.sum() * 0.0
        n_chunks = int(boundary_mask.sum())
        if n_chunks == 0:
            # Keep enabled trainable heads visible to DDP on degenerate packs.
            for module in (
                self.online_projector,
                self.predictor,
                self.identity_projector,
            ):
                if module is not None:
                    zero = zero + sum(p.sum() * 0.0 for p in module.parameters())
            return {
                "pred": zero,
                "id": zero,
                "vic": zero,
                "n_chunks": zero.detach(),
                "n_pairs": zero.detach(),
            }
        if int(chunk_cu[-1]) != n_chunks:
            raise ValueError(
                f"chunk_cu_seqlens ends at {int(chunk_cu[-1])}, "
                f"but boundary_mask contains {n_chunks} chunks"
            )

        online_frame_z = self.online_projector(confidence_frames.float())
        online_chunks = _pool_by_boundaries(online_frame_z, boundary_mask)

        target_chunks = None
        if self.target_projector is not None:
            with torch.no_grad():
                target_frame_z = self.target_projector(latent_frames.detach().float())
                target_chunks = _pool_by_boundaries(target_frame_z, boundary_mask)
        identity_chunks = None
        if self.identity_projector is not None:
            identity_frame_z = self.identity_projector(raw_frames.detach().float())
            identity_chunks = _pool_by_boundaries(identity_frame_z, boundary_mask)

        episode_ids = _chunk_episode_ids(chunk_cu)
        if episode_ids.numel() != n_chunks:
            raise ValueError("chunk_cu_seqlens is inconsistent with boundary_mask")

        if n_chunks > 1:
            src = torch.arange(n_chunks - 1, device=confidence_frames.device)
            src = src[episode_ids[:-1] == episode_ids[1:]]
        else:
            src = torch.empty(0, device=confidence_frames.device, dtype=torch.long)
        tgt = src + 1

        predicted = None
        if self.predictor is not None:
            # An empty tensor still connects predictor parameters to a zero
            # loss, which avoids DDP unused-parameter failures on short packs.
            predicted = F.normalize(self.predictor(online_chunks[src]), dim=-1)

        if self.enable_pred and src.numel():
            target = F.normalize(target_chunks[tgt], dim=-1)
            pred_loss = (2.0 - 2.0 * (predicted * target).sum(dim=-1)).mean()
        elif self.enable_pred:
            pred_loss = predicted.sum() * 0.0
        else:
            pred_loss = zero

        if self.enable_id and src.numel():
            identities = F.normalize(identity_chunks, dim=-1)
            logits = predicted @ identities.transpose(0, 1)
            logits = logits / self.temperature

            # The positive is the next chunk. Only chunks from other episodes
            # act as negatives, avoiding false negatives within one trajectory.
            allowed = episode_ids.unsqueeze(0) != episode_ids[src].unsqueeze(1)
            allowed[torch.arange(src.numel(), device=src.device), tgt] = True
            has_negative = allowed.sum(dim=1) > 1
            if bool(has_negative.any()):
                masked = logits[has_negative].masked_fill(
                    ~allowed[has_negative],
                    torch.finfo(logits.dtype).min,
                )
                id_loss = F.cross_entropy(masked, tgt[has_negative])
            else:
                id_loss = predicted.sum() * 0.0 + identity_chunks.sum() * 0.0
        elif self.enable_id:
            id_loss = predicted.sum() * 0.0 + identity_chunks.sum() * 0.0
        else:
            id_loss = zero

        if self.enable_vic and online_frame_z.shape[0] > 1:
            std = torch.sqrt(
                online_frame_z.var(dim=0, unbiased=False) + 1e-4
            )
            vic_loss = F.relu(self.vic_target_std - std).mean()
        elif self.enable_vic:
            vic_loss = online_frame_z.sum() * 0.0
        else:
            vic_loss = zero

        return {
            "pred": pred_loss,
            "id": id_loss,
            "vic": vic_loss,
            "n_chunks": zero.new_tensor(float(n_chunks)),
            "n_pairs": zero.new_tensor(float(src.numel())),
        }
