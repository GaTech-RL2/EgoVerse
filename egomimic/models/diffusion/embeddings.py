"""
Noise-level / conditioning embeddings ported from the DFoT repo.

Only the action-chunk denoising path is retained — image / Fourier-position
embedding helpers from the upstream video pipeline are dropped. Provides
``StochasticTimeEmbedding`` (sinusoidal or Fourier per-token noise embedding
with optional stochastic-unknown-token dropout) and
``RandomDropoutCondEmbedding`` (an MLP wrapper around an embedding tensor
with per-batch dropout).
"""

import math
from typing import Optional

import numpy as np
import torch
from torch import nn


class _TimestepEmbedding(nn.Module):
    """Linear-SiLU-Linear projection from raw embedding dim -> time_embed_dim.

    Mirrors ``diffusers.models.embeddings.TimestepEmbedding`` without taking
    the dependency. Defaults to SiLU activation; bias=True.
    """

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM-style).

    Accepts ``timesteps`` of shape ``(N,)`` or ``(N, M)`` (per-token noise
    levels). Returns ``(N, embedding_dim)`` or ``(N, M, embedding_dim)``.
    """
    if timesteps.dim() not in (1, 2):
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class _Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class _StochasticUnknownTimesteps(_Timesteps):
    """Sinusoidal timestep embedding with optional per-token dropout to an
    ``unknown_token`` parameter. Used by DFoT to teach the model that some
    frames may have an unknown noise level (history-guidance setup).
    """

    def __init__(self, num_channels: int, p: float = 1.0):
        super().__init__(num_channels)
        self.unknown_token = (
            nn.Parameter(torch.randn(1, num_channels)) if p > 0.0 else None
        )
        self.p = p

    def forward(
        self,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = super().forward(timesteps)
        if self.p == 0.0:
            return t_emb
        if self.training or self.p == 1.0 or mask is None:
            mask = torch.rand(t_emb.shape[:-1], device=t_emb.device) < self.p
            mask = mask[..., None].expand_as(t_emb)
            return torch.where(mask, self.unknown_token, t_emb)
        mask = mask[..., None].expand_as(t_emb)
        return torch.where(mask, self.unknown_token, t_emb)


class FourierEmbedding(nn.Module):
    """EDM2-style Fourier embedding for continuous-time noise levels."""

    def __init__(self, num_channels: int, bandwidth: float = 1.0):
        super().__init__()
        self.register_buffer(
            "freqs", 2 * math.pi * torch.randn(num_channels) * bandwidth
        )
        self.register_buffer("phases", 2 * math.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype if x.is_floating_point() else torch.float32
        y = x.to(torch.float32)
        y = y[..., None] * self.freqs.to(torch.float32)
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(out_dtype)


class StochasticTimeEmbedding(nn.Module):
    """Per-token noise-level embedding.

    ``use_fourier=False`` -> sinusoidal + optional ``p>0`` stochastic dropout
    to an unknown-token parameter (discrete-diffusion path).

    ``use_fourier=True``  -> EDM2 Fourier embedding (continuous-diffusion
    path); the ``p`` knob is forbidden in this mode.

    Both modes feed the raw embedding through an MLP projection to
    ``time_embed_dim``.
    """

    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        use_fourier: bool = False,
        p: float = 0.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert p == 0.0, "Fourier embeddings do not support stochastic timesteps"
        self.timesteps = (
            FourierEmbedding(dim, bandwidth=1)
            if use_fourier
            else _StochasticUnknownTimesteps(dim, p)
        )
        self.embedding = _TimestepEmbedding(dim, time_embed_dim)

    def forward(
        self,
        timesteps: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_fourier:
            t_emb = self.timesteps(timesteps)
        else:
            t_emb = self.timesteps(timesteps, mask)
        return self.embedding(t_emb)


class _RandomEmbeddingDropout(nn.Module):
    """Per-batch dropout-to-zero with probability ``p`` during training."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(
        self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            assert not self.training, "embedding mask is only allowed during inference"
            assert mask.ndim == 1, "embedding mask should be of shape (B,)"
        if self.training and self.p > 0:
            mask = torch.rand(emb.shape[:1], device=emb.device) < self.p
        if mask is not None:
            view = mask.view(*mask.shape, *([1] * (emb.ndim - 1)))
            emb = torch.where(view, torch.zeros_like(emb), emb)
        return emb


class RandomDropoutCondEmbedding(nn.Module):
    """MLP projection of external conditioning, with optional per-batch
    dropout-to-zero (classifier-free-guidance-style cond dropping).

    Inputs ``cond`` shape ``(B, ..., cond_dim)`` -> output ``(B, ...,
    cond_emb_dim)``. Dropout is applied per-batch (uniform across all
    trailing dims), matching the upstream DFoT contract.
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.embedding = _TimestepEmbedding(cond_dim, cond_emb_dim)
        self.dropout = (
            _RandomEmbeddingDropout(p=dropout_prob) if dropout_prob > 0 else None
        )

    def forward(
        self, cond: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.embedding(cond)
        if self.dropout is not None:
            out = self.dropout(out, mask)
        return out
