"""
DFoT denoising backbone: wraps the existing ``Isotropic`` trunk with input
projections, per-token noise-level embedding, external-cond projection +
broadcast, and an output head. Outputs match the discrete/continuous
diffusion contract: ``backbone(x, noise_levels, external_cond) -> (B, T, A)``.
"""

from typing import Optional

import torch
from torch import nn

from egomimic.algo.dfot.embeddings import (
    RandomDropoutCondEmbedding,
    StochasticTimeEmbedding,
)
from egomimic.models.hnet_nets.isotropic_builder import build_isotropic


class DFoTBackbone(nn.Module):
    """Action-chunk denoiser with per-token AdaLN cond.

    Inputs:
        x:               ``(B, T, action_dim)`` noisy action chunk.
        noise_levels:    ``(B, T)`` per-token noise level. Discrete-diffusion
                         path passes longs in ``[0, timesteps)``; continuous
                         path passes ``precond_scale * logSNR`` floats.
        external_cond:   ``(B, cond_dim)`` or ``(B, T, cond_dim)`` obs cond.
                         ``None`` to skip.

    Steps:
      1. Project ``x`` (action_dim -> d_model).
      2. Embed ``noise_levels`` per-token to ``noise_emb`` ``(B, T, time_embed_dim)``.
      3. Project ``external_cond`` to ``cond_emb`` ``(B, T, cond_emb_dim)``,
         broadcasting from ``(B, cond_dim)`` if needed.
      4. Concatenate ``noise_emb`` + ``cond_emb`` and project to ``d_cond``
         (the per-block AdaLN cond width).
      5. Run through the wrapped ``Isotropic`` trunk with per-token AdaLN.
      6. Output project back to ``action_dim``.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int = 128,
        d_cond: int = 128,
        cond_dim: int = 0,
        time_embed_dim: int = 128,
        cond_emb_dim: int = 128,
        use_fourier_time: bool = False,
        stochastic_time_p: float = 0.0,
        cond_dropout_prob: float = 0.0,
        arch_layout: str = "T6",
        num_heads: int = 4,
        d_intermediate: int = 512,
        rotary_emb_dim: int = 0,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.cond_emb_dim = int(cond_emb_dim) if cond_dim > 0 else 0

        # 1. Action input projection.
        self.x_in = nn.Linear(self.action_dim, self.d_model)

        # 2. Noise-level embedding (per-token). ``dim`` is the raw sinusoidal
        # /Fourier dimension; ``time_embed_dim`` is the MLP-projected output
        # dim used downstream.
        self.time_embed = StochasticTimeEmbedding(
            dim=self.time_embed_dim,
            time_embed_dim=self.time_embed_dim,
            use_fourier=use_fourier_time,
            p=stochastic_time_p,
        )

        # 3. External-cond projection.
        if self.cond_dim > 0:
            self.cond_embed = RandomDropoutCondEmbedding(
                cond_dim=self.cond_dim,
                cond_emb_dim=self.cond_emb_dim,
                dropout_prob=cond_dropout_prob,
            )
        else:
            self.cond_embed = None

        # 4. Combine noise + cond -> d_cond for AdaLN.
        combined_in = self.time_embed_dim + self.cond_emb_dim
        self.cond_fuse = nn.Sequential(
            nn.Linear(combined_in, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # 5. Isotropic trunk with per-token AdaLN cond.
        self.trunk = build_isotropic(
            spec={
                "arch_layout": arch_layout,
                "d_model": self.d_model,
                "d_intermediate": int(d_intermediate),
                "num_heads": int(num_heads),
                "cond": True,
                "cond_mode": "adaln",
                "rotary_emb_dim": int(rotary_emb_dim),
                "dropout": float(dropout),
                "resid_dropout": float(resid_dropout),
            },
            d_cond=self.d_cond,
            causal=bool(causal),
        )

        # 6. Output head.
        self.x_out = nn.Linear(self.d_model, self.action_dim)

    def _broadcast_cond(
        self, external_cond: Optional[torch.Tensor], B: int, T: int
    ) -> Optional[torch.Tensor]:
        if external_cond is None or self.cond_embed is None:
            return None
        if external_cond.dim() == 2:
            external_cond = external_cond.unsqueeze(1).expand(B, T, -1)
        elif external_cond.dim() != 3:
            raise ValueError(
                f"external_cond must be (B, cond_dim) or (B, T, cond_dim); "
                f"got shape {tuple(external_cond.shape)}"
            )
        return self.cond_embed(external_cond)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"x must be (B, T, action_dim); got shape {tuple(x.shape)}"
            )
        B, T, _ = x.shape
        if noise_levels.dim() == 1:
            # Broadcast a (B,) noise level to per-token (B, T).
            noise_levels = noise_levels[:, None].expand(B, T).contiguous()
        elif noise_levels.dim() != 2:
            raise ValueError(
                f"noise_levels must be (B,) or (B, T); got shape "
                f"{tuple(noise_levels.shape)}"
            )

        # 1. Project x.
        h = self.x_in(x)

        # 2. Per-token noise embedding -> (B, T, time_embed_dim).
        noise_emb = self.time_embed(noise_levels)

        # 3. External cond -> (B, T, cond_emb_dim) or None.
        cond_emb = self._broadcast_cond(external_cond, B=B, T=T)

        # 4. Fuse.
        if cond_emb is not None:
            fused = torch.cat([noise_emb, cond_emb], dim=-1)
        else:
            fused = noise_emb
        cond = self.cond_fuse(fused)  # (B, T, d_cond)

        # 5. Trunk with per-token AdaLN.
        h = self.trunk(h, cond=cond)

        # 6. Project to action.
        return self.x_out(h)
