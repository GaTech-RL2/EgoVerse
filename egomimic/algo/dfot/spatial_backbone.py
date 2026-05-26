"""``DFoTSpatialBackbone`` — DiT3D-style backbone that preserves spatial
structure for image-latent diffusion.

Architectural difference from ``DFoTBackbone``:

* Input: ``(B, T, C, H, W)`` instead of ``(B, T, action_dim)``.
* Each frame is patchified into ``num_patches = (H // patch_size) ** 2``
  spatial tokens via a stride-``patch_size`` ``Conv2d``. The transformer
  sequence becomes ``(B, T * num_patches, d_model)``.
* Per-frame AdaLN cond (noise level + external_cond) is BROADCAST across
  the patches of that frame. Spatial tokens in the same time-step share
  their conditioning, so the model can still treat the per-frame noise
  level as a single value while the spatial tokens attend over each
  other for image structure.
* Output: unpatchified back to ``(B, T, C, H, W)``.

This matches the ``DiT3D`` backbone used in
``kwsong0113/diffusion-forcing-transformer`` (the v2 "History Guided
Video Diffusion" paper), which is the proven architecture for joint
spatio-temporal diffusion.

Use cases:
  * **Image-latent diffusion** (this file's primary target). Feed VAE
    latents ``(B, T, 4, 6, 6)``; backbone denoises in latent space.
  * **Could also handle raw images** if you skip the VAE -- not the
    intended path here, latents are much cheaper.

Action prediction is NOT in the diffusion target for this backbone.
Actions enter via ``external_cond`` (and live in the AdaLN broadcast
across patches). That's the "world-model only" trade-off discussed in
the planning doc; an action-as-extra-token hybrid is a follow-up.

Packed mode: ``cu_seqlens`` is in FRAME units coming in; we multiply by
``num_patches`` before passing to the flash-attn trunk so within-episode
attention is correct in patch-token coordinates.

**Causal attention caveat**: when this backbone runs with ``causal=True``,
the 1D causal mask sees the flattened ``(t, p)`` sequence in row-major
order. That means patch ``p1`` of frame ``t1`` is BEFORE patch ``p2`` of
frame ``t1``, so ``p2`` can see ``p1`` but ``p1`` cannot see ``p2`` —
spatially adjacent patches at the same timestep are wrongly hidden from
each other. For correct DiT3D-style behaviour use ``causal=False`` and
run inference in chunk-mode (full sequence at once). True causal-AR
across time + full attention within space requires factorized attention
which the current Isotropic trunk doesn't support; that's a follow-up.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from egomimic.algo.dfot.embeddings import (
    RandomDropoutCondEmbedding,
    StochasticTimeEmbedding,
)
from egomimic.models.hnet_nets.isotropic_builder import build_isotropic


class _PatchEmbed(nn.Module):
    """Image -> patches via stride-``patch_size`` Conv2d. Each spatial
    cell of the output becomes one token of width ``d_model``.
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):  # (BT, C, H, W) -> (BT, num_patches, d_model)
        x = self.proj(x)
        return rearrange(x, "bt c h w -> bt (h w) c")


class _Unpatchify(nn.Module):
    """Inverse of ``_PatchEmbed``. Tokens of width ``out_c * patch**2``
    repack into ``(BT, out_c, H, W)``.
    """

    def __init__(self, out_channels: int, patch_size: int, grid_size: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.grid_size = int(grid_size)

    def forward(self, tokens):
        # tokens: (BT, num_patches, out_c * patch**2)
        return rearrange(
            tokens,
            "bt (h w) (p q c) -> bt c (h p) (w q)",
            h=self.grid_size,
            w=self.grid_size,
            p=self.patch_size,
            q=self.patch_size,
            c=self.out_channels,
        )


class DFoTSpatialBackbone(nn.Module):
    """Spatial diffusion backbone over ``(B, T, C, H, W)`` tensors.

    Mirrors ``DFoTBackbone``'s contract:
      ``forward(x, noise_levels, external_cond=None, cu_seqlens=None,
                max_seqlen=None, force_uncond=False) -> v_pred``

    Just the SHAPES are spatial.

    Args:
        latent_channels: ``C`` of the latent feature map (e.g. 4).
        latent_size: ``H == W`` of the latent feature map (e.g. 6).
        patch_size: side length of each patch in latent cells. ``H % patch_size == 0``.
            ``patch_size=1`` makes each latent cell its own token.
        action_horizon: unused at runtime (kept for ABI parity).
        d_model: transformer hidden width.
        d_cond: AdaLN cond width.
        cond_dim: external_cond input width (e.g. action+state dim). 0 = no cond.
        time_embed_dim / cond_emb_dim / use_fourier_time / stochastic_time_p
            / cond_dropout_prob: same as ``DFoTBackbone``.
        arch_layout / num_heads / d_intermediate / rotary_emb_dim
            / dropout / resid_dropout / causal: same as ``DFoTBackbone``;
            forwarded to the Isotropic trunk.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 6,
        patch_size: int = 1,
        action_horizon: int = 32,
        d_model: int = 512,
        d_cond: int = 512,
        cond_dim: int = 0,
        time_embed_dim: int = 512,
        cond_emb_dim: int = 512,
        use_fourier_time: bool = True,
        stochastic_time_p: float = 0.0,
        cond_dropout_prob: float = 0.0,
        arch_layout: str = "T12",
        num_heads: int = 8,
        d_intermediate: int = 2048,
        rotary_emb_dim: int = 64,
        dropout: float = 0.1,
        resid_dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(
                f"latent_size ({latent_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.patch_size = int(patch_size)
        self.num_patches_per_side = self.latent_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.cond_emb_dim = int(cond_emb_dim) if self.cond_dim > 0 else 0

        # 1. Patch embedder.
        self.patch_in = _PatchEmbed(
            in_channels=self.latent_channels,
            d_model=self.d_model,
            patch_size=self.patch_size,
        )

        # 2. Noise embedding (per-frame).
        self.time_embed = StochasticTimeEmbedding(
            dim=self.time_embed_dim,
            time_embed_dim=self.time_embed_dim,
            use_fourier=use_fourier_time,
            p=stochastic_time_p,
        )

        # 3. External-cond projection (per-frame).
        if self.cond_dim > 0:
            self.cond_embed = RandomDropoutCondEmbedding(
                cond_dim=self.cond_dim,
                cond_emb_dim=self.cond_emb_dim,
                dropout_prob=cond_dropout_prob,
            )
        else:
            self.cond_embed = None

        # 4. Fuse noise + cond -> d_cond per-frame; broadcast to patches
        #    in ``forward``.
        combined_in = self.time_embed_dim + self.cond_emb_dim
        self.cond_fuse = nn.Sequential(
            nn.Linear(combined_in, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # 5. Isotropic trunk -- reused from DFoTBackbone. Tokens are 1D
        #    over the flattened (time, patch) sequence with shared
        #    AdaLN cond per (time) cluster.
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

        # 6. Output projection: token -> patch_size**2 * latent_channels,
        #    unpatchified back to (BT, C, H, W).
        self.x_out = nn.Linear(
            self.d_model,
            self.patch_size * self.patch_size * self.latent_channels,
        )
        self.unpatchify = _Unpatchify(
            out_channels=self.latent_channels,
            patch_size=self.patch_size,
            grid_size=self.num_patches_per_side,
        )

    # ------------------------------------------------------------------
    # AdaLN cond. Per-frame value, broadcast to ``num_patches`` tokens.
    # ------------------------------------------------------------------

    def _build_per_frame_cond(
        self,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_packed: bool,
        B: Optional[int],
        T: Optional[int],
        force_uncond: bool,
    ) -> torch.Tensor:
        noise_emb = self.time_embed(noise_levels)  # per-frame
        if external_cond is not None and self.cond_embed is not None:
            if not is_packed and external_cond.dim() == 2:
                external_cond = external_cond.unsqueeze(1).expand(B, T, -1)
            cond_emb = self.cond_embed(external_cond)
            if force_uncond:
                cond_emb = torch.zeros_like(cond_emb)
            fused = torch.cat([noise_emb, cond_emb], dim=-1)
        else:
            fused = noise_emb
        return self.cond_fuse(fused)  # per-frame -> (B, T, d_cond) / (T_total, d_cond)

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Returns ``v_pred`` of the same shape as ``x``.

        Padded mode (cu_seqlens is None):
            x:            (B, T, C, H, W)
            noise_levels: (B,) or (B, T)
            external_cond: (B, cond_dim) / (B, T, cond_dim) / None
        Packed mode (cu_seqlens given):
            x:            (T_total, C, H, W)
            noise_levels: (T_total,)
            external_cond: (T_total, cond_dim) / None
            cu_seqlens:   (B+1,) int -- episode boundaries in FRAME units.
            max_seqlen:   int -- longest episode length in frames.
        """
        is_packed = cu_seqlens is not None
        P = self.num_patches  # patches per frame

        if is_packed:
            if x.dim() != 4:
                raise ValueError(
                    f"packed x must be (T_total, C, H, W); got {tuple(x.shape)}"
                )
            if noise_levels.dim() != 1:
                raise ValueError(
                    f"packed noise_levels must be (T_total,); got "
                    f"{tuple(noise_levels.shape)}"
                )

            # 1. Patchify per-frame.
            tokens = self.patch_in(x)            # (T_total, P, d_model)
            tokens = rearrange(tokens, "t p c -> (t p) c")  # (T_total * P, d_model)

            # 2. Per-frame cond -> broadcast to patches.
            cond_per_frame = self._build_per_frame_cond(
                noise_levels, external_cond,
                is_packed=True, B=None, T=None, force_uncond=force_uncond,
            )  # (T_total, d_cond)
            cond_tokens = cond_per_frame.unsqueeze(1).expand(-1, P, -1)
            cond_tokens = rearrange(cond_tokens, "t p c -> (t p) c")  # (T_total*P, d_cond)

            # 3. cu_seqlens + max_seqlen rescale to patch-token units.
            cu_patch = cu_seqlens.to(tokens.device) * P
            msl_patch = int(max_seqlen) * P if max_seqlen is not None else None

            # 4. Run trunk in packed mode.
            tokens = self.trunk(
                tokens,
                cond=cond_tokens,
                cu_seqlens=cu_patch,
                max_seqlen=msl_patch,
            )

            # 5. Output projection + unpatchify.
            tokens = self.x_out(tokens)          # (T_total*P, patch**2*C)
            tokens = rearrange(tokens, "(t p) c -> t p c", p=P)  # (T_total, P, ...)
            v_pred = self.unpatchify(tokens)     # (T_total, C, H, W)
            return v_pred

        # ----- Padded mode -----
        if x.dim() != 5:
            raise ValueError(
                f"padded x must be (B, T, C, H, W); got {tuple(x.shape)}"
            )
        B, T, C, H, W = x.shape

        if noise_levels.dim() == 1:
            noise_levels = noise_levels[:, None].expand(B, T).contiguous()
        elif noise_levels.dim() != 2:
            raise ValueError(
                f"noise_levels must be (B,) or (B, T); got shape "
                f"{tuple(noise_levels.shape)}"
            )

        # 1. Patchify each frame.
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        tokens = self.patch_in(x_flat)  # (B*T, P, d_model)
        tokens = rearrange(tokens, "(b t) p c -> b (t p) c", b=B)

        # 2. Per-frame cond.
        cond_per_frame = self._build_per_frame_cond(
            noise_levels, external_cond,
            is_packed=False, B=B, T=T, force_uncond=force_uncond,
        )  # (B, T, d_cond)
        cond_tokens = cond_per_frame.unsqueeze(2).expand(-1, -1, P, -1)
        cond_tokens = rearrange(cond_tokens, "b t p c -> b (t p) c")

        # 3. Trunk over the flattened (time * patch) sequence.
        tokens = self.trunk(tokens, cond=cond_tokens)

        # 4. Output projection + unpatchify per-frame.
        tokens = self.x_out(tokens)  # (B, T*P, patch**2 * C)
        tokens = rearrange(tokens, "b (t p) c -> (b t) p c", t=T)
        v_pred = self.unpatchify(tokens)  # (B*T, C, H, W)
        v_pred = rearrange(v_pred, "(b t) c h w -> b t c h w", b=B)
        return v_pred
