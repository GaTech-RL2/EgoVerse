"""``DFoTDiT3DBackbone`` — DiT3D backbone for spatial video diffusion.

Matches the architecture from ``kwsong0113/diffusion-forcing-transformer``
(the "History Guided Video Diffusion" paper, ICML 2025):

* **AdaLN-Zero** conditioning: each block produces (shift, scale, gate)
  from the conditioning vector. The gate is zero-initialized so residual
  branches start as identity, stabilising early training.
* **RoPE 3D** positional encoding: separate rotary frequencies for
  temporal and 2D spatial axes. No learned position embeddings.
* **DiT-style weight init**: Xavier for linear layers, zero-init for
  AdaLN modulation outputs and final projection layer.
* **pred_v objective**: the model predicts the velocity parameterisation.

Input: ``(B, T, C, H, W)`` VAE latents.
Output: ``v_pred`` of the same shape.

Per-frame AdaLN cond (noise level + optional external_cond) is broadcast
across the spatial patches of each frame, matching the reference impl.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from egomimic.algo.dfot.embeddings import (
    RandomDropoutCondEmbedding,
    StochasticTimeEmbedding,
    get_timestep_embedding,
)

# ---------------------------------------------------------------------------
# RoPE 3D
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryEmbedding3D(nn.Module):
    """Axial RoPE for (T, H, W) token grids, flattened to 1D.

    Stores per-axis inverse frequencies and computes the full 3D grid
    on-the-fly for the actual input temporal length (which may vary
    between batches due to packed episode padding).
    """

    def __init__(self, dim: int, sizes: Tuple[int, int, int], theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        match half_dim % 3:
            case 0:
                dims = (half_dim // 3,) * 3
            case 1:
                dims = (half_dim // 3 + 1, half_dim // 3, half_dim // 3)
            case 2:
                dims = (half_dim // 3, half_dim // 3 + 1, half_dim // 3 + 1)
        self._dims = dims
        self._spatial_sizes = (sizes[1], sizes[2])
        # Store per-axis inverse frequencies as buffers.
        for i, axis_dim in enumerate(dims):
            inv_freq = 1.0 / (theta ** (torch.arange(0, axis_dim * 2, 2).float() / (axis_dim * 2)))
            self.register_buffer(f"inv_freq_{i}", inv_freq, persistent=False)

    def _build_freqs(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        H, W = self._spatial_sizes
        sizes = (T, H, W)
        all_freqs = []
        for axis_idx, axis_dim in enumerate(self._dims):
            inv_freq = getattr(self, f"inv_freq_{axis_idx}").to(device)
            pos = torch.arange(sizes[axis_idx], device=device, dtype=dtype)
            freqs = torch.einsum("p, f -> p f", pos, inv_freq.to(dtype))
            axis_freqs = torch.zeros(*sizes, axis_dim, device=device, dtype=dtype)
            for i in range(sizes[axis_idx]):
                idx = [slice(None)] * 3
                idx[axis_idx] = i
                axis_freqs[tuple(idx)] = freqs[i]
            all_freqs.append(axis_freqs)
        all_freqs = torch.cat(all_freqs, dim=-1)
        all_freqs = rearrange(all_freqs, "t h w d -> (t h w) d")
        return repeat(all_freqs, "n d -> n (d r)", r=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        H, W = self._spatial_sizes
        # When non-grid tokens (e.g. per-frame action tokens) are appended to
        # the sequence, ``_n_grid_tokens`` tells us how many leading tokens are
        # the (T, H, W) image grid. The appended tokens get identity rotation
        # (zero freqs) and rely on a learned positional embedding instead.
        n_grid = getattr(self, "_n_grid_tokens", None)
        if n_grid is None:
            n_grid = seq_len
        T = n_grid // (H * W)
        grid_freqs = self._build_freqs(T, x.device, x.dtype)[:n_grid]
        if seq_len > n_grid:
            pad = torch.zeros(
                seq_len - n_grid, grid_freqs.shape[-1],
                device=x.device, dtype=x.dtype,
            )
            freqs = torch.cat([grid_freqs, pad], dim=0)
        else:
            freqs = grid_freqs[:seq_len]
        return x * freqs.cos() + _rotate_half(x) * freqs.sin()


# ---------------------------------------------------------------------------
# AdaLN-Zero
# ---------------------------------------------------------------------------

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """Produces (shift, scale, gate) from conditioning — gate is zero-init."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return _modulate(self.norm(x), shift, scale), gate


class AdaLayerNorm(nn.Module):
    """Produces (shift, scale) only — used in the final layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return _modulate(self.norm(x), shift, scale)


# ---------------------------------------------------------------------------
# DiT Block
# ---------------------------------------------------------------------------

class DiTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 qk_norm: bool = False, rope: Optional[RotaryEmbedding3D] = None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero conditioning + optional RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 rope: Optional[RotaryEmbedding3D] = None):
        super().__init__()
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = DiTAttention(hidden_size, num_heads, qkv_bias=True, rope=rope)
        self.use_mlp = mlp_ratio is not None
        self.norm2 = AdaLayerNormZero(hidden_size)
        mlp_hidden = int(hidden_size * (mlp_ratio or 4.0))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in [self.attn, self.mlp]:
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x


class DiTFinalLayer(nn.Module):
    """Final layer: AdaLN (no gate) + zero-init linear projection."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = AdaLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x, c))


# ---------------------------------------------------------------------------
# Patch embed / unpatchify
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size,
                              kernel_size=patch_size, stride=patch_size)
        # Xavier init like nn.Linear (DiT convention)
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (BT, C, H, W) -> (BT, num_patches, hidden_size)
        x = self.proj(x)
        return rearrange(x, "bt c h w -> bt (h w) c")


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class DFoTDiT3DBackbone(nn.Module):
    """DiT3D backbone matching the History Guided Video Diffusion architecture.

    Contract matches ``DFoTSpatialBackbone``:
      ``forward(x, noise_levels, external_cond=None, ...) -> v_pred``

    Args:
        latent_channels: C of the VAE latent (e.g. 4).
        latent_size: H == W of the VAE latent (e.g. 6).
        patch_size: spatial patch side length (1 = each latent cell is a token).
        action_horizon: T dimension (max frames).
        d_model: transformer hidden width.
        d_cond: AdaLN conditioning width (noise_emb + cond_emb projected to this).
        cond_dim: external_cond input width. 0 = no external cond.
        depth: number of DiT blocks.
        num_heads: attention heads.
        mlp_ratio: FFN expansion ratio.
        time_embed_dim: noise embedding MLP output width.
        cond_emb_dim: external cond embedding MLP output width.
        use_fourier_time: Fourier vs sinusoidal noise embedding.
        stochastic_time_p: dropout prob for stochastic unknown timesteps.
        cond_dropout_prob: CFG-style cond dropout prob.
        dropout: unused (kept for config compat).
        resid_dropout: unused (kept for config compat).
        causal: unused (DiT3D is always non-causal; kept for config compat).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 6,
        patch_size: int = 1,
        action_horizon: int = 32,
        d_model: int = 384,
        d_cond: int = 384,
        cond_dim: int = 0,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        cond_emb_dim: int = 256,
        use_fourier_time: bool = False,
        stochastic_time_p: float = 0.0,
        cond_dropout_prob: float = 0.1,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        causal: bool = False,
        action_token_dim: int = 0,
    ):
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(f"latent_size ({latent_size}) must be divisible by patch_size ({patch_size})")

        self.latent_channels = int(latent_channels)
        self.latent_size = int(latent_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.latent_size // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        self.cond_dim = int(cond_dim)
        self.depth = int(depth)

        out_channels = self.patch_size ** 2 * self.latent_channels

        # 1. Patch embedder
        self.patch_embed = PatchEmbed(self.latent_channels, self.d_model, self.patch_size)

        # 2. RoPE 3D: (T, grid_h, grid_w) axes
        rope = RotaryEmbedding3D(
            dim=self.d_model // num_heads,
            sizes=(self.action_horizon, self.grid_size, self.grid_size),
        )

        # 3. Noise embedding -> d_cond (matches reference: both embeddings
        #    project to d_cond independently, then are ADDED together)
        self.time_embed = StochasticTimeEmbedding(
            dim=int(time_embed_dim),
            time_embed_dim=self.d_cond,
            use_fourier=use_fourier_time,
            p=stochastic_time_p,
        )

        # 4. External cond embedding -> d_cond (same width as noise emb)
        if self.cond_dim > 0:
            self.cond_embed = RandomDropoutCondEmbedding(
                cond_dim=self.cond_dim,
                cond_emb_dim=self.d_cond,
                dropout_prob=cond_dropout_prob,
            )
        else:
            self.cond_embed = None

        # Init embedding MLPs (reference uses normal std=0.02)
        for module in [self.time_embed]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if self.cond_embed is not None:
            for m in self.cond_embed.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # 6. DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.d_model, int(num_heads), float(mlp_ratio), rope=rope)
            for _ in range(self.depth)
        ])

        # 7. Final layer (zero-init output)
        self.final_layer = DiTFinalLayer(self.d_model, out_channels)

        # 8. Optional per-frame action token (2D POLICY). When
        # action_token_dim > 0 the backbone additionally diffuses one action
        # token per frame: the noisy action is projected to d_model, appended
        # after the T*P image patch tokens (identity RoPE + learned per-frame
        # position), co-attended with the image patches through the shared DiT
        # blocks, and projected back to action-v. forward() then returns the
        # tuple (v_image, v_action).
        self.action_token_dim = int(action_token_dim)
        self._rope = rope  # forward sets _n_grid_tokens on this shared instance
        if self.action_token_dim > 0:
            self.action_in_proj = nn.Linear(self.action_token_dim, self.d_model)
            # Learned "this is an action token" marker. Temporal position is a
            # dynamic sinusoidal embedding (computed per-T in forward) so it
            # works for any episode length, not just action_horizon.
            self.action_type_embed = nn.Parameter(
                torch.randn(1, 1, self.d_model) * 0.02
            )
            self.action_out = nn.Linear(self.d_model, self.action_token_dim)
            nn.init.zeros_(self.action_out.weight)
            nn.init.zeros_(self.action_out.bias)

    def _build_cond(
        self,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        is_packed: bool,
        B: Optional[int],
        T: Optional[int],
        force_uncond: bool,
    ) -> torch.Tensor:
        """Build per-frame cond via addition (matches reference)."""
        emb = self.time_embed(noise_levels)
        if external_cond is not None and self.cond_embed is not None:
            if not is_packed and external_cond.dim() == 2:
                external_cond = external_cond.unsqueeze(1).expand(B, T, -1)
            cond_emb = self.cond_embed(external_cond)
            if force_uncond:
                cond_emb = torch.zeros_like(cond_emb)
            emb = emb + cond_emb
        return emb

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        force_uncond: bool = False,
        action: Optional[torch.Tensor] = None,
    ):
        """Returns ``v_image`` (B, T, C, H, W), or the tuple
        ``(v_image, v_action)`` when ``action_token_dim > 0`` and a noisy
        ``action`` (B, T, action_token_dim) is supplied."""
        is_packed = cu_seqlens is not None
        P = self.num_patches

        if is_packed:
            raise NotImplementedError(
                "Packed mode not yet supported for DiT3D backbone. "
                "Use padded (B, T, C, H, W) input."
            )

        if x.dim() != 5:
            raise ValueError(f"Expected (B, T, C, H, W); got {tuple(x.shape)}")
        B, T, C, H, W = x.shape

        if noise_levels.dim() == 1:
            noise_levels = noise_levels[:, None].expand(B, T).contiguous()

        # 1. Patchify
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        tokens = self.patch_embed(x_flat)  # (B*T, P, d_model)
        tokens = rearrange(tokens, "(b t) p c -> b (t p) c", b=B)

        # 2. Per-frame cond (B, T, d_cond), broadcast to image patches.
        frame_cond = self._build_cond(
            noise_levels, external_cond,
            is_packed=False, B=B, T=T, force_uncond=force_uncond,
        )
        img_cond = rearrange(
            frame_cond.unsqueeze(2).expand(-1, -1, P, -1), "b t p c -> b (t p) c"
        )

        # 2b. Optional per-frame action token (2D policy). Append after the
        # image patches; give it the frame's cond + a learned position; mark
        # the grid length so RoPE rotates only the image tokens.
        has_action = self.action_token_dim > 0 and action is not None
        if has_action:
            # Dynamic sinusoidal temporal position (any T) + learned type marker.
            frame_idx = torch.arange(T, device=x.device, dtype=torch.float32)
            act_pos = get_timestep_embedding(frame_idx, self.d_model).to(tokens.dtype)
            act_tok = (
                self.action_in_proj(action)
                + act_pos.unsqueeze(0)
                + self.action_type_embed
            )
            tokens = torch.cat([tokens, act_tok], dim=1)        # (B, T*P + T, d)
            cond = torch.cat([img_cond, frame_cond], dim=1)     # (B, T*P + T, d_cond)
            self._rope._n_grid_tokens = T * P
        else:
            cond = img_cond
            self._rope._n_grid_tokens = None

        # 3. DiT blocks (shared, bidirectional self-attention)
        for block in self.blocks:
            tokens = block(tokens, cond)

        # 4. Split image / action, final layer + unpatchify the image stream.
        if has_action:
            img_tokens = tokens[:, : T * P]
            act_tokens = tokens[:, T * P:]
        else:
            img_tokens = tokens

        img_tokens = self.final_layer(img_tokens, img_cond)
        img_tokens = rearrange(img_tokens, "b (t p) c -> (b t) p c", t=T)
        v_pred = rearrange(
            img_tokens,
            "bt (h w) (p q c) -> bt c (h p) (w q)",
            h=self.grid_size,
            w=self.grid_size,
            p=self.patch_size,
            q=self.patch_size,
            c=self.latent_channels,
        )
        v_pred = rearrange(v_pred, "(b t) c h w -> b t c h w", b=B)

        if has_action:
            v_action = self.action_out(act_tokens)  # (B, T, action_token_dim)
            return v_pred, v_action
        return v_pred
