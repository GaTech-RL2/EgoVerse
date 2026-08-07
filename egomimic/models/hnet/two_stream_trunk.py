"""Two-stream (agnostic ‖ specific) transformer trunk for the dual-stream H-Net.

EACH LAYER HOLDS SEPARATE WEIGHTS PER STREAM: per-stream RMSNorm + QKV + out_proj
+ SwiGLU MLP for the agnostic stream (A) and the specific stream (S). The
attention is CROSS-STREAM: per-stream Q/K/V are concatenated into ONE 2T-token
sequence and attended with a channel-aware allow-mask:
  * asym: A-query attends A-keys ONLY; S-query attends BOTH  (A stays agnostic);
  * sym:  every query attends both streams.
So S reads A every layer (and A reads S only in sym) = per-layer cross-attention,
with genuinely separate weights per stream. RoPE on the shared within-episode time
index (A_t, S_t both at position t).

PACKED path only (T_total tokens per stream; episode boundaries + time-causality
live in the (2T,2T) allow-mask built by DualStreamComputeStage). Reuses blocks.py
primitives unchanged: RMSNorm, SwiGLU, _apply_rotary, _RotaryCache.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.blocks import RMSNorm, SwiGLU, _apply_rotary, _RotaryCache


class TwoStreamAttention(nn.Module):
    """Cross-stream attention with separate per-stream QKV / out projections."""

    def __init__(self, d_model, num_heads, rotary_emb_dim, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)
        # separate weights per stream
        self.qkv_A = nn.Linear(d_model, 3 * d_model, bias=True)
        self.qkv_S = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_A = nn.Linear(d_model, d_model, bias=True)
        self.out_S = nn.Linear(d_model, d_model, bias=True)
        self.rotary_emb_dim = int(rotary_emb_dim)
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be <= head_dim"
        self.rotary = _RotaryCache(self.rotary_emb_dim) if self.rotary_emb_dim > 0 else None

    def forward(self, A, S, allow_mask, rope_positions):
        # A, S: (T, d_model) packed. allow_mask: (2T, 2T) bool. rope_positions: (2T,)
        T = A.shape[0]
        H, Dh = self.num_heads, self.head_dim
        qa, ka, va = self.qkv_A(A).reshape(T, 3, H, Dh).unbind(1)  # each (T,H,Dh)
        qs, ks, vs = self.qkv_S(S).reshape(T, 3, H, Dh).unbind(1)
        q = torch.cat([qa, qs], 0)  # (2T, H, Dh)  agnostic first, specific second
        k = torch.cat([ka, ks], 0)
        v = torch.cat([va, vs], 0)
        if self.rotary is not None:
            cos, sin = self.rotary.get(rope_positions, q.dtype)  # (2T, dim/2)
            cos, sin = cos[:, None, :], sin[:, None, :]          # broadcast over heads
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
        # SDPA over the packed 2T sequence: (1, H, 2T, Dh)
        q = q.permute(1, 0, 2)[None]
        k = k.permute(1, 0, 2)[None]
        v = v.permute(1, 0, 2)[None]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=allow_mask[None, None],  # (1,1,2T,2T) bool: True = attend
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out[0].permute(1, 0, 2).reshape(2 * T, self.d_model)  # (2T, d_model)
        return self.out_A(out[:T]), self.out_S(out[T:])


class TwoStreamBlock(nn.Module):
    """Pre-norm two-stream block: separate norm/attn/MLP weights per stream."""

    def __init__(self, d_model, num_heads, d_intermediate, rotary_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1_A, self.norm1_S = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = TwoStreamAttention(d_model, num_heads, rotary_emb_dim, dropout)
        self.norm2_A, self.norm2_S = RMSNorm(d_model), RMSNorm(d_model)
        self.mlp_A = SwiGLU(d_model, d_intermediate)
        self.mlp_S = SwiGLU(d_model, d_intermediate)

    def forward(self, A, S, allow_mask, rope_positions):
        oa, os = self.attn(self.norm1_A(A), self.norm1_S(S), allow_mask, rope_positions)
        A, S = A + oa, S + os
        A = A + self.mlp_A(self.norm2_A(A))
        S = S + self.mlp_S(self.norm2_S(S))
        return A, S


class TwoStreamTrunk(nn.Module):
    """N two-stream blocks + final per-stream norms. Separate weights throughout."""

    def __init__(self, d_model, num_heads, d_intermediate, n_layers, rotary_emb_dim, dropout=0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.blocks = nn.ModuleList([
            TwoStreamBlock(d_model, num_heads, d_intermediate, rotary_emb_dim, dropout)
            for _ in range(int(n_layers))
        ])
        self.norm_f_A, self.norm_f_S = RMSNorm(d_model), RMSNorm(d_model)

    @property
    def height(self) -> int:
        """Residual-stream depth (per stream): each block adds attn + mlp."""
        return 2 * self.n_layers

    def forward(self, A, S, allow_mask, rope_positions):
        for blk in self.blocks:
            A, S = blk(A, S, allow_mask, rope_positions)
        return self.norm_f_A(A), self.norm_f_S(S)
