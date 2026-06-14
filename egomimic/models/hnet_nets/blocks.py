"""
Transformer + Mamba2 blocks for the vendored HNet.

Arch types (matching upstream layout-string vocabulary):
    't' : attention-only block (no MLP)
    'T' : attention + SwiGLU MLP block
    'm' : Mamba2-only block (no MLP)        -- requires `mamba_ssm`
    'M' : Mamba2 + SwiGLU MLP block         -- requires `mamba_ssm`

Optional kernels (auto-detected, fallback if absent):
    flash_attn:  packed-mode attention uses `flash_attn_varlen_func`,
                 else SDPA + block-diagonal mask.
    mamba_ssm:   'm'/'M' arch blocks (Mamba2 mixer). Without mamba_ssm,
                 only 't'/'T' arch is supported.

Tensor shape conventions (match upstream):
    Padded mode: (B, L, D) with mask: (B, L).
    Packed mode: (T_total, D) with cu_seqlens: (B+1,) + max_seqlen: int.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet_nets.config import HNetConfig, get_stage_cfg

# Optional flash-attn varlen kernel.
try:
    from flash_attn import flash_attn_varlen_func  # type: ignore

    _HAS_FLASH_ATTN = True
except Exception:
    flash_attn_varlen_func = None  # type: ignore
    _HAS_FLASH_ATTN = False

# Optional Mamba2 SSM block.
try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2  # type: ignore

    _HAS_MAMBA = True
except Exception:
    _Mamba2 = None  # type: ignore
    _HAS_MAMBA = False


def has_flash_attn() -> bool:
    return _HAS_FLASH_ATTN


def has_mamba() -> bool:
    return _HAS_MAMBA


@dataclass
class KVCache:
    """Per-attention-layer K/V cache with per-batch-element fill counters."""

    k: torch.Tensor  # (B, num_heads, max_seqlen, head_dim)
    v: torch.Tensor  # (B, num_heads, max_seqlen, head_dim)
    offsets: torch.Tensor  # (B,) long

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.offsets.zero_()


@dataclass
class MambaCache:
    """Per-SSM-layer state for Mamba2 step inference."""

    conv_state: torch.Tensor  # (B, conv_dim, d_conv - 1)
    ssm_state: torch.Tensor  # (B, nheads, headdim, d_state)

    def reset(self):
        self.conv_state.zero_()
        self.ssm_state.zero_()


LayerCache = Union[KVCache, MambaCache]


@dataclass
class IsotropicInferenceParams:
    """One per-layer cache (KV or Mamba) per layer index."""

    layer_caches: Dict[int, Any] = field(default_factory=dict)
    max_seqlen: int = 0
    batch_size: int = 0


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype)
        return x_normed * self.weight


class AdaLNModulation(nn.Module):
    """Maps a conditioning vector to per-block (scale, shift) for AdaLN."""

    def __init__(self, d_cond, d_model):
        super().__init__()
        self.proj = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.weight._no_reinit = True

    def forward(self, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return scale, shift


def _adaln(x, scale, shift):
    """Apply AdaLN modulation with shape-flexible broadcasting.

    Padded mode: ``x`` is (B, L, D), ``scale/shift`` are (B, D)  → unsqueeze dim 1.
    Packed mode: ``x`` is (T_total, D), ``scale/shift`` are (T_total, D) → direct.
    """
    if scale.dim() < x.dim():
        scale = scale.unsqueeze(-2)
        shift = shift.unsqueeze(-2)
    return x * (1 + scale) + shift


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the leading ``rotary_dim`` channels of ``x``.

    Mirrors ``flash_attn.layers.rotary.apply_rotary_emb_torch`` (non-interleaved,
    i.e. GPT-NeoX style — rotate first/second halves). Works on any leading
    shape; the last dim of ``x`` is ``head_dim``, and ``cos/sin`` are
    broadcastable along all dims except the rotary-channel axis.

    x:        (..., head_dim)
    cos, sin: (..., rotary_dim / 2)  with positions aligned to x's positional
              axis (caller is responsible for shaping so they broadcast).
    """
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # Stack cos/sin into the rotary-channel layout: (..., rotary_dim).
    cos_full = torch.cat([cos, cos], dim=-1).to(x.dtype)
    sin_full = torch.cat([sin, sin], dim=-1).to(x.dtype)
    out_rot = x_rot * cos_full + _rotate_half(x_rot) * sin_full
    return torch.cat([out_rot, x_pass], dim=-1) if x_pass.numel() > 0 else out_rot


class _RotaryCache(nn.Module):
    """Cached cos/sin tables for RoPE.

    Holds a cache sized to the largest seqlen seen so far. ``get(positions)``
    returns ``(cos, sin)`` indexed by an arbitrary 1D position tensor —
    crucial for packed mode where positions are per-subseq, not contiguous.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "rotary_emb_dim must be even"
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        # Buffer so it follows .to(device); not persistent (no state_dict bloat).
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
        # ``_no_reinit`` keeps the residual-stream init from touching the
        # frozen freq table even though it isn't a Linear weight.
        self.inv_freq._no_reinit = True

    def _maybe_resize(self, seqlen: int, device, dtype):
        if (
            self._cos_cached is None
            or seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            inv = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv)  # (seqlen, dim/2)
            self._cos_cached = freqs.cos().to(dtype)
            self._sin_cached = freqs.sin().to(dtype)
            self._seq_len_cached = seqlen

    def get(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions: (N,) long. Returns (cos, sin) each (N, dim/2)."""
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        self._maybe_resize(max_pos, positions.device, dtype)
        return self._cos_cached[positions], self._sin_cached[positions]


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        causal=False,
        dropout=0.0,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout
        # RoPE (F6): when > 0, rotate the leading ``rotary_emb_dim`` channels
        # of q/k before attention. ``rotary_emb_dim <= head_dim``. Mirrors
        # upstream ``goombalab/hnet/hnet/modules/mha.py:CausalMHA``.
        self.rotary_emb_dim = int(rotary_emb_dim)
        assert self.rotary_emb_dim <= self.head_dim, (
            f"rotary_emb_dim ({self.rotary_emb_dim}) must be <= head_dim "
            f"({self.head_dim})"
        )
        if self.rotary_emb_dim > 0:
            self.rotary_emb = _RotaryCache(self.rotary_emb_dim, base=rotary_emb_base)
        else:
            self.rotary_emb = None

    def forward(
        self,
        x,
        mask=None,
        cu_seqlens=None,
        max_seqlen: Optional[int] = None,
    ):
        """
        Padded mode: x (B, L, D), mask (B, L). Uses SDPA.
        Packed mode: x (T_total, D), cu_seqlens (B+1,), max_seqlen int.
            Uses flash_attn_varlen_func when available; otherwise SDPA with a
            block-diagonal causal mask.
        """
        if cu_seqlens is not None:
            return self._forward_packed(x, cu_seqlens, max_seqlen)
        return self._forward_padded(x, mask)

    def _forward_padded(self, x, mask):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        # RoPE on q/k (F6). Positions are 0..L-1, broadcast over batch + heads.
        if self.rotary_emb is not None:
            positions = torch.arange(L, device=x.device)
            cos, sin = self.rotary_emb.get(positions, q.dtype)  # (L, dim/2)
            # Broadcast shape: (1, L, 1, dim/2) to match (B, L, H, head_dim).
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(dtype=torch.bool)
            attn_mask = attn_mask & attn_mask.transpose(-1, -2)

        _W = getattr(self, "window", 0)
        if _W > 0 and self.causal:
            _p = torch.arange(q.shape[2], device=q.device)
            _wm = ((_p[:, None] >= _p[None, :]) & (_p[:, None] - _p[None, :] < _W))[None, None]
            attn_mask = _wm if attn_mask is None else (attn_mask & _wm)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal,
            )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

    def _forward_packed(self, x, cu_seqlens, max_seqlen):
        # x: (T_total, D)
        T_total, D = x.shape
        qkv = self.qkv(x).reshape(T_total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each (T_total, H, Dh)

        # RoPE on q/k in packed mode (F6). Positions are PER-SUBSEQ: token t
        # at global index i belongs to subseq s with start cu_seqlens[s], so
        # its rotary position is ``i - cu_seqlens[s]``. This avoids the
        # cross-subseq position leakage that would happen with a single
        # global arange.
        if self.rotary_emb is not None:
            pos_global = torch.arange(T_total, device=x.device)
            seq_idx = (pos_global[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
            local_pos = pos_global - cu_seqlens[seq_idx]  # (T_total,)
            cos, sin = self.rotary_emb.get(local_pos, q.dtype)  # (T_total, dim/2)
            # Broadcast over heads: (T_total, 1, dim/2).
            cos = cos[:, None, :]
            sin = sin[:, None, :]
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        # flash_attn_varlen_func ONLY supports fp16 / bf16. Under bf16
        # autocast (Lightning ``trainer.precision=bf16-mixed``) q/k/v
        # are already bf16 so this is the common path; standalone fp32
        # forwards (smokes, no-autocast inference) drop to SDPA.
        if _HAS_FLASH_ATTN and x.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            cu_q = cu_seqlens.to(torch.int32)
            ms = (
                int(max_seqlen)
                if max_seqlen is not None
                else int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            )
            _W = getattr(self, "window", 0)
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_q,
                max_seqlen_q=ms,
                max_seqlen_k=ms,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=((_W - 1, 0) if (_W > 0 and self.causal) else (-1, -1)),
            )
            # flash_attn returns (T_total, H, Dh)
            out = out.reshape(T_total, D)
            return self.out_proj(out)

        # Fallback: SDPA with block-diagonal (+ optional causal) mask.
        # (T_total, H, Dh) -> (1, H, T_total, Dh)
        q_ = q.transpose(0, 1).unsqueeze(0)
        k_ = k.transpose(0, 1).unsqueeze(0)
        v_ = v.transpose(0, 1).unsqueeze(0)

        pos = torch.arange(T_total, device=x.device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        same_seq = seq_idx[:, None] == seq_idx[None, :]
        if self.causal:
            causal_mask = pos[:, None] >= pos[None, :]
            _W = getattr(self, "window", 0)
            if _W > 0:
                causal_mask = causal_mask & (pos[:, None] - pos[None, :] < _W)
            attn_mask = (same_seq & causal_mask)[None, None]
        else:
            attn_mask = same_seq[None, None]

        out = F.scaled_dot_product_attention(
            q_,
            k_,
            v_,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        # (1, H, T_total, Dh) -> (T_total, D)
        out = out.squeeze(0).transpose(0, 1).reshape(T_total, D)
        return self.out_proj(out)

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype):
        return KVCache(
            k=torch.zeros(
                batch_size,
                self.num_heads,
                max_seqlen,
                self.head_dim,
                device=device,
                dtype=dtype,
            ),
            v=torch.zeros(
                batch_size,
                self.num_heads,
                max_seqlen,
                self.head_dim,
                device=device,
                dtype=dtype,
            ),
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def step(self, x, cache: KVCache):
        # x: (B, 1, D). Per-batch K/V scatter + attn against valid prior slots.
        B, _, D = x.shape
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, 1, H, head_dim)

        # RoPE on q/k at the current AR position (F6). Each batch element may
        # be at a different ``cache.offsets`` (variable-length rollouts), so
        # we look up cos/sin per-row.
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb.get(cache.offsets, q.dtype)  # (B, dim/2)
            cos = cos[:, None, None, :]  # (B, 1, 1, dim/2) -> broadcast over (1, H)
            sin = sin[:, None, None, :]
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        batch_idx = torch.arange(B, device=x.device)
        cache.k[batch_idx, :, cache.offsets] = k.squeeze(2).to(cache.k.dtype)
        cache.v[batch_idx, :, cache.offsets] = v.squeeze(2).to(cache.v.dtype)
        cache.offsets = cache.offsets + 1

        max_T = cache.k.shape[2]
        pos = torch.arange(max_T, device=x.device)
        attn_mask = pos[None, :] < cache.offsets[:, None]
        _W = getattr(self, "window", 0)
        if _W > 0:
            attn_mask = attn_mask & (pos[None, :] >= cache.offsets[:, None] - _W)
        attn_mask = attn_mask[:, None, None, :]

        # AR inference: never apply dropout regardless of training flag.
        out = F.scaled_dot_product_attention(
            q, cache.k, cache.v, attn_mask=attn_mask, dropout_p=0.0
        )
        out = out.transpose(1, 2).reshape(B, 1, D)
        return self.out_proj(out)


class CrossMultiHeadAttention(nn.Module):
    """Cross-attention: queries from x, keys/values from cond_tokens.

    For our use case, x is the action token sequence (B, T, d_model) and
    cond_tokens is the per-frame conditioning sequence (B, T, d_cond) — same
    T, with `cond_t` derived from the obs at step t. ``causal=True`` lets the
    action token at position t attend only to cond_<=t (matching teacher-
    forcing semantics in training).
    """

    def __init__(self, d_model, d_cond, num_heads, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_cond = d_cond
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_cond, 2 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, cond_tokens, cu_seqlens=None, max_seqlen=None):
        # PER-FRAME (run E): a 4D cond ``(B/T_total, L, M, d_cond)`` means each
        # query token has its OWN private set of M spatial tokens (the spatial
        # tokens of ITS frame). This is gated on rank=4 so the existing 3D
        # chunk-path (a single shared (B, M, d_cond) set) stays byte-identical.
        if cond_tokens.dim() == (4 if cu_seqlens is None else 3):
            return self._forward_per_frame(x, cond_tokens, cu_seqlens)
        if cu_seqlens is not None:
            return self._forward_packed(x, cond_tokens, cu_seqlens, max_seqlen)
        return self._forward_padded(x, cond_tokens)

    def _forward_per_frame(self, x, cond_tokens, cu_seqlens=None):
        """Per-query-token-private cross-attention.

        Padded:  x (B, L, d_model),       cond_tokens (B, L, M, d_cond).
        Packed:  x (T_total, d_model),     cond_tokens (T_total, M, d_cond).

        Query token i attends ONLY over its own frame's M spatial tokens — no
        causal masking across frames (each token sees just its own frame, so
        train (full seq) and AR-step (one token) are mathematically identical:
        token i's output depends solely on x[i] and cond_tokens[i]).
        """
        if cu_seqlens is None:
            B, L, _ = x.shape
            M = cond_tokens.shape[2]
            xf = x.reshape(B * L, 1, self.d_model)
            cf = cond_tokens.reshape(B * L, M, self.d_cond)
        else:
            T_total = x.shape[0]
            L = 1
            M = cond_tokens.shape[1]
            xf = x.reshape(T_total, 1, self.d_model)
            cf = cond_tokens  # (T_total, M, d_cond)
            B = T_total
        N = xf.shape[0]
        q = self.q_proj(xf).reshape(N, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(cf).reshape(N, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # (N, H, 1, Dh)
        out = out.transpose(1, 2).reshape(N, self.d_model)
        out = self.out_proj(out)
        if cu_seqlens is None:
            return out.reshape(B, L, self.d_model)
        return out  # (T_total, d_model)

    def step_per_frame(self, x, cond_tokens):
        """AR-step per-frame cross-attention (NO cache).

        ``x``: (B, 1, d_model) current step's query token.
        ``cond_tokens``: (B, M, d_cond) the current frame's M spatial tokens.

        Identical math to ``_forward_per_frame`` for one query token, so the
        AR rollout matches teacher-forced training exactly.
        """
        B = x.shape[0]
        M = cond_tokens.shape[1]
        q = self.q_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(cond_tokens).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, 1, self.d_model)
        return self.out_proj(out)

    def _forward_padded(self, x, cond_tokens):
        B, T_q, _ = x.shape
        T_k = cond_tokens.shape[1]
        q = (
            self.q_proj(x)
            .reshape(B, T_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_proj(cond_tokens).reshape(B, T_k, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = self.causal and (T_q == T_k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.out_proj(out)

    def _forward_packed(self, x, cond_tokens, cu_seqlens, max_seqlen):
        T_total = x.shape[0]
        q = self.q_proj(x).reshape(T_total, self.num_heads, self.head_dim)
        kv = self.kv_proj(cond_tokens).reshape(
            T_total, 2, self.num_heads, self.head_dim
        )
        k, v = kv.unbind(dim=1)

        q_ = q.transpose(0, 1).unsqueeze(0)
        k_ = k.transpose(0, 1).unsqueeze(0)
        v_ = v.transpose(0, 1).unsqueeze(0)

        pos = torch.arange(T_total, device=x.device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        same_seq = seq_idx[:, None] == seq_idx[None, :]
        if self.causal:
            causal_mask = pos[:, None] >= pos[None, :]
            _W = getattr(self, "window", 0)
            if _W > 0:
                causal_mask = causal_mask & (pos[:, None] - pos[None, :] < _W)
            attn_mask = (same_seq & causal_mask)[None, None]
        else:
            attn_mask = same_seq[None, None]

        out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask)
        out = out.squeeze(0).transpose(0, 1).reshape(T_total, self.d_model)
        return self.out_proj(out)

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype):
        """Allocate a KV cache for cond-token attention during AR rollout.

        Mirrors ``MultiHeadAttention.allocate_inference_cache`` but the cache
        accumulates one new K/V pair per AR step (the current step's cond).
        """
        return KVCache(
            k=torch.zeros(
                batch_size,
                self.num_heads,
                max_seqlen,
                self.head_dim,
                device=device,
                dtype=dtype,
            ),
            v=torch.zeros(
                batch_size,
                self.num_heads,
                max_seqlen,
                self.head_dim,
                device=device,
                dtype=dtype,
            ),
            offsets=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def step(self, x, cond_curr, cache: "KVCache"):
        """Cross-attn at AR step time, with KV cache.

        ``x``: (B, 1, d_model) — current step's action token (Q source).
        ``cond_curr``: (B, 1, d_cond) — current step's cond (K/V source).
        ``cache``: KV cache accumulating one cond token per AR step. The
        cache is updated in-place; the Q from ``x`` attends against all
        cached K/V (positions 0..t inclusive).
        """
        B = x.shape[0]
        # Compute K/V from this step's cond and scatter into the cache.
        kv = self.kv_proj(cond_curr).reshape(B, 1, 2, self.num_heads, self.head_dim)
        k_new, v_new = kv.unbind(dim=2)  # each (B, 1, H, Dh)
        batch_idx = torch.arange(B, device=x.device)
        cache.k[batch_idx, :, cache.offsets] = k_new.squeeze(1).to(cache.k.dtype)
        cache.v[batch_idx, :, cache.offsets] = v_new.squeeze(1).to(cache.v.dtype)
        cache.offsets = cache.offsets + 1

        # Q from x, attend against valid cached K/V positions.
        q = self.q_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        max_T = cache.k.shape[2]
        pos = torch.arange(max_T, device=x.device)
        attn_mask = (pos[None, :] < cache.offsets[:, None])[:, None, None, :]
        out = F.scaled_dot_product_attention(q, cache.k, cache.v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, 1, self.d_model)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_intermediate):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    Conditioning mode (``cond_mode``):
      - ``"adaln"`` (default): cond is consumed as per-token modulation
        (scale/shift on each pre-norm output). The block expects ``cond`` to
        be per-token: (B, T, d_cond) padded or (T_total, d_cond) packed.
      - ``"cross_attn"``: an extra cross-attention layer between self-attn
        and the MLP. Queries from x, keys/values from cond tokens. Same
        per-token shape contract as adaln. ``causal=True`` constrains the
        action token at position t to attend only to cond_<=t (matching
        teacher-forcing semantics).
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_intermediate=0,
        causal=False,
        d_cond=0,
        cond_mode: str = "adaln",
        rotary_emb_dim: int = 0,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.has_mlp = d_intermediate > 0
        self.has_cond = d_cond > 0
        self.causal = causal
        self.cond_mode = cond_mode if self.has_cond else "none"
        if self.cond_mode not in ("adaln", "cross_attn", "none"):
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")
        self.norm1 = RMSNorm(d_model)
        # F6: plumb rotary_emb_dim into the self-attn mixer. Cross-attn (cond
        # tokens) intentionally does NOT use RoPE — cond positions are per-
        # frame conditioning, not autoregressive token positions.
        # ``dropout`` is the attention-softmax dropout (forwarded into MHA).
        self.mixer = MultiHeadAttention(
            d_model,
            num_heads,
            causal=causal,
            dropout=dropout,
            rotary_emb_dim=rotary_emb_dim,
        )
        # Residual-branch dropout (applied to attn/ffn outputs before the
        # residual add). Default 0.0 — keeps existing call sites unaffected.
        self.resid_dropout = float(resid_dropout)
        self.attn_resid_drop = nn.Dropout(self.resid_dropout)
        if self.has_mlp:
            self.norm2 = RMSNorm(d_model)
            self.mlp = SwiGLU(d_model, d_intermediate)
            self.ffn_resid_drop = nn.Dropout(self.resid_dropout)
        if self.cond_mode == "adaln":
            self.adaln1 = AdaLNModulation(d_cond, d_model)
            if self.has_mlp:
                self.adaln2 = AdaLNModulation(d_cond, d_model)
        elif self.cond_mode == "cross_attn":
            self.cross_norm = RMSNorm(d_model)
            self.cross_attn = CrossMultiHeadAttention(
                d_model=d_model, d_cond=d_cond, num_heads=num_heads, causal=causal
            )
            self.cross_resid_drop = nn.Dropout(self.resid_dropout)

    def forward(self, x, mask=None, cond=None, cu_seqlens=None, max_seqlen=None):
        # Self-attention.
        h = self.norm1(x)
        if self.cond_mode == "adaln" and cond is not None:
            s, b = self.adaln1(cond)
            h = _adaln(h, s, b)
        x = x + self.attn_resid_drop(
            self.mixer(h, mask=mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        )

        # Cross-attention (if enabled).
        if self.cond_mode == "cross_attn" and cond is not None:
            h = self.cross_norm(x)
            x = x + self.cross_resid_drop(
                self.cross_attn(
                    h, cond, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                )
            )

        # MLP.
        if self.has_mlp:
            h = self.norm2(x)
            if self.cond_mode == "adaln" and cond is not None:
                s, b = self.adaln2(cond)
                h = _adaln(h, s, b)
            x = x + self.ffn_resid_drop(self.mlp(h))
        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype):
        """Allocate self-attn KVCache + (optional) cross-attn KVCache.

        Returns a single cache for ``cond_mode!="cross_attn"`` (matches the
        upstream single-cache convention); a tuple ``(self_cache,
        cross_cache)`` when cross-attn is enabled.
        """
        self_cache = self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, device, dtype
        )
        if self.cond_mode == "cross_attn":
            cross_cache = self.cross_attn.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            )
            return (self_cache, cross_cache)
        return self_cache

    def step(self, x, cache, cond: Optional[torch.Tensor] = None):
        """AR step.

        Cond contract by ``cond_mode``:
          - ``"adaln"``: cond is (B, d_cond) — current step's cond.
          - ``"cross_attn"``: cond is (B, 1, d_cond) — current step's cond
            (the cross_attn block's KV cache accumulates the history).

        Cache contract:
          - ``"adaln"`` / ``"none"``: cache is the self-attn ``KVCache``.
          - ``"cross_attn"``: cache is ``(self_cache, cross_cache)``.
        """
        if isinstance(cache, tuple):
            self_cache, cross_cache = cache
        else:
            self_cache, cross_cache = cache, None

        # Self-attention. Residual dropout modules are still applied — in
        # eval/AR mode they are identity (self.training=False).
        h = self.norm1(x)
        if self.cond_mode == "adaln" and cond is not None:
            s, b = self.adaln1(cond)
            h = _adaln(h, s, b)
        x = x + self.attn_resid_drop(self.mixer.step(h, self_cache))

        # Cross-attention.
        if self.cond_mode == "cross_attn" and cond is not None:
            h = self.cross_norm(x)
            if cond.dim() == 3:
                # PER-FRAME (run E): cond is (B, M, d_cond) — the current
                # frame's M spatial tokens. No cache: each step attends over
                # JUST its own frame's tokens (matches _forward_per_frame), so
                # AR == teacher-forced exactly.
                x = x + self.cross_resid_drop(
                    self.cross_attn.step_per_frame(h, cond)
                )
            else:
                assert (
                    cross_cache is not None
                ), "cond_mode=cross_attn requires a cross-attn cache"
                # Accept both (B, d_cond) and (B, 1, d_cond) — the per-step cond
                # slice from the policy comes through as 2D; cross_attn.step wants
                # an explicit time dim.
                cond_curr = cond.unsqueeze(1) if cond.dim() == 2 else cond
                x = x + self.cross_resid_drop(
                    self.cross_attn.step(h, cond_curr, cross_cache)
                )

        # MLP.
        if self.has_mlp:
            h = self.norm2(x)
            if self.cond_mode == "adaln" and cond is not None:
                s, b = self.adaln2(cond)
                h = _adaln(h, s, b)
            x = x + self.ffn_resid_drop(self.mlp(h))
        return x


class Mamba2Mixer(nn.Module):
    """Adapter around mamba_ssm's Mamba2 with our (forward, step, allocate) interface.

    Mamba2 is inherently causal; the `causal` flag on TransformerBlock has no
    equivalent here.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        ssm_cfg: Optional[dict] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError(
                "mamba_ssm is not installed but an 'm'/'M' arch block was requested. "
                "Install mamba_ssm or change the arch_layout to use only 't'/'T'."
            )
        ssm_cfg = dict(ssm_cfg or {})
        self.mamba = _Mamba2(
            d_model=d_model, layer_idx=layer_idx, device=device, dtype=dtype, **ssm_cfg
        )
        self.layer_idx = layer_idx
        self.d_model = d_model

    def forward(self, x, mask=None, cu_seqlens=None, max_seqlen=None):
        """
        Padded mode: x (B, L, D). Mamba2 processes the full sequence (padding
            tokens at the end don't pollute earlier outputs since it's causal).
        Packed mode: x (T_total, D). We unsqueeze to (1, T_total, D) and pass
            seq_idx so the inner scan resets at sub-sequence boundaries.
        """
        if cu_seqlens is not None:
            x3 = x.unsqueeze(0)
            pos = torch.arange(x3.shape[1], device=x.device)
            seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1).to(torch.int32)
            seq_idx = seq_idx.unsqueeze(0)  # (1, T_total)
            out = self.mamba(x3, seq_idx=seq_idx)
            return out.squeeze(0)
        return self.mamba(x)

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype):
        conv_state, ssm_state = self.mamba.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype
        )
        return MambaCache(conv_state=conv_state, ssm_state=ssm_state)

    def step(self, x, cache: MambaCache):
        # x: (B, 1, D). Mamba2.step returns (out, conv_state, ssm_state).
        out, conv_state, ssm_state = self.mamba.step(
            x, cache.conv_state, cache.ssm_state
        )
        cache.conv_state.copy_(conv_state)
        cache.ssm_state.copy_(ssm_state)
        return out


class MambaBlock(nn.Module):
    """Pre-norm block with Mamba2 mixer + optional MLP + optional AdaLN cond.

    Mirrors TransformerBlock; same forward/step signature so Isotropic can
    treat them interchangeably.
    """

    def __init__(
        self,
        d_model,
        layer_idx: int,
        d_intermediate: int = 0,
        d_cond: int = 0,
        ssm_cfg: Optional[dict] = None,
        device=None,
        dtype=None,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.has_mlp = d_intermediate > 0
        self.has_cond = d_cond > 0
        self.norm1 = RMSNorm(d_model)
        self.mixer = Mamba2Mixer(
            d_model, layer_idx=layer_idx, ssm_cfg=ssm_cfg, device=device, dtype=dtype
        )
        # Residual-branch dropout (analogous to TransformerBlock).
        self.resid_dropout = float(resid_dropout)
        self.mixer_resid_drop = nn.Dropout(self.resid_dropout)
        if self.has_mlp:
            self.norm2 = RMSNorm(d_model)
            self.mlp = SwiGLU(d_model, d_intermediate)
            self.ffn_resid_drop = nn.Dropout(self.resid_dropout)
        if self.has_cond:
            self.adaln1 = AdaLNModulation(d_cond, d_model)
            if self.has_mlp:
                self.adaln2 = AdaLNModulation(d_cond, d_model)

    def forward(self, x, mask=None, cond=None, cu_seqlens=None, max_seqlen=None):
        h = self.norm1(x)
        if self.has_cond and cond is not None:
            s, b = self.adaln1(cond)
            h = _adaln(h, s, b)
        x = x + self.mixer_resid_drop(
            self.mixer(h, mask=mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        )

        if self.has_mlp:
            h = self.norm2(x)
            if self.has_cond and cond is not None:
                s, b = self.adaln2(cond)
                h = _adaln(h, s, b)
            x = x + self.ffn_resid_drop(self.mlp(h))
        return x

    def step(self, x, cache: MambaCache, cond: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        if self.has_cond and cond is not None:
            s, b = self.adaln1(cond)
            h = _adaln(h, s, b)
        x = x + self.mixer_resid_drop(self.mixer.step(h, cache))

        if self.has_mlp:
            h = self.norm2(x)
            if self.has_cond and cond is not None:
                s, b = self.adaln2(cond)
                h = _adaln(h, s, b)
            x = x + self.ffn_resid_drop(self.mlp(h))
        return x


class Isotropic(nn.Module):
    """A stack of transformer and/or Mamba2 blocks for one stage / position."""

    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        d_cond: int = 0,
        cond_here: bool = False,
        causal: bool = False,
        cond_mode: str = "adaln",
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        self.cond_mode = cond_mode
        attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)
        num_heads = attn_cfg.get("num_heads", 8)
        # F6: per-stage rotary_emb_dim. ``AttnConfig.rotary_emb_dim`` is a
        # per-stage list (parallel to ``num_heads``); 0 disables RoPE in this
        # stage.
        rotary_emb_dim = int(attn_cfg.get("rotary_emb_dim", 0) or 0)
        # Per-stage dropout knobs (default 0.0 — existing call sites that
        # don't set them in their AttnConfig stay unaffected).
        attn_dropout = float(attn_cfg.get("dropout", 0.0) or 0.0)
        resid_dropout = float(attn_cfg.get("resid_dropout", 0.0) or 0.0)
        ssm_cfg = (
            get_stage_cfg(config.ssm_cfg, stage_idx)
            if hasattr(config, "ssm_cfg")
            else {}
        )

        layout = config.arch_layout
        for _ in range(stage_idx):
            layout = layout[1]
        layout = layout[pos_idx]
        layout_parse = re.findall(r"([mMtT])(\d+)", layout)
        if not layout_parse:
            raise ValueError(
                f"Empty / unsupported arch_layout entry '{layout}'. Use t/T/m/M tokens."
            )

        blocks = []
        self.arch_full = []
        self.height = 0
        layer_idx = 0
        for arch, n_layer_str in layout_parse:
            n = int(n_layer_str)
            d_int = config.d_intermediate[stage_idx] if arch.isupper() else 0
            for _ in range(n):
                if arch in ("t", "T"):
                    blk = TransformerBlock(
                        d_model=self.d_model,
                        num_heads=num_heads,
                        d_intermediate=d_int,
                        causal=causal,
                        d_cond=d_cond if cond_here else 0,
                        cond_mode=cond_mode,
                        rotary_emb_dim=rotary_emb_dim,
                        dropout=attn_dropout,
                        resid_dropout=resid_dropout,
                    )
                else:  # 'm' or 'M'
                    blk = MambaBlock(
                        d_model=self.d_model,
                        layer_idx=layer_idx,
                        d_intermediate=d_int,
                        d_cond=d_cond if cond_here else 0,
                        ssm_cfg=ssm_cfg or None,
                        resid_dropout=resid_dropout,
                    )
                blocks.append(blk)
                self.arch_full.append(arch)
                layer_idx += 1
            self.height += (2 if arch.isupper() else 1) * n

        self.layers = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(self.d_model)

    def forward(
        self,
        x,
        mask=None,
        cond: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        for blk in self.layers:
            x = blk(
                x, mask=mask, cond=cond, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        return self.final_norm(x)

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype):
        params = IsotropicInferenceParams(
            layer_caches={},
            max_seqlen=max_seqlen,
            batch_size=batch_size,
        )
        for i, blk in enumerate(self.layers):
            # Prefer block-level allocate (handles both self-attn and
            # cross-attn caches in one call); fall back to mixer.allocate
            # for blocks that don't define their own.
            if hasattr(blk, "allocate_inference_cache"):
                params.layer_caches[i] = blk.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype
                )
            else:
                params.layer_caches[i] = blk.mixer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype
                )
        return params

    def step(
        self, x, params: IsotropicInferenceParams, cond: Optional[torch.Tensor] = None
    ):
        for i, blk in enumerate(self.layers):
            x = blk.step(x, params.layer_caches[i], cond=cond)
        return self.final_norm(x)
