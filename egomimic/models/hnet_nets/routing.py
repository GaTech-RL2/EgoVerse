"""
RoutingModule / ChunkLayer / DeChunkLayer with both padded (mask-based) and
packed (cu_seqlens-based) training paths, plus single-token `step` inference.

Padded mode:
    hidden_states: (B, L, D)
    mask:          (B, L) bool — True = valid token
    cu_seqlens:    None

Packed mode (matches upstream's 2D convention so flash_attn varlen plugs in):
    hidden_states: (T_total, D)
    cu_seqlens:    (num_subseqs + 1,) long — cumulative subseq lengths
    max_seqlen:    int — longest subseq length (passed to attention modules)
    mask:          None
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional Mamba2 selective-scan kernel for the DeChunkLayer EMA.
try:
    from einops import rearrange, repeat  # type: ignore
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,  # type: ignore
    )

    _HAS_MAMBA_SCAN = True
except Exception:
    mamba_chunk_scan_combined = None  # type: ignore
    rearrange = repeat = None  # type: ignore
    _HAS_MAMBA_SCAN = False


def has_mamba_scan() -> bool:
    return _HAS_MAMBA_SCAN


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor  # padded: (B, L, 2); packed: (T_total, 2); step: (B, 2)
    boundary_mask: torch.Tensor  # padded: (B, L);    packed: (T_total,);   step: (B,)
    selected_probs: (
        torch.Tensor
    )  # padded: (B, L, 1); packed: (T_total, 1); step: (B, 1)


@dataclass
class RoutingModuleState:
    has_seen_tokens: torch.Tensor  # (B,) bool
    last_hidden_state: torch.Tensor  # (B, D)


@dataclass
class DeChunkState:
    last_value: torch.Tensor  # (B, D)


def get_seq_idx(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """(B+1,) cu_seqlens -> (T_total,) seq_idx mapping each packed position to its subseq id."""
    T_total = int(cu_seqlens[-1].item())
    pos = torch.arange(T_total, device=cu_seqlens.device)
    return (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)


class RoutingModule(nn.Module):
    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        # F9 (upstream parity): identity init for q/k is intentional in the
        # H-Net paper / reference implementation
        # (``goombalab/hnet/hnet/modules/dc.py:RoutingModule.__init__``).
        # At init cos_sim(q(h_t), k(h_{t+1})) reduces to cosine similarity
        # of consecutive hidden states themselves; near-identical consecutive
        # embeddings give cos_sim ~ 1 → boundary_prob ~ 0, so the chunker
        # initially fires ONLY at the forced subseq-start positions
        # (``boundary_prob[cu_seqlens[:-1]] = 1.0`` below). The network is
        # expected to break this degeneracy during training. The
        # ``_no_reinit`` flags keep residual-stream-aware init from
        # overwriting these back to a normal-init. Do NOT change to a random
        # init: upstream relies on this exact behaviour.
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        # max_seqlen kept for upstream signature parity (unused here).
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[RoutingModuleState] = None,
    ) -> RoutingModuleOutput:
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask (padded) or cu_seqlens (packed) must be provided"
        if cu_seqlens is not None:
            assert (
                inference_params is None
            ), "Packed mode does not support inference_params"
            return self._forward_packed(hidden_states, cu_seqlens)
        return self._forward_padded(hidden_states, mask, inference_params)

    def _forward_padded(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        inference_params: Optional[RoutingModuleState],
    ) -> RoutingModuleOutput:
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", 1.0)  # PAD Prob 1.0
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = (
            selected_idx == 1
        ) & mask  # selected boundary not in valid token range is not chosen
        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # equivalent confidence mentioned in the paper

        # Prefill update: stash last valid token per batch so subsequent step()s
        # continue from this prefix.
        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_idx = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            B = hidden_states.shape[0]
            last_h = hidden_states[
                torch.arange(B, device=hidden_states.device), last_idx
            ]
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask.unsqueeze(-1),
                    last_h.to(inference_params.last_hidden_state.dtype),
                    inference_params.last_hidden_state,
                )
            )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def _forward_packed(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> RoutingModuleOutput:
        # hidden_states: (T_total, D)
        # Pairwise cos sim across consecutive positions, then re-pad to length T_total
        cos_sim = (
            F.normalize(self.q_proj_layer(hidden_states[:-1]), dim=-1)
            * F.normalize(self.k_proj_layer(hidden_states[1:]), dim=-1)
        ).sum(dim=-1)
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", 1.0)
        # Force boundary at the first token of every subseq
        boundary_prob = boundary_prob.clone()
        boundary_prob[cu_seqlens[:-1]] = 1.0  # PAD Prob 1.0 at subseq starts

        boundary_prob = torch.stack(
            ((1 - boundary_prob), boundary_prob), dim=-1
        )  # (T_total, 2)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1  # (T_total,)
        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # equivalent to confidence mentioned in the paper, (T_total, 1)
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(
        self, hidden_states, inference_params: RoutingModuleState
    ) -> RoutingModuleOutput:
        hidden_states = hidden_states.squeeze(1)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(
            hidden_states.to(inference_params.last_hidden_state.dtype)
        )
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        inference_params.has_seen_tokens.fill_(True)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class ChunkLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns: (next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask)
            Padded mode: next_cu_seqlens / next_max_seqlen are None.
            Packed mode: next_mask is None.
        """
        if cu_seqlens is not None:
            return self._forward_packed(hidden_states, boundary_mask, cu_seqlens)
        return self._forward_padded(hidden_states, boundary_mask)

    def _forward_padded(self, hidden_states, boundary_mask):
        _, L, D = hidden_states.shape
        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)
        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(-1, -1, D),
        )
        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
        )
        # Upstream discards next_max_seqlen at the end of the padded path — the
        # downstream Isotropic uses next_mask instead.
        next_max_seqlen = None
        return next_hidden_states, None, next_max_seqlen, next_mask

    def _forward_packed(self, hidden_states, boundary_mask, cu_seqlens):
        # hidden_states: (T_total, D), boundary_mask: (T_total,)
        next_hidden_states = hidden_states[boundary_mask]  # (M_total, D)
        next_cu_seqlens = F.pad(
            boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
        )  # (B+1,)
        next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
        return next_hidden_states, next_cu_seqlens, next_max_seqlen, None

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    """
    Maps chunked representations back to the original sequence length via an
    EMA along the chunked axis followed by index-gather plug-back.

    EMA backend (auto-selected):
        - mamba_chunk_scan_combined when mamba_ssm is installed (parallel
          selective-scan kernel; same recurrence as the Python loop).
        - Pure-PyTorch sequential recurrence otherwise.

    `dtype`, `block_size`, `headdim` are kernel knobs (mirror upstream defaults).
    """

    def __init__(self, d_model, dtype=torch.bfloat16, block_size=256, headdim=32):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert (
            d_model % self.headdim == 0
        ), f"d_model ({d_model}) must be divisible by headdim ({self.headdim})."
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        # max_seqlen kept for upstream signature parity (unused here).
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        inference_params: Optional[DeChunkState] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Padded mode:
            hidden_states (B, M, D), boundary_mask (B, L), boundary_prob (B, L, 2).
        Packed mode:
            hidden_states (M_total, D), boundary_mask (T_total,), boundary_prob (T_total, 2),
            cu_seqlens is **chunked-space** cu_seqlens (matches upstream API).
        """
        if cu_seqlens is not None:
            assert (
                inference_params is None
            ), "Packed mode does not support inference_params"
            return self._forward_packed(
                hidden_states, boundary_mask, boundary_prob, cu_seqlens
            )
        return self._forward_padded(
            hidden_states, boundary_mask, boundary_prob, inference_params
        )

    # -------------- EMA backends --------------

    def _ema_kernel(
        self, x_3d: torch.Tensor, p_2d: torch.Tensor, seq_idx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Parallel EMA via mamba_chunk_scan_combined.

        x_3d: (B, M, D), p_2d: (B, M), seq_idx: (B, M) int32 or None.
        Returns (B, M, D), dtype = x_3d.dtype.
        """
        original_dtype = x_3d.dtype
        dt = torch.log(1 / (1 - p_2d)).to(self.dtype)
        x = (x_3d / dt[..., None]).to(self.dtype)
        A = -torch.ones((self.nheads,), device=x_3d.device, dtype=torch.float32)
        b = p_2d.to(self.dtype)
        c = torch.ones_like(b)
        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")
        return out.to(original_dtype)

    def _ema_loop(
        self,
        x_3d: torch.Tensor,
        p_2d: torch.Tensor,
        seq_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sequential EMA fallback.

        x_3d: (B, M, D), p_2d: (B, M). When seq_starts is provided (1D long
        tensor of chunked-space subseq starts), reset prev at those indices.
        """
        B, M, D = x_3d.shape
        if M == 0:
            return x_3d
        starts_set: set = set()
        if seq_starts is not None:
            starts_set = set(int(s) for s in seq_starts.tolist())
        out = []
        prev = torch.zeros(B, D, device=x_3d.device, dtype=x_3d.dtype)
        for t in range(M):
            if t in starts_set:
                prev = torch.zeros(B, D, device=x_3d.device, dtype=x_3d.dtype)
            pt = p_2d[:, t : t + 1]
            prev = pt * x_3d[:, t] + (1 - pt) * prev
            out.append(prev)
        return torch.stack(out, dim=1)

    # -------------- Mode-specific forwards --------------

    def _forward_padded(
        self, hidden_states, boundary_mask, boundary_prob, inference_params=None
    ):
        _, L = boundary_mask.shape
        D = hidden_states.shape[-1]
        device = hidden_states.device
        M = hidden_states.shape[1]

        # Gather boundary probs in chunked order (boundaries first within each row).
        p_full = torch.clamp(boundary_prob[..., -1], min=1e-4, max=1 - 1e-4)
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        sorted_idx = torch.argsort(token_idx, dim=1)[:, :M]
        p = torch.gather(p_full, dim=1, index=sorted_idx)  # (B, M)

        if _HAS_MAMBA_SCAN and hidden_states.is_cuda and M > 0:
            ema = self._ema_kernel(hidden_states, p, seq_idx=None)
        else:
            ema = self._ema_loop(hidden_states, p)

        plug_back_idx = torch.clamp(torch.cumsum(boundary_mask, dim=1) - 1, min=0)
        full = torch.gather(
            ema, dim=1, index=plug_back_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        if inference_params is not None:
            inference_params.last_value.copy_(
                full[:, -1].to(inference_params.last_value.dtype)
            )
        return full

    def _forward_packed(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens):
        """`cu_seqlens` is the **chunked-space** cu_seqlens (size B+1)."""
        _, D = hidden_states.shape
        device = hidden_states.device

        # Chunked-space boundary probs.
        p_packed = torch.clamp(
            boundary_prob[..., -1][boundary_mask], min=1e-4, max=1 - 1e-4
        )  # (M_total,)

        M_total = hidden_states.shape[0]
        if _HAS_MAMBA_SCAN and hidden_states.is_cuda and M_total > 0:
            # Build chunked-space seq_idx for the scan reset.
            pos = torch.arange(M_total, device=device)
            seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1).to(torch.int32)
            ema = self._ema_kernel(
                hidden_states.unsqueeze(0),
                p_packed.unsqueeze(0),
                seq_idx=seq_idx.unsqueeze(0),
            ).squeeze(0)
        else:
            # Fallback: per-subseq EMA with reset at each subseq start.
            ema = self._ema_loop(
                hidden_states.unsqueeze(0),
                p_packed.unsqueeze(0),
                seq_starts=cu_seqlens[1:-1] if cu_seqlens.numel() > 2 else None,
            ).squeeze(0)

        # Plug back: chunked stream is monotonic in subseq order, so the global
        # cumsum aligns indices correctly.
        plug_back_idx = torch.clamp(boundary_mask.cumsum(dim=0) - 1, min=0)
        full = torch.gather(ema, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, D))
        return full  # (T_total, D)

    def step(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        inference_params: DeChunkState,
    ):
        B = boundary_mask.shape[0]
        D = inference_params.last_value.shape[-1]
        device = inference_params.last_value.device

        p = torch.zeros(B, device=device, dtype=inference_params.last_value.dtype)
        # Cast the source to p's dtype so this works under autocast (e.g.
        # bf16 forward where boundary_prob is bf16 but p stays fp32 from the
        # inference cache).
        p[boundary_mask] = (
            boundary_prob[boundary_mask, -1].clamp(min=1e-4, max=1 - 1e-4).to(p.dtype)
        )

        new_vals = torch.zeros(
            B, D, device=device, dtype=inference_params.last_value.dtype
        )
        if hidden_states.numel() > 0:
            new_vals[boundary_mask] = hidden_states.squeeze(1).to(
                inference_params.last_value.dtype
            )

        p_b = p.unsqueeze(-1)
        result = p_b * new_vals + (1 - p_b) * inference_params.last_value
        inference_params.last_value.copy_(result)
        return result.unsqueeze(1)
