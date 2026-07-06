"""
Scan-based chunk interface for the H-Net (design: vault
"Expressive Chunk Downsampler-Upsampler", 2026-06-10).

Three modules, drop-in alternatives to the cos-sim RoutingModule /
first-token ChunkLayer / duplicate+EMA DeChunkLayer (packed mode):

* ScanRouter      — p_t = sigmoid(MLP([h_{t-1}, x_t])) where h is a GRU run
                    over each SUBSEQ (reset at episode starts only — running
                    the scan over the model's own chunks would make training
                    sequential, since boundaries define the resets). Returns
                    a RoutingModuleOutput, so everything downstream (STE,
                    ratio loss, confidence) is unchanged.
* ChunkScanEncoder — downsampler: GRU over each CHUNK's tokens (reset at
                    every boundary; chunks are known post-routing so this
                    parallelizes across chunks), with a phase feature
                    appended per token. Outputs the per-token running state
                    h_t (prefix-causal within the chunk) and m temporally
                    ordered sub-latents per chunk (scan states at sub-span
                    ends). Sub-latent m-1 (the last state) is the chunk
                    summary that feeds the inner stage.
* SplineUpsampler — phase-conditioned decoder: per token t in chunk j,
                    output mixes (a) m basis vectors from the PREVIOUS
                    chunk's (inner-processed) sub-latents with position-
                    driven weights w(phi_t), and (b) the own-chunk running
                    state h_t (fresh, prefix-causal), via a learned gate.
                    FiLM on the output by phi_t. Confidence-gated by the
                    router's selected prob so the (2b_t - 1) learning path
                    for p_t is preserved. No single-source softmax anywhere
                    (the prior cross-attn upsampler failure mode).

All modules operate on the packed convention: hidden_states (T_total, D),
cu_seqlens (B+1,), boundary_mask (T_total,) with subseq starts forced True.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing import RoutingModuleOutput, get_seq_idx


def _fourier_phase(phase: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """phase: (T,) in [0, 1] -> (T, 2*n_freqs) fourier features."""
    freqs = (2.0 ** torch.arange(n_freqs, device=phase.device)) * math.pi
    ang = phase[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def _segment_meta(
    seg_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """seg_id: (T,) non-decreasing segment ids starting at 0.

    Returns (offset, seg_len_per_token, seg_len_per_seg):
      offset (T,): position of each token within its segment.
      seg_len_per_token (T,): its segment's length.
      seg_len_per_seg (N,): length per segment.
    """
    T = seg_id.shape[0]
    pos = torch.arange(T, device=seg_id.device)
    seg_len = torch.bincount(seg_id)
    seg_start_pos = pos[
        F.pad(seg_id.diff(), (1, 0), value=1).bool()
    ]  # first pos of each segment
    offset = pos - seg_start_pos[seg_id]
    return offset, seg_len[seg_id], seg_len


def _pad_by_segment(
    x: torch.Tensor, seg_id: torch.Tensor, offset: torch.Tensor, seg_len: torch.Tensor
) -> torch.Tensor:
    """Scatter packed tokens (T, D) into a (N, L_max, D) zero-padded buffer."""
    N = int(seg_len.shape[0])
    L_max = int(seg_len.max())
    out = x.new_zeros(N, L_max, x.shape[-1])
    out[seg_id, offset] = x
    return out


class ScanRouter(nn.Module):
    """Order-aware boundary head: p_t = sigmoid(MLP([h_{t-1}, x_t])).

    h is a causal scan over each subseq (reset at subseq starts ONLY).
    h_{t-1} is the state BEFORE consuming x_t, so p_t is strictly
    prefix-causal: "given the running summary of everything so far, does x_t
    continue the stream or start something new?". Drop-in for RoutingModule
    in packed mode: returns RoutingModuleOutput, prob 1.0 forced at starts.

    Backend: subseqs are EPISODE-length (hundreds-to-thousands of tokens),
    so the scan uses the repo's packed-native ``Mamba2Mixer`` (parallel
    selective-scan kernel, seq_idx resets, recurrent step for AR) on CUDA;
    a per-subseq GRU is the CPU/unit-test fallback (sequential — fine for
    tests, NOT the training path). ``backend='auto'`` picks at call time.
    """

    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        mlp_hidden: int = 128,
        backend: str = "auto",
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        if backend not in ("auto", "mamba", "gru"):
            raise ValueError(f"backend must be auto|mamba|gru, got {backend}")
        self.backend = backend
        self.d_model = d_model
        self.d_state = int(d_state or d_model)
        self._mamba = None
        if backend in ("auto", "mamba"):
            try:
                from .blocks import Mamba2Mixer

                # Mixer outputs (T, d_model): scan state dim == d_model.
                self._mamba = Mamba2Mixer(d_model=d_model, layer_idx=0, **fk)
                self.d_state = d_model
            except Exception:
                if backend == "mamba":
                    raise
        self.gru = nn.GRU(d_model, self.d_state, batch_first=True, **fk)
        self.head = nn.Sequential(
            nn.Linear(self.d_state + d_model, mlp_hidden, **fk),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1, **fk),
        )

    def _scan(self, hidden_states, cu_seqlens, seq_id, offset, seq_len):
        """Returns per-token state AFTER consuming x_t, (T, d_state)."""
        use_mamba = self._mamba is not None and hidden_states.is_cuda
        if self.backend == "gru":
            use_mamba = False
        if use_mamba:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            return self._mamba(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        padded = _pad_by_segment(hidden_states, seq_id, offset, seq_len)
        outs, _ = self.gru(padded)  # (B, L_max, H)
        return outs[seq_id, offset]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> RoutingModuleOutput:
        assert cu_seqlens is not None, "ScanRouter implements packed mode only"
        assert inference_params is None, "packed mode takes no inference_params"
        seq_id = get_seq_idx(cu_seqlens)
        offset, _seq_len_tok, seq_len = _segment_meta(seq_id)
        h_after = self._scan(
            hidden_states, cu_seqlens, seq_id, offset, seq_len
        )  # state AFTER consuming x_t
        # h BEFORE x_t: shift within subseq; zeros at subseq starts.
        h_prev = torch.zeros_like(h_after)
        not_start = offset > 0
        h_prev[not_start] = h_after[
            torch.arange(h_after.shape[0], device=h_after.device)[not_start] - 1
        ]
        logit = self.head(torch.cat([h_prev, hidden_states], dim=-1)).squeeze(-1)
        p = torch.sigmoid(logit)
        p = p.clone()
        p[cu_seqlens[:-1]] = 1.0  # forced boundary at subseq starts (contract)
        boundary_prob = torch.stack((1 - p, p), dim=-1)  # (T, 2)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1
        selected_probs = boundary_prob.gather(-1, selected_idx.unsqueeze(-1))
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )


class ChunkScanEncoder(nn.Module):
    """Downsampler: per-chunk causal GRU (reset at every boundary) + phase
    feature. Emits the per-token running state and m ordered sub-latents per
    chunk. Replaces first-token selection: the chunk token sent to the inner
    stage is sub-latent m-1 (full-chunk summary), and the m sub-latents are
    kept for the upsampler.
    """

    def __init__(
        self,
        d_model: int,
        d_out: int,
        m_sublatents: int = 4,
        n_phase_freqs: int = 4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.m = int(m_sublatents)
        self.n_phase_freqs = n_phase_freqs
        d_in = d_model + 2 * n_phase_freqs
        self.gru = nn.GRU(d_in, d_out, batch_first=True, **fk)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (T, D)
        boundary_mask: torch.Tensor,  # (T,) bool, subseq starts True
        cu_seqlens: torch.Tensor,  # (B+1,)
    ):
        """Returns (h_tok, sub_latents, chunk_summary, next_cu, next_max):
        h_tok        (T, H)    running state per token (prefix-causal).
        sub_latents  (N, m, H) scan states at the m sub-span ends.
        chunk_summary (N, H)   = sub_latents[:, -1] (inner-stage input).
        next_cu      (B+1,)    chunked-space cu_seqlens (ChunkLayer parity).
        next_max     int       max chunks per subseq.
        """
        seg_id = boundary_mask.cumsum(0) - 1  # (T,) chunk index
        offset, len_tok, seg_len = _segment_meta(seg_id)
        phase = offset.float() / len_tok.clamp(min=1).float()
        x = torch.cat(
            [hidden_states, _fourier_phase(phase, self.n_phase_freqs)], dim=-1
        )
        padded = _pad_by_segment(x, seg_id, offset, seg_len)  # (N, L_max, d_in)
        outs, _ = self.gru(padded)  # (N, L_max, H)
        h_tok = outs[seg_id, offset]  # (T, H)
        # Sub-span end offsets per chunk: ceil(len * k / m) - 1, k = 1..m.
        N = seg_len.shape[0]
        ks = torch.arange(1, self.m + 1, device=seg_len.device)  # (m,)
        ends = (
            (torch.ceil(seg_len[:, None].float() * ks[None, :] / self.m) - 1)
            .long()
            .clamp(min=0)
        )  # (N, m)
        sub_latents = outs[torch.arange(N, device=outs.device)[:, None], ends]
        chunk_summary = sub_latents[:, -1]
        next_cu = F.pad(boundary_mask.cumsum(0)[cu_seqlens[1:] - 1], (1, 0))
        next_max = int((next_cu[1:] - next_cu[:-1]).max())
        return h_tok, sub_latents, chunk_summary, next_cu, next_max


class SplineUpsampler(nn.Module):
    """Phase-conditioned upsampler. Per token t in chunk j:

      basis  = W_v(prev-chunk sub-latents)            (m vectors, complete)
      mix_t  = sum_k w_k(phi_t) * basis_k             (position-driven mix)
      fresh_t = W_h(h_t)                              (own-chunk, causal)
      ctx_t  = (1 - g_t) * mix_t + g_t * fresh_t      (learned gate)
      out_t  = c_t * (gamma(phi_t) * W_o(ctx_t) + beta(phi_t))   (FiLM + conf)

    The confidence c_t (router's selected prob) multiplies the output, so
    grad(L -> p_t) keeps the (2b_t - 1) structure. Within-chunk outputs vary
    through w(phi_t), FiLM(phi_t) and h_t — never constant by construction.
    First chunk of each subseq has no previous chunk: basis contribution is
    zeroed there (gate path only).
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        d_model: int,
        m_sublatents: int = 4,
        n_phase_freqs: int = 4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.m = int(m_sublatents)
        pf = 2 * n_phase_freqs
        self.n_phase_freqs = n_phase_freqs
        self.w_v = nn.Linear(d_inner, d_model, **fk)
        self.w_h = nn.Linear(d_state, d_model, **fk)
        self.w_weights = nn.Sequential(
            nn.Linear(pf, 64, **fk), nn.GELU(), nn.Linear(64, self.m, **fk)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, 64, **fk), nn.GELU(), nn.Linear(64, 1, **fk)
        )
        self.film = nn.Sequential(
            nn.Linear(pf, 64, **fk), nn.GELU(), nn.Linear(64, 2 * d_model, **fk)
        )
        self.w_o = nn.Linear(d_model, d_model, **fk)

    def forward(
        self,
        inner_sub_latents: torch.Tensor,  # (N, m, d_inner) inner-processed
        h_tok: torch.Tensor,  # (T, d_state) own-chunk running state
        x_outer: torch.Tensor,  # (T, d_model) outer tokens (query side)
        boundary_mask: torch.Tensor,  # (T,)
        boundary_prob: torch.Tensor,  # (T, 2)
        cu_seqlens: torch.Tensor,  # (B+1,) TOKEN-space
    ) -> torch.Tensor:
        T = x_outer.shape[0]
        seg_id = boundary_mask.cumsum(0) - 1  # (T,)
        offset, len_tok, seg_len = _segment_meta(seg_id)
        phase = offset.float() / len_tok.clamp(min=1).float()
        pf = _fourier_phase(phase, self.n_phase_freqs)

        # Previous-chunk latents, zeroed for each subseq's first chunk.
        basis_all = self.w_v(inner_sub_latents)  # (N, m, d_model)
        prev = torch.zeros_like(basis_all)
        prev[1:] = basis_all[:-1]
        first_chunk_ids = (boundary_mask.cumsum(0) - 1)[cu_seqlens[:-1]]
        prev[first_chunk_ids] = 0.0
        prev_tok = prev[seg_id]  # (T, m, d_model)

        w = torch.softmax(self.w_weights(pf), dim=-1)  # (T, m)
        mix = torch.einsum("tm,tmd->td", w, prev_tok)  # (T, d_model)
        fresh = self.w_h(h_tok)  # (T, d_model)
        g = torch.sigmoid(self.gate(torch.cat([x_outer, fresh], dim=-1)))  # (T,1)
        ctx = (1 - g) * mix + g * fresh
        gamma, beta = self.film(pf).chunk(2, dim=-1)
        out = (1.0 + gamma) * self.w_o(ctx) + beta
        # Confidence gating — preserves the p_t learning path.
        p = boundary_prob[..., -1]
        c = torch.where(boundary_mask, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)
        return out * c.unsqueeze(-1)
