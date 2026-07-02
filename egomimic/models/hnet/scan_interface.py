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
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import StateRowMixin
from .routing import DeChunkLayer, RoutingModuleOutput, get_seq_idx

try:  # RMSNorm for the gated decoder-end normalization (paper "network norm").
    from .blocks import RMSNorm  # type: ignore
except Exception:  # pragma: no cover - blocks always present in the live repo.
    RMSNorm = None  # type: ignore


def _fourier_phase(phase: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """phase: (T,) in [0, 1] -> (T, 2*n_freqs) fourier features."""
    freqs = (2.0 ** torch.arange(n_freqs, device=phase.device)) * math.pi
    ang = phase[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


# --------------------------------------------------------------------------- #
# Phase definitions: teacher-forced (full-chunk) vs prefix-causal (AR).
#
# TF forward uses phase = offset / chunk_len, where chunk_len is the FULL
# (post-completion) chunk length. That value is NOT known at AR emit-time
# (the chunk hasn't ended), so a strict one-output-per-step ``step`` cannot
# reproduce it for non-final tokens. The reconciliation (see vault
# "Expressive Chunk Downsampler-Upsampler", §"constraint: causality"): use a
# PREFIX-CAUSAL phase that is a function of the running offset ALONE,
#
#     phi_t = offset_t / (offset_t + 1)         (in [0, 1), monotone)
#
# This is reproducible identically in TF (offset known per token) and AR
# (running offset known). It agrees with the TF full-chunk phase at the
# chunk's FIRST token (0) and LAST token ((L-1)/L); interior positions
# legitimately differ (TF's interior phase looks ahead to the final length;
# the causal phase cannot, by construction). ``forward(..., causal_phase=True)``
# selects it so the consistency test can compare AR ``step`` against a TF
# reference under the SAME definition (exact match, no tolerance fudge). The
# training path keeps ``causal_phase=False`` (byte-identical to before).
# --------------------------------------------------------------------------- #


def _phase_from_offset_len(
    offset: torch.Tensor, len_tok: torch.Tensor, causal_phase: bool
) -> torch.Tensor:
    """phase per token. TF: offset/len_tok. Causal: offset/(offset+1)."""
    if causal_phase:
        o = offset.float()
        return o / (o + 1.0)
    return offset.float() / len_tok.clamp(min=1).float()


def _causal_phase_scalar(offset: torch.Tensor) -> torch.Tensor:
    """Prefix-causal phase from a per-batch running offset (B,) -> (B,)."""
    o = offset.float()
    return o / (o + 1.0)


@dataclass
class ScanRouterState(StateRowMixin):
    """AR state for ``ScanRouter``. ``has_seen_tokens`` forces a boundary on
    the first step of each episode (subseq start), mirroring the packed
    forward's ``boundary_prob[cu_seqlens[:-1]] = 1.0``. ``gru_h`` is the GRU
    hidden carry (None on the Mamba backend, which uses ``mamba_cache``).
    ``h_prev`` is the scan state BEFORE the current token (h_{t-1}); on the
    Mamba path it stashes the previous step's mixer output (the GRU path reads
    it directly from ``gru_h``)."""

    _DIM1_FIELDS = ("gru_h",)

    has_seen_tokens: torch.Tensor  # (B,) bool
    gru_h: Optional[torch.Tensor] = None  # (1, B, d_state) GRU carry
    mamba_cache: Optional["object"] = None  # MambaCache on the CUDA path
    h_prev: Optional[torch.Tensor] = None  # (B, d_state) carry for Mamba path


@dataclass
class ChunkScanEncoderState(StateRowMixin):
    """AR state for ``ChunkScanEncoder``. Per-chunk GRU carry + running offset
    + a FIXED-capacity buffer of this chunk's per-token GRU outputs (needed to
    extract the m sub-latents at chunk completion, when the full chunk length
    is known). A batched ``(B, cap, H)`` tensor (not a Python list) so nested
    slice/scatter of this state is trivial. ``cap`` bounds the max chunk
    length the AR path can summarize exactly; a chunk longer than ``cap``
    triggers an assert (raise ``max_chunk_len`` in the config / allocate)."""

    _DIM1_FIELDS = ("gru_h",)

    gru_h: torch.Tensor  # (1, B, H) per-chunk GRU carry
    offset: torch.Tensor  # (B,) running position within the current chunk
    buf: torch.Tensor  # (B, cap, H) per-token outputs of the current chunk


@dataclass
class SplineUpsamplerState(StateRowMixin):
    """AR state for ``SplineUpsampler``. ``prev_inner_sub`` is the most
    recently COMPLETED chunk's inner-processed m sub-latents (B, m, d_inner),
    used as the basis for the current chunk's tokens (shift-by-one). Zero
    until the first chunk completes (first chunk of an episode = no basis).
    ``offset`` is the running position within the current chunk."""

    prev_inner_sub: torch.Tensor  # (B, m, d_inner)
    have_prev: torch.Tensor  # (B,) bool — a previous chunk exists
    offset: torch.Tensor  # (B,) running position within the current chunk


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

    def __init__(self, d_model: int, d_state: Optional[int] = None,
                 mlp_hidden: int = 128, backend: str = "auto",
                 n_layer: int = 1, use_block: bool = False,
                 has_mlp: bool = False, mlp_dim: Optional[int] = None,
                 device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        if backend not in ("auto", "mamba", "gru"):
            raise ValueError(f"backend must be auto|mamba|gru, got {backend}")
        self.backend = backend
        self.d_model = d_model
        self.d_state = int(d_state or d_model)
        # Optional MambaBlock-stack scan (gated; DEFAULT use_block=False ->
        # the bare ``self._mamba`` mixer path below is built EXACTLY as before,
        # byte-identical to the current behaviour). When enabled, the scan is
        # routed through ``n_layer`` MambaBlocks instead -- each provides
        # pre-norm RMSNorm + residual + (optional, has_mlp) SwiGLU MLP, which
        # adds the normalization/residual the bare mixer lacks.
        self.use_block = bool(use_block)
        self.n_layer = int(n_layer)
        self.blocks = None
        self._mamba = None
        if backend in ("auto", "mamba"):
            try:
                from .blocks import Mamba2Mixer

                if self.use_block:
                    from .blocks import MambaBlock

                    # MambaBlock has no ``has_mlp`` kwarg -- the MLP is enabled
                    # by passing d_intermediate > 0 (SwiGLU width). Default the
                    # width to d_model when has_mlp is on and no explicit dim is
                    # given.
                    d_int = (
                        int(mlp_dim if mlp_dim is not None else d_model)
                        if has_mlp
                        else 0
                    )
                    self.blocks = nn.ModuleList(
                        [
                            MambaBlock(
                                d_model=d_model,
                                layer_idx=i,
                                d_intermediate=d_int,
                                **fk,
                            )
                            for i in range(self.n_layer)
                        ]
                    )
                    # The block stack's Mamba2 mixer outputs (T, d_model), so the
                    # running scan-state dim is still d_model.
                    self.d_state = d_model
                else:
                    # Mixer outputs (T, d_model): scan state dim == d_model.
                    self._mamba = Mamba2Mixer(
                        d_model=d_model, layer_idx=0, **fk
                    )
                    self.d_state = d_model
            except Exception:
                if backend == "mamba":
                    raise
        # Only create the GRU fallback when neither the bare mixer nor the block
        # stack is available. Keeping an unused GRU on the GPU/Mamba path leaves
        # its params untouched by the loss -> Lightning/DDP "unused parameters".
        self.gru = (
            nn.GRU(d_model, self.d_state, batch_first=True, **fk)
            if (self._mamba is None and self.blocks is None)
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(self.d_state + d_model, mlp_hidden, **fk),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1, **fk),
        )

    def _scan(self, hidden_states, cu_seqlens, seq_id, offset, seq_len):
        """Returns per-token state AFTER consuming x_t, (T, d_state)."""
        use_mamba = (
            (self._mamba is not None or self.blocks is not None)
            and hidden_states.is_cuda
        )
        if self.backend == "gru":
            use_mamba = False
        if use_mamba:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            if self.blocks is not None:
                # Block-stack path: each MambaBlock handles its own pre-norm
                # RMSNorm + residual (and optional SwiGLU). Sequential over the
                # stack; the packed-scan resets are carried via cu_seqlens.
                x = hidden_states
                for blk in self.blocks:
                    x = blk(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                return x
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

    # --------------------------- AR inference --------------------------- #

    def allocate_inference_cache(self, batch_size, device, dtype=None):
        gru_h = None
        mamba_cache = None
        if self.gru is not None:
            gru_h = torch.zeros(1, batch_size, self.d_state, device=device, dtype=dtype)
        h_prev = None
        if self._mamba is not None:
            # max_seqlen unused by Mamba2Mixer.allocate; pass 1.
            mamba_cache = self._mamba.allocate_inference_cache(
                batch_size, 1, device, dtype
            )
            h_prev = torch.zeros(
                batch_size, self.d_state, device=device, dtype=dtype
            )
        elif self.blocks is not None:
            # Block-stack path: one MambaCache per block's mixer (a list stashed
            # in the same ``mamba_cache`` field). ``h_prev`` stashes the previous
            # step's stacked-block output (= h_{t-1}), zeros at episode start.
            mamba_cache = [
                blk.mixer.allocate_inference_cache(batch_size, 1, device, dtype)
                for blk in self.blocks
            ]
            h_prev = torch.zeros(
                batch_size, self.d_state, device=device, dtype=dtype
            )
        return ScanRouterState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            gru_h=gru_h,
            mamba_cache=mamba_cache,
            h_prev=h_prev,
        )

    def step(self, x_t: torch.Tensor, state: ScanRouterState) -> RoutingModuleOutput:
        """One AR step. ``x_t``: (B, 1, D) or (B, D). Returns a per-batch
        RoutingModuleOutput (boundary_prob (B, 2) etc.). Boundary forced True
        on the first step of each episode (``has_seen_tokens`` False).

        The boundary logit uses h_{t-1} (the carry BEFORE consuming x_t),
        matching the forward's strictly prefix-causal ``h_prev``; the carry is
        then advanced by consuming x_t.
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B = x_t.shape[0]
        h_prev = self._step_carry(state)  # (B, d_state), state BEFORE x_t
        logit = self.head(torch.cat([h_prev, x_t], dim=-1)).squeeze(-1)
        p = torch.sigmoid(logit)
        # Advance the carry by consuming x_t (so the NEXT step sees h_t).
        self._step_advance(x_t, state)
        # Force a boundary on the first token of each episode.
        p = torch.where(state.has_seen_tokens, p, torch.ones_like(p))
        state.has_seen_tokens.fill_(True)
        boundary_prob = torch.stack((1 - p, p), dim=-1)  # (B, 2)
        boundary_mask = boundary_prob[..., 1] > 0.5
        selected_probs = boundary_prob.max(dim=-1).values.unsqueeze(-1)
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def _step_carry(self, state: ScanRouterState) -> torch.Tensor:
        """Hidden state BEFORE consuming the current token (h_{t-1}).

        GRU path: the carry IS h_{t-1} (advanced only after the logit).
        Mamba path: ``h_prev`` stashes the previous step's mixer output
        (= h_{t-1}); zeros at episode start, matching the forward's
        ``h_prev[subseq_start] = 0``.
        """
        if state.gru_h is not None:
            return state.gru_h[0]  # (B, d_state)
        return state.h_prev

    def _step_advance(self, x_t: torch.Tensor, state: ScanRouterState) -> None:
        """Consume x_t, advancing the carry to h_t."""
        if state.gru_h is not None:
            _, h_new = self.gru(x_t.unsqueeze(1), state.gru_h)
            state.gru_h = h_new
            return
        if self.blocks is not None:
            # Block-stack path: run each MambaBlock's recurrent step in order,
            # threading the per-block cache (residual handled inside .step()).
            h = x_t.unsqueeze(1)  # (B, 1, D)
            for blk, cache in zip(self.blocks, state.mamba_cache):
                h = blk.step(h, cache)
            state.h_prev = h.squeeze(1)
            return
        out = self._mamba.step(x_t.unsqueeze(1), state.mamba_cache)  # (B,1,D)
        state.h_prev = out.squeeze(1)


class ChunkScanEncoder(nn.Module):
    """Downsampler: per-chunk causal GRU (reset at every boundary) + phase
    feature. Emits the per-token running state and m ordered sub-latents per
    chunk. Replaces first-token selection: the chunk token sent to the inner
    stage is sub-latent m-1 (full-chunk summary), and the m sub-latents are
    kept for the upsampler.
    """

    def __init__(self, d_model: int, d_out: int, m_sublatents: int = 4,
                 n_phase_freqs: int = 4, layernorm: bool = False,
                 device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.m = int(m_sublatents)
        self.n_phase_freqs = n_phase_freqs
        d_in = d_model + 2 * n_phase_freqs
        self.gru = nn.GRU(d_in, d_out, batch_first=True, **fk)
        # Optional post-GRU stability LayerNorm (per-token over the feature dim,
        # so it preserves the per-chunk reset, token ordering and causality).
        # Re-added (gated, DEFAULT OFF -> byte-identical to the bare GRU path)
        # because unnormalized GRU states explode (raw grad norms ~250 observed
        # live). This is post-norm on the recurrent OUTPUT, NOT transformer
        # pre-norm -- pre-norming the GRU input would not bound output
        # explosion. Applied to ``outs`` before h_tok / sub-latents are gathered.
        self.norm = nn.LayerNorm(d_out, **fk) if layernorm else None

    def forward(
        self,
        hidden_states: torch.Tensor,  # (T, D)
        boundary_mask: torch.Tensor,  # (T,) bool, subseq starts True
        cu_seqlens: torch.Tensor,  # (B+1,)
        causal_phase: bool = False,
    ):
        """Returns (h_tok, sub_latents, chunk_summary, next_cu, next_max):
          h_tok        (T, H)    running state per token (prefix-causal).
          sub_latents  (N, m, H) scan states at the m sub-span ends.
          chunk_summary (N, H)   = sub_latents[:, -1] (inner-stage input).
          next_cu      (B+1,)    chunked-space cu_seqlens (ChunkLayer parity).
          next_max     int       max chunks per subseq.

        ``causal_phase``: use the prefix-causal phase offset/(offset+1)
        instead of the full-chunk offset/len_tok, so the per-token GRU input
        (and thus h_tok) is reproducible by the AR ``step``. Sub-latent end
        positions (ceil(len*k/m)-1) are unchanged — they are finalized at
        chunk completion, when the full length is known in AR too.
        """
        seg_id = boundary_mask.cumsum(0) - 1  # (T,) chunk index
        offset, len_tok, seg_len = _segment_meta(seg_id)
        phase = _phase_from_offset_len(offset, len_tok, causal_phase)
        x = torch.cat(
            [hidden_states, _fourier_phase(phase, self.n_phase_freqs)], dim=-1
        )
        padded = _pad_by_segment(x, seg_id, offset, seg_len)  # (N, L_max, d_in)
        L_max = padded.shape[1]
        # Run the GRU on a PACKED sequence so cuDNN's training reserve scales
        # with the total number of real tokens, NOT N*L_max. Critical at INIT:
        # the identity-init router fires boundaries only at subseq starts, so
        # every subseq is ~one giant chunk (L_max == subseq length); the dense
        # (N, L_max, H) GRU reserve otherwise OOMs (~24 GiB). Padded positions
        # are never read below (we gather at valid offsets only), so this is
        # numerically identical to self.gru(padded).
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, seg_len.to("cpu"), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        outs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=L_max
        )  # (N, L_max, H)
        if self.norm is not None:
            outs = self.norm(outs)
        h_tok = outs[seg_id, offset]  # (T, H)
        # Sub-span end offsets per chunk: ceil(len * k / m) - 1, k = 1..m.
        N = seg_len.shape[0]
        ks = torch.arange(1, self.m + 1, device=seg_len.device)  # (m,)
        ends = (
            torch.ceil(seg_len[:, None].float() * ks[None, :] / self.m) - 1
        ).long().clamp(min=0)  # (N, m)
        sub_latents = outs[torch.arange(N, device=outs.device)[:, None], ends]
        chunk_summary = sub_latents[:, -1]
        next_cu = F.pad(boundary_mask.cumsum(0)[cu_seqlens[1:] - 1], (1, 0))
        next_max = int((next_cu[1:] - next_cu[:-1]).max())
        return h_tok, sub_latents, chunk_summary, next_cu, next_max

    # --------------------------- AR inference --------------------------- #

    def allocate_inference_cache(self, batch_size, device, dtype=None, max_chunk_len=256):
        H = self.gru.hidden_size
        return ChunkScanEncoderState(
            gru_h=torch.zeros(1, batch_size, H, device=device, dtype=dtype),
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
            buf=torch.zeros(batch_size, int(max_chunk_len), H, device=device, dtype=dtype),
        )

    def _sub_positions(self, length: int) -> torch.Tensor:
        """Offsets ceil(length*k/m)-1 for k=1..m (same rule as forward)."""
        ks = torch.arange(1, self.m + 1)
        return (torch.ceil(length * ks.float() / self.m) - 1).long().clamp(min=0)

    def step(
        self,
        x_t: torch.Tensor,  # (B, D) or (B, 1, D)
        state: ChunkScanEncoderState,
        boundary_t: torch.Tensor,  # (B,) bool — boundary fired at THIS token
    ):
        """One AR step.

        Order within a step (mirrors TF chunk timing):
          1. For elements whose ``boundary_t`` is True AND have a non-empty
             open chunk: that chunk just COMPLETED -> extract its m sub-latents
             + summary from the buffered per-token outputs (full length now
             known), mark it in ``finished_mask``, reset the GRU carry/offset.
          2. Consume x_t into the (freshly-reset-if-boundary) per-chunk GRU
             with the causal phase feature -> running h_t; append to buffer.

        Returns (h_t, finished_sub, finished_summary, finished_mask):
          h_t            (B, H)     running state after consuming x_t.
          finished_sub   (B, m, H)  completed chunk's sub-latents (0 where not).
          finished_summary (B, H)   = finished_sub[:, -1].
          finished_mask  (B,) bool  which elements completed a chunk this step.
        """
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B, D = x_t.shape
        H = self.gru.hidden_size
        cap = state.buf.shape[1]
        device = x_t.device

        finished_mask = boundary_t & (state.offset > 0)
        finished_sub = x_t.new_zeros(B, self.m, H)
        # ----- 1. finalize completed chunks (per-element; B is small) ----- #
        if finished_mask.any():
            for b in torch.nonzero(finished_mask, as_tuple=False).flatten().tolist():
                L_b = int(state.offset[b].item())  # # tokens consumed in the chunk
                pos = self._sub_positions(L_b).to(device)  # (m,) sub-span ends
                finished_sub[b] = state.buf[b, pos]
        finished_summary = finished_sub[:, -1]

        # ----- 2. consume x_t (reset carry/offset where boundary) ----- #
        if boundary_t.any():
            reset = boundary_t.view(1, B, 1).to(state.gru_h.dtype)
            state.gru_h = state.gru_h * (1.0 - reset)
            state.offset = torch.where(
                boundary_t, torch.zeros_like(state.offset), state.offset
            )
        phase = _causal_phase_scalar(state.offset)  # (B,) offset BEFORE this token
        pf = _fourier_phase(phase, self.n_phase_freqs)  # (B, 2*n_freqs)
        gin = torch.cat([x_t, pf], dim=-1).unsqueeze(1)  # (B, 1, d_in)
        out, h_new = self.gru(gin, state.gru_h)  # out (B,1,H)
        state.gru_h = h_new
        h_t = out.squeeze(1)
        # Mirror the TF forward's post-GRU LayerNorm (gated, default OFF). The
        # normed h_t is what gets buffered, so the sub-latents gathered from the
        # buffer at chunk completion are also normed -- exactly as forward
        # gathers sub_latents from the normed ``outs``. Keeps the AR step and
        # the TF forward numerically consistent when the norm is enabled.
        if self.norm is not None:
            h_t = self.norm(h_t)
        # Write this token's output into the per-row buffer slot at its offset.
        off = state.offset  # (B,) offset of THIS token (0 at a chunk start)
        if int(off.max().item()) >= cap:
            raise RuntimeError(
                f"ChunkScanEncoder AR buffer capacity {cap} exceeded by a chunk "
                f"of length >{cap}; raise max_chunk_len in allocate_inference_cache."
            )
        state.buf[torch.arange(B, device=device), off] = h_t
        state.offset = off + 1
        return h_t, finished_sub, finished_summary, finished_mask


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

    def __init__(self, d_inner: int, d_state: int, d_model: int,
                 m_sublatents: int = 4, n_phase_freqs: int = 4,
                 device=None, dtype=None):
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
        causal_phase: bool = False,
    ) -> torch.Tensor:
        T = x_outer.shape[0]
        seg_id = boundary_mask.cumsum(0) - 1  # (T,)
        offset, len_tok, seg_len = _segment_meta(seg_id)
        phase = _phase_from_offset_len(offset, len_tok, causal_phase)
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

    # --------------------------- AR inference --------------------------- #

    def allocate_inference_cache(self, batch_size, d_inner, device, dtype=None):
        return SplineUpsamplerState(
            prev_inner_sub=torch.zeros(
                batch_size, self.m, d_inner, device=device, dtype=dtype
            ),
            have_prev=torch.zeros(batch_size, device=device, dtype=torch.bool),
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def update_prev(
        self,
        state: SplineUpsamplerState,
        finished_inner_sub: torch.Tensor,  # (B, m, d_inner)
        finished_mask: torch.Tensor,  # (B,) bool
    ) -> None:
        """Cache a just-completed chunk's inner-processed sub-latents as the
        basis for the upcoming chunk's tokens (shift-by-one)."""
        if finished_mask.any():
            m = finished_mask.view(-1, 1, 1)
            state.prev_inner_sub = torch.where(
                m, finished_inner_sub.to(state.prev_inner_sub.dtype),
                state.prev_inner_sub,
            )
            state.have_prev = state.have_prev | finished_mask

    def step(
        self,
        x_t: torch.Tensor,  # (B, d_model) or (B, 1, d_model) outer query
        h_t: torch.Tensor,  # (B, d_state) own-chunk running state
        boundary_t: torch.Tensor,  # (B,) bool
        boundary_prob_t: torch.Tensor,  # (B, 2)
        state: SplineUpsamplerState,
    ) -> torch.Tensor:
        """One AR step. Uses the cached previous-chunk basis + own h_t + the
        prefix-causal phase of the running offset. The basis is zeroed for
        elements with no completed previous chunk (episode's first chunk)."""
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B = x_t.shape[0]
        # Reset the running offset at chunk starts; phase uses the offset
        # BEFORE this token (offset 0 at a boundary).
        if boundary_t.any():
            state.offset = torch.where(
                boundary_t, torch.zeros_like(state.offset), state.offset
            )
        phase = _causal_phase_scalar(state.offset)  # (B,)
        pf = _fourier_phase(phase, self.n_phase_freqs)  # (B, 2*nf)

        basis_all = self.w_v(state.prev_inner_sub)  # (B, m, d_model)
        # Zero the basis for elements with no previous chunk yet.
        basis_all = basis_all * state.have_prev.view(B, 1, 1).to(basis_all.dtype)

        w = torch.softmax(self.w_weights(pf), dim=-1)  # (B, m)
        mix = torch.einsum("bm,bmd->bd", w, basis_all)  # (B, d_model)
        fresh = self.w_h(h_t)  # (B, d_model)
        g = torch.sigmoid(self.gate(torch.cat([x_t, fresh], dim=-1)))  # (B,1)
        ctx = (1 - g) * mix + g * fresh
        gamma, beta = self.film(pf).chunk(2, dim=-1)
        out = (1.0 + gamma) * self.w_o(ctx) + beta
        p = boundary_prob_t[..., -1]
        c = torch.where(boundary_t, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)
        out = out * c.unsqueeze(-1)
        state.offset = state.offset + 1
        return out


@dataclass
class MSublatentDeChunkerState(StateRowMixin):
    """AR state for ``MSublatentDeChunker.step`` (DECHUNK-AR). Mirrors
    ``SplineUpsamplerState`` plus a chunk-rate EMA carry (``ema_prev``)."""
    prev_inner_sub: torch.Tensor  # (B, m, d_inner) POST-EMA sub-latents = W_v basis source (shift-by-one)
    ema_prev: torch.Tensor        # (B, m*d_inner)  chunk-rate EMA carry
    offset: torch.Tensor          # (B,) long       within-chunk phase index (prefix-causal)
    have_prev: torch.Tensor       # (B,) bool       a completed previous chunk exists


class MSublatentDeChunker(nn.Module):
    """Bespoke m-sublatents phase-weighted dechunker (design: vault "Expressive
    Chunk Downsampler-Upsampler", §"m sub-latents per chunk").

    Takes the paper's ``DeChunkLayer`` (duplicate + EMA + c_t STE) and changes
    ONLY the token-reconstruction step: instead of *broadcasting the single
    chunk token* to every fine token of the chunk, each fine token mixes the
    chunk's ``m`` inner-processed sub-latents (register tokens) with phase-driven
    weights::

        recon_t = sum_{k=1..m} softmax(w_weights(fourier(phi_t)))_k * W_v(V_{j,k})

    where ``V_{j,k}`` is the k-th inner-processed register token of token t's OWN
    chunk j (TF-only — no shift-by-one; the AR step is a deferred follow-up) and
    ``phi_t`` is the within-chunk phase (offset / chunk_len in [0, 1)).

    The EMA recurrence + kernel knobs are REUSED from ``DeChunkLayer`` via
    composition (an owned ``DeChunkLayer`` whose ``_ema_kernel`` / ``_ema_loop``
    are untouched). FIX #1: the EMA now runs at CHUNK rate over the flattened m
    sub-latents ``(N, m*d_inner)`` and is applied BEFORE the phase m-mix —
    mirroring the canonical ``DeChunkLayer._forward_packed`` (chunk-rate EMA
    then expand). Pipeline: EMA(chunk-rate sub-latents) -> w_v + phase-softmax
    m-mix to fine tokens -> c_t gate. Because the per-token m-mix happens AFTER
    the smoothing, the within-chunk variation it introduces SURVIVES into the
    output (the prior order — m-mix then a fine-rate EMA — damped it away; that
    was the design tension the variance probe flagged, now resolved).

    FIX #4: the c_t confidence gate uses the straight-through form
    ``out = recon * _ste(c)`` (forward multiplier ≡ 1, so the output is NOT
    attenuated; gradient routed through c), matching the canonical
    ``_ste(selected_probs)`` paths — not the prior literal ``recon * c`` multiply.

    The borrowed phase machinery (``_fourier_phase``, ``_segment_meta``, the
    ``w_v`` projection and the softmax-over-m ``w_weights`` MLP on the Fourier
    phase features) comes from ``SplineUpsampler``; the gate / FiLM / fresh-h
    paths are intentionally NOT borrowed (minimal build).

    Packed convention: chunk-rate sub-latents (N, m, d_inner); boundary_mask
    (T_total,), boundary_prob (T_total, 2); ``cu_seqlens`` is TOKEN-space (B+1,)
    for the phase machinery, ``chunk_cu_seqlens`` is CHUNK-space (B+1,) for the
    chunk-rate EMA.
    """

    def __init__(self, d_inner: int, d_model: int, m_sublatents: int = 4,
                 n_phase_freqs: int = 4, component_end_norm: bool = False,
                 dtype=torch.bfloat16, block_size: int = 256, headdim: int = 32,
                 device=None, _dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": _dtype}
        self.m = int(m_sublatents)
        self.n_phase_freqs = n_phase_freqs
        pf = 2 * n_phase_freqs
        # Phase-mix params borrowed from SplineUpsampler (W_v + softmax-over-m).
        self.w_v = nn.Linear(d_inner, d_model, **fk)
        self.w_weights = nn.Sequential(
            nn.Linear(pf, 64, **fk), nn.GELU(), nn.Linear(64, self.m, **fk)
        )
        # Reuse the paper's DeChunkLayer EMA (recurrence + kernel knobs) verbatim
        # via composition. It owns NO Linear params (only the EMA), so this adds
        # no parameters beyond w_v / w_weights / the optional end-norm.
        #
        # FIX #1: the EMA now runs at CHUNK rate over the flattened m sub-latents
        # (N, m*d_inner), BEFORE the phase m-mix — mirroring the canonical
        # DeChunkLayer._forward_packed (chunk-rate EMA then expand). So it is
        # built for ``m * d_inner`` (the flattened sub-latent dim), NOT d_model.
        # m*d_inner must be divisible by headdim; default headdim=32 divides
        # 4*256=1024 cleanly. DeChunkLayer has zero parameters, so the dim change
        # is param-count-neutral (BC preserved).
        ema_dim = self.m * int(d_inner)
        assert ema_dim % headdim == 0, (
            f"m*d_inner ({ema_dim}) must be divisible by EMA headdim ({headdim}); "
            f"set a compatible headdim for the chunk-rate sub-latent EMA."
        )
        self.ema = DeChunkLayer(
            ema_dim, dtype=dtype, block_size=block_size, headdim=headdim
        )
        # Gated decoder-end RMSNorm (paper "network normalization" at the
        # component end). DEFAULT OFF -> no new param, byte-identical when unset.
        self.component_end_norm = bool(component_end_norm)
        if self.component_end_norm:
            assert RMSNorm is not None, "RMSNorm unavailable; cannot enable component_end_norm"
            # Repo RMSNorm takes (d_model, eps) only — no device/dtype kwargs
            # (matches register_interface's ``RMSNorm(d_model)`` construction).
            self.end_norm = RMSNorm(d_model)
        else:
            self.end_norm = None

    def _residual_adds(self) -> int:
        """FIX #7 (faithfulness): # residual-stream adds this UP-interface
        (dechunker) contributes, mirroring ``RegisterTrunk.n_residual_adds()``
        on the down side. The bare softmax m-mix dechunker has NO residual-block
        stack — its ``w_v`` / ``w_weights`` MLP write the output directly (not
        into a depth-stacked residual stream) and the owned EMA is parameterless
        — so it adds ZERO residual depth. Returns 2*n_blocks for a future
        has_mlp / residual-block dechunker variant. Used by
        ``ChunkerStage._init_weights`` to make the encoder-rule
        (``parent + encoder.height + decoder.height``) faithful for non-root /
        residual-dechunker topologies; the numeric value here is 0, so the
        register-down arch's ``n_residuals`` (and scaled_std=0.01) is unchanged."""
        return 0

    def forward(
        self,
        inner_sub_latents: torch.Tensor,  # (N, m, d_inner) inner-processed regs
        boundary_mask: torch.Tensor,      # (T_total,) bool, subseq starts True
        boundary_prob: torch.Tensor,      # (T_total, 2)
        cu_seqlens: torch.Tensor,         # (B+1,) TOKEN-space (phase machinery)
        chunk_cu_seqlens: Optional[torch.Tensor] = None,  # (B+1,) CHUNK-space (EMA)
        causal_phase: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """Packed TF forward. Returns the dechunked tokens (T_total, d_model).

        FIX #1 pipeline (mirrors the canonical ``DeChunkLayer._forward_packed``,
        which EMAs at CHUNK rate then expands):

          inner_sub_latents (N, m, d_inner)
            -> reshape (N, m*d_inner)
            -> chunk-rate EMA over the N chunks (chunk-space cu_seqlens, chunk-
               rate boundary probs = boundary_prob[...,-1][boundary_mask])
            -> reshape back (N, m, d_inner)
            -> w_v + softmax phase-mix to fine tokens (T, d_model)
            -> c_t straight-through gate (FIX #4: ema * _ste(c), forward mult==1)

        The EMA runs BEFORE the m-mix, so the within-chunk variation introduced
        by the per-token phase weights ``w(phi_t)`` SURVIVES into the output
        (the m-mix happens after the smoothing). This is the design fix the
        variance probe checks.

        ``chunk_cu_seqlens`` is REQUIRED (kept Optional for signature evolution):
        the register-down call site passes ``reg_cu // m``. The EMA kernel's
        ``dt = log(1/(1-p))`` reparam is rate-agnostic, so the same kernel runs
        at chunk rate unchanged.

        ``return_diagnostics`` (test/smoke hook): also return within-chunk
        variance of the FINAL fine-token output (post m-mix, post EMA) and of a
        no-EMA reference, to confirm the m-mix variation is RETAINED.
        """
        assert chunk_cu_seqlens is not None, (
            "MSublatentDeChunker.forward requires chunk_cu_seqlens (chunk-space "
            "cu_seqlens) for the chunk-rate sub-latent EMA (FIX #1)."
        )
        device = boundary_mask.device
        T = boundary_mask.shape[0]
        N, m, D_in = inner_sub_latents.shape
        seg_id = boundary_mask.cumsum(0) - 1                  # (T,) chunk id

        # FIX #5: guard the N-coupling. The dechunker derives N from the raw
        # ``boundary_mask`` (subseq starts forced True upstream), while the
        # encoder forces register/subseq starts independently. If a future
        # non-forcing router de-syncs them, the seg_id <-> sub-latent index map
        # would silently misalign. Assert they agree.
        assert int(seg_id.max().item()) + 1 == N, (
            f"N-coupling broken: boundary_mask implies {int(seg_id.max().item())+1} "
            f"chunks but inner_sub_latents has N={N}. The router must force subseq "
            f"starts (boundary_prob[cu_seqlens[:-1]]=1) so chunk ids align with the "
            f"register encoder's chunk-major layout."
        )

        # --- FIX #1: chunk-rate EMA over the flattened m sub-latents, BEFORE the
        #     m-mix. Chunk-space boundary probs = boundary_prob[...,-1] at the
        #     boundary positions (one per chunk), exactly as the canonical
        #     DeChunkLayer._forward_packed gathers them. ---
        sub_flat = inner_sub_latents.reshape(N, m * D_in)     # (N, m*d_inner)
        p_chunk = torch.clamp(
            boundary_prob[..., -1][boundary_mask], min=1e-4, max=1 - 1e-4
        )  # (N,) chunk-rate boundary probs
        from .routing import _HAS_MAMBA_SCAN  # module-level kernel availability flag
        if _HAS_MAMBA_SCAN and sub_flat.is_cuda and N > 0:
            pos = torch.arange(N, device=device)
            seq_idx = (
                pos[:, None] >= chunk_cu_seqlens[None, 1:]
            ).sum(dim=-1).to(torch.int32)
            ema_flat = self.ema._ema_kernel(
                sub_flat.unsqueeze(0), p_chunk.unsqueeze(0), seq_idx=seq_idx.unsqueeze(0)
            ).squeeze(0)
        else:
            ema_flat = self.ema._ema_loop(
                sub_flat.unsqueeze(0),
                p_chunk.unsqueeze(0),
                seq_starts=(
                    chunk_cu_seqlens[1:-1] if chunk_cu_seqlens.numel() > 2 else None
                ),
            ).squeeze(0)
        ema_sub = ema_flat.reshape(N, m, D_in)                # (N, m, d_inner)

        # --- phase m-mix to fine tokens (AFTER the EMA). ---
        offset, len_tok, _seg_len = _segment_meta(seg_id)
        phase = _phase_from_offset_len(offset, len_tok, causal_phase)
        pf = _fourier_phase(phase, self.n_phase_freqs)        # (T, 2*n_freqs)
        basis_all = self.w_v(ema_sub)                         # (N, m, d_model)
        basis_tok = basis_all[seg_id]                         # (T, m, d_model)
        w = torch.softmax(self.w_weights(pf), dim=-1)         # (T, m)
        out = torch.einsum("tm,tmd->td", w, basis_tok)        # (T, d_model)

        # --- FIX #4: c_t straight-through gate (paper form). _ste(c) has forward
        #     value == 1 (output UNSCALED in the forward pass; no attenuation),
        #     gradient routed through c — matching the canonical _ste(probs)
        #     paths. (Previously: ``out = ema * c`` literal multiply, which
        #     attenuated the forward output.) ---
        from .stages import _ste  # canonical _STE (forward==1, grad pass-through)
        p = boundary_prob[..., -1]
        c = torch.where(boundary_mask, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)  # (T,)
        out = out * _ste(c).unsqueeze(-1)

        if self.end_norm is not None:
            out = self.end_norm(out)

        if return_diagnostics:
            # The variance probe measures within-chunk variation of the FINAL
            # fine-token output (post m-mix, post EMA). Compare against a no-EMA
            # reference (same m-mix straight off the raw sub-latents) so the
            # retention ratio = how much within-chunk structure survives the
            # (now chunk-rate, pre-m-mix) EMA. Both are pre-c_t-gate (the STE
            # gate is forward-identity, so it does not change the variance).
            basis_no_ema = self.w_v(inner_sub_latents)[seg_id]   # (T, m, d_model)
            out_no_ema = torch.einsum("tm,tmd->td", w, basis_no_ema)  # (T, d_model)
            diag = {
                # Final output within-chunk variance (post chunk-rate EMA, post m-mix).
                "within_chunk_var_out": _within_chunk_var(out, seg_id),
                # Reference: same m-mix WITHOUT the EMA (raw sub-latents).
                "within_chunk_var_no_ema": _within_chunk_var(out_no_ema, seg_id),
                # Pre/post names kept for backward-compat with the probe wording:
                # "pre" = no-EMA m-mix reference, "post" = the real output.
                "within_chunk_var_pre_ema": _within_chunk_var(out_no_ema, seg_id),
                "within_chunk_var_post_ema": _within_chunk_var(out, seg_id),
            }
            return out, diag
        return out

    # --------------------------- AR inference --------------------------- #

    def allocate_inference_cache(self, batch_size, d_inner, device, dtype=None):
        return MSublatentDeChunkerState(
            prev_inner_sub=torch.zeros(batch_size, self.m, d_inner, device=device, dtype=dtype),
            ema_prev=torch.zeros(batch_size, self.m * d_inner, device=device, dtype=dtype),
            offset=torch.zeros(batch_size, device=device, dtype=torch.long),
            have_prev=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    def update_prev(self, state, finished_inner_sub, boundary_prob_t, finished_mask):
        """One chunk-rate EMA step on the just-completed chunk's m sub-latents,
        then promote the EMA'd result to the shift-by-one basis. The recurrence
        ``ema = p*sub + (1-p)*ema_prev`` reproduces the TF chunk-rate EMA
        (``DeChunkLayer._ema_loop``) one chunk at a time; ``p`` is the chunk-rate
        boundary prob = ``boundary_prob[...,-1]`` at the finalizing token. Rows
        not finalizing this step are left untouched (p=0 => identity)."""
        if not finished_mask.any():
            return
        B = finished_mask.shape[0]
        sub_flat = finished_inner_sub.reshape(B, -1).to(state.ema_prev.dtype)  # (B, m*d_inner)
        p = torch.zeros(B, device=sub_flat.device, dtype=state.ema_prev.dtype)
        p[finished_mask] = boundary_prob_t[finished_mask, -1].clamp(1e-4, 1 - 1e-4).to(p.dtype)
        p_b = p.unsqueeze(-1)
        new_ema = p_b * sub_flat + (1 - p_b) * state.ema_prev                  # (B, m*d_inner)
        state.ema_prev = torch.where(finished_mask.unsqueeze(-1), new_ema, state.ema_prev)
        fm = finished_mask.view(B, 1, 1)
        state.prev_inner_sub = torch.where(
            fm, state.ema_prev.reshape(B, self.m, -1).to(state.prev_inner_sub.dtype),
            state.prev_inner_sub,
        )
        state.have_prev = state.have_prev | finished_mask

    def step(self, boundary_t, boundary_prob_t, state):
        """One AR fine-token output. Decodes against the cached PREVIOUS chunk's
        EMA'd ``W_v`` basis (shift-by-one — the AR causality cost) using the
        prefix-causal phase of the running offset. Mirrors the TF forward's
        m-mix -> c_t gate -> end_norm. Equality with TF holds under
        ``causal_phase=True`` (the prefix phase ``o/(o+1)``)."""
        B = boundary_t.shape[0]
        if boundary_t.any():  # reset within-chunk phase at chunk starts
            state.offset = torch.where(boundary_t, torch.zeros_like(state.offset), state.offset)
        phase = _causal_phase_scalar(state.offset)            # (B,)
        pf = _fourier_phase(phase, self.n_phase_freqs)        # (B, 2*nf)
        basis = self.w_v(state.prev_inner_sub)                # (B, m, d_model)
        basis = basis * state.have_prev.view(B, 1, 1).to(basis.dtype)  # zero if no prev chunk
        w = torch.softmax(self.w_weights(pf), dim=-1)         # (B, m)
        out = torch.einsum("bm,bmd->bd", w, basis)            # (B, d_model)
        # c_t straight-through gate (forward value == 1; matches TF order: gate then end_norm).
        from .stages import _ste
        p = boundary_prob_t[..., -1]
        c = torch.where(boundary_t, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)
        out = out * _ste(c).unsqueeze(-1)
        if self.end_norm is not None:
            out = self.end_norm(out)
        state.offset = state.offset + 1
        return out


def _within_chunk_var(x: torch.Tensor, seg_id: torch.Tensor) -> float:
    """Mean (over feature dims, then over chunks) within-chunk variance of a
    packed (T, D) tensor grouped by ``seg_id`` (T,). Chunks of length 1
    contribute 0. Returned as a python float for logging."""
    xf = x.float()
    N = int(seg_id.max().item()) + 1 if seg_id.numel() > 0 else 0
    if N == 0:
        return 0.0
    D = xf.shape[-1]
    counts = torch.bincount(seg_id, minlength=N).clamp(min=1).float()  # (N,)
    sum_ = torch.zeros(N, D, device=xf.device, dtype=xf.dtype)
    sum_.index_add_(0, seg_id, xf)
    mean = sum_ / counts[:, None]                                      # (N, D)
    sq = torch.zeros(N, D, device=xf.device, dtype=xf.dtype)
    sq.index_add_(0, seg_id, xf * xf)
    var = sq / counts[:, None] - mean * mean                          # (N, D)
    # Average over chunks of length > 1 only (length-1 chunks have no spread).
    multi = torch.bincount(seg_id, minlength=N) > 1
    if multi.any():
        return float(var[multi].clamp(min=0).mean().item())
    return 0.0
