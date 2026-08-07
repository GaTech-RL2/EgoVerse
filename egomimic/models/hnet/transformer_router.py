"""
Phase-Summary Boundary Router ("PSER") for the H-Net dynamic-chunking model.

A drop-in alternative to ``scan_interface.ScanRouter`` / ``routing.RoutingModule``
(packed mode) that decides boundaries from a *causal Transformer* summary of the
stream so far plus a *hard-reset running segment summary* and a *duration prior*:

    u_t   = W_in(x_t)                                   (linear embed -> d_model)
    h     = R(u)                                        (depth causal xformer layers)
    h_t  := LayerNorm(h_t)                              (prenorm before any head)
    r_{t-1} = mean{ h_j : last_cut < j <= t-1 }         (HARD-RESET segment summary)
    tau_t = t - last_cut                                (segment dwell time)
    ell_t = MLP([h_t; r; h_t-r; h_t*r]) + w_l*g_dwell(tau_t)   (variant "pser")
          = MLP(h_t)                                            (variant "prefix_global")
    p_t   = sigmoid(ell_t / T_temp)
    p_t  <- 0  where tau_t < L_min                      (min-length mask; pser ONLY)
    p_t  <- 1  at every subseq start                    (forced boundary, contract)
    b_t   = 1[p_t > 0.5]                                (STE-hard forward decision)

Returns a ``RoutingModuleOutput`` (boundary_prob (T,2), boundary_mask (T,),
selected_probs (T,1)) so everything downstream (STE, ratio loss, confidence) is
unchanged. Mirrors ScanRouter's (forward, step, allocate_inference_cache) and the
ScanRouterState dataclass shape.

VARIANT ``prefix_global`` (the one we run) is FULLY PARALLEL: ``ell_t = MLP(h_t)``
depends only on the per-token ``h_t`` (no segment summary, no dwell, no b0), and
there is NO L_min — so boundaries = ``sigmoid(MLP(h)/T_temp) > 0.5`` + forced
subseq-starts, computed in ONE GPU pass (``_forward_prefix_global``), exactly like
the upstream cos-sim ``RoutingModule``. No scan, no per-token ``.item()``.

THE PARALLEL-SCAN CAUSALITY INVARIANT below applies to variant ``pser`` ONLY:
``b_k`` and ``tau_{k+1}`` / the segment reset all depend on ``b_k``, so there is
ONE genuine left-to-right dependency (the segmentation cannot be parallelized —
exactly as in the EMA / scan-router path). We resolve it identically:

  1. compute ``h`` FULLY in parallel (the causal transformer);
  2. then ONE segmented left-to-right pass that, at each token k, reads ``p_k``
     from the SAME ``ell_k = MLP(...) + w_l*g_dwell(tau_k)`` that ``step`` uses,
     thresholds to ``b_k``, and updates segment-id / tau / running-mean.

The cut decision thresholds on the IDENTICAL ``ell`` that ``step`` produces (the
``_logit`` core is shared verbatim). Thresholding before adding dwell, or using a
stale tau, would make TF and AR diverge and GATE-1 fail.

Gradient flows through the ``h_j`` values that build ``r`` (pser) and through the
MLP; the segment IDs (the reset *structure*) are detached — only the membership is
non-differentiable, the summed values are not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import IsotropicInferenceParams, KVCache, MultiHeadAttention, RMSNorm
from .routing import (
    RoutingModuleOutput,
    get_seq_idx,
    routing_output_from_probs,
)


# --------------------------------------------------------------------------- #
# Duration prior.
# --------------------------------------------------------------------------- #


def _g_dwell(
    tau: torch.Tensor,
    target_len: float,
    L_min: int,
    sigma: float = 0.5,
    floor: float = -30.0,
) -> torch.Tensor:
    """Log-normal-shaped dwell bonus on the boundary logit.

    Very negative for ``tau < L_min`` (a cut there is hard-masked anyway, but the
    prior also discourages it softly); peaks near the target chunk length
    ``N = 1/target_rate`` and decays for longer dwell. Shape (matches ``tau``).

    g(tau) = -((log tau - log N)^2) / (2 sigma^2),  with g(tau<L_min) = floor.

    The peak (==0) sits at tau==N; it is a *relative* bonus (a constant offset is
    absorbed by ``b0``). ``floor`` keeps the pre-L_min region finite for autograd.
    """
    tau_f = tau.float()
    logN = math.log(max(float(target_len), 1.0))
    safe_tau = tau_f.clamp(min=1.0)
    g = -((safe_tau.log() - logN) ** 2) / (2.0 * sigma * sigma)
    g = torch.where(tau_f < float(L_min), tau_f.new_full((), floor), g)
    return g


# --------------------------------------------------------------------------- #
# AR state (mirrors ScanRouterState).
# --------------------------------------------------------------------------- #


@dataclass
class PhaseSummaryRouterState:
    """AR state for ``PhaseSummaryRouter`` (mirror of ``ScanRouterState``).

    ``has_seen_tokens`` forces a boundary on the first step of each episode
    (subseq start), matching the packed forward's ``p[cu_seqlens[:-1]] = 1``.
    ``kv`` is one ``KVCache`` per causal-attention layer (RoPE applied at the
    running position ``t_idx``). ``seg_sum`` / ``seg_count`` accumulate the
    hard-reset running mean ``r`` of the post-LayerNorm ``h`` within the OPEN
    segment; ``last_cut_pos`` / ``t_idx`` drive ``tau``.
    """

    has_seen_tokens: torch.Tensor  # (B,) bool
    kv: List[KVCache]              # per-layer KV caches
    last_cut_pos: torch.Tensor    # (B,) long — position of the current segment start
    t_idx: torch.Tensor           # (B,) long — running token position within the subseq
    seg_sum: torch.Tensor         # (B, d_model) sum of h over the open segment
    seg_count: torch.Tensor       # (B,) long — # tokens in the open segment


class PhaseSummaryRouter(nn.Module):
    """Phase-summary boundary router. RoutingModuleOutput-compatible drop-in for
    ScanRouter (packed mode). Every dimension is a constructor arg with a sane
    default (build-for-the-future)."""

    def __init__(
        self,
        d_model: int,
        depth: int = 3,
        n_heads: int = 4,
        ff_mult: int = 4,
        L_min: int = 2,
        target_rate: float = 0.125,
        T_temp: float = 1.0,
        causal: bool = True,
        variant: str = "pser",
        rotary_emb_dim: Optional[int] = None,
        rotary_emb_base: float = 10000.0,
        mlp_hidden: int = 128,
        w_l: float = 1.0,
        dwell_sigma: float = 0.5,
        residual_mlp_init_scale: float = 1e-3,
        win_init_scale: float = 1e-2,
        calibrate_b0: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        if variant not in ("pser", "prefix_global"):
            raise ValueError(f"variant must be pser|prefix_global, got {variant}")
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.L_min = int(L_min)
        self.target_rate = float(target_rate)
        self.target_len = 1.0 / max(self.target_rate, 1e-6)
        self.T_temp = float(T_temp)
        self.causal = bool(causal)
        self.variant = variant
        self.w_l = float(w_l)
        self.dwell_sigma = float(dwell_sigma)
        head_dim = d_model // n_heads
        if rotary_emb_dim is None:
            # Default: rotate the full head (even) — gives the AR KV path
            # position awareness without extra config burden.
            rotary_emb_dim = head_dim - (head_dim % 2)
        self.rotary_emb_dim = int(rotary_emb_dim)

        # 1. Linear embed to d_model.
        self.W_in = nn.Linear(d_model, d_model, **fk)

        # 2. ``depth`` causal Transformer layers (REUSE the repo attention block).
        #    Pre-norm self-attn + GELU MLP, mirroring TransformerBlock's structure
        #    but kept self-contained here so PSER owns its stack. Attention is the
        #    repo's MultiHeadAttention with RoPE + KV-cache .step().
        self.attn_norms = nn.ModuleList(
            [RMSNorm(d_model) for _ in range(self.depth)]
        )
        self.attns = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model,
                    n_heads,
                    causal=causal,
                    dropout=0.0,
                    rotary_emb_dim=self.rotary_emb_dim,
                    rotary_emb_base=rotary_emb_base,
                )
                for _ in range(self.depth)
            ]
        )
        self.mlp_norms = nn.ModuleList(
            [RMSNorm(d_model) for _ in range(self.depth)]
        )
        ff_dim = int(ff_mult * d_model)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_dim, **fk),
                    nn.GELU(),
                    nn.Linear(ff_dim, d_model, **fk),
                )
                for _ in range(self.depth)
            ]
        )
        # Per-token prenorm before the head (mirrors the scan_interface prenorm
        # fix that stopped grad spikes).
        self.head_norm = nn.LayerNorm(d_model, **fk)

        # 5. LOGIT MLP. pser: [h; r; h-r; h*r] -> 4*d_model. prefix_global: [h].
        feat_dim = 4 * d_model if variant == "pser" else d_model
        self.logit_mlp = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden, **fk),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1, **fk),
        )
        # INIT: residual MLP final layer ~0 + small W_in so the boundary logit is
        # near 0 early → p_t ~ 0.5 uniformly, exactly like the upstream cos-sim
        # router at init (random projections → cos_sim~0 → prob~0.5). No
        # target-rate bias: the ratio loss drives the boundary rate to target.
        # The MLP's own final-layer bias is the only additive offset, learned.
        with torch.no_grad():
            self.logit_mlp[-1].weight.mul_(residual_mlp_init_scale)
            self.logit_mlp[-1].bias.zero_()
            self.logit_mlp[0].weight.mul_(residual_mlp_init_scale)
            self.W_in.weight.mul_(win_init_scale)
            self.W_in.bias.zero_()

    # ------------------------------------------------------------------ #
    # Shared logit core (TF parallel pass AND AR step read THIS exact fn).
    # ------------------------------------------------------------------ #

    def _logit(self, h: torch.Tensor, r: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """ell from post-LayerNorm h, running segment summary r (= h's mean over
        the open segment, BEFORE the current token), and dwell tau.

        h, r: (..., d_model). tau: (...). Returns (...,) logit (pre-sigmoid,
        pre-temperature, pre-b0 are folded in here so callers just sigmoid)."""
        if self.variant == "pser":
            feat = torch.cat([h, r, h - r, h * r], dim=-1)
        else:  # prefix_global
            feat = h
        ell = self.logit_mlp(feat).squeeze(-1)
        if self.variant == "pser":
            g = _g_dwell(tau, self.target_len, self.L_min, sigma=self.dwell_sigma)
            ell = ell + self.w_l * g
        return ell / self.T_temp

    def _encode(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run W_in + ``depth`` causal transformer layers + head LayerNorm.
        Returns post-LayerNorm h, (T, d_model)."""
        x = self.W_in(hidden_states)
        for i in range(self.depth):
            x = x + self.attns[i](
                self.attn_norms[i](x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            x = x + self.mlps[i](self.mlp_norms[i](x))
        return self.head_norm(x)

    # ------------------------------------------------------------------ #
    # Teacher-forced packed forward.
    # ------------------------------------------------------------------ #

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> RoutingModuleOutput:
        assert cu_seqlens is not None, "PhaseSummaryRouter implements packed mode only"
        assert inference_params is None, "packed mode takes no inference_params"
        device = hidden_states.device
        T = hidden_states.shape[0]
        D = self.d_model
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        h = self._encode(hidden_states, cu_seqlens, max_seqlen)  # (T, D)

        # prefix_global: ell_t = MLP(h_t) is purely PER-TOKEN — no segment
        # summary, and (since L_min was dropped) no min-length constraint — so
        # boundaries compute in ONE parallel GPU pass, exactly like the upstream
        # cos-sim RoutingModule (threshold + forced-start scatter + argmax). No
        # CPU, no .item(), fully torch.compile / CUDA-graph friendly.
        if self.variant == "prefix_global":
            return self._forward_prefix_global(h, cu_seqlens)

        # ---- pser variant ONLY past here: genuine left-to-right segment
        # recurrence (ell depends on the running segment mean r), so it cannot be
        # parallelized; kept as the eager loop. (We run prefix_global.) ----
        seq_id = get_seq_idx(cu_seqlens)  # (T,)
        starts = set(int(s) for s in cu_seqlens[:-1].tolist())

        # ONE segmented left-to-right pass. The segmentation is data-dependent
        # (b_k feeds tau_{k+1} and the reset), so this recurrence is genuinely
        # sequential — exactly as the scan/EMA path. h is already parallel; this
        # loop only does scalar/vector bookkeeping + the shared _logit core.
        p = torch.empty(T, device=device, dtype=h.dtype)
        seg_sum = h.new_zeros(D)
        seg_count = 0
        last_cut = 0
        for k in range(T):
            if k in starts:
                # Forced boundary at subseq start: open a fresh segment AT k.
                seg_sum = h[k]
                seg_count = 1
                last_cut = k
                p[k] = 1.0
                continue
            # r_{k-1}: mean of h over the OPEN segment (last_cut < j <= k-1).
            # Detach the count/structure; gradient flows through seg_sum values.
            if seg_count > 0:
                r = seg_sum / float(seg_count)
            else:
                r = torch.zeros_like(h[k])
            tau = torch.tensor(float(k - last_cut), device=device)
            ell = self._logit(h[k], r, tau)
            p_k = torch.sigmoid(ell)
            # HARD MASK: p<-0 where tau < L_min (cannot cut before min length).
            if (k - last_cut) < self.L_min:
                p_k = p_k * 0.0
            p[k] = p_k
            b_k = bool(p_k.item() > 0.5)
            if b_k:
                seg_sum = h[k]
                seg_count = 1
                last_cut = k
            else:
                seg_sum = seg_sum + h[k]
                seg_count += 1

        return routing_output_from_probs(p)

    # ------------------------------------------------------------------ #
    # Parallel TF forward for variant="prefix_global" (pure GPU, no L_min).
    # ------------------------------------------------------------------ #

    def _forward_prefix_global(
        self,
        h: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> RoutingModuleOutput:
        """Boundaries for variant='prefix_global', fully on the GPU.

        ``ell_t = MLP(h_t)/T_temp`` is purely per-token (no segment summary, no
        dwell, no b0 bias), so all probabilities compute in ONE batched op. With L_min
        dropped there is no sequential min-length constraint left, so this exactly
        mirrors the upstream ``RoutingModule._forward_packed``: threshold every
        position + force a boundary at each subseq start via a scatter, then
        argmax. No Python loop, no ``.item()`` per token, no host sync.

        Grad: forced-start positions get a constant 1.0 (MLP grad severed there,
        as in upstream); every other position carries the sigmoid(MLP) gradient.
        """
        ell = self.logit_mlp(h).squeeze(-1) / self.T_temp  # (T,)
        p = torch.sigmoid(ell)                             # (T,)
        # Force a boundary at the first token of every subseq (upstream parity:
        # ``boundary_prob[cu_seqlens[:-1]] = 1.0``). Clone first so the in-place
        # scatter is autograd-safe.
        p = p.clone()
        p[cu_seqlens[:-1]] = 1.0
        return routing_output_from_probs(p)

    # ------------------------------------------------------------------ #
    # AR inference.
    # ------------------------------------------------------------------ #

    def allocate_inference_cache(self, batch_size, device, dtype=None, T_max=None):
        if T_max is None:
            raise ValueError(
                "PhaseSummaryRouter.allocate_inference_cache needs T_max (the KV "
                "cache length). ChunkerStage passes max_seqlen through."
            )
        if not self.causal:
            raise AssertionError(
                "PhaseSummaryRouter.step is only supported for causal=True; the "
                "causal=False arm is a training/TF-only bidirectional ABLATION "
                "(see module docstring)."
            )
        kv = [
            self.attns[i].allocate_inference_cache(
                batch_size, int(T_max), device, dtype
            )
            for i in range(self.depth)
        ]
        return PhaseSummaryRouterState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            kv=kv,
            last_cut_pos=torch.zeros(batch_size, device=device, dtype=torch.long),
            t_idx=torch.zeros(batch_size, device=device, dtype=torch.long),
            seg_sum=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
            seg_count=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def _encode_step(self, x_t: torch.Tensor, state: PhaseSummaryRouterState) -> torch.Tensor:
        """One AR token through W_in + causal layers (KV-cache) + head LayerNorm.
        x_t: (B, D). Returns post-LayerNorm h_t (B, D). RoPE is applied inside
        each attention at the per-row position ``cache.offsets`` (== t_idx)."""
        x = self.W_in(x_t).unsqueeze(1)  # (B, 1, D)
        for i in range(self.depth):
            h = self.attn_norms[i](x)
            x = x + self.attns[i].step(h, state.kv[i])  # KVCache scatter + masked attn
            x = x + self.mlps[i](self.mlp_norms[i](x))
        return self.head_norm(x.squeeze(1))  # (B, D)

    def step(self, x_t: torch.Tensor, state: PhaseSummaryRouterState) -> RoutingModuleOutput:
        """One AR step. ``x_t``: (B, 1, D) or (B, D). Returns a per-batch
        RoutingModuleOutput. Boundary forced True on the first step of each
        episode (``has_seen_tokens`` False). Reads p_t from the SAME ``_logit``
        core as the TF forward so GATE-1 (TF=AR) holds exactly."""
        if not self.causal:
            raise AssertionError(
                "PhaseSummaryRouter.step requires causal=True (causal=False is a "
                "TF-only bidirectional ablation)."
            )
        if x_t.dim() == 3:
            x_t = x_t.squeeze(1)
        B = x_t.shape[0]
        h_t = self._encode_step(x_t, state)  # (B, D)

        # r = running mean of the OPEN segment BEFORE this token; 0 if empty
        # (first-of-segment) so the residual ~ 0 there.
        cnt = state.seg_count.clamp(min=1).unsqueeze(-1).to(h_t.dtype)
        r = state.seg_sum / cnt
        r = torch.where(state.seg_count.unsqueeze(-1) > 0, r, torch.zeros_like(r))

        tau = (state.t_idx - state.last_cut_pos).to(h_t.dtype)  # (B,)
        ell = self._logit(h_t, r, tau)  # (B,)
        p = torch.sigmoid(ell)
        # HARD MASK: p<-0 where tau < L_min. pser ONLY — prefix_global dropped
        # L_min in the TF forward, so skip it here too to keep GATE-1 (TF=AR).
        if self.variant == "pser":
            p = torch.where(tau < float(self.L_min), torch.zeros_like(p), p)
        # Force p=1 at the first token of each episode (subseq start).
        p = torch.where(state.has_seen_tokens, p, torch.ones_like(p))

        b_t = p > 0.5  # (B,) STE-hard decision

        # UPDATE accumulators (mirror the TF segmented pass exactly).
        #   boundary fired  -> open a fresh segment AT this token: seg=h_t, cnt=1,
        #                       last_cut = t_idx.
        #   no boundary     -> append h_t to the open segment.
        state.seg_sum = torch.where(b_t.unsqueeze(-1), h_t, state.seg_sum + h_t)
        state.seg_count = torch.where(
            b_t, torch.ones_like(state.seg_count), state.seg_count + 1
        )
        state.last_cut_pos = torch.where(b_t, state.t_idx, state.last_cut_pos)
        state.t_idx = state.t_idx + 1
        state.has_seen_tokens.fill_(True)

        return routing_output_from_probs(p)
