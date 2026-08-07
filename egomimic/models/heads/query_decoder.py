"""ACT/HPT-style action-query readout for the chunked BC-RNN-Transformer.

This is a GATED ALTERNATIVE to the ``GMMActionHead``'s default flat-linear
chunk readout (``chunk_head: "linear"``). It is selected by ``chunk_head:
"queries"`` and changes ONLY the readout shape -- everything upstream (the
ObsEncoder, the TransformerCore that produces the per-obs-step causal features
``h_0..h_{T-1}``, the obs-striding/action-chunking window machinery) and the
final GMM distribution semantics (``num_modes`` diagonal Gaussians per action
position, tanh means, low_noise_eval sampling) are IDENTICAL.

WHAT THE LINEAR HEAD DOES (the baseline, ``chunk_head: "linear"``):
    each obs-step's feature ``h_k (d_model,)`` is fed through ONE wide linear
    ``Linear(d_model -> chunk_len * per_step)`` whose output is reshaped to
    ``chunk_len`` independent GMMs. So the 8 chunk positions are read out by 8
    disjoint slices of a single projection of the SAME feature ``h_k`` -- there
    is no interaction between chunk positions and no per-position parameter
    sharing beyond the shared input ``h_k``.

WHAT THIS QUERY HEAD DOES (``chunk_head: "queries"``, ACT/HPT-style):
    ``chunk_len`` LEARNABLE query embeddings (one per chunk position) are
    refined by a small transformer decoder of ``n_layers`` blocks. In each block
    the queries (a) SELF-ATTEND among themselves (the 8 plan steps coordinate --
    inter-step plan coherence) and (b) CROSS-ATTEND over the CAUSAL context =
    the core's per-step features ``h_0..h_k`` for the CURRENT obs-step ``k``
    (a query at obs-step ``k`` may only see context positions ``<= k`` -- no
    future). Each refined query feature is then mapped by a SINGLE SHARED GMM
    projection ``Linear(d_model -> per_step)`` (ACT-style: the per-query identity
    comes entirely from the learned query embeddings, NOT from per-position
    projection weights). The ``per_step`` outputs of the 8 queries are stacked
    into the SAME flat ``(.., chunk_len*per_step)`` layout the existing
    ``GMMActionHead._make_dist`` / ``nll`` / ``decode`` consume, so all GMM
    distribution + loss + sampling code is reused unchanged.

CAUSALITY (train == rollout by construction):
    At TRAINING the decoder is run for ALL obs-steps ``k`` of a window at once.
    The queries are broadcast over the ``(N, T)`` obs-steps and the cross-
    attention uses a causal mask of shape ``(T, T)`` so the queries at obs-step
    ``k`` attend to context positions ``0..k`` only. At ROLLOUT the
    TransformerCore buffer provides exactly ``h_0..h_k`` for the current step
    ``k``, so feeding those K+1 context features (no mask needed -- the buffer
    IS the causal prefix) reproduces the SAME query refinement as training row
    ``k``. Hence train == rollout context by construction (verified by the
    parity + causality GPU proofs).
"""

from typing import Optional

import torch
import torch.nn as nn


class _QueryDecoderBlock(nn.Module):
    """One ACT-style query-refinement block: self-attn(queries) + cross-attn(
    queries -> causal context) + FF, all pre-norm residual.

    Pre-norm (norm-before-sublayer) matches ``TransformerCore``'s
    ``_CausalTransformerBlock`` so the two stacks share the stable GPT-style
    arrangement. Both attentions are ``batch_first=True``.

    The forward operates on queries FLATTENED over the (batch, obs-step) axes so
    a single call refines the queries for EVERY obs-step of EVERY window at once
    (the batched form -- no python loop over the T obs-steps). The cross-
    attention key-padding/attn mask enforces causality per obs-step.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=float(dropout), batch_first=True,
        )
        self.ln_q2 = nn.LayerNorm(d_model)
        self.ln_ctx = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=float(dropout), batch_first=True,
        )
        self.ln_q3 = nn.LayerNorm(d_model)
        hidden = int(ff_mult) * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        q: torch.Tensor,
        ctx: torch.Tensor,
        ctx_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``q (G, Q, d_model)`` queries, ``ctx (G, S, d_model)`` context.

        ``G`` is the flattened (N*T) group axis (one group per (window, obs-step)
        pair). ``ctx_key_padding_mask (G, S)`` is True where a context key is
        DISALLOWED (i.e. a future / padded context position for this obs-step).
        Returns refined ``q (G, Q, d_model)``.
        """
        # (a) self-attention among the Q queries (plan-step coherence). No mask:
        # all queries see all queries (they are a SET, jointly planned).
        h = self.ln_q1(q)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + self.drop(a)
        # (b) cross-attention: queries attend over the causal context h_<=k.
        hq = self.ln_q2(q)
        hctx = self.ln_ctx(ctx)
        c, _ = self.cross_attn(
            hq, hctx, hctx,
            key_padding_mask=ctx_key_padding_mask, need_weights=False,
        )
        q = q + self.drop(c)
        # (c) feed-forward.
        h = self.ln_q3(q)
        q = q + self.ff(h)
        return q


class QueryActionDecoder(nn.Module):
    """ACT/HPT-style action-query readout producing flat GMM params per obs-step.

    Drop-in for the ``GMMActionHead``'s flat-linear chunk projection: given the
    core's per-step features ``h (N, T, d_model)`` it returns flat GMM params
    ``(N, T, chunk_len * per_step)`` IDENTICAL in layout to
    ``GMMActionHead.proj(h)`` when ``chunk_len > 1`` -- so the caller hands the
    result straight to ``gmm_head.nll`` / ``gmm_head.decode`` / ``_make_dist``,
    all of which depend only on ``chunk_len`` / ``num_modes`` / ``action_dim``.

    The per_step (= M*(2D+1)) projection is a SINGLE shared ``Linear`` applied to
    every refined query (ACT-style weight sharing across the chunk positions);
    per-position identity comes from the ``chunk_len`` learned query embeddings.

    Args:
        d_model:    decoder width == core hidden (TransformerCore.hidden_dim).
        chunk_len:  number of action positions / queries (== gmm_head.chunk_len).
        per_step:   GMM params per action position (== gmm_head.per_step =
                    num_modes*(2*action_dim+1)). The shared head emits this many
                    numbers per query.
        n_layers:   number of query-refinement blocks (1-2 typical).
        n_heads:    attention heads (must divide d_model).
        ff_mult:    feed-forward hidden = ff_mult * d_model.
        dropout:    dropout in attention + FF + residual.
        max_window: max # of context positions (== rnn_horizon). The causal
                    cross-attention only ever sees <= this many context features.
    """

    def __init__(
        self,
        d_model: int,
        chunk_len: int,
        per_step: int,
        n_layers: int = 1,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_window: int = 10,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.chunk_len = int(chunk_len)
        self.per_step = int(per_step)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_window = int(max_window)
        if self.chunk_len < 1:
            raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        # chunk_len LEARNABLE query embeddings -- the per-position identity.
        self.query_emb = nn.Parameter(torch.zeros(self.chunk_len, self.d_model))
        nn.init.normal_(self.query_emb, std=0.02)
        # OPTIONAL learned positional embedding over the CONTEXT positions so the
        # queries can tell h_0..h_k apart by obs-step index (the TransformerCore
        # already adds window-relative positions to its OUTPUT features, but an
        # explicit context-position embedding here is cheap and lets the cross-
        # attention key the obs-step ordering directly). Indexed 0..max_window-1.
        self.ctx_pos_emb = nn.Embedding(self.max_window, self.d_model)
        self.blocks = nn.ModuleList(
            [
                _QueryDecoderBlock(
                    self.d_model, self.n_heads, int(ff_mult), float(dropout)
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        # SHARED GMM projection across the chunk_len queries (ACT-style): one
        # Linear(d_model -> per_step) applied to every refined query feature.
        self.shared_proj = nn.Linear(self.d_model, self.per_step)

    # ---- training (batched over all obs-steps of all windows) -------------

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Per-step core features ``h (N, T, d_model)`` -> flat GMM params
        ``(N, T, chunk_len * per_step)``.

        Batched over the ``T`` obs-steps: the ``chunk_len`` queries are broadcast
        across all ``N*T`` (window, obs-step) pairs and refined in one shot. The
        cross-attention causal mask lets the queries at obs-step ``k`` attend to
        context features ``h_0..h_k`` only (no future).

        Two regimes, mirroring ``TransformerCore.forward`` (so the teacher-forced
        eval overlay matches rollout semantics):

          * ``T <= max_window`` (the TRAINING path: length-``rnn_horizon``
            windows): one batched ``_decode_window`` over positions 0..T-1.

          * ``T > max_window`` (the EVAL OVERLAY path: ``forward_eval`` feeds the
            FULL episode): chunk into NON-OVERLAPPING length-``max_window``
            windows, decode each fresh (each window's obs-step k cross-attends to
            its own window positions 0..k), and concat back to ``(N, T, ...)`` --
            IDENTICAL window boundaries to the rollout's rnn_horizon re-inits and
            to ``TransformerCore``'s own overlay chunking, so the two stack up.
        """
        T = h.shape[1]
        if T <= self.max_window:
            return self._decode_window(h)
        # long-sequence (overlay) path: non-overlapping max_window chunks.
        W = self.max_window
        outs = [self._decode_window(h[:, s : s + W]) for s in range(0, T, W)]
        return torch.cat(outs, dim=1)  # (N, T, chunk_len*per_step)

    def _decode_window(self, h: torch.Tensor) -> torch.Tensor:
        """Batched per-obs-step decode over ONE window ``h (N, T<=max_window,
        d_model)`` -> flat GMM params ``(N, T, chunk_len*per_step)``."""
        N, T, Dm = h.shape
        if T > self.max_window:
            raise ValueError(
                f"window length T ({T}) exceeds max_window ({self.max_window})."
            )
        device = h.device
        # context = the SAME T per-step features for every obs-step's queries,
        # restricted causally by the key-padding mask. ctx (N, T_ctx=T, d_model).
        ctx_pos = torch.arange(T, device=device, dtype=torch.long)
        ctx = h + self.ctx_pos_emb(ctx_pos)[None, :, :]  # (N, T, d_model)
        # broadcast over the obs-step axis -> group axis G = N*T.
        # ctx_g[g] holds the FULL window context; causality is applied by the
        # mask (obs-step k disallows context positions > k).
        ctx_g = ctx[:, None, :, :].expand(N, T, T, Dm).reshape(N * T, T, Dm)
        # causal key-padding mask: for obs-step k (the second axis), disallow
        # context position j > k. mask (T_obsstep, T_ctx) -> broadcast to groups.
        obs_idx = torch.arange(T, device=device)
        ctx_idx = torch.arange(T, device=device)
        # True == DISALLOWED (future context). (T_obsstep, T_ctx).
        disallow = ctx_idx[None, :] > obs_idx[:, None]
        # expand to every window in the batch: (N, T_obsstep, T_ctx) -> (N*T, T).
        key_pad = disallow[None, :, :].expand(N, T, T).reshape(N * T, T)
        # queries: (chunk_len, d_model) -> (N*T, chunk_len, d_model).
        q = self.query_emb[None, :, :].expand(N * T, self.chunk_len, Dm)
        q = q.contiguous()
        for blk in self.blocks:
            q = blk(q, ctx_g, key_pad)
        q = self.ln_f(q)  # (N*T, chunk_len, d_model)
        # shared projection -> per_step per query, then assemble flat layout
        # (N*T, chunk_len, per_step) -> (N*T, chunk_len*per_step).
        params = self.shared_proj(q)  # (N*T, chunk_len, per_step)
        params = params.reshape(N, T, self.chunk_len * self.per_step)
        return params

    # ---- rollout (single obs-step; context = buffer h_0..h_k) -------------

    def forward_step(self, ctx_feats: torch.Tensor) -> torch.Tensor:
        """Single rollout obs-step. ``ctx_feats (B, S, d_model)`` are the core's
        per-step features for the CURRENT buffer (S = k+1 positions h_0..h_k).

        Returns flat GMM params ``(B, chunk_len * per_step)`` for this obs-step.
        No mask is needed: the buffer IS the causal prefix (all S positions are
        in the past or present), so every query attends to all S context
        features -- exactly the unmasked sub-block that training row k computes.
        """
        B, S, Dm = ctx_feats.shape
        if S > self.max_window:
            raise ValueError(
                f"buffer length S ({S}) exceeds max_window ({self.max_window})."
            )
        device = ctx_feats.device
        ctx_pos = torch.arange(S, device=device, dtype=torch.long)
        ctx = ctx_feats + self.ctx_pos_emb(ctx_pos)[None, :, :]  # (B, S, d_model)
        q = self.query_emb[None, :, :].expand(B, self.chunk_len, Dm).contiguous()
        for blk in self.blocks:
            q = blk(q, ctx, None)  # no mask: full buffer is the causal prefix
        q = self.ln_f(q)
        params = self.shared_proj(q)  # (B, chunk_len, per_step)
        return params.reshape(B, self.chunk_len * self.per_step)
