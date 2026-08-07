"""Transformer core for BC-RNN (drop-in swap for the LSTM core).

This is the SAME history mechanism as ``LSTMCore`` — it conditions on the
sequence of per-frame OBSERVATION embeddings and emits one per-step feature
vector that the GMM head decodes into an action — but the recurrence is a
CAUSAL self-attention transformer over a length-``rnn_horizon`` window instead
of an LSTM. There is NO past-action input (identical to the LSTM path): the
action is the OUTPUT decoded from each step's feature, never fed back in.

Interface CONTRACT (byte-for-byte the same shape contract as ``LSTMCore`` so
``BCRNNPolicy`` is core-agnostic):

  * ``.input_dim``  : per-frame obs embedding width (ObsEncoder.embed_dim, 66).
  * ``.hidden_dim`` : per-step OUTPUT width the head/actor-MLP reads. We make
                      this ``d_model`` so ``gmm_head.d_model`` / ``actor_mlp``
                      sizing logic in ``BCRNNPolicy`` is unchanged (it reads
                      ``core.hidden_dim``). Input proj 66 -> d_model; output is
                      d_model (no extra output proj — the transformer block
                      output IS d_model, fed straight to the head exactly like
                      the LSTM's hidden_dim output).
  * ``.output_dim`` : == hidden_dim (parity with LSTMCore.output_dim).
  * ``forward(obs_emb) -> (out, hidden)`` : TRAINING path. Unroll over a
        ``(B, T, input_dim)`` window with CAUSAL self-attention + learned
        positional embeddings indexed by WINDOW-RELATIVE position 0..T-1.
        Returns ``out (B, T, hidden_dim)`` — per-step features for the GMM NLL
        over all steps, exactly like the LSTM's per-step hidden states. The
        second return mirrors ``LSTMCore.forward``'s ``hidden`` (here the final
        rolling buffer) so callers that thread state keep working.
  * ``init_hidden(B, device, dtype) -> buffer`` : a FRESH (empty) rolling
        buffer for a new rollout window — the transformer analogue of the
        LSTM's zero ``(h, c)``. Clearing it == LSTM hidden re-zero.
  * ``step(obs_emb_t, buffer) -> (h_t, new_buffer)`` : ROLLOUT path. APPEND the
        new frame embedding to the rolling buffer, run CAUSAL self-attention
        over the whole buffer (positions 0..len-1 of the current window), and
        return the LAST position's feature ``h_t (B, hidden_dim)`` plus the
        grown buffer. ``BCRNNPolicy.step`` clears (re-inits) the buffer every
        ``rnn_horizon`` steps via ``init_hidden`` BEFORE appending — the exact
        same reinit-then-step order as the LSTM path — so the window-relative
        position structure seen at rollout is IDENTICAL to training.

TRAIN == ROLLOUT parity: in both paths step ``k`` of a window attends causally
to steps ``0..k`` of that window with positional embedding ``k``. Feeding the
same 10 obs through ``forward`` (one shot) vs 10 sequential ``step`` calls (with
a fresh buffer at k=0) therefore produces the SAME per-step features (up to
attention-kernel numeric noise) — this is the train/rollout-parity guarantee
the LSTM gets from sharing ``nn.LSTM`` weights.
"""

from typing import Optional

import torch
import torch.nn as nn


class _CausalTransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with CAUSAL self-attention.

    Pre-norm (norm-before-sublayer) is the stable GPT-style arrangement and is
    what robomimic's own transformer (GPT_Backbone) uses. The causal mask is
    supplied by the caller so the same block serves both the windowed training
    unroll and the step-by-step rollout (where the mask is over the current
    buffer length).
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(ff_mult) * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """``x (B, L, d_model)``, ``attn_mask (L, L)`` additive/bool causal mask."""
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TransformerCore(nn.Module):
    """Causal-self-attention core over per-frame obs embeddings.

    Drop-in replacement for ``LSTMCore`` (same interface; see module docstring).

    Args:
        input_dim:   per-frame obs embedding width (== ObsEncoder.embed_dim, 66).
        d_model:     transformer width (== ``hidden_dim`` exposed to the head).
        n_layers:    number of stacked causal-attention blocks.
        n_heads:     attention heads (must divide ``d_model``).
        ff_mult:     feed-forward hidden = ``ff_mult * d_model``.
        dropout:     dropout in attention + FF + residual.
        max_window:  max sequence length for the learned positional embedding
                     table (== rollout/train window length, ``rnn_horizon``).
                     A window position index must always be < max_window.

    Defaults (d_model=448, n_layers=5, n_heads=8, ff_mult=4) are chosen to be
    param-comparable to LSTM(hidden=1000)x2: core params 12.11M vs 12.28M
    (ratio 0.99). Only the CORE differs between builds; the shared obs_encoder
    + gmm_head are identical. See the train script / report for the breakdown.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 448,
        n_layers: int = 5,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_window: int = 10,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_window = int(max_window)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        # input projection 66 -> d_model (the transformer analogue of feeding
        # the 66-d obs embedding straight into the LSTM's input gate).
        self.in_proj = nn.Linear(self.input_dim, self.d_model)
        # learned positional embedding indexed by WINDOW-RELATIVE position.
        self.pos_emb = nn.Embedding(self.max_window, self.d_model)
        self.in_drop = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            [
                _CausalTransformerBlock(
                    self.d_model, self.n_heads, int(ff_mult), float(dropout)
                )
                for _ in range(self.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)

    @property
    def hidden_dim(self) -> int:
        # name parity with LSTMCore: BCRNNPolicy reads ``core.hidden_dim`` to
        # size the actor MLP / GMM head. Output width == d_model (no extra proj).
        return self.d_model

    @property
    def output_dim(self) -> int:
        return self.d_model

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        """Bool ``(L, L)`` mask: True == DISALLOWED (upper triangle, strict).

        Position i may attend to j for j <= i (so step t sees only steps <= t).
        nn.MultiheadAttention treats a True bool entry as "not allowed".
        """
        return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )

    def _encode(self, x_in: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """``x_in (B, L, input_dim)`` -> ``(B, L, d_model)`` causal features.

        ``start_pos`` is the WINDOW-RELATIVE position of the first row of
        ``x_in`` (always 0 here; both train and rollout encode positions
        0..L-1 of the current window, which is what gives train/rollout the
        identical positional structure).
        """
        B, L = x_in.shape[0], x_in.shape[1]
        if start_pos + L > self.max_window:
            raise ValueError(
                f"window position {start_pos + L} exceeds max_window "
                f"({self.max_window}); rnn_horizon must be <= max_window."
            )
        x = self.in_proj(x_in)
        pos_ids = torch.arange(
            start_pos, start_pos + L, device=x_in.device, dtype=torch.long
        )
        x = x + self.pos_emb(pos_ids)[None, :, :]
        x = self.in_drop(x)
        mask = self._causal_mask(L, x_in.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    def forward(
        self, obs_emb: torch.Tensor, hidden: Optional[dict] = None
    ):
        """Unroll over a ``(B, T, input_dim)`` sequence.

        ``hidden`` is accepted for signature parity with ``LSTMCore.forward``;
        each window always starts a fresh window (positions 0..H-1) so it is
        ignored here (matching the LSTM's zero-init-per-window training).
        Returns ``(out (B, T, hidden_dim), buffer)`` where ``buffer`` is the
        final window's cached obs, mirroring the LSTM's returned ``(h, c)``.

        Two regimes, selected purely by ``T`` (no behavior flag — the short
        path is byte-identical to the prior implementation):

          * ``T <= max_window`` (the TRAINING path: ``forward_training`` feeds
            length-``rnn_horizon`` windows, and ``rnn_horizon <= max_window``):
            one fresh ``_encode`` over positions 0..T-1. UNCHANGED.

          * ``T > max_window`` (the EVAL OVERLAY path: ``forward_eval`` feeds the
            FULL episode, T can be 100s of frames): the table only holds
            ``max_window`` positions, so chunk the episode into NON-OVERLAPPING
            length-``max_window`` windows, encode each fresh (start_pos=0), and
            concat the per-window features back to ``(B, T, hidden)``. This is
            NOT an arbitrary fallback: it makes step ``t`` of the overlay attend
            to window ``[floor(t/max_window)*max_window, t]`` with positional id
            ``t % max_window`` — IDENTICAL to how the model is used at rollout,
            where ``BCRNNPolicy.step`` re-inits the buffer every ``rnn_horizon``
            steps. So the teacher-forced overlay matches rollout semantics
            (10-step windows), not a full-length attention the model never sees
            in training. (When ``max_window == rnn_horizon``, as in every config,
            the overlay window boundaries line up exactly with the rollout's.)
        """
        T = obs_emb.shape[1]
        if T <= self.max_window:
            out = self._encode(obs_emb, start_pos=0)
            buffer = {"obs": obs_emb}
            return out, buffer
        # long-sequence (overlay) path: non-overlapping max_window chunks.
        W = self.max_window
        outs = []
        last = obs_emb[:, :W]
        for s in range(0, T, W):
            chunk = obs_emb[:, s : s + W]  # (B, <=W, input_dim)
            outs.append(self._encode(chunk, start_pos=0))
            last = chunk
        out = torch.cat(outs, dim=1)  # (B, T, hidden_dim)
        buffer = {"obs": last}
        return out, buffer

    def init_hidden(self, batch_size: int, device, dtype=None) -> dict:
        """Fresh (empty) rolling buffer for a new rollout window.

        The transformer analogue of the LSTM's zero ``(h, c)``: clearing this is
        the rollout re-init. ``dtype`` kept for signature parity with
        ``LSTMCore.init_hidden``.
        """
        return {"obs": None}

    def step(self, obs_emb_t: torch.Tensor, hidden: dict):
        """One rollout step. ``obs_emb_t (B, input_dim)`` (a single frame).

        APPEND the frame to the rolling buffer, run causal self-attention over
        the whole buffer (window positions 0..len-1), return the LAST position's
        feature ``(B, hidden_dim)`` + the grown buffer. ``BCRNNPolicy.step``
        clears the buffer (``init_hidden``) at the ``rnn_horizon`` boundary
        BEFORE calling this — same reinit-then-step order as the LSTM — so the
        window-relative positions seen here match training exactly.
        """
        x_t = obs_emb_t.unsqueeze(1)  # (B, 1, input_dim)
        prev = hidden.get("obs", None) if hidden is not None else None
        if prev is None:
            buf = x_t
        else:
            buf = torch.cat([prev, x_t], dim=1)  # (B, L, input_dim)
        out = self._encode(buf, start_pos=0)  # (B, L, hidden_dim)
        h_t = out[:, -1]  # last position's feature
        return h_t, {"obs": buf}
