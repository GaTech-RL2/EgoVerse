"""H-Net core for BC-RNN (third drop-in swap for the LSTM / Transformer core).

This is the SAME history mechanism as ``LSTMCore`` / ``TransformerCore`` — it
conditions on the sequence of per-frame OBSERVATION embeddings and emits one
per-step feature vector that the GMM head decodes into an action — but the
recurrence is the user's REAL hierarchical **dynamic-chunking H-Net** (the
``hnet_nets`` stage tree: an outer encoder/decoder Isotropic wrapped around a
``ChunkerStage`` whose ``RoutingModule`` dynamically picks chunk boundaries over
frames, with an inner ``ComputeStage`` running in the compressed chunk space).
There is NO past-action input (identical to LSTM/TX): the action is the OUTPUT
decoded from each step's feature, never fed back in.

Interface CONTRACT (byte-for-byte the same shape contract as ``LSTMCore`` /
``TransformerCore`` so ``BCRNNPolicy`` is core-agnostic):

  * ``.input_dim``  : per-frame obs embedding width (ObsEncoder.embed_dim, 66).
  * ``.hidden_dim`` : per-step OUTPUT width the head/actor-MLP reads. We make
                      this ``d_model`` so ``gmm_head.d_model`` / ``actor_mlp``
                      sizing logic in ``BCRNNPolicy`` is unchanged (it reads
                      ``core.hidden_dim``). Input proj 66 -> d_model; the H-Net
                      stage tree preserves the working dim end to end, so its
                      per-step output IS d_model, fed straight to the head
                      exactly like the LSTM's hidden_dim output.
  * ``.output_dim`` : == hidden_dim (parity with LSTMCore.output_dim).
  * ``forward(obs_emb) -> (out, state)`` : TRAINING path. Run the H-Net stage
        tree (PADDED mode, all-ones mask) over a ``(B, T, input_dim)`` window
        and return ``out (B, T, hidden_dim)`` — per-step features for the GMM
        NLL over all steps, exactly like the LSTM's per-step hidden states. The
        second return mirrors ``LSTMCore.forward``'s ``hidden`` (here the final
        rolling buffer) so callers that thread state keep working.
  * ``init_hidden(B, device, dtype) -> buffer`` : a FRESH (empty) rolling
        buffer for a new rollout window — the analogue of the LSTM's zero
        ``(h, c)``. Clearing it == LSTM hidden re-zero.
  * ``step(obs_emb_t, buffer) -> (h_t, new_buffer)`` : ROLLOUT path. APPEND the
        new frame embedding to the rolling buffer, run the H-Net forward over
        the WHOLE buffer (positions 0..len-1 of the current window), and return
        the LAST position's feature ``h_t (B, hidden_dim)`` plus the grown
        buffer. ``BCRNNPolicy.step`` clears (re-inits) the buffer every
        ``rnn_horizon`` steps via ``init_hidden`` BEFORE appending — the exact
        same reinit-then-step order as the LSTM / TX path — so the
        window-relative structure seen at rollout is IDENTICAL to training.

WHY THE BUFFER-REPLAY ``step`` (and not the H-Net's own KV-cached ``step``):
the H-Net stage tree HAS a single-token ``step`` path (KV caches + routing
``last_hidden_state`` + dechunk EMA carry), but proving it bit-matches the
windowed ``forward`` is hard (the dechunk EMA reset, the routing prefill, and
the chunk-boundary state all have to line up). The TX core already established
the robust pattern: at window 10 the per-step cost of re-running ``forward``
over the buffer is trivial, and because the EXACT SAME ``forward`` code runs in
both training and rollout, train == rollout is guaranteed by construction (up to
attention/EMA numeric noise) — see the prefix-consistency proof below.

EMA / PRECISION BACKEND (re: the "bf16 EMA/attn untested" open concern): on the
EgoVerse7 launch venv ``routing.has_mamba_scan() == False`` and
``blocks.has_flash_attn() == False`` (neither mamba_ssm nor flash_attn is
installed), and training is fp32 (``trainer.precision=32`` in the sbatch). So the
DeChunkLayer EMA runs the pure-PyTorch ``_ema_loop`` IN THE INPUT DTYPE (fp32) and
the Isotropic attention runs plain SDPA in fp32 — the bf16 mamba-kernel /
flash-attn paths (the only places a hardcoded bf16 cast lives) are NEVER hit on
this run. The fixer verified both backend flags are False on the A40 venv. ⚠️ If
mamba_ssm or flash_attn is EVER installed in the launch venv, those kernels
auto-activate (the mamba EMA hard-casts to bf16 even under fp32 — see
``routing.DeChunkLayer._ema_kernel``); re-run the causality / train-vs-step parity
proofs before trusting a run in that case.

CAUSALITY (the make-or-break — output at k must depend only on positions <= k,
INCLUDING through the dynamic chunk-boundary decisions):

  * Inner Isotropic attention (encoder / decoder / inner main_network): built
    with ``causal=True`` so SDPA masks each position to attend only to <= k.
  * ``RoutingModule`` boundary decision (the dynamic chunk boundary): the
    padded path computes ``cos_sim(q(h[:-1]), k(h[1:]))`` then ``F.pad((1,0),
    1.0)`` — so ``boundary_prob[k]`` depends ONLY on ``h_{k-1}`` and ``h_k``
    (both <= k); position 0 is a forced boundary. No future frame enters a
    boundary choice.
  * ``ChunkLayer`` keeps original order (stable argsort), so the j-th chunk
    aggregates only boundaries at positions <= its own.
  * ``DeChunkLayer`` EMA is a LEFT-TO-RIGHT recurrence along the chunked axis
    (``prev = p*x + (1-p)*prev``) and the plug-back index
    (``cumsum(boundary_mask) - 1``) maps each frame k to the most recent chunk
    at a position <= k. So the dechunked feature at k uses only chunks <= k.
  * The STE residual (``residual_proj(x)``) is per-position (causal).

Net: with ``causal=True`` on the Isotropic stacks the WHOLE chunker is causal,
and forward-over-the-full-window == forward-over-each-prefix at the matching
position (the prefix-consistency test the algo runs; maxdiff ~1e-6 like TX).

Defaults (d_model=256, n_heads=8, d_intermediate=768, outer_layers=4,
inner_layers=6, ratio=2.0) are chosen param-comparable to LSTM(1000)x2 /
TX(448,5L): see the train script / report for the exact count. Only the CORE
differs between builds; the shared obs_encoder + gmm_head are identical.
"""

from typing import Optional

import torch
import torch.nn as nn

from egomimic.models.bc_rnn_nets._hnet_vendored.context import HNetContext
from egomimic.models.bc_rnn_nets._hnet_vendored.hnet import HNet
from egomimic.models.bc_rnn_nets._hnet_vendored.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)


def _t_layout(n_layers: int) -> str:
    """arch_layout token for ``n_layers`` ``T`` (attn+SwiGLU-MLP) blocks."""
    return f"T{int(n_layers)}"


class HNetCore(nn.Module):
    """Real dynamic-chunking H-Net core over per-frame obs embeddings.

    Drop-in replacement for ``LSTMCore`` / ``TransformerCore`` (same interface;
    see module docstring). Internally:

        obs_emb (B,T,66) --in_proj--> (B,T,d_model)
            --EncoderDecoderStage(outer, causal)
                --ChunkerStage(dynamic chunking over frames, causal)
                    --ComputeStage(inner main_network, causal)
            --> (B,T,d_model) == per-step features for the GMM head.

    All sizes are config knobs so variants are a config change, not a code edit.

    Args:
        input_dim:   per-frame obs embedding width (== ObsEncoder.embed_dim, 66).
        d_model:     H-Net working/hidden dim (== ``hidden_dim`` exposed to the
                     head; the stage tree preserves it end to end so no output
                     proj is needed).
        n_heads:     attention heads in every Isotropic block (must divide
                     d_model AND d_model // n_heads must be even when RoPE is on;
                     here RoPE is off by default).
        d_intermediate: SwiGLU FFN hidden width inside each ``T`` block.
        outer_layers:   ``T`` blocks in EACH of the outer encoder and decoder.
        inner_layers:   ``T`` blocks in the inner ComputeStage main_network
                        (runs in the compressed chunk space).
        target_compression_ratio: the chunker's target avg chunk length (the
                        ratio-loss target N); pure history-mechanism knob, does
                        not change the per-step output shape.
        ratio_loss_weight: weight of the chunker's auxiliary ratio loss. NOTE:
                        the BC-RNN algo only reads ``core.forward``'s per-step
                        FEATURES and never collects the H-Net ``ctx.aux`` ratio
                        loss, so this term is INERT here (kept as a knob for
                        parity / future use). The chunker still registers aux
                        into a throwaway ctx each call.
        max_window:  max sequence length the core is asked to encode in one
                     window (== rollout/train window length, ``rnn_horizon``).
                     A guard, mirroring TransformerCore.max_window.
        dropout:     attention-softmax dropout inside the Isotropic blocks.
        resid_dropout: residual-branch dropout inside the Isotropic blocks.
        rotary_emb_dim: RoPE dim for the Isotropic self-attn (0 = off, default).
        causal:      MUST stay True for train==rollout parity (see docstring).
                     Exposed as a knob only so an ablation can flip it; the
                     prefix-consistency proof relies on True.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_intermediate: int = 768,
        outer_layers: int = 4,
        inner_layers: int = 6,
        target_compression_ratio: float = 2.0,
        ratio_loss_weight: float = 0.03,
        max_window: int = 10,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        rotary_emb_dim: int = 0,
        causal: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_intermediate = int(d_intermediate)
        self.outer_layers = int(outer_layers)
        self.inner_layers = int(inner_layers)
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.max_window = int(max_window)
        self.causal = bool(causal)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        # DeChunkLayer requires d_model % headdim == 0 (headdim default 32). Fail
        # at construction (before a wasted launch) with a clear message instead
        # of deep inside the EMA kernel.
        if self.d_model % 32 != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by 32 (the "
                f"DeChunkLayer EMA headdim) so the chunker's dechunk kernel is "
                f"valid; pick a multiple of 32."
            )

        # input projection 66 -> d_model (the H-Net analogue of feeding the
        # 66-d obs embedding straight into the LSTM's input gate).
        self.in_proj = nn.Linear(self.input_dim, self.d_model)

        # --- build the real H-Net stage tree (flat list; HNet wires the chain).
        # Outer encoder/decoder Isotropic (causal) wrapped around a ChunkerStage
        # whose inner ComputeStage runs in the compressed chunk space. Every
        # working dim == d_model so the chunker needs no proj_in/proj_out and the
        # end-to-end output dim == d_model (== hidden_dim).
        iso = lambda n: {  # noqa: E731 — tiny local spec builder
            "arch_layout": _t_layout(n),
            "d_model": self.d_model,
            "d_intermediate": self.d_intermediate,
            "num_heads": self.n_heads,
            "cond": False,  # no AdaLN cond in the BC-RNN core (obs-only history)
            "rotary_emb_dim": int(rotary_emb_dim),
            "dropout": float(dropout),
            "resid_dropout": float(resid_dropout),
        }
        outer = EncoderDecoderStage(
            input_hidden_dim=self.d_model,
            output_hidden_dim=self.d_model,
            encoder=iso(self.outer_layers),
            decoder=iso(self.outer_layers),
            cond_key=None,  # no conditioning fetched from ctx
            d_cond=0,
            causal=self.causal,
        )
        chunker = ChunkerStage(
            input_hidden_dim=self.d_model,
            output_hidden_dim=self.d_model,  # inner runs at the same dim
            target_compression_ratio=self.target_compression_ratio,
            ratio_loss_weight=self.ratio_loss_weight,
            cond_key=None,
        )
        inner = ComputeStage(
            input_hidden_dim=self.d_model,
            output_hidden_dim=self.d_model,
            main_network=iso(self.inner_layers),
            cond_key=None,
            d_cond=0,
            causal=self.causal,
        )
        self.hnet = HNet([outer, chunker, inner])
        # NOTE (param registration): HNet wires the flat stage list into a chain
        # via ``object.__setattr__(stages[i], "inner_stage", stages[i+1])`` (see
        # hnet.py) so the inner ChunkerStage/ComputeStage params are registered
        # ONCE through the ModuleList, not double-counted through the attribute
        # chain. The fixer verified all 121 named_parameters are present exactly
        # once (0 duplicate tensors, de-duped count == 12.1638M) and every param
        # gets a gradient. Do NOT "simplify" that setattr to a plain assignment —
        # it would re-register the inner stages and could silently drop them from
        # the optimizer while forward still works via the attribute chain.
        # Residual-stream-aware init exactly like the standalone H-Net (scales
        # out_proj / fc2 by initializer_range / sqrt(n_residuals); leaves the
        # _no_reinit routing-q/k identity + zero residual_proj alone).
        self.hnet.init_weights()

    @property
    def hidden_dim(self) -> int:
        # name parity with LSTMCore: BCRNNPolicy reads ``core.hidden_dim`` to
        # size the actor MLP / GMM head. Output width == d_model (stage tree
        # preserves the working dim; no output proj).
        return self.d_model

    @property
    def output_dim(self) -> int:
        return self.d_model

    def _new_ctx(self) -> HNetContext:
        """A fresh PADDED-mode context (cu_seqlens=None -> stages use the
        all-ones-mask padded paths). ``aux`` is a throwaway sink for the
        chunker's ratio-loss registration (the BC-RNN algo doesn't read it)."""
        return HNetContext()

    def _encode(self, x_in: torch.Tensor) -> torch.Tensor:
        """``x_in (B, L, input_dim)`` -> ``(B, L, d_model)`` causal features.

        Projects to d_model and runs the H-Net stage tree in padded mode. Both
        train and rollout encode positions 0..L-1 of the current window with the
        SAME forward, which is what gives train/rollout the identical structure.
        """
        L = x_in.shape[1]
        if L > self.max_window:
            raise ValueError(
                f"window length {L} exceeds max_window ({self.max_window}); "
                f"rnn_horizon must be <= max_window."
            )
        x = self.in_proj(x_in)
        ctx = self._new_ctx()
        return self.hnet(x, ctx)  # (B, L, d_model)

    def forward(self, obs_emb: torch.Tensor, hidden: Optional[dict] = None):
        """Unroll the H-Net over a ``(B, T, input_dim)`` sequence.

        ``hidden`` is accepted for signature parity with ``LSTMCore.forward``;
        each window always starts a fresh window so it is ignored here (matching
        the LSTM's zero-init-per-window training). Returns ``(out (B, T,
        hidden_dim), buffer)`` where ``buffer`` caches the window's obs,
        mirroring the LSTM's returned ``(h, c)``.

        Two regimes, selected purely by ``T`` (same structure as TransformerCore
        so the eval-overlay semantics match rollout):

          * ``T <= max_window`` (TRAINING path: ``forward_training`` feeds
            length-``rnn_horizon`` windows): one fresh ``_encode`` over 0..T-1.

          * ``T > max_window`` (EVAL OVERLAY path: ``forward_eval`` feeds the
            FULL episode): chunk into NON-OVERLAPPING length-``max_window``
            windows, encode each fresh, concat back to ``(B, T, hidden)`` — so
            the overlay matches ROLLOUT semantics (the rollout re-inits the
            buffer every ``rnn_horizon`` steps), not a full-length encode the
            model never sees in training.
        """
        T = obs_emb.shape[1]
        if T <= self.max_window:
            out = self._encode(obs_emb)
            return out, {"obs": obs_emb}
        # long-sequence (overlay) path: non-overlapping max_window chunks. This is
        # the EVAL-OVERLAY path ONLY (forward_eval feeds the FULL episode, T can be
        # hundreds of frames) — the silent chunking here is intended, NOT a missing
        # guard. The TRAINING-window length constraint (window <= max_window) is
        # enforced upstream in BCRNN.__init__ (max_window >= rnn_horizon) and the
        # rollout step() path is guarded by _encode's L>max_window raise.
        W = self.max_window
        outs = []
        last = obs_emb[:, :W]
        for s in range(0, T, W):
            chunk = obs_emb[:, s : s + W]  # (B, <=W, input_dim)
            outs.append(self._encode(chunk))
            last = chunk
        out = torch.cat(outs, dim=1)  # (B, T, hidden_dim)
        return out, {"obs": last}

    def init_hidden(self, batch_size: int, device, dtype=None) -> dict:
        """Fresh (empty) rolling buffer for a new rollout window.

        The analogue of the LSTM's zero ``(h, c)``: clearing this is the rollout
        re-init. ``dtype`` kept for signature parity with ``LSTMCore.init_hidden``.
        """
        return {"obs": None}

    def step(self, obs_emb_t: torch.Tensor, hidden: dict):
        """One rollout step. ``obs_emb_t (B, input_dim)`` (a single frame).

        APPEND the frame to the rolling buffer, run the H-Net forward over the
        whole buffer (window positions 0..len-1), return the LAST position's
        feature ``(B, hidden_dim)`` + the grown buffer. ``BCRNNPolicy.step``
        clears the buffer (``init_hidden``) at the ``rnn_horizon`` boundary
        BEFORE calling this — same reinit-then-step order as LSTM / TX — so the
        window-relative positions seen here match training exactly. Because the
        SAME ``forward`` (``_encode``) runs over the buffer, train == rollout up
        to attention/EMA numeric noise (prefix-consistency proof).
        """
        x_t = obs_emb_t.unsqueeze(1)  # (B, 1, input_dim)
        prev = hidden.get("obs", None) if hidden is not None else None
        if prev is None:
            buf = x_t
        else:
            buf = torch.cat([prev, x_t], dim=1)  # (B, L, input_dim)
        out = self._encode(buf)  # (B, L, hidden_dim)
        h_t = out[:, -1]  # last position's feature
        return h_t, {"obs": buf}
