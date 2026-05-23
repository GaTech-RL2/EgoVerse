"""
H-Net policy for EgoVerse — stage-based architecture.

The policy treats the action sequence as the input modality (autoregressive,
causal). Observations are encoded by a ``CondEncoderModule`` into a
``cond_dict`` carried by an ``HNetContext`` that is threaded through the
stage tree. Each stage reads whichever cond key it wants (or ignores cond
entirely).

Loss = action MSE (next-action prediction) +
       sum_over_chunkers( weight * ratio_loss(boundary_predictions) ).

The per-chunker ratio-loss weights live inside the chunker stages themselves;
this algo just calls ``ratio_loss_from_aux(ctx.aux)`` after forward.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.algo.hnet_outer_stage import HNetOuterStage
from egomimic.algo.input_modules import ActionInToken, InputModule
from egomimic.algo.loss import HNetLoss, Loss
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.hnet import HNet as HNetCore
from egomimic.models.hnet_nets.hnet import chunk_stats_from_aux
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class HNetPolicy(nn.Module):
    """
    action-tokenizer → stage-based H-Net → action-detokenizer.

    Owns action_in / action_out projections, BOS token, positional embedding,
    the ``CondEncoderModule``, and the ``HNetCore`` (stage tree).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        cond_encoder: CondEncoderModule,
        hnet: HNetCore,
        action_head_type: str = "linear",
        input_modules: list | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        # Composable input modules. Each emits a per-token (B,T,d_model)
        # contribution that we sum before feeding the H-Net trunk.
        # Default = legacy AR behaviour (single ActionInToken with BOS).
        if input_modules is None:
            input_modules = [ActionInToken(action_dim, d_model)]
        for mod in input_modules:
            if not isinstance(mod, InputModule):
                raise TypeError(
                    f"input_modules entries must subclass InputModule, got "
                    f"{type(mod).__name__}"
                )
        self.input_modules = nn.ModuleList(input_modules)
        if action_head_type == "linear":
            self.action_out = nn.Linear(d_model, action_dim)
        elif action_head_type == "mlp":
            # 2-layer MLP head: Linear -> SiLU -> Linear, hidden = d_model.
            # The final Linear is what the upstream init recipe scales as
            # ``out_proj`` (1/sqrt(n_residuals)).
            self.action_out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, action_dim),
            )
        else:
            raise ValueError(
                f"action_head_type must be 'linear' or 'mlp', got {action_head_type!r}"
            )
        self.action_head_type = action_head_type
        # BOS / action_in moved into ActionInToken (an InputModule).
        # See egomimic/algo/input_modules.py.
        # F6: positional info is carried by RoPE inside MHA.

        # Legacy-ckpt key remap: pre-refactor checkpoints stored
        # ``<prefix>action_in.{weight,bias}`` and ``<prefix>bos`` directly
        # on HNetPolicy. If the first input module is an ActionInToken,
        # rewrite those keys to ``<prefix>input_modules.0.action_in.*`` /
        # ``<prefix>input_modules.0.bos`` on load so legacy ckpts keep
        # working without manual surgery.
        self._register_load_state_dict_pre_hook(self._remap_legacy_input_keys)

        self.cond_encoder = cond_encoder
        self.hnet = hnet

        # Sanity-check that the stage tree's outer hidden dim matches d_model.
        if self.hnet.input_hidden_dim != d_model:
            raise ValueError(
                f"hnet.input_hidden_dim ({self.hnet.input_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )
        if self.hnet.output_hidden_dim != d_model:
            raise ValueError(
                f"hnet.output_hidden_dim ({self.hnet.output_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )

    def _remap_legacy_input_keys(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Remap pre-refactor ckpt keys (``<prefix>action_in.*``, ``<prefix>bos``)
        to the new ``input_modules.0.*`` paths when the first input module
        is an ActionInToken. No-op otherwise.
        """
        if not self.input_modules:
            return
        first = self.input_modules[0]
        if not isinstance(first, ActionInToken):
            return
        # Look for the legacy keys and move them.
        legacy_to_new = {
            prefix + "action_in.weight": prefix + "input_modules.0.action_in.weight",
            prefix + "action_in.bias": prefix + "input_modules.0.action_in.bias",
            prefix + "bos": prefix + "input_modules.0.bos",
        }
        for old_k, new_k in legacy_to_new.items():
            if old_k in state_dict and new_k not in state_dict:
                state_dict[new_k] = state_dict.pop(old_k)

    def _build_ctx(self, obs: dict) -> HNetContext:
        cond_dict = self.cond_encoder.encode(obs, self.action_horizon)
        return HNetContext(cond_dict=cond_dict, aux=[], inference_params=None)

    def forward(self, actions: torch.Tensor, obs: dict):
        """
        actions: (B, T, action_dim) ground-truth actions for teacher-forcing.
        obs:     dict of (B, ...) obs tensors.

        Returns: (pred_actions (B, T, action_dim), aux list).
        """
        B, T, _ = actions.shape
        device = actions.device
        # Materialise dtype from action_in if present, otherwise from obs.
        dtype = actions.dtype
        x = None
        for mod in self.input_modules:
            contrib = mod.forward_padded(
                actions=actions,
                obs=obs,
                B=B,
                T=T,
                device=device,
                dtype=dtype,
            )
            x = contrib if x is None else x + contrib
        if x is None:
            raise RuntimeError("input_modules produced no tokens")
        # F6: no outer pos_emb — RoPE inside MHA handles positions.
        ctx = self._build_ctx(obs)
        h = self.hnet(x, ctx)
        return self.action_out(h), ctx.aux

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Mirrors :meth:`forward` for variable-length episodes packed into a
        single flat stream (FlashAttention-style varlen). The BOS shift and
        ``pos_emb`` indexing happen per sub-sequence — for each subseq
        ``[s, e)``, position s gets BOS and positions s+1..e-1 get
        ``action_in(actions[s..e-2])``; ``pos_emb`` is indexed by ``t - s``
        within each subseq so every episode starts at position 0.

        Args:
            actions_packed: (T_total, action_dim) packed ground-truth actions.
            obs_packed:     dict of (T_total, ...) per-frame obs tensors.
            cu_seqlens:     (B+1,) long, cumulative subseq lengths (starts 0).
            max_seqlen:     int, longest subseq length.

        Returns: (pred_packed (T_total, action_dim), aux).
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        if not torch.is_tensor(cu_seqlens):
            cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.long)
        else:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        # F6: with RoPE handling positions inside each MHA, there is no
        # hard ``action_horizon`` ceiling anymore — the old check on
        # ``max_seqlen > self.action_horizon`` was tied to the size of
        # ``self.pos_emb`` and is removed.

        # 1. Sum per-token contributions from each input module. Each
        # module is responsible for its own BOS/shift conventions (e.g.
        # ActionInToken right-shifts; ObsToken does not).
        dtype = actions_packed.dtype
        x_packed = None
        for mod in self.input_modules:
            contrib = mod.forward_packed(
                actions_packed=actions_packed,
                obs_packed=obs_packed,
                cu_seqlens=cu_seqlens,
                T_total=T_total,
                device=device,
                dtype=dtype,
            )
            x_packed = contrib if x_packed is None else x_packed + contrib
        if x_packed is None:
            raise RuntimeError("input_modules produced no tokens")
        # F6: no outer per-subseq pos_emb add — RoPE applies per-subseq
        # rotary positions inside attention via cu_seqlens threading.

        # 3. Packed cond. ``cond_encoder.encode`` expects (B, T, ...); for a
        #    packed stream the simplest path is to feed (1, T_total, ...) and
        #    squeeze the leading dim back out. The encoder's per-frame branch
        #    (state x.dim()==3 / images x.dim()==5) fires correctly because
        #    obs_packed already carries the per-frame dim.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(obs_for_encode, T_action=T_total)
        cond_packed = {k: v.squeeze(0) for k, v in cond_padded.items()}

        # 4. Build packed ctx and run.
        ctx = HNetContext(
            cond_dict=cond_packed,
            aux=[],
            inference_params=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
        )
        h = self.hnet(x_packed, ctx)  # (T_total, D)
        return self.action_out(h), ctx.aux

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout from BOS for ``T`` steps (default
        ``action_horizon``). ``T`` may be < ``action_horizon`` when rolling
        out an individual episode whose length is known to be shorter — the
        pos_emb is sized at action_horizon so any T <= action_horizon
        works."""
        if T is None:
            T = self.action_horizon
        # F6: no upper bound on T from a learned pos_emb anymore — RoPE
        # handles arbitrary positions inside the MHA. The previous check
        # `T > self.action_horizon` is removed. ``action_horizon`` is still
        # consulted as a *default rollout length* when ``T`` is None.
        cond_dict = self.cond_encoder.encode(obs, T)
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        dtype = next(self.parameters()).dtype

        inference_params = self.hnet.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T,
            device=device,
            dtype=dtype,
        )

        # Per-step cond_dict slice (B, d_cond) — AdaLN broadcasts over the
        # single-token sequence dim inside the encoder.
        def slice_cond(t: int) -> dict:
            return {k: v[:, t] if v.dim() == 3 else v for k, v in cond_dict.items()}

        # F6: no outer pos_emb add; RoPE inside MHA reads positions from each
        # KVCache's ``offsets`` (set by ``MultiHeadAttention.step``).
        prev_a = None

        def _step_obs(t: int) -> dict:
            # Pull single-frame obs at step t. If a value is (B,T,...) we
            # slice; (B,...) single-frame obs is broadcast inside encoders.
            out = {}
            for k, v in obs.items():
                if torch.is_tensor(v) and v.dim() >= 3 and v.shape[1] == T:
                    out[k] = v[:, t : t + 1]
                else:
                    out[k] = v.unsqueeze(1) if torch.is_tensor(v) else v
            return out

        for t in range(T):
            cur = None
            step_obs = _step_obs(t)
            for mod in self.input_modules:
                contrib = mod.step(
                    prev_action_norm=prev_a,
                    obs_norm=step_obs,
                    t=t,
                    B=batch_size,
                    device=device,
                    dtype=dtype,
                )
                cur = contrib if cur is None else cur + contrib
            ctx = HNetContext(
                cond_dict=slice_cond(t),
                aux=[],
                inference_params=inference_params,
            )
            h = self.hnet.step(cur, ctx)
            a_t = self.action_out(h)
            actions[:, t : t + 1] = a_t
            prev_a = a_t
        return actions

    # ----- Single-step inference API for closed-loop rollout -----

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state for ``batch_size`` parallel
        rollouts of at most ``T_max`` steps. The state is opaque to the
        caller; pass it back to :meth:`step` along with the current obs.
        """
        # F6: T_max is no longer clamped by self.action_horizon — RoPE
        # handles arbitrary positions. We still need ``T_max`` to size the
        # KV cache allocations.
        T_max = int(T_max)
        dtype = dtype or next(self.parameters()).dtype
        params = self.hnet.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T_max,
            device=device,
            dtype=dtype,
        )
        # Per-step token is now assembled by input_modules.step() each call;
        # state only carries the previous prediction (for ActionInToken) and
        # the dtype/T_max contract.
        return {
            "params": params,
            "prev_action": None,
            "batch_size": int(batch_size),
            "device": device,
            "dtype": dtype,
            "T_max": T_max,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Single AR step. Returns normalized action ``(B, 1, action_dim)``.

        ``obs_norm`` is a dict of single-frame, normalized obs tensors
        (``state_agent_obj`` ``(B, D)``, ``front_img_1`` ``(B, C, H, W)``,
        …). Mutates ``state`` in place: replaces ``cur`` with the next
        step's input token.
        """
        cond_dict_seq = self.cond_encoder.encode(obs_norm, T_action=1)
        # Block step paths accept the (B, d_cond) view (AdaLN) or auto-
        # unsqueeze the (B, 1, d_cond) view (cross-attn).
        cond_2d = {k: v.squeeze(1) for k, v in cond_dict_seq.items()}

        # Per-step obs dict in (B, 1, ...) form for input_modules.
        obs_step = {
            k: v.unsqueeze(1)
            if (
                torch.is_tensor(v)
                and v.dim() < 5
                and v.shape[0] == state["batch_size"]
                and (v.dim() == 1 or v.shape[1] != 1)
            )
            else v
            for k, v in obs_norm.items()
        }
        # Assemble the per-step input token by summing module contributions.
        cur = None
        for mod in self.input_modules:
            contrib = mod.step(
                prev_action_norm=state.get("prev_action"),
                obs_norm=obs_step,
                t=t,
                B=state["batch_size"],
                device=state["device"],
                dtype=state["dtype"],
            )
            cur = contrib if cur is None else cur + contrib
        if cur is None:
            raise RuntimeError("input_modules produced no tokens at step")

        ctx = HNetContext(
            cond_dict=cond_2d,
            aux=[],
            inference_params=state["params"],
        )
        h = self.hnet.step(cur, ctx)
        a_t_norm = self.action_out(h)

        # F6: no per-step pos_emb add. Cache the prediction so the next
        # step's ActionInToken (if any) can read it.
        state["prev_action"] = a_t_norm
        return a_t_norm


class FlatFusedPolicy(nn.Module):
    """Flat transformer with interleaved [c_t, a_t] tokens.

    Drop-in replacement for ``HNetPolicy`` that bypasses the H-Net stage
    hierarchy (no chunker, no ratio loss). Each timestep contributes TWO
    tokens to the input sequence: a cond token and a (shifted) action token.
    Causal masking means the model predicting ``a_t`` (at sequence position
    2t+1) has seen ``c_0, BOS, c_1, a_0, c_2, a_1, ..., c_t, a_{t-1}``.

    Input layout (length 2T):
      x[:, 0]  = cond_in(c_0)         x[:, 1]  = BOS
      x[:, 2]  = cond_in(c_1)         x[:, 3]  = action_in(a_0)
      ...
      x[:, 2t]   = cond_in(c_t)       x[:, 2t+1] = action_in(a_{t-1})

    Output extraction:
      pred[:, t] = action_out(out[:, 2t+1])

    AR rollout walks the sequence one token at a time (2T model steps for T
    actions). Each "outer" step emits one action prediction and adds two
    tokens (the new cond + the new predicted action) to the cache.

    Same ``forward(actions, obs)`` / ``forward_packed(...)`` / ``generate(...)``
    contracts as ``HNetPolicy`` so the same ``HNet`` algo wrapper consumes
    it. Aux is always ``[]`` (no chunker contributions).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        from egomimic.models.hnet_nets.isotropic_builder import build_isotropic

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.d_cond = d_cond
        self.cond_encoder = cond_encoder

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.cond_in = nn.Linear(d_cond, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        # pos_emb covers the 2T-token interleaved sequence.
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Single Isotropic stack, causal, no per-block cond (the fusion happens
        # at the input layer).
        self.backbone = build_isotropic(
            {
                "arch_layout": arch_layout,
                "d_model": d_model,
                "d_intermediate": d_intermediate,
                "num_heads": num_heads,
                "cond": False,
                "dropout": dropout,
                "resid_dropout": resid_dropout,
            },
            d_cond=0,
            causal=True,
        )

    def _encode_cond(self, obs: dict, T: int) -> torch.Tensor:
        cond_dict = self.cond_encoder.encode(obs, T)
        c = cond_dict.get("fused_cond")
        if c is None:
            raise KeyError(
                "FlatFusedPolicy requires 'fused_cond' in cond_encoder output."
            )
        return c  # (B, T, d_cond)

    def forward(self, actions: torch.Tensor, obs: dict):
        """Padded teacher-forced forward.

        actions: (B, T, action_dim)
        obs:     dict of per-frame (B, T, ...) obs tensors.
        Returns: (pred (B, T, action_dim), aux=[]).
        """
        B, T, _ = actions.shape
        c = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(c)  # (B, T, d_model)
        a_tok = self.action_in(actions)  # (B, T, d_model)
        # Shift: BOS at position 0, a_0..a_{T-2} after.
        a_shifted = torch.cat([self.bos.expand(B, -1, -1), a_tok[:, :-1]], dim=1)

        # Interleave: x[:, 0::2] = c_tok, x[:, 1::2] = a_shifted.
        x = torch.empty(
            B, 2 * T, self.d_model, device=actions.device, dtype=a_tok.dtype
        )
        x[:, 0::2] = c_tok
        x[:, 1::2] = a_shifted
        x = x + self.pos_emb[:, : 2 * T].to(x.dtype)

        x = self.backbone(x)  # (B, 2T, d_model)
        pred = self.action_out(x[:, 1::2])  # (B, T, action_dim)
        return pred, []

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward.

        Builds a fused-token packed stream where each sub-sequence ``[s, e)``
        becomes a length-``2*(e-s)`` interleaved chunk. Returns predictions
        in packed format ``(T_total, action_dim)`` matching the input
        ``actions_packed`` layout.
        """
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        if max_seqlen > self.action_horizon:
            raise ValueError(
                f"max_seqlen={max_seqlen} exceeds action_horizon={self.action_horizon}"
            )

        # Encode cond. Feed obs as (1, T_total, ...) via cond_encoder.encode.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_seq = self._encode_cond(obs_for_encode, T_total).squeeze(
            0
        )  # (T_total, d_cond)
        c_tok = self.cond_in(cond_seq)  # (T_total, d_model)
        a_tok = self.action_in(actions_packed)  # (T_total, d_model)

        # Build BOS-shifted actions per sub-sequence.
        a_shifted = torch.empty_like(a_tok)
        bos = self.bos.squeeze(0).squeeze(0).to(a_tok.dtype)  # (d_model,)
        a_shifted[cu_seqlens[:-1]] = bos
        # For positions that aren't sub-seq starts, copy a_tok[t-1].
        non_start = torch.ones(T_total, dtype=torch.bool, device=device)
        non_start[cu_seqlens[:-1]] = False
        # Indices of non-start positions:
        idx_non_start = torch.nonzero(non_start, as_tuple=False).squeeze(-1)
        a_shifted[idx_non_start] = a_tok[idx_non_start - 1]

        # Build per-sub-seq position indices: 0, 1, ..., (e-s)-1 within each
        # sub-seq. Then the 2-token interleave doubles to 2*(e-s).
        pos = torch.arange(T_total, device=device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        local_pos = pos - cu_seqlens[seq_idx]  # (T_total,)

        # The interleaved stream has 2T_total tokens. cond_t at 2*local_pos
        # within each sub-seq; action_t at 2*local_pos+1.
        # Compute new cu_seqlens for the doubled stream.
        sub_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        new_lens = 2 * sub_lens
        new_cu = torch.zeros(len(cu_seqlens), dtype=torch.long, device=device)
        new_cu[1:] = torch.cumsum(new_lens, dim=0)
        new_T_total = int(new_cu[-1].item())

        # Build the packed interleaved stream and apply pos_emb based on
        # 2*local_pos / 2*local_pos+1.
        x = torch.empty(new_T_total, self.d_model, device=device, dtype=a_tok.dtype)
        # For each t, write c_tok[t] at new_cu[seq_idx[t]] + 2*local_pos[t]
        # and a_shifted[t] at the next position.
        target_c = new_cu[seq_idx] + 2 * local_pos
        target_a = target_c + 1
        x[target_c] = c_tok
        x[target_a] = a_shifted
        # pos_emb indexing: doubled local positions within each sub-seq.
        # pos_emb is (1, 2*action_horizon, d_model). Apply pos_emb[2*local_pos]
        # to cond positions and pos_emb[2*local_pos+1] to action positions.
        pos_c = (2 * local_pos).clamp(max=2 * self.action_horizon - 1)
        pos_a = (2 * local_pos + 1).clamp(max=2 * self.action_horizon - 1)
        x[target_c] = x[target_c] + self.pos_emb[0, pos_c].to(x.dtype)
        x[target_a] = x[target_a] + self.pos_emb[0, pos_a].to(x.dtype)

        # Run the backbone on the doubled packed stream.
        out = self.backbone(
            x,
            cu_seqlens=new_cu,
            max_seqlen=2 * int(max_seqlen),
        )

        # Predictions: out at action positions (target_a). action_out projects
        # to (T_total, action_dim). Order matches actions_packed.
        pred = self.action_out(out[target_a])
        return pred, []

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """AR rollout for T action steps over a 2T-token interleaved stream.

        Each AR outer step does TWO inner step() calls: one for the cond
        token, one for the (predicted) action token. The output of the
        action-step is the next action prediction.
        """
        if T is None:
            T = self.action_horizon
        if T > self.action_horizon:
            raise ValueError(f"generate T={T} exceeds action_horizon")

        cond_seq = self._encode_cond(obs, T)  # (B, T, d_cond)
        c_tok = self.cond_in(cond_seq)  # (B, T, d_model)
        dtype = c_tok.dtype

        # Allocate the backbone's inference cache sized for the doubled stream.
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T,
            device=device,
            dtype=dtype,
        )

        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        a_prev = self.bos.expand(batch_size, -1, -1).to(dtype)  # (B, 1, d_model)
        for t in range(T):
            # Cond step.
            x_c = c_tok[:, t : t + 1] + self.pos_emb[:, 2 * t : 2 * t + 1].to(dtype)
            _ = self.backbone.step(x_c, params)
            # Action step.
            x_a = a_prev + self.pos_emb[:, 2 * t + 1 : 2 * t + 2].to(dtype)
            h = self.backbone.step(x_a, params)
            a_t = self.action_out(h)  # (B, 1, action_dim)
            actions[:, t : t + 1] = a_t
            # Prepare next-step's a_prev (a_t becomes a_{t-1} for the next outer step).
            a_prev = self.action_in(a_t)

        return actions

    # ----- Single-step inference API for closed-loop rollout -----

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state. The flat fused policy runs a
        2-token-per-step interleaved stream, so the backbone cache is
        sized at ``2 * T_max``. ``a_prev_emb`` is the previous step's
        action embedding (BOS at t=0).
        """
        T_max = int(min(T_max, self.action_horizon))
        dtype = dtype or next(self.parameters()).dtype
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T_max,
            device=device,
            dtype=dtype,
        )
        a_prev_emb = self.bos.expand(batch_size, 1, self.d_model).to(dtype)
        return {
            "params": params,
            "a_prev_emb": a_prev_emb,
            "dtype": dtype,
            "T_max": T_max,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Single outer step. Internally runs two backbone steps: one
        for the cond token at position ``2t``, one for the action token
        at position ``2t+1``. Returns normalized action ``(B, 1,
        action_dim)``. Mutates ``state`` in place.
        """
        dtype = state["dtype"]
        cond_seq = self._encode_cond(obs_norm, T=1)  # (B, 1, d_cond)
        fused = cond_seq.squeeze(1)  # (B, d_cond)
        c_tok = (
            self.cond_in(fused).unsqueeze(1) + self.pos_emb[:, 2 * t : 2 * t + 1]
        ).to(dtype)
        _ = self.backbone.step(c_tok, state["params"])

        a_tok = (state["a_prev_emb"] + self.pos_emb[:, 2 * t + 1 : 2 * t + 2]).to(dtype)
        h = self.backbone.step(a_tok, state["params"])
        a_t_norm = self.action_out(h)
        state["a_prev_emb"] = self.action_in(a_t_norm).to(dtype)
        return a_t_norm


class HNet(Algo):
    """
    H-Net policy Algo. Single-domain action-sequence model with per-frame
    obs conditioning -- each action token sees the obs at its own timestep.
    """

    def __init__(
        self,
        outer_stage: HNetOuterStage,
        norm_stats,
        d_cond: int = None,
        loss: Optional[Loss] = None,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        init_weights_range: Optional[float] = None,
        lr_multipliers: Optional[list] = None,
        use_parameter_groups: bool = False,
        weight_decay: float = 0.0,
        train_obs_transforms: list | None = None,
        **kwargs,
    ):
        """
        Training recipe knobs (all OFF by default — opt-in):

        - ``init_weights_range``: if set (e.g. ``0.02``), call
          ``hnet.init_weights(init_weights_range)`` after policy construction
          so ``out_proj`` / ``fc2`` weights get ``1/sqrt(n_residuals)``
          scaling.
        - ``lr_multipliers``: list of per-stage LR scales (outer→inner). If
          set, call ``hnet.apply_lr_multiplier(...)`` which stamps every
          parameter's ``_optim`` dict with ``lr_multiplier``.
        - ``use_parameter_groups``: if True, expose
          ``self.parameter_groups()`` so ``pl_model.configure_optimizers``
          builds AdamW param groups (bias / norm weights get
          ``weight_decay=0``; per-group ``lr = base_lr * lr_multiplier``).
        - ``weight_decay``: the base WD used when building parameter groups
          (only consulted when ``use_parameter_groups=True``). Outside of
          that, the optimizer config in the model YAML drives WD.

        Leaving all of these at their defaults reproduces "standard
        training": PyTorch default init, single LR for all params, single
        WD for all params from the optimizer config.
        """
        super().__init__()
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = outer_stage.action_horizon
        self.d_cond = d_cond  # legacy field; not used by refactored code
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Cache training-recipe knobs for configure_optimizers.
        self.use_parameter_groups = bool(use_parameter_groups)
        self.lr_multipliers = list(lr_multipliers) if lr_multipliers else None
        self.weight_decay = float(weight_decay)
        self.train_obs_transforms = list(train_obs_transforms or [])

        # Apply opt-in training recipe BEFORE moving to device.
        # outer_stage.inner_stage is the HNetCore (stage tree) for the
        # stage-based H-Net; for flat variants (FlatFusedOuterStage) it's
        # None — the training recipe knobs are no-ops in that case.
        hnet = outer_stage.inner_stage
        if hnet is not None and init_weights_range is not None:
            hnet.init_weights(initializer_range=float(init_weights_range))
        if hnet is not None and self.lr_multipliers is not None:
            hnet.apply_lr_multiplier(self.lr_multipliers)
        self._hnet_core = hnet

        if loss is None:
            loss = HNetLoss()
        self.nets = nn.ModuleDict({"outer_stage": outer_stage, "loss": loss})
        self.nets = self.nets.float().to(self.device)

        # Resolve per-embodiment keys via norm_stats (which owns the
        # MultiDataset key topology — same surface HPT uses).
        self.embodiment_ids = {}
        self.proprio_keys = {}
        self.lang_keys = {}
        self.camera_keys = {}
        self.resolved_ac_keys = {}
        for emb in self.domains:
            emb_id = get_embodiment_id(emb)
            self.embodiment_ids[emb] = emb_id
            self.proprio_keys[emb_id] = []
            self.lang_keys[emb_id] = []
            self.camera_keys[emb_id] = []
            for key in norm_stats.keys_of_type("action_keys", emb_id):
                if (
                    norm_stats.is_key_with_embodiment(key, emb_id)
                    and key == self.ac_keys[emb]
                ):
                    self.resolved_ac_keys[emb_id] = key
            for key in norm_stats.keys_of_type("proprio_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.proprio_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("lang_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.lang_keys[emb_id].append(key)
            for key in norm_stats.keys_of_type("camera_keys", emb_id):
                if norm_stats.is_key_with_embodiment(key, emb_id):
                    self.camera_keys[emb_id].append(key)

    # ----- Convenience accessors so code that references the old
    # ``self.outer_stage`` / cond_encoder / hnet keeps working via
    # forwarding to the new outer_stage submodules.

    @property
    def outer_stage(self) -> HNetOuterStage:
        return self.nets["outer_stage"]

    @property
    def loss(self) -> Loss:
        return self.nets["loss"]

    @property
    def policy(self) -> HNetOuterStage:
        """Back-compat alias — used to be HNetPolicy; now is HNetOuterStage."""
        return self.outer_stage

    @property
    def cond_encoder(self) -> CondEncoderModule:
        return self.outer_stage.cond_encoder

    @property
    def hnet(self) -> HNetCore:
        return self.outer_stage.inner_stage

    @property
    def input_modules(self) -> nn.ModuleList:
        return self.outer_stage.input_modules

    @property
    def action_out(self) -> nn.Module:
        return self.outer_stage.action_out

    # ---- Algo API --------------------------------------------------------

    # Keys emitted by ``pack_collate`` that aren't zarr-mapped data tensors
    # and must be passed through ``process_batch_for_training`` unchanged.
    _PACKED_META_KEYS = (
        "cu_seqlens",
        "max_seq_len",
        "seq_lens",
        "batch_size",
        "embodiment",
        "episode_idx",
        "chunk_offset",
    )

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for emb_name, _batch in batch.items():
            emb_id = get_embodiment_id(emb_name)
            processed[emb_id] = {}
            # Detect packed batches by the presence of cu_seqlens. Packed and
            # padded batches have a different key topology; treat them
            # separately so we don't try to keyname-resolve the meta keys.
            is_packed = "cu_seqlens" in _batch

            for key, value in _batch.items():
                if is_packed and key in self._PACKED_META_KEYS:
                    processed[emb_id][key] = value
                    continue
                key_name = self.norm_stats.zarr_key_to_keyname(key, emb_id)
                # Pre-existing typo: tested ``key is not None`` instead of
                # ``key_name``, which caused unrelated batch keys (e.g.
                # ``metadata.robot_name`` from _read_span) to be stored under
                # the None key. Skip silently when keyname can't be resolved.
                if key_name is not None:
                    processed[emb_id][key_name] = value

            # F5: H-Net does NOT consume ``pad_mask`` — variable-length is
            # carried by ``cu_seqlens`` in packed mode and by the full-length
            # convention in padded mode. The previous code populated an
            # all-ones ``pad_mask`` (or None) here as a vestige from the
            # ACT/HPT algos; no downstream H-Net code reads it. Don't
            # re-introduce it: if you need a true valid-token mask, plumb
            # it through ``HNetContext`` instead.
            processed[emb_id]["_packed"] = is_packed
            # Per-feature normalization via MultiDataset stats: each tensor
            # gets ``(x - mean) / std`` (or quantile equivalent) broadcast
            # against (action_dim,) / (proprio_dim,) stats. Works for both
            # padded ``(B, T, D)`` and packed ``(T_total, D)`` shapes.
            processed[emb_id] = self.norm_stats.normalize(processed[emb_id], emb_id)
            # Post-normalize train-only obs transforms (e.g. GaussianObsNoise).
            # Whole-dict; transforms must preserve packed metadata like cu_seqlens.
            if self.train_obs_transforms and self.outer_stage.training:
                for t in self.train_obs_transforms:
                    processed[emb_id] = t(processed[emb_id])
            processed[emb_id]["embodiment"] = torch.tensor(
                [emb_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed[emb_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[emb_id][key] = value
        return processed

    def _build_obs(self, _batch, emb_id):
        obs = {}
        for key in (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        ):
            if key in _batch:
                obs[key] = _batch[key]
        return obs

    @override
    def forward_training(self, batch):
        """Refactored training forward — delegates to outer_stage + loss.

        Builds (per-embodiment) the inner batch dict + HNetContext, calls
        outer_stage(batch, ctx) which writes batch[pred_action] + ctx.aux,
        calls loss(batch, ctx) which combines action MSE + chunker ratio
        loss and stores the per-term breakdown on ctx (ctx.action_loss,
        ctx.ratio_loss) for the logging dict.
        """
        predictions = OrderedDict()
        outer_stage = self.outer_stage
        loss_fn = self.loss
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)
            is_packed = _batch.get("_packed", False)

            new_batch = {"actions": actions, "__obs": obs}
            ctx = HNetContext(
                cond_dict={},
                aux=[],
                inference_params=None,
                cu_seqlens=_batch["cu_seqlens"] if is_packed else None,
                max_seqlen=int(_batch["max_seq_len"]) if is_packed else None,
            )
            outer_stage(new_batch, ctx)
            total_loss = loss_fn(new_batch, ctx)
            pred = new_batch["pred_action"]
            mse = getattr(ctx, "action_loss", total_loss)
            rloss = getattr(
                ctx,
                "ratio_loss",
                torch.zeros((), device=total_loss.device, dtype=total_loss.dtype),
            )
            predictions[f"{emb_id}_pred"] = pred
            predictions[f"{emb_id}_action_loss"] = mse
            predictions[f"{emb_id}_ratio_loss"] = rloss

            # Per-chunker stats for logging.
            stats = chunk_stats_from_aux(ctx.aux)
            for k, v in stats.items():
                predictions[f"{emb_id}_{k}"] = torch.tensor(v, device=mse.device)
        return predictions

    @override
    def forward_eval(self, batch):
        """Per-frame teacher-forced eval.

        For each batch entry we run a SINGLE forward pass with the GT
        action stream (same as training) and compare per-frame predictions
        against GT in raw / unnormalized action space. AR rollout used to
        live here but is pointless when the obs sequence is fixed: the
        predicted action doesn't change what the model sees next, so AR
        just compounds exposure-bias error without testing anything
        useful. To test closed-loop behavior, run a separate
        envrollout evaluator that steps a simulator with predicted
        actions.

        Returns a dict with keys ``emb{id}_{ac_key}`` carrying
        ``(B, T_max, action_dim)`` unnormalized predictions (zero-padded
        past each episode's length) and ``emb{id}_seq_lens`` ``(B,)`` so
        downstream metric code can mask the padded positions.
        """
        unnorm = {}
        policy = self.outer_stage
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            if _batch.get("_packed", False):
                preds_padded, seq_lens = self._teacher_forced_packed(_batch, emb_id)
                preds = OrderedDict()
                preds[ac_key] = preds_padded
                unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
                for key, val in unnorm_actions.items():
                    unnorm[f"emb{emb_id}_{key}"] = val
                unnorm[f"emb{emb_id}_seq_lens"] = seq_lens
                continue
            # Padded mode (legacy): no packed dataset wired but kept for
            # completeness.
            obs = self._build_obs(_batch, emb_id)
            actions = _batch[ac_key]
            pred, _ = policy.forward_padded(actions, obs)
            preds = OrderedDict()
            preds[ac_key] = pred
            unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    @torch.no_grad()
    def _teacher_forced_packed(self, _batch: dict, emb_id: int):
        """Single-pass teacher-forced eval for a packed validation batch.

        Runs ``policy.forward_packed`` (the same path used in training)
        and unpacks the resulting ``(T_total, action_dim)`` predictions
        into ``(B, T_max, action_dim)`` zero-padded per-episode for
        downstream metric / viz code.
        """
        policy = self.outer_stage
        ac_key = self.resolved_ac_keys[emb_id]
        actions = _batch[ac_key]
        obs = self._build_obs(_batch, emb_id)
        cu = _batch["cu_seqlens"]
        max_seqlen = int(_batch["max_seq_len"])
        seq_lens = _batch["seq_lens"].clone()
        pred_packed, _ = policy.forward_packed(actions, obs, cu, max_seqlen)

        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        preds_padded = torch.zeros(
            B,
            T_max,
            action_dim,
            device=pred_packed.device,
            dtype=pred_packed.dtype,
        )
        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            preds_padded[b, : e - s] = pred_packed[s:e]
        return preds_padded, seq_lens

    @torch.no_grad()
    def _ar_rollout_packed(self, _batch: dict, emb_id: int):
        """Per-episode AR rollout for a packed validation batch.

        For each sub-sequence ``[s, e)`` in ``cu_seqlens``:
          1. Slice that episode's obs into ``(1, T_ep, ...)``.
          2. Call ``policy.generate(obs_ep, batch_size=1, T=T_ep)`` to AR
             rollout exactly ``T_ep`` steps from BOS.
          3. Stash the prediction.

        Returns:
            preds_padded: ``(B, T_max, action_dim)`` (zero-padded past each
                episode's length).
            seq_lens:     ``(B,)`` long, the per-episode rollout lengths
                (matches ``_batch['seq_lens']`` and used for masking the
                padding in downstream MSE).
        """
        policy = self.outer_stage
        cu = _batch["cu_seqlens"]
        seq_lens = _batch["seq_lens"].clone()
        B = int(seq_lens.shape[0])
        T_max = int(seq_lens.max().item())
        action_dim = policy.action_dim
        device = self.device

        # Gather the obs keys we need for the cond encoder.
        obs_keys = (
            self.proprio_keys[emb_id]
            + self.lang_keys[emb_id]
            + self.camera_keys[emb_id]
        )
        obs_keys = [k for k in obs_keys if k in _batch]

        preds_padded = torch.zeros(B, T_max, action_dim, device=device)

        for b in range(B):
            s = int(cu[b].item())
            e = int(cu[b + 1].item())
            T_ep = e - s
            # Slice each obs key to the episode's range and add a leading
            # batch dim. The packed tensor is (T_total, ...) so slicing along
            # dim 0 gives (T_ep, ...) → unsqueeze → (1, T_ep, ...).
            obs_ep = {k: _batch[k][s:e].unsqueeze(0) for k in obs_keys}
            a_ep = policy.generate(
                obs_ep,
                batch_size=1,
                device=device,
                T=T_ep,
            )  # (1, T_ep, action_dim)
            preds_padded[b, :T_ep] = a_ep.squeeze(0)

        return preds_padded, seq_lens

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for emb_id in batch.keys():
            a = predictions[f"{emb_id}_action_loss"]
            r = predictions[f"{emb_id}_ratio_loss"]
            loss_dict[f"emb{emb_id}_action_loss"] = a
            loss_dict[f"emb{emb_id}_ratio_loss"] = r
            # Ratio-loss weights are baked into each chunker stage, so r
            # is already a properly-weighted sum.
            total = total + a + r

            # Pass non-loss stats through to logging (boundary_rate /
            # avg_chunk_len, per-chunker and aggregate). They are 0-dim
            # tensors so log_info.item() still works.
            for key, value in predictions.items():
                prefix = f"{emb_id}_"
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix) :]
                if tail in ("pred", "action_loss", "ratio_loss"):
                    continue
                loss_dict[f"emb{emb_id}_{tail}"] = value
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

    # ----- Sim eval hook (SimRolloutEval.inference_step contract) ----- #
    # Single entry point. t=0 is the universal reset signal (allocates a
    # fresh AR state); t>0 steps against the cached state. The eval class
    # never touches model state — it lives entirely in self._sim_state.

    @torch.no_grad()
    def inference_step(
        self, obs_zarr: dict, t: int, emb_id: int, T_max=None
    ) -> "np.ndarray":
        """One closed-loop sim step.

        Args:
            obs_zarr: env obs in canonical zarr-key dict (already on device).
            t: timestep within the rollout. t=0 resets state.
            emb_id: embodiment id.

        Returns:
            absolute-frame action as np.float32 of shape (action_dim,).
        """
        import numpy as np

        policy = self.outer_stage
        if t == 0:
            device = next(self.outer_stage.parameters()).device
            default_T = int(getattr(policy, "action_horizon", 1024))
            T_max_use = int(T_max) if T_max is not None else default_T
            import torch

            self._sim_state = policy.init_step_state(
                batch_size=1, T_max=T_max_use, device=device, dtype=torch.bfloat16
            )
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys
            else self.ac_keys[emb_id]
        )
        obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
        action_norm = policy.step(self._sim_state, obs_norm, t)
        action_unnorm = self.norm_stats.unnormalize(
            {ac_key: action_norm.squeeze(0).squeeze(0)},
            emb_id,
        )[ac_key]
        return action_unnorm.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # ----- Optional training recipe hook for pl_model.configure_optimizers ----- #

    def parameter_groups(self, base_lr: float):
        """Return AdamW-ready ``list[dict]`` if ``use_parameter_groups``,
        else ``None`` (caller falls back to ``self.nets.parameters()``).

        Groups are built by the inner HNet stage tree via
        ``HNetCore.parameter_groups(weight_decay=self.weight_decay)``, then
        each group's ``lr`` is set to ``base_lr * lr_multiplier``. Params
        that aren't part of the HNet stage tree (e.g. ``action_in``,
        ``action_out``, ``cond_encoder``, ``bos``, ``pos_emb``) are added in
        a single extra group with ``lr_multiplier=1.0`` so optimizer
        instantiation still sees every learnable parameter exactly once.
        """
        if not self.use_parameter_groups:
            return None

        # Groups for params inside the HNet stage tree.
        groups = self._hnet_core.parameter_groups(weight_decay=self.weight_decay)
        for g in groups:
            g["lr"] = float(base_lr) * float(g.get("lr_multiplier", 1.0))

        # Extra group for everything *not* inside the HNet stage tree.
        hnet_param_ids = {id(p) for g in groups for p in g["params"]}
        extra_params, extra_bias_norm = [], []
        for name, p in self.nets.named_parameters():
            if id(p) in hnet_param_ids or not p.requires_grad:
                continue
            # Bias / norm-weight detection (same rule as parameter_groups).
            if name.endswith(".bias") or ".norm" in name or "rmsnorm" in name.lower():
                extra_bias_norm.append(p)
            else:
                extra_params.append(p)
        if extra_params:
            groups.append(
                {
                    "params": extra_params,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": self.weight_decay,
                }
            )
        if extra_bias_norm:
            groups.append(
                {
                    "params": extra_bias_norm,
                    "lr": float(base_lr),
                    "lr_multiplier": 1.0,
                    "weight_decay": 0.0,
                }
            )
        return groups


class HNetFused(HNet):
    """Flat-transformer (no chunker) variant. Thin alias for ``HNet`` —
    kept as a separate ``_target_`` for the hnet_pushshapes_fused*.yaml
    configs that historically referenced this class.

    All the flat-fused behavior (interleaved [c_t, a_t] tokens, single
    Isotropic stack, no chunker / ratio loss) lives in
    :class:`egomimic.algo.flat_fused_outer_stage.FlatFusedOuterStage`.
    Use that as the ``outer_stage`` argument.
    """

    pass
