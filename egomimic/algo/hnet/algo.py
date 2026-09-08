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
from egomimic.models.stems.input_modules import ActionInToken, InputModule
from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet as HNetCore
from egomimic.models.hnet.hnet import chunk_stats_from_aux
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class HNetLoss(nn.Module):
    """Action-MSE + per-chunker ratio-loss regulariser.

    Self-contained loss for the H-Net algo — lives here next to ``HNetPolicy``
    (each model owns its own loss). Implements the ``forward(batch, ctx) ->
    scalar`` contract the algo orchestrator calls.

    Reads:
      - ``batch["pred_action"]`` (per-token predicted actions; written by
        ``HNetOuterStage.decode``)
      - ``batch["actions"]`` (per-token GT actions)
      - ``ctx.aux`` (list of chunker-stage auxiliaries; each chunker
        contributes a ``ratio_loss`` * weight term)

    The per-chunker weight lives inside each ChunkerStage's aux already
    (see ``ratio_loss_from_aux``); this class just sums them onto the
    action MSE.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        from egomimic.models.hnet.hnet import ratio_loss_from_aux

        pred = batch["pred_action"]
        target = batch["actions"]
        action_loss = torch.mean((pred - target) ** 2)

        # Sum per-chunker boundary regularisers if any.
        aux = getattr(ctx, "aux", None) or []
        ratio_loss = (
            ratio_loss_from_aux(aux, device=action_loss.device)
            if aux
            else torch.zeros((), device=action_loss.device, dtype=action_loss.dtype)
        )
        moe_aux = _moe_aux_from_ctx(ctx, action_loss)
        # Stash the per-term breakdown on ctx so the algo class's logging
        # path can read them without recomputing.
        ctx.action_loss = action_loss
        ctx.ratio_loss = ratio_loss
        ctx.moe_aux_loss = moe_aux
        return action_loss + ratio_loss + moe_aux


class GMMLoss(nn.Module):
    """GMM negative-log-likelihood action loss + per-chunker ratio loss.

    Drop-in peer of :class:`HNetLoss` for the H-Net algo when the outer stage
    uses a GMM head (``action_head_type='gmm'``). Reads:
      - ``batch["pred_action"]`` — flat GMM *params* (written by
        ``HNetOuterStage.decode``).
      - ``batch["actions"]`` — per-token GT actions.
      - ``ctx.extras["gmm_head"]`` — the GMM head, reused for its (param-free)
        ``.nll`` / chunk-splitting config.
      - ``ctx.cu_seqlens`` — packed episode boundaries (None => padded).
      - ``ctx.aux`` — per-chunker ratio-loss regularisers.

    For ``chunk_len = C`` the target at obs-step ``t`` is the next ``C`` actions
    ``actions[t : t+C]``, repeat-padded at the episode end (repeat-unmasked,
    matching the robomimic BC-RNN chunked-GMM convention).
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _chunked_targets_packed(actions: torch.Tensor, cu_seqlens, C: int):
        # actions (T_total, D) -> (T_total, C, D), clamped within each episode.
        T_total = actions.shape[0]
        device = actions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        ep_end = torch.empty(T_total, dtype=torch.long, device=device)
        for b in range(cu.shape[0] - 1):
            ep_end[cu[b] : cu[b + 1]] = cu[b + 1]
        t = torch.arange(T_total, device=device)
        idx = t[:, None] + torch.arange(C, device=device)[None, :]
        idx = torch.minimum(idx, (ep_end - 1)[:, None])
        return actions[idx]

    @staticmethod
    def _chunked_targets_padded(actions: torch.Tensor, C: int):
        # actions (B, T, D) -> (B, T, C, D), clamped to the last frame.
        T = actions.shape[1]
        device = actions.device
        t = torch.arange(T, device=device)
        idx = torch.clamp(t[:, None] + torch.arange(C, device=device)[None, :], max=T - 1)
        return actions[:, idx]

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        from egomimic.models.hnet.hnet import ratio_loss_from_aux

        raw = batch["pred_action"]
        actions = batch["actions"]
        head = ctx.extras.get("gmm_head") if hasattr(ctx, "extras") else None
        if head is None:
            raise RuntimeError(
                "GMMLoss requires ctx.extras['gmm_head'] (set action_head_type='gmm')."
            )
        C = int(head.chunk_len)
        if batch.get("chunk_targets") is not None:
            # obs_stride path: targets were pre-built at the strided token
            # frames (FULL-resolution actions[f:f+C] per kept token), so the
            # (T_strided, C, D) targets already align with the strided preds.
            target = batch["chunk_targets"]
        elif getattr(ctx, "cu_seqlens", None) is not None:
            target = self._chunked_targets_packed(actions, ctx.cu_seqlens, C)
        else:
            target = self._chunked_targets_padded(actions, C)
        if C == 1:
            target = target.squeeze(-2)  # head.nll expects (..., D) for chunk_len==1
        action_loss = head.nll(raw, target)

        aux = getattr(ctx, "aux", None) or []
        ratio_loss = (
            ratio_loss_from_aux(aux, device=action_loss.device)
            if aux
            else torch.zeros((), device=action_loss.device, dtype=action_loss.dtype)
        )
        moe_aux = _moe_aux_from_ctx(ctx, action_loss)
        ctx.action_loss = action_loss
        ctx.ratio_loss = ratio_loss
        ctx.moe_aux_loss = moe_aux
        return action_loss + ratio_loss + moe_aux


def _moe_aux_from_ctx(ctx, ref: torch.Tensor) -> torch.Tensor:
    """Sum the per-MoE-layer load-balance losses that ``ComputeStage`` published
    on ``ctx.extras['moe_aux']`` this forward. Returns a 0-dim zero (no MoE) so
    dense models are byte-identical. Also stashes ``ctx.moe_entries`` (the raw
    per-layer diagnostics) for the algo's logging path."""
    entries = (getattr(ctx, "extras", None) or {}).get("moe_aux", []) or []
    ctx.moe_entries = entries
    if not entries:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return torch.stack([e["aux_loss"].to(ref.dtype) for e in entries]).sum()


def _init_ar_state(module, cache_owner, batch_size: int, T_max: int, device,
                   dtype=None) -> dict:
    """Allocate the opaque AR inference state shared by the H-Net policies.

    ``module`` supplies the dtype when the caller does not pin one;
    ``cache_owner`` is whichever submodule owns the KV cache (``self.hnet`` for
    HNetPolicy, ``self.inner_stage`` for HNetOuterStage).

    ``T_max`` is not clamped by ``action_horizon`` (F6) -- RoPE handles arbitrary
    positions -- but it still sizes the KV cache allocation. The per-step token is
    assembled by ``input_modules.step()`` on each call, so the state carries only
    the previous prediction (for ActionInToken) plus the dtype/T_max contract.

    The dtype line matters: allocating this state at a fixed dtype rather than the
    model's own silently produced a bf16 state under fp32 weights, which
    under-measured H-Net closed-loop coverage ~2-2.6x and read as the policy
    failing closed-loop. Keep it derived, and keep it in one place.
    """
    T_max = int(T_max)
    dtype = dtype or next(module.parameters()).dtype
    params = cache_owner.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=T_max,
        device=device,
        dtype=dtype,
    )
    return {
        "params": params,
        "prev_action": None,
        "batch_size": int(batch_size),
        "device": device,
        "dtype": dtype,
        "T_max": T_max,
    }


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
        return _init_ar_state(
            self, self.hnet, batch_size, T_max, device, dtype
        )

    @torch.no_grad()
    def step(
        self, state: dict, obs_norm: dict, t: int,
        embodiment_id: Optional[str] = None,
    ) -> torch.Tensor:
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
                embodiment_id=embodiment_id,
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
        _head = self.action_out[embodiment_id] if isinstance(self.action_out, nn.ModuleDict) else self.action_out
        a_t_norm = _head(h)

        # F6: no per-step pos_emb add. Cache the prediction so the next
        # step's ActionInToken (if any) can read it.
        state["prev_action"] = a_t_norm
        return a_t_norm


class HNetOuterStage(nn.Module):
    """Stage-based H-Net outer stage — the H-Net algo's outer stage.

    Co-located with ``HNetPolicy`` / ``PackedAlgoBase`` (each model owns its own
    pieces). Same responsibilities as the previous ``HNetPolicy``: input_modules
    (per-token summed contributions), cond_encoder (obs → AdaLN cond_dict),
    inner_stage = HNetCore (the stage tree), action_head decoding — reshaped into
    the OuterStage encode → inner_stage → decode contract. ``forward_padded`` /
    ``forward_packed`` / ``step`` mirror the prior ``HNetPolicy`` paths.

    Args:
        action_dim: action feature width.
        action_horizon: kept for legacy compatibility; RoPE-based attention
            doesn't actually clamp on this anymore, but it remains as a
            "default rollout length" for ``generate``.
        d_model: trunk hidden dim. Must match ``hnet.input_hidden_dim`` and
            ``hnet.output_hidden_dim``.
        cond_encoder: CondEncoderModule for obs → cond_dict (AdaLN).
        hnet: HNetCore — the inner_stage (stage tree).
        action_head_type: ``"linear"`` (single Linear) or ``"mlp"`` (2-layer).
        input_modules: list of InputModule subclasses. Each emits a per-token
            ``(B, T, d_model)`` contribution; they are summed before the
            inner_stage. Default: ``[ActionInToken(action_dim, d_model)]``
            (legacy AR behaviour with BOS + right-shifted actions).
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
        gmm_head: nn.Module | None = None,
        gmm_heads: dict | None = None,  # per-embodiment heads {domain: head}
    ):
        super().__init__()
        self.inner_stage = hnet
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.d_model = int(d_model)

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
            self.action_out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, action_dim),
            )
        elif action_head_type == "gmm":
            # GMM head (pre-instantiated via hydra, like WindowedBC). forward(h)
            # emits flat GMM params (chunk_len * M * (2D+1)); GMMLoss reads them
            # back through the same head's .nll. Pairs with chunk_len>1 for
            # 1-obs -> N-action chunking. gmm_heads (dict domain->head) makes the
            # head PER-EMBODIMENT, dispatched on ctx.embodiment_id in decode()
            # (so only the high-level trunk is shared between embodiments).
            if gmm_heads is not None:
                self.action_out = nn.ModuleDict(dict(gmm_heads))
            elif gmm_head is not None:
                self.action_out = gmm_head
            else:
                raise ValueError(
                    "action_head_type='gmm' requires gmm_head or gmm_heads")
        else:
            raise ValueError(
                f"action_head_type must be 'linear'|'mlp'|'gmm', got "
                f"{action_head_type!r}"
            )
        self.action_head_type = action_head_type

        self.cond_encoder = cond_encoder

        # Sanity-check inner_stage hidden-dim contract.
        if self.inner_stage.input_hidden_dim != d_model:
            raise ValueError(
                f"hnet.input_hidden_dim ({self.inner_stage.input_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )
        if self.inner_stage.output_hidden_dim != d_model:
            raise ValueError(
                f"hnet.output_hidden_dim ({self.inner_stage.output_hidden_dim}) "
                f"must equal d_model ({d_model})."
            )

    # ------------------------------------------------------------------
    # OuterStage API.
    # ------------------------------------------------------------------

    def encode(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        """Build the per-token input tensor x by summing per-input-module
        contributions. Reads actions + obs from batch; mode comes from
        ``ctx.cu_seqlens`` (packed if set, padded otherwise).
        """
        is_packed = ctx.cu_seqlens is not None
        if is_packed:
            return self._encode_packed(batch, ctx)
        return self._encode_padded(batch, ctx)

    def _encode_padded(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        actions = batch["actions"]  # (B, T, A)
        obs = batch["__obs"]
        B, T, _ = actions.shape
        device = actions.device
        dtype = actions.dtype
        x = None
        for mod in self.input_modules:
            contrib = mod.forward_padded(
                actions=actions, obs=obs, B=B, T=T, device=device, dtype=dtype,
                embodiment_id=ctx.embodiment_id,
            )
            x = contrib if x is None else x + contrib
        if x is None:
            raise RuntimeError("input_modules produced no tokens")
        # Build cond_dict from per-frame obs and stuff onto ctx so the
        # inner stages (which read it via cond_key) get it. ``embodiment_id``
        # is None for single-emb runs (CondEncoderModule ignores it) and the
        # domain name in cotrain (MultiEmbodimentCondEncoder dispatches on it).
        cond_dict = self.cond_encoder.encode(
            obs, self.action_horizon, embodiment_id=ctx.embodiment_id
        )
        ctx.cond_dict = cond_dict
        return x

    def _encode_packed(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        actions_packed = batch["actions"]  # (T_total, A)
        obs_packed = batch["__obs"]
        T_total = actions_packed.shape[0]
        device = actions_packed.device
        dtype = actions_packed.dtype
        cu_seqlens = ctx.cu_seqlens.to(device=device, dtype=torch.long)

        x_packed = None
        for mod in self.input_modules:
            contrib = mod.forward_packed(
                actions_packed=actions_packed,
                obs_packed=obs_packed,
                cu_seqlens=cu_seqlens,
                T_total=T_total,
                device=device,
                dtype=dtype,
                embodiment_id=ctx.embodiment_id,
            )
            x_packed = contrib if x_packed is None else x_packed + contrib
        if x_packed is None:
            raise RuntimeError("input_modules produced no tokens")

        # Packed cond: fake batch=1 by unsqueezing each obs, then squeeze
        # the leading dim back out.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(
            obs_for_encode, T_action=T_total, embodiment_id=ctx.embodiment_id
        )
        cond_packed = {k: v.squeeze(0) for k, v in cond_padded.items()}
        ctx.cond_dict = cond_packed
        return x_packed

    def decode(self, h: torch.Tensor, batch: dict, ctx: HNetContext) -> None:
        """Apply the action head and write ``batch["pred_action"]``.

        For the GMM head, ``pred_action`` holds the flat GMM *params* (not point
        actions); the head is stashed on ``ctx.extras`` so ``GMMLoss`` can read
        them back through the same head's ``.nll``.
        """
        head = (self.action_out[ctx.embodiment_id]
                if isinstance(self.action_out, nn.ModuleDict) else self.action_out)
        object.__setattr__(self, "_cur_head", head)  # NOT a registered submodule: self._cur_head=head adds duplicate _cur_head.* keys to state_dict and breaks strict resume
        batch["pred_action"] = head(h)
        if self.action_head_type == "gmm":
            ctx.extras["gmm_head"] = head

    def forward(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        """Dispatch padded vs packed, run encode → inner_stage → decode.
        Returns the post-trunk hidden tensor (the loss reads
        ``batch["pred_action"]`` instead, so this return value is mostly
        for diagnostic / downstream-stage chaining)."""
        x = self.encode(batch, ctx)
        h = self.inner_stage(x, ctx)
        self.decode(h, batch, ctx)
        return h

    # ------------------------------------------------------------------
    # Legacy bridge methods for code that calls (actions, obs[, cu, msl])
    # and expects (pred, aux). These wrap the new (batch, ctx) API so the
    # HNet algo class's forward_eval / _teacher_forced_packed / etc. can
    # keep working with a minimal call-site rename.
    # ------------------------------------------------------------------

    def _decode_pred_actions(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a head output to POINT actions for the (per-frame) eval path.

        Non-GMM heads already emit point actions -> returned unchanged. The GMM
        head emits flat mixture params: decode (sample under low-noise eval)
        and, for a chunked head, take the action at chunk position 0 (the action
        AT the current frame) so the teacher-forced overlay stays per-frame
        ``(..., action_dim)``. Sim rollout uses the FULL chunk separately
        (see ``inference_step`` open-loop buffer).
        """
        if self.action_head_type != "gmm":
            return raw
        head = getattr(self, "_cur_head", None)
        if head is None or isinstance(head, nn.ModuleDict):
            head = (next(iter(self.action_out.values()))
                    if isinstance(self.action_out, nn.ModuleDict) else self.action_out)
        a = head.decode(raw)
        if int(getattr(head, "chunk_len", 1)) > 1:
            a = a[..., 0, :]
        return a

    def forward_padded(
        self, actions: torch.Tensor, obs: dict, embodiment_id: Optional[str] = None
    ):
        """Legacy bridge: padded (actions, obs) -> (pred, aux)."""
        batch = {"actions": actions, "__obs": obs}
        ctx = HNetContext(
            cond_dict={}, aux=[], inference_params=None, embodiment_id=embodiment_id
        )
        self.forward(batch, ctx)
        return self._decode_pred_actions(batch["pred_action"]), ctx.aux

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        embodiment_id: Optional[str] = None,
    ):
        """Legacy bridge: packed (...) -> (pred, aux)."""
        batch = {"actions": actions_packed, "__obs": obs_packed}
        ctx = HNetContext(
            cond_dict={},
            aux=[],
            inference_params=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
            embodiment_id=embodiment_id,
        )
        self.forward(batch, ctx)
        return self._decode_pred_actions(batch["pred_action"]), ctx.aux

    # ------------------------------------------------------------------
    # Inference: AR step. Used by the algo class's closed-loop
    # path; not part of the OuterStage abstract API.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state. Returns an opaque dict for
        :meth:`step`."""
        return _init_ar_state(
            self, self.inner_stage, batch_size, T_max, device, dtype
        )

    @torch.no_grad()
    def step(
        self, state: dict, obs_norm: dict, t: int, embodiment_id: Optional[str] = None
    ) -> torch.Tensor:
        """Single online AR step. Returns ``(B, 1, action_dim)``.

        ``embodiment_id`` (domain name) is threaded onto the per-step ctx so a
        ``PerEmbodimentStage`` can dispatch to the right per-emb sub-stage in
        the AR cache; None for single-embodiment models.
        """
        cond_dict_seq = self.cond_encoder.encode(
            obs_norm, T_action=1, embodiment_id=embodiment_id
        )
        cond_2d = {k: v.squeeze(1) for k, v in cond_dict_seq.items()}

        obs_step = {
            k: v.unsqueeze(1) if (
                torch.is_tensor(v)
                and v.dim() < 5
                and v.shape[0] == state["batch_size"]
                and (v.dim() == 1 or v.shape[1] != 1)
            ) else v
            for k, v in obs_norm.items()
        }
        cur = None
        for mod in self.input_modules:
            contrib = mod.step(
                prev_action_norm=state.get("prev_action"),
                obs_norm=obs_step,
                t=t,
                B=state["batch_size"],
                device=state["device"],
                dtype=state["dtype"],
                embodiment_id=embodiment_id,
            )
            cur = contrib if cur is None else cur + contrib
        if cur is None:
            raise RuntimeError("input_modules produced no tokens at step")

        ctx = HNetContext(
            cond_dict=cond_2d,
            aux=[],
            inference_params=state["params"],
            embodiment_id=embodiment_id,
        )
        h = self.inner_stage.step(cur, ctx)
        _head = self.action_out[embodiment_id] if isinstance(self.action_out, nn.ModuleDict) else self.action_out
        a_t_norm = _head(h)
        state["prev_action"] = a_t_norm
        return a_t_norm


def _stride_packed(obs, actions, cu_seqlens, sigma, C):
    """Subsample a packed batch's obs-token stream to every ``sigma``-th frame
    within each episode, and build the matching ``(T_strided, C, D)`` GMM
    targets from FULL-resolution actions.

    Args:
        obs: dict of per-frame packed tensors, each ``(T_total, ...)`` (plus any
            non-per-frame entries, which are passed through untouched).
        actions: ``(T_total, D)`` packed FULL-resolution actions.
        cu_seqlens: ``(B+1,)`` packed episode boundaries (frame units).
        sigma: obs stride (>= 1). Token k of episode ``[s, e)`` reads frame
            ``s + sigma*k`` for k = 0 .. ceil((e-s)/sigma)-1.
        C: chunk_len. Target for the token at frame ``f`` is ``actions[f:f+C]``,
            clamped to the episode's last frame (repeat-unmasked tail).

    Returns ``(strided_obs, new_cu, new_max_seqlen, chunk_targets, sidx, ep_end)``
    where ``sidx`` is the ``(T_strided,)`` original-frame index of each kept
    token and ``ep_end`` is the ``(T_strided,)`` exclusive episode-end frame of
    each token (used by the eval to clamp the tiled per-frame reconstruction).
    """
    dev = actions.device
    cu = cu_seqlens.to(device=dev, dtype=torch.long)
    T_total = int(actions.shape[0])
    idx_list, ep_end_list, new_cu = [], [], [0]
    for b in range(int(cu.numel()) - 1):
        s, e = int(cu[b].item()), int(cu[b + 1].item())
        fr = torch.arange(s, e, sigma, device=dev)
        idx_list.append(fr)
        ep_end_list.append(torch.full_like(fr, e))
        new_cu.append(new_cu[-1] + int(fr.numel()))
    sidx = torch.cat(idx_list) if idx_list else torch.empty(0, dtype=torch.long, device=dev)
    ep_end = torch.cat(ep_end_list) if ep_end_list else torch.empty(0, dtype=torch.long, device=dev)
    new_cu_t = torch.tensor(new_cu, device=dev, dtype=torch.long)
    # Strided obs: gather per-frame tensors at sidx; pass through anything whose
    # leading dim isn't T_total (e.g. scalar/meta tensors).
    strided_obs = {}
    for k, v in obs.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == T_total:
            strided_obs[k] = v[sidx]
        else:
            strided_obs[k] = v
    # Chunk targets: for each kept frame f, actions[f:f+C] clamped to ep_end-1.
    tgt_idx = sidx[:, None] + torch.arange(C, device=dev)[None, :]
    tgt_idx = torch.minimum(tgt_idx, (ep_end - 1)[:, None])
    chunk_targets = actions[tgt_idx]  # (T_strided, C, D)
    new_max = int((new_cu_t[1:] - new_cu_t[:-1]).max().item()) if sidx.numel() else 0
    return strided_obs, new_cu_t, new_max, chunk_targets, sidx, ep_end


class PackedAlgoBase(Algo):
    """
    Packed-sequence policy Algo base. Single-domain action-sequence model with
    per-frame obs conditioning -- each action token sees the obs at its own
    timestep. Subclassed by WindowedBC (bc.py); the inner architecture is
    supplied via ``outer_stage`` and may itself be an H-Net stage tree.
    """

    def __init__(
        self,
        outer_stage: HNetOuterStage,
        norm_stats,
        d_cond: int = None,
        loss: Optional[nn.Module] = None,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        init_weights_range: Optional[float] = None,
        lr_multipliers: Optional[list] = None,
        use_parameter_groups: bool = False,
        weight_decay: float = 0.0,
        train_obs_transforms: list | None = None,
        episode_level_transforms: list | None = None,
        obs_stride: Optional[int] = None,
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
        # Reverse map emb_id -> domain name, used in cotrain so forward_*
        # can set ctx.embodiment_id for PerEmbodimentStage /
        # MultiEmbodimentCondEncoder dispatch. Empty/identity for single-emb.
        self.domain_by_id = {get_embodiment_id(emb): emb for emb in self.domains}
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
        self.episode_level_transforms = list(episode_level_transforms or [])
        # obs_stride: subsample the obs-token stream fed to the chunker so
        # routed tokens are ``obs_stride`` sim-frames apart (mirrors the
        # working WindowedBC/HNetCore recipe). With stride > 1 the RoutingModule
        # sees temporally-spaced tokens (consecutive cos_sim < 1) so boundaries
        # can fire; stride == 1 (default) preserves the original per-frame path
        # byte-for-byte. Each kept token still predicts the next chunk_len
        # actions at FULL frame resolution, so with obs_stride == chunk_len the
        # 8-action chunks tile the episode exactly (like obs_stride=8/chunk_len=8
        # on the RNN side).
        # Resolve the action head's chunk_len, robust to per-embodiment
        # ModuleDict heads (getattr on the dict itself returns no chunk_len).
        _ao = getattr(outer_stage, "action_out", None)
        if isinstance(_ao, nn.ModuleDict):
            _cls = sorted({int(getattr(h, "chunk_len", 1)) for h in _ao.values()})
            if len(_cls) > 1:
                raise ValueError(
                    f"per-embodiment heads have heterogeneous chunk_len {_cls}; "
                    f"the obs_stride<->chunk_len coupling is ambiguous."
                )
            _C = _cls[0] if _cls else 1
        else:
            _C = int(getattr(_ao, "chunk_len", 1))
        # DEFAULT obs_stride to chunk_len: a chunked head emits chunk_len actions
        # per obs-step, so the policy must re-observe every chunk_len frames.
        # Defaulting here means a chunked model can NEVER be silently trained /
        # eval'd at dense stride-1 (the bug that tanked dual-stream closed-loop:
        # trained dense @1 but eval'd @chunk_len=4). Explicit obs_stride wins.
        if obs_stride is None:
            obs_stride = _C
        self.obs_stride = int(obs_stride)
        if self.obs_stride < 1:
            raise ValueError(f"obs_stride must be >= 1, got {self.obs_stride}")
        # Only the obs_stride == chunk_len, GMM-head recipe is validated (the
        # chunk_len actions then tile the episode exactly). Fail loud on the
        # un-validated variants (sigma<C overlaps, sigma>C leaves gaps,
        # non-GMM has no chunk fan-out) instead of corrupting metrics silently.
        if self.obs_stride > 1:
            _ah = getattr(outer_stage, "action_head_type", None)
            if _ah != "gmm":
                raise ValueError(
                    f"obs_stride>1 currently requires action_head_type='gmm' (got {_ah!r})."
                )
            if self.obs_stride != _C:
                raise ValueError(
                    f"obs_stride ({self.obs_stride}) must equal the action chunk_len "
                    f"({_C}) so the strided action chunks tile the episode exactly."
                )

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
        # MultiDataset key topology — same surface HPT uses). Shared base
        # helper (collapse c5): see Algo._resolve_embodiment_keys.
        self._resolve_embodiment_keys(norm_stats)

    # ----- Convenience accessors so code that references the old
    # ``self.outer_stage`` / cond_encoder / hnet keeps working via
    # forwarding to the new outer_stage submodules.

    @property
    def outer_stage(self) -> HNetOuterStage:
        return self.nets["outer_stage"]

    @property
    def loss(self) -> nn.Module:
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
            # Episode-level transforms (e.g. PadHoldStill) — PRE-norm, may
            # change length (updates cu_seqlens). Train-only, packed-only.
            if (self.episode_level_transforms and self.outer_stage.training
                    and "cu_seqlens" in _batch):
                from egomimic.algo.hnet.episode_transforms import apply_episode_level_transforms
                _batch = apply_episode_level_transforms(_batch, self.episode_level_transforms)
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

    # _build_obs is inherited from Algo (collapse c5 — byte-identical).

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
            cu_for_ctx = _batch["cu_seqlens"] if is_packed else None
            max_for_ctx = int(_batch["max_seq_len"]) if is_packed else None
            # obs_stride: decimate the obs-token stream (and align the GMM
            # targets) so the chunker's RoutingModule sees tokens ``obs_stride``
            # frames apart. See PackedAlgoBase.__init__ note. stride==1 leaves
            # the original per-frame path untouched.
            if is_packed and self.obs_stride > 1:
                # For action_head_type='gmm', the head IS outer_stage.action_out
                # (no separate .gmm_head attr); chunk_len lives there.
                head = getattr(outer_stage, "action_out", None)
                # Per-embodiment ModuleDict heads share chunk_len; index in so we
                # don't read chunk_len off the ModuleDict itself (=> 1), which
                # would build 1-action targets instead of C-action chunks.
                if isinstance(head, nn.ModuleDict):
                    head = next(iter(head.values()))
                C = int(getattr(head, "chunk_len", 1))
                sobs, scu, smax, chunk_targets, sidx, _ = _stride_packed(
                    obs, actions, _batch["cu_seqlens"], self.obs_stride, C
                )
                new_batch["actions"] = actions[sidx]  # strided, for shape only
                new_batch["__obs"] = sobs
                new_batch["chunk_targets"] = chunk_targets  # FULL-res targets
                cu_for_ctx = scu
                max_for_ctx = smax
            ctx = HNetContext(
                cond_dict={},
                aux=[],
                inference_params=None,
                cu_seqlens=cu_for_ctx,
                max_seqlen=max_for_ctx,
                embodiment_id=self.domain_by_id.get(emb_id),
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

            # MoE load-balance aux (0 for dense models) + routing diagnostics.
            moe_aux = getattr(ctx, "moe_aux_loss", None)
            if moe_aux is not None:
                predictions[f"{emb_id}_moe_aux_loss"] = moe_aux
                moe_entries = getattr(ctx, "moe_entries", None) or []
                if moe_entries:
                    fracs = torch.stack([e["expert_frac"] for e in moe_entries])
                    # min/max per-expert routing fraction across all MoE layers
                    # (balanced == 1/num_experts; collapse -> max toward 1, min -> 0)
                    predictions[f"{emb_id}_moe_frac_min"] = fracs.min().to(mse.device)
                    predictions[f"{emb_id}_moe_frac_max"] = fracs.max().to(mse.device)
                    predictions[f"{emb_id}_moe_gate_entropy"] = (
                        torch.stack([e["gate_entropy"] for e in moe_entries])
                        .mean()
                        .to(mse.device)
                    )

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
            pred, _ = policy.forward_padded(
                actions, obs, embodiment_id=self.domain_by_id.get(emb_id)
            )
            preds = OrderedDict()
            preds[ac_key] = pred
            unnorm_actions = self.norm_stats.unnormalize(preds, emb_id)
            for key, val in unnorm_actions.items():
                unnorm[f"emb{emb_id}_{key}"] = val
        return unnorm

    @torch.no_grad()
    @torch.no_grad()
    def collect_chunkviz(self, batch):
        """Consistent chunkviz interface (user rule 2026-07-04).

        ONE teacher-forced packed forward per embodiment; every chunker level
        publishes its ``bpred`` (+ ``viz_tokens``: the canonical per-chunk
        representation — the AGNOSTIC (A) chunk tokens on dual-stream levels,
        the pre-lift chunk tokens on single-stream levels) into ``ctx.aux``.
        Exporters/probes consume THIS instead of module-type hooks.

        Returns ``{emb_id: {"levels": [(prob_packed, mask_packed), ...],
                            "tokens": float32 ndarray | None}}`` where
        ``tokens`` is the TOP (fewest-chunks) level's viz_tokens.
        """
        out = {}
        for emb_id, _batch in batch.items():
            if not _batch.get("_packed", False):
                continue
            ac_key = self.resolved_ac_keys[emb_id]
            obs = self._build_obs(_batch, emb_id)
            ctx = HNetContext(
                cond_dict={},
                aux=[],
                inference_params=None,
                cu_seqlens=_batch["cu_seqlens"],
                max_seqlen=int(_batch["max_seq_len"]),
                embodiment_id=self.domain_by_id.get(emb_id),
            )
            self.outer_stage({"actions": _batch[ac_key], "__obs": obs}, ctx)
            levels, best = [], None
            for entry in ctx.aux:
                bp = entry.get("bpred") if isinstance(entry, dict) else None
                if bp is None:
                    continue
                levels.append(
                    (
                        bp.boundary_prob[..., 1].detach().float().cpu(),
                        bp.boundary_mask.detach().cpu().to(torch.bool),
                    )
                )
                vt = entry.get("viz_tokens")
                if vt is not None and (best is None or vt.shape[0] < best.shape[0]):
                    best = vt
            out[emb_id] = {
                "levels": levels,
                "tokens": (
                    None if best is None
                    else best.detach().float().cpu().numpy()
                ),
            }
        return out

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
        if self.obs_stride > 1:
            # obs_stride eval: run the H-Net on the strided token stream, decode
            # the FULL chunk per kept token, and tile each chunk back over
            # [f:f+C] to reconstruct a per-frame (T_total, action_dim) prediction
            # aligned with the original GT. paired-MSE stays like-for-like and
            # the downstream metric/viz code is untouched. (Mirrors how the sim
            # inference_step rolls out: re-observe every chunk_len frames.)
            head = getattr(policy, "action_out", None)
            # Resolve the per-embodiment head (action_out is a ModuleDict in the
            # dual-stream); reading chunk_len/decode off the ModuleDict gives
            # chunk_len=1 and no .decode, so raw GMM params (C*M*(2D+1)) get
            # written into action slots -> shape mismatch [.,640] vs [.,2].
            if isinstance(head, nn.ModuleDict):
                head = head[self.domain_by_id.get(emb_id)]
            C = int(getattr(head, "chunk_len", 1))
            sobs, scu, smax, _, sidx, ep_end = _stride_packed(
                obs, actions, cu, self.obs_stride, C
            )
            new_batch = {"actions": actions[sidx], "__obs": sobs}
            ctx = HNetContext(
                cond_dict={},
                aux=[],
                inference_params=None,
                cu_seqlens=scu,
                max_seqlen=smax,
                embodiment_id=self.domain_by_id.get(emb_id),
            )
            policy(new_batch, ctx)
            raw = new_batch["pred_action"]  # (T_strided, C*P)
            if hasattr(head, "decode"):
                dec = head.decode(raw)
                chunk = dec if dec.dim() == 3 else dec.unsqueeze(-2)  # (Ts, C, D)
            else:
                chunk = raw.unsqueeze(-2)  # point head, C==1
            dev = actions.device
            T_total = int(actions.shape[0])
            pred_packed = actions.new_zeros((T_total, policy.action_dim))
            fill = sidx[:, None] + torch.arange(C, device=dev)[None, :]  # (Ts, C)
            valid = fill < ep_end[:, None]
            pred_packed[fill.clamp(max=T_total - 1)[valid]] = chunk[valid].to(
                pred_packed.dtype
            )
        else:
            pred_packed, _ = policy.forward_packed(
                actions, obs, cu, max_seqlen, embodiment_id=self.domain_by_id.get(emb_id)
            )

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

            # MoE load-balance aux (present only for MoE models; absent =>
            # dense model, byte-identical). Baked weight already applied in
            # MoEFFN, so add it straight onto the backprop total.
            m = predictions.get(f"{emb_id}_moe_aux_loss", None)
            if m is not None:
                loss_dict[f"emb{emb_id}_moe_aux_loss"] = m
                total = total + m

            # Pass non-loss stats through to logging (boundary_rate /
            # avg_chunk_len, per-chunker and aggregate). They are 0-dim
            # tensors so log_info.item() still works.
            for key, value in predictions.items():
                prefix = f"{emb_id}_"
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix) :]
                if tail in ("pred", "action_loss", "ratio_loss", "moe_aux_loss"):
                    continue
                loss_dict[f"emb{emb_id}_{tail}"] = value
        loss_dict["action_loss"] = total / max(len(batch), 1)
        return loss_dict

    # log_info is inherited from Algo (collapse c5 — byte-identical default).

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
        is_gmm = getattr(policy, "action_head_type", None) == "gmm"
        if t == 0:
            device = next(self.outer_stage.parameters()).device
            default_T = int(getattr(policy, "action_horizon", 1024))
            T_max_use = int(T_max) if T_max is not None else default_T
            import torch

            # Run the AR rollout in the MODEL's own dtype (fp32 when loaded
            # .float(), like training + teacher-forced overlay + txar's sim),
            # NOT a hardcoded bf16. The old bf16 hardcode made the closed-loop
            # sim a strictly lower-precision model than training — bf16 error in
            # the routing cos_sim / DeChunk EMA carry compounded over the rollout
            # and uniquely degraded the H-Net (txar's init_step_state defaults to
            # fp32, hence "txar works, H-Net doesn't"). Match training precision.
            self._sim_state = policy.init_step_state(
                batch_size=1, T_max=T_max_use, device=device,
                dtype=next(policy.parameters()).dtype,
            )
            # Open-loop action queue (mirrors WindowedBCPolicy.step): on an
            # obs-step the model emits ``chunk_len`` actions that REPLACE the
            # queue; intermediate frames pop one action with NO model call, so
            # the model re-observes only every ``chunk_len`` frames
            # (== obs_stride). chunk_len==1 / non-GMM => queue holds one action,
            # i.e. re-predict every frame (the prior behaviour).
            self._sim_action_queue: list = []
        embodiment_name = get_embodiment(emb_id).lower()
        ac_key = (
            self.ac_keys[embodiment_name]
            if embodiment_name in self.ac_keys
            else self.ac_keys[emb_id]
        )

        # OPEN-LOOP DISPENSE: queue non-empty => pop the next chunk action and
        # step the env WITHOUT querying the model (and without advancing the AR
        # cache / re-observing).
        if self._sim_action_queue:
            a_norm = self._sim_action_queue.pop(0)
        else:
            # Queue empty => OBS-STEP: re-observe, advance the AR cache one
            # token, decode a fresh chunk, refill the queue.
            obs_norm = self.norm_stats.normalize(obs_zarr, emb_id)
            raw = policy.step(
                self._sim_state, obs_norm, t,
                embodiment_id=self.domain_by_id.get(emb_id),
            )  # (B,1,*)
            if is_gmm:
                _ah = (policy.action_out[self.domain_by_id.get(emb_id)]
                       if isinstance(policy.action_out, nn.ModuleDict) else policy.action_out)
                chunk = _ah.decode(raw)  # (B,1,C,D) if C>1 else (B,1,D)
                C = int(getattr(_ah, "chunk_len", 1))
                if C > 1:
                    # (B,1,C,D) -> C tensors (B,1,D), one per chunk position.
                    # RE-PLAN CADENCE = the ARCHITECTURE's obs_stride. A model
                    # trained at obs_stride=sigma re-observes every sigma frames
                    # at inference: keep ``sigma`` chunk actions (capped at C),
                    # then the queue empties and the model re-observes. This ties
                    # the train and sim cadence to ONE config knob so they can't
                    # desync (obs_stride=1 => open-loop-1, re-observe every frame;
                    # obs_stride=C => open-loop-C). The sim harness stays fully
                    # arch-agnostic — it just calls inference_step every frame.
                    n_keep = max(1, min(int(getattr(self, "obs_stride", 1)), C))
                    self._sim_action_queue = [chunk[..., j, :] for j in range(n_keep)]
                else:
                    self._sim_action_queue = [chunk]
            else:
                self._sim_action_queue = [raw]  # one point action
            a_norm = self._sim_action_queue.pop(0)

        action_unnorm = self.norm_stats.unnormalize(
            {ac_key: a_norm.squeeze(0).squeeze(0)},
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


# Compat alias: legacy name for PackedAlgoBase. Kept for OLD checkpoints/configs
# that referenced the ``HNet`` name (now reachable at
# ``egomimic.algo.hnet.HNet``). New code should use ``PackedAlgoBase``.
HNet = PackedAlgoBase

