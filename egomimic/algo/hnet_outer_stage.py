"""``HNetOuterStage`` — OuterStage subclass for the stage-based H-Net.

Same responsibilities as the previous ``HNetPolicy`` (which it replaces):
input_modules (per-token summed contributions), cond_encoder (obs → AdaLN
cond_dict), inner_stage = HNetCore (the stage tree), action_head decoding.
Just reshaped into the OuterStage encode → inner_stage → decode contract.

Three forward paths on the class — ``forward_padded``, ``forward_packed``,
``step`` — mirror the prior ``HNetPolicy.forward``, ``forward_packed``,
``step``. The OuterStage base's ``forward(batch, ctx)`` dispatches between
padded/packed via ``ctx.is_packed``. ``step`` (online single-tick AR) stays
as an explicit method that the algo's inference path calls directly.

DIFFERENCES VS HNetPolicy:
- No more `_remap_legacy_input_keys` hook. Old checkpoints WILL NOT load
  (clean break per project policy).
- ``forward`` is now ``(batch, ctx) -> Tensor`` instead of ``(actions, obs)``.
  The HNet algo class builds the batch dict + ctx before calling.
- Pred actions are written to ``batch["pred_action"]`` so ``HNetLoss`` can
  read them; ``ctx.aux`` accumulates the chunker-stage aux for the
  boundary regularizer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from egomimic.algo.input_modules import ActionInToken, InputModule
from egomimic.algo.outer_stage import OuterStage
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.hnet import HNet as HNetCore


class HNetOuterStage(OuterStage):
    """Stage-based H-Net outer stage.

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
    ):
        super().__init__(inner_stage=hnet)
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
        else:
            raise ValueError(
                f"action_head_type must be 'linear' or 'mlp', got {action_head_type!r}"
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
            )
            x = contrib if x is None else x + contrib
        if x is None:
            raise RuntimeError("input_modules produced no tokens")
        # Build cond_dict from per-frame obs and stuff onto ctx so the
        # inner stages (which read it via cond_key) get it.
        cond_dict = self.cond_encoder.encode(obs, self.action_horizon)
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
            )
            x_packed = contrib if x_packed is None else x_packed + contrib
        if x_packed is None:
            raise RuntimeError("input_modules produced no tokens")

        # Packed cond: fake batch=1 by unsqueezing each obs, then squeeze
        # the leading dim back out.
        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_padded = self.cond_encoder.encode(obs_for_encode, T_action=T_total)
        cond_packed = {k: v.squeeze(0) for k, v in cond_padded.items()}
        ctx.cond_dict = cond_packed
        return x_packed

    def decode(self, h: torch.Tensor, batch: dict, ctx: HNetContext) -> None:
        """Apply the action head and write ``batch["pred_action"]``."""
        batch["pred_action"] = self.action_out(h)

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

    def forward_padded(self, actions: torch.Tensor, obs: dict):
        """Legacy bridge: padded (actions, obs) -> (pred, aux)."""
        batch = {"actions": actions, "__obs": obs}
        ctx = HNetContext(cond_dict={}, aux=[], inference_params=None)
        self.forward(batch, ctx)
        return batch["pred_action"], ctx.aux

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Legacy bridge: packed (...) -> (pred, aux)."""
        batch = {"actions": actions_packed, "__obs": obs_packed}
        ctx = HNetContext(
            cond_dict={},
            aux=[],
            inference_params=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=int(max_seqlen),
        )
        self.forward(batch, ctx)
        return batch["pred_action"], ctx.aux

    # ------------------------------------------------------------------
    # Inference: AR generate + step. Used by the algo class's closed-loop
    # path; not part of the OuterStage abstract API.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        """Offline AR rollout from BOS for ``T`` steps. Returns
        ``(batch_size, T, action_dim)``. ``T`` defaults to
        ``self.action_horizon`` when not given."""
        if T is None:
            T = self.action_horizon
        cond_dict = self.cond_encoder.encode(obs, T)
        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        dtype = next(self.parameters()).dtype

        inference_params = self.inner_stage.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=T,
            device=device,
            dtype=dtype,
        )

        def slice_cond(t: int) -> dict:
            return {k: v[:, t] if v.dim() == 3 else v for k, v in cond_dict.items()}

        prev_a = None

        def _step_obs(t: int) -> dict:
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
            h = self.inner_stage.step(cur, ctx)
            a_t = self.action_out(h)
            actions[:, t : t + 1] = a_t
            prev_a = a_t
        return actions

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        """Allocate the AR inference state. Returns an opaque dict for
        :meth:`step`."""
        T_max = int(T_max)
        dtype = dtype or next(self.parameters()).dtype
        params = self.inner_stage.allocate_inference_cache(
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

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Single online AR step. Returns ``(B, 1, action_dim)``."""
        cond_dict_seq = self.cond_encoder.encode(obs_norm, T_action=1)
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
            )
            cur = contrib if cur is None else cur + contrib
        if cur is None:
            raise RuntimeError("input_modules produced no tokens at step")

        ctx = HNetContext(
            cond_dict=cond_2d,
            aux=[],
            inference_params=state["params"],
        )
        h = self.inner_stage.step(cur, ctx)
        a_t_norm = self.action_out(h)
        state["prev_action"] = a_t_norm
        return a_t_norm
