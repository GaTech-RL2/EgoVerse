"""``DFoTOuterStage`` — outermost stage for the DFoT algorithm.

Owns:
- ``cond_encoder``: CondEncoderModule that turns obs into per-token AdaLN cond.
- ``backbone``: DFoTBackbone (Isotropic trunk + per-token AdaLN noise embedding).
  Stored as ``self.inner_stage`` per the OuterStage convention.
- ``diffusion``: ContinuousDiffusion (or DiscreteDiffusion) — used to call
  ``q_sample`` at training time. Has no learnable params here; lives on the
  outer stage so the loss class can call ``compute_loss`` symmetrically.

The training-time ``forward`` does:
  encode -> q_sample -> backbone -> decode (writes batch['pred_v'])

The DFoTLoss class (in egomimic/algo/loss.py) then reads ctx.q_state +
batch['pred_v'] and produces the scalar SNR-weighted epsilon-MSE.

Inference paths (closed-loop AR sample_step, chunk-mode plan-and-execute)
live on the algo class for now — they use this outer stage's cond_encoder
and backbone directly without going through ``forward``. A later refactor
can fold them in once the training-path refactor is verified.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from egomimic.algo.dfot.backbone import DFoTBackbone
from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.outer_stage import OuterStage
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule


def make_dfot_ctx(
    *,
    is_packed: bool,
    action_key: str,
    obs: dict,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> SimpleNamespace:
    """Build a minimal DFoT context object. Fields are filled in by the
    outer stage during ``encode`` (q_state, external_cond) and consumed by
    the loss class. ``cu_seqlens``/``max_seqlen`` carry packing info."""
    return SimpleNamespace(
        is_packed=is_packed,
        action_key=action_key,
        obs=obs,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        q_state=None,
        external_cond=None,
    )


class DFoTOuterStage(OuterStage):
    """Training-time outer stage for DFoT.

    Subclass-specific contract:

    * ``encode(batch, ctx)``: reads ``batch[ctx.action_key]`` (clean actions),
      encodes obs into per-token cond via ``self.cond_encoder``, samples
      per-token noise levels, runs ``diffusion.q_sample`` to get noisy
      tokens. Stores the full ``q_state`` and the encoded cond on
      ``ctx`` for the loss to read later. Returns the noisy actions
      (``x_t``) for the inner_stage (backbone).

    * ``decode(v_pred, batch, ctx)``: writes ``batch['pred_v']`` so the
      loss can pick it up. (For inference paths the algo class converts
      v -> action separately via the sampler step formulas.)

    * ``forward(batch, ctx)``: overrides the default OuterStage flow to
      thread ``cu_seqlens`` / ``max_seqlen`` / ``time_cond`` / ``external_cond``
      into the backbone, which has a richer signature than a plain
      ``inner_stage(x, ctx) -> x``.
    """

    def __init__(
        self,
        action_dim: int,
        cond_encoder: CondEncoderModule,
        backbone: DFoTBackbone,
        diffusion: nn.Module,  # ContinuousDiffusion or DiscreteDiffusion
        cond_output_key: str = "fused_cond",
    ):
        super().__init__(inner_stage=backbone)
        self.action_dim = int(action_dim)
        self.cond_encoder = cond_encoder
        self.diffusion = diffusion
        self.cond_output_key = cond_output_key

    @property
    def bundle_dim(self) -> int:
        """Width of the diffused tensor. For vanilla DFoT this equals
        ``action_dim``; subclasses that diffuse extra modalities (obs+action
        joint, etc.) override to return the full bundle width."""
        return self.action_dim

    @property
    def action_slice(self) -> slice:
        """Slice into the trailing dim of the sampled bundle that
        corresponds to actions. Vanilla DFoT is action-only, so this is
        the full slice. Subclasses (e.g. obs+action joint) override to
        point at just the action portion of their bundle."""
        return slice(0, self.action_dim)

    # -------------------------------------------------------------------
    # Per-mode cond encode helpers (lifted from algo.py to keep modality
    # encoding on the outer stage where it belongs).
    # -------------------------------------------------------------------

    def _encode_cond_padded(self, obs: dict, T: int) -> Optional[torch.Tensor]:
        cond_dict = self.cond_encoder.encode(obs, T)
        return cond_dict.get(self.cond_output_key)

    def _encode_cond_packed(self, obs: dict) -> Optional[torch.Tensor]:
        obs_3d = {
            k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in obs.items()
        }
        cond_dict = self.cond_encoder.encode(obs_3d, T_action=1)
        c = cond_dict.get(self.cond_output_key)
        if c is None:
            return None
        if c.dim() == 3 and c.shape[0] == 1:
            c = c.squeeze(0)
        return c

    # -------------------------------------------------------------------
    # Noise-level sampling. Mirrors DFoT._sample_noise_levels exactly so
    # the refactor preserves training-time noise distribution.
    # -------------------------------------------------------------------

    def _sample_noise_levels(self, shape, device) -> torch.Tensor:
        if isinstance(self.diffusion, DiscreteDiffusion):
            return torch.randint(
                0, self.diffusion.timesteps, shape, device=device, dtype=torch.long
            )
        return torch.rand(shape, device=device).clamp_(1e-5, 1.0 - 1e-5)

    # -------------------------------------------------------------------
    # OuterStage API.
    # -------------------------------------------------------------------

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        actions = batch[ctx.action_key]

        if ctx.is_packed:
            if actions.dim() != 2:
                raise ValueError(
                    f"packed actions must be (T_total, action_dim); "
                    f"got {tuple(actions.shape)}"
                )
            T_total = actions.shape[0]
            cond = self._encode_cond_packed(ctx.obs)
            t = self._sample_noise_levels((T_total,), actions.device)
        else:
            if actions.dim() != 3:
                raise ValueError(
                    f"padded actions must be (B, T, action_dim); "
                    f"got {tuple(actions.shape)}"
                )
            B, T, _ = actions.shape
            cond = self._encode_cond_padded(ctx.obs, T)
            t = self._sample_noise_levels((B, T), actions.device)

        q = self.diffusion.q_sample(actions, t)
        ctx.q_state = q
        ctx.external_cond = cond
        return q["x_t"]

    def decode(self, v_pred: torch.Tensor, batch: dict, ctx: SimpleNamespace) -> None:
        # The loss reads ``batch['pred_v']`` and combines with ``ctx.q_state``.
        batch["pred_v"] = v_pred

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        """Override the default OuterStage.forward to thread the backbone's
        extra kwargs (time_cond from q_state, external_cond, cu_seqlens,
        max_seqlen for packed) through the inner_stage call."""
        x_t = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]
        if ctx.is_packed:
            v_pred = self.inner_stage(
                x_t,
                time_cond,
                external_cond=ctx.external_cond,
                cu_seqlens=ctx.cu_seqlens,
                max_seqlen=ctx.max_seqlen,
            )
        else:
            v_pred = self.inner_stage(
                x_t,
                time_cond,
                external_cond=ctx.external_cond,
            )
        self.decode(v_pred, batch, ctx)
        return v_pred
