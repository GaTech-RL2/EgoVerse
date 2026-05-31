"""``SpatialObsActionPolicyDFoTOuterStage`` — 2D POLICY variant of obs+action
diffusion forcing.

Like ``ImageSpatialDFoTOuterStage`` (the 2D world model), the image stays a
SPATIAL VAE latent (NOT flattened) and is the spatial diffusion target. The
difference: the ACTION is ALSO a diffusion target, carried as one per-frame
"action token" appended to the DiT3D token sequence (see
``DFoTDiT3DBackbone(action_token_dim=...)``). State enters as ``external_cond``.

The backbone returns ``(v_image, v_action)``; this stage computes a structured
loss ``image_v_mse + action_loss_weight * action_v_mse`` and stashes it on
``ctx.precomputed_loss`` (``DFoTLoss`` returns it verbatim). This keeps the
structured (image, action) target out of the scalar 1D v-MSE path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from egomimic.algo.dfot.image_spatial_outer_stage import ImageSpatialDFoTOuterStage


class SpatialObsActionPolicyDFoTOuterStage(ImageSpatialDFoTOuterStage):
    def __init__(self, *args, action_loss_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_loss_weight = float(action_loss_weight)

        # State-only conditioning MLP. The parent's ``state_action_proj`` mixes
        # the action into the cond, but here the action is a TARGET, so we
        # condition on state alone.
        self.state_only_proj = nn.Sequential(
            nn.Linear(self._state_dim_total, self._state_action_proj_dim),
            nn.SiLU(),
            nn.Linear(self._state_action_proj_dim, self._state_action_proj_dim),
        )

        bb_at = int(getattr(self.inner_stage, "action_token_dim", 0))
        if bb_at != self._action_dim:
            raise ValueError(
                f"backbone.action_token_dim ({bb_at}) must equal action_dim "
                f"({self._action_dim}) for the 2D policy variant."
            )

    @property
    def action_slice(self) -> slice:
        # Action is predicted as a separate backbone output, not sliced from a
        # bundle — empty slice signals "N/A" to bundle-based extractors.
        return slice(0, 0)

    def _state_to_cond(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        pieces = [ctx.obs[k] for k in self.bundle_obs_keys]
        concat = torch.cat(pieces, dim=-1)
        return self.state_only_proj(concat)

    def encode(self, batch: dict, ctx: SimpleNamespace):
        latent = self._build_latent(ctx)        # (T_total,C,H,W) | (B,T,C,H,W)
        action = batch[ctx.action_key]          # (T_total,A)     | (B,T,A)
        state_cond = self._state_to_cond(batch, ctx)

        if ctx.is_packed:
            t = self._sample_noise_levels((latent.shape[0],), latent.device)
        else:
            t = self._sample_noise_levels((latent.shape[0], latent.shape[1]), latent.device)

        ctx.q_state = self._qstate(latent, t)
        ctx.q_action = self._qstate(action, t)   # same per-frame noise level
        ctx.external_cond = state_cond
        return ctx.q_state["x_t"], ctx.q_action["x_t"]

    def _qstate(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """Noise ``x`` at level ``t`` and return a q_state dict. Discrete
        diffusion's ``q_sample`` returns a bare tensor, so we supply explicit
        noise and assemble the dict ourselves (matching ImageSpatial)."""
        from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion as _DD

        if isinstance(self.diffusion, _DD):
            noise = torch.randn_like(x).clamp_(
                -self.diffusion.clip_noise, self.diffusion.clip_noise
            )
            x_t = self.diffusion.q_sample(x, t, noise=noise)
            return {"x_t": x_t, "k": t, "time_cond": t, "noise": noise, "x_start": x}
        return self.diffusion.q_sample(x, t)

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        x_t_latent, x_t_action = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]

        if not ctx.is_packed:
            v_latent, v_action = self.inner_stage(
                x_t_latent, time_cond,
                external_cond=ctx.external_cond, action=x_t_action,
            )
        else:
            # DiT3D has no packed mode -> run each episode as a batch-of-1.
            cu = ctx.cu_seqlens
            B = cu.shape[0] - 1
            vlat, vact = [], []
            for i in range(B):
                s, e = int(cu[i].item()), int(cu[i + 1].item())
                vl, va = self.inner_stage(
                    x_t_latent[s:e].unsqueeze(0),
                    time_cond[s:e].unsqueeze(0),
                    external_cond=ctx.external_cond[s:e].unsqueeze(0),
                    action=x_t_action[s:e].unsqueeze(0),
                )
                vlat.append(vl.squeeze(0))
                vact.append(va.squeeze(0))
            v_latent = torch.cat(vlat, dim=0)
            v_action = torch.cat(vact, dim=0)

        latent_loss = self.diffusion.compute_loss(v_latent, ctx.q_state).mean()
        action_loss = self.diffusion.compute_loss(v_action, ctx.q_action).mean()
        ctx.precomputed_loss = latent_loss + self.action_loss_weight * action_loss
        batch["pred_v"] = v_latent
        return v_latent
