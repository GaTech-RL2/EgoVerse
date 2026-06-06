"""``PixelObsActionDecoupledDFoTOuterStage`` — pixel-space obs+action DFoT policy
with the action as a SEPARATE decoupled-noise token (no VAE, no fused channels).

Unlike ``PixelObsActionPolicyDFoTOuterStage`` (action broadcast into obs CHANNELS,
one noise level per frame -> at inference you must pin the whole context frame
clean and read predictions from frame 1+ -> a 1-frame execution offset), here the
action rides as its OWN DiT3D token with an INDEPENDENT per-frame noise level
(backbone ``action_token_dim``). At inference obs_t is pinned clean while a_t is
denoised in the SAME frame -> predicts a_t from obs_t with NO offset, and the
action is never a clean input (cannot copy a_{t-1}).

Mirrors ``SpatialObsActionPolicyDFoTOuterStage`` (the VAE variant) but swaps the
VAE-latent obs for raw pixels via the proven ``PixelSpatialDFoTOuterStage`` base.

Backbone must be a ``DFoTDiT3DBackbone(action_token_dim=action_dim)`` returning
``(v_image, v_action)``. Use ``frame_sampling="full"`` with pre-windowed data
(e.g. max_seq_len=33); joint image+action cropping is not done here.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import PixelSpatialDFoTOuterStage


class PixelObsActionDecoupledDFoTOuterStage(PixelSpatialDFoTOuterStage):
    def __init__(self, *args, action_loss_weight: float = 1.0,
                 decouple_action_noise: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_loss_weight = float(action_loss_weight)
        self.decouple_action_noise = bool(decouple_action_noise)
        bb_at = int(getattr(self.inner_stage, "action_token_dim", 0))
        if bb_at != self.action_dim:
            raise ValueError(
                f"backbone.action_token_dim ({bb_at}) must equal action_dim "
                f"({self.action_dim}) for the decoupled pixel policy."
            )

    @property
    def action_slice(self) -> slice:
        # action is a separate backbone output, not sliced from the bundle
        return slice(0, 0)

    def _qstate(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        noise = torch.randn_like(x).clamp_(
            -self.diffusion.clip_noise, self.diffusion.clip_noise
        )
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        return {"x_t": x_t, "k": t, "time_cond": t, "noise": noise, "x_start": x}

    def encode(self, batch: dict, ctx: SimpleNamespace):
        images = self._extract_images(ctx)
        ac_key = getattr(ctx, "action_key", None)
        if ac_key is None or ac_key not in batch:
            raise KeyError(
                f"decoupled pixel policy needs the action key in the batch "
                f"(ctx.action_key={ac_key!r}); keys: {list(batch.keys())}"
            )
        actions = batch[ac_key].to(images.device).float()

        if ctx.is_packed:
            t = self._sample_noise_levels((images.shape[0],), images.device)
        else:
            t = self._sample_noise_levels((images.shape[0], images.shape[1]), images.device)

        ctx.q_state = self._qstate(images, t)
        t_a = (self._sample_noise_levels(tuple(t.shape), images.device)
               if self.decouple_action_noise else t)
        ctx.q_action = self._qstate(actions, t_a)
        ctx.external_cond = None
        ctx.latent_clean = images
        return ctx.q_state["x_t"], ctx.q_action["x_t"]

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        x_t, x_t_a = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]
        a_levels = ctx.q_action["time_cond"] if self.decouple_action_noise else None

        if not ctx.is_packed:
            v_img, v_act = self.inner_stage(
                x_t, time_cond, external_cond=None,
                action=x_t_a, action_noise_levels=a_levels,
            )
        else:
            cu = ctx.cu_seqlens
            B = cu.shape[0] - 1
            vi, va = [], []
            for i in range(B):
                s, e = int(cu[i].item()), int(cu[i + 1].item())
                vimg, vact = self.inner_stage(
                    x_t[s:e].unsqueeze(0), time_cond[s:e].unsqueeze(0),
                    external_cond=None,
                    action=x_t_a[s:e].unsqueeze(0),
                    action_noise_levels=(
                        a_levels[s:e].unsqueeze(0) if a_levels is not None else None
                    ),
                )
                vi.append(vimg.squeeze(0))
                va.append(vact.squeeze(0))
            v_img = torch.cat(vi, dim=0)
            v_act = torch.cat(va, dim=0)

        img_loss = self.diffusion.compute_loss(v_img, ctx.q_state).mean()
        act_loss = self.diffusion.compute_loss(v_act, ctx.q_action).mean()
        ctx.precomputed_loss = img_loss + self.action_loss_weight * act_loss
        batch["pred_v"] = v_img
        return v_img
