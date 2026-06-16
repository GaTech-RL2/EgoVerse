"""``PixelObsActionDFoTOuterStage`` — UNIFIED pixel-space obs+action DFoT policy.

This single parameterized stage subsumes the three former near-duplicate
pixel-policy outer stages, selected by the ``pixel_mode`` knob:

  * ``pixel_mode="policy"``   (Design A) — action broadcast into RGB CHANNELS,
    jointly diffused by the DiT3D (bundle = 3 RGB + C_a action planes); decode by
    global-avg-pool of the predicted action planes. Was
    ``PixelObsActionPolicyDFoTOuterStage``.
  * ``pixel_mode="regress"``  (Design B) — diffuse ONLY the RGB video
    (latent_channels=3); a conv ``action_head`` regresses the action from the
    model's predicted clean frame (x0). Action is NOT a diffusion target. Was
    ``PixelObsActionRegressPolicyDFoTOuterStage``.
  * ``pixel_mode="decoupled"`` (DEC) — action rides as its OWN DiT3D token with an
    INDEPENDENT per-frame noise level (backbone ``action_token_dim``); backbone
    returns ``(v_image, v_action)``. Was ``PixelObsActionDecoupledDFoTOuterStage``.

All three subclass the proven no-VAE pixel video model
(``PixelSpatialDFoTOuterStage``). Each mode reproduces the corresponding former
class EXACTLY — identical construction (state_dict), identical forward outputs,
identical duck-typed attribute surface consumed by the algo inference paths
(``_action_channels`` for policy, ``action_head`` for regress,
``decouple_action_noise`` for decoupled, plus the mode-correct ``action_slice``).

Per-mode kwargs (all accepted; only the selected mode's are used):
  * policy:    ``action_channels`` (default = ``action_dim``)
  * regress:   ``action_loss_weight`` (default 1.0), ``head_width`` (default 64)
  * decoupled: ``action_loss_weight`` (default 1.0),
               ``decouple_action_noise`` (default True)
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import PixelSpatialDFoTOuterStage


_PIXEL_MODES = ("policy", "regress", "decoupled")


class PixelObsActionDFoTOuterStage(PixelSpatialDFoTOuterStage):
    def __init__(
        self,
        *args,
        pixel_mode: str = "policy",
        # --- policy (Design A) ---
        action_channels: int | None = None,
        # --- regress (Design B) ---
        head_width: int = 64,
        # --- regress + decoupled ---
        action_loss_weight: float = 1.0,
        # --- decoupled ---
        decouple_action_noise: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if pixel_mode not in _PIXEL_MODES:
            raise ValueError(
                f"pixel_mode must be one of {_PIXEL_MODES}; got {pixel_mode!r}"
            )
        self.pixel_mode = str(pixel_mode)

        if self.pixel_mode == "policy":
            Ca = int(action_channels) if action_channels is not None else self.action_dim
            if Ca < self.action_dim:
                raise ValueError(
                    f"action_channels ({Ca}) must be >= action_dim ({self.action_dim}); "
                    f"each action component broadcasts to one plane."
                )
            self._action_channels = Ca
            # bundle now = image channels + action planes (drives sampler tensor alloc)
            self._bundle_shape = (
                self._image_channels + Ca,
                self._image_size,
                self._image_size,
            )

        elif self.pixel_mode == "regress":
            self.action_loss_weight = float(action_loss_weight)
            c = self._image_channels
            # conv-down: predicted frame (C,H,W) -> action (A,)  [your "project down"]
            self.action_head = nn.Sequential(
                nn.Conv2d(c, head_width, 3, stride=2, padding=1), nn.SiLU(),            # H/2
                nn.Conv2d(head_width, head_width, 3, stride=2, padding=1), nn.SiLU(),   # H/4
                nn.Conv2d(head_width, head_width, 3, stride=2, padding=1), nn.SiLU(),   # H/8
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(head_width, self.action_dim),
            )

        else:  # decoupled
            self.action_loss_weight = float(action_loss_weight)
            self.decouple_action_noise = bool(decouple_action_noise)
            bb_at = int(getattr(self.inner_stage, "action_token_dim", 0))
            if bb_at != self.action_dim:
                raise ValueError(
                    f"backbone.action_token_dim ({bb_at}) must equal action_dim "
                    f"({self.action_dim}) for the decoupled pixel policy."
                )

    # ------------------------------------------------------------------ #
    @property
    def action_slice(self) -> slice:
        if self.pixel_mode == "policy":
            # action occupies the trailing C_a channels of the per-frame tensor
            return slice(self._image_channels, self._image_channels + self._action_channels)
        # regress + decoupled: action is a separate output, not sliced from bundle
        return slice(0, 0)

    # ------------------------------------------------------------------ #
    # policy: action -> broadcast planes
    # ------------------------------------------------------------------ #
    def _action_to_planes(self, actions: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """``(N, A) -> (N, C_a, H, W)`` by broadcasting each action component
        across the spatial plane. Continuous + precise; global-avg-pool on
        decode recovers the value with prediction error averaged out."""
        n = actions.shape[0]
        a = actions[..., : self._action_channels]
        return a.reshape(n, self._action_channels, 1, 1).expand(
            n, self._action_channels, h, w
        )

    # ------------------------------------------------------------------ #
    # joint image/action frame sampling per packed episode:
    # ``_sample_windows_packed`` now lives on the parent
    # ``PixelSpatialDFoTOuterStage`` (dedup collapse c7) — inherited here.
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # decoupled: per-tensor q_state helper
    # ------------------------------------------------------------------ #
    def _qstate(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        noise = torch.randn_like(x).clamp_(
            -self.diffusion.clip_noise, self.diffusion.clip_noise
        )
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        return {"x_t": x_t, "k": t, "time_cond": t, "noise": noise, "x_start": x}

    # ------------------------------------------------------------------ #
    # encode — dispatch on pixel_mode
    # ------------------------------------------------------------------ #
    def encode(self, batch: dict, ctx: SimpleNamespace):
        if self.pixel_mode == "policy":
            return self._encode_policy(batch, ctx)
        if self.pixel_mode == "regress":
            return self._encode_regress(batch, ctx)
        return self._encode_decoupled(batch, ctx)

    def _encode_policy(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        images = self._extract_images(ctx)  # packed (T,3,H,W) | padded (B,T,3,H,W)
        ac_key = getattr(ctx, "action_key", None)
        if ac_key is None or ac_key not in batch:
            raise KeyError(
                f"PixelObsActionPolicy needs the action key in the batch "
                f"(ctx.action_key={ac_key!r}); keys: {list(batch.keys())}"
            )
        actions = batch[ac_key].to(images.device).float()

        if ctx.is_packed:
            if self._frame_sampling != "full" and self.training:
                images, actions, new_cu = self._sample_windows_packed(
                    images, actions, ctx.cu_seqlens
                )
                ctx.cu_seqlens = new_cu
                ctx.max_seqlen = max(
                    int(new_cu[i + 1] - new_cu[i]) for i in range(new_cu.shape[0] - 1)
                )
            h, w = images.shape[-2:]
            planes = self._action_to_planes(actions, h, w)          # (T, C_a, H, W)
            x_start = torch.cat([images, planes], dim=1)            # (T, 3+C_a, H, W)
            t = self._sample_noise_levels((x_start.shape[0],), images.device)
        else:
            b, T = images.shape[:2]
            h, w = images.shape[-2:]
            planes = self._action_to_planes(
                actions.reshape(b * T, -1), h, w
            ).reshape(b, T, self._action_channels, h, w)
            x_start = torch.cat([images, planes], dim=2)            # (B,T,3+C_a,H,W)
            t = self._sample_noise_levels((b, T), images.device)

        noise = torch.randn_like(x_start).clamp_(
            -self.diffusion.clip_noise, self.diffusion.clip_noise
        )
        x_t = self.diffusion.q_sample(x_start, t, noise=noise)
        ctx.q_state = {
            "x_t": x_t,
            "k": t,
            "time_cond": t,
            "noise": noise,
            "x_start": x_start,
        }
        ctx.external_cond = None
        ctx.latent_clean = x_start
        return x_t

    def _encode_regress(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        images = self._extract_images(ctx)
        ac_key = getattr(ctx, "action_key", None)
        if ac_key is None or ac_key not in batch:
            raise KeyError(f"regress policy needs action key (ctx.action_key={ac_key!r})")
        actions = batch[ac_key].to(images.device).float()

        if ctx.is_packed:
            if self._frame_sampling != "full" and self.training:
                images, actions, new_cu = self._sample_windows_packed(images, actions, ctx.cu_seqlens)
                ctx.cu_seqlens = new_cu
                ctx.max_seqlen = max(int(new_cu[i + 1] - new_cu[i]) for i in range(new_cu.shape[0] - 1))
            t = self._sample_noise_levels((images.shape[0],), images.device)
        else:
            b, T = images.shape[:2]
            t = self._sample_noise_levels((b, T), images.device)

        noise = torch.randn_like(images).clamp_(-self.diffusion.clip_noise, self.diffusion.clip_noise)
        x_t = self.diffusion.q_sample(images, t, noise=noise)
        ctx.q_state = {"x_t": x_t, "k": t, "time_cond": t, "noise": noise, "x_start": images}
        ctx.external_cond = None
        ctx._gt_actions = actions
        return x_t

    def _encode_decoupled(self, batch: dict, ctx: SimpleNamespace):
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

    # ------------------------------------------------------------------ #
    # forward — dispatch on pixel_mode
    # ------------------------------------------------------------------ #
    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        if self.pixel_mode == "policy":
            # policy reuses the base PixelSpatial forward (encode -> backbone -> decode)
            return PixelSpatialDFoTOuterStage.forward(self, batch, ctx)
        if self.pixel_mode == "regress":
            return self._forward_regress(batch, ctx)
        return self._forward_decoupled(batch, ctx)

    def _forward_regress(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        x_t = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]
        gt_actions = ctx._gt_actions

        if not ctx.is_packed:
            v_pred = self.inner_stage(x_t, time_cond, external_cond=None)
        else:
            cu = ctx.cu_seqlens
            b = cu.shape[0] - 1
            pieces = []
            for i in range(b):
                s, e = int(cu[i].item()), int(cu[i + 1].item())
                pieces.append(self.inner_stage(x_t[s:e].unsqueeze(0), time_cond[s:e].unsqueeze(0),
                                               external_cond=None).squeeze(0))
            v_pred = torch.cat(pieces, dim=0)

        video_loss = self.diffusion.compute_loss(v_pred, ctx.q_state).mean()

        # predicted clean frame -> regress action
        pred_x0 = self.diffusion.predict_start_from_v(ctx.q_state["x_t"], ctx.q_state["k"], v_pred)
        if pred_x0.dim() == 5:
            bb, tt = pred_x0.shape[:2]
            ap = self.action_head(pred_x0.reshape(bb * tt, *pred_x0.shape[2:])).reshape(bb, tt, -1)
        else:
            ap = self.action_head(pred_x0)                       # (T, A)
        gta = gt_actions[..., : self.action_dim]
        action_loss = ((ap - gta) ** 2).mean()

        ctx.precomputed_loss = video_loss + self.action_loss_weight * action_loss
        batch["pred_v"] = v_pred
        batch["pred_action"] = ap
        return v_pred

    def _forward_decoupled(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
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
