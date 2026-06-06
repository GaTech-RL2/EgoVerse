"""``PixelObsActionRegressPolicyDFoTOuterStage`` — Design B: pixel video DF +
a conv action-regression head reading the model's PREDICTED clean frame.

The DiT3D diffuses ONLY the RGB video (``latent_channels=3``, the proven
foundation). A conv head reads the model's predicted clean frame (x0,
reconstructed from the v-prediction via ``predict_start_from_v``) and regresses
the continuous action with an MSE loss. The action is **NOT** a diffusion
target — it's a deterministic regression off the predicted frame ("the
predicted image already contains the pusher's next position"). Cut-action: no
past-action input, so it cannot copy a_{t-1}.

Loss = video v-diffusion loss + ``action_loss_weight`` * MSE(action_pred, gt).
Contrast to Design A (action as broadcast channels, jointly diffused, pooled).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import PixelSpatialDFoTOuterStage


class PixelObsActionRegressPolicyDFoTOuterStage(PixelSpatialDFoTOuterStage):
    def __init__(self, *args, action_loss_weight: float = 1.0, head_width: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
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

    @property
    def action_slice(self) -> slice:
        return slice(0, 0)  # action is a regression head, not part of the diffused bundle

    def _sample_windows_packed(self, images, actions, cu):
        """Frame-sample images AND actions identically per packed episode."""
        b = cu.shape[0] - 1
        n = self._sample_n_frames
        mode = self._frame_sampling
        img_crops, act_crops = [], []
        for i in range(b):
            s, e = int(cu[i].item()), int(cu[i + 1].item())
            L = e - s
            if mode == "fixed_window" and L > n:
                st = int(torch.randint(0, L - n + 1, (1,)).item())
                sl = slice(s + st, s + st + n)
            elif mode == "start_to_end" and L > n:
                st = int(torch.randint(0, L - n + 1, (1,)).item())
                sl = slice(s + st, e)
            elif mode == "random_subsample" and L > n:
                idx = torch.randperm(L)[:n].sort().values
                img_crops.append(images[s + idx]); act_crops.append(actions[s + idx]); continue
            else:
                sl = slice(s, e)
            img_crops.append(images[sl]); act_crops.append(actions[sl])
        new_cu = torch.zeros(b + 1, dtype=cu.dtype, device=cu.device)
        for i, c in enumerate(img_crops):
            new_cu[i + 1] = new_cu[i] + c.shape[0]
        return torch.cat(img_crops, 0), torch.cat(act_crops, 0), new_cu

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
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

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
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
