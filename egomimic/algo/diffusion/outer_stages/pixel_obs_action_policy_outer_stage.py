"""``PixelObsActionPolicyDFoTOuterStage`` — pixel-space obs+action DFoT policy (no VAE).

Extends the proven no-VAE pixel video model (``PixelSpatialDFoTOuterStage``) by
carrying the ACTION as extra SPATIAL CHANNELS appended to the RGB frame:

    action (A,)  --broadcast-->  (C_a, H, W) planes        (continuous, precise)
    concat [RGB(3), action(C_a)] -> (3+C_a, H, W)           jointly diffused by DiT3D
    decode: predicted action planes --global-avg-pool--> (A,) continuous action

Design notes:
  * No VAE, no heatmap quantization — the float64 action stays a *continuous*
    regression (the broadcast target is constant across space, so global-avg-pool
    on the predicted planes recovers it; prediction error averages out).
  * Genuinely 2D: the DiT3D's spatial attention mixes action<->image at every
    patch position — the "project the action token to 2D" idea, the precise way.
  * No backbone change: the backbone just sees ``latent_channels = 3 + C_a``.
  * The action is supervised implicitly by the standard joint v-diffusion loss
    over all (3+C_a) channels — no separate head, nothing left untrained.

Config: set ``backbone.latent_channels = image_channels + action_channels`` and
``diffusion.action_dim`` to the same, and ``action_channels`` here (default =
``action_dim``). A learned conv projection (instead of broadcast) is a future
variant; broadcast is chosen first because it keeps the diffusion TARGET fixed.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from egomimic.algo.diffusion.outer_stages.pixel_spatial_outer_stage import PixelSpatialDFoTOuterStage


class PixelObsActionPolicyDFoTOuterStage(PixelSpatialDFoTOuterStage):
    def __init__(self, *args, action_channels: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
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

    @property
    def action_slice(self) -> slice:
        # action occupies the trailing C_a channels of the per-frame tensor
        return slice(self._image_channels, self._image_channels + self._action_channels)

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

    def _sample_windows_packed(self, images, actions, cu):
        """Frame-sample images AND actions IDENTICALLY per packed episode, so the
        action at frame t always lines up with image t after cropping."""
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
                img_crops.append(images[s + idx])
                act_crops.append(actions[s + idx])
                continue
            else:
                sl = slice(s, e)
            img_crops.append(images[sl])
            act_crops.append(actions[sl])
        new_cu = torch.zeros(b + 1, dtype=cu.dtype, device=cu.device)
        for i, c in enumerate(img_crops):
            new_cu[i + 1] = new_cu[i] + c.shape[0]
        return torch.cat(img_crops, 0), torch.cat(act_crops, 0), new_cu

    # ------------------------------------------------------------------ #
    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
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
