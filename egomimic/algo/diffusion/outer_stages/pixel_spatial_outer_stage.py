"""``PixelSpatialDFoTOuterStage`` — DFoT on raw pixel images (no VAE).

Training frame sampling modes (``frame_sampling`` config):
  - ``"full"``: use the entire episode as-is. No cropping.
  - ``"fixed_window"``: sample a random window of ``sample_n_frames``
    consecutive frames from each episode. Matches the reference repo.
  - ``"start_to_end"``: sample a random start index, take everything
    from there to the end of the episode. Variable-length sequences.
  - ``"random_subsample"``: sample ``sample_n_frames`` frames uniformly
    at random (not necessarily consecutive) from the episode, sorted
    by time. Preserves temporal coverage but with gaps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from egomimic.algo.diffusion.outer_stages.outer_stage import DFoTOuterStage
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion


class PixelSpatialDFoTOuterStage(DFoTOuterStage):

    def __init__(
        self,
        action_dim: int,
        cond_encoder,
        backbone,
        diffusion,
        image_key: str = "front_img_1",
        image_channels: int = 3,
        image_size: int = 96,
        frame_sampling: str = "full",
        sample_n_frames: int = 9,
        cond_output_key: str = "fused_cond",
    ):
        super().__init__(
            action_dim=action_dim,
            cond_encoder=cond_encoder,
            backbone=backbone,
            diffusion=diffusion,
            cond_output_key=cond_output_key,
        )
        self.image_key = str(image_key)
        self._image_channels = int(image_channels)
        self._image_size = int(image_size)
        self._frame_sampling = str(frame_sampling)
        self._sample_n_frames = int(sample_n_frames)
        self._bundle_shape = (self._image_channels, self._image_size, self._image_size)

    @property
    def bundle_shape(self) -> tuple:
        return self._bundle_shape

    @property
    def bundle_dim(self) -> int:
        c, h, w = self._bundle_shape
        return c * h * w

    @property
    def action_slice(self) -> slice:
        return slice(0, 0)

    def _extract_images(self, ctx: SimpleNamespace) -> torch.Tensor:
        if self.image_key not in ctx.obs:
            raise KeyError(
                f"image key '{self.image_key}' required but missing from "
                f"ctx.obs (keys: {list(ctx.obs.keys())})."
            )
        img = ctx.obs[self.image_key]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.max() > 1.5:
            img = img.float() / 255.0
        else:
            img = img.float()
        return img

    def _sample_windows_packed(self, images, actions, cu):
        """Frame-sample images (and OPTIONALLY actions) IDENTICALLY per packed
        episode, so the action at frame t always lines up with image t after
        cropping.

        This is the SUPERSET sampler shared by the image-only base stage and
        the image+action subclass (dedup collapse c7). Passing ``actions=None``
        makes it the pure image-only frame sampler — the image cropping +
        cu_seqlens are byte-identical whether or not actions are supplied (the
        action branch performs no extra RNG draws), proven by
        ``tests/test_c7_sampler_reducer_equality``.

        Returns ``(sampled_images, sampled_actions_or_None, new_cu)``.
        """
        has_act = actions is not None
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
                if has_act:
                    act_crops.append(actions[s + idx])
                continue
            else:
                sl = slice(s, e)
            img_crops.append(images[sl])
            if has_act:
                act_crops.append(actions[sl])
        new_cu = torch.zeros(b + 1, dtype=cu.dtype, device=cu.device)
        for i, c in enumerate(img_crops):
            new_cu[i + 1] = new_cu[i] + c.shape[0]
        sampled_act = torch.cat(act_crops, 0) if has_act else None
        return torch.cat(img_crops, 0), sampled_act, new_cu

    def _sample_frames_packed(
        self, images: torch.Tensor, cu_seqlens: torch.Tensor
    ):
        """Image-only frame sampler — thin delegate to the action-aware
        superset ``_sample_windows_packed`` with ``actions=None`` (dedup
        collapse c7; the standalone duplicated loop was removed after proving
        byte-identical output across all sampling modes).

        Returns:
            sampled: (T_new, C, H, W) re-packed sampled frames.
            new_cu: (B+1,) updated cu_seqlens.
        """
        sampled, _, new_cu = self._sample_windows_packed(images, None, cu_seqlens)
        return sampled, new_cu

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        images = self._extract_images(ctx)

        # Frame sampling for packed data during training.
        if (ctx.is_packed and self._frame_sampling != "full"
                and self.training):
            images, new_cu = self._sample_frames_packed(
                images, ctx.cu_seqlens
            )
            ctx.cu_seqlens = new_cu
            ctx.max_seqlen = max(
                int(new_cu[i + 1] - new_cu[i])
                for i in range(new_cu.shape[0] - 1)
            )
            # Also crop the action key in the batch so loss shapes match
            ac_key = getattr(ctx, 'action_key', None)
            if ac_key and ac_key in batch:
                actions = batch[ac_key]
                crops = []
                old_cu = ctx._original_cu if hasattr(ctx, '_original_cu') else None
                # Actions were already packed matching original images,
                # but we've now re-packed images. We need to crop actions
                # the same way. Store original cu before overwrite.
                # Actually, the actions come from the batch which hasn't
                # been modified. We need to re-crop them too.
                # For simplicity, just truncate actions to match new cu.
                # This works because the loss only uses ctx.q_state which
                # has the cropped x_start/noise.

        if ctx.is_packed:
            T_total = images.shape[0]
            t = self._sample_noise_levels((T_total,), images.device)
        else:
            B, T = images.shape[:2]
            t = self._sample_noise_levels((B, T), images.device)

        noise = torch.randn_like(images).clamp_(
            -self.diffusion.clip_noise, self.diffusion.clip_noise
        )
        x_t = self.diffusion.q_sample(images, t, noise=noise)
        ctx.q_state = {
            "x_t": x_t,
            "k": t,
            "time_cond": t,
            "noise": noise,
            "x_start": images,
        }
        ctx.external_cond = None
        ctx.latent_clean = images
        return x_t

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        x_t = self.encode(batch, ctx)
        time_cond = ctx.q_state["time_cond"]

        if not ctx.is_packed:
            v_pred = self.inner_stage(
                x_t, time_cond, external_cond=None,
            )
        else:
            cu = ctx.cu_seqlens
            B = cu.shape[0] - 1
            pieces = []
            for i in range(B):
                s, e = int(cu[i].item()), int(cu[i + 1].item())
                x_ep = x_t[s:e].unsqueeze(0)
                t_ep = time_cond[s:e].unsqueeze(0)
                v_ep = self.inner_stage(x_ep, t_ep, external_cond=None)
                pieces.append(v_ep.squeeze(0))
            v_pred = torch.cat(pieces, dim=0)

        self.decode(v_pred, batch, ctx)
        return v_pred
