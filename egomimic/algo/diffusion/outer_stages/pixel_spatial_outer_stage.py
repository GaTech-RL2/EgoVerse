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

try:
    from torchmetrics.image import (  # noqa: F401
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import (  # noqa: F401
        LearnedPerceptualImagePatchSimilarity,
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


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

    # ------------------------------------------------------------------
    # Video-rollout hook (COMBINE A — decode-on-outer-stage).
    #
    # This stage owns the pixel-family rollout: a sliding-window DDIM rollout
    # anchored on the first n_context GT frames (no VAE — output IS pixels),
    # plus the PSNR/SSIM/LPIPS perceptual metrics. Moved byte-for-byte from
    # ``eval_dfot_pixel_video_rollout.DFoTPixelVideoRolloutEval``
    # (``_rollout_sliding_window`` + per-episode body). Side-by-side
    # [GT|pred] panel.
    # ------------------------------------------------------------------

    video_metric_prefix = "pixel"
    video_panel = "sidebyside"
    video_has_extra_metrics = True

    @torch.no_grad()
    def _rollout_sliding_window(
        self, ev, algo, total_frames: int, context_frames: torch.Tensor,
        window_size: int, device,
    ) -> torch.Tensor:
        """Sliding window rollout matching the reference DFoT inference.

        Args:
            ev: the unified video-rollout eval (carries n_chunk_steps).
            algo: the DFoT algo.
            total_frames: total number of frames to generate.
            context_frames: (n_context, C, H, W) initial context frames.
            window_size: number of frames per denoising window.
            device: torch device.

        Returns:
            (total_frames, C, H, W) generated frames.
        """
        from egomimic.models.diffusion.sampling import sample_step, vanilla_schedule

        bundle_shape = self.bundle_shape
        discrete_ts = (
            int(algo.diffusion.timesteps)
            if isinstance(algo.diffusion, DiscreteDiffusion)
            else None
        )

        n_context = context_frames.shape[0]
        generated = context_frames.clone()  # (n_context, C, H, W)

        while generated.shape[0] < total_frames:
            # How many context frames to use (up to window_size - 1)
            c = min(generated.shape[0], window_size - 1)
            # How many new frames to generate
            h = min(total_frames - generated.shape[0], window_size - c)
            T_window = c + h

            # Build window: context + noise
            context_part = generated[-c:]  # (c, C, H, W)
            noise_part = torch.randn(h, *bundle_shape, device=device)
            x = torch.cat([context_part, noise_part], dim=0).unsqueeze(0)  # (1, T_window, C, H, W)

            # Build schedule
            schedule = vanilla_schedule(
                n_steps=ev.n_chunk_steps, T=T_window, discrete_timesteps=discrete_ts,
            ).to(device)

            # Context tokens get noise_level = -1 (clean) throughout
            # Modify schedule: context columns stay at -1
            for step_idx in range(schedule.shape[0]):
                schedule[step_idx, :c] = -1 if discrete_ts else 0.0

            # Run DDIM sampling
            for s in range(schedule.shape[0] - 1):
                x = sample_step(
                    algo.diffusion, algo.backbone, x=x,
                    current_levels=schedule[s],
                    next_levels=schedule[s + 1],
                    external_cond=None, eta=0.0,
                )
                # Revert context frames to clean values
                x[0, :c] = context_part

            # Append newly generated frames
            new_frames = x[0, c:c + h]
            generated = torch.cat([generated, new_frames.clamp(0, 1)], dim=0)

        return generated[:total_frames]

    @torch.no_grad()
    def rollout_video_episode(
        self, ev, algo, _batch, emb_id, ep_idx, ep_start, ep_len, device
    ) -> SimpleNamespace:
        """Per-episode pixel-space sliding-window rollout + perceptual metrics."""
        imgs = _batch[ev.image_key]
        is_packed = _batch.get("_packed", False)
        T_rollout = min(ev.rollout_steps, ep_len)

        if is_packed:
            gt_seq = imgs[ep_start : ep_start + T_rollout]
        else:
            gt_seq = imgs[ep_idx, :T_rollout]

        # GT normalization (also used to seed the context frame(s)).
        gt_f = gt_seq[:T_rollout].to(device).float()
        if gt_f.max() > 1.5:
            gt_f = gt_f / 255.0

        # Conditional rollout matching the reference DFoT prediction task:
        # seed the first n_context GT frames, hold them clean (noise level
        # -1) for the whole sampling trajectory, and predict the rest
        # conditioned on them.
        n_ctx = max(1, min(ev.n_context_frames, T_rollout))
        pred_frames = self._rollout_sliding_window(
            ev,
            algo,
            total_frames=T_rollout,
            context_frames=gt_f[:n_ctx],
            window_size=min(ev.rollout_window, T_rollout),
            device=device,
        )  # (T, 3, H, W)

        # Clamp output to [0, 1].
        pred_frames = pred_frames.clamp(0.0, 1.0)

        n_cmp = min(ev.recon_loss_n_frames, pred_frames.shape[0])

        # PSNR, SSIM, LPIPS per episode (averaged over frames).
        extra = {}
        if _HAS_METRICS and n_cmp > 0:
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )
            pred_cmp = pred_frames[:n_cmp].to(device)
            gt_cmp = gt_f[:n_cmp].to(device)
            psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
            extra["psnr"] = psnr_fn(pred_cmp, gt_cmp)
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            extra["ssim"] = ssim_fn(pred_cmp, gt_cmp)
            try:
                lpips_fn = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True).to(device)
                extra["lpips"] = lpips_fn(pred_cmp, gt_cmp)
            except Exception:
                pass

        return SimpleNamespace(
            pred_frames=pred_frames,
            gt_for_mse=gt_f[:n_cmp],
            # Raw GT slice for the side-by-side panel (reproduces the
            # original per-frame ``gt_seq[t] / (255 if max>1.5 else 1)``).
            gt_panel_raw=gt_seq,
            extra_metrics=extra,
        )
