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

Parity knobs (defaults reproduce historic behavior exactly):
  - ``image_range``: ``"01"`` (default) diffuses images in [0, 1];
    ``"pm1"`` maps to [-1, 1] right after the /255 extraction and inverts
    at every decode/rollout/viz seam (D03).
  - ``image_loss_weight`` / ``action_loss_weight_group`` /
    ``image_loss_weighting`` / ``action_loss_weighting``: when ANY is set,
    the training loss becomes separate means over image vs action bundle
    channels, each with its own per-call weighting-strategy override,
    combined as ``image_loss_weight*img + action_loss_weight_group*act``
    (D16). Unset weights default to 1.0; unset strategies use the
    diffusion core's configured strategy. All None -> single-mean path.
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
        image_range: str = "01",
        image_loss_weight: Optional[float] = None,
        action_loss_weight_group: Optional[float] = None,
        image_loss_weighting: Optional[str] = None,
        action_loss_weighting: Optional[str] = None,
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
        if image_range not in ("01", "pm1"):
            raise ValueError(f"image_range must be '01' or 'pm1'; got {image_range!r}")
        self.image_range = str(image_range)
        # Per-group loss knobs (D16 + decoupling seam). All None -> the loss
        # flows through DFoTLoss's single-mean path bit-identically; setting
        # ANY of them switches to separate image/action group means combined
        # as image_loss_weight*img + action_loss_weight_group*act.
        for s in (image_loss_weighting, action_loss_weighting):
            if s == "fused_min_snr":
                raise ValueError(
                    "fused_min_snr cannot be used as a per-group weighting "
                    "override; it couples weights across the time axis"
                )
        self.image_loss_weight = (
            None if image_loss_weight is None else float(image_loss_weight)
        )
        self.action_loss_weight_group = (
            None if action_loss_weight_group is None
            else float(action_loss_weight_group)
        )
        self.image_loss_weighting = image_loss_weighting
        self.action_loss_weighting = action_loss_weighting

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

    # ------------------------------------------------------------------
    # Image-range seam (D03). The model diffuses images in "model range":
    # [0, 1] (image_range="01", default — historic behavior) or [-1, 1]
    # (image_range="pm1", upstream-parity). Public so inference/eval code
    # that fabricates image context or decodes sampled frames can route
    # through the same mapping.
    # ------------------------------------------------------------------

    def to_model_range(self, img01: torch.Tensor) -> torch.Tensor:
        """Map images from [0, 1] into the diffusion model range."""
        if self.image_range == "pm1":
            return img01 * 2.0 - 1.0
        return img01

    def from_model_range(self, x: torch.Tensor) -> torch.Tensor:
        """Map images from the diffusion model range back to [0, 1]."""
        if self.image_range == "pm1":
            return (x + 1.0) / 2.0
        return x

    @property
    def model_range_bounds(self) -> tuple:
        """(lo, hi) clamp bounds of valid images in model range."""
        return (-1.0, 1.0) if self.image_range == "pm1" else (0.0, 1.0)

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
        return self.to_model_range(img)

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

    # ------------------------------------------------------------------
    # Per-group loss split (D16 + decoupling seam). Image channels and
    # action-plane channels of the jointly-diffused bundle get separate
    # means (each with its own per-call weighting-strategy override),
    # combined as image_loss_weight*img + action_loss_weight_group*act.
    # Disabled (all knobs None) -> ``forward`` leaves ``ctx.precomputed_loss``
    # unset and DFoTLoss runs the historic single-mean path bit-identically.
    # ------------------------------------------------------------------

    @property
    def _group_loss_enabled(self) -> bool:
        return any(
            v is not None
            for v in (
                self.image_loss_weight,
                self.action_loss_weight_group,
                self.image_loss_weighting,
                self.action_loss_weighting,
            )
        )

    def _diffusion_group_loss(
        self, v_pred: torch.Tensor, q_state: dict, strategy: Optional[str]
    ) -> torch.Tensor:
        """``diffusion.compute_loss`` with an optional per-call weighting
        override. Only forwards the kwarg when a strategy is actually set so
        the default path stays byte-identical to the historic call."""
        if strategy is None:
            return self.diffusion.compute_loss(v_pred, q_state)
        return self.diffusion.compute_loss(
            v_pred, q_state, weighting_strategy=strategy
        )

    @staticmethod
    def _slice_q_state(q_state: dict, ch_slice: slice) -> dict:
        """Channel-slice the spatial tensors of a q_state dict (axis -3 of
        (..., C, H, W)); per-token entries (k, time_cond, ...) pass through."""
        out = dict(q_state)
        for key in ("x_t", "noise", "x_start"):
            val = out.get(key)
            if torch.is_tensor(val):
                out[key] = val[..., ch_slice, :, :]
        return out

    def _compute_group_loss(self, v_pred: torch.Tensor, q_state: dict) -> torch.Tensor:
        c_img = self._image_channels
        c_total = int(v_pred.shape[-3])
        w_img = 1.0 if self.image_loss_weight is None else self.image_loss_weight
        w_act = (
            1.0 if self.action_loss_weight_group is None
            else self.action_loss_weight_group
        )

        img_sl = slice(0, c_img)
        img_loss = self._diffusion_group_loss(
            v_pred[..., img_sl, :, :],
            self._slice_q_state(q_state, img_sl),
            self.image_loss_weighting,
        ).mean()
        loss = w_img * img_loss
        if c_total > c_img:
            act_sl = slice(c_img, c_total)
            act_loss = self._diffusion_group_loss(
                v_pred[..., act_sl, :, :],
                self._slice_q_state(q_state, act_sl),
                self.action_loss_weighting,
            ).mean()
            loss = loss + w_act * act_loss
        return loss

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
        if self._group_loss_enabled:
            ctx.precomputed_loss = self._compute_group_loss(v_pred, ctx.q_state)
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
                    external_cond=None, eta=float(getattr(algo, "sampler_eta", 0.0)),
                )
                # Revert context frames to clean values
                x[0, :c] = context_part

            # Append newly generated frames. Image channels clamp to the
            # model range; extra bundle channels (action planes in the policy
            # bundle) are not pixel-valued and stay unclamped.
            new_frames = x[0, c:c + h]
            lo, hi = self.model_range_bounds
            c_img = self._image_channels
            if new_frames.shape[1] > c_img:
                new_frames = torch.cat(
                    [new_frames[:, :c_img].clamp(lo, hi), new_frames[:, c_img:]],
                    dim=1,
                )
            else:
                new_frames = new_frames.clamp(lo, hi)
            generated = torch.cat([generated, new_frames], dim=0)

        return generated[:total_frames]

    def _make_rollout_context(
        self, ctx_imgs, _batch, algo, emb_id, ep_idx, ep_start, is_packed,
    ) -> torch.Tensor:
        """Build the (n_ctx, C_bundle, H, W) clean-context bundle from the
        model-range GT context images. Base stage: bundle == image channels;
        if the bundle carries extra (non-image) channels, pad them with zeros
        so the cat against bundle-shaped noise can't crash (D06). Subclasses
        with a structured bundle override to fill those channels properly."""
        c_bundle = int(self.bundle_shape[0])
        c_img = int(ctx_imgs.shape[1])
        if c_bundle <= c_img:
            return ctx_imgs
        n_ctx, _, h, w = ctx_imgs.shape
        pad = ctx_imgs.new_zeros(n_ctx, c_bundle - c_img, h, w)
        return torch.cat([ctx_imgs, pad], dim=1)

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

        # GT normalization (also used to seed the context frame(s)). gt_f
        # stays in [0, 1] for metrics/panels; context converts to model range.
        gt_f = gt_seq[:T_rollout].to(device).float()
        if gt_f.max() > 1.5:
            gt_f = gt_f / 255.0

        # Conditional rollout matching the reference DFoT prediction task:
        # seed the first n_context GT frames, hold them clean (noise level
        # -1) for the whole sampling trajectory, and predict the rest
        # conditioned on them.
        n_ctx = max(1, min(ev.n_context_frames, T_rollout))
        context_frames = self._make_rollout_context(
            self.to_model_range(gt_f[:n_ctx]),
            _batch, algo, emb_id, ep_idx, ep_start, is_packed,
        )
        pred_frames = self._rollout_sliding_window(
            ev,
            algo,
            total_frames=T_rollout,
            context_frames=context_frames,
            window_size=min(ev.rollout_window, T_rollout),
            device=device,
        )  # (T, C_bundle, H, W)

        # Decode: keep image channels, map back to [0, 1] for metrics/viz.
        pred_frames = pred_frames[:, : self._image_channels]
        pred_frames = self.from_model_range(pred_frames).clamp(0.0, 1.0)

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
