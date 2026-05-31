"""DFoT pixel-space video-rollout eval for ``PixelSpatialDFoTOuterStage``.

Like ``DFoTSpatialVideoRolloutEval`` but skips VAE decode — the diffusion
output IS already pixel frames ``(T, 3, H, W)`` in [0, 1].

The model is a Diffusion-Forcing (random_independent) predictor, so it is
NOT an unconditional generator: rollout is conditioned on the first
``n_context_frames`` real GT frame(s), which are held clean for the whole
sampling trajectory while the remaining frames denoise (matching the
reference ``dfot_video.py`` prediction task, ``context_length=1`` for pusht).
Outputs an mp4 of ``[GT | predicted]`` side-by-side per episode plus
per-step + cumulative recon MSE.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.algo.dfot.discrete_diffusion import DiscreteDiffusion
from egomimic.algo.dfot.sampling import (
    sample as _sample,
    staircase_ar_schedule,
    vanilla_schedule,
)
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


class DFoTPixelVideoRolloutEval(EvalVideo):
    """Pixel-space video rollout for ``PixelSpatialDFoTOuterStage``.

    Args:
        rollout_steps: number of frames to generate.
        upscale_to: post-render upscale side per panel.
        max_videos: per-pass episode cap.
        limit_val_batches: per-pass batch cap.
        embodiment_name: which embodiment the packed batch carries.
        image_key: which obs key carries images.
        recon_loss_n_frames: how many leading steps to MSE.
        mode: ``"chunk"`` (default) or ``"ar"``.
        ar_chunk_size, ar_step_size, n_chunk_steps, cfg_scale:
            sampler knobs.
        video_subdir: output subdir under root_dir.
    """

    def __init__(
        self,
        rollout_steps: int = 9,
        upscale_to: int = 384,
        max_videos: int = 2,
        limit_val_batches: int = 4,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        recon_loss_n_frames: int = 9,
        n_context_frames: int = 1,
        rollout_window: int = 9,
        mode: str = "chunk",
        ar_chunk_size: int = 1,
        ar_step_size: int = 1,
        n_chunk_steps: int = 50,
        cfg_scale: float = 1.0,
        video_subdir: str = "videos_pixel_rollout",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        if mode not in {"chunk", "ar", "sliding_window"}:
            raise ValueError(f"mode must be 'chunk' or 'ar'; got {mode!r}")
        self.rollout_steps = int(rollout_steps)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self.recon_loss_n_frames = int(recon_loss_n_frames)
        self.n_context_frames = int(n_context_frames)
        self.rollout_window = int(rollout_window)
        self.mode = mode
        self.ar_chunk_size = int(ar_chunk_size)
        self.ar_step_size = int(ar_step_size)
        self.n_chunk_steps = int(n_chunk_steps)
        self.cfg_scale = float(cfg_scale)
        self._video_subdir = str(video_subdir)

    def video_dir(self):
        import os
        return os.path.join(self.root_dir(), self._video_subdir)

    # ------------------------------------------------------------------
    # Unconditional rollout (no external_cond).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout(
        self, algo, T: int, device, batch_size: int = 1
    ) -> torch.Tensor:
        """Returns predicted pixel frames ``(batch_size, T, 3, H, W)``."""
        outer = algo.outer_stage
        bundle_shape = outer.bundle_shape
        discrete_ts = (
            int(algo.diffusion.timesteps)
            if isinstance(algo.diffusion, DiscreteDiffusion)
            else None
        )
        if self.mode == "chunk":
            schedule = vanilla_schedule(
                n_steps=self.n_chunk_steps, T=T, discrete_timesteps=discrete_ts,
            ).to(device)
        else:
            schedule = staircase_ar_schedule(
                T=T,
                chunk_size=self.ar_chunk_size,
                step_size=self.ar_step_size,
                discrete_timesteps=discrete_ts,
            ).to(device)

        return _sample(
            algo.diffusion,
            algo.backbone,
            schedule_matrix=schedule,
            x_shape=bundle_shape,
            batch_size=batch_size,
            external_cond=None,  # No conditioning
            cfg_scale=self.cfg_scale,
            device=device,
        )

    @torch.no_grad()
    def _rollout_sliding_window(
        self, algo, total_frames: int, context_frames: torch.Tensor,
        window_size: int, device,
    ) -> torch.Tensor:
        """Sliding window rollout matching the reference DFoT inference.

        Args:
            algo: the DFoT algo.
            total_frames: total number of frames to generate.
            context_frames: (n_context, C, H, W) initial context frames.
            window_size: number of frames per denoising window.
            device: torch device.

        Returns:
            (total_frames, C, H, W) generated frames.
        """
        from egomimic.algo.dfot.sampling import sample_step

        outer = algo.outer_stage
        bundle_shape = outer.bundle_shape
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
                n_steps=self.n_chunk_steps, T=T_window, discrete_timesteps=discrete_ts,
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

    # ------------------------------------------------------------------
    # Main entrypoint.
    # ------------------------------------------------------------------

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        metrics: Dict[str, torch.Tensor] = {}
        images_dict: Dict[int, np.ndarray] = {}

        emb_id = get_embodiment_id(self.embodiment_name)
        if emb_id not in batch:
            return metrics, images_dict
        _batch = batch[emb_id]
        if self.image_key not in _batch:
            return metrics, images_dict

        device = self.trainer.lightning_module.device
        outer = algo.outer_stage

        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)

        # Episode indexing.
        if is_packed:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            n = min(int(cu.shape[0] - 1), self.max_videos or 99999)
            ep_starts = [int(cu[i].item()) for i in range(n)]
            ep_lens = [int(cu[i + 1].item() - cu[i].item()) for i in range(n)]
        else:
            B = imgs.shape[0]
            n = min(B, self.max_videos or B)
            ep_starts = [None] * n
            ep_lens = [imgs.shape[1]] * n

        per_step_sse = np.zeros(self.recon_loss_n_frames, dtype=np.float64)
        per_step_n = np.zeros(self.recon_loss_n_frames, dtype=np.int64)
        all_frames: List[np.ndarray] = []

        for ep_idx in range(n):
            ep_len = ep_lens[ep_idx]
            T_rollout = min(self.rollout_steps, ep_len)

            if is_packed:
                start = ep_starts[ep_idx]
                gt_seq = imgs[start : start + T_rollout]
            else:
                gt_seq = imgs[ep_idx, :T_rollout]

            # GT normalization (also used to seed the context frame(s)).
            gt_f = gt_seq[:T_rollout].to(device).float()
            if gt_f.max() > 1.5:
                gt_f = gt_f / 255.0

            # Conditional rollout matching the reference DFoT prediction task:
            # seed the first n_context GT frames, hold them clean (noise level
            # -1) for the whole sampling trajectory, and predict the rest
            # conditioned on them. The model is a Diffusion-Forcing
            # (random_independent) predictor, so unconditional all-frames-from-
            # noise sampling drifts off-manifold -> garbage; anchoring on >=1
            # real frame is what the reference does (context_length=1 for
            # pusht). _rollout (unconditional) is kept only as a baseline.
            n_ctx = max(1, min(self.n_context_frames, T_rollout))
            pred_frames = self._rollout_sliding_window(
                algo,
                total_frames=T_rollout,
                context_frames=gt_f[:n_ctx],
                window_size=min(self.rollout_window, T_rollout),
                device=device,
            )  # (T, 3, H, W)

            # Clamp output to [0, 1].
            pred_frames = pred_frames.clamp(0.0, 1.0)

            # Per-step MSE.
            n_cmp = min(self.recon_loss_n_frames, pred_frames.shape[0])
            mse_per_step = (
                (pred_frames[:n_cmp] - gt_f[:n_cmp]) ** 2
            ).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for t in range(n_cmp):
                per_step_sse[t] += float(mse_per_step[t])
                per_step_n[t] += 1

            # PSNR, SSIM, LPIPS per episode (averaged over frames).
            if _HAS_METRICS and n_cmp > 0:
                pred_cmp = pred_frames[:n_cmp].to(device)
                gt_cmp = gt_f[:n_cmp].to(device)
                # PSNR
                psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
                ep_psnr = psnr_fn(pred_cmp, gt_cmp)
                metrics.setdefault(f"Valid/emb{emb_id}_pixel_psnr_sum", torch.tensor(0.0, device=device))
                metrics[f"Valid/emb{emb_id}_pixel_psnr_sum"] += ep_psnr
                # SSIM
                ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                ep_ssim = ssim_fn(pred_cmp, gt_cmp)
                metrics.setdefault(f"Valid/emb{emb_id}_pixel_ssim_sum", torch.tensor(0.0, device=device))
                metrics[f"Valid/emb{emb_id}_pixel_ssim_sum"] += ep_ssim
                # LPIPS
                try:
                    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
                    ep_lpips = lpips_fn(pred_cmp, gt_cmp)
                    metrics.setdefault(f"Valid/emb{emb_id}_pixel_lpips_sum", torch.tensor(0.0, device=device))
                    metrics[f"Valid/emb{emb_id}_pixel_lpips_sum"] += ep_lpips
                except Exception:
                    pass

            # Side-by-side mp4 frames.
            for t in range(pred_frames.shape[0]):
                gt_t = _img_chw_to_uint8(
                    gt_seq[t] / (255.0 if gt_seq.max() > 1.5 else 1.0)
                )
                pr_t = _img_chw_to_uint8(pred_frames[t])
                gt_t = cv2.resize(
                    gt_t, (self.upscale_to, self.upscale_to),
                    interpolation=cv2.INTER_NEAREST,
                )
                pr_t = cv2.resize(
                    pr_t, (self.upscale_to, self.upscale_to),
                    interpolation=cv2.INTER_NEAREST,
                )
                all_frames.append(np.concatenate([gt_t, pr_t], axis=1))

        for t in range(self.recon_loss_n_frames):
            if per_step_n[t] > 0:
                metrics[
                    f"Valid/emb{emb_id}_pixel_recon_mse_step_{t:02d}"
                ] = torch.tensor(
                    per_step_sse[t] / per_step_n[t], device=device
                )
        if per_step_n.sum() > 0:
            metrics[
                f"Valid/emb{emb_id}_pixel_recon_mse_first{self.recon_loss_n_frames}"
            ] = torch.tensor(
                per_step_sse.sum() / per_step_n.sum(), device=device
            )

        # Average PSNR, SSIM, LPIPS over episodes.
        if n > 0 and _HAS_METRICS:
            for suffix in ["psnr", "ssim", "lpips"]:
                sum_key = f"Valid/emb{emb_id}_pixel_{suffix}_sum"
                if sum_key in metrics:
                    metrics[f"Valid/emb{emb_id}_pixel_{suffix}"] = metrics.pop(sum_key) / n

        if all_frames:
            images_dict[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images_dict
