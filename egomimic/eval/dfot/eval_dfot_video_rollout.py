"""DFoT video self-rollout evaluator.

For the joint obs+action+image DFoT variant
(``ObsActionImageDFoTOuterStage``), the diffusion bundle includes a
flattened VAE latent per step. At inference we sample a trajectory of
bundles, slice out the latent portion of each step, reshape to the
VAE's spatial form, and run the frozen VAE decoder to produce
pixel frames. The result is a video predicted by the model.

Different from ``DFoTSelfRolloutEval`` (which renders predicted STATE
through the env): this eval renders predicted IMAGE-LATENT through
the VAE. It tests the "world-model + policy" capability literally.

Output: ``<output_dir>/videos_video_rollout/epoch_N/<embodiment>/
validation_video_N.mp4`` — sequence of ``rollout_steps`` decoded frames
per starting episode.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` float -> ``(H, W, C)`` uint8 in [0, 255]."""
    x = img_chw.detach().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


class DFoTVideoRolloutEval(EvalVideo):
    """VAE-decoded self-rollout for ObsActionImageDFoTOuterStage.

    Pipeline per episode:
      1. Pick the t=0 image from the val batch.
      2. Run the algo's chunk-mode sampler for ``rollout_steps`` tokens
         producing a (T, bundle_dim) tensor.
      3. Slice the bundle's image-latent portion via
         ``algo.outer_stage.image_latent_slice``, reshape to
         ``algo.outer_stage.latent_shape``.
      4. Run frozen VAE decoder -> pixel frames (T, 3, H, W).
      5. Stack the t=0 GT frame in front as the launch condition, then
         the predicted frames -> mp4.

    Args:
        rollout_steps: number of bundle tokens to generate (= mp4 frames).
        upscale_to: post-render upscale side; default 384 for visibility.
        max_videos: cap on episodes rendered per val pass.
        limit_val_batches: cap on val batches per val pass.
        embodiment_name: which embodiment the packed val batches carry.
        image_key: which obs key carries the t=0 image.
        video_subdir: subdir under output_dir for these mp4s. Defaults
            to ``videos_video_rollout`` so it doesn't collide with the
            sim-rollout / teacher-forced eval video dirs.
    """

    def __init__(
        self,
        rollout_steps: int = 64,
        upscale_to: int = 384,
        max_videos: int = 2,
        limit_val_batches: int = 4,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        video_subdir: str = "videos_video_rollout",
        mode: str = "chunk",
        ar_chunk_size: int = 1,
        ar_step_size: int = 1,
        n_chunk_steps: int = 50,
        cfg_scale: float = 1.0,
        recon_loss_n_frames: int = 10,
        viz_func=None,
        transform_lists=None,
    ):
        """
        mode:
          * ``"chunk"`` (default): full-chunk DDIM with uniform per-
            token noise, ``n_chunk_steps`` denoise steps. Same as
            ``algo._sample_chunk``.
          * ``"ar"``: staircase causal-AR with rolling per-token noise
            schedule. Earlier tokens denoise first; later tokens stay
            noisy. ``ar_chunk_size`` = tokens per rung;
            ``ar_step_size`` = denoise steps per rung. Same matrix
            shape as ``DFoTValEval._ar_pred``.
        """
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        if mode not in {"chunk", "ar"}:
            raise ValueError(f"mode must be 'chunk' or 'ar', got {mode!r}")
        self.rollout_steps = int(rollout_steps)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self._video_subdir = str(video_subdir)
        self.mode = mode
        self.ar_chunk_size = int(ar_chunk_size)
        self.ar_step_size = int(ar_step_size)
        self.n_chunk_steps = int(n_chunk_steps)
        self.cfg_scale = float(cfg_scale)
        self.recon_loss_n_frames = int(recon_loss_n_frames)

    def video_dir(self):
        import os
        return os.path.join(self.root_dir(), self._video_subdir)

    @torch.no_grad()
    def _decode_latents_to_frames(
        self, algo, latent_flat: torch.Tensor
    ) -> torch.Tensor:
        """``(T, latent_flat_dim) -> (T, 3, H, W)``."""
        outer = algo.outer_stage
        c, h, w = outer.latent_shape
        z = latent_flat.view(-1, c, h, w)
        # Frozen VAE on the right device.
        return outer.vae.decode(z)

    @torch.no_grad()
    def _rollout(self, algo, device, cond) -> torch.Tensor:
        """Sample a (T, bundle_dim) bundle. Mode = ``"chunk"`` uses
        ``algo._sample_chunk`` (uniform per-token noise + DDIM). Mode =
        ``"ar"`` builds a staircase schedule matrix and calls
        ``sample`` directly so the per-token noise pattern mirrors
        DFoT's training AR schedule."""
        from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion
        from egomimic.models.diffusion.sampling import (
            sample as _sample,
            staircase_ar_schedule,
        )

        T = self.rollout_steps
        if self.mode == "chunk":
            bundle = algo._sample_chunk(B=1, T=T, cond=cond, device=device)
            return bundle.squeeze(0)

        # mode == "ar"
        bundle_dim = int(algo.outer_stage.bundle_dim)
        discrete_ts = (
            int(algo.diffusion.timesteps)
            if isinstance(algo.diffusion, DiscreteDiffusion)
            else None
        )
        schedule = staircase_ar_schedule(
            T=T,
            chunk_size=self.ar_chunk_size,
            step_size=self.ar_step_size,
            discrete_timesteps=discrete_ts,
        ).to(device)
        ec = cond.unsqueeze(0) if cond is not None else None
        bundle = _sample(
            algo.diffusion,
            algo.backbone,
            schedule_matrix=schedule,
            action_dim=bundle_dim,
            batch_size=1,
            external_cond=ec,
            cfg_scale=self.cfg_scale,
            device=device,
        ).squeeze(0)
        return bundle

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

        imgs = _batch[self.image_key]
        device = self.trainer.lightning_module.device

        # Pick t=0 image for each of the first ``max_videos`` episodes.
        if not _batch.get("_packed", False):
            B = imgs.shape[0]
            n = min(B, self.max_videos or B)
            starting = [imgs[i, 0] for i in range(n)]
        else:
            cu = _batch["cu_seqlens"].to(imgs.device, dtype=torch.long)
            n = min(int(cu.shape[0] - 1), self.max_videos or 99999)
            starting = [imgs[int(cu[i].item())] for i in range(n)]

        outer = algo.outer_stage
        if not hasattr(outer, "image_latent_slice"):
            # Not the obs+action+image variant; nothing to do.
            return metrics, images_dict
        lat_slice = outer.image_latent_slice

        # The model's _sample_chunk wants a cond tensor that matches the
        # backbone's external_cond shape. For the obs+action+image
        # variant the cond_encoder is usually empty (image is in the
        # bundle, not the cond), so cond should be None.
        cond = None

        # Per-step + average MSE of predicted vs GT frames for the first
        # ``recon_loss_n_frames`` rollout steps. Caveat: with the current
        # empty cond_encoder + no obs-anchoring at inference, predictions
        # are sampled from the learned prior with no link to a specific
        # GT episode. The MSE is therefore noisy episode-to-episode but
        # the mean across val episodes should still trend down as the
        # model learns the dataset's distribution.
        per_step_sse = np.zeros(self.recon_loss_n_frames, dtype=np.float64)
        per_step_n = np.zeros(self.recon_loss_n_frames, dtype=np.int64)

        all_frames: List[np.ndarray] = []
        for ep_idx, img_chw in enumerate(starting):
            bundle_pred = self._rollout(algo, device, cond)  # (T, bundle_dim)
            latent_seq = bundle_pred[:, lat_slice]
            pred_frames = self._decode_latents_to_frames(algo, latent_seq)

            # ---- per-step MSE against GT episode's first N frames ----
            n_cmp = min(self.recon_loss_n_frames, pred_frames.shape[0])
            if _batch.get("_packed", False):
                s_idx = int(cu[ep_idx].item())
                gt_seq = imgs[s_idx : s_idx + n_cmp]
            else:
                gt_seq = imgs[ep_idx, :n_cmp]
            # GT in [0,1] float; predicted also in [0,1] from VAE sigmoid.
            gt_f = gt_seq.to(device).float()
            if gt_f.max() > 1.5:
                gt_f = gt_f / 255.0
            pred_f = pred_frames[:n_cmp].to(device).float()
            # MSE per step over (C, H, W) dims.
            mse_per_step = ((pred_f - gt_f) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for t in range(n_cmp):
                per_step_sse[t] += float(mse_per_step[t])
                per_step_n[t] += 1

            # Prepend the GT t=0 frame for context.
            t0_uint = _img_chw_to_uint8(img_chw)
            t0_uint = cv2.resize(
                t0_uint, (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            all_frames.append(np.ascontiguousarray(t0_uint))
            for t in range(pred_frames.shape[0]):
                f = _img_chw_to_uint8(pred_frames[t])
                f = cv2.resize(
                    f, (self.upscale_to, self.upscale_to),
                    interpolation=cv2.INTER_NEAREST,
                )
                all_frames.append(np.ascontiguousarray(f))

        # ---- emit metrics ----
        # Per-step (so wandb plots a fan / can see degradation over time).
        for t in range(self.recon_loss_n_frames):
            if per_step_n[t] > 0:
                metrics[
                    f"Valid/emb{emb_id}_video_recon_mse_step_{t:02d}"
                ] = torch.tensor(
                    per_step_sse[t] / per_step_n[t], device=device
                )
        # Single scalar across the first N frames (averaged per episode).
        if per_step_n.sum() > 0:
            metrics[
                f"Valid/emb{emb_id}_video_recon_mse_first{self.recon_loss_n_frames}"
            ] = torch.tensor(
                per_step_sse.sum() / per_step_n.sum(), device=device
            )

        if all_frames:
            images_dict[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images_dict
