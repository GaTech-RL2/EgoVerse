"""DFoT spatial video-rollout eval for ``ImageSpatialDFoTOuterStage``.

Different from ``DFoTVideoRolloutEval``:

* The diffusion target IS already a spatial latent ``(T, C, H, W)`` —
  no slicing needed.
* External_cond is built per-step from the val batch's GT state +
  action (the model's world-modeling input). The model predicts the
  image latent sequence conditioned on that.
* Per-step MSE against the val batch's GT frames is the headline
  quality metric (this variant has GT images aligned per timestep, so
  the metric is meaningful — unlike the unconditioned-rollout variant
  in ``DFoTVideoRolloutEval``).

Outputs an mp4 of ``[GT | predicted]`` side-by-side per episode plus
per-step + cumulative recon MSE for the first ``recon_loss_n_frames``
steps.
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


def _img_chw_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().cpu().float().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


class DFoTSpatialVideoRolloutEval(EvalVideo):
    """World-model video rollout for ``ImageSpatialDFoTOuterStage``.

    Args:
        rollout_steps: number of bundle tokens to generate. mp4 length =
            this + 1 (GT t=0 frame prepended).
        upscale_to: post-render upscale side per panel.
        max_videos: per-pass episode cap.
        limit_val_batches: per-pass batch cap.
        embodiment_name: which embodiment the packed batch carries.
        image_key: which obs key carries images.
        recon_loss_n_frames: how many leading steps to MSE for the
            scalar Valid metric.
        mode: ``"chunk"`` (default) or ``"ar"`` (staircase). Note:
            spatial backbone with row-major (t, p) flattening + 1D
            causal mask isn't truly causal — use "chunk" unless you
            have a factorized-attention backbone.
        ar_chunk_size, ar_step_size, n_chunk_steps, cfg_scale:
            sampler knobs (mirror DFoTValEval / DFoTVideoRolloutEval).
        video_subdir: output subdir under root_dir.
    """

    def __init__(
        self,
        rollout_steps: int = 64,
        upscale_to: int = 384,
        max_videos: int = 2,
        limit_val_batches: int = 4,
        embodiment_name: str = "pushshapes_sim",
        image_key: str = "front_img_1",
        recon_loss_n_frames: int = 10,
        mode: str = "chunk",
        ar_chunk_size: int = 1,
        ar_step_size: int = 1,
        n_chunk_steps: int = 50,
        cfg_scale: float = 1.0,
        video_subdir: str = "videos_spatial_rollout",
        viz_func=None,
        transform_lists=None,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
            max_videos=max_videos,
        )
        if mode not in {"chunk", "ar"}:
            raise ValueError(f"mode must be 'chunk' or 'ar'; got {mode!r}")
        self.rollout_steps = int(rollout_steps)
        self.upscale_to = int(upscale_to)
        self.embodiment_name = embodiment_name
        self.image_key = str(image_key)
        self.recon_loss_n_frames = int(recon_loss_n_frames)
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
    # Rollout: external_cond built from (state, action) GT, passed
    # through the outer stage's projection, then handed to the sampler.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout(
        self, algo, cond_seq: torch.Tensor, device, batch_size: int = 1
    ) -> torch.Tensor:
        """Returns predicted latent of shape ``(batch_size, T, C, H, W)``."""
        outer = algo.outer_stage
        bundle_shape = outer.bundle_shape
        T = cond_seq.shape[1]
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
            external_cond=cond_seq,
            cfg_scale=self.cfg_scale,
            device=device,
        )

    # ------------------------------------------------------------------
    # Build per-step (state, action) -> external_cond for one episode.
    # ------------------------------------------------------------------

    def _build_cond_seq(
        self, algo, _batch: dict, emb_id: int, start: int, T: int
    ) -> torch.Tensor:
        """``(1, T, state_action_proj_dim)`` projected cond over a slice
        of the val batch starting at frame ``start``."""
        outer = algo.outer_stage
        ac_key = algo.resolved_ac_keys[emb_id]
        pieces = []
        for key in outer.bundle_obs_keys:
            pieces.append(_batch[key][start : start + T])
        pieces.append(_batch[ac_key][start : start + T])
        concat = torch.cat(pieces, dim=-1)  # (T, state_dim + action_dim)
        cond = outer.state_action_proj(concat)  # (T, cond_dim)
        return cond.unsqueeze(0)                # (1, T, cond_dim)

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
        if not hasattr(outer, "bundle_shape") or not isinstance(
            outer.bundle_shape, tuple
        ) or len(outer.bundle_shape) <= 1:
            # Not a spatial outer stage; nothing to do.
            return metrics, images_dict

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
                cond_seq = self._build_cond_seq(
                    algo, _batch, emb_id, start, T_rollout,
                )
                gt_seq = imgs[start : start + T_rollout]
            else:
                # Padded: take row ep_idx, first T_rollout frames.
                # Build cond from the padded view.
                ac_key = algo.resolved_ac_keys[emb_id]
                pieces = []
                for key in outer.bundle_obs_keys:
                    pieces.append(_batch[key][ep_idx, :T_rollout])
                pieces.append(_batch[ac_key][ep_idx, :T_rollout])
                concat = torch.cat(pieces, dim=-1)
                cond_seq = outer.state_action_proj(concat).unsqueeze(0)
                gt_seq = imgs[ep_idx, :T_rollout]

            cond_seq = cond_seq.to(device).float()
            # ---- Sampler returns (1, T, C, H, W) ----
            pred_latent = self._rollout(algo, cond_seq, device).squeeze(0)
            # ---- VAE decode -> pixel frames (T, 3, H, W) ----
            pred_frames = outer.vae.decode(outer.denormalize_latent(pred_latent))

            # ---- per-step MSE vs GT ----
            n_cmp = min(self.recon_loss_n_frames, pred_frames.shape[0])
            gt_f = gt_seq[:n_cmp].to(device).float()
            if gt_f.max() > 1.5:
                gt_f = gt_f / 255.0
            pred_f = pred_frames[:n_cmp].float()
            mse_per_step = ((pred_f - gt_f) ** 2).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for t in range(n_cmp):
                per_step_sse[t] += float(mse_per_step[t])
                per_step_n[t] += 1

            # ---- side-by-side mp4 frames ----
            for t in range(pred_frames.shape[0]):
                gt_t = _img_chw_to_uint8(gt_seq[t] / (255.0 if gt_seq.max() > 1.5 else 1.0))
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
                    f"Valid/emb{emb_id}_spatial_recon_mse_step_{t:02d}"
                ] = torch.tensor(
                    per_step_sse[t] / per_step_n[t], device=device
                )
        if per_step_n.sum() > 0:
            metrics[
                f"Valid/emb{emb_id}_spatial_recon_mse_first{self.recon_loss_n_frames}"
            ] = torch.tensor(
                per_step_sse.sum() / per_step_n.sum(), device=device
            )

        if all_frames:
            images_dict[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images_dict
