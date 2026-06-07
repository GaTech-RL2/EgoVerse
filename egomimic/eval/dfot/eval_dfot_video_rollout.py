"""Family-agnostic DFoT video self-rollout evaluator (COMBINE A).

ONE evaluator drives all three DFoT video-rollout families. Each family's
outer stage owns a ``rollout_video_episode`` hook (decode-on-outer-stage)
that turns its sampler output into ``(T, 3, H, W)`` pixel frames:

  * :class:`ObsActionImageDFoTOuterStage` — unconditional chunk/AR bundle
    sampler, slice the flat VAE-latent portion, frozen-VAE decode. Single
    panel (t=0 GT prepended). Metric prefix ``video``.
  * :class:`ImageSpatialDFoTOuterStage` — per-step (state,action) cond,
    conditional chunk/AR spatial-latent sampler (optional GT-context
    anchor), frozen-VAE decode. Side-by-side [GT|pred]. Prefix ``spatial``.
  * :class:`PixelSpatialDFoTOuterStage` — sliding-window pixel-space rollout
    anchored on the first GT frame(s), no VAE. Side-by-side [GT|pred].
    Prefix ``pixel``. Also reports PSNR / SSIM / LPIPS.

This eval owns the family-INVARIANT skeleton: episode indexing (packed /
padded), per-step recon-MSE accumulation, panel assembly, perceptual-metric
averaging, and mp4 emission. The family-VARIANT pieces live on the outer
stage; the eval dispatches on three class attributes it advertises:
``video_metric_prefix`` / ``video_panel`` / ``video_has_extra_metrics``.

The old per-family classes are kept as compat aliases at the bottom
(:class:`DFoTSpatialVideoRolloutEval`, :class:`DFoTPixelVideoRolloutEval`) —
the unified ``__init__`` is a true superset and every old config now passes
its knobs explicitly, so the aliases are pure ``_target_`` redirects.

Output: ``<output_dir>/<video_subdir>/epoch_N/<embodiment>/
validation_video_N.mp4``.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.eval.core.img_utils import img_chw_to_uint8
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


class DFoTVideoRolloutEval(EvalVideo):
    """Family-agnostic DFoT video rollout — delegates the family-specific
    sampler + decode to ``outer_stage.rollout_video_episode``.

    Args (union of the three families' knobs; per-knob defaults preserved):
        rollout_steps: bundle tokens / frames to generate. (video/spatial
            default 64; pixel configs pass 9.)
        upscale_to: post-render upscale side per panel; default 384.
        max_videos: cap on episodes rendered per val pass.
        limit_val_batches: cap on val batches per val pass.
        embodiment_name: which embodiment the packed val batches carry.
        image_key: which obs key carries the image stream.
        recon_loss_n_frames: leading steps scored for the scalar recon MSE.
        n_context_frames: GT-context anchor length. 0 = unconditional
            (video / spatial-default); pixel uses >=1 (its config passes 1).
        rollout_window: pixel sliding-window size (pixel-only knob).
        mode: ``"chunk"`` (default) or ``"ar"`` (staircase). The video
            family's chunk mode = ``algo._sample_chunk`` (uniform per-token
            noise + DDIM); spatial/pixel chunk = ``vanilla_schedule``. AR =
            staircase per-token schedule (mirrors ``DFoTValEval``).
        ar_chunk_size, ar_step_size, n_chunk_steps, cfg_scale: sampler knobs.
        video_subdir: output subdir under root_dir (per-family namespaced).
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
        n_context_frames: int = 0,
        rollout_window: int = 9,
        mode: str = "chunk",
        ar_chunk_size: int = 1,
        ar_step_size: int = 1,
        n_chunk_steps: int = 50,
        cfg_scale: float = 1.0,
        video_subdir: str = "videos_video_rollout",
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
            raise ValueError(f"mode must be 'chunk' or 'ar', got {mode!r}")
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
    # Panel renderers — one per ``outer_stage.video_panel`` layout. Each
    # reproduces its family's original frame construction byte-for-byte.
    # ------------------------------------------------------------------

    def _panel_single_t0prepend(self, res) -> List[np.ndarray]:
        """obs+action+image: prepend the GT t=0 launch frame, then preds."""
        frames: List[np.ndarray] = []
        t0_uint = img_chw_to_uint8(res.gt_t0_chw)
        t0_uint = cv2.resize(
            t0_uint, (self.upscale_to, self.upscale_to),
            interpolation=cv2.INTER_NEAREST,
        )
        frames.append(np.ascontiguousarray(t0_uint))
        pred_frames = res.pred_frames
        for t in range(pred_frames.shape[0]):
            f = img_chw_to_uint8(pred_frames[t])
            f = cv2.resize(
                f, (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            frames.append(np.ascontiguousarray(f))
        return frames

    def _panel_sidebyside(self, res) -> List[np.ndarray]:
        """spatial / pixel: [GT | pred] per step."""
        frames: List[np.ndarray] = []
        pred_frames = res.pred_frames
        gt_seq = res.gt_panel_raw
        gt_scale = 255.0 if gt_seq.max() > 1.5 else 1.0
        for t in range(pred_frames.shape[0]):
            gt_t = img_chw_to_uint8(gt_seq[t] / gt_scale)
            pr_t = img_chw_to_uint8(pred_frames[t])
            gt_t = cv2.resize(
                gt_t, (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            pr_t = cv2.resize(
                pr_t, (self.upscale_to, self.upscale_to),
                interpolation=cv2.INTER_NEAREST,
            )
            frames.append(np.concatenate([gt_t, pr_t], axis=1))
        return frames

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

        outer = algo.outer_stage
        # This eval only drives outer stages that advertise the rollout hook.
        if not hasattr(outer, "rollout_video_episode"):
            return metrics, images_dict

        prefix = outer.video_metric_prefix
        panel = outer.video_panel
        has_extra = bool(getattr(outer, "video_has_extra_metrics", False))

        device = self.trainer.lightning_module.device
        imgs = _batch[self.image_key]
        is_packed = _batch.get("_packed", False)

        # Episode indexing (packed / padded).
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
        extra_sums: Dict[str, torch.Tensor] = {}
        all_frames: List[np.ndarray] = []

        for ep_idx in range(n):
            res = outer.rollout_video_episode(
                self, algo, _batch, emb_id, ep_idx,
                ep_starts[ep_idx], ep_lens[ep_idx], device,
            )

            # ---- per-step MSE vs aligned GT ----
            pred = res.pred_frames
            gt_f = res.gt_for_mse
            n_cmp = min(self.recon_loss_n_frames, pred.shape[0], gt_f.shape[0])
            pred_f = pred[:n_cmp].to(device).float()
            gt_cmp = gt_f[:n_cmp].to(device).float()
            mse_per_step = (
                (pred_f - gt_cmp) ** 2
            ).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for t in range(n_cmp):
                per_step_sse[t] += float(mse_per_step[t])
                per_step_n[t] += 1

            # ---- per-episode extra (perceptual) metrics, summed ----
            for k, v in res.extra_metrics.items():
                extra_sums.setdefault(
                    f"Valid/emb{emb_id}_{prefix}_{k}",
                    torch.tensor(0.0, device=device),
                )
                extra_sums[f"Valid/emb{emb_id}_{prefix}_{k}"] += v

            # ---- panel frames ----
            if panel == "single_t0prepend":
                all_frames.extend(self._panel_single_t0prepend(res))
            elif panel == "sidebyside":
                all_frames.extend(self._panel_sidebyside(res))
            else:
                raise ValueError(f"unknown video_panel {panel!r}")

        # ---- emit per-step + scalar recon MSE ----
        for t in range(self.recon_loss_n_frames):
            if per_step_n[t] > 0:
                metrics[
                    f"Valid/emb{emb_id}_{prefix}_recon_mse_step_{t:02d}"
                ] = torch.tensor(
                    per_step_sse[t] / per_step_n[t], device=device
                )
        if per_step_n.sum() > 0:
            metrics[
                f"Valid/emb{emb_id}_{prefix}_recon_mse_first{self.recon_loss_n_frames}"
            ] = torch.tensor(
                per_step_sse.sum() / per_step_n.sum(), device=device
            )

        # ---- average perceptual metrics over episodes ----
        if has_extra and n > 0:
            for key, total in extra_sums.items():
                metrics[key] = total / n

        if all_frames:
            images_dict[emb_id] = np.stack(all_frames, axis=0)
        return metrics, images_dict


# ---------------------------------------------------------------------------
# Compat aliases (COMBINE A). The unified ``DFoTVideoRolloutEval`` is a true
# superset: every knob each old config passed is still accepted with identical
# semantics, and the family-specific behaviour now lives on the outer stage's
# ``rollout_video_episode`` hook. So the old per-family classes are pure
# redirects — configs that still name them resolve to the unified eval.
# ---------------------------------------------------------------------------
DFoTSpatialVideoRolloutEval = DFoTVideoRolloutEval
DFoTPixelVideoRolloutEval = DFoTVideoRolloutEval
