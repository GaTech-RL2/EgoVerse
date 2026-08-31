import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


class WAMEvalVideo(EvalVideo):
    """Evaluator for WAM (World-Action Model). Per embodiment it writes TWO
    separate validation videos:

      1. ``validation_video_*.mp4`` — predicted vs GT **action** overlay drawn on
         the GT observation frame (via ``viz_func``); BC loss + action MSE metrics.
      2. ``predicted_video_*.mp4``  — the world model's **predicted future frames**
         (sampled jointly with the action chunk, VAE-decoded). Written from the
         frames forward_eval stashes on the algo as ``_eval_frames[eid]``.
    """

    # validation_video (with action-overlay animation):
    #   container fps = 30 (aria native action rate), 192 output frames per
    #   sample. Image updates at 5 fps effective (each downsampled pixel frame
    #   held FRAME_REPEAT=6 output frames); action-arrow trail shrinks 48→1 per
    #   chunk boundary reset.
    # predicted_video (pure model-imagined frames):
    #   fps = 5 (matches the effective 5 fps content rate after video_stride=6),
    #   32 output frames per sample = the model's 32 predicted pixel frames.
    VAL_FPS = 30
    FRAME_REPEAT = 6
    PREDICTED_FPS = 5

    def __init__(
        self,
        limit_val_batches: int = 400,
        viz_func: dict = None,
        transform_lists: dict | None = None,
        teacher_force_rolling: bool = False,
    ):
        """Same signature as ``EvalVideo`` plus ``teacher_force_rolling`` —
        metadata read by the offline eval script's monkey-patch
        (``eval_dreamzero._patch_algo_use_sample_rolling``) to pick the DiT
        rolling mode. Ignored inside this class."""
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
        )
        self.teacher_force_rolling = bool(teacher_force_rolling)

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)  # samples actions + future frames

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss = 0

        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]

            # Capture the full clip BEFORE any collapse — we need it to draw
            # the 192-action trajectory (30 Hz) overlay on every one of the 32
            # downsampled GT frames (5 fps) so validation_video plays 1:1 in
            # sync with predicted_video (same 32 frames per sample at 5 fps).
            clip_by_cam = {}
            for ck in algo.camera_keys.get(embodiment_id, []):
                if (
                    ck in _batch
                    and torch.is_tensor(_batch[ck])
                    and _batch[ck].dim() == 5
                ):
                    clip_by_cam[ck] = _batch[ck]  # (B, T, C, H, W)

            loss_key = f"{name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                total_loss = loss_val if total_loss is None else total_loss + loss_val
                n_loss += 1

            pred_key = f"{name}_{ac_key}"
            if pred_key in preds:
                # MSE aligns pred + gt to their common shorter length. Both
                # can now be longer than 192: gt via bumped dataset
                # action_horizon (for extended val overlay), pred via more
                # rolling steps. Compare only where BOTH exist so shapes match.
                gt_actions = _batch[ac_key]
                pred_full = preds[pred_key]
                T_common = min(gt_actions.shape[1], pred_full.shape[1])
                T_gt = T_common
                pred_actions_cpu = pred_full[:, :T_common].cpu()
                gt_actions_cpu = gt_actions[:, :T_common].cpu()
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    pred_actions_cpu, gt_actions_cpu
                )
                # Per-block paired action MSE (DreamZero convention: report
                # per-block degradation across the horizon so compound error
                # is visible). Blocks sized by DiT's num_action_per_block; the
                # count is action_horizon / num_action_per_block. Skips
                # cleanly when the attr is missing or T_gt not divisible.
                nab = getattr(algo.nets["policy"].dit, "num_action_per_block", None)
                if nab and T_gt >= nab:
                    block_mses = []
                    for k in range(T_gt // nab):
                        b0, b1 = k * nab, (k + 1) * nab
                        # ``.contiguous()`` because MeanSquaredError does an
                        # internal ``view(-1)`` and slicing a (B, T, D) tensor
                        # on the T axis breaks stride-contiguity.
                        block_mse = MeanSquaredError()(
                            pred_actions_cpu[:, b0:b1].contiguous(),
                            gt_actions_cpu[:, b0:b1].contiguous(),
                        )
                        metrics[f"Valid/{pred_key}_paired_mse_block_{k}"] = block_mse
                        block_mses.append(block_mse)
                    # Mean-of-per-block MSE (equivalent to flat _avg for equal-
                    # size blocks, but reported explicitly for clarity + parity
                    # with the PSNR/SSIM _blocks_mean below where it differs).
                    metrics[f"Valid/{pred_key}_paired_mse_blocks_mean"] = torch.stack(
                        [
                            m if torch.is_tensor(m) else torch.tensor(m)
                            for m in block_mses
                        ]
                    ).mean()

            # Per-block video quality metrics (PSNR + SSIM) between the model's
            # decoded future frames and the GT clip — DreamZero convention.
            # ``algo._eval_frames[eid]`` is the pred video: (B, C, T_pred, H, W)
            # in [-1, 1] after the anchor drop. GT clip is (B, T_gt_pix, C, H, W)
            # in [0, 1]. Block k of size (K*4) raw frames (K=num_frame_per_block,
            # VAE compresses 4x temporally); we compare pred[k*K*4:(k+1)*K*4] to
            # gt[1+k*K*4 : 1+(k+1)*K*4] (+1 to skip the GT anchor). Skips
            # gracefully when shapes don't line up or the pred cache is empty.
            pred_video = getattr(algo, "_eval_frames", {}).get(embodiment_id)
            if pred_video is not None and clip_by_cam:
                gt_clip = next(iter(clip_by_cam.values()))  # (B, T_gt_pix, C, H, W)
                # Normalize both to [0, 1] for PSNR/SSIM
                pred_01 = ((pred_video.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
                gt_01 = gt_clip.detach().cpu().clamp(0.0, 1.0)
                # Reorder pred to (B, T, C, H, W) to match GT axis order
                pred_01 = pred_01.permute(0, 2, 1, 3, 4).contiguous()
                # Resize pred to GT spatial size if VAE decode dims differ (e.g. Wan
                # 5B decodes to 480x832; dataset serves 480x640 aria).
                B_p, T_p, C_p, H_p, W_p = pred_01.shape
                B_g, T_g, C_g, H_g, W_g = gt_01.shape
                if (H_p, W_p) != (H_g, W_g):
                    pred_01 = F.interpolate(
                        pred_01.reshape(B_p * T_p, C_p, H_p, W_p),
                        size=(H_g, W_g),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(B_p, T_p, C_p, H_g, W_g)
                dit = algo.nets["policy"].dit
                K_lat = getattr(dit, "num_frame_per_block", 1)
                block_pix = K_lat * 4  # VAE 4x temporal compression
                # Align: pred already dropped anchor; GT still has it at idx 0.
                gt_future = gt_01[:, 1 : 1 + pred_01.shape[1]]
                T_common = min(pred_01.shape[1], gt_future.shape[1])
                if T_common >= block_pix:
                    num_blocks_pix = T_common // block_pix
                    block_psnrs = []
                    block_ssims = []
                    for k in range(num_blocks_pix):
                        p0, p1 = k * block_pix, (k + 1) * block_pix
                        p = pred_01[:, p0:p1].reshape(-1, C_g, H_g, W_g)
                        g = gt_future[:, p0:p1].reshape(-1, C_g, H_g, W_g)
                        # data_range=1.0 since normalized to [0, 1]
                        psnr_k = PeakSignalNoiseRatio(data_range=1.0)(p, g)
                        ssim_k = StructuralSimilarityIndexMeasure(data_range=1.0)(p, g)
                        metrics[f"Valid/{name}_video_psnr_block_{k}"] = psnr_k
                        metrics[f"Valid/{name}_video_ssim_block_{k}"] = ssim_k
                        block_psnrs.append(psnr_k)
                        block_ssims.append(ssim_k)
                    # Flat aggregate (metric computed on all frames concatenated)
                    p_all = pred_01[:, : num_blocks_pix * block_pix].reshape(
                        -1, C_g, H_g, W_g
                    )
                    g_all = gt_future[:, : num_blocks_pix * block_pix].reshape(
                        -1, C_g, H_g, W_g
                    )
                    metrics[f"Valid/{name}_video_psnr_avg"] = PeakSignalNoiseRatio(
                        data_range=1.0
                    )(p_all, g_all)
                    metrics[f"Valid/{name}_video_ssim_avg"] = (
                        StructuralSimilarityIndexMeasure(data_range=1.0)(p_all, g_all)
                    )
                    # Mean-of-per-block PSNR/SSIM (paper convention: mean of
                    # per-block scores. Differs from flat _avg above because
                    # PSNR/SSIM are non-linear — mean-of-log ≠ log-of-mean.)
                    metrics[f"Valid/{name}_video_psnr_blocks_mean"] = torch.stack(
                        [
                            m if torch.is_tensor(m) else torch.tensor(m)
                            for m in block_psnrs
                        ]
                    ).mean()
                    metrics[f"Valid/{name}_video_ssim_blocks_mean"] = torch.stack(
                        [
                            m if torch.is_tensor(m) else torch.tensor(m)
                            for m in block_ssims
                        ]
                    ).mean()

            # Video 1: per-output-frame action-overlay animation.
            # Container fps=30, 192 output frames per sample = M*H (M=4 chunks,
            # H=48 actions/chunk at 30 Hz). Image content updates at 5 fps
            # (each downsampled pixel frame held FRAME_REPEAT=6 output frames).
            # Within each 48-frame chunk window the action-arrow trail SHRINKS
            # by 1 arrow per output frame — starts at all 48 arrows visible,
            # ends at 1. Chunk boundary resets to 48 arrows for the next chunk.
            if clip_by_cam:
                some_ck = next(iter(clip_by_cam))
                T_pix = clip_by_cam[some_ck].shape[1]  # e.g. 33
                # Derive M and H from the DiT config so this stays in sync.
                dit = algo.nets["policy"].dit
                K = getattr(dit, "num_frame_per_block", 1)
                num_action_per_block = getattr(dit, "num_action_per_block", None)
                # Fall back to inferring H if the DiT attr isn't exposed.
                if num_action_per_block is None:
                    total_actions = _batch[ac_key].shape[1]
                    M = (algo.nets["policy"].num_video_frames - 1) // K
                    num_action_per_block = total_actions // M
                # M chunks × FRAME_REPEAT effective-image frames per chunk
                # (since each chunk of K=2 latents decodes to K*(4)=8 raw
                # frames, or K*FRAME_REPEAT/K = FRAME_REPEAT... simpler: total
                # output frames = (T_pix-1) * FRAME_REPEAT, and we tile the
                # M-chunk × H-action pattern across them).
                total_out_frames = (T_pix - 1) * self.FRAME_REPEAT
                H_actions = num_action_per_block  # 48
                pred_key = f"{name}_{ac_key}"
                gt_actions_full = _batch[ac_key]
                pred_actions_full = preds.get(pred_key)

                # -------------------------------------------------------------
                # Temporally-aligned chunk trail:
                #
                # ``start``: shifted by +FRAME_REPEAT so that at each moment
                # the pixel updates (f % FR == 0), the arrow's origin action
                # index matches the current pixel's source time
                # (action[pixel_idx * FR] = hand position at pixel_idx). Then
                # advances by 1 per output frame so the trail shrinks smoothly.
                #
                # ``end``: fixed at the end of the current 48-action DiT chunk
                # (also +FR-shifted). Chunk boundaries in animation time land
                # at f = 0, 48, 96, 144 — exactly the output frames where the
                # arrow origin lines up with pixel_idx = 1, 9, 17, 25 (source
                # time 0.2, 1.8, 3.4, 5.0 s). At each of these boundaries the
                # trail resets to full length (H_actions arrows), then shrinks
                # 48 → 1 as ``start`` advances through the chunk.
                # -------------------------------------------------------------
                FR = self.FRAME_REPEAT
                # GT actions come from the dataset (action_horizon=192, ROPE-
                # locked at training time). Pred actions come from the rolling
                # sampler and are ``num_steps * num_action_per_block`` long — for
                # eva longeva (cam_horizon=99 → 3× num_steps) that's ~576. If we
                # clamp ``start``/``end`` to n_gt for BOTH slices (as we used
                # to), the pred trail gets truncated to gt's 6.4 s window even
                # though pred has data all the way out to the extended pixel
                # video's end. Track n_gt and n_pred separately, extend the trail
                # over max(n_gt, n_pred), and slice each source with its own
                # length. Past n_gt we simply stop drawing GT arrows.
                n_gt = gt_actions_full.shape[1]
                n_pred = (
                    pred_actions_full.shape[1] if pred_actions_full is not None else 0
                )
                _ = max(
                    n_gt, n_pred
                )  # kept for symmetry; loop uses n_gt/n_pred directly
                # Per-raw-frame head-pose chunk (see Mecka.get_wam_keymap ->
                # ``obs_head_pose_chunk``). Shape (B, cam_horizon, 7) xyzwxyz
                # after SubsampleKeys; row 0 == the pose the data-pipeline
                # transform used as target (H_0), row k == the pose current at
                # pixel_idx=k. If absent (embodiments without head pose in
                # zarr), we skip the per-frame reprojection and fall back to
                # H_0-frame actions unchanged.
                head_pose_chunk = _batch.get("obs_head_pose_chunk")

                per_frame_ims = []
                for f in range(total_out_frames):
                    # Which downsampled pixel frame to display (5 fps effective).
                    pixel_idx = 1 + f // FR  # skip frame 0=anchor
                    # Origin of the trail (shifted +FR to align with pixel).
                    start = f + FR
                    # Chunk-end of the trail (also +FR-shifted). Which
                    # animation-chunk we're in resets every H_actions frames.
                    chunk_bucket = f // H_actions
                    end = (chunk_bucket + 1) * H_actions + FR

                    _batch_f = dict(_batch)
                    for ck, clip in clip_by_cam.items():
                        # pixel_idx can now overshoot the read clip (rare — only
                        # when total_out_frames rounds past T_pix); freeze on
                        # the last pixel frame so the container keeps playing.
                        clip_t = clip[:, min(pixel_idx, clip.shape[1] - 1)]
                        _batch_f[ck] = clip_t  # (B, C, H, W)

                    # GT slice: clamp to n_gt; None once we've run past GT.
                    if start < n_gt:
                        gt_end = min(end, n_gt)
                        gt_slice = gt_actions_full[:, start:gt_end]
                    else:
                        gt_slice = None
                    # Pred slice: clamp to n_pred (extends further than gt).
                    if pred_actions_full is not None and start < n_pred:
                        pred_end = min(end, n_pred)
                        pred_slice = pred_actions_full[:, start:pred_end]
                    else:
                        pred_slice = None

                    # ----- Per-pixel head-frame reprojection ----------------
                    # Actions in ``_batch[ac_key]`` are in the frame-0 head
                    # frame (data pipeline anchors them to ``obs_head_pose``
                    # at the sample start). But the image at pixel_idx k was
                    # captured through the head/camera AT THAT LATER TIME —
                    # so the arrow drifts as the head moves. Convert the
                    # 12D chunk ``[L xyz L ypr R xyz R ypr]`` from H_0 frame
                    # to H_t frame before drawing, using the per-episode head
                    # pose chunk from the batch.
                    if (
                        head_pose_chunk is not None
                        and pixel_idx < head_pose_chunk.shape[1]
                    ):
                        if gt_slice is not None:
                            gt_slice = self._reproject_actions_to_head_t(
                                gt_slice, head_pose_chunk, pixel_idx
                            )
                        if pred_slice is not None:
                            pred_slice = self._reproject_actions_to_head_t(
                                pred_slice, head_pose_chunk, pixel_idx
                            )

                    # ``gt_slice`` may be None once we've run past n_gt (past
                    # the 192-action / 6.4 s dataset window). Emit an empty
                    # (B, 0, D) chunk so the renderer draws zero GT arrows for
                    # these frames while still drawing pred arrows.
                    if gt_slice is not None:
                        _batch_f[ac_key] = gt_slice
                    else:
                        _batch_f[ac_key] = gt_actions_full[:, :0]

                    preds_f = dict(preds)
                    if pred_slice is not None:
                        preds_f[pred_key] = pred_slice
                    else:
                        preds_f[pred_key] = (
                            pred_actions_full[:, :0]
                            if pred_actions_full is not None
                            else None
                        )

                    per_frame_ims.append(
                        self._visualize_preds(preds_f, _batch_f)  # (B, H, W, 3)
                    )
                stacked = np.stack(per_frame_ims, axis=0)  # (F, B, H, W, 3)
                stacked = stacked.transpose(1, 0, 2, 3, 4)  # (B, F, H, W, 3)
                B_, F_, H_, W_, C_ = stacked.shape
                images_dict[embodiment_id] = stacked.reshape(B_ * F_, H_, W_, C_)

                # ---- predicted_video overlay ----
                # Also render the ACTION OVERLAY on top of the model's imagined
                # future frames (not just the GT clip), so ``predicted_video_*.mp4``
                # shows what the DiT rolled out AND the action trajectory it
                # produced overlaid at each pixel step. Same per-frame slicing +
                # head-pose reprojection as the val_video loop above; base image
                # is the pred VAE-decoded frame instead of the GT clip frame.
                #
                # RANK-GUARD: this loop calls ``viz_func`` T_pred times per
                # sample (matplotlib/PIL under the hood). On 4×DDP training val
                # loops the per-rank time variance was tripping the 30 min
                # NCCL watchdog on the next sync collective. Only rank 0 does
                # the overlay compute + writes — other ranks skip and let their
                # ``predicted_video`` be the raw (non-overlay) VAE decode via
                # the ``_predicted_frames_all`` fallback in ``on_validation_step``.
                is_rank0 = (
                    (not torch.distributed.is_available())
                    or (not torch.distributed.is_initialized())
                    or (torch.distributed.get_rank() == 0)
                )
                pred_video_t = getattr(self.model, "_eval_frames", {}).get(
                    embodiment_id
                )
                if is_rank0 and pred_video_t is not None and clip_by_cam:
                    image_key = next(iter(clip_by_cam))
                    ref_H = clip_by_cam[image_key].shape[-2]
                    ref_W = clip_by_cam[image_key].shape[-1]
                    # (B, C, T_pred, H, W) [-1,1] → per-frame batches at
                    # GT-clip resolution so ``viz_func`` (uses intrinsics of
                    # the clip) draws arrows at the correct pixel positions.
                    Bp, Cp, T_pred, _, _ = pred_video_t.shape
                    pv = F.interpolate(
                        pred_video_t.permute(0, 2, 1, 3, 4).reshape(
                            Bp * T_pred,
                            Cp,
                            pred_video_t.shape[-2],
                            pred_video_t.shape[-1],
                        ),
                        size=(ref_H, ref_W),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(Bp, T_pred, Cp, ref_H, ref_W)
                    # Normalize to [0, 1] (viz_func expects the same range as
                    # ``clip_by_cam`` which comes from the data pipeline already
                    # in [0, 1] after JPEG-decode).
                    pv = (pv.clamp(-1, 1) + 1.0) * 0.5
                    pred_overlay_frames = []
                    for p in range(T_pred):
                        # pred_video_t already dropped the anchor in
                        # eval_dreamzero.val_rollout (viz_video = pred_frames[:, :, 1:]),
                        # so p=0 corresponds to pixel_idx=1 (first predicted frame).
                        pixel_idx = p + 1
                        f_val = p * FR  # equivalent frame in val_video's 30fps timeline
                        start = f_val + FR
                        chunk_bucket = f_val // H_actions
                        end = (chunk_bucket + 1) * H_actions + FR

                        if start < n_gt:
                            gt_slice_p = gt_actions_full[:, start : min(end, n_gt)]
                        else:
                            gt_slice_p = None
                        if pred_actions_full is not None and start < n_pred:
                            pred_slice_p = pred_actions_full[
                                :, start : min(end, n_pred)
                            ]
                        else:
                            pred_slice_p = None
                        if (
                            head_pose_chunk is not None
                            and pixel_idx < head_pose_chunk.shape[1]
                        ):
                            if gt_slice_p is not None:
                                gt_slice_p = self._reproject_actions_to_head_t(
                                    gt_slice_p, head_pose_chunk, pixel_idx
                                )
                            if pred_slice_p is not None:
                                pred_slice_p = self._reproject_actions_to_head_t(
                                    pred_slice_p, head_pose_chunk, pixel_idx
                                )
                        _batch_p = dict(_batch)
                        _batch_p[image_key] = pv[:, p]  # (B, C, H, W)
                        _batch_p[ac_key] = (
                            gt_slice_p
                            if gt_slice_p is not None
                            else gt_actions_full[:, :0]
                        )
                        preds_p = dict(preds)
                        preds_p[pred_key] = (
                            pred_slice_p
                            if pred_slice_p is not None
                            else (
                                pred_actions_full[:, :0]
                                if pred_actions_full is not None
                                else None
                            )
                        )
                        pred_overlay_frames.append(
                            self._visualize_preds(preds_p, _batch_p)  # (B, H, W, 3)
                        )
                    p_stacked = np.stack(
                        pred_overlay_frames, axis=0
                    )  # (T_pred, B, H, W, 3)
                    p_stacked = p_stacked.transpose(
                        1, 0, 2, 3, 4
                    )  # (B, T_pred, H, W, 3)
                    Bp_, Tp_, Hp_, Wp_, Cp_ = p_stacked.shape
                    if not hasattr(self, "_pred_overlay_cache"):
                        self._pred_overlay_cache = {}
                    # Cache as (B*T_pred, H, W, 3) so on_validation_step can
                    # extend ``_pred_buffer`` with the same layout that the
                    # existing (no-overlay) code path produces.
                    self._pred_overlay_cache[embodiment_id] = p_stacked.reshape(
                        Bp_ * Tp_, Hp_, Wp_, Cp_
                    )
            else:
                images_dict[embodiment_id] = self._visualize_preds(preds, _batch)

        if total_loss is not None and n_loss > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        name = get_embodiment(batch["embodiment"][0].item()).lower()
        return self.viz_func[name](predictions, batch)

    # --- Video 2: predicted future-frame video ----------------------------
    # Every val sample contributes its full 16-frame prediction (all frames
    # from one sample() call). Order per batch: sample0's 16 frames, sample1's
    # 16 frames, ..., sample{B-1}'s 16 frames. Re-conditioning happens at each
    # new sample using that sample's GT[0]. Written at 30 fps ~= source frame
    # rate of aria clips.
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Base writes the action-overlay video + logs metrics (it calls
        # compute_metrics_and_viz, which runs forward_eval and stashes frames).
        super().on_validation_step(batch, batch_idx, dataloader_idx)
        if not hasattr(self, "_pred_buffer"):
            self._pred_buffer = {}
            # Per-embodiment count of val-step contributions. Each on_validation_step
            # call with a non-None _eval_frames[eid] adds ONE sample's frames to the
            # buffer. Tracking this makes on_validation_end's mp4-per-sample split
            # robust to per-embodiment ``cam_horizon`` overrides (e.g. longeva eval
            # where eva uses cam_horizon=99 instead of 33 → 3× more pred frames per
            # sample → the old fixed ``pred_frames_per_sample = 4F-4 = 32`` split
            # each eva episode across 3 mp4s).
            self._pred_step_count = {}
            self._val_step_count = {}
        for eid, video in getattr(self.model, "_eval_frames", {}).items():
            if video is None:
                continue
            # Prefer the action-overlay-annotated pred frames built inside
            # compute_metrics_and_viz (arrows drawn on the model's imagined
            # frames, at GT-clip resolution so intrinsics line up). Fall back
            # to raw upscaled pred frames if the overlay cache is missing —
            # e.g. clip_by_cam was empty for this batch.
            overlay = getattr(self, "_pred_overlay_cache", {}).pop(eid, None)
            if overlay is not None:
                frames = overlay
            else:
                frames = self._predicted_frames_all(video)  # (B*T, Hd, Wd, 3)
            self._pred_buffer.setdefault(eid, []).extend(torch.from_numpy(frames))
            self._pred_step_count[eid] = self._pred_step_count.get(eid, 0) + 1
        # Also mirror the val-overlay buffer count so per-embodiment cam_horizon
        # is honored when slicing ``val_image_buffer`` in on_validation_end.
        for key in self.val_image_buffer.keys():
            self._val_step_count[key] = self._val_step_count.get(key, 0) + 1

    def on_validation_end(self):
        # RANK-GUARD: h264 encode + shared-FS writes ran on all 4 DDP ranks and
        # caused significant per-rank divergence. Since each rank's buffer only
        # holds frames from its OWN subset of val samples anyway (DistributedSampler
        # shards val), rank-0-only writes lose 3/4 of the videos — which is fine
        # for training-time val (a sample of predictions is enough for eyeballing).
        # Offline eval already runs on a single rank so this guard is a no-op there.
        is_rank0 = (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or (torch.distributed.get_rank() == 0)
        )
        if not is_rank0:
            # Still clear the buffers so we don't leak memory across val loops.
            self._pred_buffer = {}
            self._pred_step_count = {}
            for key in list(self.val_image_buffer.keys()):
                self.val_image_buffer[key] = []
                self.val_counter[key] = 0
            self._val_step_count = {}
            return
        # PREDICTED: one mp4 per sample. Sample size is inferred from
        # ``len(buf) // step_count`` (per eid) so an embodiment with a larger
        # cam_horizon (e.g. longeva eva) still gets ONE mp4 per episode instead
        # of being sliced by a training-time constant.
        for eid, buf in getattr(self, "_pred_buffer", {}).items():
            if not buf:
                continue
            n_samples = getattr(self, "_pred_step_count", {}).get(eid, 0)
            if n_samples <= 0:
                continue
            frames_per_sample = len(buf) // n_samples
            if frames_per_sample <= 0:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(eid)),
            )
            os.makedirs(out_dir, exist_ok=True)
            for s in range(n_samples):
                frames = torch.stack(
                    buf[s * frames_per_sample : (s + 1) * frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"predicted_video_{s}.mp4"),
                    frames,
                    fps=self.PREDICTED_FPS,
                    video_codec="h264",
                )
        self._pred_buffer = {}
        self._pred_step_count = {}

        # VAL OVERLAY: one mp4 per sample, similarly inferred from step-count.
        for key, buffer in self.val_image_buffer.items():
            if not buffer:
                continue
            n_samples = getattr(self, "_val_step_count", {}).get(key, 0)
            if n_samples <= 0:
                continue
            frames_per_sample = len(buffer) // n_samples
            if frames_per_sample <= 0:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(key)),
            )
            os.makedirs(out_dir, exist_ok=True)
            for s in range(n_samples):
                frames = torch.stack(
                    buffer[s * frames_per_sample : (s + 1) * frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"validation_video_{s}.mp4"),
                    frames,
                    fps=self.VAL_FPS,
                    video_codec="h264",
                )
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []
        self._val_step_count = {}

    @staticmethod
    def _reproject_actions_to_head_t(
        actions_slice: torch.Tensor,
        head_pose_chunk: torch.Tensor,
        pixel_idx: int,
    ) -> torch.Tensor:
        """Rotate an action chunk from H_0 head frame to H_t head frame so the
        arrow lands on the hand as the aria head moves during the sample window.

        Actions in ``actions_slice`` are 12D ``[L xyz L ypr R xyz R ypr]`` in the
        frame-0 head frame (the data pipeline's ``ActionChunkCoordinateFrameTransform``
        target). This helper composes ``T = H_t^-1 @ H_0`` and applies it to
        the left/right XYZ blocks (dims 0:3 and 6:9) — YPR blocks are left
        untouched because the WAM viz uses ``mode="traj"`` which only reads XYZ.

        Args:
            actions_slice: (B, T_chunk, 12) — action slice for the current
                animation window in H_0 frame.
            head_pose_chunk: (B, cam_horizon, 7) xyzwxyz — per-pixel head pose
                chunk. ``[:, 0]`` is H_0 (the transform's reference),
                ``[:, pixel_idx]`` is H_t (current pixel's head pose).
            pixel_idx: int — index into ``head_pose_chunk`` for the current pixel.

        Returns: (B, T_chunk, 12) with L_xyz and R_xyz reprojected; YPR unchanged.
        """
        from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

        was_tensor = isinstance(actions_slice, torch.Tensor)
        a = (
            actions_slice.detach().cpu().numpy()
            if was_tensor
            else np.asarray(actions_slice)
        )
        hpc = (
            head_pose_chunk.detach().cpu().numpy()
            if isinstance(head_pose_chunk, torch.Tensor)
            else np.asarray(head_pose_chunk)
        )
        B, T_chunk, D = a.shape
        result = a.copy()
        for b in range(B):
            H_0_mat = _xyzwxyz_to_matrix(hpc[b, 0][None])[0]  # (4, 4)
            H_t_mat = _xyzwxyz_to_matrix(hpc[b, pixel_idx][None])[0]  # (4, 4)
            try:
                T_ht_h0 = np.linalg.inv(H_t_mat) @ H_0_mat  # 4x4 h_0 -> h_t
            except np.linalg.LinAlgError:
                continue  # degenerate pose — leave this sample as-is
            for arm_start in (0, 6):  # L_xyz at 0:3, R_xyz at 6:9
                xyz = a[b, :, arm_start : arm_start + 3]  # (T_chunk, 3)
                xyz_h = np.concatenate(
                    [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1
                )
                xyz_t = (T_ht_h0 @ xyz_h.T).T[:, :3].astype(xyz.dtype)
                result[b, :, arm_start : arm_start + 3] = xyz_t
        if was_tensor:
            return (
                torch.from_numpy(result)
                .to(actions_slice.dtype)
                .to(actions_slice.device)
            )
        return result

    def _predicted_frames_all(self, video: torch.Tensor, hw=(360, 640)) -> np.ndarray:
        """(B, C, T, H, W) in [-1, 1] -> (B*T, Hd, Wd, 3) uint8. No temporal
        duplication — the 32 predicted pixel frames per sample are shown 1:1
        at PREDICTED_FPS=5 so real-time playback matches the video_stride=6
        downsampling used at data-load time."""
        B, C, T, H, W = video.shape
        v = video.clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        v = F.interpolate(v, size=tuple(hw), mode="bilinear", align_corners=False)
        v = ((v + 1.0) * 127.5).to(torch.uint8).cpu()
        return v.permute(0, 2, 3, 1).numpy()  # (B*T, Hd, Wd, 3)
