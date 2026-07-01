import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torchmetrics import MeanSquaredError

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
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )

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

                per_frame_ims = []
                for f in range(total_out_frames):
                    # Which downsampled pixel frame to display (5 fps effective).
                    pixel_idx = 1 + f // self.FRAME_REPEAT  # skip frame 0=anchor
                    # Which chunk + intra-chunk offset (j) the arrow trail is in.
                    k = f // H_actions
                    j = f % H_actions
                    start = k * H_actions + j
                    end = (k + 1) * H_actions

                    _batch_f = dict(_batch)
                    for ck, clip in clip_by_cam.items():
                        _batch_f[ck] = clip[:, pixel_idx]  # (B, C, H, W)
                    _batch_f[ac_key] = gt_actions_full[:, start:end]

                    preds_f = dict(preds)
                    if pred_actions_full is not None:
                        preds_f[pred_key] = pred_actions_full[:, start:end]

                    per_frame_ims.append(
                        self._visualize_preds(preds_f, _batch_f)  # (B, H, W, 3)
                    )
                stacked = np.stack(per_frame_ims, axis=0)  # (F, B, H, W, 3)
                stacked = stacked.transpose(1, 0, 2, 3, 4)  # (B, F, H, W, 3)
                B_, F_, H_, W_, C_ = stacked.shape
                images_dict[embodiment_id] = stacked.reshape(B_ * F_, H_, W_, C_)
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
        for eid, video in getattr(self.model, "_eval_frames", {}).items():
            if video is None:
                continue
            frames = self._predicted_frames_all(video)  # (B*T, Hd, Wd, 3)
            self._pred_buffer.setdefault(eid, []).extend(torch.from_numpy(frames))

    def on_validation_end(self):
        # PREDICTED: one mp4 per sample (32 frames/sample at PREDICTED_FPS=5).
        pred_frames_per_sample = (
            4 * self.model.nets["policy"].num_video_frames - 4
        )  # 4·F-3-1 = 32
        for eid, buf in getattr(self, "_pred_buffer", {}).items():
            if not buf:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(eid)),
            )
            os.makedirs(out_dir, exist_ok=True)
            n_samples = len(buf) // pred_frames_per_sample
            for s in range(n_samples):
                frames = torch.stack(
                    buf[s * pred_frames_per_sample : (s + 1) * pred_frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"predicted_video_{s}.mp4"),
                    frames,
                    fps=self.PREDICTED_FPS,
                    video_codec="h264",
                )
        self._pred_buffer = {}

        # VAL OVERLAY: one mp4 per sample. val_image_buffer has (T_pix-1)*FRAME_REPEAT
        # frames per sample; for cam_horizon=33 & FRAME_REPEAT=6 that's 32·6=192.
        # We infer from the DiT config to stay in sync with the animation loop.
        dit = self.model.nets["policy"].dit
        K = getattr(dit, "num_frame_per_block", 1)
        M = (self.model.nets["policy"].num_video_frames - 1) // K
        H = getattr(dit, "num_action_per_block", None)
        val_frames_per_sample = M * H if H is not None else None  # 4·48 = 192

        for key, buffer in self.val_image_buffer.items():
            if not buffer or val_frames_per_sample is None:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(key)),
            )
            os.makedirs(out_dir, exist_ok=True)
            n_samples = len(buffer) // val_frames_per_sample
            for s in range(n_samples):
                frames = torch.stack(
                    buffer[s * val_frames_per_sample : (s + 1) * val_frames_per_sample]
                )
                tvio.write_video(
                    os.path.join(out_dir, f"validation_video_{s}.mp4"),
                    frames,
                    fps=self.VAL_FPS,
                    video_codec="h264",
                )
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

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
