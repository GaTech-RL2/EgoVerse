import copy
import os
from collections.abc import Mapping

import torch
import torchvision.io as tvio
from lightning.pytorch.loggers import WandbLogger

from egomimic.eval.action_metrics import layout_metrics
from egomimic.eval.eval import Eval
from egomimic.eval.replay_viz import ReplayClip, render_replay
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment

# viz_gt_preds kwargs that ReplayClip / render_replay consume themselves.
_REPLAY_OWN_KWARGS = {
    "image_key",
    "action_key",
    "annotation_key",
    "mode",
    "gt_alpha",
    "pred_alpha",
}


class EvalVideo(Eval):
    """
    Base evaluator that buffers per-embodiment frames and writes them out as
    validation videos. ``compute_metrics_and_viz`` computes the metrics and
    produces the frames to buffer; subclasses add metrics via
    ``_head_metrics`` / ``_cam_metrics``.
    """

    # A pinned video loader plays each chunk as GT then prediction
    # (egomimic/eval/replay_viz.py), capped at REPLAY_CHUNKS chunks per pass;
    # a non-keypoint layout draws the last REPLAY_TRAIL steps of each.
    REPLAY_CHUNKS = 5
    REPLAY_TRAIL = 5

    def __init__(
        self,
        limit_val_batches: int = 400,
        viz_func: dict = None,
        transform_lists: dict | None = None,
        viz_every_n_epochs: int = 1,
        viz_max_batches: int | None = None,
        action_stride: float | dict | None = None,
        prefix: str | None = None,
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        # Render the overlay video only on validation passes where
        # (current_epoch + 1) is a multiple of this; metrics log every pass.
        self.viz_every_n_epochs = viz_every_n_epochs
        # On viz epochs, render overlay frames for only the first N val
        # batches (None = all). Rendering is CPU-bound -- ~0.1 s/frame, 4864
        # frames in 8.4 min with four ranks rendering at once -- and dominates
        # viz-epoch wall time; metrics are still computed on EVERY batch
        # regardless. Required on heads with pinned video loaders (trainHydra).
        self.viz_max_batches = viz_max_batches
        # The replay needs consecutive frames, so it runs only on a pinned
        # video loader; other loaders draw each frame's whole GT and predicted
        # chunk. ``action_stride`` is video frames per chunk step (fractional
        # when the chunk is resampled): one number, or ``{embodiment name:
        # stride}`` as trainHydra._action_strides derives it when unset.
        self._replay_now = False
        self._replay_seen = {}  # embodiment id -> chunks buffered this pass
        self.action_stride = action_stride
        # ``<prefix>/...`` metric keys and ``videos_<prefix>/``, so the
        # train_viz / opsplit heads do not collide with the canonical one.
        self.prefix = prefix
        # Per-embodiment list[Transform] applied once during eval to project
        # the model's wrist-frame actions back into cam (head) frame for the
        # viz video.
        self.transform_lists = transform_lists or {}
        self.val_image_buffer = {}
        self.val_counter = {}
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def video_dir(self):
        return os.path.join(
            self.root_dir(), f"videos_{self.prefix}" if self.prefix else "videos"
        )

    def _should_viz(self) -> bool:
        if not self.viz_every_n_epochs or self.viz_every_n_epochs <= 0:
            return False
        # Lightning runs validation when (current_epoch + 1) is a multiple of
        # check_val_every_n_epoch, with current_epoch still the pre-increment
        # value during the val hooks (19, 39, ...). Gate on the same (+1)
        # convention: a plain `current_epoch % n` never aligns with a
        # validation epoch unless check_val_every_n_epoch == 1.
        epoch = self.trainer.current_epoch + 1
        # The last epoch always renders, so short runs (trainer=debug, eval
        # mode's single validate) get a video without retuning the interval.
        max_epochs = getattr(self.trainer, "max_epochs", None)
        if max_epochs is not None and max_epochs > 0 and epoch == max_epochs:
            return True
        return epoch % self.viz_every_n_epochs == 0

    def compute_metrics_and_viz(self, batch, do_viz=True):
        """
        Run the model's eval forward and, per embodiment on its native
        (unnormalized) output, compute the val loss (``Valid/<name>_loss``,
        averaged into ``Valid/action_loss``) and ``_head_metrics``. Those are
        frame-invariant, so the ``transform_lists`` revert (back to cam frame
        for the overlay) runs only when rendering (``do_viz``).

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
            do_viz (bool): render overlay frames for this batch. When False the
                returned ``images_dict`` must be empty.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        algo = self.model
        preds = algo.forward_eval(batch)
        metrics, images_dict, losses = {}, {}, []
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{name}_{ac_key}"
            if f"{name}_loss" in preds:
                losses.append(preds[f"{name}_loss"])
                metrics[f"Valid/{name}_loss"] = losses[-1]
            metrics.update(self._head_metrics(preds, _batch, name, ac_key))

            if not do_viz or (self.viz_func is not None and name not in self.viz_func):
                # No overlay for this embodiment (e.g. joint-space actions,
                # which have no image projection): metrics only.
                continue
            gt_viz, preds_viz = _batch, preds
            transform_list = self.transform_lists.get(name)
            if transform_list is not None and pred_key in preds:
                pred_batch = copy.deepcopy(_batch)
                pred_batch[ac_key] = preds[pred_key]
                # apply_transform drops keys whose shape[0] != batch_size (e.g.
                # ``embodiment``, ``annotations``). Merge to preserve them.
                gt_viz = {
                    **_batch,
                    **Embodiment.apply_transform(_batch, transform_list),
                }
                pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                preds_viz = {**preds, pred_key: pred_t[ac_key]}
                metrics.update(
                    self._cam_metrics(pred_t[ac_key], gt_viz[ac_key], pred_key)
                )
            images_dict[embodiment_id] = self._visualize_preds(preds_viz, gt_viz)

        if losses:
            metrics["Valid/action_loss"] = sum(losses) / len(losses)
        return metrics, images_dict

    def _head_metrics(self, preds, batch, name, ac_key) -> dict:
        """One embodiment's metrics: the layout metrics of
        :mod:`egomimic.eval.action_metrics` on its main prediction."""
        pred_key = f"{name}_{ac_key}"
        if pred_key not in preds:
            return {}
        return layout_metrics(
            preds[pred_key], batch[ac_key], f"Valid/{pred_key}", ac_key
        )

    def _cam_metrics(self, pred, gt, pred_key) -> dict:
        """Metrics of the main prediction after the cam-frame revert (viz
        batches only); none by default."""
        return {}

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_name = get_embodiment(batch["embodiment"][0].item()).lower()
        viz_fn = self.viz_func[embodiment_name]
        if self._replay_now:
            return ReplayClip.from_batch(viz_fn, predictions, batch)
        return viz_fn(predictions, batch)

    def _stride(self, key) -> float:
        """Video frames per chunk step for embodiment ``key`` (1 if unknown)."""
        stride = self.action_stride or 1
        if isinstance(stride, Mapping):
            return stride.get(get_embodiment(key).lower(), 1)
        return stride

    def _render_replay(self, key, final: bool):
        """Render the buffered clip's complete chunks; keep the rest unless final."""
        clips = self.val_image_buffer[key]
        if not clips:
            return None
        clip = ReplayClip.concat(clips)
        viz_fn = self.viz_func[get_embodiment(key).lower()]
        frames, used = render_replay(
            clip,
            embodiment_cls=viz_fn.func.__self__,
            mode=viz_fn.keywords["mode"],
            stride=self._stride(key),
            trail=self.REPLAY_TRAIL,
            viz_kwargs={
                k: v for k, v in viz_fn.keywords.items() if k not in _REPLAY_OWN_KWARGS
            },
        )
        self.val_image_buffer[key] = [] if final else [clip.tail(used)]
        return torch.from_numpy(frames) if len(frames) else None

    def _is_replay_buffer(self, key) -> bool:
        buffer = self.val_image_buffer[key]
        return bool(buffer) and isinstance(buffer[0], ReplayClip)

    def _replay_done(self) -> bool:
        return bool(self._replay_seen) and all(
            n >= self.REPLAY_CHUNKS for n in self._replay_seen.values()
        )

    def _buffered_frames(self, key) -> int:
        # Overlay buffers hold one (H, W, 3) tensor per frame, not clips.
        if self._is_replay_buffer(key):
            return sum(len(c) for c in self.val_image_buffer[key])
        return len(self.val_image_buffer[key])

    def _flush(self, key, final: bool) -> None:
        if self._is_replay_buffer(key):
            frames = self._render_replay(key, final)
        else:
            buffer = self.val_image_buffer[key]
            frames = torch.stack(buffer) if buffer else None
            self.val_image_buffer[key] = []
        if frames is not None:
            self._write_video(key, frames)
            self.val_counter[key] += 1

    def _write_video(self, key, frames) -> None:
        # Every rank runs every val batch (Lightning puts no DistributedSampler
        # on the CombinedLoader val heads; measured 2026-09-16: 8 ranks = 8 x
        # 223 steps), so all ranks render the same frames to the same path:
        # rank 0 alone writes them, at the 30 fps source rate (a world_size
        # division once made every multi-GPU video a slow-motion).
        if not self.trainer.is_global_zero:
            return
        path = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(key)),
            f"validation_video_{self.val_counter[key]}.mp4",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tvio.write_video(path, frames, fps=30, video_codec="h264")
        self._upload_video(key, path)

    def _upload_video(self, key, path) -> None:
        name = f"{os.path.basename(self.video_dir())}/{get_embodiment(key)}"
        if self.val_counter[key]:
            name += f"_{self.val_counter[key]}"
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_video(
                    name,
                    [path],
                    step=self.trainer.global_step,
                    caption=[f"epoch {self.trainer.current_epoch}"],
                    format=["mp4"],
                )

    def on_validation_start(self):
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        if not self._should_viz():
            return
        for key in list(self.val_image_buffer):
            self._flush(key, final=True)
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []
        self._replay_seen = {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        """``mode`` splits metrics from video so one head can serve two loaders:

        * ``"metrics"`` -- the per-episode subsampled metric loader. Never
          renders or buffers a frame (rendering is ~1 s/frame of CPU, and a
          subsampled video would be an incoherent time-lapse anyway).
        * ``"video"`` -- the contiguous pinned-episode loader. Renders and
          buffers, and logs NOTHING, so no ``Valid/...`` key is averaged over
          it. On a non-viz epoch, or past its cap (``viz_max_batches``, or
          ``REPLAY_CHUNKS`` replayed chunks), it returns before the
          forward pass: there is nothing such a call could produce.
        * ``"both"`` (default) -- today's single-loader behaviour, which is what
          a data config without ``video_episodes`` still gets.
        """
        if mode not in ("both", "metrics", "video"):
            raise ValueError(f"unknown validation mode {mode!r}")
        replay = self._replay_now = mode == "video"
        if replay:
            # viz_max_batches still bounds a loader that yields no clips.
            nothing_yet = not self._replay_seen and (
                self.viz_max_batches is not None and batch_idx >= self.viz_max_batches
            )
            past_viz_cap = self._replay_done() or nothing_yet
        else:
            past_viz_cap = (
                self.viz_max_batches is not None and batch_idx >= self.viz_max_batches
            )
        if mode == "video" and (past_viz_cap or not self._should_viz()):
            return
        do_viz = mode != "metrics" and self._should_viz() and not past_viz_cap
        metrics, images_dict = self.compute_metrics_and_viz(batch, do_viz=do_viz)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        if do_viz:
            for key, images in images_dict.items():
                if (
                    key not in self.val_image_buffer
                    or self.val_image_buffer[key] is None
                ):
                    self.val_image_buffer[key] = []
                    self.val_counter[key] = 0
                if isinstance(images, ReplayClip):
                    self.val_image_buffer[key].append(images)
                    chunk = images.gt.shape[1] * self._stride(key)  # frames
                    self._replay_seen[key] = (
                        self._replay_seen.get(key, 0) + len(images) / chunk
                    )
                else:
                    self.val_image_buffer[key].extend(torch.from_numpy(images))
                if self._buffered_frames(key) >= 1000:
                    self._flush(key, final=False)

        if mode == "video":
            # Video-only loader: the forward pass was for the overlay. Logging
            # here would mix pinned-episode numbers into the head's metric and
            # (with add_dataloader_idx=False) collide key-for-key with it.
            return

        # add_dataloader_idx=False: with the train_viz loader Lightning would
        # otherwise suffix every key with "/dataloader_idx_N"; the heads
        # prefix their keys themselves.
        if self.prefix:
            metrics = {f"{self.prefix}/{k}": v for k, v in metrics.items()}
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )
