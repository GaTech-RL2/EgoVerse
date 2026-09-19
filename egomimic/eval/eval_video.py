import os
from abc import abstractmethod

import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment


class EvalVideo(Eval):
    """
    Base evaluator that buffers per-embodiment frames and writes them out as
    validation videos. Subclasses implement `compute_metrics_and_viz` to compute
    model-specific metrics and produce the frames to buffer.
    """

    def __init__(
        self,
        limit_val_batches: int = 400,
        viz_func: dict = None,
        transform_lists: dict | None = None,
        viz_every_n_epochs: int = 1,
        viz_max_batches: int | None = None,
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        # Render the overlay video only on validation passes where
        # (current_epoch + 1) is a multiple of this; metrics log every pass.
        self.viz_every_n_epochs = viz_every_n_epochs
        # On viz epochs, render overlay frames for only the first N val
        # batches (None = all). Rendering is CPU-bound (~1s/frame) and
        # dominates viz-epoch wall time; metrics are still computed on EVERY
        # batch regardless.
        self.viz_max_batches = viz_max_batches
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
        return os.path.join(self.root_dir(), "videos")

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

    @abstractmethod
    def compute_metrics_and_viz(self, batch, do_viz=True):
        """
        Run the model's eval forward and compute metrics and visualization frames.

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
            do_viz (bool): render overlay frames for this batch. When False the
                returned ``images_dict`` must be empty.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    def _video_fps(self, source_fps: int = 30) -> int:
        """Playback fps of the overlay video: the source rate.

        The val heads come back from ``val_dataloader()`` inside
        ``CombinedLoader``s, which Lightning does NOT wrap in a
        ``DistributedSampler`` (every rank runs every val batch; measured
        2026-09-16: 1 rank = 223 val steps, 8 ranks = 8 x 223), so rank 0
        renders every source frame and the video plays in real time at the
        source rate. An earlier version divided by ``world_size`` to undo a
        sampler stride that never applied, which made every multi-GPU video a
        ``world_size``-times slow-motion (416 frames at 4 fps instead of 30).
        """
        return int(source_fps)

    def _write_video(self, key, frames) -> None:
        path = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(key)),
            f"validation_video_{self.val_counter[key]}.mp4",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tvio.write_video(path, frames, fps=self._video_fps(), video_codec="h264")

    def on_validation_start(self):
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        if not self._should_viz():
            return
        for key, buffer in self.val_image_buffer.items():
            if len(buffer) != 0:
                self._write_video(key, torch.stack(buffer))
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        """``mode`` splits metrics from video so one head can serve two loaders:

        * ``"metrics"`` -- the per-episode subsampled metric loader. Never
          renders or buffers a frame (rendering is ~1 s/frame of CPU, and a
          subsampled video would be an incoherent time-lapse anyway).
        * ``"video"`` -- the contiguous pinned-episode loader. Renders and
          buffers, and logs NOTHING, so no ``Valid/...`` key is averaged over
          it. On a non-viz epoch, or past ``viz_max_batches``, it returns before
          the forward pass: there is nothing such a call could produce.
        * ``"both"`` (default) -- today's single-loader behaviour, which is what
          a data config without ``video_episodes`` still gets.
        """
        if mode not in ("both", "metrics", "video"):
            raise ValueError(f"unknown validation mode {mode!r}")
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
                self.val_image_buffer[key].extend(torch.from_numpy(images))
                if len(self.val_image_buffer[key]) >= 1000:
                    self._write_video(key, torch.stack(self.val_image_buffer[key]))
                    self.val_image_buffer[key].clear()
                    self.val_counter[key] += 1

        if mode == "video":
            # Video-only loader: the forward pass was for the overlay. Logging
            # here would mix pinned-episode numbers into the head's metric and
            # (with add_dataloader_idx=False) collide key-for-key with it.
            return

        # add_dataloader_idx=False: with the train_viz loader Lightning would
        # otherwise suffix every key with "/dataloader_idx_N"; the train-viz
        # wrapper prefixes its keys itself.
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )
