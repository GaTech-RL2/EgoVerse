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
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        self.viz_every_n_epochs = viz_every_n_epochs
        # Per-embodiment list[Transform] applied once during eval to project
        # the model's wrist-frame actions back into cam (head) frame. Reused for
        # both cam-frame MSE and the viz video so we don't transform twice.
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
        # convention — a plain `current_epoch % n` never aligns with a
        # validation epoch unless check_val_every_n_epoch == 1.
        return (self.trainer.current_epoch + 1) % self.viz_every_n_epochs == 0

    @abstractmethod
    def compute_metrics_and_viz(self, batch, do_viz=True):
        """
        Run the model's eval forward and compute metrics and visualization frames.

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    def _video_fps(self, source_fps: int = 30) -> int:
        """Playback fps compensating for the DistributedSampler stride.

        Distributed validation deals the (unshuffled) val indices round-robin,
        so rank 0's consecutive rendered frames are ``world_size`` source
        frames apart. Writing them at the raw source fps produces a
        world_size-times timelapse; scale playback down to restore wall-clock
        pacing."""
        world = max(1, int(getattr(self.trainer, "world_size", 1) or 1))
        return max(1, round(source_fps / world))

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
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(
                    path, frames, fps=self._video_fps(), video_codec="h264"
                )

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        do_viz = self._should_viz()
        metrics, images_dict = self.compute_metrics_and_viz(batch, do_viz=do_viz)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        ## images is now a dict
        if do_viz:
            for key, images in images_dict.items():
                os.makedirs(
                    os.path.join(
                        self.video_dir(),
                        f"epoch_{self.trainer.current_epoch}",
                        str(get_embodiment(key)),
                    ),
                    exist_ok=True,
                )
                if (
                    key not in self.val_image_buffer
                    or self.val_image_buffer[key] is None
                ):
                    self.val_image_buffer[key] = []
                    self.val_counter[key] = 0
                self.val_image_buffer[key].extend(torch.from_numpy(images))
                if len(self.val_image_buffer[key]) >= 1000:
                    frames = torch.stack(self.val_image_buffer[key])
                    path = os.path.join(
                        self.video_dir(),
                        f"epoch_{self.trainer.current_epoch}",
                        str(get_embodiment(key)),
                        f"validation_video_{self.val_counter[key]}.mp4",
                    )
                    tvio.write_video(
                        path, frames, fps=self._video_fps(), video_codec="h264"
                    )
                    self.val_image_buffer[key].clear()
                    self.val_counter[key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
