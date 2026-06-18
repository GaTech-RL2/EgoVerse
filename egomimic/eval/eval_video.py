import os
import re
from abc import abstractmethod
from collections import defaultdict

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
        one_video_per_task: bool = False,
        max_frames_per_task: int | None = 1000,
        viz_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        # Validation metrics log every validation; viz video rendering only
        # happens on epochs divisible by this (viz is far more expensive than
        # the metrics). <=0 disables viz entirely.
        self.viz_every_n_epochs = viz_every_n_epochs
        # Per-embodiment list[Transform] applied once during eval to project
        # the model's wrist-frame actions back into cam (head) frame. Reused for
        # both cam-frame MSE and the viz video so we don't transform twice.
        self.transform_lists = transform_lists or {}
        # When True, key the buffer by (embodiment_id, task) and emit exactly
        # one mp4 per (embodiment, task) at the end of validation. Used in
        # eval-only mode when the valid filter spans multiple tasks.
        self.one_video_per_task = one_video_per_task
        # Cap each (embodiment, task) bucket at this many frames. Only takes
        # effect when one_video_per_task=True. Set None to disable.
        self.max_frames_per_task = max_frames_per_task
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
        # check_val_every_n_epoch, so validation epochs are 49, 99, 149, ...
        # for check_val_every_n_epoch=50. Gate viz on the completed-epoch count
        # (current_epoch + 1) so viz_every_n_epochs aligns with those validation
        # runs instead of raw epoch index (which would never match). Always viz
        # on the very first validation (epoch 0 / val_at_start).
        epoch = self.trainer.current_epoch
        return epoch == 0 or ((epoch + 1) % self.viz_every_n_epochs) == 0

    @abstractmethod
    def compute_metrics_and_viz(self, batch, do_viz=True):
        """
        Run the model's eval forward and compute metrics and visualization frames.

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
            do_viz (bool): when False, skip producing viz frames (the expensive
                part); metrics are still computed and returned.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    def on_validation_start(self):
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    @staticmethod
    def _sanitize_task(task: str) -> str:
        # Filesystem-safe: collapse whitespace and replace path separators.
        return re.sub(r"[^\w.-]+", "_", str(task)).strip("_") or "unknown"

    def on_validation_end(self):
        if not self._should_viz():
            return
        for key, buffer in self.val_image_buffer.items():
            if self.one_video_per_task:
                embodiment_id, task = key
                emb_dir = str(get_embodiment(embodiment_id))
                filename = f"{self._sanitize_task(task)}.mp4"
            else:
                emb_dir = str(get_embodiment(key))
                filename = f"validation_video_{self.val_counter[key]}.mp4"

            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    emb_dir,
                ),
                exist_ok=True,
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    emb_dir,
                    filename,
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")

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

        ## images is now a dict (empty when do_viz is False)
        for embodiment_id, images in images_dict.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(embodiment_id)),
                ),
                exist_ok=True,
            )
            frames_tensor = torch.from_numpy(images)

            if self.one_video_per_task:
                tasks = batch[embodiment_id].get("task")
                if tasks is None:
                    raise KeyError(
                        "one_video_per_task=True requires 'task' in each batch "
                        "sample. Confirm ZarrDataset.__getitem__ attaches it."
                    )
                # Group sample indices by task so each bucket only takes one
                # extend call even when a batch straddles two tasks.
                per_task = defaultdict(list)
                for i, t in enumerate(tasks):
                    per_task[str(t)].append(i)
                for task, idxs in per_task.items():
                    key = (embodiment_id, task)
                    if key not in self.val_image_buffer:
                        self.val_image_buffer[key] = []
                        self.val_counter[key] = 0
                    if self.max_frames_per_task is not None:
                        remaining = self.max_frames_per_task - len(
                            self.val_image_buffer[key]
                        )
                        if remaining <= 0:
                            continue
                        idxs = idxs[:remaining]
                    self.val_image_buffer[key].extend(frames_tensor[idxs])
                # No mid-flush in per-task mode: clip length is bounded by
                # limit_val_batches; final write happens in on_validation_end.
            else:
                key = embodiment_id
                if key not in self.val_image_buffer:
                    self.val_image_buffer[key] = []
                    self.val_counter[key] = 0
                self.val_image_buffer[key].extend(frames_tensor)
                if len(self.val_image_buffer[key]) >= 1000:
                    frames = torch.stack(self.val_image_buffer[key])
                    path = os.path.join(
                        self.video_dir(),
                        f"epoch_{self.trainer.current_epoch}",
                        str(get_embodiment(embodiment_id)),
                        f"validation_video_{self.val_counter[key]}.mp4",
                    )
                    tvio.write_video(path, frames, fps=30, video_codec="h264")
                    self.val_image_buffer[key].clear()
                    self.val_counter[key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
