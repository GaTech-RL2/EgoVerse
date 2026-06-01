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
        max_frames_per_task: int | None = 1000,
        tasks: list[str] | None = None,
        videos_per_task: int = 1,
    ):
        super().__init__()
        if videos_per_task < 1:
            raise ValueError("videos_per_task must be >= 1")
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        # Per-embodiment list[Transform] applied once during eval to project
        # the model's wrist-frame actions back into cam (head) frame. Reused for
        # both cam-frame MSE and the viz video so we don't transform twice.
        self.transform_lists = transform_lists or {}
        task_list = [str(task) for task in (tasks or [])]
        self.tasks = task_list
        self._task_filter = set(task_list) if task_list else None
        self._write_task_videos = self._task_filter is not None
        # Cap each task video at this many frames. With multiple videos per
        # task, the buffered task limit is this value times videos_per_task.
        self.max_frames_per_task = max_frames_per_task
        self.videos_per_task = int(videos_per_task)
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

    @abstractmethod
    def compute_metrics_and_viz(self, batch):
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

    def on_validation_start(self):
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    @staticmethod
    def _sanitize_task(task: str) -> str:
        # Filesystem-safe: collapse whitespace and replace path separators.
        return re.sub(r"[^\w.-]+", "_", str(task)).strip("_") or "unknown"

    def _task_frame_limit(self):
        if self.max_frames_per_task is None:
            return None
        return self.max_frames_per_task * self.videos_per_task

    def _task_video_chunks(self, buffer):
        if self.videos_per_task == 1:
            return [buffer]
        num_chunks = min(self.videos_per_task, len(buffer))
        chunk_size = (len(buffer) + num_chunks - 1) // num_chunks
        return [
            buffer[start : start + chunk_size]
            for start in range(0, len(buffer), chunk_size)
        ][:num_chunks]

    def on_validation_end(self):
        for key, buffer in self.val_image_buffer.items():
            if self._write_task_videos:
                embodiment_id, task = key
                emb_dir = str(get_embodiment(embodiment_id))
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
                if self._write_task_videos:
                    task_name = self._sanitize_task(task)
                    for video_idx, chunk in enumerate(self._task_video_chunks(buffer)):
                        filename = (
                            f"{task_name}.mp4"
                            if self.videos_per_task == 1
                            else f"{task_name}_{video_idx}.mp4"
                        )
                        frames = torch.stack(chunk)
                        path = os.path.join(
                            self.video_dir(),
                            f"epoch_{self.trainer.current_epoch}",
                            emb_dir,
                            filename,
                        )
                        tvio.write_video(path, frames, fps=30, video_codec="h264")
                else:
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
        metrics, images_dict = self.compute_metrics_and_viz(batch)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        ## images is now a dict
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

            if self._write_task_videos:
                tasks = batch[embodiment_id].get("task")
                if tasks is None:
                    raise KeyError(
                        "Per-task video output requires 'task' in each batch sample. "
                        "Confirm ZarrDataset.__getitem__ attaches it."
                    )
                # Group sample indices by task so each bucket only takes one
                # extend call even when a batch straddles two tasks.
                per_task = defaultdict(list)
                for i, t in enumerate(tasks):
                    task = str(t)
                    if self._task_filter is not None and task not in self._task_filter:
                        continue
                    per_task[task].append(i)
                for task, idxs in per_task.items():
                    key = (embodiment_id, task)
                    if key not in self.val_image_buffer:
                        self.val_image_buffer[key] = []
                        self.val_counter[key] = 0
                    frame_limit = self._task_frame_limit()
                    if frame_limit is not None:
                        remaining = frame_limit - len(self.val_image_buffer[key])
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
