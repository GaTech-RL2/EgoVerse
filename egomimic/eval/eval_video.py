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
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
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

    @staticmethod
    def _select_rows(d: dict, keep: torch.Tensor, batch_size: int) -> dict:
        """Return a copy of ``d`` with every per-sample entry (tensor with
        ``shape[0] == batch_size`` or list of that length) indexed by the bool
        mask ``keep``. Entries that are not per-sample (scalars, the ``(1,)``
        ``embodiment`` tensor, strings) are passed through unchanged.
        """
        out = {}
        keep_cpu = keep.detach().cpu()
        keep_idx = keep_cpu.nonzero(as_tuple=False).flatten().tolist()
        for k, v in d.items():
            if k == "embodiment":
                out[k] = v
            elif torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == batch_size:
                out[k] = v[keep.to(v.device)]
            elif isinstance(v, list) and len(v) == batch_size:
                out[k] = [v[i] for i in keep_idx]
            else:
                out[k] = v
        return out

    @classmethod
    def mask_substituted(cls, batch: dict, preds: dict, embodiment_name: str):
        """Drop samples the dataset served from a different index than requested
        (see ``MultiDataset._mark_substituted``) so they contribute to neither
        metrics nor the validation video.

        ``batch`` is one embodiment's batch dict; ``preds`` is the algo's full
        prediction dict, of which only the ``{embodiment_name}_*`` per-sample
        tensors are filtered. Returns ``(batch, preds, n_kept)``. When the batch
        has no ``substituted`` key it is returned unchanged.
        """
        sub = batch.get("substituted")
        if not torch.is_tensor(sub) or sub.ndim != 1:
            return batch, preds, None
        keep = ~sub.bool()
        n_kept = int(keep.sum().item())
        if n_kept == sub.shape[0]:
            return batch, preds, n_kept
        B = sub.shape[0]
        batch = cls._select_rows(batch, keep, B)
        prefix = f"{embodiment_name}_"
        preds = dict(preds)
        for k, v in list(preds.items()):
            if (
                k.startswith(prefix)
                and torch.is_tensor(v)
                and v.ndim >= 1
                and v.shape[0] == B
            ):
                preds[k] = v[keep.to(v.device)]
        return batch, preds, n_kept

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

    def on_validation_end(self):
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
        for key, images in images_dict.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    f"epoch_{self.trainer.current_epoch}",
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if key not in self.val_image_buffer or self.val_image_buffer[key] is None:
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
                tvio.write_video(path, frames, fps=30, video_codec="h264")
                self.val_image_buffer[key].clear()
                self.val_counter[key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
