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
        self.viz_buffers = {}
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

    # ---- viz loader support (see MultiDataModuleWrapper.valid_viz_params) ----
    def _viz_dataloader_idx(self):
        dm = getattr(self.trainer, "datamodule", None)
        return getattr(dm, "viz_dataloader_idx", None)

    def _episode_label(self, embodiment_id, episode_idx: int) -> str:
        """``<idx>_<hash8>_<seen|unseen>`` from the first valid dataset that
        names its episodes; falls back to the bare index."""
        dm = getattr(self.trainer, "datamodule", None)
        for ds in (getattr(dm, "valid_datasets", None) or {}).values():
            names = getattr(ds, "episode_names", None)
            if names and 0 <= episode_idx < len(names):
                name = names[episode_idx]
                tags = getattr(ds, "_operator_seen", None) or {}
                tag = ""
                if name in tags:
                    tag = "_seen" if tags[name] else "_unseen"
                return f"ep{episode_idx:02d}_{str(name)[:8]}{tag}"
        return f"ep{episode_idx:02d}"

    def _buffer_viz_frames(self, key, raw_batch, images):
        """Append viz frames to per-episode buffers. ``raw_batch`` is the
        processed batch for this embodiment (before substituted rows were
        dropped); ``images`` are the frames of the kept rows, in order."""
        ep = raw_batch.get("episode_idx")
        sub = raw_batch.get("substituted")
        if torch.is_tensor(ep):
            ep = ep.reshape(-1)
            if torch.is_tensor(sub) and sub.numel() == ep.numel():
                ep = ep[~sub.reshape(-1).bool()]
            ep = ep.tolist()
        if not isinstance(ep, list) or len(ep) != len(images):
            ep = [-1] * len(images)
        for e, frame in zip(ep, images):
            self.viz_buffers.setdefault((key, int(e)), []).append(
                torch.from_numpy(frame)
            )

    def _write_viz_videos(self):
        if not self.trainer.is_global_zero:
            self.viz_buffers = {}
            return
        for (key, e), frames in self.viz_buffers.items():
            if not frames:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(key)),
            )
            os.makedirs(out_dir, exist_ok=True)
            label = self._episode_label(key, e) if e >= 0 else "viz"
            tvio.write_video(
                os.path.join(out_dir, f"{label}.mp4"),
                torch.stack(frames),
                fps=30,
                video_codec="h264",
            )
        self.viz_buffers = {}

    def on_validation_end(self):
        self._write_viz_videos()
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
        viz_idx = self._viz_dataloader_idx()
        metrics, images_dict = self.compute_metrics_and_viz(batch)

        if viz_idx is not None and dataloader_idx == viz_idx:
            # Viz loader: frames only, one video per episode; no metrics.
            for key, images in images_dict.items():
                self._buffer_viz_frames(key, batch[key], images)
            return

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        if viz_idx is not None:
            # Metrics loader: log without Lightning's "/dataloader_idx_N"
            # suffix and leave the video to the viz loader.
            self.trainer.lightning_module.log_dict(
                metrics, sync_dist=True, add_dataloader_idx=False
            )
            return

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
