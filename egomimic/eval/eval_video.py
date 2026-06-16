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
        viz_episodes: str | None = None,
    ):
        super().__init__()
        self.trainer = None
        self.model = None
        self.viz_func = viz_func
        self.limit_val_batches = limit_val_batches
        self.viz_every_n_epochs = viz_every_n_epochs
        # Path to a JSON list of episode hashes to visualize. When set, viz is
        # produced exclusively from these curated episodes (the inline
        # per-val-batch viz is disabled); each listed episode is rendered for up
        # to ``limit_val_batches`` batches. The per-episode dataloaders are built
        # in trainHydra and assigned to ``viz_dataloaders``
        # (embodiment_name -> {episode_hash: loader}).
        self.viz_episodes = viz_episodes
        self.viz_dataloaders = {}
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
        return (self.trainer.current_epoch % self.viz_every_n_epochs) == 0

    def _use_viz_episodes(self) -> bool:
        return bool(self.viz_episodes) and bool(self.viz_dataloaders)

    @staticmethod
    def _to_device(batch, device):
        return {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    def _write_episode_video(self, embodiment_id, episode_hash, frames):
        out_dir = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(embodiment_id)),
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{episode_hash}.mp4")
        tvio.write_video(path, torch.stack(frames), fps=30, video_codec="h264")

    def _run_viz_episode_pass(self):
        """Render one video per curated episode (rank 0 only).

        Each episode's dataloader is iterated for up to ``limit_val_batches``
        batches, reusing ``compute_metrics_and_viz`` for the model forward + viz
        drawing (metrics are discarded here; they are logged on the normal
        validation loader). ``self.model`` is the raw algo, so the forward does
        not trigger DDP collectives — safe to run on rank 0 alone.
        """
        if not self.trainer.is_global_zero:
            return
        algo = self.model
        device = self.trainer.lightning_module.device
        max_batches = self.limit_val_batches
        with torch.no_grad():
            for embodiment_name, loaders in self.viz_dataloaders.items():
                for episode_hash, loader in loaders.items():
                    frames_by_emb = {}
                    for i, raw in enumerate(loader):
                        if max_batches and i >= max_batches:
                            break
                        batch = {embodiment_name: self._to_device(raw, device)}
                        batch = algo.process_batch_for_training(batch)
                        _, images_dict = self.compute_metrics_and_viz(
                            batch, do_viz=True
                        )
                        for emb_id, images in images_dict.items():
                            frames_by_emb.setdefault(emb_id, []).extend(
                                torch.from_numpy(images)
                            )
                    for emb_id, frames in frames_by_emb.items():
                        if frames:
                            self._write_episode_video(emb_id, episode_hash, frames)

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

    def on_validation_start(self):
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        if not self._should_viz():
            return
        if self._use_viz_episodes():
            self._run_viz_episode_pass()
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
                tvio.write_video(path, frames, fps=30, video_codec="h264")

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # When curated viz episodes are configured, viz is produced by a
        # dedicated pass in on_validation_end; the inline per-val-batch viz is
        # disabled here while metrics still log every validation.
        do_viz = self._should_viz() and not self._use_viz_episodes()
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
                    tvio.write_video(path, frames, fps=30, video_codec="h264")
                    self.val_image_buffer[key].clear()
                    self.val_counter[key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
