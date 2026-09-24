import os
from abc import abstractmethod

import torch
import torchvision.io as tvio
from lightning.pytorch.loggers import WandbLogger

from egomimic.eval.eval import Eval
from egomimic.eval.replay_viz import ReplayClip, infer_stride, render_replay
from egomimic.rldb.embodiment.embodiment import get_embodiment

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
        viz_mode: str = "replay",
        replay_chunks: int = 5,
        replay_trail: int = 5,
        action_stride: float | None = None,
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
        # "replay" (egomimic/eval/replay_viz.py) plays each chunk as GT then
        # prediction; it needs consecutive frames, so it runs only on a pinned
        # video loader and caps it at ``replay_chunks`` chunks instead of
        # ``viz_max_batches``. "overlay" draws each frame's whole GT and
        # predicted chunk, and is what a head without a video loader gets.
        # ``action_stride`` is frames per chunk step, fractional when the chunk
        # is resampled (None: inferred from the GT keypoints, see
        # replay_viz.infer_stride).
        if viz_mode not in ("overlay", "replay"):
            raise ValueError(f"unknown viz_mode {viz_mode!r}")
        self.viz_mode = viz_mode
        self.replay_chunks = replay_chunks
        self._replay_now = False
        self._replay_seen = {}
        self._replay_stride = {}
        self._replay_horizon = {}
        self.replay_trail = replay_trail
        self.action_stride = action_stride
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

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_name = get_embodiment(batch["embodiment"][0].item()).lower()
        viz_fn = self.viz_func[embodiment_name]
        if self._replay_now:
            return ReplayClip.from_batch(viz_fn, predictions, batch)
        return viz_fn(predictions, batch)

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
            stride=self._replay_stride.get(key, self.action_stride),
            trail=self.replay_trail,
            viz_kwargs={
                k: v for k, v in viz_fn.keywords.items() if k not in _REPLAY_OWN_KWARGS
            },
        )
        self.val_image_buffer[key] = [] if final else [clip.tail(used)]
        return torch.from_numpy(frames) if len(frames) else None

    def _set_replay_now(self, value: bool) -> None:
        self._replay_now = value

    def _is_replay_buffer(self, key) -> bool:
        buffer = self.val_image_buffer[key]
        return bool(buffer) and isinstance(buffer[0], ReplayClip)

    def _replay_done(self) -> bool:
        return bool(self._replay_seen) and all(
            k in self._replay_stride
            and seen
            >= self.replay_chunks * self._replay_horizon[k] * self._replay_stride[k]
            for k, seen in self._replay_seen.items()
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
        # Every rank that gets here renders the same frames to the same path
        # (see _video_fps on why no DistributedSampler splits them), so without
        # this the file is whatever the last rank to finish wrote.
        if not self.trainer.is_global_zero:
            return
        path = os.path.join(
            self.video_dir(),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(key)),
            f"validation_video_{self.val_counter[key]}.mp4",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tvio.write_video(path, frames, fps=self._video_fps(), video_codec="h264")
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
        self._replay_seen, self._replay_stride = {}, {}

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0, mode="both"):
        """``mode`` splits metrics from video so one head can serve two loaders:

        * ``"metrics"`` -- the per-episode subsampled metric loader. Never
          renders or buffers a frame (rendering is ~1 s/frame of CPU, and a
          subsampled video would be an incoherent time-lapse anyway).
        * ``"video"`` -- the contiguous pinned-episode loader. Renders and
          buffers, and logs NOTHING, so no ``Valid/...`` key is averaged over
          it. On a non-viz epoch, or past its cap (``viz_max_batches``, or
          ``replay_chunks`` chunks in replay mode), it returns before the
          forward pass: there is nothing such a call could produce.
        * ``"both"`` (default) -- today's single-loader behaviour, which is what
          a data config without ``video_episodes`` still gets.
        """
        if mode not in ("both", "metrics", "video"):
            raise ValueError(f"unknown validation mode {mode!r}")
        replay = mode == "video" and self.viz_mode == "replay"
        self._set_replay_now(replay)
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
                    seen = self._replay_seen.get(key, 0) + len(images)
                    self._replay_seen[key] = seen
                    self._replay_horizon[key] = images.gt.shape[1]
                    # infer_stride compares frames up to 32 on, from up to 5 anchors.
                    if key not in self._replay_stride and seen >= 40:
                        buffered = ReplayClip.concat(self.val_image_buffer[key])
                        self._replay_stride[key] = (
                            self.action_stride or infer_stride(buffered) or 1
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
        # otherwise suffix every key with "/dataloader_idx_N"; the train-viz
        # wrapper prefixes its keys itself.
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )
