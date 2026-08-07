import os
from abc import abstractmethod

import torch
import torchvision.io as tvio

from egomimic.eval.core.eval import Eval
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
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
        max_videos: int | None = None,
        video_chunk_frames: int = 1000,
        max_episode_frames: int = 6000,
    ):
        super().__init__()  # initializes self.trainer = self.model = None
        self.viz_func = viz_func
        # Cap on number of episodes rendered per validation pass. None
        # = no cap. Each sub-eval reads this in its per-episode loop to
        # truncate B -> min(B, max_videos) so output panels stay short.
        self.max_videos = int(max_videos) if max_videos is not None else None
        # Frames per file on the LEGACY path only -- batches that arrive
        # without `episode_hash` still flush a fresh file every this many
        # frames. Episode-aware batches ignore it and cut on episode
        # boundaries instead.
        self.video_chunk_frames = int(video_chunk_frames)
        # Safety cap on a single episode's video, and the memory bound on the
        # per-episode path: an open episode is held in RAM until its boundary,
        # so peak use is about
        #     2 (embodiments) * max_episode_frames * H * W * 3 bytes
        # -- at 480x640 that is ~0.9 MB/frame, so 6000 frames is ~5.5 GB per
        # embodiment worst case. Episodes longer than this are truncated
        # rather than split, so the file still maps 1:1 to an episode.
        self.max_episode_frames = int(max_episode_frames)
        # Per-embodiment list[Transform] applied once during eval to project
        # the model's wrist-frame actions back into cam (head) frame. Reused for
        # both cam-frame MSE and the viz video so we don't transform twice.
        self.transform_lists = transform_lists or {}
        # All keyed by (val_group, image key) rather than image key alone:
        # with more than one val group the same embodiment appears in each, and
        # a key-only buffer would let one group's leftover frames spill into
        # the next group's video.
        self.val_image_buffer = {}
        self.val_counter = {}
        # episode_hash currently accumulating per (group, key), and the set of
        # hashes already written this epoch. The latter is what makes the
        # CombinedLoader's `max_size_cycle` harmless: when the shorter
        # embodiment wraps and replays episodes to match the longer one, those
        # frames are recognised as already written and dropped instead of being
        # appended a second time.
        self.val_open_episode = {}
        self.val_written = {}
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

    def val_group(self, dataloader_idx=0):
        """Name of the val group Lightning is currently iterating.

        Resolved from the datamodule's positional `valid_group_names` rather
        than plumbed through the evaluator API, so composite evaluators that
        already forward `dataloader_idx` to their children get it for free.
        Falls back to the default group when the datamodule predates groups
        (or when running under a bare eval harness with no datamodule).
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        names = getattr(datamodule, "valid_group_names", None) or []
        if 0 <= dataloader_idx < len(names):
            return names[dataloader_idx]
        return DEFAULT_VALID_GROUP

    def _group_video_dir(self, group, key):
        """`videos/epoch_N/[group/]{embodiment}` -- the group segment is
        omitted for the default group so existing runs keep their layout."""
        parts = [self.video_dir(), f"epoch_{self.trainer.current_epoch}"]
        if group != DEFAULT_VALID_GROUP:
            parts.append(str(group))
        parts.append(str(get_embodiment(key)))
        return os.path.join(*parts)

    @staticmethod
    def _namespace_metrics(metrics, group):
        """Move `Valid/...` keys into `Valid_{group}/...` for non-default groups.

        The leading path segment is what wandb uses to section a chart list, so
        this is what puts the held-out-task numbers in their own panel instead
        of overwriting the in-domain ones. Keys that are not `Valid/`-prefixed
        are left alone -- they are not per-group val metrics.
        """
        if group == DEFAULT_VALID_GROUP:
            return metrics
        prefix = f"Valid_{group}/"
        return {
            (prefix + k[len("Valid/"):] if k.startswith("Valid/") else k): v
            for k, v in metrics.items()
        }

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

    @staticmethod
    def _episode_hashes(batch, key, n_images):
        """Per-sample episode_hash for the images this embodiment produced.

        `episode_hash` is stamped on every sample by ZarrDataset and survives
        `process_batch_for_training` (it is not a registered zarr key, so it
        falls through under its own name). Returns None when it is absent or
        shorter than the image count -- callers then fall back to fixed-size
        chunking rather than mislabelling frames.
        """
        sub = batch.get(key) if isinstance(batch, dict) else None
        hashes = sub.get("episode_hash") if isinstance(sub, dict) else None
        if not isinstance(hashes, (list, tuple)) or len(hashes) < n_images:
            return None
        return [str(h) for h in hashes[:n_images]]

    def _write_video(self, out_dir, name, frames):
        os.makedirs(out_dir, exist_ok=True)
        tvio.write_video(
            os.path.join(out_dir, f"{name}.mp4"),
            torch.stack(list(frames)),
            fps=30,
            video_codec="h264",
        )

    def _flush_episode(self, buf_key, out_dir):
        """Write the open episode's buffer as `{episode_hash}.mp4`."""
        episode = self.val_open_episode.get(buf_key)
        buffer = self.val_image_buffer.get(buf_key) or []
        if episode is not None and len(buffer) != 0:
            self._write_video(out_dir, episode, buffer)
            self.val_written.setdefault(buf_key, set()).add(episode)
        self.val_image_buffer[buf_key] = []
        self.val_open_episode[buf_key] = None

    def _buffer_per_episode(self, buf_key, out_dir, frames, hashes):
        """One mp4 per episode, cut where episode_hash changes.

        Validation runs with shuffle=False and MultiDataset lays its index map
        out episode by episode, so samples arrive grouped by episode and in
        frame order -- the boundary is simply where the hash changes.
        """
        written = self.val_written.setdefault(buf_key, set())
        for frame, episode in zip(frames, hashes):
            open_episode = self.val_open_episode.get(buf_key)
            if episode in written:
                # Replay from max_size_cycle (or frames past the safety cap).
                # Seeing a finished episode also means whatever is still open
                # has ended -- without this the LAST episode of a cycling
                # embodiment never hits a boundary and keeps accumulating a
                # copy of itself on every wrap.
                if open_episode is not None:
                    self._flush_episode(buf_key, out_dir)
                continue
            if open_episode is None:
                self.val_open_episode[buf_key] = episode
                self.val_image_buffer[buf_key] = []
            elif episode != open_episode:
                self._flush_episode(buf_key, out_dir)
                self.val_open_episode[buf_key] = episode
                self.val_image_buffer[buf_key] = []
            buffer = self.val_image_buffer[buf_key]
            buffer.append(frame)
            if len(buffer) >= self.max_episode_frames:
                # Episode longer than the safety cap: write what we have and
                # mark it done so the remainder is dropped rather than
                # spilling into the next episode's file.
                self._flush_episode(buf_key, out_dir)

    def _buffer_chunked(self, buf_key, out_dir, frames):
        """Legacy path for batches with no episode_hash: fixed-size files."""
        if self.val_image_buffer.get(buf_key) is None:
            self.val_image_buffer[buf_key] = []
            self.val_counter[buf_key] = 0
        self.val_image_buffer[buf_key].extend(frames)
        if len(self.val_image_buffer[buf_key]) >= self.video_chunk_frames:
            self._write_video(
                out_dir,
                f"validation_video_{self.val_counter[buf_key]}",
                self.val_image_buffer[buf_key],
            )
            self.val_image_buffer[buf_key].clear()
            self.val_counter[buf_key] += 1

    def on_validation_start(self):
        # Per-epoch reset. `val_written` in particular MUST be cleared here:
        # it is the "already have a video for this episode" guard, and carrying
        # it across epochs would silently skip every episode after epoch 0.
        self.val_image_buffer = {}
        self.val_counter = {}
        self.val_open_episode = {}
        self.val_written = {}
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Fires once after ALL val groups have run, so the buffer key carries
        # the group each tail of frames belongs to. The last episode of every
        # (group, embodiment) is still open at this point -- nothing followed
        # it to trigger a boundary flush -- so this is what writes it.
        for buf_key in list(self.val_image_buffer):
            group, key = buf_key
            out_dir = self._group_video_dir(group, key)
            if self.val_open_episode.get(buf_key) is not None:
                self._flush_episode(buf_key, out_dir)
            elif len(self.val_image_buffer[buf_key]) != 0:
                # chunked fallback tail
                self._write_video(
                    out_dir,
                    f"validation_video_{self.val_counter.get(buf_key, 0)}",
                    self.val_image_buffer[buf_key],
                )
            self.val_counter[buf_key] = 0
            self.val_image_buffer[buf_key] = []
            self.val_open_episode[buf_key] = None

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        group = self.val_group(dataloader_idx)
        metrics, images_dict = self.compute_metrics_and_viz(batch)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        metrics = self._namespace_metrics(metrics, group)

        ## images is now a dict
        for key, images in images_dict.items():
            buf_key = (group, key)
            out_dir = self._group_video_dir(group, key)
            os.makedirs(out_dir, exist_ok=True)
            frames = torch.from_numpy(images)
            hashes = self._episode_hashes(batch, key, len(frames))
            if hashes is None:
                self._buffer_chunked(buf_key, out_dir, frames)
            else:
                self._buffer_per_episode(buf_key, out_dir, frames, hashes)

        # add_dataloader_idx=False: one dataloader per val group means Lightning
        # would otherwise append `/dataloader_idx_N` to every key, splitting one
        # chart in two and burying the group name. `_namespace_metrics` already
        # disambiguates the groups (`Valid/` vs `Valid_{group}/`), so the suffix
        # is redundant, and without it the same metric from a single-group run
        # and a multi-group run share an axis.
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )
