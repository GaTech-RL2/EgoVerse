"""Cartesian bimanual validation: per-embodiment MSE + on-image overlay videos.

Graph-native evaluator for time-indexed (baseline) cotrain runs on 14-D bimanual
cartesian action chunks. Mirrors ``PlanarActionEval`` on the metric side --
normalized and native MSE, aggregated across embodiments -- and adds the overlay
plumbing the arc-branch evaluator used: predicted vs. ground-truth trajectories
projected through per-embodiment revert transforms.

Video output matches the main EgoVerse ``EvalVideo`` format: one h264 mp4 per
episode under ``val_videos/epoch_{N}/[group/]{embodiment}/{episode_hash}.mp4``
at 30 fps, buffered across the whole val epoch and cut on ``episode_hash``
boundaries. Batches without ``episode_hash`` fall back to fixed-size
``validation_video_{i}.mp4`` chunks. Every finished mp4 is also logged to
WandB via ``wandb.Video`` at ``Val_video/{embodiment}`` (or
``Val_video_{group}/{embodiment}`` off the default group). Only rank 0
buffers/writes.

The action space here is 14-D eef-frame [L pose(6), L grip(1), R pose(6), R
grip(1)]; ``viz_func`` per-embodiment partials receive it in native units and
apply the embodiment's ``get_revert_transform_list`` to reproject onto the front
camera before drawing. Both prediction and ground truth are unnormalized before
overlay so the color-graded polylines are directly comparable in metres.

The metric families are:

  * ``Valid/MSE``                 -- normalized action-chunk MSE, macro mean.
  * ``Valid/MSE/{embodiment}``    -- per-embodiment slices of the above.
  * ``Valid/Native_MSE``          -- unnormalized (metres/radians) MSE.
  * ``Valid/Native_MSE/{embod}``  -- per-embodiment slices.

Val group namespacing (``Valid_newtask/...``) follows the same rule the planar
evaluator uses: the default group keeps the bare ``Valid/...`` prefix so runs
overlay on existing charts, and every other group gets a ``Valid_{group}/...``
suffix instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


class BimanualCartesianEval(Eval):
    """Time-indexed cartesian bimanual val: MSE + optional overlay videos."""

    _validation_group = None

    def __init__(
        self,
        action_key: str = "actions_cartesian",
        obs_pose_key: str = "observations.state.ee_pose",
        image_key: str = "observations.images.front_img_1",
        viz_func: Mapping | None = None,
        revert_transforms: Mapping | None = None,
        video_output_dir: str | None = None,
        video_chunk_frames: int = 1000,
        max_episode_frames: int = 6000,
        deterministic_seed: int = 420042,
        limit_val_batches: int | None = None,
    ):
        self.trainer = None
        self.model = None
        self.normalizer = None
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.obs_pose_key = str(obs_pose_key)
        self.image_key = str(image_key)
        self.viz_func = dict(viz_func or {})
        # Per-embodiment revert transform lists. Overlays are always drawn in
        # the front-camera frame, so eef-frame actions must be recomposed onto
        # the current EEF pose before projection. Each entry is the ALREADY-
        # instantiated list returned by ``_build_{embodiment}_revert_eef_frame_transform_list``.
        self.revert_transforms = dict(revert_transforms or {})
        self.video_output_dir = (
            None if video_output_dir is None else Path(video_output_dir)
        )
        # Frames per file on the LEGACY chunked path (batches with no
        # ``episode_hash``); episode-aware batches cut on episode boundaries.
        self.video_chunk_frames = int(video_chunk_frames)
        # Safety cap on a single episode's video and the memory bound on the
        # episode-aware path (an open episode is held in RAM until its boundary).
        self.max_episode_frames = int(max_episode_frames)
        self.deterministic_seed = int(deterministic_seed)
        # Only present so ``ConfigStore`` overrides (used by fast probes) have a
        # place to land; ``BimanualCartesianEval`` does not override trainer
        # limits by default the way ``PlanarActionEval`` does.
        self.override_dict = {}
        if limit_val_batches is not None:
            self.override_dict["limit_val_batches"] = int(limit_val_batches)
        # All keyed by (group, embodiment_name) so per-group runs don't spill
        # frames into each other.
        self.val_image_buffer: dict = {}
        self.val_counter: dict = {}
        self.val_open_episode: dict = {}
        self.val_written: dict = {}
        # (group, embodiment_name, path) accumulated across the epoch and
        # flushed to WandB in on_validation_end.
        self._written_paths: list = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def bind_data_context(self, *, normalizer):
        self.normalizer = normalizer

    def set_validation_group(self, group_name) -> None:
        """Namespace metrics by the val group Lightning is running."""
        self._validation_group = (
            None if group_name in (None, DEFAULT_VALID_GROUP) else str(group_name)
        )

    def on_validation_start(self):
        # Per-epoch reset. ``val_written`` in particular MUST clear here: it is
        # the "already have a video for this episode" guard, and carrying it
        # across epochs would silently skip every episode after epoch 0.
        self.val_image_buffer = {}
        self.val_counter = {}
        self.val_open_episode = {}
        self.val_written = {}
        self._written_paths = []
        if self.trainer is not None and getattr(self.trainer, "is_global_zero", True):
            os.makedirs(
                os.path.join(
                    self._video_dir_root(),
                    f"epoch_{int(getattr(self.trainer, 'current_epoch', 0))}",
                ),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Only rank 0 buffered / wrote frames, so only rank 0 has tails to
        # flush and paths to upload.
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return None
        for buf_key in list(self.val_image_buffer):
            group, embodiment_name = buf_key
            out_dir = self._group_video_dir(group, embodiment_name)
            if self.val_open_episode.get(buf_key) is not None:
                self._flush_episode(buf_key, out_dir)
            elif len(self.val_image_buffer[buf_key]) != 0:
                # chunked fallback tail
                self._write_video(
                    out_dir,
                    f"validation_video_{self.val_counter.get(buf_key, 0)}",
                    self.val_image_buffer[buf_key],
                    group=group,
                    embodiment_name=embodiment_name,
                )
            self.val_counter[buf_key] = 0
            self.val_image_buffer[buf_key] = []
            self.val_open_episode[buf_key] = None

        self._log_wandb_videos()
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _namespaced(self, metrics: dict) -> dict:
        """Prefix ``Valid/...`` keys with the current non-default val group."""
        if self._validation_group is None:
            return metrics
        suffix = self._validation_group
        return {
            (
                f"Valid_{suffix}/{key[len('Valid/'):]}"
                if key.startswith("Valid/")
                else key
            ): value
            for key, value in metrics.items()
        }

    @staticmethod
    def _embodiment(source_batch):
        if "embodiment" not in source_batch:
            raise KeyError("bimanual eval batch is missing embodiment metadata")
        embodiment_id = int(
            resolve_homogeneous_scalar(source_batch["embodiment"], label="embodiment")
        )
        name = get_embodiment(embodiment_id)
        if name is None:
            raise KeyError(f"Unknown bimanual embodiment id {embodiment_id}")
        return embodiment_id, name.lower()

    @staticmethod
    def _cuda_devices(batch):
        return sorted(
            {
                int(value.device.index)
                for source_batch in batch.values()
                for value in source_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )

    def _forward_deterministic(self, batch):
        """Run one prediction with a pinned seed, without leaking RNG state."""
        with torch.random.fork_rng(devices=self._cuda_devices(batch)):
            torch.manual_seed(self.deterministic_seed)
            return self.model.forward_eval(batch)

    def _native(self, normalized, embodiment_id):
        if self.normalizer is None:
            raise RuntimeError("bimanual evaluator data context was not bound")
        return self.normalizer.unnormalize(
            {self.action_key: normalized}, embodiment_id
        )[self.action_key]

    def _native_pose(self, normalized, embodiment_id):
        return self.normalizer.unnormalize(
            {self.obs_pose_key: normalized}, embodiment_id
        )[self.obs_pose_key]

    def _revert_to_camframe(
        self, *, actions: torch.Tensor, obs_pose: torch.Tensor, embodiment_name: str
    ) -> torch.Tensor | None:
        """Recompose eef-frame actions onto the obs pose and land in cam frame.

        Returns ``None`` when no revert transform is configured for this
        embodiment, which disables the overlay for that source without failing.
        """
        transform_list = self.revert_transforms.get(embodiment_name)
        if not transform_list:
            return None
        batch = {
            self.action_key: actions.detach().cpu().numpy().astype(np.float32),
            self.obs_pose_key: obs_pose.detach().cpu().numpy().astype(np.float32),
        }
        reverted = Embodiment.apply_transform(batch, list(transform_list))
        return reverted[self.action_key]

    # ------------------------------------------------------------------
    # video buffering / writing
    # ------------------------------------------------------------------

    def _video_dir_root(self) -> str:
        """Base ``videos/`` directory. Prefer ``video_output_dir`` when set,
        else fall back to ``{trainer.default_root_dir}/videos``."""
        if self.video_output_dir is not None:
            return str(self.video_output_dir)
        if self.trainer is None:
            return os.path.join(os.getcwd(), "videos")
        return os.path.join(self.trainer.default_root_dir, "videos")

    def _group_video_dir(self, group: str, embodiment_name: str) -> str:
        """``videos/epoch_N/[group/]{embodiment}`` -- the group segment is
        omitted for the default group so existing runs keep their layout."""
        parts = [
            self._video_dir_root(),
            f"epoch_{int(getattr(self.trainer, 'current_epoch', 0))}",
        ]
        if group != DEFAULT_VALID_GROUP:
            parts.append(str(group))
        parts.append(str(embodiment_name))
        return os.path.join(*parts)

    def _write_video(self, out_dir, name, frames, *, group, embodiment_name):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.mp4")
        tvio.write_video(
            path,
            torch.stack(list(frames)),
            fps=30,
            video_codec="h264",
        )
        self._written_paths.append((group, embodiment_name, path))

    def _flush_episode(self, buf_key, out_dir):
        """Write the open episode's buffer as ``{episode_hash}.mp4``."""
        episode = self.val_open_episode.get(buf_key)
        buffer = self.val_image_buffer.get(buf_key) or []
        if episode is not None and len(buffer) != 0:
            group, embodiment_name = buf_key
            self._write_video(
                out_dir,
                episode,
                buffer,
                group=group,
                embodiment_name=embodiment_name,
            )
            self.val_written.setdefault(buf_key, set()).add(episode)
        self.val_image_buffer[buf_key] = []
        self.val_open_episode[buf_key] = None

    def _buffer_per_episode(self, buf_key, out_dir, frames, hashes):
        """One mp4 per episode, cut where ``episode_hash`` changes.

        Validation runs with shuffle=False and MultiDataset lays its index map
        out episode by episode, so samples arrive grouped by episode and in
        frame order -- the boundary is simply where the hash changes.
        """
        written = self.val_written.setdefault(buf_key, set())
        for frame, episode in zip(frames, hashes):
            open_episode = self.val_open_episode.get(buf_key)
            if episode in written:
                # Replay from ``max_size_cycle`` (or frames past the safety
                # cap). Seeing a finished episode also means whatever is still
                # open has ended -- without this the LAST episode of a cycling
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
        """Legacy path for batches with no ``episode_hash``: fixed-size files."""
        if self.val_image_buffer.get(buf_key) is None:
            self.val_image_buffer[buf_key] = []
            self.val_counter[buf_key] = 0
        self.val_image_buffer[buf_key].extend(frames)
        if len(self.val_image_buffer[buf_key]) >= self.video_chunk_frames:
            group, embodiment_name = buf_key
            self._write_video(
                out_dir,
                f"validation_video_{self.val_counter[buf_key]}",
                self.val_image_buffer[buf_key],
                group=group,
                embodiment_name=embodiment_name,
            )
            self.val_image_buffer[buf_key].clear()
            self.val_counter[buf_key] += 1

    @staticmethod
    def _episode_hashes(source_batch: Mapping, n_images: int):
        """Per-sample ``episode_hash`` for this embodiment's images.

        ``episode_hash`` is stamped on every sample by ZarrDataset and falls
        through ``process_batch_for_training`` unchanged. Returns ``None`` when
        it is absent or shorter than the image count so callers fall back to
        fixed-size chunking rather than mislabelling frames.
        """
        hashes = (
            source_batch.get("episode_hash")
            if isinstance(source_batch, Mapping)
            else None
        )
        if not isinstance(hashes, (list, tuple)) or len(hashes) < n_images:
            return None
        return [str(h) for h in hashes[:n_images]]

    def _wandb_logger(self):
        """Return the WandbLogger.experiment handle if wandb is configured."""
        if self.trainer is None:
            return None
        logger = getattr(self.trainer, "logger", None)
        loggers = getattr(self.trainer, "loggers", None) or ([logger] if logger else [])
        for entry in loggers:
            experiment = getattr(entry, "experiment", None)
            if experiment is None:
                continue
            if callable(getattr(experiment, "log", None)) and hasattr(experiment, "id"):
                return experiment
        return None

    def _log_wandb_videos(self) -> None:
        """Upload every mp4 written this epoch as a ``wandb.Video`` panel.

        One log call per (group, embodiment); the panel accepts a list so all
        episode files for that pair land on the same chart.
        """
        if not self._written_paths:
            return
        experiment = self._wandb_logger()
        if experiment is None:
            return
        import wandb  # type: ignore

        grouped: dict[tuple[str, str], list[str]] = {}
        for group, embodiment_name, path in self._written_paths:
            grouped.setdefault((group, embodiment_name), []).append(path)

        payload: dict = {}
        for (group, embodiment_name), paths in grouped.items():
            prefix = (
                "Val_video" if group == DEFAULT_VALID_GROUP else f"Val_video_{group}"
            )
            payload[f"{prefix}/{embodiment_name}"] = [
                wandb.Video(p, fps=30, format="mp4") for p in paths
            ]
        experiment.log(
            payload,
            step=int(getattr(self.trainer, "global_step", 0)),
        )

    def _maybe_log_overlay(
        self,
        *,
        embodiment_name: str,
        source_batch: Mapping,
        predictions: Mapping,
        embodiment_id: int,
    ) -> None:
        """Render prediction vs GT overlays and buffer them into the epoch video.

        Rank 0 only -- other ranks racing on the same directory would clobber
        each other's files.
        """
        if self.trainer is not None and not getattr(
            self.trainer, "is_global_zero", True
        ):
            return
        viz_partial = self.viz_func.get(embodiment_name)
        if viz_partial is None:
            return
        if self.obs_pose_key not in source_batch:
            # Overlay needs the obs pose to recompose eef-frame actions into
            # cam frame; without it we can't draw a meaningful trajectory.
            return

        pred_normalized = predictions["pred_action"].detach()
        gt_normalized = source_batch[self.action_key].detach()
        if int(pred_normalized.shape[0]) <= 0:
            return

        pred_native = self._native(pred_normalized, embodiment_id).detach().cpu()
        gt_native = self._native(gt_normalized, embodiment_id).detach().cpu()

        # Unnormalize the obs pose and drop an optional T_obs=1 axis. The revert
        # transforms expect per-sample obs of shape (D,), not (T_obs, D).
        obs_pose_normalized = source_batch[self.obs_pose_key].detach()
        obs_pose_native = self._native_pose(obs_pose_normalized, embodiment_id)
        if obs_pose_native.ndim == 3 and obs_pose_native.shape[1] == 1:
            obs_pose_native = obs_pose_native.squeeze(1)
        obs_pose_native = obs_pose_native.detach().cpu()

        pred_camframe = self._revert_to_camframe(
            actions=pred_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )
        if pred_camframe is None:
            return
        gt_camframe = self._revert_to_camframe(
            actions=gt_native,
            obs_pose=obs_pose_native,
            embodiment_name=embodiment_name,
        )

        image_slab = source_batch[self.image_key]
        images = image_slab
        if images.ndim == 5:
            # (B, T=1, C, H, W) or (B, C, H, W) — collapse the optional obs-step axis.
            images = images[:, 0]
        images = images.detach().cpu()
        if images.ndim == 4 and images.shape[1] in (1, 3):
            images = images.permute(0, 2, 3, 1)

        # viz_gt_preds looks up predictions[f"{embodiment_name}_{action_key}"]
        # and reads batch[image_key], batch[action_key], batch['embodiment'],
        # batch.get('intrinsics'). Feed the cam-frame actions (numpy) so viz
        # can project them without doing its own revert.
        flat_predictions = {
            f"{embodiment_name}_{self.action_key}": pred_camframe,
        }
        flat_batch = {
            self.image_key: images,
            self.action_key: gt_camframe,
            "embodiment": source_batch["embodiment"].detach().cpu(),
        }
        if "intrinsics" in source_batch:
            flat_batch["intrinsics"] = source_batch["intrinsics"].detach().cpu()

        try:
            frames = viz_partial(
                predictions=flat_predictions,
                batch=flat_batch,
            )
        except Exception as exc:  # noqa: BLE001 -- overlay is best-effort, don't die val
            print(
                f"[BimanualCartesianEval] skipped {embodiment_name} overlay: {exc}",
                flush=True,
            )
            return

        if not isinstance(frames, np.ndarray):
            frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim == 3:
            frames = frames[None]

        # torchvision.io.write_video wants uint8 (T, H, W, 3) tensors; keep the
        # per-frame layout the numpy array already has.
        frame_tensor = torch.from_numpy(frames)
        n_images = int(frame_tensor.shape[0])

        group = self._validation_group or DEFAULT_VALID_GROUP
        buf_key = (group, embodiment_name)
        out_dir = self._group_video_dir(group, embodiment_name)
        os.makedirs(out_dir, exist_ok=True)

        hashes = self._episode_hashes(source_batch, n_images)
        if hashes is None:
            self._buffer_chunked(buf_key, out_dir, list(frame_tensor))
        else:
            self._buffer_per_episode(buf_key, out_dir, list(frame_tensor), hashes)

    # ------------------------------------------------------------------
    # main val entrypoint
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        del dataloader_idx
        result = self._forward_deterministic(batch)

        metrics: dict[str, torch.Tensor] = {}
        normalized_values: list[torch.Tensor] = []
        native_values: list[torch.Tensor] = []
        seen_labels: set[str] = set()

        for source_id, source_batch in batch.items():
            embodiment_id, label = self._embodiment(source_batch)
            if label in seen_labels:
                raise ValueError(f"Duplicate bimanual eval embodiment {label!r}")
            seen_labels.add(label)

            prediction = result[source_id]["pred_action"]
            target = source_batch[self.action_key]
            if prediction.shape != target.shape:
                raise ValueError(
                    "bimanual prediction/target shape mismatch: "
                    f"{tuple(prediction.shape)} != {tuple(target.shape)}"
                )
            normalized_mse = (prediction - target).square().mean()
            native_mse = (
                (
                    self._native(prediction, embodiment_id)
                    - self._native(target, embodiment_id)
                )
                .square()
                .mean()
            )
            metrics[f"Valid/MSE/{label}"] = normalized_mse
            metrics[f"Valid/Native_MSE/{label}"] = native_mse
            normalized_values.append(normalized_mse)
            native_values.append(native_mse)

            self._maybe_log_overlay(
                embodiment_name=label,
                source_batch=source_batch,
                predictions=result[source_id],
                embodiment_id=embodiment_id,
            )

        metrics["Valid/MSE"] = torch.stack(normalized_values).mean()
        metrics["Valid/Native_MSE"] = torch.stack(native_values).mean()

        self.trainer.lightning_module.log_dict(
            self._namespaced(metrics), sync_dist=True
        )
