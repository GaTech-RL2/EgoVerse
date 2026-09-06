"""Cartesian bimanual validation: per-embodiment MSE + on-image overlay videos.

Graph-native evaluator for time-indexed (baseline) cotrain runs on 14-D bimanual
cartesian action chunks. Mirrors ``PlanarActionEval`` on the metric side --
normalized and native MSE, aggregated across embodiments -- and adds the overlay
plumbing the arc-branch evaluator used: predicted vs. ground-truth trajectories
projected through per-embodiment revert transforms and logged to WandB as
per-sample images.

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

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.eval import Eval
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.pl_utils.pl_data_utils import DEFAULT_VALID_GROUP
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.viz_utils import save_image


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
        videos_per_epoch: int = 2,
        video_output_dir: str | None = None,
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
        self.videos_per_epoch = int(videos_per_epoch)
        if self.videos_per_epoch < 0:
            raise ValueError("videos_per_epoch must be non-negative")
        self.video_output_dir = (
            None if video_output_dir is None else Path(video_output_dir)
        )
        self.deterministic_seed = int(deterministic_seed)
        # Only present so ``ConfigStore`` overrides (used by fast probes) have a
        # place to land; ``BimanualCartesianEval`` does not override trainer
        # limits by default the way ``PlanarActionEval`` does.
        self.override_dict = {}
        if limit_val_batches is not None:
            self.override_dict["limit_val_batches"] = int(limit_val_batches)
        self._videos_saved_this_epoch = 0

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
        self._videos_saved_this_epoch = 0

    def on_validation_end(self):
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
    # WandB video logging
    # ------------------------------------------------------------------

    def _wandb_logger(self):
        """Return the WandbLogger.experiment handle if wandb is configured."""
        if self.trainer is None:
            return None
        logger = getattr(self.trainer, "logger", None)
        # PL exposes a list at trainer.loggers; the first with wandb wins.
        loggers = getattr(self.trainer, "loggers", None) or ([logger] if logger else [])
        for entry in loggers:
            experiment = getattr(entry, "experiment", None)
            if experiment is None:
                continue
            # WandB run objects expose ``.log`` and ``.id``; duck-check both.
            if callable(getattr(experiment, "log", None)) and hasattr(experiment, "id"):
                return experiment
        return None

    def _log_video_frames(
        self,
        *,
        embodiment_name: str,
        frames: np.ndarray,
        batch_idx: int,
    ) -> None:
        """Write overlays to disk (if configured) and log per-sample WandB images.

        ``frames`` is (N, H, W, 3) uint8; each row is one sample's overlay.
        """
        rank = int(self.trainer.global_rank) if self.trainer is not None else 0
        if self.video_output_dir is not None:
            epoch = int(getattr(self.trainer, "current_epoch", 0))
            step = int(getattr(self.trainer, "global_step", 0))
            destination = (
                self.video_output_dir
                / f"epoch-{epoch}-step-{step}"
                / (self._validation_group or DEFAULT_VALID_GROUP)
                / embodiment_name
            )
            destination.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                save_image(
                    frame,
                    str(destination / f"rank-{rank}-batch-{batch_idx}-sample-{i}.png"),
                )

        experiment = self._wandb_logger()
        if experiment is None:
            return
        # Lazy import so the module works without wandb installed at parse time.
        import wandb  # type: ignore

        group = self._validation_group or DEFAULT_VALID_GROUP
        key_prefix = (
            "Val_video" if group == DEFAULT_VALID_GROUP else f"Val_video_{group}"
        )
        images = [
            wandb.Image(frame, caption=f"batch{batch_idx}-sample{i}")
            for i, frame in enumerate(frames)
        ]
        experiment.log(
            {f"{key_prefix}/{embodiment_name}": images},
            step=int(getattr(self.trainer, "global_step", 0)),
        )

    def _maybe_log_overlay(
        self,
        *,
        embodiment_name: str,
        source_batch: Mapping,
        predictions: Mapping,
        embodiment_id: int,
        batch_idx: int,
    ) -> None:
        """Render prediction vs GT overlay on the first ``budget`` samples."""
        if self.videos_per_epoch <= 0:
            return
        remaining = self.videos_per_epoch - self._videos_saved_this_epoch
        if remaining <= 0:
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
        take = int(min(remaining, pred_normalized.shape[0]))
        if take <= 0:
            return

        pred_native = self._native(pred_normalized[:take], embodiment_id).detach().cpu()
        gt_native = self._native(gt_normalized[:take], embodiment_id).detach().cpu()

        # Unnormalize the obs pose and drop an optional T_obs=1 axis. The revert
        # transforms expect per-sample obs of shape (D,), not (T_obs, D).
        obs_pose_normalized = source_batch[self.obs_pose_key][:take].detach()
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
        images = image_slab[:take]
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
            "embodiment": source_batch["embodiment"][:take].detach().cpu(),
        }
        if "intrinsics" in source_batch:
            flat_batch["intrinsics"] = source_batch["intrinsics"][:take].detach().cpu()

        try:
            frames = viz_partial(
                predictions=flat_predictions,
                batch=flat_batch,
            )
        except Exception as exc:  # noqa: BLE001 -- overlay is best-effort, don't die val
            if self.trainer is not None and getattr(
                self.trainer, "is_global_zero", True
            ):
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

        self._log_video_frames(
            embodiment_name=embodiment_name,
            frames=frames,
            batch_idx=batch_idx,
        )
        self._videos_saved_this_epoch += int(frames.shape[0])

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
                batch_idx=batch_idx,
            )

        metrics["Valid/MSE"] = torch.stack(normalized_values).mean()
        metrics["Valid/Native_MSE"] = torch.stack(native_values).mean()

        self.trainer.lightning_module.log_dict(
            self._namespaced(metrics), sync_dist=True
        )
