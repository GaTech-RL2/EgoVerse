"""Teacher-forced Fold action metrics and GT/prediction overlay videos."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.pipeline import packed
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


class HumanRobotOverlayEval(EvalVideo):
    """Evaluate denoised actions in one shared, unnormalized action space.

    Predictions and targets are both converted from the training normalizer
    before metrics or rendering. This makes the reported MSE directly
    comparable across Pipeline and diffusion-policy baselines.
    """

    def __init__(
        self,
        chunk_len: int | None = None,
        image_key: str = "front_img_1",
        frame_stride: int = 10,
        max_frames: int | None = 120,
        max_frames_by_embodiment: dict[str, int] | None = None,
        log_normalized_mse: bool = False,
        log_native_mse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_len = None if chunk_len is None else int(chunk_len)
        self.image_key = str(image_key)
        self.frame_stride = int(frame_stride)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.max_frames_by_embodiment = {
            str(name).lower(): int(limit)
            for name, limit in (max_frames_by_embodiment or {}).items()
        }
        self.log_normalized_mse = bool(log_normalized_mse)
        self.log_native_mse = bool(log_native_mse)
        self._rendered_frames: dict[int, int] = {}

    def on_validation_start(self):
        self._rendered_frames.clear()
        super().on_validation_start()

    @staticmethod
    def _error_metrics(prefix: str, prediction, target) -> dict:
        squared_error = (prediction - target).square().reshape(-1)
        return {
            f"{prefix}_mse": squared_error.mean().detach(),
            f"{prefix}_squared_error_median": squared_error.median().detach(),
            f"{prefix}_squared_error_p95": torch.quantile(
                squared_error.float(), 0.95
            ).detach(),
            f"{prefix}_squared_error_max": squared_error.max().detach(),
        }

    def _unnormalized_target(self, batch, emb_id, action_key):
        unnormalized = self.model.norm_stats.unnormalize(batch, emb_id)
        target = unnormalized[action_key]
        if "cu_seqlens" not in batch:
            if target.ndim != 3:
                raise ValueError(
                    f"Expected standard actions (batch, horizon, dim), got "
                    f"{tuple(target.shape)}"
                )
            return unnormalized, target
        if self.chunk_len is None:
            raise ValueError("chunk_len is required for a packed validation loader")
        cu_seqlens = batch["cu_seqlens"].to(target.device)
        return unnormalized, packed.chunk_targets(target, cu_seqlens, self.chunk_len)

    def _normalized_target(self, batch, action_key):
        target = batch[action_key]
        if "cu_seqlens" not in batch:
            if target.ndim != 3:
                raise ValueError(
                    f"Expected standard actions (batch, horizon, dim), got "
                    f"{tuple(target.shape)}"
                )
            return target
        if self.chunk_len is None:
            raise ValueError("chunk_len is required for a packed validation loader")
        cu_seqlens = batch["cu_seqlens"].to(target.device)
        return packed.chunk_targets(target, cu_seqlens, self.chunk_len)

    def _unnormalize_prediction(self, prediction, emb_id, action_key):
        if prediction.ndim != 3:
            raise ValueError(
                "Expected prediction shaped (batch, horizon, action_dim), got "
                f"{tuple(prediction.shape)}"
            )
        # Keep the horizon axis intact. MultiDataset statistics may be either
        # per-dimension (D,) or slotwise (H, D); both broadcast correctly into
        # (B, H, D), while flattening B and H breaks slotwise arc-token stats.
        return self.model.norm_stats.unnormalize({action_key: prediction}, emb_id)[
            action_key
        ]

    @staticmethod
    def _finite_mse(prediction, target, label: str):
        if prediction.shape != target.shape:
            raise ValueError(
                f"{label} prediction/target mismatch: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        if prediction.numel() == 0:
            raise ValueError(f"{label} received an empty prediction/target")
        squared_error = (prediction - target).float().square().reshape(-1)
        if not bool(torch.isfinite(squared_error).all()):
            raise ValueError(f"{label} contains non-finite squared error")
        return (
            squared_error.mean().detach(),
            squared_error.sum().detach(),
            squared_error.numel(),
        )

    def compute_metrics_and_viz(
        self, batch: dict[int, dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], dict[int, np.ndarray]]:
        predictions = self.model.forward_eval(batch)
        metrics = {}
        images = {}
        normalized_squared_error_sum = None
        normalized_element_count = 0
        native_squared_error_sum = None
        native_element_count = 0

        for emb_id, embodiment_batch in batch.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            prediction_key = f"emb{emb_id}_{action_key}"
            if prediction_key not in predictions:
                raise KeyError(
                    f"Missing {prediction_key!r}; available={sorted(predictions)}"
                )

            normalized_prediction = predictions[prediction_key]
            normalized_target = self._normalized_target(embodiment_batch, action_key)
            normalized_count = min(
                normalized_target.shape[0], normalized_prediction.shape[0]
            )
            normalized_target = normalized_target[:normalized_count]
            normalized_prediction = normalized_prediction[:normalized_count]
            if self.log_normalized_mse:
                normalized_mse, squared_error_sum, element_count = self._finite_mse(
                    normalized_prediction,
                    normalized_target,
                    f"normalized {embodiment_name}",
                )
                metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse
                normalized_squared_error_sum = (
                    squared_error_sum
                    if normalized_squared_error_sum is None
                    else normalized_squared_error_sum + squared_error_sum
                )
                normalized_element_count += element_count

            unnormalized, target = self._unnormalized_target(
                embodiment_batch, emb_id, action_key
            )
            prediction = self._unnormalize_prediction(
                normalized_prediction, emb_id, action_key
            )
            count = min(target.shape[0], prediction.shape[0])
            target = target[:count]
            prediction = prediction[:count]
            if target.shape != prediction.shape:
                raise ValueError(
                    f"Prediction/target mismatch after unnormalizing: "
                    f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
                )

            metric_prefix = f"Valid/emb{emb_id}_{action_key}_action"
            metrics.update(self._error_metrics(metric_prefix, prediction, target))
            metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] = (
                (target[:, :1] - target).square().mean().detach()
            )

            if self.log_native_mse:
                adapter = self.model.rollout_adapter_for(emb_id)
                if adapter is None:
                    raise ValueError(
                        f"Native MSE requested but {embodiment_name} has no "
                        "rollout adapter"
                    )
                native_prediction = adapter.decode(prediction, unnormalized)
                native_target = adapter.decode(target, unnormalized)
                native_mse, squared_error_sum, element_count = self._finite_mse(
                    native_prediction,
                    native_target,
                    f"native {embodiment_name}",
                )
                metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse
                native_squared_error_sum = (
                    squared_error_sum
                    if native_squared_error_sum is None
                    else native_squared_error_sum + squared_error_sum
                )
                native_element_count += element_count

            raw_images = embodiment_batch.get(self.image_key)
            viz = None if self.viz_func is None else self.viz_func.get(embodiment_name)
            if raw_images is None or viz is None:
                continue
            if raw_images.ndim == 5:
                raw_images = raw_images[:, -1]

            selected = torch.arange(0, count, self.frame_stride)
            limit = self.max_frames_by_embodiment.get(embodiment_name, self.max_frames)
            rendered = self._rendered_frames.get(emb_id, 0)
            if limit is not None and limit > 0:
                selected = selected[: max(0, limit - rendered)]
            if len(selected) == 0:
                continue
            self._rendered_frames[emb_id] = rendered + len(selected)

            def take(value):
                return value.index_select(0, selected.to(value.device))

            viz_batch = {
                "embodiment": torch.full((len(selected),), emb_id, dtype=torch.long),
                self.image_key: take(raw_images),
                action_key: take(target),
            }
            for key in (
                "state_ee_pose",
                "viz_current_wrist_poses",
                "front_intrinsics",
                "left_camera_extrinsics",
                "right_camera_extrinsics",
            ):
                source = unnormalized if key in unnormalized else embodiment_batch
                if key in source and torch.is_tensor(source[key]):
                    viz_batch[key] = take(source[key])
            viz_predictions = {f"{embodiment_name}_{action_key}": take(prediction)}

            transforms = self.transform_lists.get(embodiment_name)
            if transforms:
                gt_transformed = Embodiment.apply_transform(viz_batch, transforms)
                pred_input = dict(viz_batch)
                pred_input[action_key] = viz_predictions[
                    f"{embodiment_name}_{action_key}"
                ]
                pred_transformed = Embodiment.apply_transform(pred_input, transforms)
                viz_batch.update(gt_transformed)
                viz_predictions[f"{embodiment_name}_{action_key}"] = pred_transformed[
                    action_key
                ]
                metrics.update(
                    self._error_metrics(
                        f"Valid/{embodiment_name}_{action_key}_camera_action",
                        pred_transformed[action_key],
                        gt_transformed[action_key],
                    )
                )

            try:
                images[emb_id] = np.asarray(
                    viz(predictions=viz_predictions, batch=viz_batch)
                )
            except Exception as error:
                print(
                    f"[HumanRobotOverlayEval] visualization skipped for "
                    f"{embodiment_name}: {type(error).__name__}: {error}",
                    flush=True,
                )

        if self.log_normalized_mse:
            if normalized_squared_error_sum is None or normalized_element_count == 0:
                raise ValueError(
                    "Normalized MSE requested for an empty validation batch"
                )
            metrics["Valid/MSE"] = (
                normalized_squared_error_sum / normalized_element_count
            ).detach()
        if self.log_native_mse:
            if native_squared_error_sum is None or native_element_count == 0:
                raise ValueError("Native MSE requested for an empty validation batch")
            metrics["Valid/Native_MSE"] = (
                native_squared_error_sum / native_element_count
            ).detach()

        return metrics, images


FoldOverlayEval = HumanRobotOverlayEval
