"""Teacher-forced Fold action metrics and GT/prediction overlay videos."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

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
        deterministic_seed: int | None = None,
        exact_epoch_metrics: bool = False,
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
        self.deterministic_seed = (
            None if deterministic_seed is None else int(deterministic_seed)
        )
        self.exact_epoch_metrics = bool(exact_epoch_metrics)
        self._rendered_frames: dict[int, int] = {}
        self._exact_sums: dict[int, dict[str, torch.Tensor | int]] = {}

    def on_validation_start(self):
        self._rendered_frames.clear()
        self._exact_sums.clear()
        super().on_validation_start()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.deterministic_seed is None:
            return super().on_validation_step(batch, batch_idx, dataloader_idx)

        device = self.trainer.lightning_module.device
        devices = []
        if device.type == "cuda":
            devices = [device.index if device.index is not None else torch.cuda.current_device()]
        rank = int(getattr(self.trainer, "global_rank", 0))
        seed = self.deterministic_seed + int(batch_idx) + rank * 1_000_003
        # Validation must neither inherit moving training RNG state nor change
        # the next training sample/noise. The same batch on the same rank gets
        # the same latent noise at every checkpoint and every validation pass.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            return super().on_validation_step(batch, batch_idx, dataloader_idx)

    def _accumulate_exact(
        self,
        emb_id: int,
        normalized_squared_error: torch.Tensor,
        native_squared_error: torch.Tensor,
    ) -> None:
        item = self._exact_sums.setdefault(
            int(emb_id),
            {
                "normalized_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "normalized_count": 0,
                "native_sum": torch.zeros(
                    (), device=native_squared_error.device, dtype=torch.float64
                ),
                "native_count": 0,
            },
        )
        item["normalized_sum"] += normalized_squared_error.double().sum().detach()
        item["normalized_count"] += normalized_squared_error.numel()
        item["native_sum"] += native_squared_error.double().sum().detach()
        item["native_count"] += native_squared_error.numel()

    def on_validation_end(self):
        if self.exact_epoch_metrics:
            metrics = {}
            normalized_mses = []
            native_mses = []
            device = self.trainer.lightning_module.device
            for emb_id in sorted(self._exact_sums):
                item = self._exact_sums[emb_id]
                totals = torch.tensor(
                    [
                        float(item["normalized_sum"].item()),
                        float(item["normalized_count"]),
                        float(item["native_sum"].item()),
                        float(item["native_count"]),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                normalized_mse = totals[0] / totals[1]
                native_mse = totals[2] / totals[3]
                embodiment_name = get_embodiment(emb_id).lower()
                metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse.float()
                metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse.float()
                metrics[f"Valid/Element_Count/{embodiment_name}"] = totals[1]
                normalized_mses.append(normalized_mse)
                native_mses.append(native_mse)
            if normalized_mses:
                metrics["Valid/MSE"] = torch.stack(normalized_mses).mean().float()
            if native_mses:
                metrics["Valid/Native_MSE"] = torch.stack(native_mses).mean().float()
            self.trainer.lightning_module.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=False
            )
        super().on_validation_end()

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
        """Return the normalized target in the same chunk shape as prediction."""
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
        return self.model.norm_stats.unnormalize(
            {action_key: prediction}, emb_id
        )[action_key]

    def compute_metrics_and_viz(
        self, batch: dict[int, dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], dict[int, np.ndarray]]:
        predictions = self.model.forward_eval(batch)
        metrics = {}
        images = {}
        normalized_mses = []
        native_mses = []

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
            unnormalized, target = self._unnormalized_target(
                embodiment_batch, emb_id, action_key
            )
            prediction = self._unnormalize_prediction(
                predictions[prediction_key], emb_id, action_key
            )
            count = min(target.shape[0], prediction.shape[0])
            normalized_target = normalized_target[:count]
            normalized_prediction = normalized_prediction[:count]
            if normalized_target.shape != normalized_prediction.shape:
                raise ValueError(
                    f"Normalized prediction/target mismatch: "
                    f"{tuple(normalized_prediction.shape)} vs "
                    f"{tuple(normalized_target.shape)}"
                )
            normalized_squared_error = (
                normalized_prediction - normalized_target
            ).square()
            normalized_mse = normalized_squared_error.mean()
            normalized_mses.append(normalized_mse.detach())
            if not self.exact_epoch_metrics:
                metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse.detach()
            target = target[:count]
            prediction = prediction[:count]
            if target.shape != prediction.shape:
                raise ValueError(
                    f"Prediction/target mismatch after unnormalizing: "
                    f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
                )

            metric_prefix = f"Valid/emb{emb_id}_{action_key}_action"
            metrics.update(self._error_metrics(metric_prefix, prediction, target))
            native_squared_error = (prediction - target).square()
            native_mse = native_squared_error.mean()
            native_mses.append(native_mse.detach())
            if self.exact_epoch_metrics:
                self._accumulate_exact(
                    emb_id, normalized_squared_error, native_squared_error
                )
            else:
                metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse.detach()
            metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] = (
                target[:, :1] - target
            ).square().mean().detach()

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

        if normalized_mses and not self.exact_epoch_metrics:
            metrics["Valid/MSE"] = torch.stack(normalized_mses).mean()
        if native_mses and not self.exact_epoch_metrics:
            metrics["Valid/Native_MSE"] = torch.stack(native_mses).mean()

        return metrics, images


FoldOverlayEval = HumanRobotOverlayEval
