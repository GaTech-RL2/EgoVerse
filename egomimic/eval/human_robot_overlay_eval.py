"""Teacher-forced Fold action metrics and GT/prediction overlay videos."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from egomimic.eval.eval_video import EvalVideo
from egomimic.pipeline import packed
from egomimic.pipeline.losses import conditional_energy_score
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
        energy_score_beta: float = 1.0,
        energy_score_normalize_by_dimension: bool = True,
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
        self.energy_score_beta = float(energy_score_beta)
        self.energy_score_normalize_by_dimension = bool(
            energy_score_normalize_by_dimension
        )
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
            devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
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
        token_squared_error: torch.Tensor,
        native_squared_error: torch.Tensor,
        energy_metrics: dict[str, torch.Tensor] | None = None,
    ) -> None:
        item = self._exact_sums.setdefault(
            int(emb_id),
            {
                "normalized_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "normalized_count": 0,
                "token_sum": torch.zeros(
                    (), device=token_squared_error.device, dtype=torch.float64
                ),
                "token_count": 0,
                "native_sum": torch.zeros(
                    (), device=native_squared_error.device, dtype=torch.float64
                ),
                "native_count": 0,
                "energy_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "energy_attraction_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "energy_repulsion_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "energy_pairwise_sum": torch.zeros(
                    (), device=normalized_squared_error.device, dtype=torch.float64
                ),
                "energy_count": 0,
            },
        )
        item["normalized_sum"] += normalized_squared_error.double().sum().detach()
        item["normalized_count"] += normalized_squared_error.numel()
        item["token_sum"] += token_squared_error.double().sum().detach()
        item["token_count"] += token_squared_error.numel()
        item["native_sum"] += native_squared_error.double().sum().detach()
        item["native_count"] += native_squared_error.numel()
        if energy_metrics is not None:
            per_condition = energy_metrics["per_condition_score"]
            item["energy_sum"] += per_condition.double().sum().detach()
            item["energy_attraction_sum"] += (
                energy_metrics["per_condition_attraction"].double().sum().detach()
            )
            item["energy_repulsion_sum"] += (
                energy_metrics["per_condition_repulsion"].double().sum().detach()
            )
            item["energy_pairwise_sum"] += (
                energy_metrics["per_condition_pairwise_distance"]
                .double()
                .sum()
                .detach()
            )
            item["energy_count"] += per_condition.numel()

    def on_validation_end(self):
        if self.exact_epoch_metrics:
            metrics = {}
            normalized_mses = []
            token_mses = []
            native_mses = []
            energy_scores = []
            energy_attractions = []
            energy_repulsions = []
            energy_pairwise_distances = []
            device = self.trainer.lightning_module.device
            for emb_id in sorted(self._exact_sums):
                item = self._exact_sums[emb_id]
                totals = torch.tensor(
                    [
                        float(item["normalized_sum"].item()),
                        float(item["normalized_count"]),
                        float(item["token_sum"].item()),
                        float(item["token_count"]),
                        float(item["native_sum"].item()),
                        float(item["native_count"]),
                        float(item["energy_sum"].item()),
                        float(item["energy_attraction_sum"].item()),
                        float(item["energy_repulsion_sum"].item()),
                        float(item["energy_pairwise_sum"].item()),
                        float(item["energy_count"]),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                normalized_mse = totals[0] / totals[1]
                token_mse = totals[2] / totals[3]
                native_mse = totals[4] / totals[5]
                embodiment_name = get_embodiment(emb_id).lower()
                metrics[f"Valid/MSE/{embodiment_name}"] = normalized_mse.float()
                metrics[f"Valid/ActionToken_MSE/{embodiment_name}"] = token_mse.float()
                metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse.float()
                metrics[f"Valid/Element_Count/{embodiment_name}"] = totals[1]
                normalized_mses.append(normalized_mse)
                token_mses.append(token_mse)
                native_mses.append(native_mse)
                if totals[10] > 0:
                    score = totals[6] / totals[10]
                    attraction = totals[7] / totals[10]
                    repulsion = totals[8] / totals[10]
                    pairwise = totals[9] / totals[10]
                    metrics[f"Valid/EnergyScore/{embodiment_name}"] = score.float()
                    metrics[f"Valid/EnergyAttraction/{embodiment_name}"] = (
                        attraction.float()
                    )
                    metrics[f"Valid/EnergyRepulsion/{embodiment_name}"] = (
                        repulsion.float()
                    )
                    metrics[f"Valid/PairwiseDistance/{embodiment_name}"] = (
                        pairwise.float()
                    )
                    energy_scores.append(score)
                    energy_attractions.append(attraction)
                    energy_repulsions.append(repulsion)
                    energy_pairwise_distances.append(pairwise)
            if normalized_mses:
                metrics["Valid/MSE"] = torch.stack(normalized_mses).mean().float()
            if token_mses:
                metrics["Valid/ActionToken_MSE"] = (
                    torch.stack(token_mses).mean().float()
                )
            if native_mses:
                metrics["Valid/Native_MSE"] = torch.stack(native_mses).mean().float()
            if energy_scores:
                metrics["Valid/EnergyScore"] = torch.stack(energy_scores).mean().float()
                metrics["Valid/EnergyAttraction"] = (
                    torch.stack(energy_attractions).mean().float()
                )
                metrics["Valid/EnergyRepulsion"] = (
                    torch.stack(energy_repulsions).mean().float()
                )
                metrics["Valid/PairwiseDistance"] = (
                    torch.stack(energy_pairwise_distances).mean().float()
                )
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
        return self.model.norm_stats.unnormalize({action_key: prediction}, emb_id)[
            action_key
        ]

    def _unnormalize_prediction_samples(self, samples, emb_id, action_key):
        if samples.ndim != 4:
            raise ValueError(
                "Expected grouped prediction shaped (batch, K, horizon, action_dim), "
                f"got {tuple(samples.shape)}"
            )
        batch_size, num_samples, horizon, action_dim = samples.shape
        flattened = samples.reshape(batch_size * num_samples, horizon, action_dim)
        return self._unnormalize_prediction(flattened, emb_id, action_key).reshape(
            batch_size, num_samples, horizon, action_dim
        )

    @staticmethod
    def _as_tensor_like(value, template: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=template.device, dtype=torch.float32)
        return torch.as_tensor(value, device=template.device, dtype=torch.float32)

    def _decode_native_actions(self, tokens, emb_id, context):
        adapter_getter = getattr(self.model, "rollout_adapter_for", None)
        adapter = None if adapter_getter is None else adapter_getter(emb_id)
        if adapter is None:
            return tokens.float(), None
        if tokens.ndim == 3:
            decoded = adapter.decode(tokens, context)
        elif tokens.ndim == 4:
            decoded = torch.stack(
                [
                    self._as_tensor_like(
                        adapter.decode(tokens[:, index], context), tokens
                    )
                    for index in range(tokens.shape[1])
                ],
                dim=1,
            )
        else:
            raise ValueError(
                "Native decoding expects (B,H,D) or (B,K,H,D), got "
                f"{tuple(tokens.shape)}"
            )
        return self._as_tensor_like(decoded, tokens), getattr(
            adapter, "native_angle_index", None
        )

    @staticmethod
    def _native_squared_error(prediction, target, angle_index):
        difference = prediction.float() - target.float()
        if angle_index is not None:
            angle_index = int(angle_index)
            angle = difference[..., angle_index]
            difference = difference.clone()
            difference[..., angle_index] = torch.atan2(
                torch.sin(angle), torch.cos(angle)
            )
        return difference.square()

    def compute_metrics_and_viz(
        self, batch: dict[int, dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], dict[int, np.ndarray]]:
        predictions = self.model.forward_eval(batch)
        metrics = {}
        images = {}
        normalized_mses = []
        token_mses = []
        native_mses = []
        energy_scores = []
        energy_attractions = []
        energy_repulsions = []
        energy_pairwise_distances = []

        for emb_id, embodiment_batch in batch.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            prediction_key = f"emb{emb_id}_{action_key}"
            if prediction_key not in predictions:
                raise KeyError(
                    f"Missing {prediction_key!r}; available={sorted(predictions)}"
                )

            normalized_prediction = predictions[prediction_key]
            samples_key = f"emb{emb_id}_{action_key}_samples"
            normalized_samples = predictions.get(samples_key)
            normalized_target = self._normalized_target(embodiment_batch, action_key)
            unnormalized, target = self._unnormalized_target(
                embodiment_batch, emb_id, action_key
            )
            prediction = self._unnormalize_prediction(
                predictions[prediction_key], emb_id, action_key
            )
            token_samples = (
                None
                if normalized_samples is None
                else self._unnormalize_prediction_samples(
                    normalized_samples, emb_id, action_key
                )
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
            grouped_energy = None
            if normalized_samples is not None and normalized_samples.shape[1] >= 2:
                normalized_samples = normalized_samples[:count]
                grouped_energy = conditional_energy_score(
                    normalized_samples,
                    normalized_target,
                    beta=self.energy_score_beta,
                    normalize_by_dimension=self.energy_score_normalize_by_dimension,
                )
                normalized_squared_error = (
                    normalized_samples - normalized_target[:, None]
                ).square()
                energy_scores.append(grouped_energy["score"].detach())
                energy_attractions.append(grouped_energy["attraction"].detach())
                energy_repulsions.append(grouped_energy["repulsion"].detach())
                energy_pairwise_distances.append(
                    grouped_energy["pairwise_distance"].detach()
                )
                if not self.exact_epoch_metrics:
                    metrics[f"Valid/EnergyScore/{embodiment_name}"] = grouped_energy[
                        "score"
                    ].detach()
                    metrics[f"Valid/EnergyAttraction/{embodiment_name}"] = (
                        grouped_energy["attraction"].detach()
                    )
                    metrics[f"Valid/EnergyRepulsion/{embodiment_name}"] = (
                        grouped_energy["repulsion"].detach()
                    )
                    metrics[f"Valid/PairwiseDistance/{embodiment_name}"] = (
                        grouped_energy["pairwise_distance"].detach()
                    )
            else:
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
            if token_samples is not None and token_samples.shape[1] >= 2:
                token_squared_error = (token_samples[:count] - target[:, None]).square()
            else:
                token_squared_error = (prediction - target).square()
            token_mse = token_squared_error.mean()
            token_mses.append(token_mse.detach())

            native_target, target_angle_index = self._decode_native_actions(
                target, emb_id, unnormalized
            )
            if token_samples is not None and token_samples.shape[1] >= 2:
                native_prediction, prediction_angle_index = self._decode_native_actions(
                    token_samples[:count], emb_id, unnormalized
                )
                native_target_for_error = native_target[:, None]
            else:
                native_prediction, prediction_angle_index = self._decode_native_actions(
                    prediction, emb_id, unnormalized
                )
                native_target_for_error = native_target
            if prediction_angle_index != target_angle_index:
                raise RuntimeError(
                    "Prediction/target rollout adapters disagree on angle index: "
                    f"{prediction_angle_index} vs {target_angle_index}"
                )
            native_squared_error = self._native_squared_error(
                native_prediction, native_target_for_error, target_angle_index
            )
            native_mse = native_squared_error.mean()
            native_mses.append(native_mse.detach())
            if self.exact_epoch_metrics:
                self._accumulate_exact(
                    emb_id,
                    normalized_squared_error,
                    token_squared_error,
                    native_squared_error,
                    energy_metrics=grouped_energy,
                )
            else:
                metrics[f"Valid/ActionToken_MSE/{embodiment_name}"] = token_mse.detach()
                metrics[f"Valid/Native_MSE/{embodiment_name}"] = native_mse.detach()
            metrics[f"Valid/emb{emb_id}_{action_key}_copybaseline_mse"] = (
                (target[:, :1] - target).square().mean().detach()
            )

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
        if token_mses and not self.exact_epoch_metrics:
            metrics["Valid/ActionToken_MSE"] = torch.stack(token_mses).mean()
        if native_mses and not self.exact_epoch_metrics:
            metrics["Valid/Native_MSE"] = torch.stack(native_mses).mean()
        if energy_scores and not self.exact_epoch_metrics:
            metrics["Valid/EnergyScore"] = torch.stack(energy_scores).mean()
            metrics["Valid/EnergyAttraction"] = torch.stack(energy_attractions).mean()
            metrics["Valid/EnergyRepulsion"] = torch.stack(energy_repulsions).mean()
            metrics["Valid/PairwiseDistance"] = torch.stack(
                energy_pairwise_distances
            ).mean()

        return metrics, images


FoldOverlayEval = HumanRobotOverlayEval
