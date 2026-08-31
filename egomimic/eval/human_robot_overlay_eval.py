"""Teacher-forced Fold action metrics and GT/prediction overlay videos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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
        energy_score: dict[str, Any] | None = None,
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
        self.energy_score = dict(energy_score or {})
        self.energy_score_enabled = bool(self.energy_score.get("enabled", False))
        self._energy_seed_bank: list[int] = []
        self._energy_seed_bank_sha256: str | None = None
        self._energy_batches_done = 0
        if self.energy_score_enabled:
            self._configure_energy_score()
        self._rendered_frames: dict[int, int] = {}

    def on_validation_start(self):
        self._rendered_frames.clear()
        self._energy_batches_done = 0
        super().on_validation_start()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _configure_energy_score(self) -> None:
        sample_count = int(self.energy_score.get("sample_count", 0))
        if sample_count != 32:
            raise ValueError("Energy Score requires sample_count=32")
        path = Path(str(self.energy_score.get("seed_bank_path", ""))).resolve()
        expected_sha = str(self.energy_score.get("seed_bank_sha256", ""))
        if not path.is_file() or self._sha256(path) != expected_sha:
            raise ValueError(f"Energy Score seed bank identity mismatch: {path}")
        payload = json.loads(path.read_text())
        seeds = payload.get("seeds") if isinstance(payload, dict) else None
        if not isinstance(seeds, list) or len(seeds) != sample_count:
            raise ValueError("Energy Score seed bank must contain exactly 32 seeds")
        self._energy_seed_bank = [int(seed) for seed in seeds]
        if len(set(self._energy_seed_bank)) != sample_count:
            raise ValueError("Energy Score seeds must be unique")
        self._energy_seed_bank_sha256 = expected_sha

        blocks = dict(self.energy_score.get("distance_blocks", {}))
        if not blocks:
            raise ValueError("Energy Score requires semantic distance blocks")
        covered = set()
        for name, raw in blocks.items():
            indices = [int(index) for index in raw.get("indices", [])]
            weight = float(raw.get("weight", 0.0))
            if not indices or weight <= 0.0 or covered.intersection(indices):
                raise ValueError(f"Invalid Energy Score block {name!r}: {raw}")
            covered.update(indices)
        if covered != set(range(int(self.energy_score.get("action_dim", 0)))):
            raise ValueError(
                "Energy Score semantic blocks must partition every action channel"
            )
        if int(self.energy_score.get("max_batches_per_rank", 0)) <= 0:
            raise ValueError("Energy Score max_batches_per_rank must be positive")
        if not str(self.energy_score.get("artifact_root", "")):
            raise ValueError("Energy Score requires an artifact_root")

    def _energy_distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return a block-balanced normalized chunk distance.

        The tensors may carry any shared leading axes. Their last two axes are
        horizon and action channel; every semantic block contributes one RMS
        term regardless of its coordinate count.
        """

        if left.shape != right.shape:
            left, right = torch.broadcast_tensors(left, right)
        terms = []
        weights = []
        for raw in self.energy_score["distance_blocks"].values():
            indices = [int(index) for index in raw["indices"]]
            weight = float(raw["weight"])
            delta = left[..., indices] - right[..., indices]
            terms.append(delta.float().square().mean(dim=(-2, -1)).sqrt() * weight)
            weights.append(weight)
        return torch.stack(terms).sum(dim=0) / sum(weights)

    def _sample_energy_predictions(
        self, batch: dict[int, dict[str, Any]]
    ) -> dict[int, torch.Tensor]:
        samples: dict[int, list[torch.Tensor]] = {emb_id: [] for emb_id in batch}
        cuda_devices = sorted(
            {
                int(value.device.index)
                for embodiment_batch in batch.values()
                for value in embodiment_batch.values()
                if torch.is_tensor(value)
                and value.device.type == "cuda"
                and value.device.index is not None
            }
        )
        for seed in self._energy_seed_bank:
            # The fixed seed bank must not perturb subsequent training RNG.
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                torch.manual_seed(seed)
                prediction = self.model.forward_eval(batch)
            for emb_id in batch:
                action_key = self.model.resolved_ac_keys[emb_id]
                key = f"emb{emb_id}_{action_key}"
                samples[emb_id].append(prediction[key].detach())
        return {
            emb_id: torch.stack(per_seed, dim=0) for emb_id, per_seed in samples.items()
        }

    def _energy_metrics_and_artifact(
        self,
        batch: dict[int, dict[str, Any]],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        sampled = self._sample_energy_predictions(batch)
        metrics: dict[str, torch.Tensor] = {}
        artifact_domains: dict[str, Any] = {}
        domain_scores = []
        domain_accuracies = []
        domain_diversities = []
        for emb_id, predictions in sampled.items():
            embodiment_name = get_embodiment(emb_id).lower()
            action_key = self.model.resolved_ac_keys[emb_id]
            target = self._normalized_target(batch[emb_id], action_key)
            if predictions.shape[1:] != target.shape:
                raise ValueError(
                    f"Energy Score prediction/target mismatch for {embodiment_name}: "
                    f"{tuple(predictions.shape)} vs {tuple(target.shape)}"
                )
            accuracy_by_condition = self._energy_distance(
                predictions,
                target.unsqueeze(0).expand_as(predictions),
            ).mean(dim=0)
            pairwise = self._energy_distance(predictions[:, None], predictions[None, :])
            sample_count = int(predictions.shape[0])
            off_diagonal = ~torch.eye(
                sample_count, device=pairwise.device, dtype=torch.bool
            )
            diversity_by_condition = (
                pairwise[off_diagonal]
                .reshape(sample_count * (sample_count - 1), predictions.shape[1])
                .mean(dim=0)
            )
            score_by_condition = accuracy_by_condition - 0.5 * diversity_by_condition
            accuracy = accuracy_by_condition.mean().detach()
            diversity = diversity_by_condition.mean().detach()
            score = score_by_condition.mean().detach()
            if not bool(
                torch.isfinite(torch.stack((score, accuracy, diversity))).all()
            ):
                raise ValueError(f"Non-finite Energy Score for {embodiment_name}")
            metrics[f"Valid/EnergyScore@32/{embodiment_name}"] = score
            metrics[f"Valid/EnergyScoreAccuracy@32/{embodiment_name}"] = accuracy
            metrics[f"Valid/EnergyScoreDiversity@32/{embodiment_name}"] = diversity
            domain_scores.append(score)
            domain_accuracies.append(accuracy)
            domain_diversities.append(diversity)
            artifact_domains[embodiment_name] = {
                "embodiment_id": int(emb_id),
                "action_key": action_key,
                "predictions": predictions.float().cpu(),
                "targets": target.detach().float().cpu(),
                "accuracy_by_condition": accuracy_by_condition.float().cpu(),
                "diversity_by_condition": diversity_by_condition.float().cpu(),
                "score_by_condition": score_by_condition.float().cpu(),
            }

        metrics["Valid/EnergyScore@32"] = torch.stack(domain_scores).mean()
        metrics["Valid/EnergyScoreAccuracy@32"] = torch.stack(domain_accuracies).mean()
        metrics["Valid/EnergyScoreDiversity@32"] = torch.stack(
            domain_diversities
        ).mean()

        root = Path(str(self.energy_score["artifact_root"])).resolve()
        step = int(self.trainer.global_step)
        epoch = int(self.trainer.current_epoch)
        rank = int(self.trainer.global_rank)
        destination = (
            root
            / f"epoch-{epoch}-step-{step}"
            / f"rank-{rank}-batch-{int(batch_idx)}.pt"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        payload = {
            "schema_version": 1,
            "metric": "EnergyScore@32",
            "sample_count": 32,
            "seed_bank": self._energy_seed_bank,
            "seed_bank_sha256": self._energy_seed_bank_sha256,
            "distance": {
                "space": "normalized_action_chunk",
                "formula": "mean_weighted_semantic_block_rms",
                "blocks": self.energy_score["distance_blocks"],
            },
            "aggregation": "condition_mean_then_equal_domain_macro_mean",
            "global_step": step,
            "epoch": epoch,
            "rank": rank,
            "batch_idx": int(batch_idx),
            "precision": str(self.trainer.precision),
            "validation_view": self.energy_score.get("validation_view"),
            "provenance": self.energy_score.get("provenance"),
            "domains": artifact_domains,
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        return metrics

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

        if self.energy_score_enabled and self._energy_batches_done < int(
            self.energy_score["max_batches_per_rank"]
        ):
            metrics.update(
                self._energy_metrics_and_artifact(batch, self._energy_batches_done)
            )
            self._energy_batches_done += 1

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
