import time
from collections import deque
from collections.abc import Mapping
from numbers import Real
from typing import Any, Dict

import hydra
import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf


def _unwrap_combined_loader_batch(batch):
    """Undo the extra tuple Lightning leaks when val groups are NESTED.

    ``MultiDataModuleWrapper.val_dataloader`` returns one CombinedLoader per val
    group. With a single group it returns that loader bare, Lightning iterates
    it directly and unpacks its ``(batch, batch_idx, dataloader_idx)`` yield, so
    ``batch`` arrives as the ``{source: data}`` mapping the pipeline expects.

    With two or more groups it returns a LIST of CombinedLoaders. Lightning
    wraps that list in a sequential CombinedLoader of its own, which treats each
    inner CombinedLoader as a plain iterable -- so the inner loader's 3-tuple is
    passed through as the batch and one unpack is not enough. The outer loop
    still hands us the correct batch_idx and dataloader_idx (the group index),
    so the inner pair is redundant and dropped. Without this a multi-group run
    dies at the first val step inside PipelineAlgo, which requires a mapping.
    """
    while isinstance(batch, tuple) and len(batch) == 3:
        batch = batch[0]
    return batch


class ModelWrapper(LightningModule):
    """
    Lightning wrapper for a configured PipelineAlgo.
    """

    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(
        self,
        pipeline=None,
        config_tree=None,
        scheduler_interval="step",
        scheduler_frequency: int = 1,
        evaluator=None,
        enable_grad_norm: bool = True,
    ):
        """
        Args:
            pipeline: an already-instantiated PipelineAlgo.
            config_tree: resolved model configuration containing ``model.pipeline``.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["pipeline"])

        if (config_tree is None) == (pipeline is None):
            raise ValueError("Provide exactly one of pipeline or config_tree")
        if config_tree is not None:
            self.model = self._instantiate_model(config_tree)
        else:
            self.model = pipeline
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        self.evaluator = evaluator

    @staticmethod
    def _as_config(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig):
            return cfg
        return OmegaConf.create(cfg)

    def _instantiate_model(self, config_tree):
        cfg = self._as_config(config_tree)
        return hydra.utils.instantiate(cfg.model.pipeline)

    @staticmethod
    def _prediction_log_metrics(predictions, reference: torch.Tensor):
        """Collect finite scalar ``log/*`` outputs under opaque source keys."""

        if not isinstance(predictions, Mapping):
            raise TypeError("Pipeline predictions must be a source mapping")
        metrics = {}
        for source, result in predictions.items():
            if not isinstance(source, str) or not source:
                raise TypeError("Pipeline source keys must be non-empty strings")
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Pipeline result for source {source!r} must be a mapping"
                )
            for key, value in result.items():
                if not isinstance(key, str) or not key.startswith("log/"):
                    continue
                metric = key.removeprefix("log/")
                if not metric:
                    raise ValueError("Pipeline log metric name must not be empty")
                if torch.is_tensor(value):
                    if value.ndim != 0:
                        raise TypeError(
                            f"Pipeline metric {key!r} for source {source!r} "
                            "must be scalar"
                        )
                    scalar = value.detach().to(
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                elif isinstance(value, Real) and not isinstance(value, bool):
                    scalar = torch.tensor(
                        float(value),
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                else:
                    raise TypeError(
                        f"Pipeline metric {key!r} for source {source!r} "
                        "must be a real scalar"
                    )
                if not bool(torch.isfinite(scalar)):
                    raise RuntimeError(
                        f"Non-finite pipeline metric {key!r} for source {source!r}"
                    )
                metrics.setdefault(metric, []).append((source, scalar))
        return metrics

    def _log_prediction_metrics(self, predictions, reference: torch.Tensor) -> None:
        for metric, source_values in self._prediction_log_metrics(
            predictions, reference
        ).items():
            for source, value in source_values:
                self.log(
                    f"Train/{metric}/{source}",
                    value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
            self.log(
                f"Train/{metric}",
                torch.stack([value for _, value in source_values]).mean(),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.train()
        t0 = time.time()
        batch = self.model.process_batch_for_training(batch)
        t1 = time.time()
        predictions = self.model.forward_training(batch)
        t2 = time.time()
        losses = self.model.compute_losses(predictions, batch)
        t3 = time.time()

        self.log(
            "Timing/Process_Batch_Sec",
            t1 - t0,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Timing/Forward_Pass_Sec",
            t2 - t1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "Timing/Compute_Losses_Sec",
            t3 - t2,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        info = {
            "losses": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in losses.items()
            }
        }
        self._log_prediction_metrics(predictions, losses["loss"])
        for k, v in self.model.log_info(info).items():
            self.log("Train/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

        # DiffusionEpsilonLossStage writes the normalized epsilon-prediction MSE
        # as ``log/diffusion_noise``.  Publish stable aggregate and per-source
        # aliases here, outside PipelineAlgo, so the generic pipeline continues
        # to treat source names as opaque loader keys.
        source_mse = []
        for index, source in enumerate(batch):
            value = losses.get(f"source_{index}_log_diffusion_noise")
            if value is None:
                continue
            source_mse.append(value)
            self.log(
                f"Train/MSE/{source}",
                value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        if source_mse:
            self.log(
                "Train/MSE",
                torch.stack(source_mse).mean(),
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

        return losses["loss"]

    def on_after_backward(self):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info = {"pipeline_grad_norms_raw": grad_norm_val}
        grad_norm_flagged = False

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["pipeline_grad_norms_mad_threshold"] = threshold
                grad_norm_flagged = grad_norm_val > threshold
                info["pipeline_grad_norms_mad_flag"] = float(grad_norm_flagged)
                if grad_norm_flagged:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    if self.trainer.is_global_zero:
                        print(
                            "[GRAD_NORM_SPIKE] "
                            f"step={self.global_step} "
                            f"grad_norm={grad_norm_val:.4f} "
                            f"median={median:.4f} "
                            f"mad={mad:.4f} "
                            f"threshold={threshold:.4f}",
                            flush=True,
                        )

        if not grad_norm_flagged:
            self.grad_norm_history.append(grad_norm_val)
        for k, v in info.items():
            self.log("Train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log(
            "Train/pipeline_grad_norms_clipped",
            float(grad_norm),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_start(self):
        if self.evaluator is None:
            return
        self.model.device = self.device

        self.evaluator.on_validation_start()

    def _valid_group_name(self, dataloader_idx: int):
        """Name of the val group Lightning is currently iterating, if known.

        The datamodule owns the positional group list; `dataloader_idx` indexes
        it. Returns None when there is no datamodule (unit tests instantiate the
        wrapper directly) or the index is out of range, and the evaluator then
        keeps its unprefixed metric names.
        """
        datamodule = getattr(self.trainer, "datamodule", None) if self._trainer else None
        names = getattr(datamodule, "valid_group_names", None)
        if not names or not 0 <= int(dataloader_idx) < len(names):
            return None
        return names[int(dataloader_idx)]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Delegate one processed validation batch to the configured evaluator."""
        if self.evaluator is None:
            return
        batch = _unwrap_combined_loader_batch(batch)
        batch = self.model.process_batch_for_training(batch)
        group = self._valid_group_name(dataloader_idx)
        if group is not None and hasattr(self.evaluator, "set_validation_group"):
            self.evaluator.set_validation_group(group)
        self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        if self.evaluator is not None:
            self.evaluator.on_validation_end()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Instantiate the optimizer and optional scheduler from model config."""
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                params=self.trainer.model.parameters(),
            )
            if callable(optimizer):
                optimizer = optimizer()
            scheduler_cfg = cfg.model.get("scheduler")
            if scheduler_cfg is not None:
                scheduler = hydra.utils.instantiate(
                    scheduler_cfg,
                    optimizer=optimizer,
                )
                if callable(scheduler):
                    scheduler = scheduler()
            else:
                scheduler = None
        else:
            raise RuntimeError("ModelWrapper optimizer requires config_tree")

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_start(self):
        self.model.device = self.device

    def on_train_epoch_start(self):
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return super().on_train_epoch_start()
