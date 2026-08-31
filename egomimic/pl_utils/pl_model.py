import copy
import random
import time
from collections import OrderedDict, deque
from typing import Any, Dict

import hydra
import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import egomimic.utils.tensor_utils as TensorUtils
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    debug_loss_spike = False
    debug_loss_spike_factor = 1000.0
    debug_loss_spike_prob = 0.03
    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(
        self,
        robomimic_model=None,
        optimizer=None,
        scheduler=None,
        config_tree=None,
        norm_stats_state=None,
        scheduler_interval="step",
        scheduler_frequency: int = 1,
        evaluator=None,
        enable_grad_norm: bool = True,
        train_metrics_on_step: bool = False,
        train_metrics_on_epoch: bool = True,
    ):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["robomimic_model"])

        if config_tree is not None:
            self.model = self._instantiate_model(config_tree, norm_stats_state)
        elif robomimic_model is not None:  # legacy support
            self.model = robomimic_model
        else:
            raise ValueError(
                "ModelWrapper requires either an instantiated robomimic_model or "
                "a config_tree with norm_stats_state."
            )
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)
        self.train_metrics_on_step = train_metrics_on_step
        self.train_metrics_on_epoch = train_metrics_on_epoch

        self.epoch_memory_stats = []  # Store memory stats per epoch
        self.evaluator = evaluator

        self._ema_config = self._resolve_ema_config(config_tree)
        if self._ema_config is not None:
            self.ema_model = copy.deepcopy(self.model)
            # Algo is an orchestration object rather than nn.Module. Register
            # its ModuleDict explicitly so Lightning moves, saves, and strictly
            # reloads the complete EMA tree.
            self.ema_nets = self.ema_model.nets
            self.ema_nets.eval()
            self.ema_nets.requires_grad_(False)
            self.register_buffer(
                "ema_optimization_step", torch.zeros((), dtype=torch.long)
            )
            self.register_buffer("ema_decay", torch.zeros((), dtype=torch.float64))

    @classmethod
    def _resolve_ema_config(cls, config_tree):
        cfg = cls._as_config(config_tree)
        if cfg is None:
            return None
        raw = OmegaConf.select(cfg, "model.ema")
        if raw is None or not bool(raw.get("enabled", False)):
            return None
        values = {
            "update_after_step": int(raw.get("update_after_step", 0)),
            "inv_gamma": float(raw.get("inv_gamma", 1.0)),
            "power": float(raw.get("power", 0.75)),
            "min_value": float(raw.get("min_value", 0.0)),
            "max_value": float(raw.get("max_value", 0.9999)),
            "use_for_validation": bool(raw.get("use_for_validation", True)),
        }
        if values["update_after_step"] < 0 or values["inv_gamma"] <= 0.0:
            raise ValueError("EMA requires update_after_step>=0 and inv_gamma>0")
        if values["power"] <= 0.0:
            raise ValueError("EMA power must be positive")
        if not 0.0 <= values["min_value"] <= values["max_value"] < 1.0:
            raise ValueError("EMA decay bounds must satisfy 0<=min<=max<1")
        return values

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "ema_model"):
            self.ema_nets.eval()
        return self

    def _ema_decay_for_step(self, optimization_step: int) -> float:
        cfg = self._ema_config
        step = max(0, int(optimization_step) - cfg["update_after_step"] - 1)
        value = 1.0 - (1.0 + step / cfg["inv_gamma"]) ** (-cfg["power"])
        if step <= 0:
            return 0.0
        return max(cfg["min_value"], min(value, cfg["max_value"]))

    @torch.no_grad()
    def _update_ema(self):
        if not hasattr(self, "ema_model"):
            return
        # Match diffusers.EMAModel exactly: increment the completed optimizer
        # step before evaluating the warm-up decay schedule.
        self.ema_optimization_step.add_(1)
        decay = self._ema_decay_for_step(int(self.ema_optimization_step.item()))
        online_params = dict(self.model.nets.named_parameters())
        ema_params = dict(self.ema_model.nets.named_parameters())
        if online_params.keys() != ema_params.keys():
            raise RuntimeError("EMA/online parameter trees differ")
        for name, parameter in online_params.items():
            averaged = ema_params[name]
            if not parameter.requires_grad:
                averaged.copy_(parameter.detach().to(dtype=averaged.dtype))
            else:
                averaged.mul_(decay).add_(
                    parameter.detach().to(dtype=averaged.dtype), alpha=1.0 - decay
                )
        online_buffers = dict(self.model.nets.named_buffers())
        ema_buffers = dict(self.ema_model.nets.named_buffers())
        if online_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("EMA/online buffer trees differ")
        for name, value in online_buffers.items():
            ema_buffers[name].copy_(value.detach().to(dtype=ema_buffers[name].dtype))
        self.ema_decay.fill_(decay)

    def on_save_checkpoint(self, checkpoint):
        """Keep runtime logging controls accurate across full-state resumes."""
        hyper_parameters = checkpoint.setdefault("hyper_parameters", {})
        hyper_parameters["enable_grad_norm"] = bool(self.enable_grad_norm)
        hyper_parameters["train_metrics_on_step"] = bool(
            self.train_metrics_on_step
        )
        hyper_parameters["train_metrics_on_epoch"] = bool(
            self.train_metrics_on_epoch
        )
        if self._ema_config is not None:
            hyper_parameters["ema_contract"] = dict(self._ema_config)
            hyper_parameters["ema_optimization_step"] = int(
                self.ema_optimization_step.item()
            )
            hyper_parameters["ema_decay"] = float(self.ema_decay.item())

    @staticmethod
    def _as_config(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig):
            return cfg
        return OmegaConf.create(cfg)

    def _instantiate_model(self, config_tree, norm_stats_state):
        cfg = self._as_config(config_tree)
        norm_stats = MultiDataset.from_state(norm_stats_state)
        return hydra.utils.instantiate(
            cfg.model.robomimic_model,
            norm_stats=norm_stats,
        )

    def _log_train_metric(self, name, value, *, sync_dist=True):
        """Log dense step metrics while retaining a separate epoch aggregate."""
        if not self.train_metrics_on_step and not self.train_metrics_on_epoch:
            return
        if self.train_metrics_on_step:
            self.log(
                name,
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=sync_dist,
            )
        if self.train_metrics_on_epoch:
            epoch_name = f"{name}_epoch" if self.train_metrics_on_step else name
            self.log(
                epoch_name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        self.train()
        loss_dicts = []

        t0 = time.time()
        batch = self.model.process_batch_for_training(batch)
        t1 = time.time()
        predictions = self.model.forward_training(batch)
        t2 = time.time()
        losses = self.model.compute_losses(predictions, batch)
        t3 = time.time()
        loss_dicts.append(losses)

        self._log_train_metric("Timing/Process_Batch_Sec", t1 - t0)
        self._log_train_metric("Timing/Forward_Pass_Sec", t2 - t1)
        self._log_train_metric("Timing/Compute_Losses_Sec", t3 - t2)

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        if (
            self.debug_loss_spike
            and random.random() < self.debug_loss_spike_prob
            and self.global_step > 100
        ):
            losses["action_loss"] = losses["action_loss"] * self.debug_loss_spike_factor
            if self.trainer.is_global_zero:
                print(
                    f"[LOSS_SPIKE] step={self.global_step} factor={self.debug_loss_spike_factor}",
                    flush=True,
                )

        info = {}
        info["losses"] = TensorUtils.detach(losses)
        for k, v in self.model.log_info(info).items():
            self._log_train_metric("Train/" + k, v)

        return losses["action_loss"]

    def on_after_backward(self):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info = {"policy_grad_norms_raw": grad_norm_val}
        grad_norm_flagged = False

        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                grad_norm_flagged = grad_norm_val > threshold
                info["policy_grad_norms_mad_flag"] = float(grad_norm_flagged)
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
            self._log_train_metric("Train/" + k, v)

    def on_before_optimizer_step(self, optimizer):
        if self.train_metrics_on_step:
            for i, param_group in enumerate(optimizer.param_groups):
                self.log(
                    f"Optimizer/param_group_{i}_lr",
                    param_group["lr"],
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self._log_train_metric("Train/policy_grad_norms_clipped", float(grad_norm))

    def on_validation_start(self):
        if self.evaluator is None:
            return
        self.model.device = self.device

        validation_model = self.model
        if self._ema_config is not None and self._ema_config["use_for_validation"]:
            self.ema_model.device = self.device
            validation_model = self.ema_model
        self.evaluator.model = validation_model

        self.evaluator.on_validation_start()

    def on_before_zero_grad(self, optimizer):
        """Update EMA exactly once after each completed optimizer step."""
        del optimizer
        self._update_ema()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.
        """
        if self.evaluator is None:
            return
        batch = self.model.process_batch_for_training(batch)
        print(
            f"[VAL_STEP] rank={self.global_rank}, batch_idx={batch_idx}",
            flush=True,
        )
        self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def on_validation_end(self):
        print(f"[ON_VALIDATION_END] rank={self.global_rank}", flush=True)
        if self.evaluator is not None:
            self.evaluator.on_validation_end()

        print(
            f"Rank {self.global_rank} on validation end, waiting for all ranks to synchronize",
            flush=True,
        )
        torch.distributed.barrier()
        print(
            f"Rank {self.global_rank} on validation end, all ranks synchronized",
            flush=True,
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        config_tree = getattr(self.hparams, "config_tree", None)
        if config_tree is not None:
            cfg = self._as_config(config_tree)
            optimizer = hydra.utils.instantiate(
                cfg.model.optimizer,
                params=(
                    parameter
                    for parameter in self.trainer.model.parameters()
                    if parameter.requires_grad
                ),
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
            optimizer = self.hparams.optimizer(
                params=(
                    parameter
                    for parameter in self.trainer.model.parameters()
                    if parameter.requires_grad
                )
            )
            scheduler = (
                self.hparams.scheduler(optimizer=optimizer)
                if self.hparams.scheduler is not None
                else None
            )

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
        if hasattr(self, "ema_model"):
            self.ema_model.device = self.device
        print(
            f"Rank {self.global_rank} on fit start, waiting for all ranks to synchronize",
            flush=True,
        )
        torch.distributed.barrier()
        print(
            f"Rank {self.global_rank} on fit start, all ranks synchronized", flush=True
        )

    def on_train_epoch_start(self):
        if not self.train_metrics_on_epoch:
            return super().on_train_epoch_start()
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr_epoch"
                if self.train_metrics_on_step
                else f"Optimizer/param_group_{i}_lr",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return super().on_train_epoch_start()
