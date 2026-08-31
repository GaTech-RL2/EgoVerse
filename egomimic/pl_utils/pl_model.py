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
        unite_flow_updates_per_reconstruction: int = 0,
        unite_gradient_telemetry_every_n_steps: int = 0,
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
        self.unite_flow_updates_per_reconstruction = int(
            unite_flow_updates_per_reconstruction
        )
        self.unite_gradient_telemetry_every_n_steps = int(
            unite_gradient_telemetry_every_n_steps
        )
        if self.unite_flow_updates_per_reconstruction < 0:
            raise ValueError("unite_flow_updates_per_reconstruction must be non-negative")
        if self.unite_gradient_telemetry_every_n_steps < 0:
            raise ValueError(
                "unite_gradient_telemetry_every_n_steps must be non-negative"
            )
        if (
            self.unite_flow_updates_per_reconstruction > 0
            and self.unite_gradient_telemetry_every_n_steps == 0
        ):
            raise ValueError(
                "Alternating UNITE updates require shared-gradient telemetry"
            )

        self.epoch_memory_stats = []  # Store memory stats per epoch
        self.evaluator = evaluator

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
        hyper_parameters["unite_flow_updates_per_reconstruction"] = int(
            self.unite_flow_updates_per_reconstruction
        )
        hyper_parameters["unite_gradient_telemetry_every_n_steps"] = int(
            self.unite_gradient_telemetry_every_n_steps
        )

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

    @staticmethod
    def _mean_loss_terms(losses, suffix: str):
        terms = [
            value
            for key, value in losses.items()
            if key.endswith(suffix) and torch.is_tensor(value) and value.ndim == 0
        ]
        if not terms:
            raise RuntimeError(f"Missing UNITE loss terms ending in {suffix!r}")
        return torch.stack(terms).mean()

    @staticmethod
    def _select_unite_update_loss(
        reconstruction_loss,
        flow_loss,
        global_step: int,
        flow_updates_per_reconstruction: int,
    ):
        ratio = int(flow_updates_per_reconstruction)
        if ratio <= 0:
            return reconstruction_loss + flow_loss, "joint", 0
        cycle_position = int(global_step) % (ratio + 1)
        if cycle_position < ratio:
            return flow_loss, "flow", cycle_position
        return reconstruction_loss, "reconstruction", cycle_position

    def _unite_shared_parameters(self):
        matches = [
            stage
            for stage in self.model.policy.stages
            if hasattr(stage, "shared_reconstruction_denoising_named_parameters")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "UNITE gradient telemetry requires exactly one latent policy; "
                f"found {len(matches)}"
            )
        named = matches[0].shared_reconstruction_denoising_named_parameters(
            self.model.domains
        )
        return tuple(name for name, _ in named), tuple(
            parameter for _, parameter in named
        )

    def _measure_unite_shared_gradients(self, reconstruction_loss, flow_loss):
        names, parameters = self._unite_shared_parameters()
        reconstruction_gradients = torch.autograd.grad(
            reconstruction_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        flow_gradients = torch.autograd.grad(
            flow_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for gradient in (*reconstruction_gradients, *flow_gradients):
                torch.distributed.all_reduce(
                    gradient,
                    op=torch.distributed.ReduceOp.SUM,
                )
                gradient.div_(world_size)

        dot = torch.zeros((), device=self.device, dtype=torch.float32)
        reconstruction_square = torch.zeros_like(dot)
        flow_square = torch.zeros_like(dot)
        for reconstruction_gradient, flow_gradient in zip(
            reconstruction_gradients, flow_gradients
        ):
            reconstruction_gradient = reconstruction_gradient.float()
            flow_gradient = flow_gradient.float()
            dot.add_(torch.sum(reconstruction_gradient * flow_gradient))
            reconstruction_square.add_(torch.sum(reconstruction_gradient.square()))
            flow_square.add_(torch.sum(flow_gradient.square()))

        reconstruction_norm = reconstruction_square.sqrt()
        flow_norm = flow_square.sqrt()
        values = torch.stack((dot, reconstruction_norm, flow_norm))
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("Non-finite UNITE shared-gradient telemetry")
        if float(reconstruction_norm) <= 0.0 or float(flow_norm) <= 0.0:
            raise RuntimeError(
                "UNITE shared-gradient telemetry encountered a zero gradient norm"
            )
        cosine = dot / (reconstruction_norm * flow_norm)
        if not bool(torch.isfinite(cosine)):
            raise RuntimeError("Non-finite UNITE shared-gradient cosine")

        self._log_train_metric(
            "log/unite_gradient_cosine", cosine.detach(), sync_dist=False
        )
        self._log_train_metric(
            "log/unite_recon_grad_norm",
            reconstruction_norm.detach(),
            sync_dist=False,
        )
        self._log_train_metric(
            "log/unite_denoise_grad_norm", flow_norm.detach(), sync_dist=False
        )
        self._log_train_metric(
            "log/unite_gradient_parameter_count",
            float(sum(parameter.numel() for parameter in parameters)),
            sync_dist=False,
        )
        self._log_train_metric(
            "log/unite_gradient_tensor_count", float(len(names)), sync_dist=False
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

        if self.unite_flow_updates_per_reconstruction > 0:
            reconstruction_loss = self._mean_loss_terms(
                losses, "_loss_unite_reconstruction"
            )
            flow_loss = self._mean_loss_terms(losses, "_loss_unite_latent")
            selected_loss, update_mode, cycle_position = (
                self._select_unite_update_loss(
                    reconstruction_loss,
                    flow_loss,
                    int(self.global_step),
                    self.unite_flow_updates_per_reconstruction,
                )
            )
            losses["action_loss"] = selected_loss
            self._log_train_metric(
                "log/unite_update_is_flow", float(update_mode == "flow")
            )
            self._log_train_metric(
                "log/unite_update_is_reconstruction",
                float(update_mode == "reconstruction"),
            )
            self._log_train_metric(
                "log/unite_update_cycle_position", float(cycle_position)
            )
            if (
                int(self.global_step)
                % self.unite_gradient_telemetry_every_n_steps
                == 0
            ):
                self._measure_unite_shared_gradients(
                    reconstruction_loss, flow_loss
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

        self.evaluator.on_validation_start()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run a validation step on the batch, and save that batch of images into the val_image_buffer.  Once the buffer hits 1000 images, save that as a 30fps video using torchvision.io.write_video.
        """
        if self.evaluator is None:
            return
        # CombinedLoader(mode="max_size") emits ``None`` for a domain after
        # that domain is exhausted. This lets a fixed multi-embodiment
        # validation pass consume every window exactly once even when the
        # domains have different lengths.
        if isinstance(batch, dict):
            batch = {name: value for name, value in batch.items() if value is not None}
        if not batch:
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
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
