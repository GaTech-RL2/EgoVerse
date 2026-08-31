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

        self.epoch_memory_stats = []  # Store memory stats per epoch
        self.evaluator = evaluator

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

    # batch is now a dict, handle on model side
    def training_step(self, batch, batch_idx):
        if batch_idx < 3:
            print(
                f"[DBG training_step] rank={self.global_rank} "
                f"epoch={self.current_epoch} batch_idx={batch_idx} "
                f"global_step={self.global_step}",
                flush=True,
            )
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
            self.log("Train/" + k, v, sync_dist=True, on_step=False, on_epoch=True)

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
            self.log("Train/" + k, v, on_step=False, on_epoch=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        self.log(
            "Train/policy_grad_norms_clipped",
            float(grad_norm),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _rank_barrier(self, tag: str) -> None:
        """Synchronize all ranks at a named point. Cheap when everyone gets
        here at roughly the same time; useful to avoid NCCL ALLREDUCE
        timeouts when one rank has been slower (e.g. first-epoch zarr JPEG
        read burst, val-loop imbalance, sporadic filesystem stalls). No-op
        outside a distributed run."""
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return
        print(f"[BARRIER {tag}] rank={self.global_rank} entering", flush=True)
        torch.distributed.barrier()
        print(f"[BARRIER {tag}] rank={self.global_rank} released", flush=True)

    def on_validation_start(self):
        # Sync before the val loop so no rank enters validation while others
        # are still finishing the previous training epoch's last step.
        self._rank_barrier("on_validation_start")
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

        # Sync after the val loop — mp4 writers on different ranks finish at
        # different times, and if we don't wait here the next train epoch's
        # first ALLREDUCE can hit the 30-min NCCL default.
        self._rank_barrier("on_validation_end")

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
        # Debug: per-rank dataloader lengths (catches DistributedSampler-caused
        # empty shards where a rank has 0 train batches and its trainer.fit()
        # returns instantly while other ranks hang on the next allreduce).
        try:
            train_dl = self.trainer.train_dataloader
            n_train = len(train_dl) if train_dl is not None else -1
        except Exception as _e:
            n_train = f"<err:{_e!r}>"
        try:
            val_dls = self.trainer.val_dataloaders
            if isinstance(val_dls, list):
                n_val = [len(v) for v in val_dls]
            elif val_dls is not None:
                n_val = len(val_dls)
            else:
                n_val = -1
        except Exception as _e:
            n_val = f"<err:{_e!r}>"
        print(
            f"[DBG on_fit_start] rank={self.global_rank}/{self.trainer.world_size} "
            f"local={self.local_rank} max_epochs={self.trainer.max_epochs} "
            f"limit_train_batches={self.trainer.limit_train_batches} "
            f"len_train_dl={n_train} len_val_dl={n_val}",
            flush=True,
        )
        self._rank_barrier("on_fit_start")

    def on_train_batch_start(self, batch, batch_idx):
        # Only print first few per epoch to avoid log spam. Catches rank-1's
        # non-participation in training_step (if this never prints on rank 1,
        # rank 1 never entered the training loop for that epoch).
        if batch_idx < 3:
            print(
                f"[DBG on_train_batch_start] rank={self.global_rank} "
                f"epoch={self.current_epoch} batch_idx={batch_idx} "
                f"global_step={self.global_step}",
                flush=True,
            )

    def on_train_end(self):
        # Fires when fit() is about to return normally. Rank 1 exiting early
        # (with `("success", JobReturn(...))` in submitit's result.pkl) means
        # this hook fires on rank 1 while other ranks are still training.
        print(
            f"[DBG on_train_end] rank={self.global_rank} "
            f"current_epoch={self.current_epoch} global_step={self.global_step}",
            flush=True,
        )

    def teardown(self, stage: str) -> None:
        print(f"[DBG teardown] rank={self.global_rank} stage={stage}", flush=True)

    def on_train_epoch_start(self):
        # Barrier at the start of each epoch — the first-epoch zarr JPEG
        # read burst is uneven across ranks (shared FS), so this pulls the
        # slow rank in before any per-step ALLREDUCE.
        self._rank_barrier("on_train_epoch_start")
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        # Sync at epoch end so no rank races into on_validation_start (or the
        # next epoch's first step) while another is still finishing its last
        # backward pass.
        self._rank_barrier("on_train_epoch_end")
        return super().on_train_epoch_end()
