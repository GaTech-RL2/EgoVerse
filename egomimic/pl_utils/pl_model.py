import random
import time
from collections import OrderedDict, deque
from typing import Any, Dict

import hydra
import numpy as np
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import egomimic.vendored.robomimic_tensor_utils as TensorUtils
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
        # Optional torch.compile of the heavy net (the HNetOuterStage that
        # forward_training calls via the `outer_stage` property). Env-gated and
        # default-off so all existing runs are byte-identical. NOTE: the H-Net
        # threads a mutable HNetContext + many .item()/dynamic packed shapes, so
        # compile will graph-break heavily; measure before trusting any speedup.
        import os

        if os.environ.get("EGO_COMPILE", "0") == "1":
            _mode = os.environ.get("EGO_COMPILE_MODE", "default")
            self.model.nets["outer_stage"] = torch.compile(
                self.model.nets["outer_stage"], dynamic=True, mode=_mode
            )
            self.nets = self.model.nets
            print(
                f"[ModelWrapper] torch.compile(outer_stage, dynamic=True, mode={_mode}) ENABLED"
            )
        try:
            self.params = self.model.nets["policy"].params
        except Exception:
            pass
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history = deque(maxlen=self.grad_norm_mad_window)

        # Optional PER-LEAF raw grad-norm logging. Gated by a model-config flag
        # so default runs are byte-identical (no new columns, no extra compute).
        # Read from the config tree the same way ``scheduler`` is, with a hard
        # default of False. Enable via CLI/config override
        # ``model.log_per_layer_grad_norms=true``. When on, logs one grad-norm
        # per parameterized LEAF module (every nn.Linear / norm / conv /
        # embedding / direct-param block) rather than a handful of coarse
        # depth-collapsed groups.
        self.log_per_layer_grad_norms = False
        cfg = self._as_config(config_tree)
        if cfg is not None and OmegaConf.select(cfg, "model") is not None:
            self.log_per_layer_grad_norms = bool(
                OmegaConf.select(cfg, "model.log_per_layer_grad_norms", default=False)
            )

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

    @staticmethod
    def _grad_norm_group_key(name: str) -> str:
        """Derive a PER-LEAF module-group key from a parameter name.

        Robust to ANY arch: the key is simply the OWNING MODULE PATH, i.e. the
        full dotted ``named_parameters`` path with the trailing tensor name
        (``.weight`` / ``.bias`` / ``.A_log`` / ...) stripped. This yields one
        group per parameterized leaf module -- every ``nn.Linear``,
        ``LayerNorm`` / ``RMSNorm``, conv, embedding, and any module that owns
        parameters directly -- so a leaf's weight + bias combine in quadrature
        while siblings stay separate. No depth collapsing: the residual /
        projection linears (taper ``proj_in`` / ``proj_out``, dechunker
        ``w_v`` / ``w_h`` / ``w_o`` / gate-MLP / FiLM-MLP, router ``W_in`` +
        head, every block's ``attn.{q,k,v,out}_proj`` + ``mlp.*``) each get
        their own entry instead of being bucketed into one trunk group.
        """
        path = name.rsplit(".", 1)[0]  # drop trailing tensor name -> owning module
        return path if path else name

    def _log_per_layer_grad_norms(self):
        """Log the L2 norm of each LEAF module's raw parameter grads.

        Computed from ``.grad`` after backward and before any clipping, so it
        mirrors ``policy_grad_norms_raw`` and reflects the raw per-leaf learning
        signal. Iterates ``self.nets.named_parameters()`` (the policy's
        ModuleDict — the canonical learnable-parameter tree, same source the
        optimizer's ``parameter_groups`` walks), groups each param under its
        owning leaf-module path, and logs the per-leaf norm under
        ``Train/grad_norm/<full.module.path>``. The quadrature sum over ALL
        per-leaf norms equals ``policy_grad_norms_raw``.
        """
        sq_sums: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, p in self.nets.named_parameters():
            if p.grad is None:
                continue
            key = self._grad_norm_group_key(name)
            sq = p.grad.detach().pow(2).sum()
            if key in sq_sums:
                sq_sums[key] = sq_sums[key] + sq
            else:
                sq_sums[key] = sq
        for key, sq in sq_sums.items():
            self.log(
                f"Train/grad_norm/{key}",
                torch.sqrt(sq),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def on_after_backward(self):
        # Per-layer raw grad norms are gated by their own flag and taken here
        # (after backward, before any clipping) independent of the global
        # grad-norm/MAD machinery below.
        if self.log_per_layer_grad_norms:
            self._log_per_layer_grad_norms()
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

        if torch.distributed.is_available() and torch.distributed.is_initialized():
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
            # Optional hook: algos with custom parameter groups (e.g. H-Net's
            # per-stage LR multipliers + WD=0 on biases/norms) can return a
            # ``list[dict]`` here. Returning ``None`` falls back to flat
            # ``self.trainer.model.parameters()``.
            groups = None
            param_groups_fn = getattr(self.model, "parameter_groups", None)
            if callable(param_groups_fn):
                try:
                    base_lr = float(
                        OmegaConf.select(cfg, "model.optimizer.lr", default=1e-4)
                    )
                    groups = param_groups_fn(base_lr=base_lr)
                except TypeError:
                    # Method exists but doesn't accept base_lr — skip.
                    groups = None

            params_arg = (
                groups if groups is not None else self.trainer.model.parameters()
            )
            # When ``params_arg`` is a ``list[dict]`` of param groups, passing
            # it through ``hydra.utils.instantiate`` (a kwarg to a _partial_
            # target) wraps the dicts as OmegaConf ``DictConfig`` objects,
            # which trips AdamW's tensor-type check. Instantiate the partial
            # first, then call it with the native-Python params arg.
            optimizer_partial = hydra.utils.instantiate(cfg.model.optimizer)
            if callable(optimizer_partial):
                optimizer = optimizer_partial(params=params_arg)
            else:
                optimizer = optimizer_partial
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
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print(
                f"Rank {self.global_rank} on fit start, waiting for all ranks to synchronize",
                flush=True,
            )
            torch.distributed.barrier()
            print(
                f"Rank {self.global_rank} on fit start, all ranks synchronized",
                flush=True,
            )

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
