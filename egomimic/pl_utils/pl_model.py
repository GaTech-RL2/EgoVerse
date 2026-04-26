from collections import deque
from typing import Any, Callable

import numpy as np
import torch
from lightning import LightningModule

from egomimic.pl_utils.graph import Graph


class GraphModel(LightningModule):
    """LightningModule wrapping a :class:`Graph` of nodes.

    The graph is dispatched with ``global_mode='train'`` in ``training_step``
    and ``'eval'`` in ``validation_step``. ``loss_fn`` is a callable
    ``(outputs, batch) -> dict`` whose returned dict must contain a scalar
    ``'loss'`` key (the value backpropped); all other keys are logged.
    """

    grad_norm_mad_scale = 3.0
    grad_norm_mad_min_count = 100
    grad_norm_mad_window = 200

    def __init__(
        self,
        graph: Graph,
        optimizer: Callable,
        scheduler: Callable | None = None,
        loss_fn: Callable[[dict, dict], dict] | None = None,
        scheduler_interval: str = "step",
        scheduler_frequency: int = 1,
        enable_grad_norm: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["graph", "loss_fn"])
        self.graph = graph
        self.loss_fn = loss_fn
        self._optimizer_partial = optimizer
        self._scheduler_partial = scheduler
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.enable_grad_norm = enable_grad_norm
        self.grad_norm_history: deque[float] = deque(maxlen=self.grad_norm_mad_window)

    def _log_losses(self, losses: dict[str, Any]) -> None:
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self.log(f"Train/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self.graph(batch, global_mode="train")
        losses = self.loss_fn(outputs, batch)
        self._log_losses(losses)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        return self.graph(batch, global_mode="eval")

    def on_after_backward(self):
        if not self.enable_grad_norm:
            return
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=float("inf")
        )
        grad_norm_val = float(grad_norm)
        info: dict[str, float] = {"policy_grad_norms_raw": grad_norm_val}
        flagged = False
        if len(self.grad_norm_history) >= self.grad_norm_mad_min_count:
            values = np.array(self.grad_norm_history, dtype=np.float32)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                threshold = median + self.grad_norm_mad_scale * mad
                info["policy_grad_norms_mad_threshold"] = threshold
                flagged = grad_norm_val > threshold
                info["policy_grad_norms_mad_flag"] = float(flagged)
                if flagged:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=median)
                    if self.trainer.is_global_zero:
                        print(
                            f"[GRAD_NORM_SPIKE] step={self.global_step} "
                            f"grad_norm={grad_norm_val:.4f} median={median:.4f} "
                            f"mad={mad:.4f} threshold={threshold:.4f}",
                            flush=True,
                        )
        if not flagged:
            self.grad_norm_history.append(grad_norm_val)
        for k, v in info.items():
            self.log(f"Train/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

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

    def on_train_epoch_start(self):
        for i, group in enumerate(self.optimizers().param_groups):
            self.log(
                f"Optimizer/param_group_{i}_lr",
                group["lr"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return super().on_train_epoch_start()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self._optimizer_partial(params=self.trainer.model.parameters())
        if self._scheduler_partial is None:
            return {"optimizer": optimizer}
        scheduler = self._scheduler_partial(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
            },
        }
