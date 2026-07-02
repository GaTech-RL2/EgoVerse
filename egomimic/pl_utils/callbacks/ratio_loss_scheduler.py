"""Lightning callback that turns the ChunkerStage ratio-loss regularizer OFF
partway through training.

Motivation: the ratio loss forces the chunker toward its
``target_compression_ratio`` early (so it actually chunks instead of sitting
idle); switching it OFF in the back half lets the task loss reshape the
chunking freely and probes whether the learned compression *survives* without
the regularizer (task-useful) or collapses back to idle (regularizer-induced).

Epoch-based so "halfway through training" is robust to the exact steps/epoch.
By default switches at ``switch_fraction * trainer.max_epochs``. Config-gated;
compose via ``callbacks=[checkpoints, ratio_loss_scheduler]``.
"""
from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl

from egomimic.models.hnet.stages import ChunkerStage


class RatioLossScheduler(pl.Callback):
    def __init__(
        self,
        on_value: float = 0.03,
        off_value: float = 0.0,
        switch_fraction: float = 0.5,
        switch_epoch: Optional[int] = None,
        log_key: str = "train/ratio_loss_weight",
    ):
        super().__init__()
        self.on_value = float(on_value)
        self.off_value = float(off_value)
        self.switch_fraction = float(switch_fraction)
        # Explicit epoch overrides the fraction when provided.
        self.switch_epoch = None if switch_epoch is None else int(switch_epoch)
        self.log_key = log_key
        self._cached: Optional[float] = None

    def _switch_epoch(self, trainer) -> int:
        if self.switch_epoch is not None:
            return self.switch_epoch
        return int(round(self.switch_fraction * int(trainer.max_epochs)))

    def _apply(self, pl_module, value: float) -> None:
        if value == self._cached:
            return
        self._cached = value
        n = 0
        for m in pl_module.modules():
            if isinstance(m, ChunkerStage):
                m.ratio_loss_weight = value
                n += 1
        print(f"[RatioLossScheduler] set ratio_loss_weight={value} on {n} ChunkerStage(s)", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        sw = self._switch_epoch(trainer)
        value = self.on_value if int(trainer.current_epoch) < sw else self.off_value
        self._apply(pl_module, value)
        pl_module.log(self.log_key, float(value), on_step=False, on_epoch=True, prog_bar=False)
