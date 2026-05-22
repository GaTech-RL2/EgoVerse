"""Lightning callback that drives ChunkerStage.residual_scale over training.

Wired through Hydra ``callbacks=[..., chunker_residual_scheduler]``. The
schedule is a list of ``(step, value)`` knots, interpolated piecewise-linearly
by ``egomimic.utils.schedulers.piecewise_linear``. Setting

    schedule: [[0, 0.0], [319, 0.0], [320, 1.0]]

reproduces "skip path off for first half, on for second half" with a hard
step at 50% of a 640-step run. Tweak to taste.
"""
from __future__ import annotations

import lightning.pytorch as pl

from egomimic.models.hnet_nets.stages import ChunkerStage
from egomimic.utils.schedulers import piecewise_linear


class ChunkerResidualScheduler(pl.Callback):
    def __init__(self, schedule, log_key: str = "train/chunker_residual_scale"):
        super().__init__()
        if not schedule:
            raise ValueError("ChunkerResidualScheduler.schedule must be non-empty")
        self.schedule = [(int(s), float(v)) for s, v in schedule]
        self.log_key = log_key
        self._cached_scale: float | None = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = int(trainer.global_step)
        scale = piecewise_linear(step, self.schedule)
        if scale != self._cached_scale:
            self._cached_scale = scale
            # Walk modules once per change rather than every batch.
            for m in pl_module.modules():
                if isinstance(m, ChunkerStage):
                    m.residual_scale = scale
        pl_module.log(self.log_key, float(scale), on_step=True, on_epoch=False, prog_bar=False)
