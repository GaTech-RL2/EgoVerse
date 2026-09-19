"""Steady-state step timing for a real trainHydra run.

Lightning's `simple` profiler reports the MEAN over every training batch, which
buries what we want to compare: the first batch carries worker spin-up, cudnn
autotuning and (with `model.compile.enabled`) a whole inductor compile, and at
240 steps a 17 s compile alone shifts the mean by 70 ms/step. This callback
reports the median of the steps after a warmup instead, split into the wait for
the batch and the step itself.
"""

from __future__ import annotations

import statistics
import time

import torch
from lightning import Callback


class StepTimer(Callback):
    def __init__(self, warmup_steps: int = 40, label: str = "step"):
        self.warmup_steps = int(warmup_steps)
        self.label = label
        self._waits: list[float] = []
        self._steps: list[float] = []
        self._t_end = None
        self._t_start = None
        self._epoch_start = None
        self._epoch_first_waits: list[float] = []

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if batch_idx == 0 and self._epoch_start is not None:
            # Wait for the FIRST batch of an epoch. Without persistent_workers
            # Lightning rebuilds the iterator here, which forks the pool and
            # refills every prefetch queue before this returns.
            self._epoch_first_waits.append(now - self._epoch_start)
        elif self._t_end is not None:
            self._waits.append(now - self._t_end)
        self._t_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        self._t_end = time.perf_counter()
        self._steps.append(self._t_end - self._t_start)

    def on_fit_end(self, trainer, pl_module):
        n = self.warmup_steps
        steps, waits = self._steps[n:], self._waits[n:]
        if not steps:
            print(f"[{self.label}] only {len(self._steps)} steps, all warmup")
            return

        def stats(xs):
            xs = sorted(xs)
            return (
                statistics.median(xs) * 1e3,
                xs[len(xs) // 10] * 1e3,
                xs[-max(1, len(xs) // 10)] * 1e3,
            )

        s_med, s_p10, s_p90 = stats(steps)
        w_med, w_p10, w_p90 = stats(waits) if waits else (0.0, 0.0, 0.0)
        print(
            f"[{self.label}] steady state over {len(steps)} steps "
            f"(first {n} dropped): "
            f"step {s_med:.1f} ms (p10 {s_p10:.1f} / p90 {s_p90:.1f}), "
            f"batch wait {w_med:.1f} ms (p10 {w_p10:.1f} / p90 {w_p90:.1f}), "
            f"first step {self._steps[0] * 1e3:.0f} ms",
            flush=True,
        )
        epoch_waits = self._epoch_first_waits
        if epoch_waits:
            per_epoch = ", ".join(f"{w * 1e3:.0f}" for w in epoch_waits)
            print(
                f"[{self.label}] epoch-start batch wait (ms), one per epoch: "
                f"{per_epoch}  (mid-epoch median {w_med:.1f})",
                flush=True,
            )
