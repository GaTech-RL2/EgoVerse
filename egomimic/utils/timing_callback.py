import time

import torch
from pytorch_lightning import Callback


class WandbProfilerLogger(Callback):
    """Logs Lightning profiler durations (when a profiler is configured), the
    mean wall-clock per training step since the last log point, and the CUDA
    peak memory to W&B every N steps."""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self._step_start = None
        self._step_times = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._step_start is not None:
            self._step_times.append(time.perf_counter() - self._step_start)
            self._step_start = None
        if batch_idx % self.log_every_n_steps != 0:
            return

        metrics_to_log = {}
        if trainer.profiler is not None and hasattr(
            trainer.profiler, "recorded_durations"
        ):
            for action_name, durations in trainer.profiler.recorded_durations.items():
                if len(durations) > 0:
                    recent_time = durations[-1]
                    metrics_to_log[f"profiler/{action_name}_time_sec"] = recent_time

        if self._step_times:
            metrics_to_log["perf/step_time_sec"] = sum(self._step_times) / len(
                self._step_times
            )
            metrics_to_log["perf/steps_averaged"] = len(self._step_times)
            self._step_times = []
        if torch.cuda.is_available():
            gib = 1024**3
            metrics_to_log["perf/gpu_max_mem_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / gib
            )
            metrics_to_log["perf/gpu_max_mem_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / gib
            )

        if metrics_to_log and trainer.logger:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
