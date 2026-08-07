from pytorch_lightning import Callback


class WandbProfilerLogger(Callback):
    """Logs Lightning profiler durations to W&B every N steps."""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        if trainer.profiler is not None and hasattr(
            trainer.profiler, "recorded_durations"
        ):
            metrics_to_log = {}
            for action_name, durations in trainer.profiler.recorded_durations.items():
                if len(durations) > 0:
                    recent_time = durations[-1]
                    metrics_to_log[f"profiler/{action_name}_time_sec"] = recent_time

            if metrics_to_log and trainer.logger:
                trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
