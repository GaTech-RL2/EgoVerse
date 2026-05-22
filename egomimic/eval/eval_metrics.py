import torch

from egomimic.eval.eval import Eval


class EvalMetrics(Eval):
    """Evaluator that runs full validation passes and logs metrics only.

    Unlike :class:`EvalVideo`, this evaluator does not buffer or write any
    validation videos. The images returned by ``forward_eval_logging`` are
    discarded so the eval pass is purely metric-driven. ``limit_val_batches``
    defaults to ``1.0`` so the full validation set is used.
    """

    def __init__(self):
        super().__init__()
        self.trainer = None
        self.model = None
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": 1.0,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def on_validation_start(self):
        return

    def on_validation_end(self):
        return

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics, _ = self.model.forward_eval_logging(batch)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
