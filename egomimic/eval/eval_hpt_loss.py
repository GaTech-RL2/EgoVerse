import json
from pathlib import Path

import torch

from egomimic.eval.eval import Eval


class HPTLossEval(Eval):
    """Validation/test metric only: `Valid/action_loss`. No video writes."""

    def __init__(self, limit_val_batches: int = 40):
        self.trainer = None
        self.model = None
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def on_validation_start(self):
        return

    def on_validation_end(self):
        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if hasattr(value, "item"):
                metrics[key] = float(value.item())
            else:
                metrics[key] = float(value)
        path = Path(self.root_dir()) / "val_metrics.json"
        path.write_text(json.dumps(metrics, indent=2))

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.model.forward_training(batch)
        losses = self.model.compute_losses(predictions, batch)
        loss = losses["action_loss"]
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, device=self.trainer.lightning_module.device)
        self.trainer.lightning_module.log(
            "Valid/action_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
