"""Per-top-module gradient-norm logging (restored from the old repo, done
right: MODULE-TREE walk, not name matching — the baseline7 param-split lesson).

Logs every `every_n_steps`: grad_norm/total plus grad_norm/<top-child> for each
direct child of the policy (stages list -> obs encoders, trunk, heads, ...).
Read-only; no clamping (the old MAD clamper stays dead)."""
from __future__ import annotations

import torch
from lightning.pytorch.callbacks import Callback


class GradNormLogger(Callback):
    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every = int(every_n_steps)

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.every:
            return
        policy = getattr(getattr(pl_module, "nets", pl_module), "policy", None)
        root = policy if policy is not None else pl_module
        total_sq = 0.0
        logs = {}
        for name, child in root.named_children():
            sq = 0.0
            for p in child.parameters(recurse=True):
                if p.grad is not None:
                    sq += float(p.grad.detach().float().pow(2).sum())
            if sq > 0:
                logs[f"grad_norm/{name}"] = sq ** 0.5
                total_sq += sq
        logs["grad_norm/total"] = total_sq ** 0.5
        pl_module.log_dict(logs, on_step=True, on_epoch=False, sync_dist=False)
