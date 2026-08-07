"""Exponential moving average of model weights (diffusion-policy style).

Maintains a shadow copy of every floating-point entry in the module's
state_dict (params AND float buffers), updated after each training batch:

    shadow = decay * shadow + (1 - decay) * live

The shadow is written into every Lightning checkpoint under
``ema_state_dict`` (same key layout as ``state_dict``), so offline evals opt
in via ``ckpt_loading --use-ema`` without touching the default path. Resume
restores the shadow from the checkpoint, so EMA survives requeues.

Note on BatchNorm: averaging conv weights while the live BN running stats
match only the LIVE weights is the mismatch that made diffusion-policy swap
BN->GroupNorm. Pair this callback with ``VisualCore(norm_layer="group")``.
(Float BN buffers are EMA'd here too, which is second-best but coherent;
GroupNorm has no such buffers and is the recommended pairing.)
"""
from __future__ import annotations

import torch
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(self, decay: float = 0.9999, start_step: int = 0):
        super().__init__()
        self.decay = float(decay)
        self.start_step = int(start_step)
        self._shadow: dict | None = None

    def _init_shadow(self, pl_module) -> None:
        self._shadow = {
            k: v.detach().clone()
            for k, v in pl_module.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step < self.start_step:
            return
        if self._shadow is None:
            self._init_shadow(pl_module)
            return
        msd = pl_module.state_dict()
        d = self.decay
        for k, s in self._shadow.items():
            v = msd[k]
            if s.device != v.device:
                s.data = s.data.to(v.device)
                self._shadow[k] = s
            s.mul_(d).add_(v.detach(), alpha=1.0 - d)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self._shadow is not None:
            checkpoint["ema_state_dict"] = {
                k: v.detach().cpu().clone() for k, v in self._shadow.items()
            }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        ema = checkpoint.get("ema_state_dict")
        if ema is not None:
            self._shadow = {k: v.clone() for k, v in ema.items()}
