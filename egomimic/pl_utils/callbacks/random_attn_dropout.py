"""Per-batch random attention-dropout rate, sampled from a list.

Walks ``pl_module.modules()``, finds every ``MultiHeadAttention``-like module
that exposes a ``dropout`` attribute, and writes the sampled rate to it. The
attention forward / step paths read ``self.dropout`` each call, so this takes
effect on the next forward.

Wire via:
    callbacks=[..., random_attn_dropout]
with the YAML at hydra_configs/callbacks/random_attn_dropout.yaml.

Logs the per-step rate as ``train/attn_dropout`` for wandb visibility.
"""
from __future__ import annotations

import random
import lightning.pytorch as pl

from egomimic.models.hnet_nets.blocks import MultiHeadAttention


class RandomAttnDropoutScheduler(pl.Callback):
    def __init__(
        self,
        dropout_values=(0.0, 0.5, 0.9, 0.95),
        weights=None,
        seed: int | None = None,
        log_key: str = "train/attn_dropout",
    ):
        super().__init__()
        if not dropout_values:
            raise ValueError("dropout_values must be non-empty")
        self.dropout_values = [float(v) for v in dropout_values]
        if weights is None:
            self.weights = [1.0] * len(self.dropout_values)
        else:
            if len(weights) != len(self.dropout_values):
                raise ValueError("weights must match dropout_values length")
            self.weights = [float(w) for w in weights]
        self.log_key = log_key
        self._rng = random.Random(seed)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Same rate for every attn layer this batch — keeps the regime
        # uniform across the trunk so the model can't route around it.
        rate = self._rng.choices(self.dropout_values, weights=self.weights, k=1)[0]
        for m in pl_module.modules():
            if isinstance(m, MultiHeadAttention) and hasattr(m, "dropout"):
                m.dropout = float(rate)
        pl_module.log(self.log_key, float(rate), on_step=True, on_epoch=False, prog_bar=False)
