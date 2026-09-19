"""Per-step train loss and a fixed held-out loss, written to JSONL.

For comparing two training runs that differ only in the augmentation path. The
held-out pass runs in eval mode, so it uses the deterministic eval augs and is
unaffected by the change under test -- it measures what the model learned, not
what it was shown. The RNG is reseeded at the start of every pass so the
flow-matching noise and timesteps are identical across passes and across runs.
"""

from __future__ import annotations

import json
import statistics
import time

import torch
from lightning import Callback


class ABMetrics(Callback):
    def __init__(self, out_path: str, eval_batches: int = 12, eval_seed: int = 4242):
        self.out_path = out_path
        self.eval_batches = int(eval_batches)
        self.eval_seed = int(eval_seed)
        self.train_losses: list[tuple[int, float]] = []
        self.heldout: list[tuple[int, float]] = []
        self._loader = None
        self._t0 = None

    def on_train_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()
        dm = trainer.datamodule
        self._loader = dm._build_val_style_loader(
            dm.valid_datasets, dm.valid_dataloader_params, "valid"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.train_losses.append((int(trainer.global_step), float(loss)))

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        if self._loader is None:
            return
        model = pl_module.model
        was_training = model.nets.training
        model.nets.eval()
        torch.manual_seed(self.eval_seed)
        losses = []
        for i, batch in enumerate(self._loader):
            if i >= self.eval_batches:
                break
            if (
                isinstance(batch, tuple)
                and len(batch) == 3
                and isinstance(batch[0], dict)
            ):
                batch = batch[0]
            processed = model.process_batch_for_training(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model.forward_training(processed)
                losses.append(
                    float(model.compute_losses(preds, processed)["action_loss"])
                )
        if was_training:
            model.nets.train()
        if losses:
            value = statistics.fmean(losses)
            self.heldout.append((int(trainer.global_step), value))
            print(
                f"[ab] step {trainer.global_step} heldout_loss {value:.6f} "
                f"({len(losses)} batches, {time.perf_counter() - self._t0:.0f}s)",
                flush=True,
            )

    def on_fit_end(self, trainer, pl_module):
        payload = {
            "train_losses": self.train_losses,
            "heldout": self.heldout,
            "wall_s": time.perf_counter() - self._t0 if self._t0 else None,
        }
        with open(self.out_path, "w") as f:
            json.dump(payload, f)
        print(f"[ab] wrote {self.out_path}", flush=True)
