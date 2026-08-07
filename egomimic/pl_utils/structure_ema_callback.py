"""Freeze a transplanted param group, then let it move ONLY via a slow EMA.

Two-phase schedule (the user's recipe):
  * epoch <  freeze_until_epoch : HARD FREEZE (requires_grad=False). The
    transplanted structure holds its donor (aligned) weights as a fixed target
    while the fresh encoder/trunk/head learn to feed & read it.
  * epoch >= freeze_until_epoch : SLOW EMA. requires_grad is re-enabled (the
    params are already in the optimizer, dormant), and after every optimizer
    step we pull each param back onto a high-decay EMA of itself:
        shadow = decay*shadow + (1-decay)*param ;  param <- shadow
    so the *effective* step is ~ (1-decay) of the optimizer's step -> the
    structure adapts slowly, anchored near its aligned init, instead of being
    yanked by the (now-reasonable) fresh compute.

Knobs: freeze_until_epoch=0 -> EMA from the start; =10**9 -> frozen forever.
Shadow + active flag are checkpointed so a requeue resumes mid-schedule.
"""
from __future__ import annotations

import torch
from lightning.pytorch.callbacks import Callback

from egomimic.pl_utils.param_groups import resolve_param_groups


class StructureEMACallback(Callback):
    def __init__(self, param_groups: dict, target_group: str = "structure",
                 freeze_until_epoch: int = 499, decay: float = 0.999):
        super().__init__()
        self.specs = dict(param_groups)
        self.target_group = target_group
        self.freeze_until = int(freeze_until_epoch)
        self.decay = float(decay)
        self._target = None          # list[(name, param)]
        self._shadow = None          # dict[name -> tensor]
        self._active = False
        self._pending = None         # shadow loaded from ckpt before setup

    def setup(self, trainer, pl_module, stage=None):
        if self._target is not None or stage not in (None, "fit"):
            return
        groups = resolve_param_groups(pl_module.named_parameters(), self.specs)
        self._target = groups[self.target_group]
        for _, p in self._target:
            p.requires_grad_(False)
        n = sum(p.numel() for _, p in self._target) / 1e6
        print(f"[struct-ema] froze {len(self._target)} params ({n:.1f}M) of "
              f"group '{self.target_group}' until ep{self.freeze_until} "
              f"(decay {self.decay})")

    def _activate(self, epoch):
        for _, p in self._target:
            p.requires_grad_(True)
        if self._pending is not None:              # resume: restore saved shadow
            self._shadow = {k: v for k, v in self._pending.items()}
        else:
            self._shadow = {n: p.detach().clone() for n, p in self._target}
        self._active = True
        print(f"[struct-ema] ACTIVATED slow-EMA at ep{epoch}")

    def on_train_epoch_start(self, trainer, pl_module):
        # Runs AFTER on_load_checkpoint, so _active/_pending are populated.
        # Fire _activate() on either: (a) reaching the freeze boundary, or
        # (b) a resume that restored _active=True but not _shadow (setup ran
        # before the ckpt was loaded) -> _activate applies _pending.
        resume_needs_shadow = self._active and self._shadow is None
        if resume_needs_shadow or (
                not self._active
                and trainer.current_epoch >= self.freeze_until):
            self._activate(trainer.current_epoch)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._active:
            return
        d = self.decay
        for n, p in self._target:
            s = self._shadow[n]
            if s.device != p.device:
                s = s.to(p.device); self._shadow[n] = s
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
            p.data.copy_(s)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["struct_ema"] = {
            "active": self._active,
            "shadow": None if self._shadow is None else
            {k: v.detach().cpu().clone() for k, v in self._shadow.items()},
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        st = checkpoint.get("struct_ema")
        if st:
            self._active = bool(st["active"])
            self._pending = st["shadow"]  # applied on next _activate
