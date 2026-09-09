"""Fail-closed parameter EMA with optional Diffusion Policy warmup."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


class EMACallback(Callback):
    """Track online parameters separately and validate with averaged values."""

    def __init__(
        self,
        decay: float = 0.9999,
        validate_with_ema: bool = True,
        use_warmup: bool = False,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_decay: float = 0.0,
    ):
        super().__init__()
        self.decay = float(decay)
        self.validate_with_ema = bool(validate_with_ema)
        self.use_warmup = bool(use_warmup)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_decay = float(min_decay)
        if not 0 <= self.min_decay <= self.decay < 1 or not math.isfinite(self.decay):
            raise ValueError("EMA decay must satisfy 0 <= min_decay <= decay < 1")
        if self.update_after_step < 0 or self.inv_gamma <= 0 or self.power <= 0:
            raise ValueError("EMA warmup parameters must be positive")
        self._shadow: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._loaded_shadow = None
        self._num_updates = 0
        self._last_global_step = 0
        self._using_ema = False

    @staticmethod
    def _parameters(module):
        return OrderedDict(module.named_parameters())

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def _schedule(self, update: int) -> float:
        step = max(0, int(update) - self.update_after_step - 1)
        if step == 0:
            return 0.0
        if self.use_warmup:
            value = 1.0 - (1.0 + step / self.inv_gamma) ** -self.power
        else:
            value = self.decay
        return max(self.min_decay, min(self.decay, value))

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        parameters = self._parameters(pl_module)
        if not parameters:
            raise RuntimeError("EMA requires model parameters")
        if self._loaded_shadow is None:
            self._shadow = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in parameters.items()
            )
        else:
            if set(parameters) != set(self._loaded_shadow):
                raise RuntimeError("EMA resume parameter set does not match the model")
            self._shadow = OrderedDict(
                (
                    name,
                    self._loaded_shadow[name].to(parameter.device, parameter.dtype),
                )
                for name, parameter in parameters.items()
            )
            self._loaded_shadow = None
        # Lightning can call this hook before restoring loop progress. Preserve
        # the checkpoint baseline so the first resumed optimizer step is seen
        # as exactly one EMA update instead of a jump from zero.
        self._last_global_step = max(int(trainer.global_step), self._num_updates)

    @torch.no_grad()
    def _update(self, pl_module: LightningModule) -> float:
        parameters = self._parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("model parameters changed after EMA initialization")
        self._num_updates += 1
        decay = self._schedule(self._num_updates)
        for name, parameter in parameters.items():
            self._shadow[name].mul_(decay).add_(parameter.detach(), alpha=1 - decay)
        return decay

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        global_step = int(trainer.global_step)
        updates = global_step - self._last_global_step
        if updates not in (0, 1):
            raise RuntimeError("EMA observed an invalid optimizer-step transition")
        if updates:
            current_decay = self._update(pl_module)
            self._last_global_step = global_step
            pl_module.log("log/ema_decay", current_decay, on_step=True, sync_dist=False)
            pl_module.log(
                "log/ema_num_updates",
                float(self._num_updates),
                on_step=True,
                sync_dist=False,
            )

    @torch.no_grad()
    def _swap(self, pl_module: LightningModule) -> None:
        parameters = self._parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("EMA swap parameter mismatch")
        for name, parameter in parameters.items():
            online = parameter.detach().clone()
            parameter.copy_(self._shadow[name])
            self._shadow[name].copy_(online)
        self._using_ema = not self._using_ema

    def on_validation_start(self, trainer, pl_module) -> None:
        if self.validate_with_ema and not self._using_ema:
            self._swap(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self.validate_with_ema and self._using_ema:
            self._swap(pl_module)

    def on_exception(self, trainer, pl_module, exception) -> None:
        if self._using_ema:
            self._swap(pl_module)

    def on_save_checkpoint(
        self, trainer, pl_module, checkpoint: dict[str, Any]
    ) -> None:
        parameters = self._parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("EMA checkpoint parameter mismatch")
        ema_state = parameters if self._using_ema else self._shadow
        if self._using_ema:
            for name, online in self._shadow.items():
                checkpoint["state_dict"][name] = online.detach()
        checkpoint["ema_state_dict"] = OrderedDict(
            (name, value.detach()) for name, value in ema_state.items()
        )
        checkpoint["ema_config"] = {
            "decay": self.decay,
            "use_warmup": self.use_warmup,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_decay": self.min_decay,
        }
        checkpoint["ema_num_updates"] = self._num_updates

    def on_load_checkpoint(
        self, trainer, pl_module, checkpoint: dict[str, Any]
    ) -> None:
        del trainer, pl_module
        state = checkpoint.get("ema_state_dict")
        if state is None:
            if int(checkpoint.get("global_step", 0)):
                raise RuntimeError("cannot resume EMA training without EMA state")
            return
        if checkpoint.get("ema_config") != {
            "decay": self.decay,
            "use_warmup": self.use_warmup,
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_decay": self.min_decay,
        }:
            raise RuntimeError("EMA checkpoint configuration mismatch")
        self._loaded_shadow = dict(state)
        self._num_updates = int(checkpoint.get("ema_num_updates", 0))
        checkpoint_global_step = int(checkpoint.get("global_step", 0))
        if self._num_updates != checkpoint_global_step:
            raise RuntimeError(
                "EMA update count must equal checkpoint global_step: "
                f"ema_num_updates={self._num_updates} "
                f"global_step={checkpoint_global_step}"
            )
        self._last_global_step = checkpoint_global_step
