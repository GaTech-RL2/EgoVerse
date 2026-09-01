"""Fail-closed exponential moving average for training and validation."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer


class EMACallback(Callback):
    """Track model parameters and run validation with their EMA values.

    The checkpoint keeps the ordinary online ``state_dict`` for exact resume
    and writes the averaged parameters separately as top-level
    ``ema_state_dict``. This matches the repository evaluation loader.
    """

    def __init__(self, decay: float = 0.9978, validate_with_ema: bool = True):
        super().__init__()
        self.decay = float(decay)
        self.validate_with_ema = bool(validate_with_ema)
        if not math.isfinite(self.decay) or not 0.0 < self.decay < 1.0:
            raise ValueError("EMA decay must be finite and strictly between 0 and 1")
        self._shadow: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._num_updates = 0
        self._last_global_step = 0
        self._using_ema = False
        self._loaded_shadow: dict[str, torch.Tensor] | None = None

    @property
    def num_updates(self) -> int:
        return self._num_updates

    def _named_parameters(
        self, pl_module: LightningModule
    ) -> OrderedDict[str, torch.nn.Parameter]:
        return OrderedDict(pl_module.named_parameters())

    def _initialize(self, pl_module: LightningModule) -> None:
        parameters = self._named_parameters(pl_module)
        if not parameters:
            raise RuntimeError("EMA requires at least one model parameter")
        if self._loaded_shadow is None:
            self._shadow = OrderedDict(
                (name, parameter.detach().clone())
                for name, parameter in parameters.items()
            )
            return
        if set(self._loaded_shadow) != set(parameters):
            missing = sorted(set(parameters) - set(self._loaded_shadow))
            unexpected = sorted(set(self._loaded_shadow) - set(parameters))
            raise RuntimeError(
                "EMA checkpoint parameter mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        self._shadow = OrderedDict(
            (
                name,
                self._loaded_shadow[name].to(
                    device=parameter.device, dtype=parameter.dtype
                ),
            )
            for name, parameter in parameters.items()
        )
        self._loaded_shadow = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._initialize(pl_module)
        self._last_global_step = int(trainer.global_step)

    @torch.no_grad()
    def _update(self, pl_module: LightningModule) -> None:
        parameters = self._named_parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("Model parameter set changed after EMA initialization")
        one_minus_decay = 1.0 - self.decay
        for name, parameter in parameters.items():
            shadow = self._shadow[name]
            if not torch.is_floating_point(parameter):
                continue
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=one_minus_decay)
        self._num_updates += 1

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        global_step = int(trainer.global_step)
        if global_step < self._last_global_step:
            raise RuntimeError("Trainer global_step moved backwards while tracking EMA")
        updates = global_step - self._last_global_step
        if updates > 1:
            raise RuntimeError(
                "EMA observed more than one optimizer update in one train batch"
            )
        if updates == 1:
            self._update(pl_module)
            self._last_global_step = global_step
            pl_module.log(
                "log/unite_ema_decay",
                self.decay,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            pl_module.log(
                "log/unite_ema_num_updates",
                float(self._num_updates),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

    @torch.no_grad()
    def _swap(self, pl_module: LightningModule) -> None:
        parameters = self._named_parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("EMA swap parameter mismatch")
        for name, parameter in parameters.items():
            temporary = parameter.detach().clone()
            parameter.copy_(self._shadow[name])
            self._shadow[name].copy_(temporary)
        self._using_ema = not self._using_ema

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.validate_with_ema and not self._using_ema:
            self._swap(pl_module)

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.validate_with_ema and self._using_ema:
            self._swap(pl_module)

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        if self._using_ema:
            self._swap(pl_module)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        parameters = self._named_parameters(pl_module)
        if set(parameters) != set(self._shadow):
            raise RuntimeError("EMA checkpoint parameter mismatch")

        if self._using_ema:
            ema_state = OrderedDict(
                (name, parameter.detach()) for name, parameter in parameters.items()
            )
            online_state = self._shadow
            for name, value in online_state.items():
                if name not in checkpoint["state_dict"]:
                    raise RuntimeError(f"Online checkpoint is missing EMA key {name!r}")
                checkpoint["state_dict"][name] = value.detach()
        else:
            ema_state = self._shadow

        checkpoint["ema_state_dict"] = OrderedDict(
            (name, value.detach()) for name, value in ema_state.items()
        )
        checkpoint["ema_decay"] = self.decay
        checkpoint["ema_num_updates"] = self._num_updates
        checkpoint["ema_validate_with_ema"] = self.validate_with_ema

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        stored_decay = checkpoint.get("ema_decay")
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None:
            if int(checkpoint.get("global_step", 0)) > 0:
                raise RuntimeError("Cannot resume EMA training from a checkpoint without EMA")
            return
        if stored_decay is None or not math.isclose(
            float(stored_decay), self.decay, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise RuntimeError(
                f"EMA decay mismatch: checkpoint={stored_decay} configured={self.decay}"
            )
        self._loaded_shadow = dict(ema_state)
        self._num_updates = int(checkpoint.get("ema_num_updates", 0))
