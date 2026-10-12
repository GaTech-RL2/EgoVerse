"""Exponential moving average of a model's trainable weights."""

from __future__ import annotations

import torch


class WeightEMA:
    """EMA shadow of ``named_params`` with RDT's decay warmup.

    Decay after ``n`` updates is ``1 - (1 + (n - 1) / inv_gamma) ** -power``,
    capped at ``max_decay`` (0 for the first two updates), as in RDT's
    ``models/ema_model.py``. With power 0.75 it passes 0.999 near 10k updates
    and reaches 0.9999 near 215k, so the average window grows with the run.
    """

    def __init__(
        self,
        named_params,
        power: float = 0.75,
        max_decay: float = 0.9999,
        inv_gamma: float = 1.0,
    ):
        named_params = list(named_params)
        self.names = [name for name, _ in named_params]
        self.params = [param for _, param in named_params]
        self.shadow = [param.detach().float().clone() for param in self.params]
        self.power = power
        self.max_decay = max_decay
        self.inv_gamma = inv_gamma
        self.num_updates = 0
        self._backup = None

    def decay(self) -> float:
        step = self.num_updates - 1
        if step <= 0:
            return 0.0
        return min(1 - (1 + step / self.inv_gamma) ** -self.power, self.max_decay)

    @torch.no_grad()
    def update(self) -> float:
        decay = self.decay()
        current = [p.detach().to(s.dtype) for p, s in zip(self.params, self.shadow)]
        torch._foreach_lerp_(self.shadow, current, 1.0 - decay)
        self.num_updates += 1
        return decay

    @property
    def swapped(self) -> bool:
        return self._backup is not None

    @torch.no_grad()
    def swap_in(self) -> None:
        """Load the EMA weights into the model, keeping the raw ones aside."""
        if self.swapped:
            return
        # On CPU: a second GPU copy of a 1B model can OOM validation.
        self._backup = [p.detach().to("cpu", copy=True) for p in self.params]
        for param, shadow in zip(self.params, self.shadow):
            param.copy_(shadow)

    @torch.no_grad()
    def swap_out(self) -> None:
        if not self.swapped:
            return
        for param, raw in zip(self.params, self._backup):
            param.copy_(raw)
        self._backup = None

    def raw_by_id(self) -> dict:
        """``id(param) -> raw weights`` while swapped in, else empty."""
        if not self.swapped:
            return {}
        return {id(p): raw for p, raw in zip(self.params, self._backup)}

    def state_dict(self) -> dict:
        return {
            "shadow": dict(zip(self.names, self.shadow)),
            "num_updates": self.num_updates,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        missing = [name for name in self.names if name not in state["shadow"]]
        if missing:
            raise KeyError(
                f"EMA state is missing {len(missing)} weight(s), e.g. {missing[:5]}"
            )
        for name, shadow in zip(self.names, self.shadow):
            shadow.copy_(state["shadow"][name])
        self.num_updates = int(state["num_updates"])
