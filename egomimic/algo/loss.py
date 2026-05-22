"""``Loss`` — scalar-loss class consumed by algorithm orchestrators.

The class is separated from ``OuterStage`` so that:

* loss policy is data (a hydra block) rather than inheritance — adding
  / removing / reweighting loss terms doesn't require subclassing the
  outer stage,
* multiple algorithms with very different I/O contracts (DFoT's
  noise-MSE, H-Net's action-MSE + boundary regulariser) plug into the
  same algorithm-level call site,
* ``CompositeLoss`` lets a config list arbitrary weighted terms
  without writing a new class per combination.

A ``Loss`` reads ``batch[pred_*]`` keys (written by ``OuterStage.decode``)
plus any training-time state attached to ``ctx`` by ``OuterStage.encode``
(sampled noise, noise levels, boundary probs, ...) and returns a single
scalar tensor.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


class Loss(nn.Module):
    """Abstract base. Subclasses produce a scalar loss from ``(batch, ctx)``."""

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        raise NotImplementedError


class CompositeLoss(Loss):
    """Sum of weighted ``Loss`` terms.

    Each term is itself a ``Loss``; the composite calls all of them and
    sums their outputs (already weighted by each term's own scalar
    ``weight`` attribute if set, else weight=1.0). A logging dict is
    attached as ``ctx.loss_terms`` so each term's contribution can be
    tracked by the training loop.

    Example yaml::

        loss:
          _target_: egomimic.algo.loss.CompositeLoss
          terms:
            - { _target_: egomimic.algo.loss.MSELoss,
                pred_key: pred_action, target_key: actions, weight: 1.0 }
            - { _target_: egomimic.algo.loss.MSELoss,
                pred_key: pred_front_img_1, target_key: front_img_1,
                weight: 1.0 }
    """

    def __init__(self, terms: Iterable[Loss]):
        super().__init__()
        self.terms = nn.ModuleList(list(terms))

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        # Initialise to a 0-tensor on the right device; populated as the
        # first term's loss accumulates.
        total: Optional[torch.Tensor] = None
        per_term: List[float] = []
        for term in self.terms:
            t = term(batch, ctx)
            per_term.append(float(t.detach().cpu().item()))
            total = t if total is None else total + t
        if total is None:
            raise RuntimeError("CompositeLoss has no terms")
        # Attach per-term breakdown to ctx for logging by the trainer.
        if hasattr(ctx, "loss_terms"):
            ctx.loss_terms.extend(per_term)
        else:
            setattr(ctx, "loss_terms", list(per_term))
        return total


class MSELoss(Loss):
    """Per-modality MSE between ``batch[pred_key]`` and ``batch[target_key]``.

    A convenience concrete term for ``CompositeLoss`` — covers the common
    "predicted action vs. ground-truth action", "predicted image vs. true
    image" cases without needing a custom Loss subclass.

    The optional ``weight`` field scales the term inside ``CompositeLoss``.
    Reduction is ``mean`` over all elements (loss is a scalar tensor).
    """

    def __init__(self, pred_key: str, target_key: str, weight: float = 1.0):
        super().__init__()
        self.pred_key = pred_key
        self.target_key = target_key
        self.weight = float(weight)

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        pred = batch[self.pred_key]
        target = batch[self.target_key]
        return self.weight * torch.mean((pred - target) ** 2)
