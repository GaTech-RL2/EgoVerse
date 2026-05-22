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


class HNetLoss(Loss):
    """Action-MSE + per-chunker ratio-loss regulariser.

    Reads:
      - ``batch["pred_action"]`` (per-token predicted actions; written by
        ``HNetOuterStage.decode``)
      - ``batch["actions"]`` (per-token GT actions)
      - ``ctx.aux`` (list of chunker-stage auxiliaries; each chunker
        contributes a ``ratio_loss`` * weight term)

    The per-chunker weight lives inside each ChunkerStage's aux already
    (see ``ratio_loss_from_aux``); this class just sums them onto the
    action MSE.
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        from egomimic.models.hnet_nets.hnet import ratio_loss_from_aux

        pred = batch["pred_action"]
        target = batch["actions"]
        action_loss = torch.mean((pred - target) ** 2)

        # Sum per-chunker boundary regularisers if any.
        aux = getattr(ctx, "aux", None) or []
        ratio_loss = (
            ratio_loss_from_aux(aux, device=action_loss.device)
            if aux
            else torch.zeros((), device=action_loss.device, dtype=action_loss.dtype)
        )
        # Stash the per-term breakdown on ctx so the algo class's logging
        # path can read them without recomputing.
        ctx.action_loss = action_loss
        ctx.ratio_loss = ratio_loss
        return action_loss + ratio_loss


class DFoTLoss(Loss):
    """DFoT epsilon-MSE with sigmoid (SNR-style) weighting.

    Reads:
      - ``batch["pred_v"]``: v-prediction emitted by the DFoT backbone
        (written by ``DFoTOuterStage.decode``).
      - ``ctx.q_state``: dict produced by ``diffusion.q_sample`` during
        ``DFoTOuterStage.encode``. Carries ``x_t``, ``noise``, ``alpha_t``,
        ``sigma_t``, ``logsnr``.

    The actual math lives in ``diffusion.compute_loss``; this class is the
    interface that fits the OuterStage-orchestrated training loop and
    reduces the per-token loss to a scalar.

    ``diffusion`` is the same instance held by the outer stage — it has
    no learnable params (in continuous mode) so this is a free reference,
    not a duplicate submodule.
    """

    def __init__(self, diffusion: nn.Module):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        v_pred = batch["pred_v"]
        q_state = ctx.q_state
        per_token = self.diffusion.compute_loss(v_pred, q_state)
        return per_token.mean()


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
