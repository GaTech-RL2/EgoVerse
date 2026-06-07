"""``Loss`` — scalar-loss class consumed by algorithm orchestrators.

The class is separated from ``OuterStage`` so that:

* loss policy is data (a hydra block) rather than inheritance — adding
  / removing / reweighting loss terms doesn't require subclassing the
  outer stage,
* multiple algorithms with very different I/O contracts (DFoT's
  noise-MSE, H-Net's action-MSE + boundary regulariser) plug into the
  same algorithm-level call site,
* concrete subclasses (``HNetLoss``, ``DFoTLoss``) implement each
  algorithm's loss directly.

A ``Loss`` reads ``batch[pred_*]`` keys (written by ``OuterStage.decode``)
plus any training-time state attached to ``ctx`` by ``OuterStage.encode``
(sampled noise, noise levels, boundary probs, ...) and returns a single
scalar tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Loss(nn.Module):
    """Abstract base. Subclasses produce a scalar loss from ``(batch, ctx)``."""

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        raise NotImplementedError


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
        from egomimic.models.hnet.hnet import ratio_loss_from_aux

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
        # Structured-target outer stages (e.g. the 2D spatial-image + action
        # policy) compute their own multi-term loss and stash it here, since a
        # single (image, action) target can't flow through the scalar v-MSE
        # path below. No-op for every existing 1D/spatial stage.
        precomputed = getattr(ctx, "precomputed_loss", None)
        if precomputed is not None:
            return precomputed
        v_pred = batch["pred_v"]
        q_state = ctx.q_state
        per_token = self.diffusion.compute_loss(v_pred, q_state)
        return per_token.mean()
