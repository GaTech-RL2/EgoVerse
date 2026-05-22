"""``OuterStage`` — the outermost stage in an algorithm's stage tree.

Owns the boundary between raw batch data and the trunk's tensor space, plus
the inverse on the way out. Subclasses implement algorithm-specific I/O
contracts (DFoT noise sampling at encode time, H-Net per-modality token
construction, joint obs+action fusion, etc.).

Convention: an ``OuterStage`` instance HAS an ``inner_stage`` (the trunk) and
calls it internally. The algorithm class holds the ``OuterStage`` and calls
it once per training step / inference step.

This file deliberately ships ONLY the abstract base; concrete subclasses
(``HNetOuterStage``, ``DFoTOuterStage``) live in their respective algo
modules.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class OuterStage(nn.Module):
    """Abstract outermost stage.

    Subclasses must implement ``encode`` (raw batch -> trunk-input tensor)
    and ``decode`` (trunk-output tensor -> per-modality prediction keys on
    the batch dict). The default ``forward`` orchestrates::

        x = self.encode(batch, ctx)
        x = self.inner_stage(x, ctx) if self.inner_stage is not None else x
        self.decode(x, batch, ctx)
        return x

    All three of encode / decode / forward receive ``(batch, ctx)``.

    * ``batch`` is the batch dict — modality keys (``front_img_1``,
      ``state_agent_obj``, ``actions``, ...) on input; subclasses write
      prediction keys (``pred_action``, ``pred_front_img_1``, ...) during
      ``decode``.
    * ``ctx`` is the algorithm-specific context object (``HNetContext`` for
      H-Net, equivalent for DFoT). It carries packing info (``cu_seqlens``,
      ``max_seqlen``), inference state (``inference_params``), and any
      training-time state subclasses want to thread to the loss (sampled
      noise, noise levels, boundary probs, ...). Subclasses are free to
      attach attributes to ``ctx`` as needed; the loss reads them later.
    """

    def __init__(self, inner_stage: Optional[nn.Module] = None):
        super().__init__()
        # ``inner_stage`` is typically a ``_BaseStage`` subclass (the trunk),
        # but the OuterStage doesn't constrain its type — subclasses may
        # use a plain ``nn.Module`` (e.g. DFoT's Isotropic backbone) or a
        # full H-Net stage tree.
        self.inner_stage = inner_stage

    # -------------------------------------------------------------------
    # Abstract methods. Subclasses implement these.
    # -------------------------------------------------------------------

    def encode(self, batch: dict, ctx) -> torch.Tensor:
        """Read raw modality keys from ``batch``, return the trunk-input
        tensor ``x``. May also sample noise / record training-time state
        on ``ctx`` for the loss to use later (DFoT does this).
        """
        raise NotImplementedError

    def decode(self, x: torch.Tensor, batch: dict, ctx) -> None:
        """Read the trunk-output tensor ``x``, write per-modality
        prediction keys onto ``batch`` (e.g. ``batch["pred_action"]``).
        """
        raise NotImplementedError

    # -------------------------------------------------------------------
    # Default orchestration. Subclasses usually don't override this.
    # -------------------------------------------------------------------

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        """Default flow: encode -> inner_stage -> decode. Returns the
        post-trunk tensor (useful when the algo needs it for a loss that
        operates pre-decode, e.g. DFoT's v-prediction in d_model)."""
        x = self.encode(batch, ctx)
        if self.inner_stage is not None:
            x = self.inner_stage(x, ctx)
        self.decode(x, batch, ctx)
        return x
