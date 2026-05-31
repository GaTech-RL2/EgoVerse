"""LR scheduler factories that need the optimizer + dynamic step counts.

Hydra-friendly: target one of these with ``_partial_: true`` and Lightning will
pass the optimizer at instantiation time.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def warmup_cosine_scheduler(
    optimizer: Optimizer,
    max_steps: int,
    warmup_steps: int = 500,
    warmup_start_factor: float = 0.01,
    eta_min: float = 2e-5,
) -> LRScheduler:
    """Linear warmup → CosineAnnealingLR.

    Args:
        optimizer: bound at Lightning instantiation time.
        max_steps: total number of optimizer steps for the whole training run.
            Cosine annealing spans the post-warmup window
            ``max_steps - warmup_steps``.
        warmup_steps: number of warmup steps. LR linearly goes from
            ``base_lr * warmup_start_factor`` to ``base_lr``.
        warmup_start_factor: fraction of base_lr to start the warmup at.
        eta_min: minimum LR at the end of cosine annealing.

    Stepping convention: this expects ``scheduler_interval="step"`` (the
    default in the EgoVerse pl_model wrapper).
    """
    warmup_steps = max(1, int(warmup_steps))
    max_steps = max(warmup_steps + 1, int(max_steps))
    warmup = LinearLR(
        optimizer,
        start_factor=float(warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=float(eta_min),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
