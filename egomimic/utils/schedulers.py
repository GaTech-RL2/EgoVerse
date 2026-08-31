"""LR scheduler factories that need the optimizer + dynamic step counts.

Hydra-friendly: target one of these with ``_partial_: true`` and Lightning will
pass the optimizer at instantiation time.
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)


def constant_scheduler(optimizer: Optimizer) -> LRScheduler:
    """Constant LR (no warmup, no decay) — robomimic BC study default.

    robomimic's bc.json uses ``learning_rate.epoch_schedule=[]`` with a
    MultiStepLR, which with an empty milestone list never decays — i.e. a
    constant learning rate for the whole run. We replicate that exactly with a
    LambdaLR whose factor is always 1.0. Works with ``scheduler_interval="step"``
    (the EgoVerse pl_model default) — every step keeps base_lr unchanged.

    Ported from EgoVerse2 (branch hpt-hnet-pusher-nc3) for the BC-RNN paper-exact
    + tx configs, which target ``egomimic.utils.schedulers.constant_scheduler``.
    """
    return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


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


def piecewise_linear(step: int, schedule):
    """Piecewise-linear interpolation. ``schedule`` is a list of
    ``(step, value)`` knots (sorted by step). Returns a float.

    - For ``step <= schedule[0][0]``: returns ``schedule[0][1]``.
    - For ``step >= schedule[-1][0]``: returns ``schedule[-1][1]``.
    - Otherwise: linear interpolation between the two surrounding knots.

    Use the same shape for any "scalar driven by training step" — residual
    scale, KL beta, dropout rate, etc.
    """
    schedule = list(schedule)
    if not schedule:
        raise ValueError("schedule must be non-empty")
    if step <= schedule[0][0]:
        return float(schedule[0][1])
    if step >= schedule[-1][0]:
        return float(schedule[-1][1])
    for (s0, v0), (s1, v1) in zip(schedule[:-1], schedule[1:]):
        if s0 <= step <= s1:
            if s1 == s0:
                return float(v1)
            t = (step - s0) / (s1 - s0)
            return float(v0 + t * (v1 - v0))
    return float(schedule[-1][1])
