"""Light wrappers for hydra-instantiable LR schedulers.

These helpers exist because torch's ``SequentialLR`` expects already-built
sub-schedulers (each holding a reference to the optimizer), so it doesn't
compose cleanly with hydra ``_partial_: true`` on the sub-schedulers. Wrap the
construction in a function and ``_partial_: true`` on the function instead.
"""

from __future__ import annotations

import torch


def warmup_then_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 0.0,
    warmup_start_factor: float = 1.0e-3,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing.

    Args:
        optimizer: optimizer to schedule.
        warmup_epochs: number of epochs to linearly warm up from
            ``warmup_start_factor * peak_lr`` to ``peak_lr``.
        total_epochs: total epochs over which the schedule is defined. The
            cosine phase runs from ``warmup_epochs`` to ``total_epochs``.
        eta_min: minimum LR at the end of cosine annealing.
        warmup_start_factor: initial multiplier on the peak LR at epoch 0.

    Returns:
        torch.optim.lr_scheduler.SequentialLR
    """
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be > 0")
    if total_epochs <= warmup_epochs:
        raise ValueError("total_epochs must be > warmup_epochs")
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def warmup_then_cosine_min_lr(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_rate: float = 0.1,
    last_epoch: int = -1,
    **_unused,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup -> cosine decay to ``min_lr_rate * peak_lr``, CONSTANT at
    the floor beyond ``num_training_steps`` (never re-oscillates).

    Implemented as a ``LambdaLR`` deliberately: lambdas are NOT serialized in
    the scheduler ``state_dict``, so resuming a checkpoint keeps THIS schedule
    with the restored step position. (``CosineAnnealingLR``-based schedules
    restore ``T_max``/``eta_min`` from the checkpoint on ``load_state_dict``,
    silently reverting a mid-run schedule change.)

    ``**_unused`` swallows leftover kwargs merged in from a parent hydra
    config's scheduler node (e.g. ``num_cycles``).
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        if step >= num_training_steps:
            return min_lr_rate
        progress = (step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cos

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )
