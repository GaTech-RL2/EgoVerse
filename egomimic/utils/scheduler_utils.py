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
