import pytest
import torch

from egomimic.utils.schedulers import warmup_cosine_scheduler


def test_warmup_cosine_scheduler_reaches_peak_then_eta_min():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        max_steps=6,
        warmup_steps=2,
        warmup_start_factor=0.1,
        eta_min=0.2,
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[:3] == pytest.approx([0.1, 0.55, 1.0])
    assert rates[-1] == pytest.approx(0.2)
