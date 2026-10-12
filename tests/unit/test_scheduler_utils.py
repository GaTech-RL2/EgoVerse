"""warmup_stable_decay LR shape, per-group scaling, and resume."""

import pytest
import torch

from egomimic.utils.scheduler_utils import warmup_stable_decay


def _opt(lrs=(1e-4, 1e-5)):
    return torch.optim.AdamW(
        [{"params": [torch.zeros(1, requires_grad=True)], "lr": lr} for lr in lrs]
    )


def _lrs_at(sched, opt, steps):
    out, step = {}, 0
    for target in sorted(steps):
        while step < target:
            opt.step()
            sched.step()
            step += 1
        out[target] = [g["lr"] for g in opt.param_groups]
    return out


def test_wsd_shape_and_group_scale():
    opt = _opt()
    sched = warmup_stable_decay(opt, warmup=10, stable=90, decay=100)
    lrs = _lrs_at(sched, opt, [0, 5, 10, 50, 100, 150, 200, 250])
    assert lrs[0][0] == pytest.approx(1e-7)
    assert lrs[5][0] == pytest.approx(1e-7 + (1e-4 - 1e-7) * 0.5)
    assert lrs[10][0] == pytest.approx(1e-4)
    assert lrs[50][0] == pytest.approx(1e-4)
    assert lrs[100][0] == pytest.approx(1e-4)
    assert lrs[150][0] == pytest.approx(5e-5)
    assert lrs[200][0] == pytest.approx(0.0)
    assert lrs[250][0] == pytest.approx(0.0)
    for v in lrs.values():
        assert v[1] == pytest.approx(v[0] * 0.1)


def test_wsd_min_factor_floor():
    opt = _opt((1.0,))
    sched = warmup_stable_decay(opt, warmup=0, stable=2, decay=4, min_factor=0.1)
    lrs = _lrs_at(sched, opt, [0, 2, 4, 6, 10])
    assert [lrs[s][0] for s in (0, 2, 4, 6, 10)] == pytest.approx(
        [1.0, 1.0, 0.55, 0.1, 0.1]
    )


def test_wsd_resume_from_state_dict():
    opt = _opt()
    sched = warmup_stable_decay(opt, warmup=10, stable=20, decay=40)
    _lrs_at(sched, opt, [45])
    state = sched.state_dict()

    opt2 = _opt()
    sched2 = warmup_stable_decay(opt2, warmup=10, stable=20, decay=40)
    sched2.load_state_dict(state)
    opt2.step()
    sched2.step()
    opt.step()
    sched.step()
    assert [g["lr"] for g in opt2.param_groups] == pytest.approx(
        [g["lr"] for g in opt.param_groups]
    )
