import pytest
import torch

from astra_reversal.reasoning_learning.guidance import GuidanceConfig, generate


def test_rtc_coefficient_matches_paper_under_native_time_convention():
    cfg = GuidanceConfig(schedule="rtc_pigdm", strength=1)
    for native_t in (0.1, 0.2, 0.5, 0.9):
        tau = 1 - native_t
        r2 = (1 - tau) ** 2 / (tau**2 + (1 - tau) ** 2)
        assert cfg.coefficient(native_t) == pytest.approx((1 - tau) / (tau * r2))
    assert cfg.coefficient(0) == cfg.coefficient(1) == 100
    off = GuidanceConfig(schedule="rtc_pigdm", strength=0)
    assert off.coefficient(0) == off.coefficient(1) == 0


def test_full_vjp_preserves_coupled_direction_removed_by_extra_projection():
    z = torch.tensor([[[0.0, 1.0]]])
    mask = torch.tensor([[[1.0, 0.0]]])

    def velocity(x, t):
        return torch.stack((x[..., 1], x[..., 1] * 0), -1)

    target = torch.tensor([[[1.0, 0.0]]])
    projected, _ = generate(velocity, z, target, mask, GuidanceConfig(steps=1))
    full, _ = generate(
        velocity, z, target, mask, GuidanceConfig(steps=1, project_gradient=False)
    )
    assert projected[0, 0, 1] == z[0, 0, 1]
    assert full[0, 0, 1] != z[0, 0, 1]
