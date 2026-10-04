import pytest
import torch

from astra_reversal.flow import generate as native_generate
from astra_reversal.reasoning_learning.guidance import (
    GuidanceConfig,
    endpoint_gradient,
    generate,
)


def test_zero_guidance_is_exact_native_euler():
    z = torch.randn(1, 10, 32, generator=torch.Generator().manual_seed(5))

    def velocity(x, t):
        return 0.2 * x + t

    expected = native_generate(velocity, z, steps=10).value
    actual, receipt = generate(
        velocity, z, torch.zeros_like(z), torch.ones_like(z), GuidanceConfig(strength=0)
    )
    assert torch.equal(expected, actual)
    assert receipt["gradient_evaluations"] == 0


@pytest.mark.parametrize("t", [0.05, 0.5, 1.0])
def test_vjp_matches_finite_difference_with_coupled_denoiser(t):
    x = torch.tensor([[[0.2, -0.3]]], dtype=torch.float64)
    matrix = torch.tensor([[0.1, 0.3], [-0.4, 0.2]], dtype=x.dtype)

    def velocity(a, _):
        return torch.tanh(a @ matrix)

    target = torch.tensor([[[0.4, 0.1]]], dtype=x.dtype)
    mask = torch.tensor([[[1.0, 0.25]]], dtype=x.dtype)
    _, gradient, _ = endpoint_gradient(velocity, x, t, target, mask)

    def energy(a):
        return 0.5 * (((a - t * velocity(a, t)) - target) * mask).square().sum()

    numerical = torch.zeros_like(x)
    for j in range(2):
        shift = torch.zeros_like(x)
        shift[0, 0, j] = 1e-6
        numerical[0, 0, j] = (energy(x + shift) - energy(x - shift)) / 2e-6
    torch.testing.assert_close(gradient, numerical, atol=1e-8, rtol=1e-6)


def test_native_time_sign_moves_towards_target_and_restricts_direct_edits():
    z = torch.zeros(1, 10, 32)
    target = torch.ones_like(z)
    mask = torch.zeros_like(z)
    mask[:, :5, 2] = 1
    result, _ = generate(
        lambda x, t: 0.1 * x, z, target, mask, GuidanceConfig(max_gradient_norm=100)
    )
    assert torch.all(result[:, :5, 2] > 0)
    assert torch.all(result[:, :5, 2] < 1)
    assert torch.count_nonzero(result) == 5


def test_vjp_does_not_accumulate_parameter_gradients_or_update_weights():
    module = torch.nn.Linear(2, 2)
    before = {k: v.clone() for k, v in module.state_dict().items()}
    generate(
        lambda x, t: module(x),
        torch.zeros(1, 3, 2),
        torch.ones(1, 3, 2),
        torch.ones(1, 3, 2),
    )
    assert all(p.grad is None for p in module.parameters())
    assert all(
        torch.equal(value, module.state_dict()[key]) for key, value in before.items()
    )


@pytest.mark.parametrize("strength", [-1, 11, float("nan")])
def test_strength_bounds(strength):
    with pytest.raises(ValueError):
        GuidanceConfig(strength=strength)


def test_nonfinite_gradient_fails_closed():
    with pytest.raises((ValueError, FloatingPointError)):
        generate(
            lambda x, t: x / 0,
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, 2),
            torch.ones(1, 2, 2),
        )


def test_gradient_clipping_bounds_a_single_step():
    x = torch.zeros(1, 3, 2)
    result, receipt = generate(
        lambda a, t: 0 * a,
        x,
        x + 1000,
        x + 1,
        GuidanceConfig(steps=1, max_gradient_norm=0.1),
    )
    assert result.norm() == pytest.approx(0.1)
    assert receipt["trace"][0]["clipped"]


def test_soft_mask_is_squared_in_energy():
    x = torch.zeros(1, 1, 2)
    _, gradient, energy = endpoint_gradient(lambda a, t: 0 * a, x, 0.5, x + 2, x + 0.5)
    torch.testing.assert_close(gradient, x - 0.5)
    assert energy == pytest.approx(1)
