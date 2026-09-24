import numpy as np
import pytest

from astra_reversal.flow import error_metrics, generate, integrate, invert, mix_noise


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_constant_field_directions_and_full_channels(solver):
    x = np.arange(2 * 10 * 32, dtype=np.float64).reshape(2, 10, 32) / 100
    original = x.copy()
    visited = []

    def velocity(state, t):
        visited.append(t)
        return np.ones_like(state) * 3

    result = invert(velocity, x, steps=10, solver=solver, save_trace=True)
    np.testing.assert_allclose(result.value, x + 3)
    assert visited[0] == 0
    assert len(result.grid) == 11
    assert result.grid[-1] == 1
    assert len(result.states) == 11
    replay = generate(velocity, result.value, steps=10, solver=solver)
    np.testing.assert_allclose(replay.value, original, atol=1e-12)
    np.testing.assert_array_equal(x, original)
    assert result.velocity_evaluations == (10 if solver == "euler" else 20)


def test_linear_round_trip_converges_but_euler_is_not_exact():
    noise = np.ones((1, 10, 32), np.float64)
    errors = []
    for steps in (10, 20, 50):
        actions = generate(lambda x, t: x, noise, steps=steps).value
        recovered = invert(lambda x, t: x, actions, steps=steps).value
        errors.append(error_metrics(noise, recovered)["rmse"])
    assert 0 < errors[2] < errors[1] < errors[0]
    assert errors[2] < errors[0] / 3


def test_time_dependent_field_uses_correct_endpoints():
    times = []

    def velocity(x, t):
        times.append(t)
        return np.full_like(x, t)

    x = np.zeros((1, 1, 8), np.float64)
    np.testing.assert_allclose(invert(velocity, x, steps=10).value, 0.45)
    assert times == [j / 10 for j in range(10)]
    times.clear()
    np.testing.assert_allclose(generate(velocity, x, steps=10).value, -0.55)
    assert times == [1 - j / 10 for j in range(10)]


@pytest.mark.parametrize("steps", [0, -1, 2.5, True])
def test_invalid_solver_resolution(steps):
    with pytest.raises(ValueError):
        invert(lambda x, t: x, np.ones((1, 10, 32)), steps=steps)


def test_invalid_tensor_and_velocity():
    with pytest.raises(ValueError):
        invert(lambda x, t: x, np.ones((10, 7)), steps=10)
    with pytest.raises(ValueError):
        invert(lambda x, t: x[..., :7], np.ones((1, 10, 32)), steps=10)
    with pytest.raises(FloatingPointError):
        invert(lambda x, t: np.full_like(x, np.nan), np.ones((1, 10, 32)), steps=10)
    with pytest.raises(ValueError):
        integrate(lambda x, t: x, np.ones((1, 10, 32)), start=0, end=2, steps=10)


def test_partial_solver_requires_explicit_start_and_noise_mix_endpoints():
    x = np.ones((1, 10, 32))
    partial = integrate(lambda x, t: np.ones_like(x), x, start=0, end=0.4, steps=4)
    replay = integrate(
        lambda x, t: np.ones_like(x), partial.value, start=0.4, end=0, steps=4
    )
    np.testing.assert_allclose(replay.value, x)
    np.testing.assert_array_equal(mix_noise(x, x * 3, 0), x)
    np.testing.assert_array_equal(mix_noise(x, x * 3, 1), x * 3)


def test_native_torch_integration_preserves_device_and_float64_precision():
    torch = pytest.importorskip("torch")
    from astra_reversal.records import to_numpy

    x = torch.ones((1, 10, 32), dtype=torch.float64)
    inverse = invert(lambda x, t: x * 0.125, x, steps=20, solver="heun")
    forward = generate(lambda x, t: x * 0.125, inverse.value, steps=20, solver="heun")
    assert forward.value.device == x.device and forward.value.dtype == torch.float64
    assert to_numpy(forward.value).dtype == np.float64
    np.testing.assert_allclose(to_numpy(forward.value), 1, atol=1e-8, rtol=0)
