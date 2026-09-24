import numpy as np
import pytest

from astra_reversal.adaptive_probe import integrate_reference


@pytest.mark.parametrize("method", ["RK45", "DOP853"])
def test_adaptive_reference_matches_analytic_flow_and_reverse(method):
    initial = np.linspace(-1, 1, 32).reshape(1, 2, 16)

    # Non-autonomous affine field; both directions have a known solution.
    def velocity(x, t):
        return 2 * x + t

    kwargs = dict(rtol=1e-9, atol=1e-11, method=method, save_trace=True)
    forward = integrate_reference(velocity, initial, start=0, end=1, **kwargs)
    exact = (initial + 0.25) * np.exp(2) - 0.75
    np.testing.assert_allclose(forward.value, exact, rtol=2e-9, atol=2e-9)
    reverse = integrate_reference(velocity, forward.value, start=1, end=0, **kwargs)
    np.testing.assert_allclose(reverse.value, initial, rtol=2e-9, atol=2e-9)
    assert forward.grid[0] == 0 and forward.grid[-1] == 1
    assert reverse.grid[0] == 1 and reverse.grid[-1] == 0
    assert forward.velocity_evaluations > len(forward.grid)
    assert len(reverse.states) == len(reverse.grid)


def test_adaptive_reference_bounds_gpu_work():
    with pytest.raises(RuntimeError, match="evaluation limit"):
        integrate_reference(
            lambda x, t: x,
            np.ones((1, 1, 1)),
            start=1,
            end=0,
            rtol=1e-6,
            atol=1e-8,
            max_evaluations=1,
        )
