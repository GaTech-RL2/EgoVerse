from dataclasses import replace

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionAdapter
from astra_reversal.config import RunConfig
from astra_reversal.diagnostics import diagnose, require_diagnostics
from astra_reversal.flow import generate, invert

from .conftest import SyntheticEnvironment, SyntheticPolicy


def test_power_grid_has_same_endpoints_and_reverses_direction():
    initial = np.ones((1, 10, 32), dtype=np.float64)

    def velocity(x, t):
        return 2 * x + t

    kwargs = dict(steps=50, solver="rk4", time_power=3)
    result = invert(velocity, initial, **kwargs)
    exact = (initial + 0.25) * np.exp(2) - 0.75
    np.testing.assert_allclose(result.value, exact, rtol=2e-6)
    replay = generate(velocity, result.value, **kwargs)
    np.testing.assert_allclose(replay.value, initial, rtol=1e-6)
    np.testing.assert_allclose(result.grid, np.asarray(replay.grid)[::-1], atol=1e-15)
    assert result.grid[1] == (1 / 50) ** 3
    assert result.velocity_evaluations == 4 * 50


def test_nonuniform_diagnostics_gate_rejects_different_schedule(spec):
    policy = SyntheticPolicy()
    actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    report = diagnose(
        policy,
        actions,
        SyntheticEnvironment().observe(),
        "move cup",
        resolutions=(10,),
        solver="rk4",
        solver_options={"time_power": 3},
        action_atol=1e-4,
        noise_atol=1e-4,
        parity_atol=1e-4,
    )
    assert report["results"][0]["passed"]
    config = RunConfig()
    config = replace(config, flow=replace(config.flow, integrator="rk4", time_power=3))
    require_diagnostics(report, policy, actions, config)
    with pytest.raises(ValueError, match="No passing"):
        require_diagnostics(
            report,
            policy,
            actions,
            replace(config, flow=replace(config.flow, time_power=1)),
        )


def test_controller_reuses_latent_with_power_schedule(make_controller):
    config = RunConfig(method="inversion_only")
    config = replace(config, flow=replace(config.flow, integrator="rk4", time_power=3))
    controller, policy, _, recorder = make_controller(config=config)
    result = controller.run_episode(
        SyntheticEnvironment(stop_after=16),
        episode_id="power-schedule-test",
        instruction="move cup",
        seed=0,
    )
    assert result["success"] and result["agent_calls"] == 0
    assert len(policy.inverse_inputs) == 1
    generated = [
        row
        for row in recorder.events
        if row["kind"] == "flow" and row["role"] == "generation"
    ]
    assert len(generated) == 4
    assert len({row["condition_id"] for row in generated}) == 4
    for row in generated[1:]:
        np.testing.assert_array_equal(row["input"], generated[0]["input"])
