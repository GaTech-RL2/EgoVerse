"""Checkpoint-specific numerical acceptance gates; synthetic tests are separate."""

import time

import numpy as np

from .flow import error_metrics
from .records import digest, to_numpy


def diagnose(
    policy,
    actions,
    observation,
    instruction,
    *,
    seed=0,
    resolutions=(10, 20, 50),
    solver="euler",
    action_atol,
    noise_atol,
    parity_atol,
    controller_actions=None,
    solver_options=None,
):
    solver_options = solver_options or {}
    for value in (action_atol, noise_atol, parity_atol):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "Declare positive finite tolerances before running diagnostics"
            )
    condition = policy.prepare(observation, digest(observation), instruction)
    noise = policy.noise(np.random.default_rng(seed))
    rows = []
    for steps in resolutions:
        began = time.perf_counter()
        original = policy.sample(
            condition, noise, steps=steps, solver=solver, **solver_options
        )
        endpoint = (
            original.value
            if controller_actions is None
            else policy.tensor(
                actions.encode(
                    controller_actions, {**observation, "prompt": instruction}
                )
            )
        )
        inverse = policy.invert(
            condition, endpoint, steps=steps, solver=solver, **solver_options
        )
        replay = policy.sample(
            condition, inverse.value, steps=steps, solver=solver, **solver_options
        )
        full_error = error_metrics(endpoint, replay.value)
        decoded_original = actions.output_transform(
            {
                "actions": to_numpy(endpoint)[0].copy(),
                "state": to_numpy(condition.state)[0].copy(),
            }
        )["actions"]
        decoded_replay = actions.output_transform(
            {
                "actions": to_numpy(replay.value)[0].copy(),
                "state": to_numpy(condition.state)[0].copy(),
            }
        )["actions"]
        # Compare Euler against the actual upstream Euler sampler, even when the
        # reconstruction experiment uses Heun.
        euler = (
            original
            if solver == "euler" and not solver_options
            else policy.sample(condition, noise, steps=steps, solver="euler")
        )
        decoded_euler = actions.output_transform(
            {
                "actions": to_numpy(euler.value)[0].copy(),
                "state": to_numpy(condition.state)[0].copy(),
            }
        )["actions"]
        upstream = policy.reference_actions(condition, noise, steps=steps)
        parity = error_metrics(upstream, decoded_euler)
        known_inverse = (
            inverse
            if controller_actions is None
            else policy.invert(
                condition, original.value, steps=steps, solver=solver, **solver_options
            )
        )
        recovery = error_metrics(noise, known_inverse.value)
        controller_error = error_metrics(decoded_original, decoded_replay)
        row = {
            "steps": steps,
            "full_internal_reconstruction": full_error,
            "controller_reconstruction": controller_error,
            "known_noise_recovery": recovery,
            "known_noise_action_channels": error_metrics(
                to_numpy(noise)[..., :7], to_numpy(known_inverse.value)[..., :7]
            ),
            "known_noise_padding_channels": error_metrics(
                to_numpy(noise)[..., 7:], to_numpy(known_inverse.value)[..., 7:]
            ),
            "reference_sampler_parity": parity,
            "velocity_evaluations": (
                original.velocity_evaluations
                + inverse.velocity_evaluations
                + replay.velocity_evaluations
                + steps
                + (euler.velocity_evaluations if euler is not original else 0)
                + (
                    known_inverse.velocity_evaluations
                    if known_inverse is not inverse
                    else 0
                )
            ),
            "latency_seconds": time.perf_counter() - began,
            "passed": full_error["max_abs"] <= action_atol
            and controller_error["max_abs"] <= action_atol
            and recovery["max_abs"] <= noise_atol
            and parity["max_abs"] <= parity_atol,
        }
        rows.append(row)
    return {
        "schema_version": "1.0",
        "solver": solver,
        "solver_options": solver_options,
        "checkpoint": policy.metadata,
        "action_spec_id": actions.spec.action_spec_id,
        "observation_id": condition.observation_id,
        "condition_id": condition.condition_id,
        "seed": seed,
        "tolerances": {
            "action_atol": action_atol,
            "noise_atol": noise_atol,
            "parity_atol": parity_atol,
        },
        "proposal_source": "supplied_controller_actions"
        if controller_actions is not None
        else "policy_generated_numerical_diagnostic",
        "results": rows,
        "experiments_run": False,
        "interpretation": "Reconstruction and reference parity only; no control-success or generalization claim",
    }


def require_diagnostics(report, policy, actions, config):
    if (
        report["checkpoint"] != policy.metadata
        or report["action_spec_id"] != actions.spec.action_spec_id
    ):
        raise ValueError(
            "Diagnostics belong to a different checkpoint, normalization, or controller"
        )
    if config.flow.inversion_steps != config.flow.generation_steps:
        raise ValueError(
            "This diagnostic gate covers equal inverse/forward resolutions only"
        )
    row = next(
        (r for r in report["results"] if r["steps"] == config.flow.inversion_steps),
        None,
    )
    if (
        report["solver"] != config.flow.integrator
        or report.get("solver_options", {}) != config.flow.solver_options
        or row is None
        or not row["passed"]
    ):
        raise ValueError(
            "No passing checkpoint diagnostic for the configured solver/resolution"
        )
