from dataclasses import replace

import numpy as np
import pytest

from astra_reversal.config import METHODS, RunConfig
from astra_reversal.controller import ProgressMonitor

from .conftest import SyntheticEnvironment


def run(controller, env=None, episode="e1"):
    env = env or SyntheticEnvironment()
    result = controller.run_episode(
        env, episode_id=episode, instruction="move cup", seed=1
    )
    return env, result


def test_two_rate_reuse_with_fresh_conditions(make_controller):
    controller, policy, backend, recorder = make_controller()
    env, result = run(controller)
    assert result["success"] and result["actions"] == 26
    assert [r["observation_step"] for r in backend.requests] == [0, 20]
    assert len(policy.inverse_inputs) == 2
    assert len(policy.sample_inputs) == 6
    noise = [x[1] for x in policy.sample_inputs]
    for item in noise[1:4]:
        np.testing.assert_array_equal(item, noise[0])
    assert not np.array_equal(noise[4], noise[0])
    assert len({x[0].condition_id for x in policy.sample_inputs}) == 6
    np.testing.assert_allclose(env.actions[0], 0.2, atol=1e-6)
    assert not np.allclose(env.actions[5], env.actions[0])
    assert [
        x["step"].observation_step
        for x in recorder.events
        if x["kind"] == "control_step"
    ] == list(range(26))


def test_invalid_and_stale_responses_get_one_retry_then_fallback(make_controller):
    controller, policy, backend, recorder = make_controller(
        mutate=lambda response, req: response.update(observation_step=-1)
    )
    _, result = run(controller, SyntheticEnvironment(stop_after=10))
    assert result["success"]
    assert len(backend.requests) == 2
    assert not policy.inverse_inputs
    assert result["fallbacks"] == 2
    assert (
        len(
            [
                x
                for x in recorder.events
                if x["kind"] == "agent_response" and not x["accepted"]
            ]
        )
        == 2
    )
    assert backend.requests[1]["validation_error"] is not None
    assert all(x[0].prompt == "move cup" for x in policy.sample_inputs)


def test_retries_never_renew_subgoal_deadline_or_episode_budget(make_controller):
    controller, _, backend, recorder = make_controller()
    _, result = run(controller, SyntheticEnvironment(stop_after=211))
    timeouts = [x for x in recorder.events if x["kind"] == "subgoal_timeout"]
    assert [x["observation_step"] for x in timeouts] == [60, 120, 180]
    assert timeouts[-1]["exhausted"]
    assert result["actions"] == 211
    assert result["fallbacks"] > 0
    assert len(backend.requests) < 20


def test_episode_reset_clears_latents_history_completion_and_seen_ids(make_controller):
    controller, policy, backend, _ = make_controller()
    run(controller, SyntheticEnvironment(stop_after=11), episode="first")
    run(controller, SyntheticEnvironment(stop_after=11), episode="second")
    assert len(policy.inverse_inputs) == 2
    second = [r for r in backend.requests if r["episode_id"] == "second"]
    assert (
        len(second) == 1
        and second[0]["history"] == []
        and second[0]["active_subgoal"] is None
    )
    assert controller.subgoals["reach"].began == 0


def test_non_divisible_period_clears_superseded_execution(make_controller):
    config = RunConfig()
    config = replace(
        config,
        policy=replace(config.policy, execute_steps=7),
        agent=replace(config.agent, refresh_env_steps=10),
        controller=replace(config.controller, latent_max_age_env_steps=10),
    )
    controller, policy, backend, recorder = make_controller(config=config)
    run(controller, SyntheticEnvironment(stop_after=21))
    assert [r["observation_step"] for r in backend.requests] == [0, 10, 20]
    assert [
        x["observation_step"]
        for x in recorder.events
        if x["kind"] == "generated_actions"
    ] == [0, 7, 10, 17, 20]
    np.testing.assert_allclose(policy.sample_inputs[0][1], policy.sample_inputs[1][1])


def test_direct_astra_consumes_all_rows_once_without_repeating_or_synthesizing(
    make_controller,
):
    def mutate(response, request):
        response["action_chunk"] = np.repeat(
            np.arange(10, dtype=float)[:, None] / 10, 7, axis=1
        ).tolist()

    controller, policy, backend, _ = make_controller("direct_astra", mutate=mutate)
    env, _ = run(controller, SyntheticEnvironment(stop_after=20))
    np.testing.assert_allclose(
        np.asarray(env.actions)[:, 0], np.tile(np.arange(10) / 10, 2)
    )
    assert [x["observation_step"] for x in backend.requests] == [0, 10]
    assert not policy.sample_inputs and not policy.inverse_inputs


def test_progress_requires_stable_evidence(make_controller):
    controller, _, _, _ = make_controller()
    run(controller, SyntheticEnvironment(stop_after=1))
    proposal = controller.current_subgoal
    proposal.completion = {
        "type": "gripper_width",
        "parameters": {"target": 0.04, "tolerance": 0.001},
    }
    monitor = ProgressMonitor(positive_checks=2)
    observation = SyntheticEnvironment().observe()
    assert not monitor.check(observation, proposal)["complete"]
    assert monitor.check(observation, proposal)["complete"]
    observation["observation/state"][6] = 0.01
    assert monitor.check(observation, proposal)["streak"] == 0


def test_stage2_reference_full_tensor_and_conditioning_paths(make_controller):
    controller, policy, _, recorder = make_controller("stage2")
    env, _ = run(controller, SyntheticEnvironment(stop_after=11))
    references = [x["reference"] for x in recorder.events if x["kind"] == "reference"]
    assert len(references) == 1
    reference = references[0]
    assert reference["model_actions"].shape == (1, 10, 32)
    np.testing.assert_array_equal(
        policy.inverse_inputs[0][1], reference["model_actions"]
    )
    assert policy.inverse_inputs[0][0].prompt != "move cup"
    # Identical augmented inversion/forward reconstructs the raw reference;
    # there is no immediate improvement or second normalization step.
    expected = np.clip(reference["model_actions"][0, 0, :7] * 2 + 0.05, -1, 1)
    np.testing.assert_allclose(env.actions[0], expected, atol=1e-5)
    events = [
        e for x in recorder.events if x["kind"] == "augmentation" for e in x["events"]
    ]
    assert any(e["event"] == "annotation_expired" for e in events)


def test_known_noise_transfer_matches_redundant_raw_inverse_control(make_controller):
    a, pa, _, _ = make_controller("conditioning_transfer")
    b, pb, _, _ = make_controller("known_noise_transfer")
    ea, _ = run(a, SyntheticEnvironment(stop_after=16))
    eb, _ = run(b, SyntheticEnvironment(stop_after=16))
    np.testing.assert_allclose(ea.actions, eb.actions, atol=2e-6)
    assert len(pa.inverse_inputs) == 1
    assert not pb.inverse_inputs


@pytest.mark.parametrize("method", METHODS)
def test_all_method_paths_have_bounded_successful_execution(method, make_controller):
    controller, _, _, _ = make_controller(method)
    _, result = run(controller, SyntheticEnvironment(stop_after=21))
    assert result["success"] and result["actions"] == 21 and result["failure"] is None
    assert result["fallbacks"] == 0


def test_episode_budget_is_exact_even_without_success(make_controller):
    controller, _, _, _ = make_controller("policy_fresh")
    _, result = run(controller, SyntheticEnvironment(stop_after=999))
    assert result["actions"] == 520 and not result["success"]


def test_expired_latent_cannot_finish_a_chunk_after_its_deadline(make_controller):
    config = RunConfig()
    config = replace(
        config, controller=replace(config.controller, latent_max_age_env_steps=7)
    )
    controller, _, backend, _ = make_controller(config=config)
    _, result = run(controller, SyntheticEnvironment(stop_after=16))
    assert [r["observation_step"] for r in backend.requests] == [0, 7, 14]
    assert result["fallbacks"] == 0


def test_duplicate_plan_ids_cannot_keep_old_actions_alive(make_controller):
    controller, _, backend, recorder = make_controller(
        mutate=lambda response, req: response.update(plan_id="same")
    )
    _, result = run(controller)
    assert len(backend.requests) == 3
    assert result["invalid_responses"] == 2
    assert result["fallbacks"] == 2
    steps = [x["step"] for x in recorder.events if x["kind"] == "control_step"]
    assert steps[20].plan_id is None and steps[20].latent_id is None


def test_generation_failure_uses_fresh_full_task_fallback(make_controller, monkeypatch):
    controller, policy, _, _ = make_controller()
    sample = policy.sample
    calls = 0

    def fail_once(condition, noise, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("bad generated actions")
        return sample(condition, noise, **kwargs)

    monkeypatch.setattr(policy, "sample", fail_once)
    _, result = run(controller, SyntheticEnvironment(stop_after=6))
    assert result["success"] and result["fallbacks"] == 2
    assert all(condition.prompt == "move cup" for condition, _ in policy.sample_inputs)


def test_image_progress_throttles_calls_without_recounting_cached_evidence(
    make_controller,
):
    controller, _, _, _ = make_controller()
    run(controller, SyntheticEnvironment(stop_after=1))
    proposal = controller.current_subgoal
    proposal.completion = {
        "type": "image_progress",
        "parameters": {"criterion": "cup reached"},
    }
    calls = []

    def assessor(obs, criterion):
        calls.append(criterion)
        return {"positive": True, "evidence": "synthetic test evidence"}

    monitor = ProgressMonitor(positive_checks=2, assessor=assessor, assessor_period=20)
    observation = SyntheticEnvironment().observe()
    assert not monitor.check(observation, proposal, step=5)["complete"]
    assert not monitor.check(observation, proposal, step=10)["complete"]
    assert not monitor.check(observation, proposal, step=20)["complete"]
    assert monitor.check(observation, proposal, step=25)["complete"]
    assert len(calls) == 2


@pytest.mark.parametrize("target,candidates", [("reversal", 2), ("stage2", 3)])
def test_compute_baseline_matches_extra_solves_per_window(
    make_controller, target, candidates
):
    proposed, _, _, _ = make_controller(target)
    config = RunConfig(method="compute_matched")
    config = replace(
        config,
        controller=replace(config.controller, compute_matched_candidates=candidates),
    )
    baseline, _, _, _ = make_controller("compute_matched", config=config)
    _, a = run(proposed, SyntheticEnvironment(stop_after=26))
    _, b = run(baseline, SyntheticEnvironment(stop_after=26))
    assert a["velocity_evaluations"] == b["velocity_evaluations"]


def test_stall_assessments_trigger_bounded_early_recovery(make_controller):
    controller, _, backend, recorder = make_controller(
        mutate=lambda response, req: response.update(
            completion={
                "type": "image_progress",
                "parameters": {"criterion": "cup reached"},
            }
        )
    )
    controller.monitor.assessor = lambda obs, criterion: {
        "positive": False,
        "stalled": True,
    }
    _, result = run(controller, SyntheticEnvironment(stop_after=51))
    stalls = [event for event in recorder.events if event["kind"] == "subgoal_stalled"]
    assert [event["observation_step"] for event in stalls] == [5, 25, 45]
    assert stalls[-1]["exhausted"]
    assert result["fallbacks"] > 0
    assert len(backend.requests) <= 6
