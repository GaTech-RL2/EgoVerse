"""Episode freshness, failure holds and budget accounting for online steering."""

import copy
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.astra_client import ClientError
from astra_reversal.interpolation_agent import InterpolationClient, summarize_calls
from astra_reversal.interpolation_search import (
    InterpolationSearch,
    OnlineInterpolation,
    arm_summary,
    load_protocol,
)
from astra_reversal.osmo.interpolation import assignment
from astra_reversal.records import Recorder, digest

from .conftest import SyntheticPolicy


def observation(value):
    return {
        "observation/image": np.full((8, 8, 3), value % 256, dtype=np.uint8),
        "observation/wrist_image": np.full(
            (8, 8, 3), (value + 1) % 256, dtype=np.uint8
        ),
        "observation/state": np.full(8, value, dtype=np.float32),
    }


class ObservedClient:
    def __init__(self, fail=()):
        self.requests = []
        self.fail = fail

    def propose(self, request):
        self.requests.append(copy.deepcopy(request))
        index = request["decision_index"]
        if index in self.fail:
            raise ClientError("recorded provider failure")
        identity = (
            "schema_version",
            "episode_id",
            "attempt_id",
            "decision_index",
            "observation_step",
            "interpolation_mode",
            "request_fingerprint",
        )
        return {
            **{key: request[key] for key in identity},
            "decision_id": f"{request['attempt_id']}_decision_{index}",
            "source_a_id": "a",
            "source_b_id": "b",
            "alpha": 1 / index,
            "observed_phase": "test phase",
            "rationale": "test visual decision",
            "vision": [
                {
                    "camera": "observation/image",
                    "kind": "point",
                    "coordinates": [3, 4],
                    "gain": 0.4,
                }
            ]
            if request["vision_enabled"]
            else [],
        }


def controller(client, **kwargs):
    return OnlineInterpolation(
        client,
        episode_id="case",
        attempt_id="astra_tli_2",
        operator="tli",
        task="put object in bowl",
        catalog=[
            {"source_id": "a", "prompt": "pick object"},
            {"source_id": "b", "prompt": "place in bowl"},
        ],
        spec=None,
        protocol=load_protocol(),
        **kwargs,
    )


def test_fresh_observations_hold_language_and_expire_vision_on_next_replan():
    client = ObservedClient(fail=(2,))
    online = controller(client, vision=True)
    original = observation(0)
    active, marks = online.update(original, 0)
    assert marks and active["alpha"] == 1
    assert np.all(original["observation/image"] == 0)
    for step in range(5, 30, 5):
        held, marks = online.update(observation(step), step)
        assert held == active
        assert marks == []
    assert [row["observation_step"] for row in client.requests] == [0, 25]
    assert [row["step"] for row in client.requests[-1]["observations"]] == [
        10,
        15,
        20,
        25,
    ]
    assert (
        client.requests[-1]["observations"][-1]["observation"]["observation/state"]
        == [25] * 8
    )
    assert online.decisions[-1]["accepted"] is False
    assert online.decisions[-1]["proposal"] is None
    for step in range(30, 55, 5):
        new, marks = online.update(observation(step), step)
    assert new["alpha"] == pytest.approx(1 / 3)
    assert marks
    request = client.requests[-1]
    assert [row["accepted"] for row in request["previous_decisions"]] == [True, False]
    assert request["active_interpolation"] == active


def test_first_failure_falls_back_to_native_and_uses_no_hidden_retry():
    client = ObservedClient(fail=(1,))
    online = controller(client)
    assert online.update(observation(0), 0) == (None, [])
    for step in range(5, 25, 5):
        assert online.update(observation(step), step) == (None, [])
    assert len(client.requests) == 1


def test_new_attempt_sees_previous_raw_rollout_without_reusing_its_active_choice():
    previous = {
        "feedback": {
            "attempt_id": "recovered_noise_1",
            "success": False,
            "executed_actions": 300,
            "termination": "budget_exhausted",
            "error": None,
        },
        "decisions": [],
        "snapshots": [{"label": "last", "step": 300, "observation": observation(90)}],
    }
    client = ObservedClient()
    online = controller(
        client, previous_attempt=previous, feedback=[previous["feedback"]]
    )
    online.update(observation(0), 0)
    request = client.requests[0]
    assert request["active_interpolation"] is None
    assert request["previous_attempt"]["snapshots"][0]["step"] == 300
    assert request["observations"][-1]["step"] == 0
    assert request["completed_rollout_feedback"][0]["success"] is False


def test_consecutive_steps_and_twelve_call_cap():
    client = ObservedClient()
    online = controller(client)
    with pytest.raises(ValueError, match="consecutive"):
        online.update(observation(5), 5)
    for step in range(0, 300, 5):
        online.update(observation(step), step)
    assert len(client.requests) == 12
    with pytest.raises(ValueError, match="budget"):
        online.update(observation(0), 300)


def test_censoring_distinguishes_rollout_revisions_from_online_decisions():
    attempts = [
        {
            "iteration": 1,
            "success": False,
            "actions_executed": 300,
            "velocity_evaluations": 600,
            "wall_seconds": 1,
            "decisions": [],
        },
        {
            "iteration": 2,
            "success": False,
            "actions_executed": 300,
            "velocity_evaluations": 600,
            "wall_seconds": 2,
            "decisions": [{"accepted": False}] * 12,
        },
        {
            "iteration": 3,
            "success": True,
            "actions_executed": 105,
            "velocity_evaluations": 210,
            "wall_seconds": 3,
            "decisions": [{"accepted": True}] * 5,
        },
    ]
    summary = arm_summary(attempts, 3)
    assert summary["first_success_attempt"] == 3
    assert summary["full_rollout_revisions_to_success"] == 2
    assert summary["within_successful_rollout_decisions"] == 5
    assert summary["decisions_through_success_or_cap"] == 17
    assert summary["success_by_attempt"] == [False, False, True]
    assert summary["provider"]["provider_calls"] == 0
    failed = arm_summary(attempts[:2], 3)
    assert failed["censored_without_success"] is True
    assert failed["full_rollout_revisions_to_success"] is None


def test_workers_cover_nine_donors_and_twenty_cases_without_overlap():
    sources = [
        source
        for worker in range(8)
        for source in assignment("bank", worker)["source_ids"]
    ]
    assert len(set(sources)) == len(sources) == 9
    cases = []
    for worker in range(8):
        target = assignment("evaluation", worker)
        cases.extend(
            (target["suite"], task)
            for task in range(10)[target["case_shard"] :: target["case_shards"]]
        )
    assert len(set(cases)) == len(cases) == 20
    for invalid in (True, -1, 8):
        with pytest.raises(ValueError):
            assignment("bank", invalid)


class AccountingOpener:
    """Real transport/parser path with synthetic provider responses and usage."""

    model = "synthetic-interpolation-orchestration"
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "total_tokens": 13,
        "completion_tokens_details": {"reasoning_tokens": 1},
    }

    def __init__(self, *, fail=()):
        self.fail, self.requests = set(fail), []

    def open(self, request, timeout):
        context = json.loads(
            json.loads(request.data)["messages"][1]["content"][0]["text"]
        )["request"]
        self.requests.append(context)
        envelope = {"model": self.model, "usage": self.usage}
        identity = (context["attempt_id"], context["decision_index"])
        if identity in self.fail:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "synthetic unavailable response",
                {},
                io.BytesIO(json.dumps(envelope).encode()),
            )
        proposal = ObservedClient().propose(context)
        envelope["choices"] = [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(proposal)},
            }
        ]
        result = io.BytesIO(json.dumps(envelope).encode())
        result.status = 200
        return result


def test_online_preflight_failure_consumes_slot_without_inventing_provider_cost(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    client = InterpolationClient(
        model=AccountingOpener.model, response_log=tmp_path / "provider.jsonl"
    )
    client.opener = AccountingOpener()
    events = []
    online = controller(client, record=lambda kind, **data: events.append((kind, data)))
    assert online.update(observation(0), 0) == (None, [])
    assert not client.opener.requests
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-test-key")
    for step in range(5, 30, 5):
        active, _ = online.update(observation(step), step)
    assert active["alpha"] == 0.5
    assert len(client.opener.requests) == 1
    request = client.opener.requests[0]
    assert request["decision_index"] == 2
    assert request["previous_decisions"][0]["accepted"] is False
    assert "not set" in request["previous_decisions"][0]["error"]
    assert request["active_interpolation"] is None
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == 2
    assert summary["preflight_failures"] == summary["provider_calls"] == 1
    assert summary["accepted_proposals"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 13
    decisions = [data for kind, data in events if kind == "interpolation_decision"]
    assert decisions[0]["raw_condition_fallback"]
    assert not decisions[1]["raw_condition_fallback"]


def test_online_billable_http_failure_keeps_held_text_and_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-test-key")
    client = InterpolationClient(
        model=AccountingOpener.model, response_log=tmp_path / "provider.jsonl"
    )
    client.opener = AccountingOpener(fail={("astra_tli_2", 2)})
    events = []
    online = controller(
        client, vision=True, record=lambda kind, **data: events.append((kind, data))
    )
    initial, marks = online.update(observation(0), 0)
    assert marks
    for step in range(5, 30, 5):
        held, marks = online.update(observation(step), step)
        assert held == initial and marks == []
    assert len(client.opener.requests) == 2
    summary = summarize_calls(client.response_log)
    assert summary["provider_calls"] == 2
    assert summary["accepted_proposals"] == summary["failed_calls"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 26
    decisions = [data for kind, data in events if kind == "interpolation_decision"]
    assert decisions[-1]["language_hold_after_failure"]
    assert decisions[-1]["vision_valid_until_step"] is None


@pytest.fixture
def phase_search_factory(monkeypatch, tmp_path, spec):
    """Actual search, controller, parser and ledger; synthetic policy/reset harness."""
    from astra_reversal import interpolation_agent, intervention_rollout

    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-phase-test-key")

    def make(*, baseline_success=False, development=False, all_provider_fail=False):
        protocol = load_protocol()
        protocol["arms"] = ["astra_tei", "astra_tli", "astra_tli_vision"]
        protocol["action_budget"] = 30
        protocol["astra"]["max_calls_per_rollout"] = 2
        protocol["astra"]["model"] = AccountingOpener.model
        protocol["execution_solver"] = {
            "solver": "euler",
            "steps": 10,
            "time_power": 1.0,
        }
        policy = SyntheticPolicy()
        policy.observation_image_size = 8

        def prepare_interpolated(
            raw, observation_id, prompt, *, source_prompts, alpha, operator, **kwargs
        ):
            condition = policy.prepare(raw, observation_id, prompt)
            return condition, {"has_effect": operator == "tei" or alpha != 0.5}

        policy.prepare_interpolated = prepare_interpolated
        entry = {
            "episode_id": "synthetic:seed29:task0:state0",
            "task_id": 0,
            "initial_state_id": 0,
            "seed": 29,
            "instruction": "put object in bowl",
        }
        search = InterpolationSearch(
            policy,
            lambda *args: (SimpleNamespace(), None, None),
            SimpleNamespace(suite="synthetic"),
            entry,
            protocol,
            tmp_path / "case",
            banks={"a": None, "b": None},
            catalog=[
                {"source_id": "a", "prompt": "pick object"},
                {"source_id": "b", "prompt": "place in bowl"},
            ],
            oracle={"source_a_id": "a", "source_b_id": "b", "lambda_calls": 14},
            development=development,
        )
        monkeypatch.setattr(
            "astra_reversal.interpolation_search.ActionSpec.from_environment",
            lambda *args: spec,
        )

        def initialize_noise(*args):
            search.known = np.zeros((1, policy.horizon, policy.action_dim), np.float32)
            search.recovered = search.known.copy()
            search.report["initialization"] = {
                "passed": True,
                "velocity_evaluations": 1230,
            }

        def check_interpolation(*args):
            search.interpolation_checked = True
            search.report["interpolation_gate"] = {
                "passed": True,
                "velocity_evaluations": 30,
            }

        monkeypatch.setattr(search, "initialize_noise", initialize_noise)
        monkeypatch.setattr(search, "check_interpolation", check_interpolation)
        failed = (
            {
                (f"{arm}_{iteration}", index)
                for arm in protocol["arms"]
                for iteration in (2, 3)
                for index in (1, 2)
            }
            if all_provider_fail
            else {("astra_tli_2", 2)}
        )
        opener = AccountingOpener(fail=failed)

        class RecordedClient(InterpolationClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.opener = opener

        monkeypatch.setattr(interpolation_agent, "InterpolationClient", RecordedClient)
        rollouts = []

        def run_rollout(
            env,
            entry,
            benchmark,
            callback,
            *,
            execute_steps,
            action_budget,
            expected_reset,
            policy_image_size,
            video_path,
        ):
            attempt_id = Path(video_path).stem
            mode, iteration = attempt_id.rsplit("_", 1)
            iteration = int(iteration)
            rollouts.append((mode, iteration))
            success = (mode == "recovered_noise" and baseline_success) or (
                mode.startswith("astra_")
                and iteration == (2 if mode == "astra_tli" else 3)
            )
            action_count = (
                10
                if mode in ("known_noise", "policy_fresh")
                or (mode == "recovered_noise" and baseline_success)
                else action_budget
            )
            for step in range(0, action_count, execute_steps):
                actions = callback(observation(step + iteration), step)
                assert actions.shape == (policy.horizon, 7)
            snapshots = [
                {"label": "first", "step": 0, "observation": observation(iteration)},
                {
                    "label": "last",
                    "step": action_count,
                    "observation": observation(action_count + iteration),
                },
            ]
            reset = {"synthetic_reset": "constant"}
            assert expected_reset in (None, reset)
            return {
                "success": success,
                "actions_executed": action_count,
                "policy_replans": action_count // execute_steps,
                "wall_seconds": 2.0,
                "initial_success": False,
                "terminated": False,
                "reset_audit": reset,
                "snapshots": snapshots,
            }

        monkeypatch.setattr(intervention_rollout, "run_rollout", run_rollout)
        return search, opener.requests, rollouts

    return make


def test_search_revisions_use_only_own_feedback_and_unique_physical_cost(
    phase_search_factory,
):
    search, requests, rollouts = phase_search_factory()
    report = search.run()
    assert report["status"] == "complete"
    assert rollouts == [
        ("recovered_noise", 1),
        ("known_noise", 1),
        ("policy_fresh", 1),
        ("astra_tei", 2),
        ("astra_tei", 3),
        ("astra_tli", 2),
        ("astra_tli_vision", 2),
        ("astra_tli_vision", 3),
    ]
    for request in requests:
        arm, iteration = request["attempt_id"].rsplit("_", 1)
        previous = request["previous_attempt"]
        expected_attempt = "recovered_noise_1" if iteration == "2" else f"{arm}_2"
        assert previous["feedback"]["attempt_id"] == expected_attempt
        assert [row["attempt_id"] for row in request["completed_rollout_feedback"]] == (
            ["recovered_noise_1"]
            if iteration == "2"
            else ["recovered_noise_1", f"{arm}_2"]
        )
        assert all(
            row["proposal"]["attempt_id"] == expected_attempt
            for row in previous["decisions"]
            if row["accepted"]
        )
        if request["decision_index"] == 1:
            assert request["active_interpolation"] is None
            assert request["previous_decisions"] == []
        assert "oracle" not in request and "oracle" not in json.dumps(previous)
    cost = report["physical_cost"]
    assert cost["rollouts"] == 8
    assert cost["simulated_actions"] == 200
    assert cost["rollout_velocity_evaluations"] == 400
    assert cost["velocity_evaluations"] == 1660
    assert cost["token_usage"]["provider_calls"] == 10
    assert cost["token_usage"]["failed_calls"] == 1
    assert cost["token_usage"]["tokens"]["total_tokens"]["sum"] == 130
    tei = report["arms"]["astra_tei"]["summary"]
    assert tei["first_success_attempt"] == 3
    assert tei["full_rollout_revisions_to_success"] == 2
    assert tei["provider_through_success_or_cap"]["provider_calls"] == 4
    assert tei["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 52
    assert tei["standalone_velocity_evaluations_through_success_or_cap"] == 1440
    held = report["arms"]["astra_tli"]["attempts"][-1]
    assert held["accepted_decisions_executed"] == 1
    assert held["actions_with_held_text_after_failed_call"] == 5
    assert held["actions_with_accepted_decision"] == 30
    assert held["actions_with_nonzero_text"] == 30
    vision = report["arms"]["astra_tli_vision"]["attempts"][-1]
    assert vision["actions_with_changed_vision"] == 10
    assert vision["vision_changed_policy_calls"] == 2
    assert vision["accepted_decisions_executed"] == 2


@pytest.mark.parametrize("development", [False, True])
def test_successful_baseline_has_zero_tokens_to_success_despite_development_cost(
    phase_search_factory, development
):
    search, requests, rollouts = phase_search_factory(
        baseline_success=True, development=development
    )
    report = search.run()
    assert len(rollouts) == 3 + 3 * int(development)
    assert len(requests) == 6 * int(development)
    for arm in report["arms"].values():
        summary = arm["summary"]
        assert summary["first_success_attempt"] == 1
        assert summary["full_rollout_revisions_to_success"] == 0
        assert summary["actions_through_success_or_cap"] == 10
        assert summary["velocity_evaluations_through_success_or_cap"] == 20
        assert summary["standalone_velocity_evaluations_through_success_or_cap"] == 1280
        assert summary["decisions_through_success_or_cap"] == 0
        assert summary["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 0
        assert summary["provider_through_success_or_cap"]["provider_calls"] == 0
        assert summary["provider"]["provider_calls"] == 2 * int(development)
        assert summary["development_extra_rollouts_after_success"] == int(development)
    assert report["physical_cost"]["token_usage"]["tokens"]["total_tokens"][
        "sum"
    ] == 78 * int(development)
    assert report["physical_cost"]["velocity_evaluations"] == 1320 + 180 * int(
        development
    )
    if development:
        assert report["development_validation"]["passed"]
        assert all(
            row["passed"] for row in report["development_validation"]["arms"].values()
        )


def test_development_all_provider_errors_preserves_evidence_then_fails_gate(
    phase_search_factory,
):
    search, requests, rollouts = phase_search_factory(
        baseline_success=True, development=True, all_provider_fail=True
    )
    with pytest.raises(ValueError, match="accepted-and-executed"):
        search.run()
    report = json.loads((search.directory / "summary.json").read_text())
    assert report["status"] == "development_failed"
    assert len(requests) == 6 and len(rollouts) == 6
    assert not report["development_validation"]["passed"]
    assert all(
        not row["passed"] for row in report["development_validation"]["arms"].values()
    )
    cost = report["physical_cost"]["token_usage"]
    assert cost["provider_calls"] == cost["failed_calls"] == 6
    assert cost["accepted_proposals"] == 0
    assert cost["tokens"]["total_tokens"]["sum"] == 78
    for arm in report["arms"].values():
        assert arm["attempts"][-1]["native_condition_fallback_actions"] == 30
        assert arm["attempts"][-1]["accepted_decisions_executed"] == 0


@pytest.mark.parametrize(
    "case", ["pass", "identity_residual", "padding_only", "solve_error"]
)
def test_weighted_gate_records_all_measured_failures_and_returned_vf(tmp_path, case):
    class ProbePolicy:
        @staticmethod
        def tensor(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def prepare(*args):
            return SimpleNamespace(condition_id="native", delta=0.0)

        @staticmethod
        def prepare_interpolated(
            raw,
            observation_id,
            prompt,
            *,
            source_prompts,
            alpha,
            operator,
            text_latents,
        ):
            nonzero = alpha == 0.0
            condition = SimpleNamespace(
                condition_id=operator if nonzero else "native",
                delta=0.1 if nonzero else 0.0,
            )
            if case == "identity_residual" and operator == "tei" and not nonzero:
                condition.delta = 0.01
            if nonzero and operator == "tli":
                assert text_latents == {
                    "a": "synthetic-bank-a",
                    "b": "synthetic-bank-b",
                }
            return condition, {"has_effect": nonzero}

        @staticmethod
        def sample(condition, latent, **solver):
            if case == "solve_error" and condition.condition_id == "tli":
                raise RuntimeError("synthetic last-solve failure")
            values = latent.copy()
            if case == "padding_only" and condition.condition_id == "tli":
                values[:, :, 7:] += condition.delta
            else:
                values += condition.delta
            return SimpleNamespace(value=values, velocity_evaluations=10)

        @staticmethod
        def output_transform(data):
            return {"actions": data["actions"][:, :7] * 2}

    search = InterpolationSearch.__new__(InterpolationSearch)
    search.policy = ProbePolicy()
    search.entry = {"instruction": "target task"}
    search.known = np.zeros((1, 10, 32), np.float32)
    search.sources = {"a": "donor a", "b": "donor b"}
    search.banks = {"a": "synthetic-bank-a", "b": "synthetic-bank-b"}
    search.oracle = {"source_a_id": "a", "source_b_id": "b"}
    search.protocol = {
        "execution_solver": {"solver": "euler", "steps": 10, "time_power": 1.0}
    }
    search.report = {}
    search.directory = tmp_path / "gate"
    search.progress = lambda: None
    search.recorder = Recorder(search.directory)
    search.interpolation_checked = False
    if case == "pass":
        search.check_interpolation(observation(0))
    else:
        with pytest.raises((ValueError, RuntimeError)):
            search.check_interpolation(observation(0))
    gate = json.loads((search.directory / "summary.json").read_text())[
        "interpolation_gate"
    ]
    assert gate == search.report["interpolation_gate"]
    assert gate["passed"] == search.interpolation_checked == (case == "pass")
    assert gate["velocity_evaluations"] == (40 if case == "solve_error" else 50)
    assert gate["velocity_evaluations_complete"] == (case != "solve_error")
    assert gate["complete"] == (case != "solve_error")
    events = [
        json.loads(line)
        for line in (search.directory / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["kind"] == "interpolation_gate"
    assert events[-1]["passed"] == gate["passed"]
    measured = [row for row in events if row["kind"] == "interpolation_gate_check"]
    assert len(measured) == (3 if case == "solve_error" else 4)
    if case == "padding_only":
        row = gate["checks"]["tli_nonzero_oracle_banks"]
        assert row["errors"]["max_abs"] > 0
        assert row["controlled_channel_errors"]["max_abs"] == 0
        assert not row["passed"]
    if case == "identity_residual":
        assert not gate["checks"]["tei_identical_sources"]["passed"]
        assert gate["checks"]["tli_nonzero_oracle_banks"]["passed"]
    for row in measured:
        for name in ("generated_actions", "decoded_actions"):
            ref = row[name]
            values = np.load(search.directory / ref["array"], allow_pickle=False)
            assert digest(values) == ref["sha256"]
