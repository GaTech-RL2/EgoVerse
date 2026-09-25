"""CPU orchestration tests with synthetic rollouts and no provider/network access."""

import copy
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from astra_reversal.intervention_agent import InterventionClient
from astra_reversal.intervention_search import InterventionSearch, load_protocol

from .conftest import SyntheticEnvironment, SyntheticPolicy

MODEL = "synthetic-provider/search-orchestration"
USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 2,
    "total_tokens": 13,
    "completion_tokens_details": {"reasoning_tokens": 1},
}


def proposal_for(request):
    iteration = request["iteration"]
    candidate_id = f"candidate_{iteration}"
    return {
        **{
            key: request[key]
            for key in (
                "schema_version",
                "episode_id",
                "iteration",
                "arm",
                "request_fingerprint",
            )
        },
        "candidate_id": candidate_id,
        "best_candidate_id": candidate_id,
        "rationale": "Synthetic revision based on the observed prior attempt.",
        "language": {"target_text": "Place the cup in the bowl.", "scale": 0.5},
        "vision": [
            {
                "camera": "observation/image",
                "kind": "point",
                "coordinates": [5, 5],
                "gain": 0.2,
            }
        ],
        "noise": {
            "basis_id": request["basis_id"],
            "coefficients": [iteration / 100] * 8,
            "perturbation_scale": 0.3,
        },
    }


class ScriptedOpener:
    def __init__(self, fail_iteration):
        self.fail_iteration = fail_iteration

    def open(self, request, timeout):
        context = json.loads(
            json.loads(request.data)["messages"][1]["content"][0]["text"]
        )["request"]
        if context["iteration"] == self.fail_iteration:
            body = {"model": MODEL, "id": "synthetic-http-error", "usage": USAGE}
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Synthetic unavailable response",
                {},
                io.BytesIO(json.dumps(body).encode()),
            )
        envelope = {
            "model": MODEL,
            "id": f"synthetic-{context['iteration']}",
            "usage": USAGE,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(proposal_for(context)),
                    },
                }
            ],
        }
        result = io.BytesIO(json.dumps(envelope).encode())
        result.status = 200
        return result


@pytest.fixture
def search_factory(monkeypatch, tmp_path, spec):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-search-test-credential")

    def make(*, baseline_success=False, development=False, fail_iteration=2):
        protocol = copy.deepcopy(load_protocol())
        protocol["arms"] = ["joint"]
        protocol["astra"]["model"] = MODEL
        policy = SyntheticPolicy()
        policy.max_token_len = 200
        policy.prompt_length = lambda prompt, observation: 100
        search = InterventionSearch(
            policy,
            None,
            SimpleNamespace(suite="synthetic"),
            {
                "episode_id": "synthetic:seed19:task0:state0",
                "task_id": 0,
                "initial_state_id": 0,
                "seed": 19,
                "instruction": "Place the cup in the bowl.",
            },
            protocol,
            tmp_path / "case",
            development=development,
        )
        search.spec = spec
        search.report["initialization"] = {"velocity_evaluations": 1210}
        requests, rollouts = [], []

        class RecordingClient(InterventionClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.opener = ScriptedOpener(fail_iteration)

            def propose(self, request):
                requests.append(copy.deepcopy(request))
                return super().propose(request)

        def rollout(mode, iteration, proposal=None):
            rollouts.append((mode, iteration, copy.deepcopy(proposal)))
            success = (iteration == 1 and baseline_success) or (
                mode == "joint" and iteration == 4
            )
            observation = SyntheticEnvironment().observe()
            observation["observation/image"][:] = iteration
            attempt = {
                "iteration": iteration,
                "candidate_id": proposal["candidate_id"] if proposal else mode,
                "proposal": proposal,
                "success": success,
                "rollout_executed": True,
                "status": "success" if success else "budget_exhausted",
                "actions_executed": 20,
                "velocity_evaluations": 40,
                "wall_seconds": 2.0,
            }
            snapshots = [
                {"label": "first", "step": 0, "observation": observation},
                {"label": "last", "step": 20, "observation": observation},
            ]
            return attempt, snapshots

        search.client_type = RecordingClient
        monkeypatch.setattr(search, "rollout", rollout)
        return search, requests, rollouts

    return make


def test_failed_provider_attempt_then_revision_success_counts_history_and_cost(
    search_factory,
):
    search, requests, rollouts = search_factory()
    report = search.run()
    arm = report["arms"]["joint"]
    assert report["status"] == "complete"
    assert [request["iteration"] for request in requests] == [2, 3, 4]
    assert [len(request["prior_candidates"]) for request in requests] == [1, 2, 3]
    failed = requests[1]["prior_candidates"][1]
    assert failed["iteration"] == 2
    assert failed["outcome"]["success"] is None
    assert failed["outcome"]["executed_steps"] == 0
    assert "HTTP 503" in failed["outcome"]["error"]
    assert requests[1]["incumbent_candidate_id"] == "reversal_identity"
    prior_revision = requests[2]["prior_candidates"][2]
    assert prior_revision["proposal"]["noise"]["coefficients"] == [0.03] * 8
    assert prior_revision["outcome"]["success"] is False
    assert requests[2]["incumbent_candidate_id"] == "candidate_3"
    assert requests[2]["observations"] != requests[0]["observations"]
    assert [(mode, iteration) for mode, iteration, _ in rollouts] == [
        ("reversal_identity", 1),
        ("known_noise", 1),
        ("policy_fresh", 1),
        ("joint", 3),
        ("joint", 4),
    ]
    summary = arm["summary"]
    assert summary["first_success_attempt"] == 4
    assert summary["intervention_iterations_to_success"] == 3
    assert summary["candidate_rollouts"] == 3
    assert summary["proposal_failures"] == 1
    assert summary["success_by_attempt"] == [False, False, False, True, True]
    assert summary["velocity_evaluations_through_success_or_cap"] == 120
    assert summary["standalone_velocity_evaluations_through_success_or_cap"] == 1330
    assert [
        row["cumulative_token_usage"]["tokens"]["total_tokens"]["sum"]
        for row in arm["attempts"][1:]
    ] == [13, 26, 39]
    for token_usage in (
        arm["token_usage"],
        arm["tokens_to_first_success"],
        report["physical_cost"]["token_usage"],
    ):
        assert token_usage["provider_calls"] == 3
        assert token_usage["accepted_proposals"] == 2
        assert token_usage["failed_calls"] == 1
        assert token_usage["tokens"]["total_tokens"] == {
            "sum": 39,
            "available_calls": 3,
            "missing_calls": 0,
            "complete": True,
        }
        assert token_usage["tokens"]["reasoning_tokens"]["sum"] == 3
    assert report["physical_cost"]["rollouts"] == 5
    assert report["physical_cost"]["velocity_evaluations"] == 1410


def test_development_rejection_consumes_budget_before_required_hook_rollout(
    search_factory,
):
    search, requests, rollouts = search_factory(baseline_success=True, development=True)
    report = search.run()
    assert [request["iteration"] for request in requests] == [2, 3]
    assert [(mode, iteration) for mode, iteration, _ in rollouts][-1] == ("joint", 3)
    arm = report["arms"]["joint"]
    assert arm["summary"]["first_success_attempt"] == 1
    assert arm["summary"]["proposal_failures"] == 1
    assert arm["token_usage"]["tokens"]["total_tokens"]["sum"] == 26
    assert arm["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 0


@pytest.mark.parametrize("development", [False, True])
def test_successful_baseline_has_zero_tokens_to_success_even_if_dev_exercises_hook(
    search_factory, development
):
    search, requests, rollouts = search_factory(
        baseline_success=True, development=development, fail_iteration=None
    )
    report = search.run()
    arm = report["arms"]["joint"]
    assert arm["summary"]["first_success_attempt"] == 1
    assert arm["summary"]["intervention_iterations_to_success"] == 0
    assert arm["tokens_to_first_success"]["provider_calls"] == 0
    assert arm["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 0
    assert len(requests) == int(development)
    assert len(rollouts) == 3 + int(development)
    physical = report["physical_cost"]["token_usage"]
    assert physical["provider_calls"] == int(development)
    assert physical["tokens"]["total_tokens"]["sum"] == 13 * int(development)
    assert arm["summary"]["velocity_evaluations_through_success_or_cap"] == 40
    assert (
        arm["summary"]["standalone_velocity_evaluations_through_success_or_cap"] == 1250
    )
