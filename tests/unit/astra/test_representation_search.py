"""Budget isolation and reproducible matched controls for representation trials."""

import copy
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.representation_search import (
    ARMS,
    RepresentationSearch,
    keyed_rng,
    load_protocol,
    random_choice,
)

from .test_representation_agent import library as synthetic_library
from .test_representation_agent import snapshot


def test_policy_stream_is_keyed_and_independent_of_global_rng():
    entry = {"seed": 47, "episode_id": "libero_goal_ood:seed47:task2:state0"}
    first = keyed_rng(entry, 1, 25).standard_normal((1, 10, 32))
    np.random.seed(827)
    np.random.normal(size=10000)
    np.testing.assert_array_equal(
        first, keyed_rng(entry, 1, 25).standard_normal(first.shape)
    )
    for revision, step, stream in ((2, 25, 0), (1, 30, 0), (1, 25, 1)):
        assert not np.array_equal(
            first, keyed_rng(entry, revision, step, stream).standard_normal(first.shape)
        )


def test_random_controls_use_only_declared_catalogs():
    rng = np.random.default_rng(19)
    for mode in ("tli", "vei", "vli"):
        for _ in range(30):
            choice = random_choice(mode, rng, ["1", "2", "3"], ["frame-a", "frame-b"])
            if mode == "tli":
                assert choice["vision"] is None
                lang = choice["language"]
                assert lang["source_a_id"] != lang["source_b_id"]
                assert {lang["source_a_id"], lang["source_b_id"]} <= {"1", "2", "3"}
            else:
                assert choice["language"] is None
                assert choice["vision"]["donor_id"] in {"frame-a", "frame-b"}
    with pytest.raises(ValueError):
        random_choice("oracle", rng, ["1", "2"], ["frame-a"])


@pytest.mark.parametrize(
    "baseline_success,development", [(False, False), (True, False), (True, True)]
)
def test_each_arm_gets_only_its_own_feedback_and_shared_baseline(
    baseline_success, development
):
    # Exercise the orchestration with completed physical rollout records. Only
    # one arm succeeds at revision1; other arms must not see or reuse its result.
    search = RepresentationSearch.__new__(RepresentationSearch)
    search.protocol = load_protocol()
    search.development = development
    search.report = {"physical_rollouts": [], "arms": {}}
    search.save = lambda: None
    search.progress = lambda: None
    seen = []

    def rollout(arm, revision, previous=None, history=()):
        seen.append(
            (arm, revision, copy.deepcopy(previous), [r["attempt_id"] for r in history])
        )
        if revision:
            assert previous["arm"] in ("native", arm)
            assert all(r["arm"] in ("native", arm) for r in history)
        result = {
            "arm": arm,
            "attempt_id": f"{arm}-{revision}",
            "revision": revision,
            "success": baseline_success
            if revision == 0
            else arm == "astra_tli" and revision == 1,
            "actions_executed": 50,
            "decisions": [],
            "provider_records": [],
            "velocity_evaluations": 100,
            "parity_velocity_evaluations": 0,
            "probe_velocity_evaluations": 0,
            "donor_captures": 0,
            "donor_capture_seconds": 0,
        }
        search.report["physical_rollouts"].append(result)
        return result, {"arm": arm, "revision": revision}

    search.rollout = rollout
    result = search.run()
    expected = (
        1 + len(ARMS)
        if baseline_success and development
        else 1
        if baseline_success
        else 2 * len(ARMS)
    )
    assert len(seen) == expected
    assert result["physical_cost"]["rollouts"] == expected
    assert result["physical_cost"]["actions"] == 50 * expected
    assert sum(arm == "native" for arm, *_ in seen) == 1
    if baseline_success:
        assert all(r["first_success_revision"] == 0 for r in result["arms"].values())
        assert all(
            r["development_extra_rollouts"] == int(development)
            for r in result["arms"].values()
        )
    else:
        assert result["arms"]["astra_tli"]["first_success_revision"] == 1
        assert result["arms"]["astra_vli"]["censored"]
        assert result["arms"]["astra_vli"]["success_by_revision"] == [
            False,
            False,
            False,
        ]


@pytest.mark.parametrize("action_count", [28, 54, 300])
def test_real_client_contract_cadence_failure_clear_and_executed_prefix_costs(
    monkeypatch, tmp_path, action_count
):
    """Use real HTTP serialization/parser/ledger with a synthetic in-memory opener.

    The first call steers, the second returns a billable HTTP503, and all later
    calls explicitly choose native. Short terminal chunks must not be charged
    as five executed actions. No model, simulator or GPU is invoked.
    """
    from astra_reversal import representation_agent as agent
    from astra_reversal import representation_search as runner
    from astra_reversal.action_adapter import ActionSpec
    from astra_reversal.records import digest

    sources, catalog, sheets = synthetic_library()
    library = SimpleNamespace(
        library_id="a" * 64,
        catalog=lambda: copy.deepcopy(catalog),
        contact_sheets=lambda: copy.deepcopy(sheets),
    )
    monkeypatch.setattr(runner, "donor_catalog", lambda: copy.deepcopy(sources))
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-only-test-value")
    requests, clients = [], []
    actual_client = agent.RepresentationClient
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }

    class Opener:
        def open(self, request, timeout):
            assert timeout == 170
            payload = json.loads(request.data)
            context = json.loads(payload["messages"][1]["content"][0]["text"])[
                "request"
            ]
            requests.append(context)
            if context["decision_index"] == 2:
                raise urllib.error.HTTPError(
                    "https://synthetic.invalid",
                    503,
                    "synthetic failure",
                    {},
                    io.BytesIO(
                        json.dumps(
                            {"usage": usage, "model": "synthetic/model"}
                        ).encode()
                    ),
                )
            active = context["decision_index"] == 1
            proposal = {
                "request_id": context["request_id"],
                "mode": "interpolate" if active else "native",
                "language": {
                    "source_a_id": "10",
                    "source_b_id": "13",
                    "alpha": 0.25,
                }
                if active
                else None,
                "vision": None,
                "observed_phase": "synthetic phase",
                "rationale": "Synthetic contract check only.",
            }
            response = io.BytesIO(
                json.dumps(
                    {
                        "model": "synthetic/model",
                        "usage": usage,
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(proposal),
                                },
                            }
                        ],
                    }
                ).encode()
            )
            response.status = 200
            return response

    def client_factory(**kwargs):
        kwargs["model"] = "synthetic/model"
        client = actual_client(**kwargs)
        client.opener = Opener()
        clients.append(client)
        return client

    monkeypatch.setattr(agent, "RepresentationClient", client_factory)
    spec = ActionSpec(
        action_spec_id="synthetic",
        horizon=10,
        model_action_dim=32,
        timestep_seconds=0.05,
        lower=(-1.0,) * 7,
        upper=(1.0,) * 7,
        semantics={},
    )
    monkeypatch.setattr(runner.ActionSpec, "from_environment", lambda *args: spec)
    monkeypatch.setattr(
        runner,
        "ActionAdapter",
        lambda *args: SimpleNamespace(
            decode=lambda value, state: (value[0, :, :7], {"count": 0})
        ),
    )
    policy = SimpleNamespace(
        horizon=10,
        action_dim=32,
        observation_image_size=224,
        input_transform=None,
        output_transform=None,
        noise=lambda rng: rng.normal(size=(1, 10, 32)),
        sample=lambda *args, **kwargs: SimpleNamespace(
            value=np.zeros((1, 10, 32), np.float32),
            velocity_evaluations=10,
            latency_seconds=0.0,
        ),
    )
    applied, raw_hashes = [], []

    def condition(observation, choice, mode, counters):
        assert mode == "tli"
        applied.append(None if choice is None else choice["mode"])
        raw_hashes.append(digest(observation))
        return SimpleNamespace(
            preparation_seconds=0.0,
            state=np.zeros(8),
            condition_id="synthetic",
        ), {"has_effect": choice is not None and choice["mode"] == "interpolate"}

    def fake_rollout(env, entry, benchmark, act, **kwargs):
        assert kwargs["execute_steps"] == 5
        assert kwargs["action_budget"] == 300
        for step in range(0, action_count, 5):
            observation = snapshot(step, value=step % 200)["observation"]
            before = digest(observation)
            assert act(observation, step).shape == (10, 7)
            assert digest(observation) == raw_hashes[-1] == before
        Path(kwargs["video_path"]).write_bytes(b"synthetic video placeholder")
        return {
            "success": False,
            "actions_executed": action_count,
            "initial_success": False,
            "reset_audit": {},
            "terminated": action_count < 300,
            "video_path": str(kwargs["video_path"]),
            "snapshots": [snapshot(action_count)],
            "wall_seconds": 1.0,
            "policy_seconds": 0.8,
            "environment_seconds": 0.1,
        }

    monkeypatch.setattr(runner, "run_rollout", fake_rollout)
    search = RepresentationSearch(
        policy,
        lambda *args: (None, None, None),
        SimpleNamespace(),
        {
            "episode_id": "synthetic:seed47:task0:state0",
            "suite": "synthetic",
            "task_id": 0,
            "instruction": "Pick up the bowl.",
            "seed": 47,
            "initial_state_id": 0,
        },
        load_protocol(),
        tmp_path / "case",
        library=library,
        banks={},
    )
    search.parity_checked = True  # The weighted CUDA gate is a separate test.
    search._condition = condition
    baseline = {
        "attempt_id": "native_revision0",
        "success": False,
        "actions_executed": 300,
        "status": "budget_exhausted",
    }
    previous = {
        "feedback": runner.feedback(baseline),
        "decisions": [],
        "snapshots": [snapshot(300, label="previous raw final")],
    }
    result, retained = search.rollout("astra_tli", 1, previous, [baseline])
    call_count = (action_count + 24) // 25
    assert len(requests) == len(clients[0].records) == call_count <= 12
    assert [r["observation_step"] for r in requests] == list(range(0, action_count, 25))
    assert all(
        len(r["observations"]) <= 4
        and r["observations"][-1]["step"] == r["observation_step"]
        for r in requests
    )
    assert all(
        len(r["previous_decisions"]) == min(2, index)
        for index, r in enumerate(requests)
    )
    if call_count > 2:
        assert requests[2]["previous_decisions"][1]["proposal"] is None
        assert "503" in requests[2]["previous_decisions"][1]["error"]
    assert all(r["previous_attempt"]["decisions"] == [] for r in requests)
    assert all(
        [r["attempt_id"] for r in row["completed_rollout_feedback"]]
        == ["native_revision0"]
        for row in requests
    )
    assert applied[:5] == ["interpolate"] * 5
    assert all(mode is None for mode in applied[5:10])
    assert all(mode == "native" for mode in applied[10:])
    fallback = min(25, action_count - 25)
    assert result["provider_failure_fallback_actions"] == fallback
    assert result["accepted_decision_actions"] == action_count - fallback
    assert result["explicit_native_actions"] == max(0, action_count - 50)
    assert result["nonzero_intervention_actions"] == 25
    assert result["accepted_decisions_executed"] == call_count - 1
    assert len(retained["decisions"]) == 2
    costs = agent.summarize_calls(result["provider_records"])
    assert costs["provider_calls"] == call_count
    assert costs["accepted_proposals"] == call_count - 1
    assert costs["tokens"]["total_tokens"]["sum"] == 15 * call_count
    assert costs["usage_unavailable_calls"] == 0
