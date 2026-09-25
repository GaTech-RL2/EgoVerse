"""Image scheduling/search integration with synthetic HTTP and rollout outcomes."""

import copy
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.astra_client import ClientError
from astra_reversal.image_perturbation_agent import (
    ImagePerturbationClient,
    parse_proposal,
    summarize_calls,
)
from astra_reversal.image_perturbation_search import (
    ImagePerturbationSearch,
    OnlineImagePerturbation,
    load_protocol,
    random_image_operations,
)
from astra_reversal.image_perturbations import (
    CAMERAS,
    ImagePerturbationLimits,
    apply_image_perturbations,
    validate_image_perturbations,
)

from .conftest import SyntheticPolicy
from .test_image_perturbation_agent import proposal_for, snapshot, synthetic_library


class SyntheticLibrary:
    def catalog(self):
        return copy.deepcopy(synthetic_library()[0])

    def contact_sheets(self):
        return synthetic_library()[1]

    def metadata(self):
        return {"library_id": self.catalog()[0]["library_id"], "synthetic": True}

    def resolve(self, donor_id, camera):
        rows = self.catalog()
        index = next(
            index for index, row in enumerate(rows) if row["donor_id"] == donor_id
        )
        row = rows[index]
        return SimpleNamespace(
            donor_id=donor_id,
            camera=camera,
            pixels=np.full((224, 224, 3), 31 + index, np.uint8),
            provenance={
                "library_id": row["library_id"],
                "sample_sha256": row["sample_sha256"],
                "pixels_sha256": row["cameras"][camera]["pixels_sha256"],
            },
        )


def observation(value):
    return snapshot(value=value % 200)["observation"]


class ObservedClient:
    def __init__(self, fail=()):
        self.requests, self.fail = [], set(fail)

    def propose(self, request):
        self.requests.append(copy.deepcopy(request))
        if request["decision_index"] in self.fail:
            raise ClientError("synthetic recorded provider failure")
        return parse_proposal(proposal_for(request), request)


def controller(client, *, image_mode="occlusion", **kwargs):
    return OnlineImagePerturbation(
        client,
        episode_id="synthetic-image-case",
        attempt_id=f"astra_{image_mode}_2",
        image_mode=image_mode,
        task="put object in bowl",
        library=SyntheticLibrary(),
        spec=None,
        protocol=load_protocol(),
        **kwargs,
    )


def test_fresh_pixels_edited_for_five_chunks_then_failed_refresh_clears_without_retry():
    client, events = ObservedClient(fail=(2,)), []
    online = controller(client, record=lambda kind, **row: events.append((kind, row)))
    initial = online.update(observation(0), 0)
    assert initial and online.active_decision_id
    for step in range(5, 25, 5):
        raw = observation(step)
        active = online.update(raw, step)
        assert active == initial
        edited, _ = apply_image_perturbations(raw, active)
        assert edited[CAMERAS[0]][0, 0, 0] != raw[CAMERAS[0]][0, 0, 0]
        assert np.all(raw[CAMERAS[0]] == step)
    for step in range(25, 50, 5):
        assert online.update(observation(step), step) == []
        assert online.active_decision_id is None
    assert len(client.requests) == 2
    assert [row["step"] for row in client.requests[-1]["observations"]] == [
        10,
        15,
        20,
        25,
    ]
    assert online.update(observation(50), 50) == initial
    assert len(client.requests) == 3
    assert [row["accepted"] for row in client.requests[-1]["previous_decisions"]] == [
        True,
        False,
    ]
    decisions = [row for kind, row in events if kind == "image_decision"]
    assert [row["valid_until_step"] for row in decisions] == [25, 50, 75]
    assert [row["raw_condition_fallback"] for row in decisions] == [False, True, False]


def test_previous_rollout_is_separate_and_does_not_activate_an_edit_before_acceptance():
    feedback = {
        "attempt_id": "recovered_noise_1",
        "success": False,
        "executed_actions": 300,
        "termination": "budget_exhausted",
        "error": None,
    }
    previous = {
        "feedback": feedback,
        "decisions": [],
        "snapshots": [snapshot(300, value=91)],
    }
    client = ObservedClient(fail=(1,))
    online = controller(client, previous_attempt=previous, feedback=[feedback])
    assert online.update(observation(0), 0) == []
    assert online.active_decision_id is None
    request = client.requests[0]
    assert request["previous_attempt"]["snapshots"][0]["step"] == 300
    assert request["observations"][0]["step"] == 0
    assert request["completed_rollout_feedback"] == [feedback]


def test_scheduler_requires_consecutive_policy_steps_and_caps_twelve_calls():
    client = ObservedClient()
    online = controller(client)
    with pytest.raises(ValueError, match="consecutive"):
        online.update(observation(5), 5)
    for step in range(0, 300, 5):
        online.update(observation(step), step)
    assert [row["observation_step"] for row in client.requests] == list(
        range(0, 300, 25)
    )
    with pytest.raises(ValueError, match="budget"):
        online.update(observation(300), 300)
    assert len(client.requests) == 12


@pytest.mark.parametrize("mode", ["occlusion", "demo_blend"])
def test_matched_random_is_reproducible_within_the_same_bounds(mode):
    raw, library = observation(37), SyntheticLibrary()
    donor_ids = [row["donor_id"] for row in library.catalog()]
    first, second = np.random.default_rng(73), np.random.default_rng(73)
    controls = []
    for _ in range(100):
        operations = random_image_operations(mode, raw, donor_ids, first)
        assert operations == random_image_operations(mode, raw, donor_ids, second)
        assert (
            validate_image_perturbations(
                operations,
                {camera: raw[camera].shape for camera in CAMERAS},
                donor_ids=donor_ids,
                limits=ImagePerturbationLimits(allowed_kinds=(mode,)),
            )
            == operations
        )
        controls.append(operations)
    assert any(not ops for ops in controls)
    assert any(len(ops) == 2 for ops in controls)
    assert np.all(raw[CAMERAS[0]] == 37)


class AccountingOpener:
    model = "synthetic-image-orchestration"
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
        if (context["attempt_id"], context["decision_index"]) in self.fail:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "synthetic failure",
                {},
                io.BytesIO(json.dumps(envelope).encode()),
            )
        envelope["choices"] = [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal_for(context)),
                },
            }
        ]
        result = io.BytesIO(json.dumps(envelope).encode())
        result.status = 200
        return result


def test_provider_and_preflight_slots_keep_distinct_costs(monkeypatch, tmp_path):
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
    client = ImagePerturbationClient(
        model=AccountingOpener.model, response_log=tmp_path / "calls.jsonl"
    )
    client.opener = AccountingOpener(fail={("astra_occlusion_2", 2)})
    online = controller(client)
    assert online.update(observation(0), 0) == []
    assert not client.opener.requests
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-image-scheduler-key")
    for step in range(5, 50, 5):
        assert online.update(observation(step), step) == []
    assert online.update(observation(50), 50)
    assert len(client.opener.requests) == 2
    summary = summarize_calls(client.response_log)
    assert summary["client_attempts"] == 3 and summary["preflight_failures"] == 1
    assert summary["provider_calls"] == 2 and summary["failed_calls"] == 1
    assert summary["tokens"]["total_tokens"]["sum"] == 26
    assert [
        row["accepted"] for row in client.opener.requests[-1]["previous_decisions"]
    ] == [False, False]


@pytest.fixture
def image_search_factory(monkeypatch, tmp_path, spec):
    from astra_reversal import image_perturbation_agent, intervention_rollout

    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-image-search-key")

    def make(*, baseline_success=False, development=False, all_provider_fail=False):
        protocol = load_protocol()
        protocol["action_budget"] = 30
        protocol["astra"]["max_calls_per_rollout"] = 2
        protocol["astra"]["model"] = AccountingOpener.model
        policy = SyntheticPolicy()
        policy.observation_image_size = 224
        entry = {
            "episode_id": "synthetic:seed37:task0:state0",
            "task_id": 0,
            "initial_state_id": 0,
            "seed": 37,
            "instruction": "put object in bowl",
        }
        search = ImagePerturbationSearch(
            policy,
            lambda *args: (SimpleNamespace(), None, None),
            SimpleNamespace(suite="synthetic"),
            entry,
            protocol,
            tmp_path / "case",
            library=SyntheticLibrary(),
            development=development,
        )
        monkeypatch.setattr(
            "astra_reversal.image_perturbation_search.ActionSpec.from_environment",
            lambda *args: spec,
        )

        def initialize_noise(*args):
            search.known = np.zeros((1, policy.horizon, policy.action_dim), np.float32)
            search.recovered = search.known.copy()
            search.report["initialization"] = {
                "passed": True,
                "velocity_evaluations": 1230,
            }

        def check_images(*args):
            search.image_checked = True
            search.report["image_gate"] = {"passed": True, "velocity_evaluations": 60}

        monkeypatch.setattr(search, "initialize_noise", initialize_noise)
        monkeypatch.setattr(search, "check_images", check_images)
        failed = (
            {
                (f"{arm}_{iteration}", index)
                for arm in ("astra_occlusion", "astra_demo_blend")
                for iteration in (2, 3)
                for index in (1, 2)
            }
            if all_provider_fail
            else {("astra_occlusion_2", 2)}
        )
        opener = AccountingOpener(fail=failed)

        class RecordedClient(ImagePerturbationClient):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.opener = opener

        monkeypatch.setattr(
            image_perturbation_agent, "ImagePerturbationClient", RecordedClient
        )
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
            mode, iteration = Path(video_path).stem.rsplit("_", 1)
            iteration = int(iteration)
            rollouts.append((mode, iteration))
            success = (mode == "recovered_noise" and baseline_success) or (
                mode.startswith("astra_")
                and iteration == (3 if mode == "astra_occlusion" else 2)
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
                "snapshots": [
                    snapshot(0, value=iteration),
                    snapshot(action_count, value=action_count + iteration),
                ],
            }

        monkeypatch.setattr(intervention_rollout, "run_rollout", run_rollout)
        return search, opener.requests, rollouts

    return make


def test_search_own_history_real_ledger_prefix_and_physical_dedup(image_search_factory):
    search, requests, rollouts = image_search_factory()
    report = search.run()
    assert report["status"] == "complete"
    assert len(rollouts) == 12
    for request in requests:
        arm, iteration = request["attempt_id"].rsplit("_", 1)
        prior = "recovered_noise_1" if iteration == "2" else f"{arm}_2"
        assert request["previous_attempt"]["feedback"]["attempt_id"] == prior
        assert [row["attempt_id"] for row in request["completed_rollout_feedback"]] == (
            ["recovered_noise_1"]
            if iteration == "2"
            else ["recovered_noise_1", f"{arm}_2"]
        )
        assert all(
            row["proposal"]["attempt_id"] == prior
            for row in request["previous_attempt"]["decisions"]
            if row["accepted"]
        )
    cost = report["physical_cost"]
    assert cost["rollouts"] == 12 and cost["simulated_actions"] == 320
    assert cost["rollout_velocity_evaluations"] == 640
    assert cost["velocity_evaluations"] == 1930
    assert cost["token_usage"]["provider_calls"] == 6
    assert cost["token_usage"]["failed_calls"] == 1
    assert cost["token_usage"]["tokens"]["total_tokens"]["sum"] == 78
    summary = report["arms"]["astra_occlusion"]["summary"]
    assert (
        summary["first_success_attempt"] == 3
        and summary["full_rollout_revisions_to_success"] == 2
    )
    assert summary["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 52
    assert summary["standalone_velocity_evaluations_through_success_or_cap"] == 1470
    first = report["arms"]["astra_occlusion"]["attempts"][1]
    assert (
        first["actions_with_accepted_decision"]
        == first["actions_with_changed_image"]
        == 25
    )
    assert first["native_condition_fallback_actions"] == 5
    generations = [
        json.loads(line)
        for line in (search.directory / "events.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "image_generation"
    ]
    assert all(row["instruction"] == search.entry["instruction"] for row in generations)
    recovered_hashes = {
        row["latent"]["sha256"]
        for row in generations
        if row["mode"] in ("recovered_noise", "astra_occlusion", "astra_demo_blend")
    }
    assert len(recovered_hashes) == 1


@pytest.mark.parametrize("development", [False, True])
def test_baseline_success_stops_evaluation_but_dev_extra_cost_stays_physical(
    image_search_factory, development
):
    search, requests, rollouts = image_search_factory(
        baseline_success=True, development=development
    )
    report = search.run()
    assert len(rollouts) == 3 + 5 * int(development)
    assert len(requests) == 4 * int(development)
    for row in report["arms"].values():
        summary = row["summary"]
        assert summary["first_success_attempt"] == 1
        assert summary["full_rollout_revisions_to_success"] == 0
        assert summary["provider_through_success_or_cap"]["provider_calls"] == 0
        assert summary["tokens_to_first_success"]["tokens"]["total_tokens"]["sum"] == 0
        assert summary["development_extra_rollouts_after_success"] == int(development)
    assert report["physical_cost"]["token_usage"]["tokens"]["total_tokens"][
        "sum"
    ] == 52 * int(development)
    assert report["physical_cost"]["velocity_evaluations"] == 1350 + 300 * int(
        development
    )
    if development:
        assert report["development_validation"]["passed"]


def test_development_all_errors_preserves_billed_failures_and_fails_gate(
    image_search_factory,
):
    search, requests, _ = image_search_factory(
        baseline_success=True, development=True, all_provider_fail=True
    )
    with pytest.raises(ValueError, match="accepted-and-executed"):
        search.run()
    report = json.loads((search.directory / "summary.json").read_text())
    assert report["status"] != "complete"
    assert len(requests) == 4
    assert not report["development_validation"]["passed"]
    assert report["physical_cost"]["token_usage"]["failed_calls"] == 4
    assert report["physical_cost"]["token_usage"]["tokens"]["total_tokens"]["sum"] == 52
    for arm in ("astra_occlusion", "astra_demo_blend"):
        assert (
            report["arms"][arm]["attempts"][-1]["native_condition_fallback_actions"]
            == 30
        )


def test_physical_cost_refuses_duplicate_rollout_identity(image_search_factory):
    search, _, _ = image_search_factory(baseline_success=True)
    search.run()
    search.report["controls"]["policy_fresh"]["attempt_id"] = "known_noise_1"
    with pytest.raises(ValueError, match="double count"):
        search.physical_cost()
