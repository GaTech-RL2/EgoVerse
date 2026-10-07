"""Guidance preserves sensors/reset identity and cannot hide failed teacher calls."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import codex_executor, codex_relay
from astra_reversal.astra_client import ClientError
from astra_reversal.complex_manipulation import language_teacher as teacher
from astra_reversal.complex_manipulation.language_guidance import (
    LanguageGuide,
    check_reset,
)


def observation():
    return {
        "prompt": "Place the tupperware on the top shelf, then close the fridge.",
        "observation/state": np.arange(16, dtype=np.float32),
        **{
            key: np.full((8, 8, 3), 20 + i, np.uint8)
            for i, key in enumerate(teacher.CAMERAS.values())
        },
    }


def request():
    obs = observation()
    return teacher.build_request(
        episode_id="synthetic:matched:attempt1",
        request_index=0,
        step=0,
        snapshots=[teacher.live_snapshot(obs, 0)],
        context={
            "original_instruction": obs["prompt"],
            "remaining_calls_including_this": 16,
        },
    )


def response():
    return {
        "method": "phase_prompt",
        "subgoal": "Lift the held container above the shelf lip.",
        "observed_evidence": "The container appears to be below the shelf edge.",
        "completion_signal": "Container clears the shelf lip.",
        "next_review_controls": 50,
    }


def test_all_three_cameras_survive_codex_payload_and_foreign_episode_is_rejected():
    req = request()
    assert codex_executor._module(req) is teacher
    assert codex_relay._module("complex_language") is teacher
    payload = teacher.build_payload(req, "gpt-6-astra")
    images = [
        p["image_url"]["url"]
        for p in payload["messages"][1]["content"]
        if p["type"] == "image_url"
    ]
    assert len(images) == len(set(images)) == 3
    proposal = teacher.parse_proposal(response(), req)
    assert codex_relay._proposal(proposal, req, teacher, "complex_language") == proposal
    proposal["episode_id"] = "another-reset"
    with pytest.raises(ClientError, match="identity differs"):
        codex_relay._proposal(proposal, req, teacher, "complex_language")
    req["snapshots"][0]["images"].pop("right")
    with pytest.raises(ValueError, match="identity differs"):
        teacher.build_payload(req, "gpt-6-astra")


@pytest.mark.parametrize(
    "change",
    [
        {"method": "native"},
        {"subgoal": " "},
        {"next_review_controls": True},
        {"next_review_controls": 51},
        {"subgoal": "x" * 161},
        {"motor_commands": [0] * 12},
    ],
)
def test_illegal_intervention_or_review_cannot_be_applied(change):
    with pytest.raises(ValueError):
        teacher.parse_proposal({**response(), **change}, request())


def test_seed_equality_is_insufficient_for_a_matched_reset():
    expected = {
        "seed": 0,
        "instruction": "goal",
        "observation_sha256": "image-a",
        "state_sha256": "state-a",
        "model_sha256": "xml-a",
        "horizon": 1500,
        "execution_prefix": 5,
        "initial_success": False,
    }
    assert check_reset(expected, expected)["matched"]
    for key in ("observation_sha256", "state_sha256", "model_sha256", "instruction"):
        with pytest.raises(ValueError, match=key):
            check_reset({**expected, key: "different"}, expected)


def guide(tmp_path, propose):
    return LanguageGuide(
        client=SimpleNamespace(propose=propose, records=[]),
        baseline={
            "frames": [],
            "episode_id": "native-reset",
            "executed_controls": 1500,
            "stop_reason": "horizon",
            "video_sha256": "video-sha",
        },
        output=tmp_path,
    )


def test_phase_persists_without_extra_calls_then_expires_at_budget(tmp_path):
    requests = []

    def propose(req):
        requests.append(req)
        return teacher.parse_proposal(response(), req)

    scheduler = guide(tmp_path, propose)
    obs = observation()
    original = copy.deepcopy(obs)
    first, metadata = scheduler.prepare(obs, step=0, episode_id="case")
    assert first["prompt"].startswith(original["prompt"])
    assert metadata["assisted"]
    for key in teacher.CAMERAS.values():
        assert first[key] is obs[key]
        np.testing.assert_array_equal(first[key], original[key])
    for step in range(5, 50, 5):
        assert (
            scheduler.prepare(obs, step=step, episode_id="case")[0]["prompt"]
            == first["prompt"]
        )
    assert len(requests) == 1
    for step in range(50, 800, 50):
        scheduler.prepare(obs, step=step, episode_id="case")
    assert len(requests) == 16
    assert requests[-1]["context"]["remaining_calls_including_this"] == 1
    final, metadata = scheduler.prepare(obs, step=800, episode_id="case")
    assert final["prompt"] == original["prompt"]
    assert metadata["budget_exhausted"] and not metadata["assisted"]
    assert obs["prompt"] == original["prompt"]


def test_provider_outage_is_charged_and_propagates_before_policy_execution(tmp_path):
    def unavailable(req):
        raise ClientError("provider unavailable")

    scheduler = guide(tmp_path, unavailable)
    with pytest.raises(ClientError, match="provider unavailable"):
        scheduler.prepare(observation(), step=0, episode_id="case")
    assert scheduler.calls == 1
    assert (tmp_path / "request_00.json").exists()
    assert not (tmp_path / "proposal_00.json").exists()
    assert scheduler.active is None


def test_history_must_not_claim_unrecorded_proprioception():
    obs = observation()
    historic = {
        "origin": "prior_native_failure",
        "step": 1500,
        "state": [0] * 16,
        "images": {
            "left_right_wrist_panorama": teacher.png_wire(obs["observation/image"])
        },
    }
    with pytest.raises(ValueError, match="does not supply robot state"):
        teacher.build_request(
            episode_id="case",
            request_index=0,
            step=0,
            snapshots=[historic, teacher.live_snapshot(obs, 0)],
            context={
                "original_instruction": obs["prompt"],
                "remaining_calls_including_this": 16,
            },
        )
