"""Observation binding, arm isolation and protected instruction token alignment."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import codex_executor, codex_relay
from astra_reversal.complex_manipulation import representation_teacher as teacher
from astra_reversal.complex_manipulation.language_guidance import LanguageGuide
from astra_reversal.complex_manipulation.text_slots import alignment_indices


def observation():
    return {
        "prompt": "Put the food container in the fridge and close the door.",
        "observation/state": np.arange(16, dtype=np.float64),
        **{
            k: np.full((8, 8, 3), 30 + i, np.uint8)
            for i, k in enumerate(teacher.CAMERAS.values())
        },
    }


def request(method="tei"):
    obs = observation()
    return teacher.build_request(
        episode_id="synthetic:matched:tei:attempt0",
        request_index=0,
        step=0,
        snapshots=[teacher.live_snapshot(obs, 0)],
        context={
            "original_instruction": obs["prompt"],
            "remaining_calls_including_this": 16,
            "allowed_intervention": method,
        },
    )


def response(method="tei"):
    return {
        "method": method,
        "alpha": 0.25,
        "subgoal": "Lift the held container above the fridge shelf lip.",
        "observed_evidence": "The container is below the shelf edge.",
        "completion_signal": "The container clears the shelf lip.",
        "next_review_controls": 100,
    }


@pytest.mark.parametrize("method", ["tei", "tli"])
def test_teacher_payload_arm_binding_and_native_escape(method):
    req = request(method)
    assert codex_executor._module(req) is teacher
    assert codex_relay._module("complex_representation") is teacher
    payload = teacher.build_payload(req, "gpt-6-astra")
    assert sum(p["type"] == "image_url" for p in payload["messages"][1]["content"]) == 3
    proposal = teacher.parse_proposal(response(method), req)
    assert (
        codex_relay._proposal(proposal, req, teacher, "complex_representation")
        == proposal
    )
    native = {**response(), "method": "native", "alpha": 0, "subgoal": ""}
    assert teacher.parse_proposal(native, req)["method"] == "native"
    with pytest.raises(ValueError, match="Unknown"):
        teacher.parse_proposal(response("tli" if method == "tei" else "tei"), req)


@pytest.mark.parametrize(
    "change",
    [
        {"alpha": True},
        {"alpha": float("nan")},
        {"alpha": 0.3},
        {"subgoal": ""},
        {"next_review_controls": True},
        {"method": "phase_prompt"},
        {"method": "native"},
        {"motor_commands": [0] * 12},
    ],
)
def test_invalid_choices_cannot_reach_model(change):
    with pytest.raises(ValueError):
        teacher.parse_proposal({**response(), **change}, request())


def test_scheduler_preserves_original_prompt_state_and_images(tmp_path):
    def propose(req):
        return teacher.parse_proposal(response(), req)

    scheduler = LanguageGuide(
        client=SimpleNamespace(propose=propose, records=[]),
        output=tmp_path,
        teacher_module=teacher,
        representation_method="tei",
        baseline={
            "frames": [],
            "episode_id": "native",
            "executed_controls": 1500,
            "stop_reason": "horizon",
            "video_sha256": "known-video",
        },
    )
    obs = observation()
    before = copy.deepcopy(obs)
    first, metadata = scheduler.prepare(obs, step=0, episode_id="same-reset")
    assert first["prompt"] == before["prompt"]
    assert metadata["intervention"]["alpha"] == 0.25
    for key in ("observation/state", *teacher.CAMERAS.values()):
        assert first[key] is obs[key]
        np.testing.assert_array_equal(first[key], before[key])
    for step in (5, 50, 95):
        _, next_metadata = scheduler.prepare(obs, step=step, episode_id="same-reset")
        assert next_metadata["intervention"] == metadata["intervention"]
    assert scheduler.calls == 1
    scheduler.calls = 16
    _, expired = scheduler.prepare(obs, step=100, episode_id="same-reset")
    assert expired["intervention"] is None
    assert not expired["assisted"]


def test_alignment_only_reads_and_writes_instruction_positions():
    source = np.array([False, True, False, True, False])
    target = np.array([False, False, True, True, True])
    indices, valid = alignment_indices(source, target)
    assert indices.tolist() == [0, 0, 1, 3, 0]
    assert valid.tolist() == [False, False, True, True, False]
    assert not (valid & ~target).any()
    assert source[indices[valid]].all()
