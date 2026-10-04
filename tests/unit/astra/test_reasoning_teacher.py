import numpy as np
import pytest

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.codex_executor import _module
from astra_reversal.codex_relay import _proposal
from astra_reversal.reasoning_learning import teacher


def request(role="diagnose", **context):
    observation = {
        "observation/state": np.zeros(8),
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
    }
    return teacher.build_request(
        role=role,
        episode_id="ep",
        request_index=1,
        step=0,
        snapshots=[{"step": 0, "observation": observation}],
        context={
            "native": np.zeros((10, 7)).tolist(),
            "candidates": {"g0": np.zeros((10, 7)).tolist()},
            **context,
        },
    )


def diagnosis():
    return {
        "intervene": True,
        "plan_complete": False,
        "evidence": "Proposed low approach",
        "rule": "Lift before transferring",
        "completion": "Object clears rim",
        "edits": [{"start": 0, "end": 5, "channel": 2, "delta": 0.2}],
    }


def test_codex_dispatch_and_bound_response():
    r = request()
    assert _module(r) is teacher
    p = teacher.parse_proposal(diagnosis(), r)
    assert _proposal(p, r, teacher, "reasoning_learning") == p
    p["observation_step"] = 5
    with pytest.raises(Exception, match="identity"):
        _proposal(p, r, teacher, "reasoning_learning")


def test_both_real_cameras_are_forwarded_with_labels():
    payload = teacher.build_payload(request(), "gpt-6-astra")
    parts = payload["messages"][1]["content"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 2
    assert "No simulator lookahead" in payload["messages"][0]["content"]


@pytest.mark.parametrize("delta", [True, float("nan"), 0.6, 0])
def test_malformed_corrections_are_rejected(delta):
    d = diagnosis()
    d["edits"][0]["delta"] = delta
    with pytest.raises(ValueError):
        teacher.parse_proposal(d, request())


def test_uncertainty_cannot_authorize_execution():
    response = {
        "selected": "g0",
        "judgments": [
            {"candidate_id": "g0", "preference": "uncertain", "evidence": "Occluded"}
        ],
    }
    with pytest.raises(ValueError):
        teacher.parse_proposal(response, request("compare"))
    response["selected"] = "native"
    assert teacher.parse_proposal(response, request("compare"))["selected"] == "native"


def test_controller_target_has_only_declared_edits_and_rejects_out_of_bounds():
    spec = ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {})
    target, mask = teacher.controller_target(
        np.zeros((10, 7)), diagnosis()["edits"], spec
    )
    assert np.count_nonzero(target) == np.count_nonzero(mask) == 5
    assert target[0, 2] == pytest.approx(0.2)
    native = np.zeros((10, 7))
    native[:, 2] = 0.9
    with pytest.raises(ValueError, match="bounds"):
        teacher.controller_target(native, diagnosis()["edits"], spec)


def test_gripper_revision_can_close_without_relaxing_motion_or_hardware_bounds():
    limits = [0.5] * 6 + [2.0]
    spec = ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {})
    d = diagnosis()
    d["edits"] = [{"start": 0, "end": 5, "channel": 6, "delta": 1.5}]
    teacher.parse_proposal(d, request(controller_delta_limits=limits))
    native = np.zeros((10, 7))
    native[:, 6] = -1
    target, mask = teacher.controller_target(
        native, d["edits"], spec, delta_limits=limits
    )
    np.testing.assert_array_equal(target[:5, 6], 0.5)
    np.testing.assert_array_equal(target[5:, 6], -1)
    assert np.count_nonzero(mask) == 5
    with pytest.raises(ValueError, match="Accumulated"):
        teacher.controller_target(native, d["edits"] * 2, spec, delta_limits=limits)
    d["edits"][0]["channel"] = 2
    teacher.parse_proposal(d, request(controller_delta_limits=limits))
    with pytest.raises(ValueError, match="pilot bounds"):
        teacher.controller_target(native, d["edits"], spec, delta_limits=limits)
    d["edits"][0]["channel"] = 6
    with pytest.raises(ValueError, match="bounds"):
        teacher.controller_target(
            np.zeros((10, 7)), d["edits"], spec, delta_limits=limits
        )


def test_explicit_execution_prefix_prevents_edits_to_unexecuted_tail():
    r = request(execution_prefix_steps=5)
    value = diagnosis()
    teacher.parse_proposal(value, r)
    value["edits"][0]["end"] = 6
    with pytest.raises(ValueError, match="outside bounds"):
        teacher.parse_proposal(value, r)
    # Historical requests keep their original ten-action contract.
    teacher.parse_proposal(value, request())
    content = teacher.build_payload(r, "model")["messages"][0]["content"]
    assert teacher.PREFIX_EXTENSION in content
    assert (
        teacher.PREFIX_EXTENSION
        not in teacher.build_payload(request(), "model")["messages"][0]["content"]
    )
    for wrong in (True, 4, 6):
        with pytest.raises(ValueError, match="exactly five"):
            request(execution_prefix_steps=wrong)
