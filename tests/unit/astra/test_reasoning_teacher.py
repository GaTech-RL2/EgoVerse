import numpy as np
import pytest

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.codex_executor import _module
from astra_reversal.codex_relay import _proposal
from astra_reversal.reasoning_learning import teacher


def request(role="diagnose"):
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
