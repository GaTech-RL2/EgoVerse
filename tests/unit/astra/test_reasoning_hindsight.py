import copy

import numpy as np
import pytest

from astra_reversal.codex_executor import _module
from astra_reversal.demo_skill_agent import png_wire
from astra_reversal.reasoning_learning import hindsight
from astra_reversal.records import digest


def request():
    image = png_wire(np.zeros((16, 16, 3), dtype=np.uint8))
    return hindsight.build_request(
        workflow="owned-collection",
        episode_id="libero_goal_ood:seed173:task6:state1",
        instruction="put the wine bottle in the bowl",
        frames=[
            {"step": step, "state": [0.0] * 8, "images": [image, image]}
            for step in (0, 10, 16)
        ],
        segments=[
            {
                "segment_id": 0,
                "start_step": 0,
                "end_step_exclusive": 10,
                "actions": [[0.0] * 7] * 10,
            },
            {
                "segment_id": 1,
                "start_step": 10,
                "end_step_exclusive": 16,
                "actions": [[0.0] * 7] * 6,
            },
        ],
        success=True,
    )


def resign(value):
    value.pop("request_fingerprint")
    value["request_fingerprint"] = digest(value)
    return value


def proposal():
    return {
        "episode_assessment": "The real trajectory completes the placement.",
        "segments": [
            {
                "segment_id": i,
                "outcome": "observed_useful",
                "confidence": 0.9,
                "phase": "placement",
                "evidence": "Visible movement into the bowl.",
            }
            for i in range(2)
        ],
    }


def test_hindsight_binds_one_completed_trajectory_and_original_images():
    value = request()
    assert _module(value) is hindsight
    payload = hindsight.build_payload(
        value, "gpt-6-astra", sampling={"reasoning_effort": "medium"}
    )
    images = [p for p in payload["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 6
    assert (
        hindsight.parse_proposal(proposal(), value)["request_fingerprint"]
        == value["request_fingerprint"]
    )
    assert payload["reasoning_effort"] == "medium"


@pytest.mark.parametrize(
    "case",
    [
        "gap",
        "future_frame",
        "incomplete_actions",
        "extra_object_poses",
        "missing_camera",
        "boolean_index",
    ],
)
def test_hindsight_rejects_unbound_or_unobserved_data(case):
    value = request()
    if case == "gap":
        value["segments"][1]["start_step"] = 11
    elif case == "future_frame":
        value["frames"][-1]["step"] = 17
    elif case == "incomplete_actions":
        value["segments"][1]["actions"].pop()
    elif case == "extra_object_poses":
        value["frames"][0]["object_positions"] = [0, 0, 0]
    elif case == "missing_camera":
        value["frames"][0]["images"].pop()
    else:
        value["segments"][0]["segment_id"] = False
    with pytest.raises(ValueError):
        hindsight._validate_request(resign(value))


def test_hindsight_rejects_changed_success_and_duplicate_credit():
    value = request()
    changed = copy.deepcopy(value)
    changed["environment_success"] = False
    with pytest.raises(ValueError, match="fingerprint"):
        hindsight._validate_request(changed)
    labels = proposal()
    labels["segments"][1]["segment_id"] = 0
    with pytest.raises(ValueError, match="exactly one"):
        hindsight.parse_proposal(labels, value)
