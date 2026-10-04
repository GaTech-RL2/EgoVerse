import copy
import json

import numpy as np
import pytest

from astra_reversal.demo_skill_agent import png_wire
from astra_reversal.reasoning_learning import credit_data, hindsight
from astra_reversal.reasoning_learning.evidence import training_windows
from astra_reversal.records import digest, file_sha256


def source():
    raw = {
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((16, 16, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
    }
    before = {**raw, "prompt": "test task"}
    actual = [
        {
            "episode_id": "e",
            "step": i,
            "observation_id": digest(before),
            "action": [0.0] * 7,
            "executed": True,
            "policy_version": 0,
            "batch_id": str(i // 5),
            "event_id": None,
            "preference": None,
            "stage": "continuation",
            "evidence": "ambiguous",
            "environment_success": i == 15,
            "terminated": False,
            "after_observation_sha256": digest(raw),
        }
        for i in range(16)
    ]
    request = hindsight.build_request(
        workflow="owned",
        episode_id="e",
        instruction="test task",
        success=True,
        frames=[
            {
                "step": i,
                "state": [0.0] * 8,
                "images": [
                    png_wire(raw[k])
                    for k in ("observation/image", "observation/wrist_image")
                ],
            }
            for i in (0, 10, 16)
        ],
        segments=[
            {
                "segment_id": i,
                "start_step": a,
                "end_step_exclusive": b,
                "actions": [r["action"] for r in actual[a:b]],
            }
            for i, (a, b) in enumerate(((0, 10), (10, 16)))
        ],
    )
    proposal = {
        "request_fingerprint": request["request_fingerprint"],
        "episode_assessment": "Successful placement.",
        "segments": [
            {
                "segment_id": i,
                "outcome": "observed_useful",
                "confidence": 0.9,
                "phase": "placement",
                "evidence": "Bottle is visibly lowered into the bowl.",
            }
            for i in range(2)
        ],
    }
    result = {
        "episode_id": "e",
        "success": True,
        "assisted_chunks": 0,
        "actions_executed": 16,
        "admitted_windows": 2,
        "initialization_steps": 10,
        "total_control_steps": 26,
    }
    return before, actual, request, proposal, result


def labels(request, proposal, actual, result):
    return credit_data.hindsight_labels(
        request, proposal, actual, result, instruction="test task", workflow="owned"
    )


def test_credit_keeps_actual_windows_and_never_labels_an_unknown_tail():
    _, actual, request, proposal, result = source()
    rows = labels(request, proposal, actual, result)
    windows = training_windows(rows, horizon=10)
    assert [w["start_step"] for w in windows] == [0, 5]
    assert all(w["end_step_exclusive"] <= 16 for w in windows)
    assert all(r["evidence"] == "ambiguous" for r in actual)
    proposal["segments"][1]["confidence"] = 0.79
    assert (
        len(training_windows(labels(request, proposal, actual, result), horizon=10))
        == 1
    )
    actual[0].update(stage="correction", preference="uncertain")
    assert [
        w["start_step"]
        for w in training_windows(labels(request, proposal, actual, result), horizon=10)
    ] == []


@pytest.mark.parametrize(
    "change", ["action", "state", "image", "success", "final_frame", "response_binding"]
)
def test_credit_rejects_changed_observed_evidence_even_if_request_is_resigned(change):
    _, actual, request, proposal, result = source()
    if change == "action":
        request["segments"][0]["actions"][0][0] = 0.25
    elif change == "state":
        request["frames"][0]["state"][0] = 0.25
    elif change == "image":
        request["frames"][0]["images"][0] = png_wire(
            np.full((16, 16, 3), 50, dtype=np.uint8)
        )
    elif change == "success":
        request["environment_success"] = False
    elif change == "final_frame":
        actual[-1]["after_observation_sha256"] = "bad"
    else:
        proposal["request_fingerprint"] = "wrong"
        with pytest.raises(ValueError, match="another request"):
            labels(request, proposal, actual, result)
        return
    request.pop("request_fingerprint")
    request["request_fingerprint"] = digest(request)
    proposal["request_fingerprint"] = request["request_fingerprint"]
    with pytest.raises(ValueError):
        labels(request, proposal, actual, result)


def test_success_bc_is_separate_evidence_and_requires_unassisted_success():
    _, actual, _, _, result = source()
    windows = credit_data.successful_windows(actual, result)
    assert [w["start_step"] for w in windows] == [0, 5]
    assert all(w["evidence"] == "successful_episode" for w in windows)
    assert credit_data.successful_windows(actual, {**result, "success": False}) == []
    with pytest.raises(ValueError, match="unassisted"):
        credit_data.successful_windows(actual, {**result, "assisted_chunks": 1})


def test_credit_loader_binds_all_files_and_all_source_controls(tmp_path):
    raw, actual, request, proposal, result = source()
    folder = tmp_path / "rollout_0"
    folder.mkdir()
    online = copy.deepcopy(actual)
    for row in online:
        row["evidence"] = "observed_useful"
    (folder / "admission.jsonl").write_text(
        json.dumps({"steps": online, "windows": training_windows(online, horizon=10)})
        + "\n"
    )
    (folder / "executed_steps.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in actual)
    )
    for name, value in (
        ("result", result),
        ("credit_request", request),
        ("credit_proposal", proposal),
    ):
        (folder / (name + ".json")).write_text(json.dumps(value))
    np.savez_compressed(folder / (digest(raw) + ".npz"), **raw)
    manifest = {
        "schema": "reasoning-credit-selection-data-1",
        "source_workflow": "owned",
        "source_protocol": {"teacher": {"frs_action_steering": False}},
        "instruction": "test task",
        "episodes": [{"directory": "rollout_0", "episode_id": "e"}],
        "source_collection_control_steps": 26,
        "variant_window_counts": {k: 2 for k in credit_data.VARIANTS},
        "files": {
            str(p.relative_to(tmp_path)): file_sha256(p) for p in folder.iterdir()
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    variants, observations, _ = credit_data.load(tmp_path, file_sha256(path))
    assert set(variants) == set(credit_data.VARIANTS)
    assert len(observations) == 1
    manifest["source_collection_control_steps"] = 16
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="interactions"):
        credit_data.load(tmp_path)
