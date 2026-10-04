import copy
import json

import pytest

from astra_reversal.reasoning_learning.native_reference import (
    SIGNATURE_FIELDS,
    reuse_native_evaluation,
)
from astra_reversal.records import file_sha256


def fixture(tmp_path):
    metadata = {key: "same-value" for key in SIGNATURE_FIELDS}
    entry = {
        "initial_state_id": 20,
        "episode_id": "e",
        "reset_state_sha256": "state",
        "reset_model_sha256": "model",
        "bddl_sha256": "bddl",
    }
    protocol = {
        "checkpoint": {"weights": "same"},
        "environment": {"actions": 300},
        "pilot": {"autonomous_evaluation_reset_indices": [20]},
    }
    kwargs = {
        "protocol": protocol,
        "metadata": metadata,
        "preflight": {"zero_adapter_max_abs": 0, "native_zero_guidance_max_abs": 0},
        "task": {"task_id": 6},
        "seed": 173,
        "manifest": {"episodes": [entry]},
    }
    reference = {
        "task": kwargs["task"],
        "seed": 173,
        "workflow": "measured-original",
        "policy_signature": metadata,
        "checkpoint": protocol["checkpoint"],
        "environment": protocol["environment"],
        "point": {
            "policy_version": 0,
            "collection_rollouts": 0,
            "collection_steps": 0,
            "rollouts": 1,
            "successes": 1,
            "episodes": [{"episode_id": "e", "success": True, "reset_audit": entry}],
            "evaluation_steps_cumulative": 100,
        },
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    return path, file_sha256(path), kwargs


def test_shared_native_reference_never_creates_new_samples(tmp_path):
    path, checksum, kwargs = fixture(tmp_path)
    result = reuse_native_evaluation(path, checksum, **kwargs)
    assert result["successes"] == 1
    assert (
        result["evaluation_steps_cumulative"]
        == result["new_evaluation_interactions"]
        == 0
    )
    assert result["reference_evaluation_steps"] == 100
    assert result["independent_new_evaluation_replicate"] is False


@pytest.mark.parametrize("change", ["weights", "reset", "adapter", "seed"])
def test_identity_changes_forbid_reusing_a_native_score(tmp_path, change):
    path, checksum, original = fixture(tmp_path)
    kwargs = copy.deepcopy(original)
    if change == "weights":
        kwargs["metadata"]["artifact_sha256"] = "different"
    elif change == "reset":
        kwargs["manifest"]["episodes"][0]["reset_model_sha256"] = "different"
    elif change == "adapter":
        kwargs["preflight"]["zero_adapter_max_abs"] = 1e-8
    else:
        kwargs["seed"] = 179
    with pytest.raises(ValueError):
        reuse_native_evaluation(path, checksum, **kwargs)
