import copy

import pytest

from astra_reversal.reasoning_learning.evaluation_checkpoint import validate
from astra_reversal.reasoning_learning.native_reference import SIGNATURE_FIELDS


def fixture():
    task = {"suite": "libero_goal_ood", "task_id": 6}
    protocol = {
        "pilot": {
            "autonomous_evaluation_reset_indices": [20, 21],
            "evaluation_after_collection_rollouts": [0, 2, 4, 8],
        }
    }
    episodes = [{"total_control_steps": 100}] * 4
    metadata = {k: None for k in SIGNATURE_FIELDS}
    source = {
        "source_workflow": "original-worker",
        "source_task": task,
        "source_seed": 173,
        "source_protocol": protocol,
        "source_policy_signature": metadata,
        "episodes": episodes,
        "evaluation_reset_indices": [20, 21],
        "policy_version": 4,
        "adapter_parameter_sha256": "parameters",
        "files": {"adapters.pt": "checkpoint"},
        "source_collection_control_steps": 400,
        "source_teacher_total_tokens": 1000,
        "source_evaluation_control_steps_retained": 500,
    }
    values = {
        "retained_run.json": {
            "workflow": "original-worker",
            "task": task,
            "seed": 173,
            "latest_updated_policy_version": 4,
            "latest_update_has_autonomous_evaluation": False,
            "collection_steps_retained": 400,
            "teacher_usage": {"total_tokens": 1000},
            "evaluation_steps_retained": 500,
        },
        "runtime.json": {"task": task, "seed": 173},
        "protocol.json": protocol,
        "updates.json": [
            {
                "policy_version": 4,
                "after_sha256": "parameters",
                "checkpoint": {"sha256": "checkpoint"},
            }
        ],
        "collection.json": episodes,
        "checkpoint.json": metadata,
        "preflight.json": {
            "zero_adapter_max_abs": 0,
            "native_zero_guidance_max_abs": 0,
        },
    }
    return copy.deepcopy(source), copy.deepcopy(values)


def test_resume_requires_the_actual_latest_unevaluated_checkpoint():
    source, values = fixture()
    validate(source, values)
    source["policy_version"] = 2
    with pytest.raises(ValueError, match="latest saved"):
        validate(source, values)
    source["policy_version"] = 4
    values["retained_run.json"]["latest_update_has_autonomous_evaluation"] = True
    with pytest.raises(ValueError, match="latest saved"):
        validate(source, values)


@pytest.mark.parametrize(
    "key",
    [
        "source_collection_control_steps",
        "source_teacher_total_tokens",
        "source_evaluation_control_steps_retained",
    ],
)
def test_evaluation_restart_cannot_erase_original_costs(key):
    source, values = fixture()
    source[key] = 0
    with pytest.raises(ValueError, match="costs differ"):
        validate(source, values)


def test_evaluation_restart_cannot_select_a_subset_of_resets():
    source, values = fixture()
    source["evaluation_reset_indices"] = [21]
    with pytest.raises(ValueError, match="schedule differs"):
        validate(source, values)
