import copy
import json

import pytest

from astra_reversal.reasoning_learning.audit_intervention_credit import collection


def fixture(directory):
    actual = [
        {
            "episode_id": "e",
            "step": i,
            "observation_id": "o",
            "action": [0.0] * 7,
            "executed": True,
            "policy_version": 0,
            "batch_id": f"b{i // 5}",
            "event_id": "correction",
            "preference": "win" if i < 5 else None,
            "stage": "correction" if i < 5 else "continuation",
            "evidence": "ambiguous",
            "environment_success": False,
            "terminated": False,
            "after_observation_sha256": "after",
        }
        for i in range(10)
    ]
    labels = copy.deepcopy(actual)
    for r in labels:
        r.update(stage="correction", evidence="observed_useful")
    # The local reviewer calls native recovery a correction phase. That phase
    # alone is not evidence that the teacher modified a motor command.
    (directory / "executed_steps.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in actual)
    )
    (directory / "admission.jsonl").write_text(
        json.dumps({"steps": labels, "windows": []}) + "\n"
    )
    (directory / "result.json").write_text(
        json.dumps(
            {
                "episode_id": "e",
                "success": False,
                "actions_executed": 10,
                "initialization_steps": 10,
                "total_control_steps": 20,
                "admitted_windows": 0,
                "assisted_chunks": 1,
            }
        )
    )
    (directory / "outcomes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "start_step": i,
                    "end_step": i + 5,
                    "review": {"outcome": "observed_useful"},
                }
            )
            + "\n"
            for i in (0, 5)
        )
    )
    (directory / "decisions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "step": i,
                    "batch_id": f"b{i // 5}",
                    "selected": "guided" if i == 0 else "native",
                    "judgments": {"guided": {"preference": "win"}},
                }
            )
            + "\n"
            for i in (0, 5)
        )
    )


def test_phase_labels_do_not_inflate_actual_assistance_or_training_coverage(tmp_path):
    fixture(tmp_path)
    row = collection(tmp_path)
    assert (
        row["assisted_actions_executed"] == row["assisted_actions_locally_useful"] == 5
    )
    assert row["assisted_actions_in_complete_training_windows"] == 0
    assert row["assisted_prefixes_executed"] == row["assisted_prefixes_reviewed"] == 1
    assert row["assisted_prefix_teacher_outcomes"] == {"observed_useful": 1}


@pytest.mark.parametrize("change", ["selected", "batch_id"])
def test_review_cannot_refer_to_another_executed_proposal(tmp_path, change):
    fixture(tmp_path)
    path = tmp_path / "decisions.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[0][change] = "native" if change == "selected" else "other"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="executed proposal"):
        collection(tmp_path)
