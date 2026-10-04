import json

import pytest

from astra_reversal.reasoning_learning.study_report import (
    paired_native_comparison,
    summarize_run,
    teacher_decisions,
    usage_by_episode,
    wilson,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_predicted_wins_are_not_observed_success_or_training_admission(tmp_path):
    path = tmp_path / "collection/rollout_0/teacher_responses.jsonl"
    path.parent.mkdir(parents=True)
    compare = {
        "decision_id": "batch",
        "role": "compare",
        "judgments": [{"preference": "win"}, {"preference": "uncertain"}],
        "selected": "candidate_a",
    }
    rows = [
        {"decision_id": "diagnosis", "role": "diagnose", "intervene": True},
        compare,
        compare,
        {"decision_id": "review", "role": "assess", "outcome": "failed"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = teacher_decisions(tmp_path)
    assert result["requested_interventions"] == 1
    assert result["roles"]["compare"] == 1
    assert result["candidate_preferences"] == {"win": 1, "uncertain": 1}
    assert result["selected_non_native_proposals"] == 1
    assert result["observed_prefix_outcomes"] == {"failed": 1}
    rows.append({**compare, "selected": "native"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="Conflicting"):
        teacher_decisions(tmp_path)


def test_missing_evaluation_and_partial_collection_are_not_scored_as_failures(tmp_path):
    path = tmp_path / "collection/rollout_0"
    path.mkdir(parents=True)
    (path / "executed_steps.jsonl").write_text("{}\n" * 3)
    result = summarize_run(
        {
            "directory": str(tmp_path),
            "label": "pilot",
            "method": "teacher",
            "workflow": "test",
        }
    )
    assert result["points"] == []
    assert result["completed_collection_rollouts"] == 0
    assert result["collected_successes"] == 0
    assert result["collection_steps_retained"] == 13
    assert result["collection_steps_may_be_lower_bound"]
    assert result["threshold_crossing"] is None


def test_report_rejects_incomplete_scheduled_evaluation(tmp_path):
    write(
        tmp_path / "protocol.json",
        {"pilot": {"autonomous_evaluation_reset_indices": [20, 21]}},
    )
    write(
        tmp_path / "learning_curve.json",
        [
            {
                "episodes": [{"episode_id": "e", "success": True}],
                "rollouts": 1,
                "successes": 1,
            }
        ],
    )
    with pytest.raises(ValueError, match="Incomplete scheduled"):
        summarize_run(
            {
                "directory": str(tmp_path),
                "label": "pilot",
                "method": "teacher",
                "workflow": "test",
            }
        )


def test_report_counts_local_completion_even_without_worker_receipt(tmp_path):
    jobs = tmp_path / "jobs/one"
    write(jobs / "request.json", {"episode_id": "e"})
    write(
        jobs / "completed.json",
        {
            "result": {
                "receipt": {
                    "token_usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                    "raw_usage": {"cached_input_tokens": 80},
                    "latency_seconds": 2,
                }
            }
        },
    )
    result = summarize_run(
        {
            "directory": str(tmp_path / "no_worker_artifacts"),
            "teacher_jobs": str(jobs.parent),
            "label": "pilot",
            "method": "teacher",
            "workflow": "test",
        }
    )
    assert result["teacher_usage"]["total_tokens"] == 110
    assert result["teacher_usage"]["uncached_input_tokens"] == 20
    assert result["teacher_usage"]["completed_calls"] == 1


def test_wilson_interval_is_not_an_evaluation_denominator_or_certification():
    assert wilson(5, 10) == pytest.approx([0.2365930905, 0.7634069095])
    assert wilson(10, 10)[0] < 0.8
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_reused_teacher_data_is_attributed_without_double_charging_new_usage(tmp_path):
    source = {
        "source_teacher_total_tokens": 100,
        "source_collection_control_steps": 620,
    }
    write(tmp_path / "replayed_training_data.json", source)
    point = {
        "policy_version": 1,
        "collection_rollouts": 2,
        "collection_steps": 620,
        "successes": 0,
        "rollouts": 1,
        "teacher_tokens_from_reused_data": 100,
        "episodes": [
            {
                "episode_id": "evaluation",
                "success": False,
                "actions_executed": 300,
                "total_control_steps": 310,
            }
        ],
    }
    write(tmp_path / "learning_curve.json", [point])
    spec = {
        "directory": str(tmp_path),
        "label": "replay",
        "method": "teacher_replay",
        "workflow": "test",
    }
    result = summarize_run(spec)
    assert (
        result["collection_steps_retained"]
        == result["teacher_usage"]["total_tokens"]
        == 0
    )
    assert result["points"][0]["teacher_total_tokens"] == 100
    assert result["points"][0]["collection_steps"] == 620
    point["teacher_tokens_from_reused_data"] = 0
    write(tmp_path / "learning_curve.json", [point])
    with pytest.raises(ValueError, match="Reused-data checkpoint costs"):
        summarize_run(spec)


def test_failed_offline_provider_call_keeps_unknown_usage_explicit(tmp_path):
    receipt = tmp_path / "failed.json"
    write(
        receipt,
        {"result": {"receipt": {"provider_unavailable": True, "token_usage": None}}},
    )
    result = usage_by_episode(tmp_path, receipt_files=[receipt])["offline_diagnostic"]
    assert result["completed_calls"] == result["failed_provider_calls"] == 1
    assert result["incomplete_usage_receipts"] == 1
    assert result["total_tokens"] == 0  # Known lower bound, never a known zero bill.


def test_initial_score_does_not_evaluate_a_later_update(tmp_path):
    write(tmp_path / "updates.json", [{"policy_version": 1}])
    write(
        tmp_path / "learning_curve.json",
        [
            {
                "policy_version": 0,
                "collection_rollouts": 0,
                "collection_steps": 0,
                "rollouts": 1,
                "successes": 1,
                "episodes": [
                    {
                        "episode_id": "native",
                        "success": True,
                        "actions_executed": 5,
                        "total_control_steps": 15,
                    }
                ],
            }
        ],
    )
    result = summarize_run(
        {
            "directory": str(tmp_path),
            "label": "pilot",
            "method": "teacher",
            "workflow": "test",
        }
    )
    assert result["latest_evaluated_policy_version"] == 0
    assert result["latest_updated_policy_version"] == 1
    assert not result["latest_update_has_autonomous_evaluation"]


def test_native_comparison_distinguishes_equal_scores_with_different_successes():
    reset = {
        "reset_state_sha256": "state",
        "reset_model_sha256": "scene",
        "bddl_sha256": "task",
    }
    reference = {
        "successes": 1,
        "rollouts": 2,
        "success_rate": 0.5,
        "episode_results": [
            {"episode_id": "a", "success": True, "reset_identity": reset},
            {"episode_id": "b", "success": False, "reset_identity": reset},
        ],
    }
    point = {
        **reference,
        "episode_results": [
            {**r, "success": not r["success"]} for r in reference["episode_results"]
        ],
    }
    result = paired_native_comparison(point, reference)
    assert result["delta_success_rate"] == 0
    assert result["gained_episode_ids"] == ["b"]
    assert result["regressed_episode_ids"] == ["a"]
    point["episode_results"][0]["reset_identity"] = {
        **reset,
        "bddl_sha256": "different task",
    }
    with pytest.raises(ValueError, match="reset scenes"):
        paired_native_comparison(point, reference)
