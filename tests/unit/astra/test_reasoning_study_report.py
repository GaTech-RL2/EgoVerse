import json

import pytest

from astra_reversal.reasoning_learning.study_report import summarize_run, wilson


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


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
