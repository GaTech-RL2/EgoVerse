"""Task-level measured SR, search cost and timing; never fill missing usage."""

import csv
from pathlib import Path

from .codex_accounting import summarize_codex_calls
from .demo_segments import write_json


def metrics(report):
    if report["status"] != "complete":
        raise ValueError("Primary results require a complete paired evaluation")
    tasks = []
    providers = report["provider_records"]
    rollouts = report["physical_rollouts"]
    for task_key, task in report["tasks"].items():
        suite, task_id = task_key.rsplit("_task", 1)
        cohort = [
            r for r in rollouts if r["suite"] == suite and r["task_id"] == int(task_id)
        ]
        evaluation = [r for r in cohort if r["split"] == "evaluation"]
        native = [r for r in evaluation if r["attempt_id"].endswith("_native")]
        selected = [r for r in evaluation if r["attempt_id"].endswith("_composed")]
        if len(native) != len(selected) or len(native) != len(
            report["protocol"]["evaluation_states"]
        ):
            raise ValueError("Missing paired evaluation trials")
        start, end = task["provider_record_start"], task["provider_record_end"]
        usage = summarize_codex_calls(providers[start:end])
        first_end = task["first_success_provider_record_end"]
        first_usage = (
            summarize_codex_calls(providers[start:first_end])
            if first_end is not None
            else None
        )
        total = usage["tokens"]["total_tokens"]
        row = {
            "task": task_key,
            "arm": report["arm"],
            "native_successes": sum(r["success"] for r in native),
            "native_trials": len(native),
            "selected_successes": sum(r["success"] for r in selected),
            "selected_trials": len(selected),
            "native_sr": sum(r["success"] for r in native) / len(native),
            "selected_sr": sum(r["success"] for r in selected) / len(selected),
            "validation_successes": task["validation_successes"],
            "candidate_programs_tested": len(task["candidates"]),
            "physical_rollouts": len(cohort),
            "first_success_candidate": task["first_success_candidate"],
            "first_success_censored": task["first_success_censored"],
            "search_codex_jobs": usage["codex_jobs"],
            "search_known_total_tokens": total["sum"],
            "search_total_tokens_complete": total["complete"],
            "search_amortized_tokens_per_evaluation_rollout": total["sum"]
            / len(selected)
            if total["complete"]
            else None,
            "search_provider_seconds": usage["latency_seconds"],
            "evaluation_codex_jobs": 0,
            "native_mean_rollout_seconds": sum(r["wall_seconds"] for r in native)
            / len(native),
            "selected_mean_rollout_seconds": sum(r["wall_seconds"] for r in selected)
            / len(selected),
            "native_mean_policy_seconds": sum(r["policy_seconds"] for r in native)
            / len(native),
            "selected_mean_policy_seconds": sum(r["policy_seconds"] for r in selected)
            / len(selected),
            "search_usage": usage,
            "search_to_first_success_usage": first_usage,
        }
        row["sr_difference_percentage_points"] = 100 * (
            row["selected_sr"] - row["native_sr"]
        )
        tasks.append(row)
    if not tasks:
        raise ValueError("A completed study must contain tasks")
    native_successes = sum(t["native_successes"] for t in tasks)
    selected_successes = sum(t["selected_successes"] for t in tasks)
    trials = sum(t["selected_trials"] for t in tasks)
    return {
        "arm": report["arm"],
        "status": "complete",
        "tasks": tasks,
        "native_successes": native_successes,
        "selected_successes": selected_successes,
        "paired_trials": trials,
        "native_sr": native_successes / trials,
        "selected_sr": selected_successes / trials,
        "sr_difference_percentage_points": 100
        * (selected_successes - native_successes)
        / trials,
        "search_usage": summarize_codex_calls(providers),
        "scope": "two-task cached-source pilot; not the all-20 study"
        if report.get("pilot")
        else report["protocol"]["evaluation_scope"],
        "cost_scope": "online search only; offline demonstration curation is separate and not assumed free",
    }


def write_metrics(report, directory):
    value = metrics(report)
    directory = Path(directory)
    write_json(directory / "metrics.json", value)
    rows = [
        {
            k: v
            for k, v in row.items()
            if k not in ("search_usage", "search_to_first_success_usage")
        }
        for row in value["tasks"]
    ]
    # The full, potentially incomplete token accounting is retained in JSON.
    with (directory / "task_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return value
