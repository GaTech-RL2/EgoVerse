"""Audit the passage from actual assisted commands to complete training windows."""

import argparse
import json
from collections import Counter
from pathlib import Path

from astra_reversal.records import file_sha256

from .replay_data import json_lines, verify_episode


def collection(directory):
    directory = Path(directory)
    windows, result = verify_episode(directory, horizon=10)
    labels = json_lines(directory / "admission.jsonl")[0]["steps"]
    outcomes = json_lines(directory / "outcomes.jsonl")
    decisions = {r["step"]: r for r in json_lines(directory / "decisions.jsonl")}
    reviewed = []
    for outcome in outcomes:
        decision = decisions[outcome["start_step"]]
        rows = labels[outcome["start_step"] : outcome["end_step"]]
        if not rows or any(r["batch_id"] != decision["batch_id"] for r in rows):
            raise ValueError("Outcome review must cover one executed proposal")
        selected = decision["selected"]
        expected_preference = "win" if selected != "native" else None
        if any(r["preference"] != expected_preference for r in rows):
            raise ValueError("Actual assistance differs from the executed proposal")
        if selected != "native":
            if decision["judgments"][selected]["preference"] != "win":
                raise ValueError("An actual intervention lacked its predicted-win gate")
            reviewed.append({"selected": selected, **outcome["review"]})
    # Native execution can be retrospectively called a "correction" phase by
    # the reviewer. The immutable preference field identifies actual assistance.
    assisted = {r["step"] for r in labels if r["preference"] == "win"}
    batches = {r["batch_id"] for r in labels if r["preference"] == "win"}
    if len(batches) != result["assisted_chunks"]:
        raise ValueError("Actual intervention count differs from the rollout result")
    trained = {
        i for w in windows for i in range(w["start_step"], w["end_step_exclusive"])
    }
    useful = {r["step"] for r in labels if r["evidence"] == "observed_useful"}
    return {
        "episode_id": result["episode_id"],
        "success": result["success"],
        "collection_controls": result["total_control_steps"],
        "assisted_prefixes_executed": result["assisted_chunks"],
        "assisted_prefixes_reviewed": len(reviewed),
        "selected_methods_reviewed": dict(Counter(r["selected"] for r in reviewed)),
        "assisted_prefix_teacher_outcomes": dict(
            Counter(r["outcome"] for r in reviewed)
        ),
        "assisted_actions_executed": len(assisted),
        "assisted_actions_locally_useful": len(assisted & useful),
        "assisted_actions_in_complete_training_windows": len(assisted & trained),
        "all_admitted_windows": len(windows),
        "correction_containing_windows": sum(
            "correction" in w["stages"] for w in windows
        ),
        "source_sha256": {
            name: file_sha256(directory / name)
            for name in (
                "outcomes.jsonl",
                "decisions.jsonl",
                "admission.jsonl",
                "executed_steps.jsonl",
                "result.json",
            )
        },
    }


def audit(manifest):
    runs = []
    for spec in json.loads(Path(manifest).read_text())["runs"]:
        directory = Path(spec["directory"])
        if (
            not spec["method"].startswith("teacher_")
            or spec["method"].startswith("teacher_replay")
        ):
            continue
        rows = [
            collection(p)
            for p in sorted((directory / "collection").glob("rollout_*"))
            if (p / "result.json").exists() and (p / "admission.jsonl").exists()
        ]
        if not rows:
            continue
        numeric = (
            "collection_controls",
            "assisted_prefixes_executed",
            "assisted_prefixes_reviewed",
            "assisted_actions_executed",
            "assisted_actions_locally_useful",
            "assisted_actions_in_complete_training_windows",
            "all_admitted_windows",
            "correction_containing_windows",
        )
        total = {k: sum(r[k] for r in rows) for k in numeric}
        for key in ("selected_methods_reviewed", "assisted_prefix_teacher_outcomes"):
            total[key] = dict(sum((Counter(r[key]) for r in rows), Counter()))
        runtime = json.loads((directory / "runtime.json").read_text())
        runs.append(
            {
                **{k: spec[k] for k in ("method", "workflow")},
                "task": runtime["task"],
                "seed": runtime["seed"],
                "collections": rows,
                "total": total,
            }
        )
    return {
        "schema": "reasoning-executed-intervention-credit-audit-1",
        "scope": "Completed development collections only, including finalized collections in ongoing workers. Pre-execution clear wins are predictions; local post-execution usefulness is also Astra judgment, not privileged ground truth or proof of superiority over an unexecuted alternative. Actual commands and original admission ledgers determine training coverage. Overlapping windows do not multiply unique actions. This audit does not assert a later checkpoint was evaluated.",
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.manifest), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
