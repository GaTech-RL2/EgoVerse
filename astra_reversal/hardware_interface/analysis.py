"""Paired task-macro analysis with initial-state clusters and explicit missingness."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import digest, strict_json


def audit(runs):
    reports = []
    for path in sorted(Path(runs).glob("*/events.jsonl")):
        previous, count, applied, started, ended = "0" * 64, 0, 0, False, False
        failures = []
        for line in path.read_text().splitlines():
            row = strict_json(line)
            expected = row.pop("sha256")
            if (
                row["previous_sha256"] != previous
                or row["sequence"] != count
                or digest(row) != expected
            ):
                failures.append("event_chain")
            previous, count = expected, count + 1
            started |= row["event"] == "trial_start"
            ended |= row["event"] == "trial_end"
            if row["event"] == "action_applied":
                applied += 1
                if row["simulator_step_after"] != applied:
                    failures.append("action_continuity")
        outcome = path.parent / "outcome.json"
        if outcome.exists():
            result = strict_json(outcome.read_bytes())
            if (
                result["actor_started"] != started
                or not ended
                or result["sim_steps"] != applied
            ):
                failures.append("outcome_continuity")
        else:
            failures.append("outcome_missing")
        reports.append(
            {
                "trial": path.parent.name,
                "events": count,
                "applied_steps": applied,
                "status": "passed" if not failures else "failed",
                "errors": sorted(set(failures)),
            }
        )
    return {
        "trials": reports,
        "status": "passed"
        if reports and all(r["status"] == "passed" for r in reports)
        else "failed",
    }


def _distribution(values):
    known = [x for x in values if x is not None]
    if not known:
        return {"n_known": 0, "n_unknown": len(values), "median": None, "iqr": None}
    return {
        "n_known": len(known),
        "n_unknown": len(values) - len(known),
        "median": float(np.median(known)),
        "iqr": np.quantile(known, [0.25, 0.75]).tolist(),
    }


def analyze(outcomes, manifest, *, split):
    selected = [r for r in outcomes if r["split"] == split and r["actor_started"]]
    groups = defaultdict(dict)
    for row in selected:
        key = (row["task_id"], row["init_state_index"], row["replicate"])
        if row["condition"] in groups[key]:
            raise ValueError("duplicate_trial_requires_explicit_rerun_policy")
        groups[key][row["condition"]] = row
    per_task, summaries = {}, {}
    for condition in manifest["conditions"]:
        rows = [r for r in selected if r["condition"] == condition]
        tasks = {}
        for task in manifest["task_ids"]:
            trials = [r for r in rows if r["task_id"] == task]
            tasks[str(task)] = {
                "n": len(trials),
                "successes": sum(r["success"] for r in trials),
                "success_rate": np.mean([r["success"] for r in trials]).item()
                if trials
                else None,
            }
        per_task[condition] = tasks
        complete = all(v["n"] for v in tasks.values())
        summaries[condition] = {
            "n": len(rows),
            "successes": sum(r["success"] for r in rows),
            "task_macro_success": float(
                np.mean([v["success_rate"] for v in tasks.values()])
            )
            if complete
            else None,
            "all_trial_wall_seconds": _distribution([r["wall_s"] for r in rows]),
            "all_trial_steps": _distribution([r["sim_steps"] for r in rows]),
            "success_only_wall_seconds": _distribution(
                [r["wall_s"] for r in rows if r["success"]]
            ),
            "tokens": _distribution(
                [
                    r["known_workflow_tokens"]
                    if not r["unknown_usage_records"]
                    else None
                    for r in rows
                ]
            ),
            "cost": _distribution([r["estimated_cost_usd"] for r in rows]),
            "wall_timeouts_right_censored": sum(r["censored_wall"] for r in rows),
            "invalid_actions": sum(r["invalid_actions"] for r in rows),
            "safety_attempts": sum(r["safety_attempts"] for r in rows),
            "applied_safety_violations": sum(
                r["applied_safety_violations"] for r in rows
            ),
        }
    comparisons = {}
    for other in ("B0", "B"):
        by_task = defaultdict(lambda: defaultdict(list))
        discordant = [0, 0]
        for (task, index, _), block in groups.items():
            if "F" not in block or other not in block:
                continue
            f, b = block["F"], block[other]
            if (
                f["init_state_hash"] != b["init_state_hash"]
                or f["env_seed"] != b["env_seed"]
            ):
                raise ValueError("unmatched_initial_state")
            delta = int(f["success"]) - int(b["success"])
            by_task[task][index].append(delta)
            if delta:
                discordant[int(delta < 0)] += 1
        if set(by_task) != set(manifest["task_ids"]):
            comparisons["F_minus_" + other] = {
                "status": "incomplete_task_pairs",
                "decision": "inconclusive",
            }
            continue
        values = [
            np.array([np.mean(v) for v in states.values()])
            for _, states in sorted(by_task.items())
        ]
        rng = np.random.default_rng(manifest["analysis"]["bootstrap_seed"])
        count = manifest["analysis"]["bootstrap_samples"]
        bootstrap = np.mean(
            [
                rng.choice(v, size=(count, len(v)), replace=True).mean(axis=1)
                for v in values
            ],
            axis=0,
        )
        ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        delta = float(np.mean([v.mean() for v in values]))
        n = sum(discordant)
        p = (
            min(
                1.0,
                2 * sum(math.comb(n, i) for i in range(min(discordant) + 1)) / (2**n),
            )
            if n
            else 1.0
        )
        indices = (
            manifest["pilot_indices"]
            if split == "pilot"
            else manifest["confirmatory_indices"]
        )
        expected = {
            (task, index, rep)
            for task in manifest["task_ids"]
            for index in indices
            for rep in range(manifest["replicates"])
        }
        complete = set(groups) == expected and all(
            set(block) == set(manifest["conditions"]) for block in groups.values()
        )
        confirm = (
            split == "confirmatory"
            and manifest["power"]["confirmation_frozen"]
            and complete
        )
        comparisons["F_minus_" + other] = {
            "status": "paired_descriptive",
            "risk_difference": delta,
            "bootstrap_95pct": ci,
            "method": "resample initial-state clusters within each fixed task; equal task weight",
            "discordant_F_only_other_only": discordant,
            "mcnemar_exact_p": p if manifest["replicates"] == 1 else None,
            "primary": other == "B0",
            "confirmatory_eligible": confirm and other == "B0",
            "decision": "success_improvement"
            if confirm and other == "B0" and delta > 0 and ci[0] > 0
            else "inconclusive",
        }
    return {
        "split": split,
        "analysis_population": "all started trials; no automatic infrastructure exclusions",
        "per_task": per_task,
        "arms": summaries,
        "comparisons": comparisons,
        "limitations": [
            "Pilot is unscored and cannot establish the primary claim.",
            "Safety observations do not establish safety equivalence.",
            "All-trial wall values retain observed censoring; success-only values are selected.",
        ],
    }
