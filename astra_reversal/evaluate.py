"""Per-task summaries and paired state-cluster bootstrap intervals."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAIR_KEYS = ("suite", "task_id", "initial_state_id", "seed")


def read_events(path):
    path = Path(path)
    events = path / "events.jsonl" if path.is_dir() else path
    with events.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read_episodes(path):
    return [row for row in read_events(path) if row["kind"] == "episode_end"]


def compare_runs(left, right, **kwargs):
    a = json.loads((Path(left) / "manifest.json").read_text())
    b = json.loads((Path(right) / "manifest.json").read_text())
    for name in ("checkpoint", "action_spec", "task_manifest_sha256"):
        if a[name] != b[name]:
            raise ValueError(f"Run provenance differs in {name}")
    return paired_comparison(read_episodes(left), read_episodes(right), **kwargs)


def summarize(episodes):
    if not episodes:
        raise ValueError("No completed episode records")
    if len({row.get("method") for row in episodes}) != 1:
        raise ValueError("Summarize one method at a time")
    protocols = {row["protocol"] for row in episodes if "protocol" in row}
    if len(protocols) > 1:
        raise ValueError("Do not aggregate standard and modified-LIBERO protocols")
    tasks = defaultdict(list)
    for row in episodes:
        if type(row["success"]) is not bool:
            raise ValueError("Success must come from the environment as a boolean")
        tasks[(row["suite"], row["task_id"])].append(row)
    output = {}
    for (suite, task), rows in sorted(tasks.items()):
        successes = [r for r in rows if r["success"]]
        output[f"{suite}:{task}"] = {
            "episodes": len(rows),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "actions_to_success": float(np.mean([r["actions"] for r in successes]))
            if successes
            else None,
            "seconds_to_success": float(np.mean([r["wall_seconds"] for r in successes]))
            if successes
            else None,
            "mean_actions": float(np.mean([r["actions"] for r in rows])),
            "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
            "mean_velocity_evaluations": float(
                np.mean([r["velocity_evaluations"] for r in rows])
            ),
            "mean_fallbacks": float(np.mean([r["fallbacks"] for r in rows])),
            "mean_agent_calls": float(np.mean([r.get("agent_calls", 0) for r in rows])),
            "invalid_response_rate": (
                sum(r.get("invalid_responses", 0) for r in rows)
                / sum(r.get("agent_calls", 0) for r in rows)
                if sum(r.get("agent_calls", 0) for r in rows)
                else None
            ),
        }
    suites = {}
    for suite in sorted({key[0] for key in tasks}):
        rates = [
            v["success_rate"] for k, v in output.items() if k.startswith(suite + ":")
        ]
        suites[suite] = {
            "macro_success": float(np.mean(rates)),
            "tasks": len(rates),
            "complete_ten_task_suite": len(rates) == 10,
        }
    return {
        "per_task": output,
        "per_suite": suites,
        "macro_success": float(np.mean([x["success_rate"] for x in output.values()])),
        "episodes": len(episodes),
        "interpretation": "Measured subset only; missing tasks are never silently assigned zero or omitted from a full-suite claim",
    }


def paired_comparison(left, right, *, samples=2000, seed=0):
    if type(samples) is not int or samples < 1:
        raise ValueError("Bootstrap samples must be a positive integer")

    def index(rows):
        values = {}
        for row in rows:
            key = tuple(row[name] for name in PAIR_KEYS)
            if key in values:
                raise ValueError(f"Duplicate paired episode: {key}")
            values[key] = row
        return values

    a, b = index(left), index(right)
    if not a or set(a) != set(b):
        raise ValueError(
            "Paired methods must contain identical task/state/seed keys; incomplete pairing is not accepted"
        )
    states = defaultdict(lambda: defaultdict(list))
    for key, row in a.items():
        other = b[key]
        for name in ("reset_state_sha256", "protocol", "task_action_budget", "split"):
            if name not in row or name not in other or row[name] != other[name]:
                raise ValueError(f"Paired episode {key} differs in {name}")
        if type(row["success"]) is not bool or type(other["success"]) is not bool:
            raise ValueError("Invalid success value")
        states[key[:2]][key[2]].append(float(other["success"]) - float(row["success"]))
    # Average repeated seeds inside each state before resampling states within
    # each task. Treating seed replicates as independent states understates error.
    differences = {
        task: np.asarray([np.mean(values) for values in task_states.values()])
        for task, task_states in states.items()
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(samples)
    for values in differences.values():
        bootstrap += values[
            rng.integers(0, len(values), size=(samples, len(values)))
        ].mean(axis=1) / len(differences)
    return {
        "difference_right_minus_left": float(
            np.mean([v.mean() for v in differences.values()])
        ),
        "paired_bootstrap_95_percent": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "paired_episodes": len(a),
        "tasks": len(differences),
        "bootstrap_samples": samples,
        "resampling_unit": "initial states within task; repeated seeds averaged within state",
    }


def trace_costs(path):
    counts = defaultdict(int)
    latency = defaultdict(list)
    failures = defaultdict(int)
    for row in read_events(path):
        kind = row["kind"]
        if kind == "agent_response":
            counts["agent_calls"] += 1
            counts["invalid_responses"] += int(not row["accepted"])
            counts["retry_calls"] += int(row["attempt"] > 0)
        if kind == "flow":
            counts["velocity_evaluations"] += row["velocity_evaluations"]
            counts[f"{row['role']}_calls"] += 1
        if kind == "progress":
            counts["assessor_calls"] += row["progress"].get("assessor_calls", 0)
            counts["assessor_errors"] += int("assessor_error" in row["progress"])
            if "assessor_latency_seconds" in row["progress"]:
                latency["assessor"].append(row["progress"]["assessor_latency_seconds"])
        if kind == "condition":
            latency["condition_preparation"].append(row["preparation_seconds"])
        if kind == "fallback":
            failures[row["reason"].split(":", 1)[0]] += 1
        if kind == "augmentation":
            counts["expired_annotations"] += sum(
                x["count"] for x in row["events"] if x["event"] == "annotation_expired"
            )
        if "latency_seconds" in row:
            latency[kind].append(row["latency_seconds"])
    total = counts["agent_calls"]
    return {
        "counts": dict(counts),
        "invalid_response_rate": counts["invalid_responses"] / total if total else None,
        "fallback_categories": dict(failures),
        "latency_seconds": {
            key: {
                "total": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for key, values in latency.items()
        },
    }
