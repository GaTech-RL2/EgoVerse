"""Optional recovery routing, independent of frozen scientific protocols."""

import re

from astra_reversal.records import digest

TASK_IDS_ENV = "ASTRA_EVALUATION_TASK_IDS"


def select_assignment(phase, original, requested):
    """Restrict an original assignment, preserving its order and default shape."""
    if requested is None:
        return original
    if phase != "evaluation":
        raise ValueError(f"{TASK_IDS_ENV} is only allowed during evaluation")
    if not isinstance(requested, str) or not re.fullmatch(
        r"[0-9]+(?:,[0-9]+)*", requested
    ):
        raise ValueError(f"{TASK_IDS_ENV} must be nonempty comma-separated task IDs")
    selected = [int(value) for value in requested.split(",")]
    if len(selected) != len(set(selected)):
        raise ValueError(f"{TASK_IDS_ENV} contains duplicate task IDs")
    assigned = original.get("task_ids")
    if assigned is None:
        assigned = list(range(original["case_shard"], 10, original["case_shards"]))
    if not set(selected) <= set(assigned):
        raise ValueError(
            f"{TASK_IDS_ENV} contains IDs outside the original worker shard"
        )
    return {**original, "task_ids": [task for task in assigned if task in selected]}


def selected_entries(entries, task_ids):
    """Filter after original sharding; never renumber or regenerate reset entries."""
    if not set(task_ids) <= {row["task_id"] for row in entries}:
        raise ValueError(
            "Selected task IDs are missing from the original assigned episodes"
        )
    return [row for row in entries if row["task_id"] in task_ids]


def selected_manifest(manifest, entries):
    """Record the executed subset after all original reset captures have run.

    Capturing first retains the original reset order and RNG consumption. Entry
    identities and hashes are unchanged; only the enclosing manifest is rebound.
    """
    identities = {row["episode_id"] for row in entries}
    episodes = [row for row in manifest["episodes"] if row["episode_id"] in identities]
    cases = {(row["task_id"], row["initial_state_id"]) for row in episodes}
    result = {
        **manifest,
        "cases": [case for case in manifest["cases"] if tuple(case) in cases],
        "episodes": episodes,
    }
    result.pop("sha256")
    result["sha256"] = digest(result)
    return result
