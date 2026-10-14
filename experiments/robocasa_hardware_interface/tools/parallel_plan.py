"""Assign every frozen RoboCasa matched block to exactly one OSMO worker."""

from astra_reversal.hardware_interface.common import digest
from astra_reversal.hardware_interface.robocasa_protocol import schedule
from experiments.libero_hardware_interface.tools.parallel_plan import (
    block,
    design_hash,
    validate_plan,
)


def create_plan(manifest, source_commit, source_sha256, shards=4):
    rows = schedule(manifest)
    groups = {}
    for row in rows:
        groups.setdefault(block(row), []).append(row)
    if type(shards) is not int or not 1 <= shards <= len(groups):
        raise ValueError("invalid_parallel_worker_count")
    assignments, loads = [set() for _ in range(shards)], [0] * shards
    # Balance using the frozen episode horizons, never observed policy outcomes.
    for group in sorted(
        groups.values(), key=lambda group: -sum(row["horizon"] for row in group)
    ):
        index = min(range(shards), key=lambda index: (loads[index], index))
        assignments[index].update(row["trial_id"] for row in group)
        loads[index] += sum(row["horizon"] for row in group)
    plan = {
        "version": 1,
        "benchmark": "robocasa365",
        "experiment_source_commit": source_commit,
        "experiment_source_sha256": source_sha256,
        "design_sha256": design_hash(manifest),
        "schedule_sha256": digest(rows),
        "parent_trial_ids": [],
        "shards": [
            {
                "index": index,
                "trial_ids": [
                    row["trial_id"] for row in rows if row["trial_id"] in selected
                ],
                "scheduled_native_steps": loads[index],
            }
            for index, selected in enumerate(assignments)
        ],
        "assignment": "Whole matched blocks balanced by official horizons; original schedule order preserved within each worker.",
    }
    validate_plan(plan, manifest, rows)
    return plan


def assigned_rows(plan, manifest, index, source_commit, source_sha256):
    rows = schedule(manifest)
    validate_plan(plan, manifest, rows)
    if plan.get("version") != 1 or plan.get("benchmark") != "robocasa365":
        raise ValueError("unknown_robocasa_execution_plan")
    if plan["parent_trial_ids"]:
        raise ValueError("fresh_cohort_requires_no_parent_ownership")
    if (
        plan["experiment_source_commit"] != source_commit
        or plan["experiment_source_sha256"] != source_sha256
    ):
        raise ValueError("parallel_source_changed")
    if type(index) is not int or not 0 <= index < len(plan["shards"]):
        raise ValueError("invalid_parallel_worker_index")
    selected = set(plan["shards"][index]["trial_ids"])
    if not selected:
        raise ValueError("empty_parallel_assignment")
    return [row for row in rows if row["trial_id"] in selected]
