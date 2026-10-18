"""Disjoint execution ownership for an unchanged, frozen LIBERO schedule."""

import copy

from astra_reversal.hardware_interface.common import digest


def design_hash(manifest):
    value = copy.deepcopy(manifest)
    value.pop("resolved", None)
    value.pop("readiness", None)
    return digest(value)


def block(row):
    return (int(row["task_id"]), int(row["init_state_index"]), int(row["replicate"]))


def validate_plan(plan, manifest, rows):
    """Require exact schedule coverage, unchanged design, and co-located arms."""
    if plan["design_sha256"] != design_hash(manifest):
        raise ValueError("parallel_design_changed")
    if plan["schedule_sha256"] != digest(rows):
        raise ValueError("parallel_schedule_changed")
    pilot = [r for r in rows if r["split"] == "pilot"]
    owners = [plan["parent_trial_ids"]] + [s["trial_ids"] for s in plan["shards"]]
    flat = [item for owner in owners for item in owner]
    expected = [r["trial_id"] for r in pilot]
    if len(flat) != len(set(flat)) or set(flat) != set(expected):
        raise ValueError("parallel_missing_or_duplicate_ownership")
    if expected[: len(owners[0])] != owners[0]:
        raise ValueError("parallel_parent_must_own_prefix")
    ownership = {trial: i for i, ids in enumerate(owners) for trial in ids}
    seen = {}
    for row in pilot:
        key = block(row)
        owner = ownership[row["trial_id"]]
        if key in seen and seen[key] != owner:
            raise ValueError("parallel_matched_block_split")
        seen[key] = owner
    for ids in owners:
        if [r["trial_id"] for r in pilot if r["trial_id"] in set(ids)] != ids:
            raise ValueError("parallel_order_changed")
    if [s["index"] for s in plan["shards"]] != list(range(len(plan["shards"]))):
        raise ValueError("parallel_shard_indices")


def create_plan(manifest, rows, *, started, source, parent_workflow, shards=4):
    pilot = [r for r in rows if r["split"] == "pilot"]
    ids = [r["trial_id"] for r in pilot]
    if not set(started) <= set(ids):
        raise ValueError("unknown_started_trial")
    groups = []
    for row in pilot:
        if not groups or block(groups[-1][0]) != block(row):
            groups.append([])
        groups[-1].append(row)
    last = max(
        (
            i
            for i, group in enumerate(groups)
            if any(r["trial_id"] in started for r in group)
        ),
        default=-1,
    )
    # Finish the active matched block plus one additional block. The existing
    # process is stopped at the next exclusive-create boundary, never mid-trial.
    parent_groups = min(last + 2, len(groups))
    parent = [r["trial_id"] for group in groups[:parent_groups] for r in group]
    assignments = [[] for _ in range(shards)]
    for i, group in enumerate(groups[parent_groups:]):
        assignments[i % shards].extend(r["trial_id"] for r in group)
    result = {
        "version": 1,
        "parent_workflow": parent_workflow,
        "experiment_source_commit": source,
        "experiment_source_sha256": manifest["resolved"]["adapter_sha256"],
        "design_sha256": design_hash(manifest),
        "schedule_sha256": digest(rows),
        "parent_trial_ids": parent,
        "boundary_trial_id": ids[len(parent)] if len(parent) < len(ids) else None,
        "shards": [{"index": i, "trial_ids": ids} for i, ids in enumerate(assignments)],
    }
    validate_plan(result, manifest, rows)
    return result


def project(rows, expected, selected):
    """Check the complete schedule before projecting a worker's execution rows."""
    if len(rows) != len(expected) or any(
        set(row) != set(reference)
        or any(str(row[key]) != str(reference[key]) for key in reference)
        for row, reference in zip(rows, expected)
    ):
        raise ValueError("parallel_schedule_changed")
    return [r for r in rows if r["split"] != "pilot" or r["trial_id"] in selected]


def create_fresh_plan(manifest, rows, *, source, source_sha256, cohort, shards=4):
    """Assign every block of a new cohort; there is no already-running parent."""
    if type(shards) is not int or not 1 <= shards <= 4:
        raise ValueError("fresh_cohort_requires_one_to_four_shards")
    assignments = [[] for _ in range(shards)]
    pilot = [r for r in rows if r["split"] == "pilot"]
    owner_by_block = {}
    for row in pilot:
        owner = owner_by_block.setdefault(block(row), len(owner_by_block) % shards)
        assignments[owner].append(row["trial_id"])
    if any(not ids for ids in assignments):
        raise ValueError("fresh_cohort_empty_shard")
    result = {
        "version": 1,
        "parent_workflow": cohort,
        "fresh_cohort": True,
        "experiment_source_commit": source,
        "experiment_source_sha256": source_sha256,
        "design_sha256": design_hash(manifest),
        "schedule_sha256": digest(rows),
        "parent_trial_ids": [],
        "boundary_trial_id": pilot[0]["trial_id"],
        "shards": [{"index": i, "trial_ids": ids} for i, ids in enumerate(assignments)],
    }
    validate_plan(result, manifest, rows)
    return result
