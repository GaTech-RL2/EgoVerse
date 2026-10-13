"""Scheduling controls must neither change a trial nor assign it twice."""

import copy

import pytest

from experiments.libero_hardware_interface.tools.parallel_plan import (
    create_plan,
    project,
    validate_plan,
)


def fixture(conditions=("F", "B0", "B")):
    manifest = {
        "conditions": list(conditions),
        "resolved": {"adapter_sha256": "a" * 64},
        "readiness": {"ok": True},
        "limits": {"steps": 1000},
    }
    rows = [
        {
            "trial_id": "%s-%d-%d-%s" % (split, task, init, method),
            "split": split,
            "task_id": task,
            "init_state_index": init,
            "replicate": 0,
            "condition": method,
            "limit": None,
        }
        for split in ("pilot", "confirmatory")
        for task in range(3)
        for init in range(3)
        for method in conditions
    ]
    return manifest, rows


@pytest.mark.parametrize("conditions", [("F", "B0", "B"), ("F", "F_adaptive")])
def test_disjoint_full_coverage_and_matched_blocks(conditions):
    manifest, rows = fixture(conditions)
    started = [r["trial_id"] for r in rows[: len(conditions) + 1]]
    plan = create_plan(
        manifest, rows, started=started, source="abc", parent_workflow="owned", shards=4
    )
    validate_plan(plan, manifest, rows)
    assert set(started) <= set(plan["parent_trial_ids"])
    assert len(plan["parent_trial_ids"]) == 3 * len(conditions)
    for shard in plan["shards"]:
        selected = set(shard["trial_ids"])
        actual = project(copy.deepcopy(rows), rows, selected)
        assert [r for r in actual if r["split"] == "pilot"] == [
            r for r in rows if r["trial_id"] in selected
        ]


def test_plan_rejects_duplicates_missing_and_split_arms():
    manifest, rows = fixture()
    plan = create_plan(
        manifest, rows, started=[], source="abc", parent_workflow="owned"
    )
    for mutation in ("duplicate", "missing", "split"):
        bad = copy.deepcopy(plan)
        if mutation == "duplicate":
            bad["shards"][1]["trial_ids"].append(bad["shards"][0]["trial_ids"][0])
        elif mutation == "missing":
            bad["shards"][0]["trial_ids"].pop()
        else:
            a, b = bad["shards"][:2]
            a["trial_ids"][0], b["trial_ids"][0] = b["trial_ids"][0], a["trial_ids"][0]
        with pytest.raises(ValueError):
            validate_plan(bad, manifest, rows)


def test_projection_rejects_tampered_full_schedule():
    _, rows = fixture()
    bad = copy.deepcopy(rows)
    bad[-1]["condition"] = "changed"
    with pytest.raises(ValueError, match="schedule_changed"):
        project(bad, rows, {rows[0]["trial_id"]})


def test_changed_protocol_cannot_be_pooled():
    manifest, rows = fixture()
    plan = create_plan(
        manifest, rows, started=[], source="abc", parent_workflow="owned"
    )
    manifest["limits"]["steps"] = 500
    with pytest.raises(ValueError, match="design_changed"):
        validate_plan(plan, manifest, rows)
