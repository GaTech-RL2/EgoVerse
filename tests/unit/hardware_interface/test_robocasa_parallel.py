"""Sharding must keep every frozen trial and its matched methods together."""

import copy
from pathlib import Path

import pytest
import yaml

from astra_reversal.hardware_interface.robocasa_protocol import schedule
from experiments.robocasa_hardware_interface.tools.parallel_plan import (
    assigned_rows,
    create_plan,
)


def manifest(adaptive=False):
    root = Path(__file__).resolve().parents[3]
    name = "preregistration-adaptive.yaml" if adaptive else "preregistration.yaml"
    return yaml.safe_load(
        (root / "experiments/robocasa_hardware_interface" / name).read_text()
    )


@pytest.mark.parametrize("adaptive,total", [(False, 150), (True, 100)])
def test_complete_disjoint_coverage_preserves_matched_blocks_and_row_bytes(
    adaptive, total
):
    frozen = manifest(adaptive)
    before = copy.deepcopy(frozen)
    plan = create_plan(frozen, "commit", "adapter", shards=4)
    assert plan == create_plan(frozen, "commit", "adapter", shards=4)
    rows = schedule(frozen)
    combined, ownership = [], {}
    for index in range(4):
        selected = assigned_rows(plan, frozen, index, "commit", "adapter")
        assert selected
        assert selected == [row for row in rows if row in selected]
        assert (
            sum(row["horizon"] for row in selected)
            == plan["shards"][index]["scheduled_native_steps"]
        )
        for row in selected:
            key = row["task_id"], row["env_seed"]
            assert ownership.setdefault(key, index) == index
        combined.extend(selected)
    assert len(combined) == len({row["trial_id"] for row in combined}) == total
    assert sorted(combined, key=lambda row: row["order"]) == rows
    assert frozen == before


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "split", "order", "source", "design"]
)
def test_invalid_assignment_is_rejected_before_any_actor(fault):
    frozen = manifest()
    plan = create_plan(frozen, "commit", "adapter")
    if fault == "duplicate":
        plan["shards"][1]["trial_ids"].append(plan["shards"][0]["trial_ids"][0])
    elif fault == "missing":
        plan["shards"][0]["trial_ids"].pop()
    elif fault == "split":
        a, b = plan["shards"][:2]
        a["trial_ids"][0], b["trial_ids"][0] = b["trial_ids"][0], a["trial_ids"][0]
    elif fault == "order":
        plan["shards"][0]["trial_ids"].reverse()
    elif fault == "source":
        plan["experiment_source_sha256"] = "other"
    else:
        frozen["randomization_seed"] += 1
    with pytest.raises(ValueError):
        assigned_rows(plan, frozen, 0, "commit", "adapter")


@pytest.mark.parametrize("count", [0, -1, 51, True, 1.5])
def test_invalid_worker_count(count):
    with pytest.raises(ValueError, match="invalid_parallel_worker_count"):
        create_plan(manifest(), "commit", "adapter", count)


def test_commissioning_receipts_do_not_change_frozen_design():
    frozen = manifest()
    plan = create_plan(frozen, "commit", "adapter")
    frozen["resolved"] = {
        "source_sha256": "adapter",
        "native_fixture_order_fix": "recorded",
    }
    assert assigned_rows(plan, frozen, 0, "commit", "adapter")
