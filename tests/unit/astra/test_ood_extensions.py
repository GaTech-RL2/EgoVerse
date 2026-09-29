"""Task-definition safety and novelty checks, without simulator dependencies."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from astra_reversal.ood_extensions import (
    goal_keys,
    parse,
    section,
    serialize,
    spatial_keys,
    validate,
)

TASKS = Path(__file__).resolve().parents[3] / "astra_reversal/reports/ood_extensions/v1"


@pytest.fixture
def release():
    return json.loads((TASKS / "manifest.json").read_text())


def test_release_hashes_references_and_independent_compositions(release):
    assert release["families"] == {
        "astra_goal_composition": 8,
        "astra_spatial_composition": 8,
    }
    assert release["reference_task_count"] == 150
    seen = set()
    for task in release["tasks"]:
        payload = (TASKS / task["bddl"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == task["bddl_sha256"]
        tree = parse(payload.decode())
        assert validate(tree)
        assert " ".join(section(tree, ":language")[1:]) == task["instruction"]
        assert task["source_object"] in section(tree, ":obj_of_interest")
        key = (
            goal_keys(tree)[0]
            if task["family"] == "astra_goal_composition"
            else spatial_keys(tree)[0]
        )
        assert list(key) == task["novelty_key"]
        assert (task["family"], key) not in seen
        seen.add((task["family"], key))
    assert release["success_rate"] is None and release["new_policy_rollouts"] == 0


@pytest.mark.parametrize(
    "text", ["()", "", "(define", "define)", "(define))", "(define)(define)"]
)
def test_malformed_problem_rejected(text):
    with pytest.raises(ValueError):
        parse(text)


def test_nested_region_ranges_survive_serialization(release):
    tree = parse((TASKS / release["tasks"][0]["bddl"]).read_text())
    restored = parse(serialize(tree))
    assert section(restored, ":regions") == section(tree, ":regions")
    assert validate(restored)


def test_missing_goal_object_is_rejected(release):
    tree = parse((TASKS / release["tasks"][0]["bddl"]).read_text())
    section(tree, ":goal")[1][1][1] = "nonexistent_carton_1"
    with pytest.raises(ValueError, match="Unresolved predicate reference"):
        validate(tree)


def test_novelty_uses_object_type_not_misleading_instance_name(release):
    tree = parse((TASKS / release["tasks"][0]["bddl"]).read_text())
    actual = goal_keys(tree)
    alias = copy.deepcopy(tree)

    def change(value):
        return (
            [change(x) for x in value]
            if isinstance(value, list)
            else ("orange_juice_1" if value == "milk_1" else value)
        )

    alias = change(alias)
    assert goal_keys(alias) == actual
    assert actual[0][1] == "milk"


def test_spatial_signature_retains_source_relation(release):
    spatial = release["tasks"][8:]
    same_destination = [
        task for task in spatial if task["destination"] == "flat_stove_1_cook_region"
    ]
    trees = [parse((TASKS / task["bddl"]).read_text()) for task in same_destination]
    assert len({goal_keys(tree)[0] for tree in trees}) == 1
    assert len({spatial_keys(tree)[0] for tree in trees}) == 4
