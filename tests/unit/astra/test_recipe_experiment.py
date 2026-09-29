import copy

import pytest

from astra_reversal.recipe_experiment import (
    assignment,
    recorded_schedules,
    scheduled_choice,
)
from astra_reversal.representation_search import keyed_rng


def sample(trajectory, arm, step, alpha, revision=1, study="phase"):
    return {
        "kind": "correction",
        "trajectory_id": trajectory,
        "observation_step": step,
        "source_study": study,
        "arm": arm,
        "revision": revision,
        "episode_id": "task:seed29:state0",
        "original_prompt": "move object",
        "source_receipt_sha256": "a" * 64,
        "choice": {
            "operator": arm.removeprefix("astra_"),
            "source_a_id": "10",
            "source_b_id": "14",
            "alpha": alpha,
        },
    }


def test_schedule_uses_declared_teacher_order_and_absolute_clock():
    rows = [
        sample("tei", "astra_tei", 0, 0.1),
        sample("tli", "astra_tli", 0, 0.25),
        sample("tli", "astra_tli", 25, 0.4),
        sample("pilot", "astra_tli", 0, 0.9, study="representation"),
    ]
    schedules = recorded_schedules(rows)
    assert schedules["move object"]["trajectory_id"] == "tli"
    assert scheduled_choice(schedules, "move object", 24)["alpha"] == 0.25
    assert scheduled_choice(schedules, "move object", 25)["alpha"] == 0.4
    assert scheduled_choice(schedules, "move object", 299)["alpha"] == 0.4
    assert scheduled_choice(schedules, "unseen task", 0) == {"operator": "native"}
    result = scheduled_choice(schedules, "move object", 25)
    result["alpha"] = 0.8
    assert scheduled_choice(schedules, "move object", 25)["alpha"] == 0.4
    assert recorded_schedules(list(reversed(rows))) == schedules


def test_five_workers_partition_all_cases_without_using_outcomes():
    assigned = []
    for worker in range(5):
        suite, tasks = assignment(worker)
        assigned.extend((suite, t, r) for t in tasks for r in (1, 2))
    assert len(assigned) == len(set(assigned)) == 48
    assert sum(suite.endswith("_ood") for suite, _, _ in assigned) == 40
    with pytest.raises(ValueError):
        assignment(5)


def test_policy_noise_is_matched_across_arms_and_changes_with_reset():
    entry = {"episode_id": "libero_goal_ood:seed61:task6:state1", "seed": 61}
    first = keyed_rng(entry, 0, 25).standard_normal((1, 10, 32))
    assert (
        first == keyed_rng(copy.deepcopy(entry), 0, 25).standard_normal((1, 10, 32))
    ).all()
    other = {**entry, "episode_id": "libero_goal_ood:seed61:task6:state2"}
    assert not (first == keyed_rng(other, 0, 25).standard_normal((1, 10, 32))).all()
