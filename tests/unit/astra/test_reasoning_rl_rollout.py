from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.reasoning_learning.rl_rollout import RLRollout, discounted_step_cost


def test_step_cost_counts_failed_steps_and_terminal_success():
    assert discounted_step_cost([False, False, True], gamma=0.9) == pytest.approx(-1.9)
    with pytest.raises(ValueError, match="executed"):
        discounted_step_cost([])


@pytest.mark.parametrize("evaluation", [False, True])
def test_rl_transition_contains_only_executed_prefix_and_own_observation(
    tmp_path, evaluation
):
    policy = SimpleNamespace(
        input_transform=lambda raw: {
            "actions": np.pad(raw["actions"], ((0, 0), (0, 25)))
        },
        output_transform=lambda data: {"actions": data["actions"][:, :7]},
        prepare=lambda *args: SimpleNamespace(state=torch.zeros(1, 8)),
        sample=lambda condition, noise, steps: SimpleNamespace(value=noise),
    )
    learner = SimpleNamespace(
        version=0, noise=lambda obs, evaluation: torch.zeros(1, 10, 32)
    )
    spec = ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {})
    loop = RLRollout(
        policy,
        learner,
        spec,
        tmp_path / "episode",
        instruction="lift",
        evaluation=evaluation,
        seed=1,
        retain_evaluation_observations=False,
    )
    before = {
        "observation/image": np.zeros((8, 8, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
    }
    after = {**before, "observation/state": np.ones(8, np.float32)}
    commands = loop.action(before, 0)
    assert loop.transitions == []
    for step in range(3):
        loop.observed_step(before, commands[step], step, after, step == 2, False)
    loop.finish_prefix(after, end=True)
    if evaluation:
        loop.action(after, 3)
        assert not loop.transitions
        assert len(list((tmp_path / "episode").glob("observation_*.npz"))) == 1
        return
    row = loop.transitions[0]
    assert len(row["executed_actions"]) == 3
    assert row["terminal"] and row["ppo_reward"] == 1
    assert row["observation_id"] != row["next_observation_id"]
    assert np.array_equal(
        row["observation"]["observation/state"], before["observation/state"]
    )
    assert np.array_equal(
        row["next_observation"]["observation/state"], after["observation/state"]
    )
    loop.action(after, 3)
    assert len(list((tmp_path / "episode").glob("observation_*.npz"))) == 2
