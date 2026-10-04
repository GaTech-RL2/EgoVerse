import numpy as np
import pytest
import torch

from astra_reversal.reasoning_learning.rl_recipes import (
    collect_on_policy,
    collection_screen,
    settings,
)
from astra_reversal.reasoning_learning.rlinf_ppo import advantages_and_returns


def test_collection_screen_keeps_order_and_scheduled_complete_updates():
    resets = list(range(8))
    schedule = [0, 2, 4, 8]
    assert collection_screen(resets, schedule) == resets
    assert collection_screen(resets, schedule, 2, update_period=2) == [0, 1]
    assert resets == list(range(8))
    for invalid in (0, 1, 3, 6, 9, True, 2.0):
        with pytest.raises(ValueError, match="scheduled full-update evaluation"):
            collection_screen(resets, schedule, invalid)
    with pytest.raises(ValueError, match="scheduled full-update evaluation"):
        collection_screen(resets, schedule, 2, update_period=4)


def test_tuning_batches_cannot_cross_evaluation_or_policy_boundaries():
    kwargs = dict(collection_rollouts=8, evaluation_schedule=[0, 2, 4, 8])
    assert settings("standard", "dsrl", **kwargs)["updates"] == 200
    assert settings("more_reuse", "dsrl", **kwargs)["updates"] == 600
    assert (
        settings("more_reuse", "ppo", **kwargs)["collection_rollouts_per_update"] == 2
    )
    with pytest.raises(ValueError, match="evaluation boundary"):
        settings(
            "more_reuse", "ppo", collection_rollouts=8, evaluation_schedule=[0, 1, 2]
        )
    episode = [
        {"policy_version": 2, "terminal": False},
        {"policy_version": 2, "terminal": True},
    ]
    batch = collect_on_policy([], episode, 2)
    assert len(collect_on_policy(batch, episode, 2)) == 4
    assert (
        len(batch) == 2
    )  # Adding an episode does not mutate the recorded prior batch.
    with pytest.raises(ValueError, match="policy versions"):
        collect_on_policy(batch, [{"policy_version": 3, "terminal": True}], 3)
    with pytest.raises(ValueError, match="end before batching"):
        collect_on_policy([], episode[:1], 2)


def test_batched_gae_does_not_leak_reward_across_episode_boundaries():
    # A terminal failed episode precedes a success. Its return must remain zero.
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.zeros(5)
    terminals = torch.tensor([False, True, False, True])
    _, returns = advantages_and_returns(
        rewards, values, terminals, gamma=1, gae_lambda=1
    )
    np.testing.assert_array_equal(returns.numpy(), [0, 0, 1, 1])
