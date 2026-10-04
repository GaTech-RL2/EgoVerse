import pytest
import torch

from astra_reversal.reasoning_learning.rlinf_ppo import (
    advantages_and_returns,
    clipped_losses,
)


def test_terminal_mask_blocks_bootstrap_and_next_episode_reward():
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.2, 0.3, 100.0])
    _, returns = advantages_and_returns(
        rewards, values, torch.tensor([True, True]), gamma=1, gae_lambda=1
    )
    torch.testing.assert_close(returns, rewards)
    _, returns = advantages_and_returns(
        rewards, values, torch.tensor([False, True]), gamma=1, gae_lambda=1
    )
    torch.testing.assert_close(returns, torch.ones(2))


def test_ppo_clips_improved_ratio_without_rewarding_further_probability_change():
    logprob = torch.tensor(1.5).log().requires_grad_()
    actor, critic = clipped_losses(
        logprob,
        torch.tensor(0.0),
        torch.tensor(2.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    assert float(actor) == pytest.approx(-2.4)
    assert float(critic) == 0
    actor.backward()
    assert float(logprob.grad) == 0


def test_value_loss_penalizes_regression_beyond_the_clip():
    value = torch.tensor(-0.5, requires_grad=True)
    _, critic = clipped_losses(
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
        value,
        torch.tensor(0.0),
        torch.tensor(2.0),
    )
    critic.backward()
    assert float(critic) == pytest.approx(3.125)
    assert float(value.grad) == pytest.approx(-2.5)
