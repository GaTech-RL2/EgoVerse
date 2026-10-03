"""Optional CPU source-integration test against the pinned RLinf checkout."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal.reasoning_learning.rlinf_bridge import import_core
from astra_reversal.reasoning_learning.rlinf_sac import DSRLLearner


def test_released_dsrl_actor_learns_from_real_transition_structure():
    root = os.environ.get("EGOVERSE_TEST_RLINF_ROOT")
    if not root:
        pytest.skip("Set EGOVERSE_TEST_RLINF_ROOT to the pinned source checkout")
    import_core(root)
    torch.set_num_threads(2)
    policy = SimpleNamespace(device=torch.device("cpu"), horizon=10)
    learner = DSRLLearner(policy, seed=173, batch_size=2, updates=12)
    obs = {
        "observation/image": np.zeros((64, 64, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
    }
    noise = learner.noise(obs, evaluation=False)
    assert noise.shape == (1, 10, 32)
    assert torch.equal(noise[:, 0], noise[:, 9])
    before = {k: v.clone() for k, v in learner.network.state_dict().items()}
    rows = [
        dict(
            observation=obs,
            next_observation=obs,
            noise=noise,
            executed_actions=[[0] * 7] * 5,
            terminal=True,
            sac_reward=1.0,
        )
        for _ in range(10)
    ]
    result = learner.update(rows)
    assert result["updated"]
    assert any(
        not torch.equal(before[k], v)
        for k, v in learner.network.state_dict().items()
        if k.startswith("dsrl_action_noise_net")
    )
    assert all(p.grad is None for p in learner.target.parameters())
    assert all(np.isfinite(row["critic_loss"]) for row in result["history"])
