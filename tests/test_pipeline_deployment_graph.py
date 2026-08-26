import numpy as np
import pytest
import torch

from egomimic.eval.inference_graph import ActionCacheState
from egomimic.pipeline.algo import PipelineAlgo


class _TinyPipeline(PipelineAlgo):
    """Minimal current-Pipeline rollout surface for graph contract tests."""

    def __init__(self, *, n_keep=2):
        self.domain_by_id = {15: "pushshapes_sim"}
        self.resolved_ac_keys = {15: "actions"}
        self.inference_stages = None
        self.inference_cache_overrides = {"n_keep": n_keep}
        self.rollout_adapter = None
        self._inference_action_cache = ActionCacheState()
        self._inference_graph = self._build_inference_graph()
        self.preprocess_calls = 0
        self.model_calls = 0

    def process_batch_for_rollout(self, batch):
        self.preprocess_calls += 1
        return {15: batch["pushshapes_sim"]}

    def forward_rollout(self, batch, rollout_t=0):
        del batch
        self.model_calls += 1
        base = 10 * rollout_t
        chunk = torch.tensor(
            [[[base + 1.0], [base + 2.0], [base + 3.0]]]
        )
        return {"emb15_actions": chunk}


def _act(policy, t):
    return policy.inference_step(
        {"x": torch.tensor([float(t)])}, t, 15
    )


def test_cache_hits_skip_preprocess_and_model():
    policy = _TinyPipeline(n_keep=2)

    np.testing.assert_array_equal(_act(policy, 0), [1.0])
    assert policy.preprocess_calls == policy.model_calls == 1
    np.testing.assert_array_equal(_act(policy, 1), [2.0])
    assert policy.preprocess_calls == policy.model_calls == 1
    np.testing.assert_array_equal(_act(policy, 2), [21.0])
    assert policy.preprocess_calls == policy.model_calls == 2


def test_runtime_observation_adapter_runs_only_on_cache_miss():
    policy = _TinyPipeline(n_keep=2)
    adapted = []

    def adapter(obs):
        adapted.append(float(obs["x"][0]))
        return obs

    policy.inference_obs_adapter = adapter
    _act(policy, 0)
    _act(policy, 1)
    _act(policy, 2)

    assert adapted == [0.0, 2.0]


def test_t0_resets_episode_scoped_cache():
    policy = _TinyPipeline(n_keep=3)
    _act(policy, 0)
    assert len(policy._inference_action_cache.actions) == 2
    np.testing.assert_array_equal(_act(policy, 0), [1.0])
    assert len(policy._inference_action_cache.actions) == 2
    assert policy.model_calls == 2


def test_inference_requires_t0_and_known_embodiment():
    policy = _TinyPipeline()
    with pytest.raises(RuntimeError, match="begin with t == 0"):
        _act(policy, 1)
    with pytest.raises(KeyError, match="does not support embodiment"):
        policy.inference_step({}, 0, 999)
