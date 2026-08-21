import numpy as np
import torch

from egomimic.eval.inference_graph import ActionCacheState
from egomimic.pipeline.algo import PipelineAlgo


class _NormStats:
    def __init__(self):
        self.normalize_calls = 0

    def normalize(self, obs, emb_id):
        self.normalize_calls += 1
        return dict(obs)

    def unnormalize(self, values, emb_id):
        return values


class _TinyHNet(PipelineAlgo):
    """Minimal rollout surface; no training pipeline construction required."""

    def __init__(self, inference_stages=None):
        self.nets = torch.nn.ModuleDict({"anchor": torch.nn.Linear(1, 1)})
        self.norm_stats = _NormStats()
        self.action_horizon = 32
        self.replan_stride = 2
        self.domain_by_id = {15: "pushshapes_sim"}
        self.ac_keys = {"pushshapes_sim": "actions"}
        self.inference_stages = inference_stages
        self._sim_action_cache = ActionCacheState()
        self._sim_action_queue = self._sim_action_cache.actions
        self._inference_graph = self._build_inference_graph()
        self.model_calls = 0
        self.reset_calls = 0

    def init_step_state(self, batch_size, T_max, device, dtype):
        self.reset_calls += 1
        return {"T_max": T_max}

    def step(self, state, obs_norm, t, embodiment_id=None):
        self.model_calls += 1
        base = 10 * t
        return torch.tensor(
            [[base + 1.0], [base + 2.0], [base + 3.0]],
            device=next(self.nets.parameters()).device,
        )


def _act(policy, t):
    return policy.inference_step({"x": torch.tensor([float(t)])}, t, 15)


def test_hnet_graph_cache_hit_skips_preprocess_and_model():
    policy = _TinyHNet()

    np.testing.assert_array_equal(_act(policy, 0), np.array([1], np.float32))
    assert policy.model_calls == 1
    assert policy.norm_stats.normalize_calls == 1
    assert len(policy._sim_action_cache.actions) == 1

    np.testing.assert_array_equal(_act(policy, 1), np.array([2], np.float32))
    assert policy.model_calls == 1
    assert policy.norm_stats.normalize_calls == 1
    assert not policy._sim_action_cache

    np.testing.assert_array_equal(_act(policy, 2), np.array([21], np.float32))
    assert policy.model_calls == 2
    assert policy.norm_stats.normalize_calls == 2


def test_runtime_obs_adapter_runs_only_after_cache_miss():
    policy = _TinyHNet()
    calls = []

    def adapt(obs):
        calls.append(float(obs["x"][0]))
        return {"x": obs["x"] + 100}

    policy.inference_obs_adapter = adapt
    _act(policy, 0)  # miss: preprocess and model
    _act(policy, 1)  # hit: graph exits before environment/model adaptation
    _act(policy, 2)  # miss again

    assert calls == [0.0, 2.0]


def test_hnet_graph_t0_resets_history_and_cache_per_episode():
    policy = _TinyHNet()
    _act(policy, 0)
    assert policy._sim_action_cache
    _act(policy, 0)
    assert policy.reset_calls == 2
    assert policy.model_calls == 2
    assert len(policy._sim_action_cache.actions) == 1


def test_hnet_graph_state_is_policy_instance_scoped():
    first, second = _TinyHNet(), _TinyHNet()
    _act(first, 0)
    assert first._sim_action_cache
    assert not second._sim_action_cache
    _act(second, 0)
    assert first._sim_action_cache.actions is not second._sim_action_cache.actions


def test_hnet_graph_accepts_literal_key_remapping():
    cfg = {
        "terminal": "hnet.action",
        "nodes": {
            "check_cache": {
                "in": {"obs": "raw.obs"},
                "out": {"action": "hnet.action"},
            },
            "inference_preprocess": {
                "in": {"obs": "raw.obs"},
                "out": {"request": "hnet.request"},
            },
            "update_cache": {
                "in": {"plan": "hnet.plan", "obs": "raw.obs"},
                "out": {"action": "hnet.action"},
            },
        },
        "model": {
            "in": {"request": "hnet.request"},
            "out": {"plan": "hnet.plan"},
        },
    }
    policy = _TinyHNet(cfg)
    policy._reset_inference_graph()
    policy._sim_t = 0
    policy._sim_emb_id = 15
    policy._sim_ac_key = "actions"
    out = policy._inference_graph(**{"raw.obs": {"x": torch.tensor([0.0])}})
    np.testing.assert_array_equal(out, np.array([1], np.float32))
