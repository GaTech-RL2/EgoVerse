from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.core.ckpt_loading import (
    _assert_completed_seed_rollouts,
    _override_sampler_inference_steps,
    _parse_init_seeds,
    _strict_state_dict,
)
from egomimic.eval.core.eval_sim import SimRolloutEval
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes_sim import (
    _ENV_TO_ZARR,
    _state_to_env_init,
)


class _IdentityNormStats:
    _keys = {
        "proprio_keys": ["state_agent_obj"],
        "camera_keys": [],
        "lang_keys": [],
        "action_keys": ["actions"],
    }

    def keys_of_type(self, key_type, embodiment_id):
        del embodiment_id
        return list(self._keys.get(key_type, []))

    def is_key_with_embodiment(self, key, embodiment_id):
        del embodiment_id
        return key in {"state_agent_obj", "actions"}

    def keyname_to_zarr_key(self, key, embodiment_id):
        del embodiment_id
        return key

    def zarr_key_to_keyname(self, key, embodiment_id):
        del embodiment_id
        return key if key in {"state_agent_obj", "actions"} else None

    def normalize(self, data, embodiment_id):
        del embodiment_id
        return dict(data)

    def unnormalize(self, data, embodiment_id):
        del embodiment_id
        return dict(data)


def test_explicit_protocol_seed_parser_is_strict():
    assert _parse_init_seeds("2011, 2022,2033") == [2011, 2022, 2033]
    assert _parse_init_seeds(None) is None
    with pytest.raises(ValueError, match="duplicates"):
        _parse_init_seeds("2011,2011")
    with pytest.raises(ValueError, match="invalid"):
        _parse_init_seeds("2011,nope")


class _Condition(Stage):
    reads = ["obs/state_agent_obj"]
    writes = ["condition"]
    rollout_obs_steps = 1

    def forward(self, batch):
        batch["condition"] = batch["obs/state_agent_obj"]
        return batch


class _TwoObsCondition(_Condition):
    rollout_obs_steps = 2


class _BF16Prediction(Stage):
    reads = ["condition", "embodiment"]
    writes = ["pred_action"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        batch["pred_action"] = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                ]
            ],
            dtype=torch.bfloat16,
            device=batch["condition"].device,
        )
        return batch


class _IndexedPrediction(Stage):
    reads = ["condition", "embodiment"]
    writes = ["pred_action"]

    def __init__(self, horizon=10):
        super().__init__()
        self.horizon = int(horizon)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        indices = torch.arange(
            self.horizon,
            dtype=torch.float32,
            device=batch["condition"].device,
        ).view(1, self.horizon, 1)
        batch["pred_action"] = indices.expand(-1, -1, 4).clone()
        return batch


class _NativeFourAdapter:
    def decode(self, actions, context=None):
        del context
        return actions[..., :4]


def test_pipeline_inference_step_keeps_native_width_and_bf16_safe_queue():
    prediction = _BF16Prediction()
    algo = PipelineAlgo(
        stages=[_Condition(), prediction],
        norm_stats=_IdentityNormStats(),
        domains=["pushshapes_sim_chain_gripper"],
        ac_keys={"pushshapes_sim_chain_gripper": "actions"},
        rollout_adapter=_NativeFourAdapter(),
        action_horizon=3,
        device=torch.device("cpu"),
    )
    algo.replan_every = 2
    emb_id = get_embodiment_id("pushshapes_sim_chain_gripper")
    obs = {"state_agent_obj": torch.zeros((1, 6), dtype=torch.float32)}

    first = algo.inference_step(obs, 0, emb_id)
    second = algo.inference_step(obs, 1, emb_id)
    third = algo.inference_step(obs, 2, emb_id)

    assert first.dtype == np.float32
    assert first.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert second.tolist() == [7.0, 8.0, 9.0, 10.0]
    assert third.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert prediction.calls == 2


def _indexed_pipeline(horizon=10):
    prediction = _IndexedPrediction(horizon=horizon)
    algo = PipelineAlgo(
        stages=[_TwoObsCondition(), prediction],
        norm_stats=_IdentityNormStats(),
        domains=["pushshapes_sim_chain_gripper"],
        ac_keys={"pushshapes_sim_chain_gripper": "actions"},
        rollout_adapter=_NativeFourAdapter(),
        action_horizon=horizon,
        device=torch.device("cpu"),
    )
    return algo, prediction


def test_pipeline_action_chunk_start_one_queues_tokens_one_through_eight():
    algo, prediction = _indexed_pipeline(horizon=10)
    algo.action_chunk_start_index = 1
    algo.replan_every = 8
    emb_id = get_embodiment_id("pushshapes_sim_chain_gripper")
    obs = {"state_agent_obj": torch.zeros((1, 6), dtype=torch.float32)}

    executed = [algo.inference_step(obs, t, emb_id)[0] for t in range(8)]

    assert executed == pytest.approx(list(range(1, 9)))
    assert prediction.calls == 1

    # Exhausting the queue replans from the same explicit start index.
    assert algo.inference_step(obs, 8, emb_id)[0] == pytest.approx(1.0)
    assert prediction.calls == 2
    # A new episode must discard a partially consumed queue and plan afresh.
    assert algo.inference_step(obs, 0, emb_id)[0] == pytest.approx(1.0)
    assert prediction.calls == 3


def test_pipeline_action_chunk_start_defaults_to_zero_without_family_inference():
    algo, prediction = _indexed_pipeline(horizon=10)
    algo.replan_every = 2
    emb_id = get_embodiment_id("pushshapes_sim_chain_gripper")
    obs = {"state_agent_obj": torch.zeros((1, 6), dtype=torch.float32)}

    first = algo.inference_step(obs, 0, emb_id)
    second = algo.inference_step(obs, 1, emb_id)

    assert algo.action_chunk_start_index == 0
    assert first[0] == pytest.approx(0.0)
    assert second[0] == pytest.approx(1.0)
    assert prediction.calls == 1


@pytest.mark.parametrize(
    "start_index,replan_every,message",
    [
        (-1, 1, "must be non-negative"),
        (3, 1, "must be smaller than the decoded horizon"),
        (1, 3, "exceeds the decoded horizon"),
        (0, 0, "replan_every must be positive"),
    ],
)
def test_pipeline_action_chunk_slice_validation(
    start_index, replan_every, message
):
    algo, _ = _indexed_pipeline(horizon=3)
    algo.action_chunk_start_index = start_index
    algo.replan_every = replan_every
    emb_id = get_embodiment_id("pushshapes_sim_chain_gripper")
    obs = {"state_agent_obj": torch.zeros((1, 6), dtype=torch.float32)}

    with pytest.raises(ValueError, match=message):
        algo.inference_step(obs, 0, emb_id)


def _obs():
    return {
        "agent_pos": np.array([10.0, 20.0]),
        "agent_angle": np.array([0.25]),
        "object_pose": np.array([30.0, 40.0, -0.5]),
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_pushshapes_eval_glue_preserves_oriented_state_schema():
    circle = _ENV_TO_ZARR["pushshapes_sim"](_obs(), torch.device("cpu"))
    chain = _ENV_TO_ZARR["pushshapes_sim_chain_gripper"](_obs(), torch.device("cpu"))
    socket = _ENV_TO_ZARR["pushshapes_sim_u_socket"](_obs(), torch.device("cpu"))

    assert circle["state_agent_obj"].shape == (1, 5)
    assert chain["state_agent_obj"].shape == (1, 6)
    assert torch.equal(chain["state_agent_obj"], socket["state_agent_obj"])
    assert chain["state_agent_obj"][0].tolist() == pytest.approx(
        [10.0, 20.0, 0.25, 30.0, 40.0, -0.5]
    )

    init = _state_to_env_init(
        np.array([10.0, 20.0, 0.25, 30.0, 40.0, -0.5]),
        "pushshapes_sim_chain_gripper",
    )
    assert init == {
        "agent_pos": (10.0, 20.0),
        "agent_angle": 0.25,
        "object_pose": (30.0, 40.0, -0.5),
    }


class _FakeEnv:
    def __init__(self):
        self.actions = []
        self.reset_seeds = []
        self.action_space = SimpleNamespace(shape=(4,))

    def reset(self, seed=None):
        self.reset_seeds.append(seed)

    def _get_obs(self):
        return _obs()

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return _obs(), 0.0, True, False, {"coverage": 0.5}

    def render(self):
        return None


class _FakeAlgo:
    def inference_step(self, obs_zarr, t, emb_id, T_max=None):
        del obs_zarr, t, emb_id, T_max
        return np.array([100.0, 200.0, 0.3, 0.7], dtype=np.float32)


class _FakeChainEval(SimRolloutEval):
    def __init__(
        self,
        env,
        *,
        init_mode="random",
        init_seeds=None,
        limit_val_batches=1,
    ):
        super().__init__(
            embodiment_name="pushshapes_sim_chain_gripper",
            init_mode=init_mode,
            init_seeds=init_seeds,
            max_steps=1,
            rollout_timeout_s=0,
            limit_val_batches=limit_val_batches,
        )
        self._fake_env = env

    def _get_env(self):
        return self._fake_env

    def _infer_n_episodes(self, batch):
        del batch
        return 1

    def batch_to_env_init(self, batch, b_idx, emb_id):
        del batch, b_idx, emb_id
        return None


def test_sim_evaluator_passes_full_chain_action_to_environment(tmp_path):
    env = _FakeEnv()
    evaluator = _FakeChainEval(env)
    evaluator.model = _FakeAlgo()
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu")),
        default_root_dir=str(tmp_path),
    )

    coverage, frames = evaluator._rollout_one_impl(None, 20, 0)

    assert coverage == pytest.approx(0.5)
    assert frames == []
    assert len(env.actions) == 1
    assert env.actions[0].tolist() == pytest.approx([100.0, 200.0, 0.3, 0.7])


def test_seed_mode_rolls_every_seed_even_when_batch_reports_one_episode(tmp_path):
    env = _FakeEnv()
    seeds = [2011, 2022, 2033]
    evaluator = _FakeChainEval(
        env,
        init_mode="seeds",
        init_seeds=seeds,
        limit_val_batches=len(seeds),
    )
    evaluator.model = _FakeAlgo()
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(device=torch.device("cpu")),
        default_root_dir=str(tmp_path),
    )

    metrics, _ = evaluator.compute_metrics_and_viz({20: {}})

    assert env.reset_seeds == seeds
    assert len(evaluator._last_per_ep_coverages[20]) == len(seeds)
    assert metrics["Valid/emb20_sim_coverage"].item() == pytest.approx(0.5)


def test_completed_seed_rollout_count_assertion_is_strict():
    evaluator = SimpleNamespace(_last_per_ep_coverages={20: [0.1, 0.2, 0.3]})

    assert _assert_completed_seed_rollouts(evaluator, [20], 3) == {20: 3}
    with pytest.raises(RuntimeError, match=r"emb20 completed=3 expected=4"):
        _assert_completed_seed_rollouts(evaluator, [20], 4)


def _registered_ema_checkpoint():
    return {
        "state_dict": {
            "nets.policy.weight": torch.tensor([1.0]),
            "nets.policy.bias": torch.tensor([2.0]),
            "ema_nets.policy.weight": torch.tensor([3.0]),
            "ema_nets.policy.bias": torch.tensor([4.0]),
            "ema_optimization_step": torch.tensor(17),
            "ema_decay": torch.tensor(0.999),
        }
    }


def test_strict_state_dict_selects_complete_registered_ema_or_online_tree():
    ckpt = _registered_ema_checkpoint()

    online = _strict_state_dict(ckpt, use_ema=False)
    ema = _strict_state_dict(ckpt, use_ema=True)

    assert set(online) == {"policy.weight", "policy.bias"}
    assert set(ema) == set(online)
    assert online["policy.weight"].item() == pytest.approx(1.0)
    assert ema["policy.weight"].item() == pytest.approx(3.0)


def test_strict_state_dict_rejects_partial_or_foreign_registered_ema_state():
    ckpt = _registered_ema_checkpoint()
    del ckpt["state_dict"]["ema_nets.policy.bias"]
    with pytest.raises(RuntimeError, match="do not exactly match"):
        _strict_state_dict(ckpt, use_ema=True)

    ckpt = _registered_ema_checkpoint()
    ckpt["state_dict"]["foreign_buffer"] = torch.tensor(1)
    with pytest.raises(RuntimeError, match="non-model state keys"):
        _strict_state_dict(ckpt, use_ema=True)

    ckpt = _registered_ema_checkpoint()
    del ckpt["state_dict"]["ema_decay"]
    with pytest.raises(RuntimeError, match="missing wrapper buffers"):
        _strict_state_dict(ckpt, use_ema=True)


def test_sampler_override_finds_selected_nested_domain_policy():
    chain = SimpleNamespace(num_inference_steps=100)
    socket = SimpleNamespace(num_inference_steps=100)
    stage = SimpleNamespace(
        policies={
            "pushshapes_sim_chain_gripper": chain,
            "pushshapes_sim_u_socket": socket,
        }
    )
    algo = SimpleNamespace(policy=SimpleNamespace(stages=[stage]))

    stage_name, original = _override_sampler_inference_steps(
        algo,
        32,
        embodiment_name="pushshapes_sim_chain_gripper",
    )

    assert stage_name == "SimpleNamespace"
    assert original == 100
    assert chain.num_inference_steps == 32
    assert socket.num_inference_steps == 100


def test_sampler_override_rejects_ambiguous_direct_candidates():
    algo = SimpleNamespace(
        policy=SimpleNamespace(
            stages=[
                SimpleNamespace(num_inference_steps=100),
                SimpleNamespace(num_inference_steps=100),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        _override_sampler_inference_steps(algo, 50)
