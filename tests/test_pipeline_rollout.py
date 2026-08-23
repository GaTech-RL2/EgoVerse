import numpy as np
import pytest
import torch

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rollout.core import RolloutPipeline
from egomimic.rollout.eva import (
    EvaActionCodec,
    EvaObservationCodec,
    EvaObservationWindow,
)
from egomimic.rollout.nodes import (
    ActionDequeue,
    ChunkCommit,
    ObsCadence,
    PolicyStep,
)


class _IdentityStats:
    def unnormalize(self, data, embodiment_id):
        return dict(data)


class _FakePolicy:
    def __init__(self, chunk):
        self.chunk = chunk
        self.calls = 0

    def forward_rollout(self, domain, observation, **kwargs):
        self.calls += 1
        assert domain == "eva_bimanual"
        assert observation["state_ee_pose"].shape == (1, 2, 20)
        return self.chunk[None]


def _identity_action(left_xyz=(0.1, 0.2, 0.3), right_xyz=(0.4, 0.5, 0.6)):
    rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return torch.tensor(
        [*left_xyz, *rotation, 0.25, *right_xyz, *rotation, 0.75],
        dtype=torch.float32,
    )


def _obs():
    return {
        "front_img_1": np.zeros((8, 10, 3), dtype=np.uint8),
        "left_wrist_img": np.zeros((8, 10, 3), dtype=np.uint8),
        "right_wrist_img": np.zeros((8, 10, 3), dtype=np.uint8),
        "ee_poses": np.zeros(14, dtype=np.float32),
        "joint_positions": np.zeros(14, dtype=np.float32),
    }


def test_eva_pipeline_reuses_two_frame_history_and_decodes_base_command():
    chunk = torch.stack([_identity_action(), _identity_action((0.2, 0.0, 0.0))])
    policy = _FakePolicy(chunk)
    rollout = RolloutPipeline(
        [
            EvaObservationWindow(2),
            ObsCadence(mode="every_n", every_n=2),
            EvaObservationCodec(),
            PolicyStep(policy, "eva_bimanual"),
            ChunkCommit(n_keep=2),
            ActionDequeue(on_empty="raise"),
            EvaActionCodec(_IdentityStats(), get_embodiment_id("eva_bimanual")),
        ]
    )
    state = rollout.reset()
    obs = _obs()
    obs["ee_poses"][[0, 7]] = [1.0, -1.0]
    state = rollout.step(state, obs)
    np.testing.assert_allclose(state["command"][:3], [1.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state["command"][7:10], [-0.6, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(state["command"][[6, 13]], [0.25, 0.75], atol=1e-6)
    state = rollout.step(state, obs)
    np.testing.assert_allclose(state["command"][:3], [1.2, 0.0, 0.0], atol=1e-6)
    assert policy.calls == 1


class _FakeNormStats:
    def __init__(self):
        self.normalize_calls = 0

    def keys_of_type(self, key_type, embodiment_id):
        if key_type == "proprio_keys":
            return ["state_ee_pose"]
        if key_type == "action_keys":
            return ["actions_cartesian"]
        return []

    def is_key_with_embodiment(self, key, embodiment_id):
        return True

    def zarr_key_to_keyname(self, key, embodiment_id):
        return key if key in {"state_ee_pose", "actions_cartesian"} else None

    def normalize(self, data, embodiment_id):
        self.normalize_calls += 1
        out = dict(data)
        out["state_ee_pose"] = out["state_ee_pose"] * 2
        return out


class _RolloutStage(Stage):
    reads = ("obs/state_ee_pose", "embodiment", "rollout_t")
    writes = ("pred_action",)

    def forward(self, batch):
        value = batch["obs/state_ee_pose"]
        batch["pred_action"] = value[:, :1, :1]
        return batch


class _TrainLoss(Stage):
    train_only = True
    reads = ("pred_action", "target")
    writes = ("loss/test",)

    def forward(self, batch):
        raise AssertionError("train-only loss ran during rollout")


def test_pipeline_algo_rollout_normalizes_raw_observation_exactly_once():
    stats = _FakeNormStats()
    algo = PipelineAlgo(
        stages=[_RolloutStage(), _TrainLoss()],
        norm_stats=stats,
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        action_horizon=1,
        device=torch.device("cpu"),
    )
    prediction = algo.forward_rollout(
        "eva_bimanual", {"state_ee_pose": torch.ones(1, 2, 20)}
    )
    assert stats.normalize_calls == 1
    torch.testing.assert_close(prediction, torch.full((1, 1, 1), 2.0))


def test_pipeline_policy_rejects_cadence_longer_than_action_horizon():
    from types import SimpleNamespace

    stats = _FakeNormStats()
    algo = PipelineAlgo(
        stages=[_RolloutStage()],
        norm_stats=stats,
        domains=["eva_bimanual"],
        ac_keys={"eva_bimanual": "actions_cartesian"},
        action_horizon=1,
        device=torch.device("cpu"),
    )
    config = SimpleNamespace(
        arm="both",
        cartesian=True,
        action_frame="base",
        query_frequency=2,
        annotation_path=None,
        embodiment_id=get_embodiment_id("eva_bimanual"),
    )
    with pytest.raises(ValueError, match="action_horizon"):
        algo.create_rollout_policy(config)
