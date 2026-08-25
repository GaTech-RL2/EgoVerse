import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva_frames import (
    EVA_DATASET_FROM_HARDWARE_ROTATION,
    hardware_ypr_pose_to_dataset_wxyz,
    hardware_ypr_pose_to_dataset_ypr,
)
from egomimic.rldb.embodiment.fold_span_transforms import eva_normal_transforms
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
    def __init__(self):
        self.unnormalize_calls = 0

    def unnormalize(self, data, embodiment_id):
        self.unnormalize_calls += 1
        return dict(data)


class _AffineStats:
    scale = 1.7
    offset = -0.23

    def __init__(self):
        self.unnormalize_calls = 0

    def normalize_action(self, action):
        return (action - self.offset) / self.scale

    def unnormalize(self, data, embodiment_id):
        self.unnormalize_calls += 1
        return {key: value * self.scale + self.offset for key, value in data.items()}


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


def _rot6d(rotation):
    return np.concatenate([rotation[..., :, 0], rotation[..., :, 1]], axis=-1)


def _dataset_state_arm(hardware_pose, gripper):
    dataset_pose = hardware_ypr_pose_to_dataset_ypr(hardware_pose)
    rotation = Rotation.from_euler("ZYX", dataset_pose[..., 3:6]).as_matrix()
    return np.concatenate(
        [dataset_pose[..., :3], _rot6d(rotation), np.asarray([gripper])]
    )


def _relative_dataset_action(current_hardware, target_hardware, gripper):
    current = hardware_ypr_pose_to_dataset_ypr(current_hardware)
    target = hardware_ypr_pose_to_dataset_ypr(target_hardware)
    current_rotation = Rotation.from_euler("ZYX", current[3:6]).as_matrix()
    target_rotation = Rotation.from_euler("ZYX", target[3:6]).as_matrix()
    relative_xyz = current_rotation.T @ (target[:3] - current[:3])
    relative_rotation = current_rotation.T @ target_rotation
    return np.concatenate(
        [relative_xyz, _rot6d(relative_rotation), np.asarray([gripper])]
    )


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
    np.testing.assert_allclose(state["command"][:3], [1.3, -0.1, -0.2], atol=1e-6)
    np.testing.assert_allclose(state["command"][7:10], [-0.4, -0.4, -0.5], atol=1e-6)
    np.testing.assert_allclose(state["command"][[3, 4, 5, 10, 11, 12]], 0, atol=1e-6)
    np.testing.assert_allclose(state["command"][[6, 13]], [0.25, 0.75], atol=1e-6)
    next_obs = _obs()
    next_obs["ee_poses"] = np.asarray(
        [
            9.0,
            8.0,
            7.0,
            0.4,
            -0.3,
            0.2,
            0.1,
            -9.0,
            -8.0,
            -7.0,
            -0.5,
            0.2,
            -0.1,
            0.9,
        ],
        dtype=np.float32,
    )
    state = rollout.step(state, next_obs)
    np.testing.assert_allclose(state["command"][:3], [1.0, -0.2, 0.0], atol=1e-6)
    assert policy.calls == 1


def test_eva_observation_codec_matches_zarr_tool_axis_convention():
    ee = np.asarray(
        [
            [
                0.31,
                0.22,
                0.41,
                0.37,
                -0.41,
                0.22,
                0.2,
                0.46,
                -0.18,
                0.29,
                -0.61,
                0.28,
                -0.35,
                0.8,
            ],
            [
                0.34,
                0.19,
                0.43,
                0.42,
                -0.33,
                0.17,
                0.3,
                0.43,
                -0.15,
                0.32,
                -0.55,
                0.31,
                -0.27,
                0.7,
            ],
        ],
        dtype=np.float64,
    )
    images = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    state = {
        "should_query": True,
        "obs_window": {
            "front_img_1": images,
            "left_wrist_img": images,
            "right_wrist_img": images,
            "ee_poses": ee,
            "joint_positions": np.zeros((2, 14), dtype=np.float64),
        },
    }
    EvaObservationCodec()(state)

    expected = []
    naive = []
    for row in ee:
        left_rotation = (
            EVA_DATASET_FROM_HARDWARE_ROTATION
            @ Rotation.from_euler("ZYX", row[3:6]).as_matrix()
        )
        right_rotation = (
            EVA_DATASET_FROM_HARDWARE_ROTATION
            @ Rotation.from_euler("ZYX", row[10:13]).as_matrix()
        )
        expected.append(
            np.concatenate(
                [
                    row[:3],
                    _rot6d(left_rotation),
                    row[6:7],
                    row[7:10],
                    _rot6d(right_rotation),
                    row[13:14],
                ]
            )
        )
        naive.append(
            np.concatenate(
                [
                    row[:3],
                    _rot6d(Rotation.from_euler("ZYX", row[3:6]).as_matrix()),
                    row[6:7],
                    row[7:10],
                    _rot6d(Rotation.from_euler("ZYX", row[10:13]).as_matrix()),
                    row[13:14],
                ]
            )
        )

    actual = state["native_state_ee_pose"]
    assert actual.shape == (2, 20)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.max(np.abs(actual - np.asarray(naive))) > 0.5


def test_eva_action_codec_reverts_dataset_pose_to_hardware_convention():
    left_current = np.asarray([0.31, 0.22, 0.41, 0.37, -0.41, 0.22])
    right_current = np.asarray([0.46, -0.18, 0.29, -0.61, 0.28, -0.35])
    left_target = np.asarray([0.35, 0.19, 0.43, 0.52, -0.29, 0.11])
    right_target = np.asarray([0.42, -0.14, 0.26, -0.48, 0.36, -0.21])
    assert np.linalg.norm(left_target[:3] - left_current[:3]) > 0.02

    native_state = np.concatenate(
        [
            _dataset_state_arm(left_current, 0.2),
            _dataset_state_arm(right_current, 0.8),
        ]
    )
    action = np.concatenate(
        [
            _relative_dataset_action(left_current, left_target, 0.35),
            _relative_dataset_action(right_current, right_target, 0.65),
        ]
    )
    stats = _AffineStats()
    state = {
        "action": torch.from_numpy(stats.normalize_action(action)).float(),
        "native_state_ee_pose": np.stack([native_state, native_state]),
    }
    EvaActionCodec(stats, get_embodiment_id("eva_bimanual"))(state)

    command = state["command"]
    assert command.shape == (14,)
    assert command.dtype == np.float32
    np.testing.assert_allclose(command[:3], left_target[:3], atol=1e-6)
    np.testing.assert_allclose(command[7:10], right_target[:3], atol=1e-6)
    np.testing.assert_allclose(command[[6, 13]], [0.35, 0.65], atol=1e-6)
    np.testing.assert_allclose(
        Rotation.from_euler("ZYX", command[3:6]).as_matrix(),
        Rotation.from_euler("ZYX", left_target[3:6]).as_matrix(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        Rotation.from_euler("ZYX", command[10:13]).as_matrix(),
        Rotation.from_euler("ZYX", right_target[3:6]).as_matrix(),
        atol=1e-6,
    )
    assert stats.unnormalize_calls == 1


def test_eva_action_codec_canonicalizes_only_numerical_gripper_residue():
    current = np.zeros(6, dtype=np.float64)
    native_state = np.concatenate(
        [
            _dataset_state_arm(current, 0.5),
            _dataset_state_arm(current, 0.5),
        ]
    )
    action = np.concatenate(
        [
            _relative_dataset_action(current, current, 1.0000009536743164),
            _relative_dataset_action(current, current, -0.0000009536743164),
        ]
    )
    stats = _IdentityStats()
    state = {
        "action": torch.from_numpy(action).float(),
        "native_state_ee_pose": np.stack([native_state, native_state]),
    }

    EvaActionCodec(stats, get_embodiment_id("eva_bimanual"))(state)

    np.testing.assert_array_equal(state["command"][[6, 13]], [1.0, 0.0])
    np.testing.assert_allclose(
        state["command"][[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]],
        0.0,
        atol=1e-6,
    )
    assert stats.unnormalize_calls == 1


@pytest.mark.parametrize(
    ("gripper", "expected"),
    [
        (1.000002, 1.0),
        (-0.000002, 0.0),
        (-0.001948229968547821, 0.0),
        (-0.018, 0.0),
        (-10.0, 0.0),
        (10.0, 1.0),
    ],
)
def test_eva_action_codec_clamps_finite_gripper_predictions(gripper, expected):
    current = np.zeros(6, dtype=np.float64)
    native_state = np.concatenate(
        [
            _dataset_state_arm(current, 0.5),
            _dataset_state_arm(current, 0.5),
        ]
    )
    action = np.concatenate(
        [
            _relative_dataset_action(current, current, gripper),
            _relative_dataset_action(current, current, 0.5),
        ]
    )
    stats = _IdentityStats()
    state = {
        "action": torch.from_numpy(action).float(),
        "native_state_ee_pose": np.stack([native_state, native_state]),
    }

    EvaActionCodec(stats, get_embodiment_id("eva_bimanual"))(state)

    assert state["command"][6] == expected
    np.testing.assert_allclose(
        state["command"][[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]],
        0.0,
        atol=1e-6,
    )
    assert stats.unnormalize_calls == 1


@pytest.mark.parametrize("gripper", [np.nan, np.inf, -np.inf])
def test_eva_action_codec_rejects_nonfinite_gripper_predictions(gripper):
    current = np.zeros(6, dtype=np.float64)
    native_state = np.concatenate(
        [
            _dataset_state_arm(current, 0.5),
            _dataset_state_arm(current, 0.5),
        ]
    )
    action = np.concatenate(
        [
            _relative_dataset_action(current, current, gripper),
            _relative_dataset_action(current, current, 0.5),
        ]
    )
    stats = _IdentityStats()
    state = {
        "action": torch.from_numpy(action).float(),
        "native_state_ee_pose": np.stack([native_state, native_state]),
    }

    with pytest.raises(ValueError, match="gripper command must be finite"):
        EvaActionCodec(stats, get_embodiment_id("eva_bimanual"))(state)
    assert "command" not in state
    assert stats.unnormalize_calls == 1


def test_eva_live_codecs_roundtrip_exact_training_transform():
    left_obs = np.asarray(
        [
            [0.30, 0.23, 0.40, 0.31, -0.44, 0.25],
            [0.31, 0.22, 0.41, 0.37, -0.41, 0.22],
        ]
    )
    right_obs = np.asarray(
        [
            [0.47, -0.19, 0.28, -0.64, 0.24, -0.39],
            [0.46, -0.18, 0.29, -0.61, 0.28, -0.35],
        ]
    )
    left_cmd = np.asarray(
        [
            left_obs[-1],
            [0.35, 0.19, 0.43, 0.52, -0.29, 0.11],
        ]
    )
    right_cmd = np.asarray(
        [
            right_obs[-1],
            [0.42, -0.14, 0.26, -0.48, 0.36, -0.21],
        ]
    )
    left_obs_grip = np.asarray([[0.25], [0.2]])
    right_obs_grip = np.asarray([[0.75], [0.8]])
    left_cmd_grip = np.asarray([[0.2], [0.35]])
    right_cmd_grip = np.asarray([[0.8], [0.65]])

    training_batch = {
        "left.obs_ee_pose": hardware_ypr_pose_to_dataset_wxyz(left_obs),
        "right.obs_ee_pose": hardware_ypr_pose_to_dataset_wxyz(right_obs),
        "left.cmd_ee_pose": hardware_ypr_pose_to_dataset_wxyz(left_cmd),
        "right.cmd_ee_pose": hardware_ypr_pose_to_dataset_wxyz(right_cmd),
        "left.obs_gripper": left_obs_grip,
        "right.obs_gripper": right_obs_grip,
        "left.cmd_gripper": left_cmd_grip,
        "right.cmd_gripper": right_cmd_grip,
    }
    for transform in eva_normal_transforms():
        training_batch = transform.transform(training_batch)

    ee_history = np.concatenate(
        [
            left_obs,
            left_obs_grip,
            right_obs,
            right_obs_grip,
        ],
        axis=-1,
    )
    images = np.zeros((2, 8, 10, 3), dtype=np.uint8)
    live = {
        "should_query": True,
        "obs_window": {
            "front_img_1": images,
            "left_wrist_img": images,
            "right_wrist_img": images,
            "ee_poses": ee_history,
            "joint_positions": np.zeros((2, 14), dtype=np.float64),
        },
    }
    EvaObservationCodec()(live)
    np.testing.assert_allclose(
        live["native_state_ee_pose"],
        training_batch["state_ee_pose"].numpy(),
        atol=1e-6,
    )

    stats = _IdentityStats()
    live["action"] = training_batch["actions_cartesian"][0]
    EvaActionCodec(stats, get_embodiment_id("eva_bimanual"))(live)
    command = live["command"]

    np.testing.assert_allclose(command[:3], left_cmd[-1, :3], atol=1e-6)
    np.testing.assert_allclose(command[7:10], right_cmd[-1, :3], atol=1e-6)
    np.testing.assert_allclose(command[[6, 13]], [0.35, 0.65], atol=1e-6)
    np.testing.assert_allclose(
        Rotation.from_euler("ZYX", command[3:6]).as_matrix(),
        Rotation.from_euler("ZYX", left_cmd[-1, 3:6]).as_matrix(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        Rotation.from_euler("ZYX", command[10:13]).as_matrix(),
        Rotation.from_euler("ZYX", right_cmd[-1, 3:6]).as_matrix(),
        atol=1e-6,
    )
    assert stats.unnormalize_calls == 1


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
