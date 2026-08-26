import numpy as np
import pytest

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import (
    build_fold_cartesian_wristframe_revert_transform_list,
)
from egomimic.rldb.embodiment.fold_span_transforms import (
    eva_rollout_obs_transforms,
)
from egomimic.robot.rollout import EMBODIMENT_MAP, PolicyRollout


def _apply(transforms, batch):
    for transform in transforms:
        batch = transform.transform(batch)
    return batch


def test_robot_arm_names_resolve_through_canonical_embodiment_enum():
    assert get_embodiment_id(EMBODIMENT_MAP["right"]) == 4
    assert get_embodiment_id(EMBODIMENT_MAP["left"]) == 5
    assert get_embodiment_id(EMBODIMENT_MAP["both"]) == 6


def test_fold_robot_observation_adapter_matches_training_20d_layout():
    # Raw zarr/robot poses are xyz + identity quaternion (wxyz).
    batch = {
        "left.obs_ee_pose": np.array([1, 2, 3, 1, 0, 0, 0], np.float32),
        "right.obs_ee_pose": np.array([4, 5, 6, 1, 0, 0, 0], np.float32),
        "left.obs_gripper": np.array([0.25], np.float32),
        "right.obs_gripper": np.array([0.75], np.float32),
    }
    out = _apply(eva_rollout_obs_transforms(), batch)["state_ee_pose"]

    assert tuple(out.shape) == (20,)
    np.testing.assert_allclose(out[[0, 1, 2, 9]], [1, 2, 3, 0.25])
    np.testing.assert_allclose(out[[10, 11, 12, 19]], [4, 5, 6, 0.75])


def test_fold_wrist_action_decodes_to_base_frame_robot_layout():
    # Identity rot6d is columns e1,e2. Wrist-relative translations must be
    # composed with the query-time base-frame wrist poses.
    identity6d = [1, 0, 0, 0, 1, 0]
    action = np.array(
        [1, 2, 3, *identity6d, 0.25,
         4, 5, 6, *identity6d, 0.75],
        dtype=np.float32,
    )
    state = np.array(
        [10, 0, 0, *identity6d, 0.1,
         0, 20, 0, *identity6d, 0.9],
        dtype=np.float32,
    )
    batch = _apply(
        build_fold_cartesian_wristframe_revert_transform_list(),
        {"actions_cartesian": action[None], "state_ee_pose": state[None]},
    )
    robot = np.asarray(batch["actions_cartesian"])[0]

    assert robot.shape == (14,)
    np.testing.assert_allclose(robot, [
        11, 2, 3, 0, 0, 0, 0.25,
        4, 25, 6, 0, 0, 0, 0.75,
    ], atol=1e-6)


def test_policy_rollout_uses_saved_query_pose_to_decode_cached_actions():
    identity6d = [1, 0, 0, 0, 1, 0]
    action = np.array(
        [1, 0, 0, *identity6d, 0.25,
         0, 2, 0, *identity6d, 0.75],
        dtype=np.float32,
    )
    policy = object.__new__(PolicyRollout)
    policy.cartesian = True
    policy.arm = "both"
    policy.action_revert_list = (
        build_fold_cartesian_wristframe_revert_transform_list()
    )
    policy._query_state_ee_pose = np.array(
        [[10, 0, 0, *identity6d, 0.1,
          0, 20, 0, *identity6d, 0.9]],
        dtype=np.float32,
    )

    first = policy._model_action_to_robot(action)
    cached = policy._model_action_to_robot(action)

    np.testing.assert_allclose(first, cached)
    np.testing.assert_allclose(first[[0, 1, 7, 8]], [11, 0, 0, 22])


def test_policy_rollout_rejects_wrist_action_without_query_pose():
    policy = object.__new__(PolicyRollout)
    policy.cartesian = True
    policy.arm = "both"
    policy._query_state_ee_pose = None
    policy.action_revert_list = (
        build_fold_cartesian_wristframe_revert_transform_list()
    )

    with pytest.raises(RuntimeError, match="query-time state"):
        policy._model_action_to_robot(np.zeros(20, dtype=np.float32))
