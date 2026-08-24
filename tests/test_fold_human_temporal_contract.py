import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation

from egomimic.rldb.embodiment.fold_span_transforms import (
    ACTION_HORIZON,
    HUMAN_ACTION_SLOWDOWN,
    HUMAN_ACTION_SOURCE_HORIZON,
    BimanualCartesianBaseToCamera,
    build_bimanual_keypoint_wrist_revert_transforms,
    build_eva_rot6d_wrist_to_camera_transforms,
    human_normal_keymap,
    human_normal_keymap_kp,
    human_normal_transforms,
    human_normal_transforms_kp,
    rot6d_to_matrix,
)


def _wxyz_from_z_degrees(degrees):
    xyzw = Rotation.from_euler(
        "z", np.asarray(degrees)[:, None], degrees=True
    ).as_quat()
    return np.concatenate((xyzw[..., 3:4], xyzw[..., :3]), axis=-1)


def _apply(transforms, batch):
    for transform in transforms:
        batch = transform.transform(batch)
    return batch


def _source_pose_chunk(x_end=0.33, rotation_end_degrees=99.0):
    poses = np.zeros((HUMAN_ACTION_SOURCE_HORIZON, 7), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, x_end, HUMAN_ACTION_SOURCE_HORIZON)
    poses[:, 3:] = _wxyz_from_z_degrees(
        np.linspace(0.0, rotation_end_degrees, HUMAN_ACTION_SOURCE_HORIZON)
    )
    return poses


def test_human_keymaps_fetch_28_source_actions_plus_obs_alignment():
    expected_fetch = HUMAN_ACTION_SOURCE_HORIZON + 1
    cart = human_normal_keymap()
    kp = human_normal_keymap_kp()

    assert cart["left.act_ee_pose"]["horizon"] == expected_fetch
    assert kp["left.act_keypoints"]["horizon"] == expected_fetch
    assert HUMAN_ACTION_SLOWDOWN == 11.0 / 3.0


def test_cartesian_human_actions_interpolate_all_28_poses_to_100_with_slerp():
    source = _source_pose_chunk()
    identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # The loader fetches one alignment frame before the 28-action source chunk.
    fetched = np.concatenate((identity[None], source), axis=0)
    obs = np.stack((identity, identity))
    batch = {
        "left.act_ee_pose": fetched.copy(),
        "right.act_ee_pose": fetched.copy(),
        "left.obs_ee_pose": obs.copy(),
        "right.obs_ee_pose": obs.copy(),
        "obs_head_pose": obs.copy(),
    }

    out = _apply(human_normal_transforms(), batch)["actions_cartesian"]

    assert out.shape == (ACTION_HORIZON, 20)
    np.testing.assert_allclose(out[[0, -1], 0], [0.0, 0.33], atol=1e-7)
    rotation_matrices = rot6d_to_matrix(out[:, 3:9])
    z_degrees = Rotation.from_matrix(rotation_matrices).as_euler("ZYX", degrees=True)[
        :, 0
    ]
    np.testing.assert_allclose(z_degrees[[0, 50, -1]], [0.0, 50.0, 99.0], atol=1e-4)
    np.testing.assert_allclose(
        np.linalg.det(rotation_matrices), np.ones(ACTION_HORIZON), atol=1e-6
    )


def test_keypoint_human_actions_share_exact_interpolation_timestamps():
    source_pose = _source_pose_chunk()
    identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    fetched_pose = np.concatenate((identity[None], source_pose), axis=0)
    points = np.zeros((HUMAN_ACTION_SOURCE_HORIZON, 21, 3), dtype=np.float64)
    points[..., 0] = np.linspace(0.0, 0.33, HUMAN_ACTION_SOURCE_HORIZON)[:, None]
    points[..., 2] = 0.5
    fetched_points = np.concatenate((points[:1], points), axis=0).reshape(
        HUMAN_ACTION_SOURCE_HORIZON + 1, 63
    )
    obs_pose = np.stack((identity, identity))
    obs_points = np.zeros((2, 63), dtype=np.float64)
    batch = {
        "left.act_keypoints": fetched_points.copy(),
        "right.act_keypoints": fetched_points.copy(),
        "left.act_wrist_pose": fetched_pose.copy(),
        "right.act_wrist_pose": fetched_pose.copy(),
        "left.obs_keypoints": obs_points.copy(),
        "right.obs_keypoints": obs_points.copy(),
        "left.obs_wrist_pose": obs_pose.copy(),
        "right.obs_wrist_pose": obs_pose.copy(),
        "obs_head_pose": obs_pose.copy(),
    }

    out = _apply(human_normal_transforms_kp(), batch)["actions_keypoints"]

    assert out.shape == (ACTION_HORIZON, 126)
    np.testing.assert_allclose(out[[0, -1], 0], [0.0, 0.33], atol=1e-7)
    np.testing.assert_allclose(out[[0, -1], 63], [0.0, 0.33], atol=1e-7)


def test_canonical_126d_keypoints_revert_from_each_wrist_frame():
    actions = np.zeros((4, 126), dtype=np.float32)
    actions[:, :63] = np.tile(np.array([1.0, 0.0, 0.0]), 21)
    actions[:, 63:] = np.tile(np.array([0.0, 1.0, 0.0]), 21)
    identity = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    wrists = np.concatenate((identity, identity)).astype(np.float32)
    batch = {
        "actions_keypoints": actions,
        "viz_current_wrist_poses": wrists,
    }

    out = _apply(build_bimanual_keypoint_wrist_revert_transforms(), batch)[
        "actions_keypoints"
    ]

    assert out.shape == (4, 126)
    np.testing.assert_allclose(out[:, :63], actions[:, :63])
    np.testing.assert_allclose(out[:, 63:], actions[:, 63:])


def test_eva_wrist_actions_revert_through_each_robot_base_into_camera():
    identity_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    left_state = np.concatenate(([1.0, 2.0, 4.0], identity_rot6d, [0.0]))
    right_state = np.concatenate(([-2.0, 1.0, 3.0], identity_rot6d, [0.0]))
    state = np.concatenate((left_state, right_state))[None]

    left_wrist = np.concatenate(([1.0, 0.0, 1.0], identity_rot6d, [0.25]))
    right_wrist = np.concatenate(([0.0, 2.0, 0.0], identity_rot6d, [0.75]))
    actions = np.repeat(np.concatenate((left_wrist, right_wrist))[None], 3, axis=0)

    left_extrinsics = np.eye(4)
    left_extrinsics[:3, :3] = Rotation.from_euler(
        "z", 90.0, degrees=True
    ).as_matrix()
    left_extrinsics[:3, 3] = [1.0, 2.0, 3.0]
    right_extrinsics = np.eye(4)
    right_extrinsics[:3, :3] = Rotation.from_euler(
        "z", -90.0, degrees=True
    ).as_matrix()
    right_extrinsics[:3, 3] = [-2.0, 1.0, 1.0]
    batch = {
        "actions_cartesian": actions,
        "state_ee_pose": state,
        "left_camera_extrinsics": left_extrinsics,
        "right_camera_extrinsics": right_extrinsics,
    }

    out = _apply(build_eva_rot6d_wrist_to_camera_transforms(), batch)[
        "actions_cartesian"
    ]

    np.testing.assert_allclose(out[:, :3], [[0.0, -1.0, 2.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(out[:, 7:10], [[-2.0, 0.0, 2.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(out[:, 6], 0.25)
    np.testing.assert_allclose(out[:, 13], 0.75)


def test_eva_camera_revert_rejects_nonfinite_extrinsics():
    left_extrinsics = np.eye(4)
    left_extrinsics[0, 0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        BimanualCartesianBaseToCamera().transform(
            {
                "actions_cartesian": np.zeros((2, 14)),
                "left_camera_extrinsics": left_extrinsics,
                "right_camera_extrinsics": np.eye(4),
            }
        )
