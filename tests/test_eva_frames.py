import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment.eva_frames import (
    EVA_DATASET_FROM_HARDWARE_ROTATION,
    dataset_ypr_pose_to_hardware_ypr,
    hardware_ypr_pose_to_dataset_wxyz,
    hardware_ypr_pose_to_dataset_ypr,
)


def test_eva_hardware_dataset_pose_roundtrip_matches_frozen_matrix():
    np.testing.assert_array_equal(
        EVA_DATASET_FROM_HARDWARE_ROTATION,
        np.asarray([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64),
    )
    hardware = np.asarray(
        [
            [0.31, 0.22, 0.41, 0.37, -0.41, 0.22],
            [0.46, -0.18, 0.29, -0.61, 0.28, -0.35],
        ],
        dtype=np.float64,
    )
    expected_rotation = (
        EVA_DATASET_FROM_HARDWARE_ROTATION
        @ Rotation.from_euler("ZYX", hardware[:, 3:6]).as_matrix()
    )

    dataset_ypr = hardware_ypr_pose_to_dataset_ypr(hardware)
    dataset_wxyz = hardware_ypr_pose_to_dataset_wxyz(hardware)
    actual_ypr_rotation = Rotation.from_euler("ZYX", dataset_ypr[:, 3:6]).as_matrix()
    actual_wxyz_rotation = Rotation.from_quat(dataset_wxyz[:, [4, 5, 6, 3]]).as_matrix()

    np.testing.assert_array_equal(dataset_ypr[:, :3], hardware[:, :3])
    np.testing.assert_array_equal(dataset_wxyz[:, :3], hardware[:, :3])
    np.testing.assert_allclose(actual_ypr_rotation, expected_rotation, atol=1e-12)
    np.testing.assert_allclose(actual_wxyz_rotation, expected_rotation, atol=1e-12)

    recovered = dataset_ypr_pose_to_hardware_ypr(dataset_ypr)
    np.testing.assert_array_equal(recovered[:, :3], hardware[:, :3])
    np.testing.assert_allclose(
        Rotation.from_euler("ZYX", recovered[:, 3:6]).as_matrix(),
        Rotation.from_euler("ZYX", hardware[:, 3:6]).as_matrix(),
        atol=1e-12,
    )

    single_ypr = hardware_ypr_pose_to_dataset_ypr(hardware[0])
    single_wxyz = hardware_ypr_pose_to_dataset_wxyz(hardware[0])
    assert single_ypr.shape == (6,)
    assert single_wxyz.shape == (7,)
    assert single_ypr.dtype == np.float64
    assert single_wxyz.dtype == np.float64


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (hardware_ypr_pose_to_dataset_ypr, np.zeros(5)),
        (hardware_ypr_pose_to_dataset_wxyz, np.zeros(7)),
        (dataset_ypr_pose_to_hardware_ypr, np.zeros(5)),
        (
            hardware_ypr_pose_to_dataset_ypr,
            np.asarray([0.0, 0.0, 0.0, np.nan, 0.0, 0.0]),
        ),
    ],
)
def test_eva_frame_helpers_reject_invalid_poses(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_eva_to_zarr_split_uses_shared_frame_for_both_arms():
    from egomimic.scripts.eva_process.eva_to_zarr import _split_per_arm

    left = np.asarray(
        [
            [0.31, 0.22, 0.41, 0.37, -0.41, 0.22, 0.2],
            [0.34, 0.19, 0.43, 0.42, -0.33, 0.17, 0.3],
        ],
        dtype=np.float64,
    )
    right = np.asarray(
        [
            [0.46, -0.18, 0.29, -0.61, 0.28, -0.35, 0.8],
            [0.43, -0.15, 0.32, -0.55, 0.31, -0.27, 0.7],
        ],
        dtype=np.float64,
    )
    combined = np.concatenate([left, right], axis=-1)
    split = _split_per_arm(
        {
            "obs_ee_pose": combined,
            "cmd_ee_pose": combined.copy(),
            "obs_joints": combined.copy(),
            "cmd_joints": combined.copy(),
            "timestamps": np.asarray([1.0, 2.0]),
        },
        "both",
    )

    frozen_matrix = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    for side, source in (("left", left), ("right", right)):
        expected_rotation = (
            frozen_matrix @ Rotation.from_euler("ZYX", source[:, 3:6]).as_matrix()
        )
        stored = split[f"{side}.obs_ee_pose"]
        stored_rotation = Rotation.from_quat(stored[:, [4, 5, 6, 3]]).as_matrix()
        np.testing.assert_array_equal(stored[:, :3], source[:, :3])
        np.testing.assert_allclose(stored_rotation, expected_rotation, atol=1e-12)
        np.testing.assert_array_equal(split[f"{side}.obs_gripper"], source[:, 6:7])
        np.testing.assert_array_equal(split[f"{side}.cmd_gripper"], source[:, 6:7])
    np.testing.assert_array_equal(split["timestamps"], [1.0, 2.0])
