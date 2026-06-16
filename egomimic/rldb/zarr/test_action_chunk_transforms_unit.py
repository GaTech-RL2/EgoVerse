import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    InterpolatePose,
    XYZWXYZ_to_XYZYPR,
    build_aria_bimanual_transform_list,
    build_eva_bimanual_transform_list,
)
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


def _shape_map(batch: dict) -> dict[str, tuple]:
    return {k: tuple(np.asarray(v).shape) for k, v in batch.items()}


def _run_and_capture(transform_list, batch: dict):
    snapshots = []
    data = {k: np.asarray(v).copy() for k, v in batch.items()}
    for transform in transform_list:
        data = transform.transform(data)
        snapshots.append(
            (transform.__class__.__name__, set(data.keys()), _shape_map(data))
        )
    return snapshots


def _assert_snapshot(
    snapshots,
    idx: int,
    expected_name: str,
    expected_keys: set[str],
    expected_shapes: dict[str, tuple],
) -> None:
    name, keys, shapes = snapshots[idx]
    assert name == expected_name, (
        f"step {idx}: transform mismatch; expected {expected_name}, got {name}"
    )

    missing = expected_keys - keys
    extra = keys - expected_keys
    assert not missing and not extra, (
        f"step {idx} ({name}): key set mismatch; missing={sorted(missing)}, "
        f"extra={sorted(extra)}"
    )

    shape_mismatch = {
        k: (expected_shapes[k], shapes.get(k))
        for k in expected_shapes
        if shapes.get(k) != expected_shapes[k]
    }
    assert not shape_mismatch, f"step {idx} ({name}): shape mismatch {shape_mismatch}"


def test_xyzwxyz_to_matrix_converts_wxyz_quaternion() -> None:
    yaw = np.pi / 2.0
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)

    poses = np.array(
        [
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, qw, 0.0, 0.0, qz],
        ],
        dtype=np.float64,
    )

    mats = _xyzwxyz_to_matrix(poses)

    np.testing.assert_allclose(mats[0, :3, 3], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(mats[0, :3, :3], np.eye(3), atol=1e-7)

    expected_rot = R.from_euler("Z", yaw, degrees=False).as_matrix()
    np.testing.assert_allclose(mats[1, :3, :3], expected_rot, atol=1e-7)
    np.testing.assert_allclose(mats[1, :3, 3], np.zeros(3), atol=1e-7)


def test_action_chunk_coordinate_frame_transform_accepts_quat_wxyz_input() -> None:
    yaw = np.pi / 2.0
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)

    transform = ActionChunkCoordinateFrameTransform(
        target_world="target_world",
        chunk_world="chunk_world",
        transformed_key_name="chunk_target",
        is_quat=True,
    )

    batch = {
        "target_world": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "chunk_world": np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, qw, 0.0, 0.0, qz],
            ],
            dtype=np.float64,
        ),
    }

    out = transform.transform(batch)
    chunk_target = np.asarray(out["chunk_target"])

    assert chunk_target.shape == (2, 7)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, qw, 0.0, 0.0, qz],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(chunk_target, expected, atol=1e-6)


def test_action_chunk_coordinate_frame_transform_ypr_mode_invalid_chunk_shape_raises() -> (
    None
):
    transform = ActionChunkCoordinateFrameTransform(
        target_world="target_world",
        chunk_world="chunk_world",
        transformed_key_name="chunk_target",
        is_quat=False,
    )
    batch = {
        "target_world": np.zeros(6, dtype=np.float64),
        "chunk_world": np.zeros((2, 7), dtype=np.float64),
    }

    with pytest.raises(ValueError, match=r"Expected \(B, 6\) array"):
        transform.transform(batch)


def test_action_chunk_coordinate_frame_transform_quat_mode_invalid_chunk_shape_raises() -> (
    None
):
    transform = ActionChunkCoordinateFrameTransform(
        target_world="target_world",
        chunk_world="chunk_world",
        transformed_key_name="chunk_target",
        is_quat=True,
    )
    batch = {
        "target_world": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "chunk_world": np.zeros((2, 6), dtype=np.float64),
    }

    with pytest.raises(ValueError, match=r"Expected \(B, 7\) array"):
        transform.transform(batch)


def test_action_chunk_coordinate_frame_transform_invalid_target_shape_raises() -> None:
    transform = ActionChunkCoordinateFrameTransform(
        target_world="target_world",
        chunk_world="chunk_world",
        transformed_key_name="chunk_target",
        is_quat=False,
    )
    batch = {
        "target_world": np.zeros((2, 6), dtype=np.float64),
        "chunk_world": np.zeros((2, 6), dtype=np.float64),
    }

    with pytest.raises(ValueError, match=r"Expected \(B, 6\) array"):
        transform.transform(batch)


def test_interpolate_pose_quat_wxyz_slerp_happy_path() -> None:
    yaw = np.pi / 2.0
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)

    transform = InterpolatePose(
        new_chunk_length=5,
        action_key="actions",
        output_action_key="actions_out",
        is_quat=True,
    )
    batch = {
        "actions": np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, qw, 0.0, 0.0, qz],
            ],
            dtype=np.float64,
        )
    }

    out = np.asarray(transform.transform(batch)["actions_out"])
    assert out.shape == (5, 7)

    quat_norms = np.linalg.norm(out[:, 3:7], axis=1)
    np.testing.assert_allclose(quat_norms, np.ones(5), atol=1e-7)
    np.testing.assert_allclose(out[0], batch["actions"][0], atol=1e-7)
    np.testing.assert_allclose(out[-1], batch["actions"][-1], atol=1e-7)

    yaws = R.from_quat(out[:, [4, 5, 6, 3]]).as_euler("ZYX", degrees=False)[:, 0]
    np.testing.assert_allclose(yaws[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(yaws[-1], yaw, atol=1e-7)
    assert np.all(np.diff(yaws) >= -1e-7)


def test_interpolate_pose_quat_wxyz_invalid_shape_raises() -> None:
    transform = InterpolatePose(
        new_chunk_length=4,
        action_key="actions",
        output_action_key="actions_out",
        is_quat=True,
    )
    batch = {"actions": np.zeros((2, 6), dtype=np.float64)}
    with pytest.raises(ValueError, match=r"InterpolatePose expects \(T, 7\)"):
        transform.transform(batch)


def test_xyzwxyz_to_xyzypr_single_pose_conversion() -> None:
    yaw = np.pi / 2.0
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)

    transform = XYZWXYZ_to_XYZYPR(keys=["pose"])
    batch = {"pose": np.array([1.0, 2.0, 3.0, qw, 0.0, 0.0, qz], dtype=np.float64)}

    out = np.asarray(transform.transform(batch)["pose"])
    assert out.shape == (6,)
    np.testing.assert_allclose(out[:3], np.array([1.0, 2.0, 3.0]), atol=1e-7)
    np.testing.assert_allclose(out[3:], np.array([yaw, 0.0, 0.0]), atol=1e-6)


def test_xyzwxyz_to_xyzypr_chunk_conversion() -> None:
    yaw = np.pi / 2.0
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)

    transform = XYZWXYZ_to_XYZYPR(keys=["poses"])
    batch = {
        "poses": np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, qw, 0.0, 0.0, qz],
            ],
            dtype=np.float64,
        )
    }

    out = np.asarray(transform.transform(batch)["poses"])
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out[0, 3:], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(out[1, 3:], np.array([yaw, 0.0, 0.0]), atol=1e-6)


def test_xyzwxyz_to_xyzypr_strict_shape_raises() -> None:
    transform = XYZWXYZ_to_XYZYPR(keys=["poses"])
    batch = {"poses": np.zeros((2, 6), dtype=np.float64)}
    with pytest.raises(
        ValueError, match=r"XYZWXYZ_to_XYZYPR expects key 'poses' to have shape"
    ):
        transform.transform(batch)


def test_eva_builder_orders_xyzwxyz_to_xyzypr_after_interpolate_before_concat() -> None:
    transform_list = build_eva_bimanual_transform_list(is_quat=True)
    converter_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, XYZWXYZ_to_XYZYPR)
    ]
    interpolate_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, InterpolatePose)
    ]
    concat_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, ConcatKeys)
    ]

    assert len(converter_indices) == 1
    converter_idx = converter_indices[0]
    assert converter_idx > max(interpolate_indices)
    assert converter_idx < min(concat_indices)
    assert set(transform_list[converter_idx].keys) == {
        "left.cmd_ee_pose_camframe",
        "right.cmd_ee_pose_camframe",
        "left.obs_ee_pose",
        "right.obs_ee_pose",
    }


def test_aria_builder_orders_xyzwxyz_to_xyzypr_after_interpolate_before_concat() -> (
    None
):
    transform_list = build_aria_bimanual_transform_list(target_world_is_quat=True)
    converter_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, XYZWXYZ_to_XYZYPR)
    ]
    interpolate_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, InterpolatePose)
    ]
    concat_indices = [
        i for i, t in enumerate(transform_list) if isinstance(t, ConcatKeys)
    ]

    assert len(converter_indices) == 1
    converter_idx = converter_indices[0]
    assert converter_idx > max(interpolate_indices)
    assert converter_idx < min(concat_indices)
    assert set(transform_list[converter_idx].keys) == {
        "left.action_ee_pose_headframe",
        "right.action_ee_pose_headframe",
        "left.obs_ee_pose_headframe",
        "right.obs_ee_pose_headframe",
    }


def test_eva_transform_list_stepwise_keys_and_shapes() -> None:
    transform_list = build_eva_bimanual_transform_list(
        chunk_length=4, stride=1, is_quat=True
    )
    cmd_pose = np.zeros((5, 7), dtype=np.float64)
    cmd_pose[:, 3] = 1.0
    obs_pose = np.zeros((7,), dtype=np.float64)
    obs_pose[3] = 1.0
    batch = {
        "left_extrinsics_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "right_extrinsics_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "left.cmd_ee_pose": cmd_pose.copy(),
        "right.cmd_ee_pose": cmd_pose.copy(),
        "left.obs_ee_pose": obs_pose.copy(),
        "right.obs_ee_pose": obs_pose.copy(),
        "left.gripper": np.zeros((5, 1), dtype=np.float64),
        "right.gripper": np.zeros((5, 1), dtype=np.float64),
        "left.obs_gripper": np.zeros((1,), dtype=np.float64),
        "right.obs_gripper": np.zeros((1,), dtype=np.float64),
    }
    snapshots = _run_and_capture(transform_list, batch)

    expected_names = [
        "ActionChunkCoordinateFrameTransform",
        "ActionChunkCoordinateFrameTransform",
        "PoseCoordinateFrameTransform",
        "PoseCoordinateFrameTransform",
        "InterpolatePose",
        "InterpolatePose",
        "InterpolateLinear",
        "InterpolateLinear",
        "XYZWXYZ_to_XYZYPR",
        "ConcatKeys",
        "ConcatKeys",
        "DeleteKeys",
    ]
    assert [name for name, _, _ in snapshots] == expected_names

    base_keys = {
        "left_extrinsics_pose",
        "right_extrinsics_pose",
        "left.cmd_ee_pose",
        "right.cmd_ee_pose",
        "left.obs_ee_pose",
        "right.obs_ee_pose",
        "left.gripper",
        "right.gripper",
        "left.obs_gripper",
        "right.obs_gripper",
    }

    _assert_snapshot(
        snapshots,
        0,
        "ActionChunkCoordinateFrameTransform",
        base_keys | {"left.cmd_ee_pose_camframe"},
        {
            "left_extrinsics_pose": (7,),
            "right_extrinsics_pose": (7,),
            "left.cmd_ee_pose": (5, 7),
            "right.cmd_ee_pose": (5, 7),
            "left.obs_ee_pose": (7,),
            "right.obs_ee_pose": (7,),
            "left.gripper": (5, 1),
            "right.gripper": (5, 1),
            "left.obs_gripper": (1,),
            "right.obs_gripper": (1,),
            "left.cmd_ee_pose_camframe": (5, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        1,
        "ActionChunkCoordinateFrameTransform",
        base_keys | {"left.cmd_ee_pose_camframe", "right.cmd_ee_pose_camframe"},
        {
            "left.cmd_ee_pose_camframe": (5, 7),
            "right.cmd_ee_pose_camframe": (5, 7),
            "left.obs_ee_pose": (7,),
            "right.obs_ee_pose": (7,),
        },
    )
    _assert_snapshot(
        snapshots,
        4,
        "InterpolatePose",
        base_keys | {"left.cmd_ee_pose_camframe", "right.cmd_ee_pose_camframe"},
        {
            "left.cmd_ee_pose_camframe": (4, 7),
            "right.cmd_ee_pose_camframe": (5, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        5,
        "InterpolatePose",
        base_keys | {"left.cmd_ee_pose_camframe", "right.cmd_ee_pose_camframe"},
        {
            "left.cmd_ee_pose_camframe": (4, 7),
            "right.cmd_ee_pose_camframe": (4, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        8,
        "XYZWXYZ_to_XYZYPR",
        base_keys | {"left.cmd_ee_pose_camframe", "right.cmd_ee_pose_camframe"},
        {
            "left.cmd_ee_pose_camframe": (4, 6),
            "right.cmd_ee_pose_camframe": (4, 6),
            "left.obs_ee_pose": (6,),
            "right.obs_ee_pose": (6,),
        },
    )
    _assert_snapshot(
        snapshots,
        9,
        "ConcatKeys",
        {
            "actions_cartesian",
            "left_extrinsics_pose",
            "right_extrinsics_pose",
            "left.cmd_ee_pose",
            "right.cmd_ee_pose",
            "left.obs_ee_pose",
            "right.obs_ee_pose",
            "left.obs_gripper",
            "right.obs_gripper",
        },
        {
            "actions_cartesian": (4, 14),
            "left.obs_ee_pose": (6,),
            "right.obs_ee_pose": (6,),
        },
    )
    _assert_snapshot(
        snapshots,
        10,
        "ConcatKeys",
        {
            "actions_cartesian",
            "observations.state.ee_pose",
            "left_extrinsics_pose",
            "right_extrinsics_pose",
            "left.cmd_ee_pose",
            "right.cmd_ee_pose",
        },
        {
            "actions_cartesian": (4, 14),
            "observations.state.ee_pose": (14,),
        },
    )
    _assert_snapshot(
        snapshots,
        11,
        "DeleteKeys",
        {"actions_cartesian", "observations.state.ee_pose"},
        {
            "actions_cartesian": (4, 14),
            "observations.state.ee_pose": (14,),
        },
    )


def test_aria_transform_list_stepwise_keys_and_shapes() -> None:
    transform_list = build_aria_bimanual_transform_list(
        chunk_length=4,
        stride=2,
        target_world_is_quat=True,
        left_action_world="left.action_ee_pose",
        right_action_world="right.action_ee_pose",
    )
    action_pose = np.zeros((6, 7), dtype=np.float64)
    action_pose[:, 3] = 1.0
    obs_pose = np.zeros((7,), dtype=np.float64)
    obs_pose[3] = 1.0
    batch = {
        "obs_head_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "left.action_ee_pose": action_pose.copy(),
        "right.action_ee_pose": action_pose.copy(),
        "left.obs_ee_pose": obs_pose.copy(),
        "right.obs_ee_pose": obs_pose.copy(),
    }
    snapshots = _run_and_capture(transform_list, batch)

    expected_names = [
        "ActionChunkCoordinateFrameTransform",
        "ActionChunkCoordinateFrameTransform",
        "PoseCoordinateFrameTransform",
        "PoseCoordinateFrameTransform",
        "InterpolatePose",
        "InterpolatePose",
        "XYZWXYZ_to_XYZYPR",
        "ConcatKeys",
        "ConcatKeys",
        "DeleteKeys",
    ]
    assert [name for name, _, _ in snapshots] == expected_names

    base_keys = {
        "obs_head_pose",
        "left.action_ee_pose",
        "right.action_ee_pose",
        "left.obs_ee_pose",
        "right.obs_ee_pose",
    }
    _assert_snapshot(
        snapshots,
        0,
        "ActionChunkCoordinateFrameTransform",
        base_keys | {"left.action_ee_pose_headframe"},
        {
            "left.action_ee_pose_headframe": (6, 7),
            "left.obs_ee_pose": (7,),
            "right.obs_ee_pose": (7,),
        },
    )
    _assert_snapshot(
        snapshots,
        1,
        "ActionChunkCoordinateFrameTransform",
        base_keys | {"left.action_ee_pose_headframe", "right.action_ee_pose_headframe"},
        {
            "left.action_ee_pose_headframe": (6, 7),
            "right.action_ee_pose_headframe": (6, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        4,
        "InterpolatePose",
        base_keys
        | {
            "left.action_ee_pose_headframe",
            "right.action_ee_pose_headframe",
            "left.obs_ee_pose_headframe",
            "right.obs_ee_pose_headframe",
        },
        {
            "left.action_ee_pose_headframe": (4, 7),
            "right.action_ee_pose_headframe": (6, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        5,
        "InterpolatePose",
        base_keys
        | {
            "left.action_ee_pose_headframe",
            "right.action_ee_pose_headframe",
            "left.obs_ee_pose_headframe",
            "right.obs_ee_pose_headframe",
        },
        {
            "left.action_ee_pose_headframe": (4, 7),
            "right.action_ee_pose_headframe": (4, 7),
        },
    )
    _assert_snapshot(
        snapshots,
        6,
        "XYZWXYZ_to_XYZYPR",
        base_keys
        | {
            "left.action_ee_pose_headframe",
            "right.action_ee_pose_headframe",
            "left.obs_ee_pose_headframe",
            "right.obs_ee_pose_headframe",
        },
        {
            "left.action_ee_pose_headframe": (4, 6),
            "right.action_ee_pose_headframe": (4, 6),
            "left.obs_ee_pose_headframe": (6,),
            "right.obs_ee_pose_headframe": (6,),
        },
    )
    _assert_snapshot(
        snapshots,
        7,
        "ConcatKeys",
        base_keys
        | {
            "left.obs_ee_pose_headframe",
            "right.obs_ee_pose_headframe",
            "actions_cartesian",
        },
        {
            "actions_cartesian": (4, 12),
            "left.obs_ee_pose_headframe": (6,),
            "right.obs_ee_pose_headframe": (6,),
        },
    )
    _assert_snapshot(
        snapshots,
        8,
        "ConcatKeys",
        base_keys | {"actions_cartesian", "observations.state.ee_pose"},
        {
            "actions_cartesian": (4, 12),
            "observations.state.ee_pose": (12,),
        },
    )
    _assert_snapshot(
        snapshots,
        9,
        "DeleteKeys",
        {"actions_cartesian", "observations.state.ee_pose"},
        {
            "actions_cartesian": (4, 12),
            "observations.state.ee_pose": (12,),
        },
    )
