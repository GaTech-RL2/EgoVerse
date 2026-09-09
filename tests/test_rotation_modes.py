"""Rotation-representation plumbing shared by Eva and Human transform lists."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.action_chunk_transforms import (
    PadGripperZeros,
    XYZWXYZ_to_XYZRot6D,
    XYZWXYZ_to_XYZYPR,
    transforms_for_rotation_mode,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzrot6d,
    _xyzrot6d_to_matrix,
    _xyzwxyz_to_matrix,
)


def _random_poses(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quats_xyzw = R.random(n, random_state=seed).as_quat()
    wxyz = np.concatenate([quats_xyzw[:, 3:4], quats_xyzw[:, :3]], axis=-1)
    return np.concatenate([rng.normal(size=(n, 3)), wxyz], axis=-1)


def test_rot6d_round_trips_through_matrix():
    mats = _xyzwxyz_to_matrix(_random_poses(8, seed=1))
    recovered = _xyzrot6d_to_matrix(_matrix_to_xyzrot6d(mats))
    np.testing.assert_allclose(recovered, mats, atol=1e-10)


def test_rot6d_stores_first_two_columns_column_major():
    mats = _xyzwxyz_to_matrix(_random_poses(3, seed=2))
    rot6d = _matrix_to_xyzrot6d(mats)
    np.testing.assert_allclose(rot6d[:, 3:6], mats[:, :3, 0])
    np.testing.assert_allclose(rot6d[:, 6:9], mats[:, :3, 1])


def test_rot6d_gram_schmidt_orthonormalises_a_perturbed_input():
    mats = _xyzwxyz_to_matrix(_random_poses(4, seed=3))
    rot6d = _matrix_to_xyzrot6d(mats)
    rot6d[:, 3:9] += 1e-3  # a denoiser never emits an exactly orthonormal pair
    rotations = _xyzrot6d_to_matrix(rot6d)[:, :3, :3]
    identity = np.einsum("bij,bkj->bik", rotations, rotations)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), (4, 3, 3)), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(rotations), np.ones(4), atol=1e-9)


@pytest.mark.parametrize(
    "rotation_mode, types, width",
    [
        ("quat", (), 7),
        ("euler", (XYZWXYZ_to_XYZYPR,), 6),
        ("6D", (XYZWXYZ_to_XYZRot6D,), 9),
    ],
)
def test_transforms_for_rotation_mode_selects_and_sizes(rotation_mode, types, width):
    transforms = transforms_for_rotation_mode(keys=["pose"], rotation_mode=rotation_mode)
    assert tuple(type(t) for t in transforms) == types

    batch = {"pose": _random_poses(5, seed=4)}
    for transform in transforms:
        batch = transform.transform(batch)
    assert batch["pose"].shape == (5, width)


def test_transforms_for_rotation_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown rotation_mode"):
        transforms_for_rotation_mode(keys=["pose"], rotation_mode="ypr")


def test_rot6d_transform_accepts_a_single_unbatched_pose():
    batch = {"pose": _random_poses(1, seed=5)[0]}
    out = XYZWXYZ_to_XYZRot6D(keys=["pose"]).transform(batch)
    assert out["pose"].shape == (9,)


def test_rot6d_transform_rejects_a_non_quaternion_width():
    with pytest.raises(ValueError, match="shape"):
        XYZWXYZ_to_XYZRot6D(keys=["pose"]).transform({"pose": np.zeros((4, 6))})


@pytest.mark.parametrize("pose_dim, expected", [(6, 14), (7, 16), (9, 20)])
def test_pad_gripper_zeros_follows_pose_dim(pose_dim, expected):
    chunk = np.arange(3 * 2 * pose_dim, dtype=np.float64).reshape(3, 2 * pose_dim)
    out = PadGripperZeros(action_key="a", pose_dim=pose_dim).transform({"a": chunk})
    padded = out["a"]
    assert padded.shape == (3, expected)
    # A zero slot lands at the end of each arm, and the poses are untouched.
    np.testing.assert_allclose(padded[:, pose_dim], 0.0)
    np.testing.assert_allclose(padded[:, -1], 0.0)
    np.testing.assert_allclose(padded[:, :pose_dim], chunk[:, :pose_dim])
    np.testing.assert_allclose(padded[:, pose_dim + 1 : -1], chunk[:, pose_dim:])


def test_pad_gripper_zeros_reports_the_width_it_wanted():
    with pytest.raises(ValueError, match="last-dim 18"):
        PadGripperZeros(action_key="a", pose_dim=9).transform({"a": np.zeros((2, 12))})


@pytest.mark.parametrize("rotation_mode", ["euler", "quat", "6D"])
@pytest.mark.parametrize("coord_frame", ["camframe", "eef_frame"])
def test_eva_builds_every_rotation_mode(coord_frame, rotation_mode):
    transforms = Eva.get_transform_list(
        action_mode="cartesian", coord_frame=coord_frame, rotation_mode=rotation_mode
    )
    assert transforms
    kinds = {type(t) for t in transforms}
    assert (XYZWXYZ_to_XYZRot6D in kinds) == (rotation_mode == "6D")
    assert (XYZWXYZ_to_XYZYPR in kinds) == (rotation_mode == "euler")


def test_eva_rejects_unknown_axes():
    with pytest.raises(ValueError, match="unknown coord_frame"):
        Eva.get_transform_list(coord_frame="headframe")
    with pytest.raises(ValueError, match="unknown action_mode"):
        Eva.get_transform_list(action_mode="keypoints")


@pytest.mark.parametrize("rotation_mode", ["euler", "quat", "6D"])
@pytest.mark.parametrize("action_mode", ["cartesian", "keypoints"])
def test_human_builds_every_rotation_mode(action_mode, rotation_mode):
    transforms = Human.get_transform_list(
        action_mode=action_mode, coord_frame="camframe", rotation_mode=rotation_mode
    )
    assert transforms
    kinds = {type(t) for t in transforms}
    assert (XYZWXYZ_to_XYZRot6D in kinds) == (rotation_mode == "6D")


@pytest.mark.parametrize("rotation_mode, pose_dim", [("euler", 6), ("quat", 7), ("6D", 9)])
def test_human_gripper_padding_matches_the_rotation_width(rotation_mode, pose_dim):
    transforms = Human.get_transform_list(
        action_mode="cartesian_gripper_padded", rotation_mode=rotation_mode
    )
    pads = [t for t in transforms if isinstance(t, PadGripperZeros)]
    # One for the action chunk, one for the proprio pose.
    assert [t.pose_dim for t in pads] == [pose_dim, pose_dim]
    assert {t.action_key for t in pads} == {
        "actions_cartesian",
        "observations.state.ee_pose",
    }


def test_human_rejects_an_unknown_action_mode():
    with pytest.raises(ValueError, match="Unsupported action_mode"):
        Human.get_transform_list(action_mode="cartesian_padded")
