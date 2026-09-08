"""rot6d (continuous 6D rotation) representation: pose helpers, the two
conversion transforms, the Human `cartesian_rot6d` mode and the viz split."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
from egomimic.rldb.zarr.action_chunk_transforms import (
    XYZRot6D_to_XYZYPR,
    XYZWXYZ_to_XYZRot6D,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzrot6d,
    _split_action_pose,
    _xyzrot6d_to_matrix,
    _xyzwxyz_to_matrix,
)


def _random_xyzwxyz(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quat_xyzw = R.random(n, random_state=seed).as_quat()
    xyz = rng.normal(size=(n, 3))
    return np.concatenate([xyz, quat_xyzw[:, [3, 0, 1, 2]]], axis=-1)


def test_rot6d_is_first_two_rows_and_round_trips() -> None:
    poses = _random_xyzwxyz(32)
    mats = _xyzwxyz_to_matrix(poses)
    rot6d = _matrix_to_xyzrot6d(mats)
    assert rot6d.shape == (32, 9)
    np.testing.assert_allclose(rot6d[:, :3], mats[:, :3, 3])
    np.testing.assert_allclose(rot6d[:, 3:6], mats[:, 0, :3])
    np.testing.assert_allclose(rot6d[:, 6:9], mats[:, 1, :3])
    # every component is bounded, unlike euler angles
    assert np.all(np.abs(rot6d[:, 3:]) <= 1.0 + 1e-12)
    back = _xyzrot6d_to_matrix(rot6d)
    np.testing.assert_allclose(back, mats, atol=1e-10)


def test_rot6d_inverse_is_gram_schmidt_tolerant() -> None:
    poses = _random_xyzwxyz(8, seed=1)
    mats = _xyzwxyz_to_matrix(poses)
    rot6d = _matrix_to_xyzrot6d(mats)
    noisy = rot6d.copy()
    noisy[:, 3:] += np.random.default_rng(1).normal(scale=0.05, size=(8, 6))
    back = _xyzrot6d_to_matrix(noisy)[:, :3, :3]
    # proper rotation matrices regardless of the noise
    eye = np.broadcast_to(np.eye(3), back.shape)
    np.testing.assert_allclose(back @ back.transpose(0, 2, 1), eye, atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(back), 1.0, atol=1e-10)


def test_rot6d_is_continuous_where_ypr_wraps() -> None:
    # Yaw sweeping through +-pi: ypr jumps by ~2pi, rot6d moves smoothly.
    yaw = np.linspace(np.pi - 0.05, np.pi + 0.05, 11)
    quat_xyzw = R.from_euler(
        "ZYX", np.stack([yaw, np.zeros(11), np.zeros(11)], 1)
    ).as_quat()
    poses = np.concatenate([np.zeros((11, 3)), quat_xyzw[:, [3, 0, 1, 2]]], axis=-1)
    ypr = XYZWXYZ_to_XYZYPR(keys=["p"]).transform({"p": poses.copy()})["p"]
    rot6d = XYZWXYZ_to_XYZRot6D(keys=["p"]).transform({"p": poses.copy()})["p"]
    assert rot6d.shape == (11, 9)
    assert np.abs(np.diff(ypr[:, 3])).max() > 6.0
    assert np.abs(np.diff(rot6d[:, 3:], axis=0)).max() < 0.02


def test_rot6d_transforms_single_pose_and_strict_shapes() -> None:
    pose = _random_xyzwxyz(1)[0]
    out = XYZWXYZ_to_XYZRot6D(keys=["p"]).transform({"p": pose.copy()})["p"]
    assert out.shape == (9,)
    ypr = XYZRot6D_to_XYZYPR(keys=["p"]).transform({"p": out.copy()})["p"]
    ref = XYZWXYZ_to_XYZYPR(keys=["p"]).transform({"p": pose.copy()})["p"]
    np.testing.assert_allclose(ypr, ref, atol=1e-10)
    with pytest.raises(ValueError, match="XYZWXYZ_to_XYZRot6D expects"):
        XYZWXYZ_to_XYZRot6D(keys=["p"]).transform({"p": np.zeros((4, 6))})
    with pytest.raises(ValueError, match="XYZRot6D_to_XYZYPR expects"):
        XYZRot6D_to_XYZYPR(keys=["p"]).transform({"p": np.zeros((4, 7))})


def test_human_cartesian_rot6d_mode_yields_18_dims() -> None:
    from egomimic.rldb.embodiment.human import Human

    transform_list = Human.get_transform_list(
        mode="cartesian_rot6d", stride=1, chunk_length=5
    )
    names = [type(t).__name__ for t in transform_list]
    assert "XYZWXYZ_to_XYZRot6D" in names and "XYZWXYZ_to_XYZYPR" not in names
    ref_names = [
        type(t).__name__
        for t in Human.get_transform_list(mode="cartesian", stride=1, chunk_length=5)
    ]
    assert [
        n.replace("XYZWXYZ_to_XYZYPR", "XYZWXYZ_to_XYZRot6D") for n in ref_names
    ] == names

    action_pose = _random_xyzwxyz(6, seed=2)
    obs_pose = _random_xyzwxyz(1, seed=3)[0]
    batch = {
        "obs_head_pose": _random_xyzwxyz(1, seed=4)[0],
        "left.action_ee_pose": action_pose.copy(),
        "right.action_ee_pose": action_pose.copy(),
        "left.obs_ee_pose": obs_pose.copy(),
        "right.obs_ee_pose": obs_pose.copy(),
    }
    for t in transform_list:
        batch = t.transform(batch)
    assert batch["actions_cartesian"].shape == (5, 18)
    assert batch["observations.state.ee_pose"].shape == (18,)
    # the viz split converts rot6d back to ypr; compare with the ypr pipeline
    ypr_batch = {
        "obs_head_pose": _random_xyzwxyz(1, seed=4)[0],
        "left.action_ee_pose": action_pose.copy(),
        "right.action_ee_pose": action_pose.copy(),
        "left.obs_ee_pose": obs_pose.copy(),
        "right.obs_ee_pose": obs_pose.copy(),
    }
    for t in Human.get_transform_list(mode="cartesian", stride=1, chunk_length=5):
        ypr_batch = t.transform(ypr_batch)
    lx, lr, rx, rr = _split_action_pose(batch["actions_cartesian"])
    lx0, lr0, rx0, rr0 = _split_action_pose(ypr_batch["actions_cartesian"])
    np.testing.assert_allclose(lx, lx0, atol=1e-10)
    np.testing.assert_allclose(rx, rx0, atol=1e-10)
    # same rotation up to the euler wrap
    d = R.from_euler("ZYX", lr).inv() * R.from_euler("ZYX", lr0)
    assert d.magnitude().max() < 1e-8
    d = R.from_euler("ZYX", rr).inv() * R.from_euler("ZYX", rr0)
    assert d.magnitude().max() < 1e-8
