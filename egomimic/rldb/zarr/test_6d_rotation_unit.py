"""Unit + integration tests for the continuous 6D rotation migration.

Covers the whole 6D path end to end:
  * numpy pose helpers (``_matrix_to_xyz6d`` / ``_xyz6d_to_matrix``) and their
    bit-for-bit parity with the torch Gram-Schmidt used at decode time;
  * the new ``action_chunk_transforms`` building blocks (``XYZWXYZ_to_XYZ6D``,
    ``XYZ6D_to_XYZYPR``, the ``xyz6d`` coordinate-frame mode);
  * the per-embodiment ``cartesian_6d`` / ``cartesian_wristframe_6d`` builders
    (shape progression to 18D human / 20D robot);
  * the 32-dim action converters (``RobotBimanualCartesian6D`` /
    ``HumanBimanualCartesian6D``) round-trips;
  * the wrist-frame revert producing cam-frame ypr, validated to be equivalent
    to the legacy ypr revert;
  * the HPT gripper-padding generalization;
  * the shared bimanual-cartesian index layout used by bounds + eval.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.eva import (
    Eva,
    _build_eva_bimanual_revert_eef_frame_transform_list,
)
from egomimic.rldb.embodiment.human import (
    Aria,
    Mecka,
    Scale,
    _build_aria_cartesian_revert_eef_frame_transform_list,
)
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToXYZ6D,
    ConcatKeys,
    InterpolatePose,
    QuaternionPoseToXYZ6D,
    XYZ6D_to_XYZYPR,
    XYZWXYZ_to_XYZ6D,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.action_utils import (
    HumanBimanualCartesian6D,
    RobotBimanualCartesian6D,
    _reconstruct_R_from_cols,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyz6d,
    _matrix_to_xyzwxyz,
    _matrix_to_xyzypr,
    _xyz6d_to_matrix,
    bimanual_cartesian_layout,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_se3(n: int, seed: int) -> np.ndarray:
    """n random SE3 matrices (proper rotation + translation)."""
    rng = np.random.default_rng(seed)
    rots = R.random(n, random_state=seed).as_matrix()
    mats = np.tile(np.eye(4), (n, 1, 1))
    mats[:, :3, :3] = rots
    mats[:, :3, 3] = rng.uniform(-2.0, 2.0, size=(n, 3))
    return mats


def _xyzwxyz_from_mats(mats: np.ndarray) -> np.ndarray:
    return _matrix_to_xyzwxyz(mats)


# ---------------------------------------------------------------------------
# A. pose_utils 6D helpers
# ---------------------------------------------------------------------------
def test_matrix_to_xyz6d_extracts_translation_and_first_two_columns():
    yaw = 0.7
    mat = np.eye(4)
    mat[:3, :3] = R.from_euler("Z", yaw).as_matrix()
    mat[:3, 3] = [1.0, 2.0, 3.0]

    out = _matrix_to_xyz6d(mat[None])
    assert out.shape == (1, 9)
    np.testing.assert_allclose(out[0, :3], [1.0, 2.0, 3.0])
    # column 1 = [cos, sin, 0], column 2 = [-sin, cos, 0]
    np.testing.assert_allclose(out[0, 3:6], [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-12)
    np.testing.assert_allclose(
        out[0, 6:9], [-np.sin(yaw), np.cos(yaw), 0.0], atol=1e-12
    )


def test_xyz6d_matrix_round_trip_is_identity():
    mats = _random_se3(64, seed=0)
    six = _matrix_to_xyz6d(mats)
    back = _xyz6d_to_matrix(six)
    np.testing.assert_allclose(back, mats, atol=1e-10)


def test_xyz6d_to_matrix_round_trip_from_6d():
    mats = _random_se3(32, seed=1)
    six = _matrix_to_xyz6d(mats)
    six2 = _matrix_to_xyz6d(_xyz6d_to_matrix(six))
    np.testing.assert_allclose(six2, six, atol=1e-10)


def test_xyz6d_to_matrix_orthonormalizes_non_orthonormal_columns():
    # Two arbitrary, non-orthonormal, non-unit columns.
    six = np.array([[5.0, -6.0, 7.0, 2.0, 0.0, 0.0, 0.3, 4.0, 0.0]])
    mat = _xyz6d_to_matrix(six)
    Rm = mat[0, :3, :3]
    # Proper rotation: orthonormal columns, det +1.
    np.testing.assert_allclose(Rm @ Rm.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(Rm), 1.0, atol=1e-10)
    # Translation preserved verbatim.
    np.testing.assert_allclose(mat[0, :3, 3], [5.0, -6.0, 7.0])
    # First column points along the (normalized) input c1 direction.
    np.testing.assert_allclose(Rm[:, 0], [1.0, 0.0, 0.0], atol=1e-10)


def test_xyz6d_to_matrix_is_scale_invariant_in_columns():
    """Gram-Schmidt only uses column *directions*, so scaling them must not
    change the recovered rotation (key property the model relies on)."""
    six = np.array([[0.0, 0.0, 0.0, 1.0, 0.2, -0.1, -0.2, 1.0, 0.05]])
    scaled = six.copy()
    scaled[:, 3:6] *= 3.7
    scaled[:, 6:9] *= 0.25
    np.testing.assert_allclose(
        _xyz6d_to_matrix(six), _xyz6d_to_matrix(scaled), atol=1e-10
    )


def test_numpy_and_torch_gram_schmidt_are_bit_compatible():
    """The numpy revert (``_xyz6d_to_matrix``) and the torch decode
    (``_reconstruct_R_from_cols``) must agree, including on non-orthonormal
    input, or train/eval/deploy would disagree."""
    rng = np.random.default_rng(7)
    c1 = rng.uniform(-3, 3, size=(50, 3))
    c2 = rng.uniform(-3, 3, size=(50, 3))
    xyz = rng.uniform(-1, 1, size=(50, 3))
    six = np.concatenate([xyz, c1, c2], axis=-1)

    np_mats = _xyz6d_to_matrix(six)
    torch_R = _reconstruct_R_from_cols(
        torch.from_numpy(c1)[:, None, :], torch.from_numpy(c2)[:, None, :]
    )[:, 0].numpy()
    np.testing.assert_allclose(np_mats[:, :3, :3], torch_R, atol=1e-12)


def test_xyz6d_helpers_reject_bad_shapes():
    with pytest.raises(ValueError, match=r"Expected \(B, 9\) array"):
        _xyz6d_to_matrix(np.zeros((3, 6)))
    with pytest.raises(ValueError, match=r"Expected \(B, 4, 4\) array"):
        _matrix_to_xyz6d(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# A'. shared bimanual-cartesian index layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [12, 14, 18, 20])
def test_bimanual_layout_partitions_every_channel(width):
    layout = bimanual_cartesian_layout(width)
    assert layout is not None
    all_idx = list(layout["xyz"]) + list(layout["rot"]) + list(layout["grip"])
    assert sorted(all_idx) == list(range(width))  # no overlap, full cover
    assert len(layout["xyz"]) == 6  # 3 per arm


@pytest.mark.parametrize("width,n_grip", [(12, 0), (14, 2), (18, 0), (20, 2)])
def test_bimanual_layout_gripper_counts(width, n_grip):
    assert len(bimanual_cartesian_layout(width)["grip"]) == n_grip


def test_bimanual_layout_unknown_width_is_none():
    for w in (0, 6, 7, 9, 13, 32):
        assert bimanual_cartesian_layout(w) is None


# ---------------------------------------------------------------------------
# B. action_chunk_transforms 6D building blocks
# ---------------------------------------------------------------------------
def test_xyzwxyz_to_xyz6d_single_and_chunk_shapes():
    yaw = np.pi / 3
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    single = XYZWXYZ_to_XYZ6D(keys=["p"]).transform(
        {"p": np.array([1.0, 2.0, 3.0, qw, 0.0, 0.0, qz])}
    )["p"]
    assert single.shape == (9,)
    np.testing.assert_allclose(single[3:6], [np.cos(yaw), np.sin(yaw), 0.0], atol=1e-7)

    chunk = XYZWXYZ_to_XYZ6D(keys=["p"]).transform(
        {"p": np.tile([0, 0, 0, qw, 0, 0, qz], (4, 1)).astype(float)}
    )["p"]
    assert chunk.shape == (4, 9)


def test_xyzwxyz_to_xyz6d_then_xyz6d_to_ypr_matches_direct_ypr():
    """quat -> 6d -> ypr equals quat -> ypr away from gimbal lock."""
    mats = _random_se3(20, seed=3)
    poses = _xyzwxyz_from_mats(mats)

    via6d = XYZWXYZ_to_XYZ6D(keys=["p"]).transform({"p": poses.copy()})["p"]
    via6d = XYZ6D_to_XYZYPR(keys=["p"]).transform({"p": via6d})["p"]
    direct = XYZWXYZ_to_XYZYPR(keys=["p"]).transform({"p": poses.copy()})["p"]
    np.testing.assert_allclose(via6d, direct, atol=1e-7)


def test_xyz6d_to_xypr_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"XYZ6D_to_XYZYPR expects key 'p'"):
        XYZ6D_to_XYZYPR(keys=["p"]).transform({"p": np.zeros((2, 6))})


def test_quaternion_pose_to_xyz6d_single_and_batch_agree():
    mats = _random_se3(5, seed=4)
    poses = _xyzwxyz_from_mats(mats)
    batch = BatchQuaternionPoseToXYZ6D(pose_key="p", output_key="o").transform(
        {"p": poses.copy()}
    )["o"]
    assert batch.shape == (5, 9)
    for i in range(5):
        single = QuaternionPoseToXYZ6D(pose_key="p", output_key="o").transform(
            {"p": poses[i].copy()}
        )["o"]
        np.testing.assert_allclose(single, batch[i], atol=1e-9)


def test_quaternion_pose_to_xyz6d_rejects_bad_shape():
    with pytest.raises(ValueError, match="QuaternionPoseToXYZ6D expects shape"):
        QuaternionPoseToXYZ6D(pose_key="p", output_key="o").transform(
            {"p": np.zeros(9)}
        )


def test_action_chunk_xyz6d_mode_identity_target_preserves_chunk():
    """With an identity target frame, the xyz6d coordinate transform should
    return the input rotation unchanged (columns already orthonormal)."""
    mats = _random_se3(6, seed=5)
    chunk6d = _matrix_to_xyz6d(mats)
    identity6d = _matrix_to_xyz6d(np.eye(4)[None])[0]  # (9,)

    out = ActionChunkCoordinateFrameTransform(
        target_world="t",
        chunk_world="c",
        transformed_key_name="o",
        mode="xyz6d",
        inverse=False,
    ).transform({"t": identity6d, "c": chunk6d})["o"]

    assert out.shape == (6, 9)
    np.testing.assert_allclose(out, chunk6d, atol=1e-9)


def test_action_chunk_xyz6d_mode_dispatches_width9_target():
    """A width-9 target world must be parsed as xyz6d (revert path), giving the
    same result as composing the rotations by hand."""
    target_mat = _random_se3(1, seed=6)[0]
    chunk_mats = _random_se3(4, seed=7)
    target6d = _matrix_to_xyz6d(target_mat[None])[0]
    chunk6d = _matrix_to_xyz6d(chunk_mats)

    out = ActionChunkCoordinateFrameTransform(
        target_world="t",
        chunk_world="c",
        transformed_key_name="o",
        mode="xyz6d",
        inverse=False,
    ).transform({"t": target6d, "c": chunk6d})["o"]

    expected = _matrix_to_xyz6d(target_mat[None] @ chunk_mats)
    np.testing.assert_allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# C. 32-dim action converters
# ---------------------------------------------------------------------------
def test_robot6d_converter_round_trip_and_layout():
    conv = RobotBimanualCartesian6D()
    x = torch.randn(2, 5, 20)
    out32 = conv.to32(x)
    assert out32.shape == (2, 5, 32)
    # 20 native dims land verbatim in the first 20 slots; rest is zero pad.
    torch.testing.assert_close(out32[..., :20], x)
    torch.testing.assert_close(out32[..., 20:], torch.zeros(2, 5, 12))
    torch.testing.assert_close(conv.from32(out32), x)


def test_robot6d_converter_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected 20-dim"):
        RobotBimanualCartesian6D().to32(torch.randn(1, 1, 18))


def test_human6d_converter_round_trip_and_gripper_slots():
    conv = HumanBimanualCartesian6D()
    x = torch.randn(3, 4, 18)
    out32 = conv.to32(x)
    assert out32.shape == (3, 4, 32)
    # left block 0:9 == left arm, right block 10:19 == right arm.
    torch.testing.assert_close(out32[..., 0:9], x[..., 0:9])
    torch.testing.assert_close(out32[..., 10:19], x[..., 9:18])
    # inserted gripper slots (9, 19) are zero.
    torch.testing.assert_close(out32[..., 9], torch.zeros(3, 4))
    torch.testing.assert_close(out32[..., 19], torch.zeros(3, 4))
    torch.testing.assert_close(conv.from32(out32), x)


def test_human6d_converter_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected 18-dim"):
        HumanBimanualCartesian6D().to32(torch.randn(1, 1, 20))


def test_human6d_from32_drops_gripper_slots():
    # from32 must ignore whatever sits in the gripper slots (model may emit
    # non-zero there).
    conv = HumanBimanualCartesian6D()
    a32 = torch.randn(2, 2, 32)
    out = conv.from32(a32)
    assert out.shape == (2, 2, 18)
    torch.testing.assert_close(out[..., 0:9], a32[..., 0:9])
    torch.testing.assert_close(out[..., 9:18], a32[..., 10:19])


def test_2d_input_is_promoted_to_bsd():
    # converters accept (B, D) and unsqueeze a sequence axis.
    out = RobotBimanualCartesian6D().to32(torch.randn(4, 20))
    assert out.shape == (4, 1, 32)


# ---------------------------------------------------------------------------
# D. builder shape progression (integration over the real transform lists)
# ---------------------------------------------------------------------------
def _eva_cartesian_batch(T=5):
    cmd = np.zeros((T, 7))
    cmd[:, 3] = 1.0
    obs = np.zeros((7,))
    obs[3] = 1.0
    return {
        "left.cmd_ee_pose": cmd.copy(),
        "right.cmd_ee_pose": cmd.copy(),
        "left.obs_ee_pose": obs.copy(),
        "right.obs_ee_pose": obs.copy(),
        "left.cmd_gripper": np.zeros((T, 1)),
        "right.cmd_gripper": np.zeros((T, 1)),
        "left.obs_gripper": np.zeros((1,)),
        "right.obs_gripper": np.zeros((1,)),
    }


def _run(transform_list, batch):
    data = {k: np.asarray(v).copy() for k, v in batch.items()}
    for t in transform_list:
        data = t.transform(data)
    return data


def test_eva_cartesian_6d_produces_20d_action_and_obs():
    out = _run(Eva.get_transform_list("cartesian_6d"), _eva_cartesian_batch())
    assert tuple(out["actions_cartesian"].shape)[-1] == 20
    assert tuple(out["observations.state.ee_pose"].shape) == (20,)


def test_eva_cartesian_wristframe_6d_produces_20d():
    out = _run(
        Eva.get_transform_list("cartesian_wristframe_6d"), _eva_cartesian_batch()
    )
    assert tuple(out["actions_cartesian"].shape)[-1] == 20
    assert tuple(out["observations.state.ee_pose"].shape) == (20,)


def _aria_cartesian_batch(T=6):
    act = np.zeros((T, 7))
    act[:, 3] = 1.0
    obs = np.zeros((7,))
    obs[3] = 1.0
    return {
        "obs_head_pose": np.array([0.0, 0, 0, 1.0, 0, 0, 0]),
        "left.action_ee_pose": act.copy(),
        "right.action_ee_pose": act.copy(),
        "left.obs_ee_pose": obs.copy(),
        "right.obs_ee_pose": obs.copy(),
    }


@pytest.mark.parametrize("emb", [Aria, Mecka, Scale])
def test_human_cartesian_6d_produces_18d(emb):
    out = _run(emb.get_transform_list("cartesian_6d"), _aria_cartesian_batch())
    assert tuple(out["actions_cartesian"].shape)[-1] == 18
    assert tuple(out["observations.state.ee_pose"].shape) == (18,)


def test_eva_6d_builder_converter_placed_after_interpolate_before_concat():
    tl = Eva.get_transform_list("cartesian_6d")
    conv = [i for i, t in enumerate(tl) if isinstance(t, XYZWXYZ_to_XYZ6D)]
    interp = [i for i, t in enumerate(tl) if isinstance(t, InterpolatePose)]
    concat = [i for i, t in enumerate(tl) if isinstance(t, ConcatKeys)]
    assert len(conv) == 1
    assert max(interp) < conv[0] < min(concat)


def test_cartesian_6d_rotation_columns_are_bounded():
    """Random ee poses -> the 6D rotation columns stay within [-1, 1] (unit
    matrix columns), which is what makes per-dim normalization meaningful."""
    mats = _random_se3(8, seed=11)
    cmd = _matrix_to_xyzwxyz(mats)
    obs = _matrix_to_xyzwxyz(_random_se3(1, seed=12))[0]
    batch = {
        "obs_head_pose": np.array([0.0, 0, 0, 1.0, 0, 0, 0]),
        "left.action_ee_pose": cmd.copy(),
        "right.action_ee_pose": cmd.copy(),
        "left.obs_ee_pose": obs.copy(),
        "right.obs_ee_pose": obs.copy(),
    }
    out = _run(Aria.get_transform_list("cartesian_6d"), batch)
    layout = bimanual_cartesian_layout(18)
    rot = np.asarray(out["actions_cartesian"])[..., list(layout["rot"])]
    assert rot.min() >= -1.0 - 1e-6 and rot.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# E. revert: model 6D wrist-frame -> cam-frame ypr, equivalent to ypr revert
# ---------------------------------------------------------------------------
def test_human_6d_revert_matches_ypr_revert():
    """The 6D revert (Gram-Schmidt + xyz6d frame compose + collapse to ypr)
    must yield the same cam-frame ypr as the legacy ypr revert fed the
    equivalent rotations."""
    T = 5
    obs_l = _random_se3(1, seed=20)[0]
    obs_r = _random_se3(1, seed=21)[0]
    wrist_l = _random_se3(T, seed=22)
    wrist_r = _random_se3(T, seed=23)

    # ypr inputs (12D obs / per-frame action)
    obs_ypr = np.concatenate(
        [_matrix_to_xyzypr(obs_l[None])[0], _matrix_to_xyzypr(obs_r[None])[0]]
    )
    act_ypr = np.concatenate(
        [_matrix_to_xyzypr(wrist_l), _matrix_to_xyzypr(wrist_r)], axis=-1
    )
    ypr_out = _run(
        _build_aria_cartesian_revert_eef_frame_transform_list(rot_repr="ypr"),
        {"observations.state.ee_pose": obs_ypr, "actions_cartesian": act_ypr},
    )["actions_cartesian"]

    # 6D inputs (18D obs / per-frame action)
    obs_6d = np.concatenate(
        [_matrix_to_xyz6d(obs_l[None])[0], _matrix_to_xyz6d(obs_r[None])[0]]
    )
    act_6d = np.concatenate(
        [_matrix_to_xyz6d(wrist_l), _matrix_to_xyz6d(wrist_r)], axis=-1
    )
    six_out = _run(
        _build_aria_cartesian_revert_eef_frame_transform_list(rot_repr="6d"),
        {"observations.state.ee_pose": obs_6d, "actions_cartesian": act_6d},
    )["actions_cartesian"]

    assert six_out.shape == ypr_out.shape == (T, 12)
    np.testing.assert_allclose(six_out, ypr_out, atol=1e-6)


def test_eva_6d_revert_matches_ypr_revert_with_grippers():
    T = 4
    obs_l = _random_se3(1, seed=30)[0]
    obs_r = _random_se3(1, seed=31)[0]
    wrist_l = _random_se3(T, seed=32)
    wrist_r = _random_se3(T, seed=33)
    glo, gro = 0.3, 0.7  # obs grippers
    gla = np.linspace(0, 1, T)[:, None]
    gra = np.linspace(1, 0, T)[:, None]

    def obs_concat(to_pose):
        return np.concatenate(
            [to_pose(obs_l[None])[0], [glo], to_pose(obs_r[None])[0], [gro]]
        )

    def act_concat(to_pose):
        return np.concatenate([to_pose(wrist_l), gla, to_pose(wrist_r), gra], axis=-1)

    ypr_out = _run(
        _build_eva_bimanual_revert_eef_frame_transform_list(
            rot_repr="ypr", is_quat=False
        ),
        {
            "observations.state.ee_pose": obs_concat(_matrix_to_xyzypr),
            "actions_cartesian": act_concat(_matrix_to_xyzypr),
        },
    )["actions_cartesian"]

    six_out = _run(
        _build_eva_bimanual_revert_eef_frame_transform_list(rot_repr="6d"),
        {
            "observations.state.ee_pose": obs_concat(_matrix_to_xyz6d),
            "actions_cartesian": act_concat(_matrix_to_xyz6d),
        },
    )["actions_cartesian"]

    assert six_out.shape == ypr_out.shape == (T, 14)
    np.testing.assert_allclose(six_out, ypr_out, atol=1e-6)


def test_eva_6d_revert_orthonormalizes_noisy_model_output():
    """A non-orthonormal 6D 'prediction' must still revert to a valid ypr
    (matching the orthonormalized rotation), exercising the decode-time
    Gram-Schmidt inside the xyz6d revert."""
    T = 3
    obs_l = _random_se3(1, seed=40)[0]
    obs_r = _random_se3(1, seed=41)[0]
    wrist_l = _random_se3(T, seed=42)
    wrist_r = _random_se3(T, seed=43)

    clean_l = _matrix_to_xyz6d(wrist_l)
    clean_r = _matrix_to_xyz6d(wrist_r)
    noisy_l = clean_l.copy()
    noisy_r = clean_r.copy()
    # scale columns (direction preserved) + jitter c2 off-orthogonal
    noisy_l[:, 3:6] *= 1.5
    noisy_r[:, 6:9] += 0.05

    obs_6d = np.concatenate(
        [_matrix_to_xyz6d(obs_l[None])[0], _matrix_to_xyz6d(obs_r[None])[0]]
    )
    out = _run(
        _build_aria_cartesian_revert_eef_frame_transform_list(rot_repr="6d"),
        {
            "observations.state.ee_pose": obs_6d,
            "actions_cartesian": np.concatenate([noisy_l, noisy_r], axis=-1),
        },
    )["actions_cartesian"]
    assert out.shape == (T, 12)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# F. full chain: data 6D -> converter to32 -> from32 -> revert (pi0.5 path)
# ---------------------------------------------------------------------------
def test_human6d_converter_preserves_data_pipeline_output():
    """A native 18D vector from the data pipeline survives to32/from32 intact,
    so the (un)normalized values the model is trained on are exactly the ones
    decoded back."""
    out = _run(Aria.get_transform_list("cartesian_6d"), _aria_cartesian_batch())
    native = torch.as_tensor(np.asarray(out["actions_cartesian"]))[
        None
    ].float()  # (1,T,18)
    conv = HumanBimanualCartesian6D()
    round_tripped = conv.from32(conv.to32(native))
    torch.testing.assert_close(round_tripped, native)
