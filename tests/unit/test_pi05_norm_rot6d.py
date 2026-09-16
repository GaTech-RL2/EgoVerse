"""Round-trip tests for the normalized continuous-6D rotation encoding.

Covers the data transform (ypr <-> 6D) and the converter packers
(``to32_norm_6d`` / ``from32_norm_6d``) for both the robot bimanual (14D ypr /
20D 6D, with gripper) and human bimanual (12D ypr / 18D 6D, no gripper) layouts,
plus the proprio ee_pose: the 6D transform modes convert the proprio too (a
single pose vector, same per-arm layout as one action row), and the 6D revert
lists convert it back before the eef-frame revert reads it.
"""

import math

import numpy as np
import pytest
import torch

from egomimic.rldb.zarr.action_chunk_transforms import (
    CartesianRot6DToYPR,
    CartesianYPRToRot6D,
)
from egomimic.utils.action_utils import (
    BaseActionConverter,
    HumanBimanualKeypoints,
    RobotBimanualCartesianEuler,
    pad_to_width,
)


def _eva_ypr_chunk(T: int = 5) -> np.ndarray:
    # [L xyz ypr g, R xyz ypr g]; moderate angles to avoid gimbal/wrap ambiguity.
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1.0, 1.0, size=(T, 3))
    ypr = rng.uniform(-1.0, 1.0, size=(T, 3))  # radians, well inside (-pi, pi)
    g = rng.uniform(0.0, 1.0, size=(T, 1))
    arm = np.concatenate([xyz, ypr, g], axis=-1)
    return np.concatenate([arm, arm], axis=-1)  # 14D


def _aria_ypr_chunk(T: int = 5) -> np.ndarray:
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1.0, 1.0, size=(T, 3))
    ypr = rng.uniform(-1.0, 1.0, size=(T, 3))
    arm = np.concatenate([xyz, ypr], axis=-1)
    return np.concatenate([arm, arm], axis=-1)  # 12D


@pytest.mark.parametrize(
    "chunk_fn,ypr_dim,six_dim",
    [(_eva_ypr_chunk, 14, 20), (_aria_ypr_chunk, 12, 18)],
)
def test_cartesian_ypr_rot6d_transform_round_trips(chunk_fn, ypr_dim, six_dim):
    ypr = chunk_fn()
    assert ypr.shape[-1] == ypr_dim

    fwd = CartesianYPRToRot6D(action_key="actions_cartesian")
    rev = CartesianRot6DToYPR(action_key="actions_cartesian")

    batch = {"actions_cartesian": ypr.copy()}
    batch = fwd.transform(batch)
    assert batch["actions_cartesian"].shape[-1] == six_dim

    batch = rev.transform(batch)
    np.testing.assert_allclose(batch["actions_cartesian"], ypr, atol=1e-6)


def test_ypr_to_rot6d_transform_is_continuous_at_yaw_wrap():
    # The whole point of encoding rotation as 6D BEFORE normalization: a yaw
    # of +pi-eps and -pi+eps are the same orientation. As ypr they differ by
    # ~2pi (and a normalized-ypr target would jump); as 6D columns they are
    # within O(eps) of each other, so the regression target is continuous.
    eps = 1e-3
    a = np.zeros((1, 12))
    b = np.zeros((1, 12))
    a[0, 3], b[0, 3] = math.pi - eps, -math.pi + eps  # L yaw
    a[0, 9], b[0, 9] = math.pi - eps, -math.pi + eps  # R yaw
    fwd = CartesianYPRToRot6D(action_key="actions_cartesian")
    six_a = fwd.transform({"actions_cartesian": a.copy()})["actions_cartesian"]
    six_b = fwd.transform({"actions_cartesian": b.copy()})["actions_cartesian"]
    assert np.abs(a - b).max() > 6.0, "ypr inputs straddle the wrap"
    assert np.abs(six_a - six_b).max() < 5 * eps, "6D columns must not jump"


def test_transform_preserves_tensor_type():
    ypr = torch.from_numpy(_eva_ypr_chunk())
    out = CartesianYPRToRot6D().transform({"actions_cartesian": ypr})[
        "actions_cartesian"
    ]
    assert isinstance(out, torch.Tensor)
    assert out.shape[-1] == 20


def test_robot_bimanual_norm_6d_pack_round_trips():
    converter = RobotBimanualCartesianEuler()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": _eva_ypr_chunk()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 20)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    decoded = converter.from32_norm_6d(packed)
    torch.testing.assert_close(decoded, six6d, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "chunk_fn,ypr_dim,six_dim",
    [(_eva_ypr_chunk, 14, 20), (_aria_ypr_chunk, 12, 18)],
)
def test_proprio_pose_vector_round_trips(chunk_fn, ypr_dim, six_dim):
    # The proprio ee_pose is a single pose vector (D,) with the same per-arm
    # layout as one action row; the same transforms must handle it.
    pose = chunk_fn(T=1)[0]
    assert pose.shape == (ypr_dim,)

    fwd = CartesianYPRToRot6D(action_key="observations.state.ee_pose")
    rev = CartesianRot6DToYPR(action_key="observations.state.ee_pose")

    batch = {"observations.state.ee_pose": pose.copy()}
    batch = fwd.transform(batch)
    assert batch["observations.state.ee_pose"].shape == (six_dim,)

    batch = rev.transform(batch)
    np.testing.assert_allclose(batch["observations.state.ee_pose"], pose, atol=1e-6)


def _keys_of(transforms, cls):
    return {t.action_key for t in transforms if isinstance(t, cls)}


# cartesian_wristframe_6d is covered by the test_wrist6d_roundtrip pipeline tests.
@pytest.mark.parametrize("mode", ["cartesian_6d"])
def test_6d_modes_convert_action_and_proprio(mode):
    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.rldb.embodiment.human import Human

    for cls in (Eva, Human):
        transform_list = cls.get_transform_list(mode)
        assert _keys_of(transform_list, CartesianYPRToRot6D) == {
            "actions_cartesian",
            "observations.state.ee_pose",
        }, f"{cls.__name__} {mode} must 6D-encode both action and proprio"


def test_6d_revert_lists_revert_proprio():
    from egomimic.rldb.embodiment.eva import (
        _build_eva_cartesian_revert_6d_transform_list,
        _build_eva_cartesian_revert_6d_wristframe_transform_list,
    )
    from egomimic.rldb.embodiment.human import (
        _build_human_cartesian_revert_6d_transform_list,
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    )

    for build in (
        _build_eva_cartesian_revert_6d_transform_list,
        _build_eva_cartesian_revert_6d_wristframe_transform_list,
        _build_human_cartesian_revert_6d_transform_list,
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    ):
        transform_list = build()
        assert _keys_of(transform_list, CartesianRot6DToYPR) == {
            "actions_cartesian",
            "observations.state.ee_pose",
        }, f"{build.__name__} must revert both action and proprio to ypr"


def test_vendor_embodiment_names_collapse_to_human():
    # Mirror episodes written by the vendor-split registry carry names like
    # MECKA_BIMANUAL in their zarr metadata; locally all human demo data is
    # one embodiment, so these must resolve to the HUMAN_* ids.
    from egomimic.rldb.embodiment.embodiment import (
        EMBODIMENT,
        get_embodiment_id,
        is_legacy_vendor_embodiment,
    )

    for vendor in ("mecka", "scale", "aria", "lightwheel"):
        assert (
            get_embodiment_id(f"{vendor}_bimanual") == EMBODIMENT.HUMAN_BIMANUAL.value
        )
        assert (
            get_embodiment_id(f"{vendor}_right_arm") == EMBODIMENT.HUMAN_RIGHT_ARM.value
        )
        assert (
            get_embodiment_id(f"{vendor}_left_arm") == EMBODIMENT.HUMAN_LEFT_ARM.value
        )
        assert is_legacy_vendor_embodiment(f"{vendor.upper()}_BIMANUAL")
    assert get_embodiment_id("human_bimanual") == EMBODIMENT.HUMAN_BIMANUAL.value
    assert get_embodiment_id("eva_bimanual") == EMBODIMENT.EVA_BIMANUAL.value
    assert not is_legacy_vendor_embodiment("human_bimanual")
    with pytest.raises(KeyError):
        get_embodiment_id("yam_bimanual")  # robot names are never aliased


def test_base_converter_rejects_norm_6d_encoding():
    converter = BaseActionConverter()
    with pytest.raises(NotImplementedError, match="normalized-rot6d"):
        converter.to32_norm_6d(torch.zeros(1, 1, 20))
    with pytest.raises(NotImplementedError, match="normalized-rot6d"):
        converter.from32_norm_6d(torch.zeros(1, 1, 32))


def test_unpad_gripper_zeros_inverts_pad_and_noops_unpadded():
    from egomimic.rldb.zarr.action_chunk_transforms import (
        PadGripperZeros,
        UnpadGripperZeros,
    )

    rng = np.random.default_rng(6)
    for width in (12, 18):
        v = rng.uniform(-1, 1, size=(width,))
        padded = PadGripperZeros(action_key="k").transform({"k": v.copy()})["k"]
        assert padded.shape == (width + 2,)
        back = UnpadGripperZeros(action_key="k").transform({"k": padded.copy()})["k"]
        np.testing.assert_allclose(back, v)
        # no-op on already-unpadded widths
        same = UnpadGripperZeros(action_key="k").transform({"k": v.copy()})["k"]
        np.testing.assert_allclose(same, v)


def test_human_6d_reverts_unpad_proprio():
    from egomimic.rldb.embodiment.human import (
        _build_human_cartesian_revert_6d_transform_list,
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    )
    from egomimic.rldb.zarr.action_chunk_transforms import UnpadGripperZeros

    for build in (
        _build_human_cartesian_revert_6d_transform_list,
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    ):
        assert any(isinstance(t, UnpadGripperZeros) for t in build()), build.__name__


def test_rotate_local_frame_flips_left_wrist_convention():
    # Right-multiplying by Rz(180 deg) must flip the pose's own x/y axes, keep
    # z (knuckle-forward) and the position, skip zero-quat padding rows, and
    # handle both (7,) poses and (T, 7) chunks.
    from scipy.spatial.transform import Rotation as R

    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame

    rng = np.random.default_rng(4)
    q = R.random(3, random_state=5)
    chunk = np.zeros((4, 7))
    chunk[:3, :3] = rng.uniform(-1, 1, size=(3, 3))
    chunk[:3, 3:] = q.as_quat()[:, [3, 0, 1, 2]]  # wxyz; row 3 stays zero-padded

    t = RotateLocalFrame(keys=["k"])
    out = t.transform({"k": chunk.copy()})["k"]

    np.testing.assert_allclose(out[:, :3], chunk[:, :3])  # positions unchanged
    np.testing.assert_allclose(out[3], np.zeros(7))  # padding untouched
    R_old = q.as_matrix()
    R_new = R.from_quat(out[:3, [4, 5, 6, 3]]).as_matrix()
    np.testing.assert_allclose(R_new[:, :, 0], -R_old[:, :, 0], atol=1e-12)  # x flip
    np.testing.assert_allclose(R_new[:, :, 1], -R_old[:, :, 1], atol=1e-12)  # y flip
    np.testing.assert_allclose(R_new[:, :, 2], R_old[:, :, 2], atol=1e-12)  # z kept

    single = t.transform({"k": chunk[0].copy()})["k"]
    np.testing.assert_allclose(single, out[0], atol=1e-12)


def test_keypoint_converter_is_identity_and_pads_to_model_width():
    # The 144-D keypoint action is packed identity-first; PI pads it (and the
    # cartesian 32-slot layout) up to model.action_dim and the decode slices
    # the native width back out of the wider vector.
    conv = HumanBimanualKeypoints()
    kp = torch.randn(2, 4, 144)
    packed = conv.to32_norm_6d(kp)
    torch.testing.assert_close(packed, kp)
    torch.testing.assert_close(conv.from32_norm_6d(pad_to_width(packed, 160)), kp)
    with pytest.raises(ValueError, match="expected 144-dim"):
        conv.to32_norm_6d(torch.zeros(1, 1, 138))

    eva32 = RobotBimanualCartesianEuler().to32_norm_6d(torch.ones(1, 3, 20))
    wide = pad_to_width(eva32, 144)
    assert wide.shape[-1] == 144 and torch.all(wide[..., 32:] == 0)
    torch.testing.assert_close(
        RobotBimanualCartesianEuler().from32_norm_6d(wide)[..., :20],
        torch.ones(1, 3, 20),
    )
    with pytest.raises(ValueError, match="action_dim is 32"):
        pad_to_width(kp, 32)


def test_fix_mecka_left_wrist_flag_prepends_correction():
    from egomimic.rldb.embodiment.human import Human
    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame

    tl = Human.get_transform_list(
        "cartesian_wristframe_6d", stride=1, fix_mecka_left_wrist=True
    )
    assert isinstance(tl[0], RotateLocalFrame)
    assert set(tl[0].keys) == {"left.action_ee_pose", "left.obs_ee_pose"}
    # default off: other vendors' data must be untouched
    tl_off = Human.get_transform_list("cartesian_wristframe_6d", stride=1)
    assert not isinstance(tl_off[0], RotateLocalFrame)
    # keypoints modes correct the wrist_pose keys their frames are built on
    # (the converter wrote the same double-mirrored rotation into both), plus
    # the ee_pose keys when those are built too.
    tl_kp = Human.get_transform_list(
        "keypoints_wristframe_6d", fix_mecka_left_wrist=True
    )
    assert isinstance(tl_kp[0], RotateLocalFrame)
    assert set(tl_kp[0].keys) == {"left.action_wrist_pose", "left.obs_wrist_pose"}
    tl_kp_ee = Human.get_transform_list(
        "keypoints_wristframe_6d", fix_mecka_left_wrist=True, include_ee_pose=True
    )
    assert set(tl_kp_ee[0].keys) == {
        "left.action_wrist_pose",
        "left.obs_wrist_pose",
        "left.action_ee_pose",
        "left.obs_ee_pose",
    }
