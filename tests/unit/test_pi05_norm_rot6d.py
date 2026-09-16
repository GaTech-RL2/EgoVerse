"""Round-trip tests for the normalized continuous-6D rotation encoding.

Covers the data transform (ypr <-> 6D) and the converter packers
(``to32_norm_6d`` / ``from32_norm_6d``) for both the robot bimanual (14D ypr /
20D 6D, with gripper) and human bimanual (12D ypr / 18D 6D, no gripper) layouts,
plus the proprio ee_pose: the 6D transform modes convert the proprio too (a
single pose vector, same per-arm layout as one action row), and the 6D revert
lists convert it back before the eef-frame revert reads it. Also the pieces
that ride along with the 6D pipeline: the rotation-free bounds check, the
mecka left-wrist fix, embodiment aliasing, the sampler fallback and the eval
helpers.
"""

import math
from types import SimpleNamespace

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
    RobotBimanualCartesian6D,
    pad_to_width,
)
from egomimic.utils.pose_utils import _ypr_to_rot6d


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
    converter = RobotBimanualCartesian6D()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": _eva_ypr_chunk()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 20)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    decoded = converter.from32_norm_6d(packed)
    torch.testing.assert_close(decoded, six6d, atol=1e-6, rtol=1e-6)


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

    eva32 = RobotBimanualCartesian6D().to32_norm_6d(torch.ones(1, 3, 20))
    wide = pad_to_width(eva32, 144)
    assert wide.shape[-1] == 144 and torch.all(wide[..., 32:] == 0)
    torch.testing.assert_close(
        RobotBimanualCartesian6D().from32_norm_6d(wide)[..., :20],
        torch.ones(1, 3, 20),
    )
    with pytest.raises(ValueError, match="action_dim is 32"):
        pad_to_width(kp, 32)


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


@pytest.mark.parametrize(
    "cls_name,mode",
    [
        ("Eva", "cartesian"),
        ("Eva", "cartesian_wristframe_ypr"),
        ("Eva", "cartesian_wristframe_quat"),
        ("Human", "cartesian"),
        ("Human", "cartesian_padded"),
        ("Human", "cartesian_wristframe_ypr"),
    ],
)
def test_legacy_rotation_modes_need_an_explicit_opt_in(cls_name, mode):
    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.rldb.embodiment.human import Human

    cls = {"Eva": Eva, "Human": Human}[cls_name]
    with pytest.raises(ValueError, match="discontinuous rotation"):
        cls.get_transform_list(mode)
    assert cls.get_transform_list(mode, allow_legacy_rotation=True)


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


def _bounds_check_dataset(key: str, width: int):
    """Minimal MultiDataset shell exposing _check_bounds with +-1 quantile
    bounds on ``key`` for embodiment 0."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    md.norm_stats = {
        0: {
            key: {
                "quantile_1": np.full(width, -1.0, dtype=np.float32),
                "quantile_99": np.full(width, 1.0, dtype=np.float32),
            }
        }
    }
    md.zarr_keys = {0: {key: key}}
    md._warned_violations = set()
    return md


@pytest.mark.parametrize("key", ["actions_cartesian", "observations.state.ee_pose"])
@pytest.mark.parametrize("width,rot_idx,xyz_idx", [(14, 3, 0), (20, 4, 0), (18, 5, 9)])
def test_bounds_check_ignores_rotation_channels(key, width, rot_idx, xyz_idx):
    # Rotation channels (Euler wraps at +-pi; 6D columns are ~[-1, 1]) must be
    # excluded from quantile bounds checking, while translation/gripper
    # channels are still checked and NaN/Inf still rejects the full vector.
    md = _bounds_check_dataset(key, width)
    arr = np.zeros((5, width), dtype=np.float32)

    arr[2, rot_idx] = 50.0  # far outside +-1, but a rotation channel
    assert md._check_bounds({"embodiment": 0, key: arr.copy()}, None, 0, "ep") is None

    bad = arr.copy()
    bad[2, xyz_idx] = 50.0  # translation channel out of bounds -> violation
    assert md._check_bounds({"embodiment": 0, key: bad}, None, 0, "ep") is not None

    nan = arr.copy()
    nan[2, rot_idx] = np.nan  # NaN anywhere (even rotation) -> violation
    assert md._check_bounds({"embodiment": 0, key: nan}, None, 0, "ep") is not None


@pytest.mark.parametrize("key", ["actions_keypoints", "observations.state.keypoints"])
@pytest.mark.parametrize("width,rot_idx,kp_idx", [(138, 4, 10), (144, 78, 90)])
def test_bounds_check_ignores_wrist_rotation_in_keypoint_layout(
    key, width, rot_idx, kp_idx
):
    md = _bounds_check_dataset(key, width)
    arr = np.zeros((5, width), dtype=np.float32)
    arr[1, rot_idx] = 50.0  # wrist rotation channel: not bounds-checked
    assert md._check_bounds({"embodiment": 0, key: arr.copy()}, None, 0, "ep") is None
    bad = arr.copy()
    bad[1, kp_idx] = 50.0  # a keypoint coordinate: checked
    assert md._check_bounds({"embodiment": 0, key: bad}, None, 0, "ep") is not None


def test_bounds_check_full_vector_for_other_keys():
    # Keys without a known layout (or unrecognized widths) keep the
    # full-vector check.
    md = _bounds_check_dataset("some_other_key", 20)
    arr = np.zeros((5, 20), dtype=np.float32)
    arr[2, 4] = 50.0
    assert (
        md._check_bounds({"embodiment": 0, "some_other_key": arr}, None, 0, "ep")
        is not None
    )

    md16 = _bounds_check_dataset("actions_cartesian", 16)
    arr16 = np.zeros((5, 16), dtype=np.float32)
    arr16[2, 4] = 50.0
    assert (
        md16._check_bounds({"embodiment": 0, "actions_cartesian": arr16}, None, 0, "ep")
        is not None
    )


def test_constant_cell_exemption_follows_norm_mode():
    """A channel the zscore normalizer still divides by must still be
    bounds-checked, even though its quantile range is zero: 0 in 99.6 % of
    samples and 1.0 in the rest gives q99 - q1 = 0 but std = 0.063."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    stats = {
        "quantile_1": np.zeros(18, dtype=np.float32),
        "quantile_99": np.zeros(18, dtype=np.float32),
        "mean": np.full(18, 0.004, dtype=np.float32),
        "std": np.full(18, 0.063, dtype=np.float32),
    }
    arr = np.zeros((5, 18), dtype=np.float32)
    arr[0, 0] = 50.0  # corrupt value in a quantile-collapsed channel

    for mode, rejected in (("quantile", False), ("zscore", True)):
        md = MultiDataset.__new__(MultiDataset)
        md.norm_mode = mode
        md.norm_stats = {0: {"actions_cartesian": stats}}
        md.zarr_keys = {0: {"actions_cartesian": "actions_cartesian"}}
        md._warned_violations = set()
        got = md._check_bounds(
            {"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep"
        )
        assert (got is not None) is rejected, mode


def test_bounds_check_tolerates_roundoff_on_collapsed_bounds():
    """Wrist-frame t=0 cells have bounds [0, 0]; values there must not reject
    (normalize() maps such constant cells to 0), while other cells are still
    checked."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    q99 = np.ones(18, dtype=np.float32)
    q99[0] = 0.0  # channel 0 collapsed to [0, 0]
    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = "quantile"
    md.norm_stats = {
        0: {
            "actions_cartesian": {
                "quantile_1": np.zeros(18, dtype=np.float32),
                "quantile_99": q99,
            }
        }
    }
    md.zarr_keys = {0: {"actions_cartesian": "actions_cartesian"}}
    md._warned_violations = set()
    arr = np.zeros((5, 18), dtype=np.float32)
    arr[0, 0] = 1e-9  # a roundoff-scale xyz value at a [0, 0] bound
    assert (
        md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep")
        is None
    )
    arr[0, 0] = 1e-3  # off-convention offset at the constant cell: admitted
    assert (
        md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep")
        is None
    )
    arr[0, 1] = 50.0  # a real violation on a regular cell is still caught
    assert (
        md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep")
        is not None
    )


def test_bounds_check_warns_once_on_stat_shape_mismatch(caplog):
    md = _bounds_check_dataset("actions_cartesian", 18)
    arr = np.zeros((5, 20), dtype=np.float32)  # stats are 18-wide
    with caplog.at_level("WARNING"):
        assert (
            md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep")
            is None
        )
        assert (
            md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 1, "ep")
            is None
        )
    msgs = [r.message for r in caplog.records if "bounds check skipped" in r.message]
    assert len(msgs) == 1, msgs


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


def test_fix_left_wrist_convention_flag_prepends_correction():
    from egomimic.rldb.embodiment.human import Human
    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame

    tl = Human.get_transform_list(
        "cartesian_wristframe_6d", stride=1, fix_left_wrist_convention=True
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
        "keypoints_wristframe_6d", fix_left_wrist_convention=True
    )
    assert isinstance(tl_kp[0], RotateLocalFrame)
    assert set(tl_kp[0].keys) == {"left.action_wrist_pose", "left.obs_wrist_pose"}
    tl_kp_ee = Human.get_transform_list(
        "keypoints_wristframe_6d", fix_left_wrist_convention=True, include_ee_pose=True
    )
    assert set(tl_kp_ee[0].keys) == {
        "left.action_wrist_pose",
        "left.obs_wrist_pose",
        "left.action_ee_pose",
        "left.obs_ee_pose",
    }


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


def _fallback_dataset():
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.index_map = [("bad", i) for i in range(10)] + [("good", i) for i in range(1000)]
    md._global_indices_by_dataset = {
        "bad": list(range(10)),
        "good": list(range(10, 1010)),
    }
    return md


def test_fallback_widens_to_global_after_local_attempts():
    # A wholly-bad episode must not exhaust the sampler: retries stay inside
    # the failing episode for GLOBAL_FALLBACK_ATTEMPTS, then widen to the full
    # index space.
    md = _fallback_dataset()
    attempts = None
    seen_local, seen_global = set(), set()
    idx = 0
    for _ in range(md.GLOBAL_FALLBACK_ATTEMPTS):
        idx, attempts = md._next_after_failure(0, "bad", attempts, reason="r")
        seen_local.add(md.index_map[idx][0])
    assert seen_local == {"bad"}, "early retries must stay within the episode"

    for _ in range(50):
        idx, attempts = md._next_after_failure(idx, "bad", attempts, reason="r")
        seen_global.add(md.index_map[idx][0])
    assert "good" in seen_global, "post-threshold retries must sample globally"


def test_fallback_cap_raises():
    # The cap is a systemic-failure detector: everything it can reach has
    # already been drawn uniformly from the whole index space and failed.
    md = _fallback_dataset()
    attempts, idx, n = None, 0, 0
    with pytest.raises(RuntimeError, match="consecutive bad samples"):
        for n in range(1, md.MAX_FALLBACK_ATTEMPTS + 2):
            idx, attempts = md._next_after_failure(idx, "bad", attempts, reason="boom")
    assert n == md.MAX_FALLBACK_ATTEMPTS


def test_video_fps_compensates_for_world_size():
    # Distributed val strides an episode's frames by world_size on each rank;
    # playback fps must scale down to keep videos wall-clock real-time.
    from egomimic.eval.eval_video import EvalVideo

    class _Stub(EvalVideo):
        def compute_metrics_and_viz(self, batch, do_viz=True):
            raise NotImplementedError

    ev = _Stub.__new__(_Stub)
    for world, expected in [(1, 30), (2, 15), (4, 8), (8, 4)]:
        ev.trainer = SimpleNamespace(world_size=world)
        assert ev._video_fps() == expected, (world, ev._video_fps())
    ev.trainer = SimpleNamespace()  # no world_size attr -> assume 1
    assert ev._video_fps() == 30


def test_viz_gate_follows_lightning_epoch_convention():
    from egomimic.eval.eval_video import EvalVideo

    class _Stub(EvalVideo):
        def compute_metrics_and_viz(self, batch, do_viz=True):
            raise NotImplementedError

    ev = _Stub(viz_every_n_epochs=20)
    # Lightning validates at current_epoch 19, 39, ... for check_val_every_n_epoch=20
    for epoch, expected in [(0, False), (19, True), (20, False), (39, True)]:
        ev.trainer = SimpleNamespace(current_epoch=epoch)
        assert ev._should_viz() is expected, epoch
    ev.viz_every_n_epochs = 0
    assert ev._should_viz() is False


def test_viz_gate_always_renders_the_final_epoch():
    from egomimic.eval.eval_video import EvalVideo

    class _Stub(EvalVideo):
        def compute_metrics_and_viz(self, batch, do_viz=True):
            raise NotImplementedError

    ev = _Stub(viz_every_n_epochs=100)
    # trainer=debug: 4 epochs, val at current_epoch 1 and 3 -> video only at 3
    for epoch, expected in [(1, False), (3, True)]:
        ev.trainer = SimpleNamespace(current_epoch=epoch, max_epochs=4)
        assert ev._should_viz() is expected, epoch
    # eval mode validates once at epoch 0 with max_epochs=1
    ev.trainer = SimpleNamespace(current_epoch=0, max_epochs=1)
    assert ev._should_viz() is True
    # unbounded runs (max_epochs None / -1) only follow the interval
    for max_epochs in (None, -1):
        ev.trainer = SimpleNamespace(current_epoch=0, max_epochs=max_epochs)
        assert ev._should_viz() is False
    # viz_every_n_epochs=0 still turns video off entirely
    ev.viz_every_n_epochs = 0
    ev.trainer = SimpleNamespace(current_epoch=3, max_epochs=4)
    assert ev._should_viz() is False


def test_frechet_helper():
    from egomimic.utils.metrics import frechet_gaussian_over_time

    B, T, D = 4, 10, 6
    torch.manual_seed(0)
    pred = torch.randn(B, T, D)
    fd = frechet_gaussian_over_time(pred, pred.clone())
    assert fd.shape == (B,)
    assert fd.max() < 1e-2, "identical distributions must score ~0"
    fd_shift = frechet_gaussian_over_time(pred + 5.0, pred)
    assert (fd_shift > fd).all(), "mean shift must increase the distance"


def test_reverse_kl_is_gone():
    import egomimic.utils.metrics as metrics

    assert not hasattr(metrics, "reverse_kl_from_samples")


def test_train_viz_wrapper_prefixes_and_forwards():
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo
    from egomimic.eval.eval_video import EvalVideo

    class _Base(EvalVideo):
        def __init__(self):
            super().__init__(viz_func={}, transform_lists={}, viz_every_n_epochs=7)
            self.seen = None

        def compute_metrics_and_viz(self, batch, do_viz=True):
            self.seen = (self.model, do_viz)
            return {"Valid/x": 1.0}, {}

    base = _Base()
    tv = TrainVizEvalVideo(base)
    algo = object()
    tv.model = algo  # property setter forwards to base too
    metrics, _ = tv.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"train_viz/Valid/x"}, metrics
    assert base.seen == (algo, False)
    assert tv.viz_every_n_epochs == 7, "wrapper inherits the base viz gate"
    tv.trainer = SimpleNamespace(default_root_dir="/tmp/run")
    assert base.trainer is tv.trainer
    assert tv.video_dir().endswith("videos_train_viz")

    op = TrainVizEvalVideo(base, prefix="unseen_op_valid")
    metrics, _ = op.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"unseen_op_valid/Valid/x"}, metrics
    op.trainer = SimpleNamespace(default_root_dir="/tmp/run")
    assert op.video_dir().endswith("videos_unseen_op_valid")


def test_dtw_distance_matches_bruteforce_and_tolerates_shift():
    from egomimic.utils.metrics import dtw_distance

    def _dtw_ref(x, y):
        t1, t2 = len(x), len(y)
        cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        acc = np.full((t1 + 1, t2 + 1), np.inf)
        acc[0, 0] = 0.0
        for i in range(1, t1 + 1):
            for j in range(1, t2 + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(
                    acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
                )
        return acc[t1, t2]

    torch.manual_seed(3)
    pred = torch.randn(3, 9, 4)
    tgt = torch.randn(3, 9, 4)
    got = dtw_distance(pred, tgt, normalize=False)
    for b in range(3):
        ref = _dtw_ref(pred[b].numpy(), tgt[b].numpy())
        assert abs(got[b].item() - ref) < 1e-4, (b, got[b].item(), ref)

    # identical trajectories -> 0
    assert dtw_distance(pred, pred.clone()).max() < 1e-6

    # a time-shifted copy of the same smooth trajectory: DTW must forgive the
    # shift (score far below the paired per-step distance of the shifted pair)
    t = torch.linspace(0, 6.28, 40)
    traj = torch.stack([torch.sin(t), torch.cos(t)], dim=-1)[None]  # (1, 40, 2)
    shifted = torch.roll(traj, shifts=3, dims=1)
    dtw_shift = dtw_distance(traj, shifted, normalize=False).item()
    paired = (traj - shifted).norm(dim=-1).sum().item()
    assert dtw_shift < 0.5 * paired, (dtw_shift, paired)


def test_split_mse_is_stateless_and_matches_manual():
    from egomimic.eval.action_metrics import _paired_mse, _split_mse

    rng = np.random.default_rng(21)
    pred = torch.from_numpy(rng.normal(size=(4, 10, 18))).float()
    gt = torch.from_numpy(rng.normal(size=(4, 10, 18))).float()
    xyz_idx = [0, 1, 2, 9, 10, 11]
    rot_idx = [i for i in range(18) if i not in xyz_idx]
    xyz, rot = _split_mse(pred, gt)
    torch.testing.assert_close(
        xyz, (pred[..., xyz_idx] - gt[..., xyz_idx]).pow(2).mean()
    )
    torch.testing.assert_close(
        rot, (pred[..., rot_idx] - gt[..., rot_idx]).pow(2).mean()
    )
    # stateless: a second, unrelated call is unaffected by the first
    a, b = torch.zeros(2, 3, 18), torch.ones(2, 3, 18)
    torch.testing.assert_close(_split_mse(a, b)[0], torch.tensor(1.0))
    torch.testing.assert_close(_paired_mse(a, b), torch.tensor(1.0))
    assert _split_mse(torch.zeros(2, 3, 7), torch.zeros(2, 3, 7)) == (None, None)


def test_rot_geodesic_error_matches_angle_and_survives_gimbal_lock():
    from scipy.spatial.transform import Rotation as R

    from egomimic.eval.action_metrics import _rot_geodesic_error

    rng = np.random.default_rng(22)
    ypr = rng.uniform(-1.0, 1.0, size=(6, 5, 3))
    theta = 0.3
    # rotate every pose by theta about a random axis -> geodesic error == theta
    axes = rng.normal(size=(6, 5, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    Rg = R.from_euler("ZYX", ypr.reshape(-1, 3))
    Rp = R.from_rotvec(theta * axes.reshape(-1, 3)) * Rg
    ypr_p = Rp.as_euler("ZYX").reshape(6, 5, 3)

    # 12-dim ypr layout (both arms the same pose)
    gt12 = torch.from_numpy(np.concatenate([ypr, ypr], -1)).float()
    gt12 = torch.cat(
        [torch.zeros(6, 5, 3), gt12[..., :3], torch.zeros(6, 5, 3), gt12[..., 3:]], -1
    )
    pr12 = torch.from_numpy(np.concatenate([ypr_p, ypr_p], -1)).float()
    pr12 = torch.cat(
        [torch.zeros(6, 5, 3), pr12[..., :3], torch.zeros(6, 5, 3), pr12[..., 3:]], -1
    )
    assert abs(_rot_geodesic_error(pr12, gt12).item() - theta) < 1e-4
    assert _rot_geodesic_error(gt12, gt12).item() < 1e-5

    # 18-dim 6D layout, same rotations -> same answer
    def to18(y):
        six = _ypr_to_rot6d(y)
        arm = np.concatenate([np.zeros(y.shape[:-1] + (3,)), six], -1)
        return torch.from_numpy(np.concatenate([arm, arm], -1)).float()

    assert abs(_rot_geodesic_error(to18(ypr_p), to18(ypr)).item() - theta) < 1e-4
    assert _rot_geodesic_error(torch.zeros(2, 3, 7), torch.zeros(2, 3, 7)) is None

    # Gimbal lock: pitch ~ pi/2, pred = gt rotated 0.004 rad about local y.
    # A ypr MSE (even wrap-aware) explodes because yaw and roll trade off;
    # the geodesic error reports the true 0.004.
    gt = np.array([0.3, np.pi / 2 - 0.002, 0.2])
    Rgt = R.from_euler("ZYX", gt)
    Rpr = Rgt * R.from_rotvec([0.0, 0.004, 0.0])
    pr = Rpr.as_euler("ZYX")
    v_gt = torch.tensor([[0, 0, 0, *gt, 0, 0, 0, *gt]], dtype=torch.float32)
    v_pr = torch.tensor([[0, 0, 0, *pr, 0, 0, 0, *pr]], dtype=torch.float32)
    geo = _rot_geodesic_error(v_pr, v_gt).item()
    assert abs(geo - 0.004) < 1e-3, geo
    euler_mse = (v_pr - v_gt).pow(2).mean().item()
    assert euler_mse > 0.1, euler_mse  # the Euler metric is fooled here


# ------------------------------------------------ PIEvalVideo metric contract
def _rand_6d_chunks(seed, B=3, T=6):
    """(pred, gt) native 18D human chunks with a small, known perturbation,
    plus the per-(batch, arm) rotation matrices so a test can re-express both
    in another frame. 6D columns are built directly from rotation matrices
    (col 0 | col 1), matching ``_reconstruct_R_from_cols``."""
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(seed)
    xyz_g = rng.uniform(-1.0, 1.0, size=(B, T, 2, 3))
    R_g = R.random(B * T * 2, random_state=seed).as_matrix().reshape(B, T, 2, 3, 3)
    xyz_p = xyz_g + 0.05 * rng.normal(size=xyz_g.shape)
    R_p = (
        (
            R.from_rotvec(0.1 * rng.normal(size=(B * T * 2, 3)))
            * R.from_matrix(R_g.reshape(-1, 3, 3))
        )
        .as_matrix()
        .reshape(B, T, 2, 3, 3)
    )
    return (xyz_p, R_p), (xyz_g, R_g)


def _chunk18(xyz, Rm):
    six = np.concatenate([Rm[..., :, 0], Rm[..., :, 1]], -1)  # (B,T,2,6)
    arm = np.concatenate([xyz, six], -1)  # (B,T,2,9)
    return torch.from_numpy(arm.reshape(arm.shape[0], arm.shape[1], 18)).float()


def _run_pi_eval(pred, gt, ac_key="actions_cartesian"):
    from egomimic.eval.eval_pi import PIEvalVideo
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment

    eid = EMBODIMENT.HUMAN_BIMANUAL.value
    name = get_embodiment(eid).lower()

    class _NormStats:
        def unnormalize(self, batch, embodiment_id):
            return batch

    class _Algo:
        ac_keys = {eid: ac_key}
        norm_stats = _NormStats()

        def forward_eval(self, batch):
            return {f"{name}_loss": torch.tensor(0.5), f"{name}_{ac_key}": pred}

    ev = PIEvalVideo(viz_func={}, transform_lists={}, viz_every_n_epochs=1)
    ev.model = _Algo()
    batch = {eid: {ac_key: gt, "embodiment": torch.tensor([eid])}}
    metrics, images = ev.compute_metrics_and_viz(batch, do_viz=False)
    return f"Valid/{name}_{ac_key}_", f"Valid/{name}_loss", metrics, images


def test_pi_eval_metric_set_and_identity():
    (xyz_p, R_p), (xyz_g, R_g) = _rand_6d_chunks(40)
    gt = _chunk18(xyz_g, R_g)
    prefix, loss_key, metrics, images = _run_pi_eval(gt.clone(), gt)
    expected = {
        prefix + k
        for k in (
            "xyz_paired_mse_avg",
            "xyz_final_mse_avg",
            "rot6d_paired_mse_avg",
            "rot_err_deg_avg",
            "rot_err_deg_final",
            "xyz_dtw_avg",
            "xyz_frechet_gauss_avg",
            "xyz_l2_early",
            "xyz_l2_mid",
            "xyz_l2_late",
        )
    } | {loss_key, "Valid/action_loss"}
    assert set(metrics) == expected, sorted(metrics)
    assert images == {}, "do_viz=False must not render"
    assert float(metrics[prefix + "xyz_paired_mse_avg"]) < 1e-10
    assert float(metrics[prefix + "rot_err_deg_avg"]) < 1e-3
    assert float(metrics[prefix + "xyz_dtw_avg"]) < 1e-6
    assert float(metrics[prefix + "xyz_l2_late"]) < 1e-5
    assert float(metrics[prefix + "xyz_frechet_gauss_avg"]) < 1e-2


def test_pi_eval_metrics_are_invariant_to_a_shared_rigid_transform():
    # The evaluator logs no cam-frame twins of the native metrics: the revert
    # applies the SAME rigid transform (the observed wrist pose) to prediction
    # and gt, and every kept metric is invariant to that. Re-express pred and
    # gt in a random frame per (batch, arm) and check.
    from scipy.spatial.transform import Rotation as R

    (xyz_p, R_p), (xyz_g, R_g) = _rand_6d_chunks(41)
    B = xyz_g.shape[0]
    R_w = R.random(B * 2, random_state=7).as_matrix().reshape(B, 1, 2, 3, 3)
    t_w = np.random.default_rng(8).uniform(-2, 2, size=(B, 1, 2, 3))

    def move(xyz, Rm):
        return np.einsum("btaij,btaj->btai", R_w, xyz) + t_w, R_w @ Rm

    prefix, _, native, _ = _run_pi_eval(_chunk18(xyz_p, R_p), _chunk18(xyz_g, R_g))
    prefix, _, moved, _ = _run_pi_eval(
        _chunk18(*move(xyz_p, R_p)), _chunk18(*move(xyz_g, R_g))
    )
    for key, atol in (
        ("xyz_paired_mse_avg", 1e-5),
        ("xyz_final_mse_avg", 1e-5),
        ("rot_err_deg_avg", 1e-3),
        ("rot_err_deg_final", 1e-3),
        ("xyz_dtw_avg", 1e-4),
        ("xyz_frechet_gauss_avg", 1e-3),
        ("xyz_l2_early", 1e-5),
        ("xyz_l2_late", 1e-5),
    ):
        a, b = float(native[prefix + key]), float(moved[prefix + key])
        assert abs(a - b) < atol, (key, a, b)
    # sanity: the perturbation is visible in the headline metrics
    assert float(native[prefix + "rot_err_deg_avg"]) > 1.0
    assert float(native[prefix + "xyz_paired_mse_avg"]) > 1e-4


def test_pi_eval_keypoint_metrics():
    from scipy.spatial.transform import Rotation as R

    rng = np.random.default_rng(50)
    B, T = 2, 5
    wrist_xyz = rng.uniform(-1, 1, size=(B, T, 2, 3))
    ypr = rng.uniform(-1, 1, size=(B, T, 2, 3))
    kp = rng.uniform(-0.1, 0.1, size=(B, T, 2, 63))

    def chunk(wx, y, k):
        hand = np.concatenate([wx, _ypr_to_rot6d(y), k], -1)  # (B,T,2,72)
        return torch.from_numpy(hand.reshape(B, T, 144)).float()

    gt = chunk(wrist_xyz, ypr, kp)
    theta = 0.2
    axes = rng.normal(size=(B * T * 2, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    ypr_p = (
        (R.from_rotvec(theta * axes) * R.from_euler("ZYX", ypr.reshape(-1, 3)))
        .as_euler("ZYX")
        .reshape(B, T, 2, 3)
    )
    pred = chunk(wrist_xyz + 0.1, ypr_p, kp + 0.02)
    prefix, loss_key, metrics, _ = _run_pi_eval(pred, gt, ac_key="actions_keypoints")
    expected = {
        prefix + k
        for k in (
            "kp_l2_avg",
            "kp_l2_final",
            "wrist_xyz_paired_mse_avg",
            "wrist_xyz_final_mse_avg",
            "wrist_rot_err_deg_avg",
            "wrist_rot_err_deg_final",
            "kp_dtw_avg",
            *(
                f"{m}_l2_{seg}"
                for m in ("kp", "wrist_xyz")
                for seg in ("early", "mid", "late")
            ),
        )
    } | {loss_key, "Valid/action_loss"}
    assert set(metrics) == expected, sorted(metrics)
    # every keypoint is off by 0.02 in each axis -> L2 = 0.02 * sqrt(3)
    assert abs(float(metrics[prefix + "kp_l2_avg"]) - 0.02 * math.sqrt(3)) < 1e-5
    assert abs(float(metrics[prefix + "wrist_xyz_paired_mse_avg"]) - 0.01) < 1e-6
    assert (
        abs(float(metrics[prefix + "wrist_rot_err_deg_avg"]) - math.degrees(theta))
        < 1e-2
    )
