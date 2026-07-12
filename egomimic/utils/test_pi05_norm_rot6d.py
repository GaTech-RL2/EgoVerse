"""Round-trip tests for the normalized continuous-6D rotation encoding.

Covers the data transform (ypr <-> 6D) and the converter 32D packers
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
    HumanBimanualCartesianEuler,
    RobotBimanualCartesianEuler,
)
from egomimic.utils.pose_utils import _rot6d_to_ypr, _ypr_to_rot6d


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


def test_ypr_rot6d_helpers_round_trip():
    ypr = np.random.default_rng(2).uniform(-1.0, 1.0, size=(7, 3))
    six = _ypr_to_rot6d(ypr)
    assert six.shape == (7, 6)
    np.testing.assert_allclose(_rot6d_to_ypr(six), ypr, atol=1e-6)


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


def test_transform_preserves_tensor_type():
    ypr = torch.from_numpy(_eva_ypr_chunk())
    out = CartesianYPRToRot6D().transform({"actions_cartesian": ypr})[
        "actions_cartesian"
    ]
    assert isinstance(out, torch.Tensor)
    assert out.shape[-1] == 20


def test_robot_bimanual_norm_6d_pack_round_trips():
    converter = RobotBimanualCartesianEuler()
    six = torch.from_numpy(_eva_ypr_chunk()).float()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": six.numpy()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 20)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    decoded = converter.from32_norm_6d(packed)
    torch.testing.assert_close(decoded, six6d, atol=1e-6, rtol=1e-6)


def test_human_bimanual_norm_6d_pack_round_trips_and_zeros_gripper():
    converter = HumanBimanualCartesianEuler()
    six6d = torch.from_numpy(
        CartesianYPRToRot6D().transform({"actions_cartesian": _aria_ypr_chunk()})[
            "actions_cartesian"
        ]
    ).float()[None]  # (1, T, 18)

    packed = converter.to32_norm_6d(six6d)
    assert packed.shape[-1] == 32
    # gripper slots (9, 19) must be zero for human (no gripper signal).
    torch.testing.assert_close(packed[..., 9], torch.zeros_like(packed[..., 9]))
    torch.testing.assert_close(packed[..., 19], torch.zeros_like(packed[..., 19]))

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


@pytest.mark.parametrize("mode", ["cartesian_6d", "cartesian_wristframe_6d"])
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


def _bounds_check_dataset(key: str, width: int):
    """Minimal MultiDataset shell exposing _check_bounds with ±1 quantile
    bounds on ``key`` for embodiment 0."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
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
    # Rotation channels (Euler wraps at ±π; 6D columns are ~[-1, 1]) must be
    # excluded from quantile bounds checking — matching the remote pipeline —
    # while translation/gripper channels are still checked and NaN/Inf still
    # rejects the full vector.
    md = _bounds_check_dataset(key, width)
    arr = np.zeros((5, width), dtype=np.float32)

    arr[2, rot_idx] = 50.0  # far outside ±1, but a rotation channel
    assert md._check_bounds({"embodiment": 0, key: arr.copy()}, None, 0, "ep") is None

    bad = arr.copy()
    bad[2, xyz_idx] = 50.0  # translation channel out of bounds -> violation
    assert md._check_bounds({"embodiment": 0, key: bad}, None, 0, "ep") is not None

    nan = arr.copy()
    nan[2, rot_idx] = np.nan  # NaN anywhere (even rotation) -> violation
    assert md._check_bounds({"embodiment": 0, key: nan}, None, 0, "ep") is not None


def test_bounds_check_full_vector_for_other_keys():
    # Keys without the bimanual cartesian layout (or unrecognized widths) keep
    # the full-vector check.
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


def test_rotate_local_frame_flips_left_wrist_convention():
    # Right-multiplying by Rz(180°) must flip the pose's own x/y axes, keep z
    # (knuckle-forward) and the position, skip zero-quat padding rows, and
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


def test_fix_mecka_left_wrist_flag_prepends_correction():
    from egomimic.rldb.embodiment.human import Human
    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame

    tl = Human.get_transform_list(
        "cartesian_wristframe_6d", stride=1, fix_mecka_left_wrist=True
    )
    assert isinstance(tl[0], RotateLocalFrame)
    assert set(tl[0].keys) == {"left.action_ee_pose", "left.obs_ee_pose"}
    # default off — other vendors' data must be untouched
    tl_off = Human.get_transform_list("cartesian_wristframe_6d", stride=1)
    assert not isinstance(tl_off[0], RotateLocalFrame)
    with pytest.raises(ValueError, match="keypoints"):
        Human.get_transform_list("keypoints_headframe_ypr", fix_mecka_left_wrist=True)


def test_vendor_embodiment_names_collapse_to_human():
    # Mirror episodes written by the vendor-split registry carry names like
    # MECKA_BIMANUAL in their zarr metadata; locally all human demo data is
    # one embodiment, so these must resolve to the HUMAN_* ids.
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment_id

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
    assert get_embodiment_id("human_bimanual") == EMBODIMENT.HUMAN_BIMANUAL.value
    assert get_embodiment_id("eva_bimanual") == EMBODIMENT.EVA_BIMANUAL.value
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


def test_fallback_widens_to_global_after_local_attempts():
    # A wholly-bad episode must not exhaust the sampler: retries stay inside
    # the failing episode for GLOBAL_FALLBACK_ATTEMPTS, then widen to the full
    # index space, and only a systemic failure (MAX_FALLBACK_ATTEMPTS) raises.
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.index_map = [("bad", i) for i in range(10)] + [("good", i) for i in range(1000)]
    md._global_indices_by_dataset = {
        "bad": list(range(10)),
        "good": list(range(10, 1010)),
    }

    attempts = None
    seen_local, seen_global = set(), set()
    for _ in range(md.GLOBAL_FALLBACK_ATTEMPTS):
        idx, attempts = md._next_after_failure(0, "bad", attempts, reason="r")
        seen_local.add(md.index_map[idx][0])
    assert seen_local == {"bad"}, "early retries must stay within the episode"

    for _ in range(200):
        idx, attempts = md._next_after_failure(idx, "bad", attempts, reason="r")
        seen_global.add(md.index_map[idx][0])
    assert "good" in seen_global, "post-threshold retries must sample globally"

    with pytest.raises(RuntimeError, match="consecutive bad samples"):
        while True:
            idx, attempts = md._next_after_failure(idx, "bad", attempts, reason="r")


def test_wrap_aware_mse_handles_pi_boundary():
    # A yaw of +pi-eps vs -pi+eps is physically ~perfect: wrapped MSE ~0,
    # unwrapped ~(2pi)^2 on that dim. xyz errors must pass through untouched,
    # and 6D-rotation widths (18) must not be wrapped at all.
    import torch

    from egomimic.eval.eval_pi import _wrap_aware_mse

    eps = 1e-3
    gt = torch.zeros(2, 12)
    pred = torch.zeros(2, 12)
    pred[:, 3] = math.pi - eps  # L yaw
    gt[:, 3] = -math.pi + eps
    wrapped, nowrap = _wrap_aware_mse(pred, gt)
    assert wrapped < 1e-4, f"wrap failed: {wrapped}"
    assert nowrap > 3.0, f"nowrap should show inflation: {nowrap}"

    # translation error passes through identically
    pred2 = torch.zeros(2, 12)
    pred2[:, 0] = 0.5
    w2, nw2 = _wrap_aware_mse(pred2, torch.zeros(2, 12))
    assert torch.isclose(w2, nw2) and torch.isclose(w2, torch.tensor(0.25 / 12))

    # 6D width: no angle dims -> wrapped == unwrapped even for large values
    pred3 = torch.full((2, 18), 4.0)
    w3, nw3 = _wrap_aware_mse(pred3, torch.zeros(2, 18))
    assert torch.isclose(w3, nw3)


def test_video_fps_compensates_for_world_size():
    # Distributed val strides an episode's frames by world_size on each rank;
    # playback fps must scale down to keep videos wall-clock real-time.
    from types import SimpleNamespace

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
