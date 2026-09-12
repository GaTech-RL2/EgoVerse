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


def test_train_viz_wrapper_prefixes_and_disables_sampling():
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo
    from egomimic.eval.eval_video import EvalVideo

    class _Base(EvalVideo):
        def __init__(self):
            super().__init__(viz_func={}, transform_lists={}, viz_every_n_epochs=7)
            self.seen_samples = None

        def compute_metrics_and_viz(self, batch, do_viz=True):
            self.seen_samples = self.model.val_samples
            return {"Valid/x": 1.0}, {}

    class _Algo:
        val_samples = 8

    base = _Base()
    tv = TrainVizEvalVideo(base)
    tv.model = _Algo()  # property setter forwards to base too
    metrics, _ = tv.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"train_viz/Valid/x"}, metrics
    assert base.seen_samples == 1, "M-sample metrics must be forced off in train viz"
    assert tv.model.val_samples == 8, "val_samples must be restored after the call"
    assert tv.viz_every_n_epochs == 7, "wrapper inherits the base viz gate"
    from types import SimpleNamespace

    tv.trainer = SimpleNamespace(default_root_dir="/tmp/run")
    assert tv.video_dir().endswith("videos_train_viz")


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
    from egomimic.eval.eval_pi import _paired_mse, _split_mse

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

    from egomimic.eval.eval_pi import _rot_geodesic_error
    from egomimic.utils.pose_utils import _ypr_to_rot6d

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

    # Gimbal lock: pitch ≈ π/2, pred = gt rotated 0.004 rad about local y.
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


def _run_pi_eval(pred, gt, val_samples=1):
    from egomimic.eval.eval_pi import PIEvalVideo
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment

    eid = EMBODIMENT.HUMAN_BIMANUAL.value
    name = get_embodiment(eid).lower()

    class _NormStats:
        def unnormalize(self, batch, embodiment_id):
            return batch

    class _Algo:
        ac_keys = {eid: "actions_cartesian"}
        norm_stats = _NormStats()

        def __init__(self):
            self.val_samples = val_samples

        def forward_eval(self, batch):
            return {
                f"{name}_loss": torch.tensor(0.5),
                f"{name}_actions_cartesian": pred,
            }

        def sample_action_chunks(self, _batch, embodiment_id, M):
            return torch.stack([pred + 0.01 * i for i in range(M)], dim=0)

    ev = PIEvalVideo(viz_func={}, transform_lists={}, viz_every_n_epochs=1)
    ev.model = _Algo()
    batch = {eid: {"actions_cartesian": gt, "embodiment": torch.tensor([eid])}}
    metrics, images = ev.compute_metrics_and_viz(batch, do_viz=False)
    return f"Valid/{name}_actions_cartesian_", f"Valid/{name}_loss", metrics, images


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
        )
    } | {loss_key, "Valid/action_loss"}
    assert set(metrics) == expected, sorted(metrics)
    assert images == {}, "do_viz=False must not render"
    assert float(metrics[prefix + "xyz_paired_mse_avg"]) < 1e-10
    assert float(metrics[prefix + "rot_err_deg_avg"]) < 1e-3
    assert float(metrics[prefix + "xyz_dtw_avg"]) < 1e-6
    assert float(metrics[prefix + "xyz_frechet_gauss_avg"]) < 1e-2

    # M-sample metrics appear only when val_samples > 1
    _, _, m4, _ = _run_pi_eval(gt.clone(), gt, val_samples=4)
    assert set(m4) - expected == {
        prefix + "bestof4_paired_mse",
        prefix + "sample_diversity_M4",
    }, sorted(set(m4) - expected)


def test_pi_eval_metrics_are_invariant_to_a_shared_rigid_transform():
    # The evaluator no longer logs cam-frame twins of the native metrics:
    # the revert applies the SAME rigid transform (the observed wrist pose)
    # to prediction and gt, and every kept metric is invariant to that.
    # Re-express pred and gt in a random frame per (batch, arm) and check.
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
    ):
        a, b = float(native[prefix + key]), float(moved[prefix + key])
        assert abs(a - b) < atol, (key, a, b)
    # sanity: the perturbation is visible in the headline metrics
    assert float(native[prefix + "rot_err_deg_avg"]) > 1.0
    assert float(native[prefix + "xyz_paired_mse_avg"]) > 1e-4
