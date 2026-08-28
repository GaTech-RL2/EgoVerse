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


def test_frechet_and_reverse_kl_helpers():
    from egomimic.utils.metrics import (
        frechet_gaussian_over_time,
        reverse_kl_from_samples,
    )

    B, T, D = 4, 10, 6
    torch.manual_seed(0)
    pred = torch.randn(B, T, D)
    fd = frechet_gaussian_over_time(pred, pred.clone())
    assert fd.shape == (B,)
    assert fd.max() < 1e-2, "identical distributions must score ~0"
    fd_shift = frechet_gaussian_over_time(pred + 5.0, pred)
    assert (fd_shift > fd).all(), "mean shift must increase the distance"

    samples = torch.randn(3, B, T, D)
    rkl = reverse_kl_from_samples(samples, pred)
    assert rkl.ndim == 0 and torch.isfinite(rkl)


def test_train_viz_wrapper_prefixes_and_disables_rkl():
    from egomimic.eval.eval_train_viz import TrainVizEvalVideo
    from egomimic.eval.eval_video import EvalVideo

    class _Base(EvalVideo):
        def __init__(self):
            super().__init__(viz_func={}, transform_lists={}, viz_every_n_epochs=7)
            self.seen_rkl = None

        def compute_metrics_and_viz(self, batch, do_viz=True):
            self.seen_rkl = self.model.rkl_samples
            return {"Valid/x": 1.0}, {}

    class _Algo:
        rkl_samples = 8

    base = _Base()
    tv = TrainVizEvalVideo(base)
    tv.model = _Algo()  # property setter forwards to base too
    metrics, _ = tv.compute_metrics_and_viz({}, do_viz=False)
    assert set(metrics) == {"train_viz/Valid/x"}, metrics
    assert base.seen_rkl == 1, "M-sample metrics must be forced off in train viz"
    assert tv.model.rkl_samples == 8, "rkl_samples must be restored after the call"
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


def test_resize_image_keys_unifies_mixed_resolutions():
    # abc eva ships 480x640 / 480x848 / 720x1280 episodes; the collate-level
    # resize must unify camera images (both dataset-style "images" keys and
    # PI-style "*_rgb" keys) so default_collate can stack, while leaving
    # non-image tensors untouched.
    from torch.utils.data._utils.collate import default_collate

    from egomimic.pl_utils.pl_data_utils import _resize_image_keys

    batch = [
        {
            "base_0_rgb": torch.rand(3, 480, 640),
            "observations.images.left_wrist_img": torch.rand(3, 480, 848),
            "actions_cartesian": torch.rand(100, 20),
        },
        {
            "base_0_rgb": torch.rand(3, 720, 1280),
            "observations.images.left_wrist_img": torch.rand(3, 480, 640),
            "actions_cartesian": torch.rand(100, 20),
        },
    ]
    _resize_image_keys(batch)
    for sample in batch:
        assert sample["base_0_rgb"].shape == (3, 480, 640)
        assert sample["observations.images.left_wrist_img"].shape == (3, 480, 640)
        assert sample["actions_cartesian"].shape == (100, 20), "non-image resized!"
    stacked = default_collate(batch)
    assert stacked["base_0_rgb"].shape == (2, 3, 480, 640)

    # (T, C, H, W) temporal stacks resize only the trailing spatial dims,
    # and dtype is preserved.
    b2 = [
        {"images.front_1": torch.randint(0, 255, (4, 3, 480, 848), dtype=torch.uint8)}
    ]
    _resize_image_keys(b2)
    assert b2[0]["images.front_1"].shape == (4, 3, 480, 640)
    assert b2[0]["images.front_1"].dtype == torch.uint8


def _hand(pinch_m, curl_ratio):
    # Synthetic 21-kp hand with controlled pinch distance and curl ratio:
    # wrist at origin, middle MCP 10cm out (hand size 0.1).
    kp = np.zeros((21, 3))
    kp[9] = [0.10, 0.0, 0.0]
    kp[5], kp[13], kp[17] = [0.09, 0.02, 0], [0.09, -0.02, 0], [0.08, -0.04, 0]
    palm = np.mean([kp[5], kp[9], kp[13], kp[17]], axis=0)
    for t in (8, 12, 16, 20):  # fingertips at the target curl radius
        kp[t] = palm + np.array([curl_ratio * 0.1, 0, 0])
    kp[4] = kp[8] + np.array([0, pinch_m, 0])  # thumb tip at pinch distance
    return kp


def test_keypoints_to_gripper_pinch_and_curl():
    from egomimic.rldb.zarr.action_chunk_transforms import KeypointsToGripper

    t = KeypointsToGripper(chunk_length=10, stride=1)
    # open hand: wide pinch + extended fingers
    assert t._openness(_hand(0.12, 1.5)) == 1.0
    # pinch closed even with extended fingers
    assert t._openness(_hand(0.005, 1.5)) == 0.0
    # power grasp: fist curls while pinch stays wide
    assert t._openness(_hand(0.12, 0.65)) == 0.0
    # untracked hand reads fully open
    assert t._openness(np.zeros((21, 3))) == 1.0

    # end-to-end: horizon series + obs frame -> chunked grip + scalar
    batch = {
        "left.action_grip_keypoints": np.stack([_hand(0.12, 1.5)] * 6).reshape(6, 63),
        "left.obs_grip_keypoints": _hand(0.005, 1.5).reshape(63),
    }
    out = t.transform(batch)
    assert out["left.action_grip"].shape == (10, 1)
    np.testing.assert_allclose(out["left.action_grip"], 1.0)
    assert out["left.obs_grip"].shape == (1,) and out["left.obs_grip"][0] == 0.0
    assert "left.action_grip_keypoints" not in out


def test_insert_gripper_channels_layout():
    from egomimic.rldb.zarr.action_chunk_transforms import InsertGripperChannels
    from egomimic.utils.pose_utils import bimanual_cartesian_layout

    T = 4
    actions = np.arange(T * 18, dtype=np.float64).reshape(T, 18)
    batch = {
        "actions_cartesian": actions.copy(),
        "left.action_grip": np.full((T, 1), 0.25),
        "right.action_grip": np.full((T, 1), 0.75),
    }
    out = InsertGripperChannels(
        action_key="actions_cartesian",
        left_grip_key="left.action_grip",
        right_grip_key="right.action_grip",
    ).transform(batch)
    a = out["actions_cartesian"]
    assert a.shape == (T, 20)
    grip_idx = bimanual_cartesian_layout(20)["grip"]
    np.testing.assert_allclose(a[:, grip_idx[0]], 0.25)
    np.testing.assert_allclose(a[:, grip_idx[1]], 0.75)
    # non-grip channels preserved in order
    keep = [i for i in range(20) if i not in grip_idx]
    np.testing.assert_allclose(a[:, keep], actions)
    assert "left.action_grip" not in out


def test_keypoint_gripper_transform_list_wiring():
    from egomimic.rldb.embodiment.human import (
        Human,
        _build_human_cartesian_revert_6d_wristframe_grip_transform_list,
    )
    from egomimic.rldb.zarr.action_chunk_transforms import (
        InsertGripperChannels,
        KeypointsToGripper,
        UnpadGripperZeros,
    )

    tl = Human.get_transform_list("cartesian_wristframe_6d", keypoint_gripper=True)
    assert isinstance(tl[0], KeypointsToGripper), "grip extraction must run first"
    inserts = [t for t in tl if isinstance(t, InsertGripperChannels)]
    assert {t.action_key for t in inserts} == {
        "actions_cartesian",
        "observations.state.ee_pose",
    }
    with pytest.raises(ValueError):
        Human.get_transform_list(
            "cartesian_wristframe_6d", keypoint_gripper=True, pad_proprio_gripper=True
        )

    rl = _build_human_cartesian_revert_6d_wristframe_grip_transform_list()
    assert isinstance(rl[0], UnpadGripperZeros)
    assert rl[0].action_key == "actions_cartesian"

    km = Human.get_keymap("cartesian_pi", include_grip_keypoints=True)
    assert km["left.action_grip_keypoints"]["horizon"] == Human.ACTION_HORIZON
    assert km["right.obs_grip_keypoints"]["zarr_key"] == "right.obs_keypoints"


def test_split_mse_is_stateless_and_matches_manual():
    from egomimic.eval.eval_pi import _paired_mse, _split_mse

    rng = np.random.default_rng(21)
    pred = torch.from_numpy(rng.normal(size=(4, 10, 18))).float()
    gt = torch.from_numpy(rng.normal(size=(4, 10, 18))).float()
    xyz_idx = [0, 1, 2, 9, 10, 11]
    rot_idx = [i for i in range(18) if i not in xyz_idx]
    xyz, rot = _split_mse(pred, gt)
    torch.testing.assert_close(xyz, (pred[..., xyz_idx] - gt[..., xyz_idx]).pow(2).mean())
    torch.testing.assert_close(rot, (pred[..., rot_idx] - gt[..., rot_idx]).pow(2).mean())
    # stateless: a second, unrelated call is unaffected by the first
    a, b = torch.zeros(2, 3, 18), torch.ones(2, 3, 18)
    torch.testing.assert_close(_split_mse(a, b)[0], torch.tensor(1.0))
    torch.testing.assert_close(_paired_mse(a, b), torch.tensor(1.0))
    assert _split_mse(torch.zeros(2, 3, 7), torch.zeros(2, 3, 7)) == (None, None)


def test_rot_geodesic_error_matches_angle_and_survives_gimbal_lock():
    from scipy.spatial.transform import Rotation as R

    from egomimic.eval.eval_pi import _rot_geodesic_error, _wrap_aware_mse
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
    gt12 = torch.cat([torch.zeros(6, 5, 3), gt12[..., :3], torch.zeros(6, 5, 3), gt12[..., 3:]], -1)
    pr12 = torch.from_numpy(np.concatenate([ypr_p, ypr_p], -1)).float()
    pr12 = torch.cat([torch.zeros(6, 5, 3), pr12[..., :3], torch.zeros(6, 5, 3), pr12[..., 3:]], -1)
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
    # The ypr MSE (even wrap-aware) explodes because yaw and roll trade off;
    # the geodesic error reports the true 0.004.
    gt = np.array([0.3, np.pi / 2 - 0.002, 0.2])
    Rgt = R.from_euler("ZYX", gt)
    Rpr = Rgt * R.from_rotvec([0.0, 0.004, 0.0])
    pr = Rpr.as_euler("ZYX")
    v_gt = torch.tensor([[0, 0, 0, *gt, 0, 0, 0, *gt]], dtype=torch.float32)
    v_pr = torch.tensor([[0, 0, 0, *pr, 0, 0, 0, *pr]], dtype=torch.float32)
    wrapped, _ = _wrap_aware_mse(v_pr, v_gt)
    geo = _rot_geodesic_error(v_pr, v_gt).item()
    assert abs(geo - 0.004) < 1e-3, geo
    assert wrapped.item() > 0.1, wrapped  # the Euler metric is fooled here
