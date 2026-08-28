"""Independent end-to-end checks of the wrist-frame 6D pipeline.

The round-trip tests in ``test_pi05_norm_rot6d.py`` compare the pipeline
against itself (``_rot6d_to_ypr(_ypr_to_rot6d(x)) == x``), which a
consistently-wrong convention would pass, and the transform-list tests there
only assert which Transform classes are present. These tests instead run the
REAL transform lists on synthetic world-frame poses and compare every stage
against plain numpy/scipy SE(3) math that shares no code with the pipeline:

  raw world-frame poses
    -> Human/Eva.get_transform_list("cartesian_wristframe_6d")   (data)
    -> quantile normalize (the dataset formula)
    -> to32_norm_6d / from32_norm_6d                            (model I/O)
    -> unnormalize
    -> _build_*_cartesian_revert_6d_wristframe_transform_list   (evaluator)
    -> head/cam-frame xyz+ypr  ==  independent inv(T_head) @ T_action
"""

import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    CartesianYPRToRot6D,
    SplitKeys,
)
from egomimic.utils.action_utils import (
    HumanBimanualCartesianEuler,
    RobotBimanualCartesianEuler,
    _apply_norm_one,
    _apply_unnorm_one,
    _matrix_to_ypr,
    _reconstruct_R_from_cols,
    _ypr_to_matrix,
)
from egomimic.utils.pose_utils import _rot6d_to_ypr, _ypr_to_rot6d

T = 100


# ---------------------------------------------------------------- helpers
def _rng(seed):
    return np.random.default_rng(seed)


def _rand_pose(rng, scale=1.0):
    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()  # xyzw
    return np.concatenate([rng.uniform(-scale, scale, 3), q[[3, 0, 1, 2]]])


def _rand_chunk(rng, start):
    """Smooth random walk of xyz+quat(wxyz) poses starting AT ``start``."""
    out = np.zeros((T, 7))
    p = start[:3].copy()
    r = R.from_quat(start[[4, 5, 6, 3]])
    for t in range(T):
        if t > 0:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        q = r.as_quat()
        out[t] = np.concatenate([p, q[[3, 0, 1, 2]]])
    return out


def _T(p7):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(p7[[4, 5, 6, 3]]).as_matrix()
    M[:3, 3] = p7[:3]
    return M


def _T_chunk(c):
    return np.stack([_T(row) for row in c])


def _xyzypr(M):
    return np.concatenate(
        [M[..., :3, 3], R.from_matrix(M[..., :3, :3]).as_euler("ZYX")], axis=-1
    )


def _R_of_ypr(ypr):
    return R.from_euler("ZYX", ypr).as_matrix()


def _apply(transform_list, sample):
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    for t in transform_list:
        s = t.transform(s)
    return s


def _assert_pose12_close(got, ref, atol):
    """Compare (..., 12) xyz+ypr vectors: xyz directly, rotation via matrices
    (immune to ±π wrap and gimbal ambiguity)."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    for off in (0, 6):
        np.testing.assert_allclose(got[..., off : off + 3], ref[..., off : off + 3], atol=atol)
        Rg = _R_of_ypr(got[..., off + 3 : off + 6].reshape(-1, 3))
        Rr = _R_of_ypr(ref[..., off + 3 : off + 6].reshape(-1, 3))
        np.testing.assert_allclose(Rg, Rr, atol=atol)


def _quantile_stats(x, width):
    flat = x.reshape(-1, width)
    return {
        "quantile_1": np.percentile(flat, 1, axis=0).astype(np.float32),
        "quantile_99": np.percentile(flat, 99, axis=0).astype(np.float32),
    }


def _wrist6d_ref(obs_pose12, act_pose12, side):
    """Independent wrist-frame 6D block (T, 9) for one arm from head-frame
    xyz+ypr proprio (12,) and actions (T, 12)."""
    o = 6 * side
    To = np.eye(4)
    To[:3, :3] = _R_of_ypr(obs_pose12[o + 3 : o + 6])
    To[:3, 3] = obs_pose12[o : o + 3]
    Ta = np.stack([np.eye(4)] * act_pose12.shape[0])
    Ta[:, :3, :3] = _R_of_ypr(act_pose12[:, o + 3 : o + 6])
    Ta[:, :3, 3] = act_pose12[:, o : o + 3]
    Tw = np.linalg.inv(To)[None] @ Ta
    return np.concatenate([Tw[:, :3, 3], Tw[:, :3, 0], Tw[:, :3, 1]], axis=-1)


# ------------------------------------------------ convention cross-checks
def test_torch_ypr_matrix_matches_scipy_zyx():
    """The torch packers and the numpy transforms must agree on what 'ypr'
    means: intrinsic Z-Y-X (yaw about z, then pitch about the new y, then
    roll about the new x), radians."""
    ypr = _rng(0).uniform(-3.0, 3.0, size=(64, 3))
    R_torch = _ypr_to_matrix(torch.from_numpy(ypr)).numpy()
    R_scipy = R.from_euler("ZYX", ypr).as_matrix()
    np.testing.assert_allclose(R_torch, R_scipy, atol=1e-12)
    # ...and _matrix_to_ypr inverts it on the principal branch.
    back = _matrix_to_ypr(torch.from_numpy(R_scipy)).numpy()
    np.testing.assert_allclose(R.from_euler("ZYX", back).as_matrix(), R_scipy, atol=1e-12)
    # numpy 6D helpers use the same columns as the torch packers.
    six = _ypr_to_rot6d(ypr)
    np.testing.assert_allclose(six[:, :3], R_scipy[:, :, 0], atol=1e-12)
    np.testing.assert_allclose(six[:, 3:], R_scipy[:, :, 1], atol=1e-12)
    np.testing.assert_allclose(
        R.from_euler("ZYX", _rot6d_to_ypr(six)).as_matrix(), R_scipy, atol=1e-12
    )


def test_gram_schmidt_matches_independent_and_is_proper():
    rng = _rng(1)
    c1 = rng.normal(size=(32, 3))
    c2 = rng.normal(size=(32, 3))
    Rt = _reconstruct_R_from_cols(torch.from_numpy(c1), torch.from_numpy(c2)).numpy()
    # independent GS
    a = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
    b = c2 - (c2 * a).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    Rn = np.stack([a, b, np.cross(a, b)], axis=-1)
    np.testing.assert_allclose(Rt, Rn, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(Rt), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        Rt @ np.transpose(Rt, (0, 2, 1)), np.broadcast_to(np.eye(3), Rt.shape), atol=1e-12
    )


# ------------------------------------------------------------ human path
@pytest.mark.parametrize("fix_left", [False, True])
def test_human_wristframe_6d_pipeline_round_trips_to_headframe(fix_left):
    from egomimic.rldb.embodiment.human import (
        Human,
        _build_human_cartesian_revert_6d_wristframe_transform_list,
    )

    rng = _rng(2 + int(fix_left))
    B = 3
    raws = []
    for _ in range(B):
        head, lobs, robs = _rand_pose(rng), _rand_pose(rng), _rand_pose(rng)
        raws.append(
            {
                "obs_head_pose": head,
                "left.obs_ee_pose": lobs,
                "right.obs_ee_pose": robs,
                "left.action_ee_pose": _rand_chunk(rng, lobs),
                "right.action_ee_pose": _rand_chunk(rng, robs),
            }
        )
    fwd = Human.get_transform_list(
        "cartesian_wristframe_6d",
        stride=1,
        fix_mecka_left_wrist=fix_left,
        pad_proprio_gripper=True,
    )
    outs = [_apply(fwd, r) for r in raws]
    act6 = np.stack([o["actions_cartesian"] for o in outs])
    obs6 = np.stack([o["observations.state.ee_pose"] for o in outs])
    assert act6.shape == (B, T, 18) and obs6.shape == (B, 20)
    assert set(outs[0]) == {"actions_cartesian", "observations.state.ee_pose"}
    assert np.all(obs6[:, [9, 19]] == 0.0)

    # Independent head-frame ground truth. The Rz(180°) fix relabels the
    # LEFT hand's local axes (right-multiply), before any frame math.
    Rfix = np.eye(4)
    Rfix[:3, :3] = R.from_euler("z", np.pi).as_matrix()
    gt_act = np.zeros((B, T, 12))
    gt_obs = np.zeros((B, 12))
    for b, r in enumerate(raws):
        Th = _T(r["obs_head_pose"])
        for si, side in enumerate(("left", "right")):
            Ta = _T_chunk(r[f"{side}.action_ee_pose"])
            To = _T(r[f"{side}.obs_ee_pose"])
            if fix_left and side == "left":
                Ta = Ta @ Rfix
                To = To @ Rfix
            gt_act[b, :, 6 * si : 6 * si + 6] = _xyzypr(np.linalg.inv(Th)[None] @ Ta)
            gt_obs[b, 6 * si : 6 * si + 6] = _xyzypr(np.linalg.inv(Th) @ To)

    # (1) proprio = head-frame obs pose, 6D-encoded, grip slots zero
    obs_ref = np.stack(
        [CartesianYPRToRot6D(action_key="k").transform({"k": g})["k"] for g in gt_obs]
    )
    keep = [i for i in range(20) if i not in (9, 19)]
    np.testing.assert_allclose(obs6[:, keep], obs_ref, atol=1e-9)

    # (2) actions = each arm's pose in that arm's obs-wrist frame, 6D-encoded
    for b in range(B):
        for si in range(2):
            np.testing.assert_allclose(
                act6[b, :, 9 * si : 9 * si + 9],
                _wrist6d_ref(gt_obs[b], gt_act[b], si),
                atol=1e-9,
            )
    # t = 0 is the identity pose exactly (reference IS the obs pose) — the
    # bounds-check tolerance in MultiDataset._check_bounds relies on this.
    np.testing.assert_allclose(act6[:, 0, [0, 1, 2, 9, 10, 11]], 0.0, atol=1e-12)

    # (3) normalize -> 32D pack -> unpack -> unnormalize is exact
    st_act = _quantile_stats(act6, 18)
    st_obs = _quantile_stats(obs6, 20)
    a_t, o_t = torch.from_numpy(act6).float(), torch.from_numpy(obs6).float()
    a_n = _apply_norm_one(a_t, st_act, "quantile")
    o_n = _apply_norm_one(o_t, st_obs, "quantile")
    conv = HumanBimanualCartesianEuler()
    a32 = conv.to32_norm_6d(a_n)
    assert a32.shape == (B, T, 32)
    assert torch.all(a32[..., [9, 19]] == 0) and torch.all(a32[..., 20:] == 0)
    torch.testing.assert_close(conv.from32_norm_6d(a32), a_n, atol=0, rtol=0)
    a_un = _apply_unnorm_one(conv.from32_norm_6d(a32), st_act, "quantile")
    o_un = _apply_unnorm_one(o_n, st_obs, "quantile")
    torch.testing.assert_close(a_un, a_t, atol=1e-5, rtol=0)
    torch.testing.assert_close(o_un, o_t, atol=1e-5, rtol=0)

    # (4) evaluator revert (batched, like eval_pi) lands back in head frame
    rev = _build_human_cartesian_revert_6d_wristframe_transform_list()
    out = Embodiment.apply_transform(
        {"actions_cartesian": a_un, "observations.state.ee_pose": o_un}, rev
    )
    assert np.asarray(out["actions_cartesian"]).shape == (B, T, 12)
    assert np.asarray(out["observations.state.ee_pose"]).shape == (B, 12)
    _assert_pose12_close(out["actions_cartesian"], gt_act, atol=1e-4)
    _assert_pose12_close(out["observations.state.ee_pose"], gt_obs, atol=1e-4)

    # (5) a noisy (non-orthonormal) prediction still reverts to finite poses
    a_noisy = _apply_unnorm_one(a_n + 0.05 * torch.randn_like(a_n), st_act, "quantile")
    out_n = Embodiment.apply_transform(
        {"actions_cartesian": a_noisy, "observations.state.ee_pose": o_un}, rev
    )
    assert np.isfinite(np.asarray(out_n["actions_cartesian"])).all()


def test_human_wristframe_actions_are_headframe_invariant():
    """Wrist-relative targets must not depend on the head pose at all."""
    from egomimic.rldb.embodiment.human import Human

    rng = _rng(7)
    lobs, robs = _rand_pose(rng), _rand_pose(rng)
    raw = {
        "left.obs_ee_pose": lobs,
        "right.obs_ee_pose": robs,
        "left.action_ee_pose": _rand_chunk(rng, lobs),
        "right.action_ee_pose": _rand_chunk(rng, robs),
    }
    fwd = Human.get_transform_list("cartesian_wristframe_6d", stride=1)
    a1 = _apply(fwd, {**raw, "obs_head_pose": _rand_pose(rng)})["actions_cartesian"]
    a2 = _apply(fwd, {**raw, "obs_head_pose": _rand_pose(rng)})["actions_cartesian"]
    np.testing.assert_allclose(a1, a2, atol=1e-12)


# -------------------------------------------------------------- eva path
def _eva_extrinsics_variants():
    """Every calibration the checkout knows about. Older trees expose one
    ``Eva.EXTRINSICS`` dict; newer ones a keyed ``EVA_EXTRINSICS`` registry
    selected via ``get_transform_list(..., extrinsics_key=...)``."""
    import egomimic.rldb.embodiment.eva as eva_mod

    registry = getattr(eva_mod, "EVA_EXTRINSICS", None)
    if registry is None:
        return [pytest.param(None, eva_mod.Eva.EXTRINSICS, id="default")]
    return [pytest.param(k, v, id=k) for k, v in registry.items()]


@pytest.mark.parametrize("extrinsics_key,extrinsics", _eva_extrinsics_variants())
def test_eva_wristframe_6d_pipeline_round_trips_to_camframe(extrinsics_key, extrinsics):
    from egomimic.rldb.embodiment.eva import (
        Eva,
        _build_eva_cartesian_revert_6d_wristframe_transform_list,
    )

    rng = _rng(11)
    B = 3
    raws = []
    for _ in range(B):
        lobs, robs = _rand_pose(rng), _rand_pose(rng)
        raws.append(
            {
                "left.obs_ee_pose": lobs,
                "right.obs_ee_pose": robs,
                "left.cmd_ee_pose": _rand_chunk(rng, lobs),
                "right.cmd_ee_pose": _rand_chunk(rng, robs),
                "left.obs_gripper": rng.uniform(0, 1, (1,)),
                "right.obs_gripper": rng.uniform(0, 1, (1,)),
                "left.cmd_gripper": rng.uniform(0, 1, (T, 1)),
                "right.cmd_gripper": rng.uniform(0, 1, (T, 1)),
            }
        )
    kwargs = {} if extrinsics_key is None else {"extrinsics_key": extrinsics_key}
    fwd = Eva.get_transform_list("cartesian_wristframe_6d", **kwargs)
    outs = [_apply(fwd, r) for r in raws]
    act6 = np.stack([o["actions_cartesian"] for o in outs])
    obs6 = np.stack([o["observations.state.ee_pose"] for o in outs])
    assert act6.shape == (B, T, 20) and obs6.shape == (B, 20)

    # independent cam-frame GT: T_cam = inv(E_side) @ T_base
    gt_act = np.zeros((B, T, 14))
    gt_obs = np.zeros((B, 14))
    for b, r in enumerate(raws):
        for si, side in enumerate(("left", "right")):
            Einv = np.linalg.inv(np.asarray(extrinsics[side]))
            gt_act[b, :, 7 * si : 7 * si + 6] = _xyzypr(Einv[None] @ _T_chunk(r[f"{side}.cmd_ee_pose"]))
            gt_act[b, :, 7 * si + 6] = r[f"{side}.cmd_gripper"][:, 0]
            gt_obs[b, 7 * si : 7 * si + 6] = _xyzypr(Einv @ _T(r[f"{side}.obs_ee_pose"]))
            gt_obs[b, 7 * si + 6] = r[f"{side}.obs_gripper"][0]

    # the x5Dec13_2 rig calibration is ~6e-9 off orthonormal, hence 1e-7
    obs_ref = np.stack(
        [CartesianYPRToRot6D(action_key="k").transform({"k": g})["k"] for g in gt_obs]
    )
    np.testing.assert_allclose(obs6, obs_ref, atol=1e-7)
    pose_idx = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    for b in range(B):
        for si in range(2):
            ref = _wrist6d_ref(gt_obs[b][pose_idx], gt_act[b][:, pose_idx], si)
            got = act6[b, :, 10 * si : 10 * si + 10]
            np.testing.assert_allclose(got[:, :9], ref, atol=1e-7)
            np.testing.assert_allclose(got[:, 9], gt_act[b, :, 7 * si + 6], atol=1e-12)

    st_act, st_obs = _quantile_stats(act6, 20), _quantile_stats(obs6, 20)
    a_t, o_t = torch.from_numpy(act6).float(), torch.from_numpy(obs6).float()
    a_n = _apply_norm_one(a_t, st_act, "quantile")
    o_n = _apply_norm_one(o_t, st_obs, "quantile")
    conv = RobotBimanualCartesianEuler()
    a32 = conv.to32_norm_6d(a_n)
    torch.testing.assert_close(conv.from32_norm_6d(a32), a_n, atol=0, rtol=0)
    a_un = _apply_unnorm_one(conv.from32_norm_6d(a32), st_act, "quantile")
    o_un = _apply_unnorm_one(o_n, st_obs, "quantile")

    rev = _build_eva_cartesian_revert_6d_wristframe_transform_list()
    out = Embodiment.apply_transform(
        {"actions_cartesian": a_un, "observations.state.ee_pose": o_un}, rev
    )
    ra = np.asarray(out["actions_cartesian"])
    ro = np.asarray(out["observations.state.ee_pose"])
    assert ra.shape == (B, T, 14) and ro.shape == (B, 14)
    _assert_pose12_close(ra[..., pose_idx], gt_act[..., pose_idx], atol=1e-4)
    np.testing.assert_allclose(ra[..., [6, 13]], gt_act[..., [6, 13]], atol=1e-5)
    _assert_pose12_close(ro[..., pose_idx], gt_obs[..., pose_idx], atol=1e-4)


# ------------------------------------------------------ silent-failure guards
def test_split_keys_rejects_width_mismatch():
    """A ypr revert list fed a 6D batch used to slice 'xyz + col0' as
    'xyz + ypr' without complaint; now it must fail loudly."""
    sk = SplitKeys(input_key="k", output_key_list=[("a", 6), ("b", 6)])
    ok = sk.transform({"k": np.zeros((4, 12))})
    assert ok["a"].shape == (4, 6) and ok["b"].shape == (4, 6)
    with pytest.raises(ValueError, match="last dim 18"):
        sk.transform({"k": np.zeros((4, 18))})
    with pytest.raises(ValueError, match="sums to 12"):
        sk.transform({"k": torch.zeros(4, 20)})


def test_ypr_revert_on_6d_batch_fails_loudly():
    """The evaluator/data-config mismatch the guard is for: eval_pi.yaml's
    ypr revert applied to a cartesian_wristframe_6d batch."""
    from egomimic.rldb.embodiment.human import (
        _build_human_cartesian_revert_eef_frame_transform_list,
    )

    rev = _build_human_cartesian_revert_eef_frame_transform_list(is_quat=False)
    batch = {
        "actions_cartesian": torch.zeros(2, T, 18),
        "observations.state.ee_pose": torch.zeros(2, 20),
    }
    with pytest.raises(ValueError, match="SplitKeys"):
        Embodiment.apply_transform(batch, rev)


def _bounds_dataset(key, width, q_low, q_high):
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.norm_stats = {
        0: {
            key: {
                "quantile_1": np.full(width, q_low, dtype=np.float32),
                "quantile_99": np.full(width, q_high, dtype=np.float32),
            }
        }
    }
    md.zarr_keys = {0: {key: key}}
    md._warned_violations = set()
    return md


def test_bounds_check_tolerates_roundoff_on_collapsed_bounds():
    """Wrist-frame t=0 cells have bounds [0, 0]; roundoff must not reject."""
    md = _bounds_dataset("actions_cartesian", 18, 0.0, 0.0)
    arr = np.zeros((5, 18), dtype=np.float32)
    arr[0, 0] = 1e-9  # a roundoff-scale xyz value at a [0, 0] bound
    assert md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep") is None
    arr[0, 0] = 1e-3  # a real violation is still caught
    assert md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep") is not None


def test_bounds_check_warns_once_on_stat_shape_mismatch(caplog):
    md = _bounds_dataset("actions_cartesian", 18, -1.0, 1.0)
    arr = np.zeros((5, 20), dtype=np.float32)  # stats are 18-wide
    with caplog.at_level("WARNING"):
        assert md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 0, "ep") is None
        assert md._check_bounds({"embodiment": 0, "actions_cartesian": arr}, None, 1, "ep") is None
    msgs = [r.message for r in caplog.records if "bounds check skipped" in r.message]
    assert len(msgs) == 1, msgs


def test_precomputed_norm_stats_provenance_is_checked(tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    writer = MultiDataset.__new__(MultiDataset)
    writer.norm_mode = "quantile"
    writer._norm_run_metadata = None
    writer.norm_stats = {
        1: {
            "actions_cartesian": {"quantile_1": np.zeros((T, 18)), "quantile_99": np.ones((T, 18))},
            "observations.state.ee_pose": {"quantile_1": np.zeros(20), "quantile_99": np.ones(20)},
        }
    }
    writer.cache_stats(str(tmp_path))
    path = tmp_path / "norm_stats" / "norm_stats.json"
    payload = json.loads(path.read_text())
    assert payload["provenance"]["norm_mode"] == "quantile"
    assert payload["provenance"]["stat_shapes"]["1"]["actions_cartesian"] == [T, 18]

    def reader(norm_mode):
        md = MultiDataset.__new__(MultiDataset)
        md.norm_mode = norm_mode
        md.norm_stats = {1: {}}
        md._norm_run_metadata = None
        return md

    keys = ["actions_cartesian", "observations.state.ee_pose"]
    good = reader("quantile")
    good._load_precomputed_stats(str(path), 1, keys)
    assert set(good.norm_stats[1]) == set(keys)
    with pytest.raises(ValueError, match="norm_mode='zscore'"):
        reader("zscore")._load_precomputed_stats(str(path), 1, keys)
    with pytest.raises(ValueError, match="different keymap/transform mode"):
        reader("quantile")._load_precomputed_stats(str(path), 1, ["actions_cartesian"])
    with pytest.raises(ValueError, match="no entry for embodiment id 2"):
        reader("quantile")._load_precomputed_stats(str(path), 2, keys)
