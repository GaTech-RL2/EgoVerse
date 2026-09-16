"""egomimic.eval.action_metrics, the trajectory metrics it uses, and the PI
evaluator's metric set."""

import math

import numpy as np
import pytest
import torch

from egomimic.eval.action_metrics import (
    cartesian_metrics,
    keypoint_metrics,
    layout_metrics,
)
from egomimic.utils.pose_utils import _ypr_to_rot6d


def test_layout_metrics_dispatch_on_the_action_key_not_just_the_width():
    pred, gt = torch.zeros(2, 5, 14), torch.ones(2, 5, 14)
    assert layout_metrics(pred, gt, "p", "actions_joints") == {}
    cart = layout_metrics(pred, gt, "p", "actions_cartesian")
    assert "p_xyz_paired_mse_avg" in cart
    assert "p_ypr_paired_mse_avg" in cart and "p_rot6d_paired_mse_avg" not in cart
    six = layout_metrics(
        torch.zeros(2, 5, 20), torch.ones(2, 5, 20), "p", "actions_cartesian"
    )
    assert "p_rot6d_paired_mse_avg" in six
    kp = layout_metrics(
        torch.zeros(2, 5, 144), torch.ones(2, 5, 144), "p", "actions_keypoints"
    )
    assert "p_kp_l2_avg" in kp


def test_pi_eval_falls_back_to_plain_mse_for_other_action_keys():
    from egomimic.eval.eval_pi import PIEvalVideo
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment

    eid = EMBODIMENT.EVA_BIMANUAL.value
    name = get_embodiment(eid).lower()
    ac_key = "actions_joints"
    gt = torch.zeros(2, 5, 14)
    pred = torch.ones(2, 5, 14)

    class _NormStats:
        def unnormalize(self, batch, embodiment_id):
            return batch

    class _Algo:
        ac_keys = {eid: ac_key}
        norm_stats = _NormStats()

        def forward_eval(self, batch):
            return {f"{name}_{ac_key}": pred}

    ev = PIEvalVideo(viz_func={}, transform_lists={}, viz_every_n_epochs=1)
    ev.model = _Algo()
    batch = {eid: {ac_key: gt, "embodiment": torch.tensor([eid])}}
    metrics, _ = ev.compute_metrics_and_viz(batch, do_viz=False)
    prefix = f"Valid/{name}_{ac_key}_"
    assert set(metrics) == {prefix + "paired_mse_avg", prefix + "final_mse_avg"}
    assert float(metrics[prefix + "paired_mse_avg"]) == 1.0


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


def test_segment_l2_separates_early_from_late_error():
    B, T = 2, 100
    gt = torch.zeros(B, T, 20)
    pred = gt.clone()
    pred[:, :10, 0] = 0.03  # 3 cm on the left arm x, first 10 % only
    m = cartesian_metrics(pred, gt, "p")
    # mean over both arms: 3 cm on one of two
    assert m["p_xyz_l2_early"].item() == pytest.approx(0.015)
    assert m["p_xyz_l2_mid"].item() == pytest.approx(0.0)
    assert m["p_xyz_l2_late"].item() == pytest.approx(0.0)


def test_segment_l2_on_keypoints():
    B, T = 1, 100
    gt = torch.zeros(B, T, 144)
    pred = gt.clone()
    pred[:, 50:, 9:72] = 0.02  # every left-hand keypoint off by (2,2,2) cm late
    m = keypoint_metrics(pred, gt, "k")
    expected = 0.5 * float(np.linalg.norm([0.02] * 3))  # one hand of two
    assert m["k_kp_l2_late"].item() == pytest.approx(expected, rel=1e-5)
    assert m["k_kp_l2_early"].item() == pytest.approx(0.0)
    assert m["k_wrist_xyz_l2_late"].item() == pytest.approx(0.0)
