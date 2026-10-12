"""Padding exclusion and per-third MSE in the layout metrics."""

import pytest
import torch

from egomimic.eval.action_metrics import cartesian_metrics, keypoint_metrics

B, T = 3, 12
# cartesian 18D: xyz at 0:3 and 9:12; keypoint 144D: wrist xyz at 0:3 and 72:75
CART_XYZ, KP_WRIST = 0, 0


def _identity_rot6d(width):
    x = torch.zeros(B, T, width)
    if width == 18:
        for arm in (0, 9):
            x[..., arm + 3] = 1.0
            x[..., arm + 7] = 1.0
    else:
        for hand in (0, 72):
            x[..., hand + 3] = 1.0
            x[..., hand + 7] = 1.0
    return x


@pytest.mark.parametrize(
    "fn, width", [(cartesian_metrics, 18), (keypoint_metrics, 144)]
)
def test_all_ones_mask_matches_no_mask(fn, width):
    gt = _identity_rot6d(width)
    pred = gt + 0.01 * torch.randn(
        B, T, width, generator=torch.Generator().manual_seed(0)
    )
    plain = fn(pred, gt, "p")
    masked = fn(pred, gt, "p", mask=torch.ones(B, T, 1))
    assert plain.keys() == masked.keys()
    for k in plain:
        torch.testing.assert_close(
            torch.as_tensor(masked[k]), torch.as_tensor(plain[k]), msg=k
        )


def _padded_case(width, xyz):
    """Row 0 is real for 4 steps, then padding with a huge error; the real
    steps carry a 0.1 error on one xyz channel."""
    gt = _identity_rot6d(width)
    pred = gt.clone()
    pred[:, :, xyz] += 0.1
    pred[0, 4:, xyz] += 100.0
    mask = torch.ones(B, T)
    mask[0, 4:] = 0
    return pred, gt, mask


def test_padding_is_excluded_from_the_cartesian_metrics():
    pred, gt, mask = _padded_case(18, CART_XYZ)
    m = cartesian_metrics(pred, gt, "p", mask=mask)
    # one of six xyz channels off by 0.1
    expected = 0.1**2 / 6
    assert m["p_xyz_paired_mse_avg"].item() == pytest.approx(expected, rel=1e-5)
    assert m["p_xyz_final_mse_avg"].item() == pytest.approx(expected, rel=1e-5)
    for third in ("early", "mid", "late"):
        assert m[f"p_xyz_paired_mse_{third}_third"].item() == pytest.approx(
            expected, rel=1e-5
        )
    assert m["p_xyz_l2_late"].item() == pytest.approx(0.05, rel=1e-5)
    assert cartesian_metrics(pred, gt, "p")["p_xyz_paired_mse_avg"] > 100


def test_padding_is_excluded_from_the_keypoint_metrics():
    pred, gt, mask = _padded_case(144, KP_WRIST)
    m = keypoint_metrics(pred, gt, "k", mask=mask)
    expected = 0.1**2 / 6
    assert m["k_wrist_xyz_paired_mse_avg"].item() == pytest.approx(expected, rel=1e-5)
    assert m["k_wrist_xyz_final_mse_avg"].item() == pytest.approx(expected, rel=1e-5)
    for third in ("early", "mid", "late"):
        assert m[f"k_wrist_xyz_paired_mse_{third}_third"].item() == pytest.approx(
            expected, rel=1e-5
        )


def test_thirds_localize_the_error_and_skip_rows_without_real_steps():
    gt = _identity_rot6d(144)
    pred = gt.clone()
    pred[:, 2 * T // 3 :, KP_WRIST] += 0.3
    m = keypoint_metrics(pred, gt, "k")
    assert m["k_wrist_xyz_paired_mse_early_third"].item() == 0
    assert m["k_wrist_xyz_paired_mse_mid_third"].item() == 0
    assert m["k_wrist_xyz_paired_mse_late_third"].item() == pytest.approx(
        0.09 / 6, rel=1e-5
    )
    # row 0 ends in the first third: it votes there only
    mask = torch.ones(B, T)
    mask[0, 2:] = 0
    pred[0, :2, KP_WRIST] += 0.6
    m = keypoint_metrics(pred, gt, "k", mask=mask)
    assert m["k_wrist_xyz_paired_mse_late_third"].item() == pytest.approx(
        0.09 / 6, rel=1e-5
    )
    assert m["k_wrist_xyz_paired_mse_early_third"].item() == pytest.approx(
        (0.36 / 6) / B, rel=1e-5
    )
    # no row reaches the last third: it is omitted, not zero
    mask[:, 2 * T // 3 :] = 0
    m = keypoint_metrics(pred, gt, "k", mask=mask)
    assert "k_wrist_xyz_paired_mse_late_third" not in m


def test_each_trajectory_counts_once():
    gt = _identity_rot6d(18)
    pred = gt.clone()
    pred[0, :, CART_XYZ] += 0.1
    mask = torch.ones(B, T)
    mask[0, 1:] = 0  # row 0 has one real step
    m = cartesian_metrics(pred, gt, "p", mask=mask)
    assert m["p_xyz_paired_mse_avg"].item() == pytest.approx((0.01 / 6) / B, rel=1e-5)


def test_a_mismatched_mask_is_rejected():
    gt = _identity_rot6d(18)
    with pytest.raises(ValueError, match="pad mask shape"):
        cartesian_metrics(gt, gt, "p", mask=torch.ones(B, T - 1))


def test_pi_evaluator_passes_the_pad_mask():
    from egomimic.eval.eval_pi import PIEvalVideo
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT

    eid = EMBODIMENT.HUMAN_BIMANUAL.value
    pred, gt, mask = _padded_case(144, KP_WRIST)

    class _NormStats:
        def unnormalize(self, batch, embodiment_id):
            return batch

    class _Algo:
        ac_keys = {eid: "actions_keypoints"}
        norm_stats = _NormStats()

        def forward_eval(self, batch):
            return {"human_bimanual_actions_keypoints": pred}

    ev = PIEvalVideo(viz_func={}, transform_lists={}, viz_every_n_epochs=1)
    ev.model = _Algo()
    batch = {eid: {"actions_keypoints": gt, "action_pad_mask": mask}}
    metrics, _ = ev.compute_metrics_and_viz(batch, do_viz=False)
    key = "Valid/human_bimanual_actions_keypoints_wrist_xyz_paired_mse_avg"
    assert metrics[key].item() == pytest.approx(0.1**2 / 6, rel=1e-5)
