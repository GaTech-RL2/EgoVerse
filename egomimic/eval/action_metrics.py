"""Layout-aware action metrics shared by the PI and HPT evaluators.

Every metric here is computed on the model's native (unnormalized) output and
is invariant to the rigid transform the evaluator's cam-frame revert applies
to prediction and ground truth alike, so it never needs the revert.

Two layouts are recognised by width:

- bimanual cartesian (12/14 ypr, 18/20 rot6d; see
  ``pose_utils.bimanual_cartesian_layout``): position MSE, geodesic rotation
  error, raw rotation-channel MSE, DTW + Frechet on xyz;
- wrist-first bimanual hand keypoints (138 ypr / 144 rot6d wrist; see
  ``pose_utils.bimanual_keypoint_layout``): mean keypoint L2 (m), wrist
  position MSE, wrist geodesic rotation error, DTW on the keypoint block.

Unknown widths yield no metrics.
"""

from __future__ import annotations

import math

import torch

from egomimic.utils.action_utils import _reconstruct_R_from_cols, _ypr_to_matrix
from egomimic.utils.metrics import dtw_distance, frechet_gaussian_over_time
from egomimic.utils.pose_utils import (
    bimanual_cartesian_layout,
    bimanual_keypoint_layout,
)

_RAD2DEG = 180.0 / math.pi


def _paired_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Plain elementwise MSE, stateless (no torchmetrics accumulator)."""
    return (pred.float() - gt.float()).pow(2).mean()


# Horizon segments for the per-segment L2 metrics, as fractions of the chunk:
# early = the first 10 % of timesteps, mid = 10-50 %, late = the second half.
# They separate near-term precision from long-range drift, which one
# chunk-mean error hides (and which per-timestep vs horizon-pooled norm stats
# trade against each other).
_HORIZON_SEGMENTS = (("early", 0.0, 0.1), ("mid", 0.1, 0.5), ("late", 0.5, 1.0))


def _segment_l2(pred: torch.Tensor, gt: torch.Tensor, idx: list, prefix: str) -> dict:
    """Mean euclidean error (m) of the xyz triples at ``idx`` per horizon
    segment of a ``(B, T, D)`` chunk."""
    T = pred.shape[1]
    diff = pred[..., idx].float() - gt[..., idx].float()
    l2 = diff.reshape(*diff.shape[:-1], -1, 3).norm(dim=-1)  # (B, T, P)
    out = {}
    for name, lo, hi in _HORIZON_SEGMENTS:
        start = min(int(round(lo * T)), T - 1)
        end = max(int(round(hi * T)), start + 1)
        out[f"{prefix}_{name}"] = l2[:, start:end].mean()
    return out


def _split_mse(pred_t: torch.Tensor, gt_t: torch.Tensor):
    """(translation MSE, rotation-channel MSE) over a bimanual cartesian
    vector, so a translation problem reads apart from a rotation one. The
    rotation channels are whatever the layout holds: the continuous 6D
    rotation-matrix columns for the 18D (human) / 20D (robot) widths,
    yaw/pitch/roll for the 12/14D widths. Either way this is a plain
    per-channel MSE (pre-Gram-Schmidt, axis-dependent, unitless for 6D; +-pi
    wrap and gimbal lock for ypr), i.e. a training-tracking diagnostic, not a
    rotation error; see :func:`_rot_geodesic_error` for that. Returns
    (None, None) for an unknown width.
    """
    layout = bimanual_cartesian_layout(pred_t.shape[-1])
    if layout is None:
        return None, None
    xyz_idx = list(layout["xyz"])
    rot_idx = list(layout["rot"])
    xyz = _paired_mse(pred_t[..., xyz_idx], gt_t[..., xyz_idx])
    rot = _paired_mse(pred_t[..., rot_idx], gt_t[..., rot_idx])
    return xyz, rot


def _geodesic_from_rot_channels(pred: torch.Tensor, gt: torch.Tensor, rot: list):
    """Mean geodesic angle (rad) over both arms given the rotation channel
    indices of a per-arm-concatenated vector (3 ypr or 6 rot6d per arm)."""
    per_arm = len(rot) // 2
    # float64: arccos near 1 loses ~sqrt(eps), which in float32 puts a
    # ~1e-4 rad floor under identical rotations; double makes it ~1e-8.
    pred = pred.double()
    gt = gt.double()
    errs = []
    for arm in (rot[:per_arm], rot[per_arm:]):
        p, g = pred[..., arm], gt[..., arm]
        if per_arm == 3:
            Rp, Rg = _ypr_to_matrix(p), _ypr_to_matrix(g)
        else:
            Rp = _reconstruct_R_from_cols(p[..., 0:3], p[..., 3:6])
            Rg = _reconstruct_R_from_cols(g[..., 0:3], g[..., 3:6])
        tr = (Rp.transpose(-1, -2) @ Rg).diagonal(dim1=-2, dim2=-1).sum(-1)
        errs.append(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
    return torch.stack(errs, dim=-1).mean().float()


def _rot_geodesic_error(pred: torch.Tensor, gt: torch.Tensor):
    """Mean geodesic rotation error in radians over batch / time / both arms
    of a bimanual cartesian vector.

    Euler-free: per arm, builds proper rotation matrices (``_ypr_to_matrix``
    for the 12/14-dim ypr widths; Gram-Schmidt on the two 6D columns for the
    18/20-dim widths, the same reconstruction the model decode uses) and
    takes ``arccos((tr(R_pred^T R_gt) - 1) / 2)``. Unlike a ypr MSE this is
    immune to the +-pi wrap AND to the yaw/roll degeneracy at pitch ~ +-pi/2.
    Returns None for an unknown width.
    """
    layout = bimanual_cartesian_layout(pred.shape[-1])
    if layout is None:
        return None
    return _geodesic_from_rot_channels(pred, gt, list(layout["rot"]))


def cartesian_metrics(pred: torch.Tensor, gt: torch.Tensor, prefix: str) -> dict:
    """Metrics for a bimanual cartesian chunk ``(B, T, D)``; ``{}`` if ``D``
    is not a cartesian width. ``pred``/``gt`` may live on any device."""
    pred_cpu, gt_cpu = pred.detach().cpu(), gt.detach().cpu()
    layout = bimanual_cartesian_layout(pred_cpu.shape[-1])
    if layout is None:
        return {}
    metrics = {}
    xyz_idx = list(layout["xyz"])
    xyz_p, rot_p = _split_mse(pred_cpu, gt_cpu)
    metrics[f"{prefix}_xyz_paired_mse_avg"] = xyz_p
    metrics[f"{prefix}_rot6d_paired_mse_avg"] = rot_p
    metrics[f"{prefix}_xyz_final_mse_avg"] = _paired_mse(
        pred_cpu[:, -1, xyz_idx], gt_cpu[:, -1, xyz_idx]
    )
    metrics.update(_segment_l2(pred_cpu, gt_cpu, xyz_idx, f"{prefix}_xyz_l2"))
    # Distributional / alignment metrics on the position channels only
    # (metres), so a unitless 6D column never trades off against a metre.
    # DTW forgives temporal misalignment (a correct motion executed early /
    # late scores near zero where paired MSE penalizes it); Frechet compares
    # the time-distribution shape of the motion.
    pred_xyz = pred[..., xyz_idx]
    gt_xyz = gt[..., xyz_idx].to(pred_xyz.device)
    metrics[f"{prefix}_xyz_dtw_avg"] = dtw_distance(pred_xyz, gt_xyz).mean().item()
    metrics[f"{prefix}_xyz_frechet_gauss_avg"] = (
        frechet_gaussian_over_time(pred_xyz, gt_xyz).mean().item()
    )
    # Rotation error on the manifold, in degrees: what the decode actually
    # produces after Gram-Schmidt, free of wrap / gimbal artefacts.
    metrics[f"{prefix}_rot_err_deg_avg"] = (
        _rot_geodesic_error(pred_cpu, gt_cpu) * _RAD2DEG
    )
    metrics[f"{prefix}_rot_err_deg_final"] = (
        _rot_geodesic_error(pred_cpu[:, -1], gt_cpu[:, -1]) * _RAD2DEG
    )
    return metrics


def keypoint_metrics(pred: torch.Tensor, gt: torch.Tensor, prefix: str) -> dict:
    """Metrics for a wrist-first bimanual keypoint chunk ``(B, T, D)``
    (D = 138 ypr wrist or 144 rot6d wrist); ``{}`` for other widths.

    ``kp_l2`` is the mean euclidean keypoint error in metres, in whatever
    frame the layout holds (the wrist frame for the ``*_wristframe_*`` modes,
    head frame otherwise); ``wrist_xyz`` / ``wrist_rot_err_deg`` score the
    wrist pose channels; ``kp_dtw`` is DTW over the flattened keypoint block.
    """
    pred_cpu, gt_cpu = pred.detach().cpu(), gt.detach().cpu()
    layout = bimanual_keypoint_layout(pred_cpu.shape[-1])
    if layout is None:
        return {}
    kp_idx = list(layout["keypoints"])
    wrist_idx = list(layout["wrist_xyz"])
    rot_idx = list(layout["rot"])

    def _kp_l2(p, g):
        diff = p[..., kp_idx].float() - g[..., kp_idx].float()
        return diff.reshape(*diff.shape[:-1], -1, 3).norm(dim=-1).mean()

    metrics = {
        f"{prefix}_kp_l2_avg": _kp_l2(pred_cpu, gt_cpu),
        f"{prefix}_kp_l2_final": _kp_l2(pred_cpu[:, -1], gt_cpu[:, -1]),
        f"{prefix}_wrist_xyz_paired_mse_avg": _paired_mse(
            pred_cpu[..., wrist_idx], gt_cpu[..., wrist_idx]
        ),
        f"{prefix}_wrist_xyz_final_mse_avg": _paired_mse(
            pred_cpu[:, -1, wrist_idx], gt_cpu[:, -1, wrist_idx]
        ),
        f"{prefix}_wrist_rot_err_deg_avg": (
            _geodesic_from_rot_channels(pred_cpu, gt_cpu, rot_idx) * _RAD2DEG
        ),
        f"{prefix}_wrist_rot_err_deg_final": (
            _geodesic_from_rot_channels(pred_cpu[:, -1], gt_cpu[:, -1], rot_idx)
            * _RAD2DEG
        ),
    }
    metrics.update(_segment_l2(pred_cpu, gt_cpu, kp_idx, f"{prefix}_kp_l2"))
    metrics.update(_segment_l2(pred_cpu, gt_cpu, wrist_idx, f"{prefix}_wrist_xyz_l2"))
    pred_kp = pred[..., kp_idx]
    gt_kp = gt[..., kp_idx].to(pred_kp.device)
    metrics[f"{prefix}_kp_dtw_avg"] = dtw_distance(pred_kp, gt_kp).mean().item()
    return metrics


def layout_metrics(pred: torch.Tensor, gt: torch.Tensor, prefix: str) -> dict:
    """Dispatch on the last-dim width: cartesian, keypoint, or ``{}``."""
    metrics = cartesian_metrics(pred, gt, prefix)
    if metrics:
        return metrics
    return keypoint_metrics(pred, gt, prefix)
