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

``mask`` is the dataset's ``action_pad_mask`` (``(B, T)`` or ``(B, T, 1)``,
1 = real step). Past an episode's end the chunk repeats its last frame, which
a model predicts trivially; with a mask those steps are left out. Means are
per trajectory first (each sampled start counts once), then over the batch;
``*_final`` reads each trajectory's last real step. ``*_paired_mse_*_third``
splits the chunk into three contiguous thirds; a trajectory with no real step
in a third does not vote there, and a third no trajectory reaches is omitted.
DTW and Frechet still see the whole chunk.
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


def _valid_steps(mask, shape) -> torch.Tensor | None:
    """``mask`` as a CPU ``(B, T)`` bool tensor, checked against ``shape``."""
    if mask is None:
        return None
    valid = torch.as_tensor(mask).detach().cpu().to(torch.bool)
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    if valid.shape != tuple(shape[:2]):
        raise ValueError(
            f"pad mask shape {tuple(valid.shape)} does not match the action "
            f"chunk {tuple(shape[:2])}"
        )
    if not valid.any(dim=1).all():
        raise ValueError("every trajectory needs at least one real step")
    return valid


def _trajectory_mean(values: torch.Tensor, valid) -> torch.Tensor:
    """Mean of per-step ``values`` ``(B, T)`` over each trajectory's real
    steps, then over trajectories."""
    if valid is None:
        return values.mean()
    values = values.masked_fill(~valid, 0)
    return (values.sum(dim=1) / valid.sum(dim=1)).mean()


def _step_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` squared error, averaged over channels."""
    return (pred.float() - gt.float()).pow(2).mean(dim=-1)


def _paired_mse(pred: torch.Tensor, gt: torch.Tensor, valid=None) -> torch.Tensor:
    """Plain elementwise MSE, stateless (no torchmetrics accumulator)."""
    if valid is None:
        return (pred.float() - gt.float()).pow(2).mean()
    return _trajectory_mean(_step_mse(pred, gt), valid)


def _last_real(x: torch.Tensor, valid) -> torch.Tensor:
    """``x[:, -1]``, or each trajectory's last real step under a mask."""
    if valid is None:
        return x[:, -1]
    T = valid.shape[1]
    last = T - 1 - valid.flip(1).to(torch.int64).argmax(dim=1)
    return x[torch.arange(x.shape[0]), last]


def _thirds(per_step: torch.Tensor, valid, prefix: str) -> dict:
    """Trajectory-mean of ``per_step`` ``(B, T)`` over each contiguous third
    of the chunk, keyed ``{prefix}_{early,mid,late}_third``."""
    T = per_step.shape[1]
    bounds = (0, T // 3, 2 * T // 3, T)
    out = {}
    for name, start, stop in zip(("early", "mid", "late"), bounds[:-1], bounds[1:]):
        if start == stop:
            continue
        part = per_step[:, start:stop]
        part_valid = None if valid is None else valid[:, start:stop]
        if part_valid is not None:
            rows = part_valid.any(dim=1)
            if not rows.any():
                continue
            part, part_valid = part[rows], part_valid[rows]
        out[f"{prefix}_{name}_third"] = _trajectory_mean(part, part_valid)
    return out


# Horizon segments for the per-segment L2 metrics, as fractions of the chunk:
# early = the first 10 % of timesteps, mid = 10-50 %, late = the second half.
# They separate near-term precision from long-range drift, which one
# chunk-mean error hides (and which per-timestep vs horizon-pooled norm stats
# trade against each other).
_HORIZON_SEGMENTS = (("early", 0.0, 0.1), ("mid", 0.1, 0.5), ("late", 0.5, 1.0))


def _segment_l2(
    pred: torch.Tensor, gt: torch.Tensor, idx: list, prefix: str, valid=None
) -> dict:
    """Mean euclidean error (m) of the xyz triples at ``idx`` per horizon
    segment of a ``(B, T, D)`` chunk."""
    T = pred.shape[1]
    diff = pred[..., idx].float() - gt[..., idx].float()
    l2 = diff.reshape(*diff.shape[:-1], -1, 3).norm(dim=-1).mean(dim=-1)  # (B, T)
    out = {}
    for name, lo, hi in _HORIZON_SEGMENTS:
        start = min(int(round(lo * T)), T - 1)
        end = max(int(round(hi * T)), start + 1)
        part = l2[:, start:end]
        part_valid = None if valid is None else valid[:, start:end]
        if part_valid is not None:
            rows = part_valid.any(dim=1)
            if not rows.any():
                continue
            part, part_valid = part[rows], part_valid[rows]
        out[f"{prefix}_{name}"] = _trajectory_mean(part, part_valid)
    return out


def _split_mse(pred_t: torch.Tensor, gt_t: torch.Tensor, valid=None):
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
    xyz = _paired_mse(pred_t[..., xyz_idx], gt_t[..., xyz_idx], valid)
    rot = _paired_mse(pred_t[..., rot_idx], gt_t[..., rot_idx], valid)
    return xyz, rot


def _geodesic_per_step(pred: torch.Tensor, gt: torch.Tensor, rot: list):
    """Geodesic angle (rad), averaged over both arms, at every leading index
    of a per-arm-concatenated vector (3 ypr or 6 rot6d per arm)."""
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
    return torch.stack(errs, dim=-1).mean(dim=-1).float()


def _geodesic_from_rot_channels(
    pred: torch.Tensor, gt: torch.Tensor, rot: list, valid=None
):
    """Mean geodesic angle (rad) over both arms given the rotation channel
    indices of a per-arm-concatenated vector (3 ypr or 6 rot6d per arm)."""
    err = _geodesic_per_step(pred, gt, rot)
    if valid is None or err.ndim != 2:
        return err.mean()
    return _trajectory_mean(err, valid)


def _rot_geodesic_error(pred: torch.Tensor, gt: torch.Tensor, valid=None):
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
    return _geodesic_from_rot_channels(pred, gt, list(layout["rot"]), valid)


def cartesian_metrics(
    pred: torch.Tensor, gt: torch.Tensor, prefix: str, mask=None
) -> dict:
    """Metrics for a bimanual cartesian chunk ``(B, T, D)``; ``{}`` if ``D``
    is not a cartesian width. ``pred``/``gt`` may live on any device."""
    pred_cpu, gt_cpu = pred.detach().cpu(), gt.detach().cpu()
    layout = bimanual_cartesian_layout(pred_cpu.shape[-1])
    if layout is None:
        return {}
    valid = _valid_steps(mask, pred_cpu.shape)
    metrics = {}
    xyz_idx = list(layout["xyz"])
    rot_idx = list(layout["rot"])
    xyz_p, rot_p = _split_mse(pred_cpu, gt_cpu, valid)
    metrics[f"{prefix}_xyz_paired_mse_avg"] = xyz_p
    metrics[f"{prefix}_rot6d_paired_mse_avg"] = rot_p
    for name, idx in (("xyz", xyz_idx), ("rot6d", rot_idx)):
        per_step = _step_mse(pred_cpu[..., idx], gt_cpu[..., idx])
        metrics.update(_thirds(per_step, valid, f"{prefix}_{name}_paired_mse"))
    metrics[f"{prefix}_xyz_final_mse_avg"] = _paired_mse(
        _last_real(pred_cpu, valid)[:, xyz_idx], _last_real(gt_cpu, valid)[:, xyz_idx]
    )
    metrics.update(_segment_l2(pred_cpu, gt_cpu, xyz_idx, f"{prefix}_xyz_l2", valid))
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
        _rot_geodesic_error(pred_cpu, gt_cpu, valid) * _RAD2DEG
    )
    metrics[f"{prefix}_rot_err_deg_final"] = (
        _rot_geodesic_error(_last_real(pred_cpu, valid), _last_real(gt_cpu, valid))
        * _RAD2DEG
    )
    return metrics


def keypoint_metrics(
    pred: torch.Tensor, gt: torch.Tensor, prefix: str, mask=None
) -> dict:
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
    valid = _valid_steps(mask, pred_cpu.shape)
    kp_idx = list(layout["keypoints"])
    wrist_idx = list(layout["wrist_xyz"])
    rot_idx = list(layout["rot"])

    def _kp_l2(p, g, v=None):
        diff = p[..., kp_idx].float() - g[..., kp_idx].float()
        l2 = diff.reshape(*diff.shape[:-1], -1, 3).norm(dim=-1).mean(dim=-1)
        return l2.mean() if v is None else _trajectory_mean(l2, v)

    pred_last, gt_last = _last_real(pred_cpu, valid), _last_real(gt_cpu, valid)
    metrics = {
        f"{prefix}_kp_l2_avg": _kp_l2(pred_cpu, gt_cpu, valid),
        f"{prefix}_kp_l2_final": _kp_l2(pred_last, gt_last),
        f"{prefix}_wrist_xyz_paired_mse_avg": _paired_mse(
            pred_cpu[..., wrist_idx], gt_cpu[..., wrist_idx], valid
        ),
        f"{prefix}_wrist_xyz_final_mse_avg": _paired_mse(
            pred_last[:, wrist_idx], gt_last[:, wrist_idx]
        ),
        f"{prefix}_wrist_rot_err_deg_avg": (
            _geodesic_from_rot_channels(pred_cpu, gt_cpu, rot_idx, valid) * _RAD2DEG
        ),
        f"{prefix}_wrist_rot_err_deg_final": (
            _geodesic_from_rot_channels(pred_last, gt_last, rot_idx) * _RAD2DEG
        ),
    }
    metrics.update(
        _thirds(
            _step_mse(pred_cpu[..., wrist_idx], gt_cpu[..., wrist_idx]),
            valid,
            f"{prefix}_wrist_xyz_paired_mse",
        )
    )
    metrics.update(_segment_l2(pred_cpu, gt_cpu, kp_idx, f"{prefix}_kp_l2", valid))
    metrics.update(
        _segment_l2(pred_cpu, gt_cpu, wrist_idx, f"{prefix}_wrist_xyz_l2", valid)
    )
    pred_kp = pred[..., kp_idx]
    gt_kp = gt[..., kp_idx].to(pred_kp.device)
    metrics[f"{prefix}_kp_dtw_avg"] = dtw_distance(pred_kp, gt_kp).mean().item()
    return metrics


def layout_metrics(
    pred: torch.Tensor, gt: torch.Tensor, prefix: str, mask=None
) -> dict:
    """Dispatch on the last-dim width: cartesian, keypoint, or ``{}``."""
    metrics = cartesian_metrics(pred, gt, prefix, mask)
    if metrics:
        return metrics
    return keypoint_metrics(pred, gt, prefix, mask)
