import copy
import math

import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.action_utils import _reconstruct_R_from_cols, _ypr_to_matrix
from egomimic.utils.metrics import dtw_distance, frechet_gaussian_over_time
from egomimic.utils.pose_utils import bimanual_cartesian_layout

_RAD2DEG = 180.0 / math.pi


def _paired_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Plain elementwise MSE, stateless (no torchmetrics accumulator)."""
    return (pred.float() - gt.float()).pow(2).mean()


def _split_mse(pred_t: torch.Tensor, gt_t: torch.Tensor):
    """(translation MSE, rotation-channel MSE) over a bimanual cartesian
    vector, so a translation problem reads apart from a rotation one. The
    rotation channels are whatever the layout holds: the continuous 6D
    rotation-matrix columns for the native 18D (human) / 20D (robot) widths,
    yaw/pitch/roll for the 12/14D widths. Either way this is a plain
    per-channel MSE (pre-Gram-Schmidt, axis-dependent, unitless for 6D; ±π
    wrap and gimbal lock for ypr), i.e. a training-tracking diagnostic, not a
    rotation error — see ``_rot_geodesic_error`` for that. Returns
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


def _rot_geodesic_error(pred: torch.Tensor, gt: torch.Tensor):
    """Mean geodesic rotation error in radians over batch / time / both arms.

    Euler-free: per arm, builds proper rotation matrices (``_ypr_to_matrix``
    for the 12/14-dim ypr widths; Gram-Schmidt on the two 6D columns for the
    18/20-dim widths — the same reconstruction the model decode uses) and
    takes ``arccos((tr(R_pred^T R_gt) - 1) / 2)``. Unlike a ypr MSE this is
    immune to the ±π wrap AND to the yaw/roll degeneracy at pitch ≈ ±π/2,
    where two nearly identical orientations can differ by ~π in both yaw and
    roll. Returns None for an unknown width.
    """
    layout = bimanual_cartesian_layout(pred.shape[-1])
    if layout is None:
        return None
    rot = list(layout["rot"])
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


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Per embodiment, on the model's native
    (unnormalized xyz+6D) output, computes:
      - ``action_loss``: flow-matching val loss, same as training (also logged
        per embodiment as ``<name>_loss``)
      - ``xyz_paired_mse_avg`` / ``xyz_final_mse_avg``: position MSE (m²) over
        the chunk / at the last timestep (longest-horizon, hardest prediction)
      - ``rot_err_deg_avg`` / ``rot_err_deg_final``: geodesic rotation error
        in degrees between the Gram-Schmidt-decoded prediction and gt
      - ``rot6d_paired_mse_avg``: MSE over the raw 6D rotation columns (the
        channels the loss regresses, up to per-dim normalization); a
        training-tracking diagnostic, not a rotation error
      - ``xyz_dtw_avg`` / ``xyz_frechet_gauss_avg``: temporal-misalignment-
        tolerant DTW and time-distribution Fréchet distance on xyz only
      - ``bestof{M}_paired_mse`` / ``sample_diversity_M{M}``: multimodal
        coverage and spread from M stochastic draws, only when the algo's
        ``val_samples`` > 1

    Metrics are frame-invariant (the cam-frame revert is a rigid transform
    shared by prediction and gt), so the ``transform_lists`` revert runs only
    when rendering the viz video (``do_viz``), never for metrics.
    """

    def compute_metrics_and_viz(self, batch, do_viz=True):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        total_loss = None
        n_loss_embodiments = 0

        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            loss_key = f"{embodiment_name}_loss"

            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                total_loss = total_loss + loss_val
                n_loss_embodiments += 1

            if pred_key in preds:
                pred_cpu = preds[pred_key].cpu()
                gt_cpu = _batch[ac_key].cpu()
                layout = bimanual_cartesian_layout(pred_cpu.shape[-1])

                xyz_p, rot_p = _split_mse(pred_cpu, gt_cpu)
                if xyz_p is not None:
                    metrics[f"Valid/{pred_key}_xyz_paired_mse_avg"] = xyz_p
                    metrics[f"Valid/{pred_key}_rot6d_paired_mse_avg"] = rot_p
                    xyz_idx = list(layout["xyz"])
                    metrics[f"Valid/{pred_key}_xyz_final_mse_avg"] = _paired_mse(
                        pred_cpu[:, -1, xyz_idx], gt_cpu[:, -1, xyz_idx]
                    )
                    # Distributional / alignment metrics on the position
                    # channels only (metres), so a unitless 6D column never
                    # trades off against a metre. DTW forgives temporal
                    # misalignment (a correct motion executed early/late scores
                    # near zero where paired MSE penalizes it); Fréchet
                    # compares the time-distribution shape of the motion.
                    pred_xyz = preds[pred_key][..., xyz_idx]
                    gt_xyz = _batch[ac_key][..., xyz_idx].to(pred_xyz.device)
                    metrics[f"Valid/{pred_key}_xyz_dtw_avg"] = (
                        dtw_distance(pred_xyz, gt_xyz).mean().item()
                    )
                    metrics[f"Valid/{pred_key}_xyz_frechet_gauss_avg"] = (
                        frechet_gaussian_over_time(pred_xyz, gt_xyz).mean().item()
                    )

                # Rotation error on the manifold, in degrees: the rotation the
                # decode actually produces after Gram-Schmidt, independent of
                # the 6D/ypr format and free of wrap / gimbal artefacts.
                geo = _rot_geodesic_error(pred_cpu, gt_cpu)
                if geo is not None:
                    metrics[f"Valid/{pred_key}_rot_err_deg_avg"] = geo * _RAD2DEG
                    metrics[f"Valid/{pred_key}_rot_err_deg_final"] = (
                        _rot_geodesic_error(pred_cpu[:, -1], gt_cpu[:, -1]) * _RAD2DEG
                    )

                M = int(getattr(algo, "val_samples", 1) or 1)
                if M > 1:
                    # Feed the ORIGINAL normalized batch element, not the loop's
                    # unnormalized ``_batch`` — ``norm_stats.unnormalize`` also
                    # denormalizes proprio obs keys, so sampling must run on the
                    # normalized obs (same as ``forward_eval``).
                    samples = algo.sample_action_chunks(
                        batch[embodiment_id], embodiment_id, M
                    )  # (M, B, T, D)
                    gt_tensor = _batch[ac_key].to(samples.device)
                    # ``bestof`` = does the policy produce a good chunk in M
                    # tries (multimodal coverage); ``diversity`` = mean
                    # per-element std across the M draws (spread).
                    per_sample_mse = (
                        ((samples - gt_tensor.unsqueeze(0)) ** 2)
                        .flatten(start_dim=2)
                        .mean(dim=2)
                    )  # (M, B)
                    metrics[f"Valid/{pred_key}_bestof{M}_paired_mse"] = (
                        per_sample_mse.min(dim=0).values.mean().item()
                    )
                    metrics[f"Valid/{pred_key}_sample_diversity_M{M}"] = (
                        samples.std(dim=0).mean().item()
                    )

            if do_viz:
                gt_batch_viz = _batch
                preds_for_viz = preds
                transform_list = self.transform_lists.get(embodiment_name)
                if transform_list is not None and pred_key in preds:
                    # Revert to cam (head) frame + xyz+ypr layout for the
                    # overlay only; the metrics above are frame-invariant.
                    pred_batch = copy.deepcopy(_batch)
                    pred_batch[ac_key] = preds[pred_key]
                    gt_t = Embodiment.apply_transform(_batch, transform_list)
                    pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                    # apply_transform drops keys whose shape[0] != batch_size
                    # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
                    gt_batch_viz = {**_batch, **gt_t}
                    pred_batch_viz = {**_batch, **pred_t}
                    preds_for_viz = dict(preds)
                    preds_for_viz[pred_key] = pred_batch_viz[ac_key]
                ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
                images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return self.viz_func[embodiment_name](predictions, batch)
