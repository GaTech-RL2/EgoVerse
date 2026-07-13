import copy
import math

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.metrics import (
    dtw_distance,
    frechet_gaussian_over_time,
    reverse_kl_from_samples,
)
from egomimic.utils.pose_utils import bimanual_cartesian_layout


def _wrap_aware_mse(pred: torch.Tensor, gt: torch.Tensor):
    """(wrapped, unwrapped) MSE over a bimanual cartesian YPR vector.

    Euler angles wrap at ±π: a prediction of +π-ε against a target of -π+ε is
    physically near-perfect but scores ~(2π)² per dim unwrapped, and a handful
    of wrap events dominates the batch average. Wrap the rotation-dim errors
    to (-π, π] before squaring; positions/grippers are untouched. Falls back
    to plain MSE when the trailing width has no known layout.
    """
    diff = (pred - gt).float()
    nowrap = diff.pow(2).mean()
    layout = bimanual_cartesian_layout(diff.shape[-1])
    # 6D-rotation layouts (18/20) have no angle dims — only wrap YPR widths.
    if layout is None or len(layout["rot"]) != 6:
        return nowrap, nowrap
    rot = list(layout["rot"])
    diff[..., rot] = torch.remainder(diff[..., rot] + math.pi, 2 * math.pi) - math.pi
    return diff.pow(2).mean(), nowrap


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Per embodiment, computes:
      - val loss (flow-matching loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE in the model's native wrist frame
      - paired/final MSE in cam frame, when a ``transform_lists`` entry is configured
    The revert transform is applied once and reused for both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch, do_viz=True):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0

        # Split the bimanual cartesian vector into a translation MSE and a
        # rotation MSE so a translation problem reads apart from a rotation
        # one. Handles all four native widths via ``bimanual_cartesian_layout``:
        #   - native model output: 18D (human) / 20D (robot) continuous 6D cols
        #     — this metric is clean (6D has no ±π wrap).
        #   - reverted cam-frame output: 12D (human) / 14D (robot) ypr — the
        #     rotation split here keeps the ±π caveat (the headline cam MSE is
        #     wrap-corrected; the split is a secondary diagnostic).
        # The metric keys keep the ``_ypr_`` name for dashboard continuity; it
        # denotes "rotation channels" regardless of encoding.
        def _split_mse(pred_t, gt_t):
            layout = bimanual_cartesian_layout(pred_t.shape[-1])
            if layout is None:
                return None, None
            xyz_idx = list(layout["xyz"])
            rot_idx = list(layout["rot"])
            xyz = MeanSquaredError()(
                pred_t[..., xyz_idx].cpu().contiguous(),
                gt_t[..., xyz_idx].cpu().contiguous(),
            )
            rot = MeanSquaredError()(
                pred_t[..., rot_idx].cpu().contiguous(),
                gt_t[..., rot_idx].cpu().contiguous(),
            )
            return xyz, rot

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
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )
                # Last-timestep-only MSE: the end of the chunk is the
                # longest-horizon (hardest) prediction, so this reads as a
                # worst-end signal vs the chunk-wide ``paired`` average.
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu().contiguous(),
                    _batch[ac_key][:, -1].cpu().contiguous(),
                )
                xyz_p, ypr_p = _split_mse(preds[pred_key], _batch[ac_key])
                if xyz_p is not None:
                    metrics[f"Valid/{pred_key}_xyz_paired_mse_avg"] = xyz_p
                    metrics[f"Valid/{pred_key}_ypr_paired_mse_avg"] = ypr_p

                # Distributional metrics (native frame only). Fréchet compares
                # the time-distribution shape of the single prediction; reverse
                # KL needs M independent stochastic samples and is gated on the
                # algo's ``rkl_samples`` (M extra sampling passes per batch).
                fd = frechet_gaussian_over_time(preds[pred_key], _batch[ac_key])
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

                # DTW: trajectory similarity tolerant to temporal misalignment
                # — a correct motion executed early/late scores near zero here
                # while paired MSE penalizes it. Normalized per path step, so
                # units are avg per-frame euclidean distance in native space.
                dtw = dtw_distance(preds[pred_key], _batch[ac_key])
                metrics[f"Valid/{pred_key}_dtw_avg"] = dtw.mean().item()

                if getattr(algo, "rkl_samples", 1) and algo.rkl_samples > 1:
                    M = int(algo.rkl_samples)
                    gt_tensor = _batch[ac_key].to(algo.device)
                    # Feed the ORIGINAL normalized batch element, not the loop's
                    # unnormalized ``_batch`` — ``norm_stats.unnormalize`` also
                    # denormalizes proprio obs keys, so sampling must run on the
                    # normalized obs (same as ``forward_eval``).
                    samples = algo.sample_action_chunks(
                        batch[embodiment_id], embodiment_id, M
                    )
                    rkl = reverse_kl_from_samples(samples, gt_tensor)
                    metrics[f"Valid/{pred_key}_reverse_kl_M{M}"] = rkl.item()

                    # Best-of-K coverage from the SAME M samples (no extra
                    # sampling): per-sample paired MSE to GT, reduced over the
                    # chunk. ``bestof`` = does the policy produce a good action
                    # in M tries (multimodal coverage); ``mean`` = avg sample
                    # quality; ``worstof`` = how bad the worst draw is;
                    # ``diversity`` = mean per-element std across samples.
                    per_sample_mse = (
                        ((samples - gt_tensor.unsqueeze(0)) ** 2)
                        .flatten(start_dim=2)
                        .mean(dim=2)
                    )  # (M, B)
                    metrics[f"Valid/{pred_key}_bestof{M}_paired_mse"] = (
                        per_sample_mse.min(dim=0).values.mean().item()
                    )
                    metrics[f"Valid/{pred_key}_mean{M}_paired_mse"] = (
                        per_sample_mse.mean().item()
                    )
                    metrics[f"Valid/{pred_key}_worstof{M}_paired_mse"] = (
                        per_sample_mse.max(dim=0).values.mean().item()
                    )
                    metrics[f"Valid/{pred_key}_sample_diversity_M{M}"] = (
                        samples.std(dim=0).mean().item()
                    )

            transform_list = self.transform_lists.get(embodiment_name)
            gt_batch_viz = _batch
            preds_for_viz = preds
            if transform_list is not None and pred_key in preds:
                pred_batch = copy.deepcopy(_batch)
                pred_batch[ac_key] = preds[pred_key]
                gt_t = Embodiment.apply_transform(_batch, transform_list)
                pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                # apply_transform drops keys whose shape[0] != batch_size
                # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
                gt_batch_viz = {**_batch, **gt_t}
                pred_batch_viz = {**_batch, **pred_t}

                # Cam-frame vectors are xyz+YPR — wrap angle errors to ±π so
                # boundary predictions don't blow up the MSE. The unwrapped
                # value is kept as ``*_nowrap`` to quantify the inflation.
                paired_w, paired_nw = _wrap_aware_mse(
                    pred_batch_viz[ac_key].cpu(), gt_batch_viz[ac_key].cpu()
                )
                final_w, final_nw = _wrap_aware_mse(
                    pred_batch_viz[ac_key][:, -1].cpu(),
                    gt_batch_viz[ac_key][:, -1].cpu(),
                )
                metrics[f"Valid/{pred_key}_cam_paired_mse_avg"] = paired_w
                metrics[f"Valid/{pred_key}_cam_final_mse_avg"] = final_w
                metrics[f"Valid/{pred_key}_cam_paired_mse_nowrap"] = paired_nw
                metrics[f"Valid/{pred_key}_cam_final_mse_nowrap"] = final_nw
                xyz_cp, ypr_cp = _split_mse(
                    pred_batch_viz[ac_key], gt_batch_viz[ac_key]
                )
                if xyz_cp is not None:
                    metrics[f"Valid/{pred_key}_cam_xyz_paired_mse_avg"] = xyz_cp
                    metrics[f"Valid/{pred_key}_cam_ypr_paired_mse_avg"] = ypr_cp

                preds_for_viz = dict(preds)
                preds_for_viz[pred_key] = pred_batch_viz[ac_key]

            if do_viz:
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
