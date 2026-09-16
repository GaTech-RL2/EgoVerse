import copy

import torch

from egomimic.eval.action_metrics import (  # noqa: F401  (re-exported for tests)
    _paired_mse,
    _rot_geodesic_error,
    _split_mse,
    layout_metrics,
)
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Per embodiment, on the model's native
    (unnormalized) output, computes:
      - ``action_loss``: flow-matching val loss, same as training (also logged
        per embodiment as ``<name>_loss``)
      - the layout metrics of :mod:`egomimic.eval.action_metrics`:
        cartesian widths get ``xyz_paired_mse_avg`` / ``xyz_final_mse_avg``
        (m^2), ``rot_err_deg_avg`` / ``rot_err_deg_final`` (geodesic, after
        Gram-Schmidt), ``rot6d_paired_mse_avg`` (raw rotation channels, a
        training-tracking diagnostic), ``xyz_dtw_avg`` and
        ``xyz_frechet_gauss_avg``; keypoint widths get ``kp_l2_avg`` /
        ``kp_l2_final`` (m), ``wrist_xyz_*_mse``, ``wrist_rot_err_deg_*`` and
        ``kp_dtw_avg``.

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
                metrics.update(
                    layout_metrics(preds[pred_key], _batch[ac_key], f"Valid/{pred_key}")
                )

            if do_viz:
                gt_batch_viz = _batch
                preds_for_viz = preds
                transform_list = self.transform_lists.get(embodiment_name)
                if transform_list is not None and pred_key in preds:
                    # Revert to cam (head) frame + the ypr layout for the
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
