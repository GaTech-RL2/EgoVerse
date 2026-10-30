import copy

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.action_metrics import (
    fill_untracked_from_pred,
    layout_metrics,
    per_tracked_hand,
    tracked_hand_fraction,
)
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.metrics import frechet_gaussian_over_time


class HPTEvalVideo(EvalVideo):
    """
    Eval class for HPT models. Per embodiment, computes:
      - val loss (BC loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE + Frechet over time for the main / shared / auxiliary heads
      - the layout metrics of :mod:`egomimic.eval.action_metrics` on the main
        ``ac_key`` (position / rotation / keypoint errors, frame-invariant)
    Metrics are frame-invariant, so the ``transform_lists`` revert runs only
    when rendering the viz video (``do_viz``), never for metrics.
    """

    def compute_metrics_and_viz(self, batch, do_viz=True):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            _batch = fill_untracked_from_pred(
                _batch, ac_key, preds.get(f"{embodiment_name}_{ac_key}")
            )

            loss_key = f"{embodiment_name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                total_loss = total_loss + loss_val
                n_loss_embodiments += 1

            if f"{embodiment_name}_{ac_key}" in preds and ac_key != algo.shared_ac_key:
                pred_key = f"{embodiment_name}_{ac_key}"
                main = {
                    f"Valid/{pred_key}_paired_mse_avg": mse(
                        preds[pred_key].cpu(), _batch[ac_key].cpu()
                    ),
                    f"Valid/{pred_key}_final_mse_avg": mse(
                        preds[pred_key][:, -1].cpu().contiguous(),
                        _batch[ac_key][:, -1].cpu().contiguous(),
                    ),
                }
                fd = frechet_gaussian_over_time(preds[pred_key], _batch[ac_key])
                main[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                main[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                main[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()
                main.update(
                    layout_metrics(
                        preds[pred_key], _batch[ac_key], f"Valid/{pred_key}", ac_key
                    )
                )
                metrics.update(
                    per_tracked_hand(main, tracked_hand_fraction(_batch, ac_key))
                )

            if embodiment_name in algo.auxiliary_ac_keys:
                for aux_key in algo.auxiliary_ac_keys[embodiment_name]:
                    pred_key = f"{embodiment_name}_{aux_key}"
                    if pred_key in preds:
                        metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                            preds[pred_key].cpu(), _batch[aux_key].cpu()
                        )
                        metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                            preds[pred_key][:, -1].cpu().contiguous(),
                            _batch[aux_key][:, -1].cpu().contiguous(),
                        )
                        fd = frechet_gaussian_over_time(
                            preds[pred_key], _batch[aux_key]
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = (
                            fd.mean().item()
                        )
                        metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                        metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if (
                algo.shared_ac_key
                and f"{embodiment_name}_{algo.shared_ac_key}" in preds
            ):
                pred_key = f"{embodiment_name}_{algo.shared_ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[algo.shared_ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu().contiguous(),
                    _batch[algo.shared_ac_key][:, -1].cpu().contiguous(),
                )
                fd = frechet_gaussian_over_time(
                    preds[pred_key], _batch[algo.shared_ac_key]
                )
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()

            if not do_viz:
                continue

            transform_list = self.transform_lists.get(embodiment_name)
            main_pred_key = f"{embodiment_name}_{ac_key}"
            gt_batch_viz = _batch
            preds_for_viz = preds
            if transform_list is not None and main_pred_key in preds:
                pred_batch = copy.deepcopy(_batch)
                pred_batch[ac_key] = preds[main_pred_key]
                gt_t = Embodiment.apply_transform(_batch, transform_list)
                pred_t = Embodiment.apply_transform(pred_batch, transform_list)
                # apply_transform drops keys whose shape[0] != batch_size
                # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
                gt_batch_viz = {**_batch, **gt_t}
                pred_batch_viz = {**_batch, **pred_t}
                preds_for_viz = dict(preds)
                preds_for_viz[main_pred_key] = pred_batch_viz[ac_key]

            ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
            images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        return metrics, images_dict
