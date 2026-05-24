import copy

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Per embodiment, computes:
      - val loss (flow-matching loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE in the model's native wrist frame
      - paired/final MSE in cam frame, when a ``transform_lists`` entry is configured
    The revert transform is applied once and reused for both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0

        # Bimanual 12-D layout from `HumanBimanualCartesianEuler.from32`:
        # [L_xyz(3), L_ypr(3), R_xyz(3), R_ypr(3)]. Split lets us tell a
        # translation problem apart from a rotation-reconstruction artifact
        # (6D-cols → matrix → YPR can blow up near gimbal lock / ±π wrap).
        def _split_mse(pred_t, gt_t):
            if pred_t.shape[-1] != 12:
                return None, None
            xyz_idx = [0, 1, 2, 6, 7, 8]
            ypr_idx = [3, 4, 5, 9, 10, 11]
            xyz = MeanSquaredError()(
                pred_t[..., xyz_idx].cpu().contiguous(),
                gt_t[..., xyz_idx].cpu().contiguous(),
            )
            ypr = MeanSquaredError()(
                pred_t[..., ypr_idx].cpu().contiguous(),
                gt_t[..., ypr_idx].cpu().contiguous(),
            )
            return xyz, ypr

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
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(), _batch[ac_key][:, -1].cpu()
                )
                xyz_p, ypr_p = _split_mse(preds[pred_key], _batch[ac_key])
                if xyz_p is not None:
                    metrics[f"Valid/{pred_key}_xyz_paired_mse_avg"] = xyz_p
                    metrics[f"Valid/{pred_key}_ypr_paired_mse_avg"] = ypr_p
                xyz_f, ypr_f = _split_mse(
                    preds[pred_key][:, -1:], _batch[ac_key][:, -1:]
                )
                if xyz_f is not None:
                    metrics[f"Valid/{pred_key}_xyz_final_mse_avg"] = xyz_f
                    metrics[f"Valid/{pred_key}_ypr_final_mse_avg"] = ypr_f

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

                # ``.contiguous()`` because ``apply_transform`` returns CPU tensors,
                # so ``.cpu()`` here is a no-op and ``[:, -1]`` leaves a non-contiguous
                # view that torchmetrics' MSE doesn't accept.
                metrics[f"Valid/{pred_key}_cam_paired_mse_avg"] = mse(
                    pred_batch_viz[ac_key].cpu().contiguous(),
                    gt_batch_viz[ac_key].cpu().contiguous(),
                )
                metrics[f"Valid/{pred_key}_cam_final_mse_avg"] = mse(
                    pred_batch_viz[ac_key][:, -1].cpu().contiguous(),
                    gt_batch_viz[ac_key][:, -1].cpu().contiguous(),
                )
                xyz_cp, ypr_cp = _split_mse(
                    pred_batch_viz[ac_key], gt_batch_viz[ac_key]
                )
                if xyz_cp is not None:
                    metrics[f"Valid/{pred_key}_cam_xyz_paired_mse_avg"] = xyz_cp
                    metrics[f"Valid/{pred_key}_cam_ypr_paired_mse_avg"] = ypr_cp
                xyz_cf, ypr_cf = _split_mse(
                    pred_batch_viz[ac_key][:, -1:], gt_batch_viz[ac_key][:, -1:]
                )
                if xyz_cf is not None:
                    metrics[f"Valid/{pred_key}_cam_xyz_final_mse_avg"] = xyz_cf
                    metrics[f"Valid/{pred_key}_cam_ypr_final_mse_avg"] = ypr_cf

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
