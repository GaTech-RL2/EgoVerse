import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.core._viz_shared import cam_frame_mse_and_viz_batches
from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


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

            gt_batch_viz, preds_for_viz = cam_frame_mse_and_viz_batches(
                transform_list=self.transform_lists.get(embodiment_name),
                pred_key=pred_key,
                ac_key=ac_key,
                _batch=_batch,
                preds=preds,
                metrics=metrics,
                mse=mse,
            )

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
