from egomimic.eval.action_metrics import _paired_mse
from egomimic.eval.eval_video import EvalVideo
from egomimic.utils.metrics import frechet_gaussian_over_time


class HPTEvalVideo(EvalVideo):
    """
    Eval class for HPT models. Per embodiment, EvalVideo's val loss and layout
    metrics (main ``ac_key`` only), plus:
      - paired/final MSE + Frechet over time for the main / auxiliary / shared
        heads
      - paired/final MSE in cam frame on the main ``ac_key``, when a
        ``transform_lists`` entry is configured (viz batches only)
    """

    def _head_metrics(self, preds, batch, name, ac_key) -> dict:
        algo = self.model
        shared = algo.shared_ac_key
        keys = [ac_key] if ac_key != shared else []
        keys += list(algo.auxiliary_ac_keys.get(name, ())) + (
            [shared] if shared else []
        )
        metrics = {}
        for key in keys:
            pred_key = f"{name}_{key}"
            if pred_key not in preds:
                continue
            pred, gt = preds[pred_key], batch[key]
            fd = frechet_gaussian_over_time(pred, gt)
            metrics[f"Valid/{pred_key}_paired_mse_avg"] = _paired_mse(
                pred.cpu(), gt.cpu()
            )
            metrics[f"Valid/{pred_key}_final_mse_avg"] = _paired_mse(
                pred[:, -1].cpu(), gt[:, -1].cpu()
            )
            metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
            metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
            metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()
        if ac_key != shared:
            metrics.update(super()._head_metrics(preds, batch, name, ac_key))
        return metrics

    def _cam_metrics(self, pred, gt, pred_key) -> dict:
        return {
            f"Valid/{pred_key}_cam_paired_mse_avg": _paired_mse(pred.cpu(), gt.cpu()),
            f"Valid/{pred_key}_cam_final_mse_avg": _paired_mse(
                pred[:, -1].cpu(), gt[:, -1].cpu()
            ),
        }
