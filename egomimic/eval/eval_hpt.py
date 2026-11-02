import copy

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.action_metrics import layout_metrics
from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment
from egomimic.utils.metrics import frechet_gaussian_over_time

EVALUATION_TRACKING_POLICY = "tracking_valid_windows_v1"
TRACKING_VALID_KEY = "evaluation_tracking_valid"


def tracking_valid_mask(samples, size=None, required=False):
    flag = samples.get(TRACKING_VALID_KEY)
    if flag is None:
        if required:
            raise ValueError("Exact validation requires evaluation_tracking_valid")
        return None
    flag = torch.as_tensor(flag)
    if flag.ndim != 1 or (size is not None and len(flag) != size):
        raise ValueError("evaluation_tracking_valid must have one value per sample")
    if not ((flag == 0) | (flag == 1)).all():
        raise ValueError("evaluation_tracking_valid must be boolean or binary")
    return flag.bool()


def _select_sample_rows(value, keep, size):
    if isinstance(value, dict):
        return {
            key: _select_sample_rows(item, keep, size) for key, item in value.items()
        }
    if torch.is_tensor(value) and value.ndim and value.shape[0] == size:
        return value[keep.to(value.device)]
    if isinstance(value, (list, tuple)) and len(value) == size:
        selected = [item for item, valid in zip(value, keep.tolist()) if valid]
        return tuple(selected) if isinstance(value, tuple) else selected
    return value


def filter_tracking_batch(batch, required=False):
    """Keep only fully tracked windows before any target-conditioned forward.

    No flag means legacy behavior. A flagged batch must flag every embodiment;
    sample identifiers and values are sliced, never replaced or imputed.
    """
    active = required or any(
        TRACKING_VALID_KEY in samples for samples in batch.values()
    )
    if not active:
        return batch, {}, False
    filtered, counts = {}, {}
    for embodiment, samples in batch.items():
        keep = tracking_valid_mask(samples, required=True)
        counts[embodiment] = int(keep.sum().item())
        if counts[embodiment]:
            filtered[embodiment] = _select_sample_rows(samples, keep, len(keep))
    return filtered, counts, True


def _metric_sample_count(key, samples, action_key, count):
    """Respect existing per-third reductions that omit all-padding rows."""
    pad = samples.get("action_pad_mask")
    if pad is None:
        return count
    valid = torch.as_tensor(pad).bool()
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    horizon = samples[action_key].shape[1]
    for part, third, lo, hi in (
        ("early", 0, 0.0, 0.1),
        ("mid", 1, 0.1, 0.5),
        ("late", 2, 0.5, 1.0),
    ):
        if key.endswith(f"_paired_mse_{part}_third"):
            bounds = (0, horizon // 3, 2 * horizon // 3, horizon)
            start, stop = bounds[third : third + 2]
        elif key.endswith(f"_l2_{part}"):
            start = min(int(round(lo * horizon)), horizon - 1)
            stop = max(int(round(hi * horizon)), start + 1)
        else:
            continue
        return int(valid[:, start:stop].any(dim=1).sum().item())
    return count


class HPTEvalVideo(EvalVideo):
    """
    Eval class for HPT models. Per embodiment, computes:
      - val loss (BC loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE + Frechet over time for the main / shared / auxiliary heads
      - the layout metrics of :mod:`egomimic.eval.action_metrics` on the main
        ``ac_key`` (position / rotation / keypoint errors, frame-invariant)
      - paired/final MSE in cam frame on the main ``ac_key``, when a
        ``transform_lists`` entry is configured (viz batches only)
    The revert transform is applied once and reused by both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch, do_viz=True):
        for embodiment_id, samples in batch.items():
            if TRACKING_VALID_KEY in samples:
                tracking_valid_mask(
                    samples,
                    size=samples[self.model.ac_keys[embodiment_id]].shape[0],
                    required=True,
                )
        batch, scored_counts, tracking_active = filter_tracking_batch(batch)
        self.metric_batch_sizes = {} if tracking_active else None
        if not batch:
            return {}, {}
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

            loss_key = f"{embodiment_name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                weight = scored_counts[embodiment_id] if tracking_active else 1
                total_loss = total_loss + loss_val * weight
                n_loss_embodiments += weight

            if f"{embodiment_name}_{ac_key}" in preds and ac_key != algo.shared_ac_key:
                pred_key = f"{embodiment_name}_{ac_key}"
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu().contiguous(),
                    _batch[ac_key][:, -1].cpu().contiguous(),
                )
                fd = frechet_gaussian_over_time(preds[pred_key], _batch[ac_key])
                metrics[f"Valid/{pred_key}_frechet_gauss_avg"] = fd.mean().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_min"] = fd.min().item()
                metrics[f"Valid/{pred_key}_frechet_gauss_max"] = fd.max().item()
                metrics.update(
                    layout_metrics(
                        preds[pred_key],
                        _batch[ac_key],
                        f"Valid/{pred_key}",
                        mask=_batch.get("action_pad_mask"),
                    )
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

                # ``.contiguous()`` because ``apply_transform`` returns CPU tensors,
                # so ``.cpu()`` here is a no-op and ``[:, -1]`` leaves a non-contiguous
                # view that torchmetrics' MSE doesn't accept.
                metrics[f"Valid/{main_pred_key}_cam_paired_mse_avg"] = mse(
                    pred_batch_viz[ac_key].cpu().contiguous(),
                    gt_batch_viz[ac_key].cpu().contiguous(),
                )
                metrics[f"Valid/{main_pred_key}_cam_final_mse_avg"] = mse(
                    pred_batch_viz[ac_key][:, -1].cpu().contiguous(),
                    gt_batch_viz[ac_key][:, -1].cpu().contiguous(),
                )

                preds_for_viz = dict(preds)
                preds_for_viz[main_pred_key] = pred_batch_viz[ac_key]

            ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
            images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        if tracking_active:
            for embodiment_id, count in scored_counts.items():
                prefix = f"Valid/{get_embodiment(embodiment_id).lower()}_"
                self.metric_batch_sizes.update(
                    {
                        key: _metric_sample_count(
                            key,
                            batch[embodiment_id],
                            algo.ac_keys[embodiment_id],
                            count,
                        )
                        for key in metrics
                        if key.startswith(prefix)
                    }
                )
            if "Valid/action_loss" in metrics:
                self.metric_batch_sizes["Valid/action_loss"] = n_loss_embodiments
            if set(self.metric_batch_sizes) != set(metrics):
                raise RuntimeError(
                    "Tracking-filtered metrics have missing sample weights"
                )
        return metrics, images_dict
