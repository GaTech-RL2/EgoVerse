"""MSE over randomly sampled trajectory starts, independent of video pins."""

import torch

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.utils.pose_utils import bimanual_cartesian_layout


def trajectory_mse(pred, target, mask=None):
    """Return one MSE per trajectory, excluding repeat-last padding."""
    error = (pred.float() - target.float()).square()
    if mask is None:
        return error.mean(dim=(1, 2))
    valid = mask.to(device=error.device, dtype=torch.bool)
    if valid.ndim == 3:
        valid = valid.squeeze(-1)
    if valid.shape != error.shape[:2] or not valid.any(dim=1).all():
        raise ValueError("Each sampled trajectory needs at least one valid step")
    error = error.masked_fill(~valid.unsqueeze(-1), 0)
    return error.sum(dim=(1, 2)) / (valid.sum(dim=1) * error.shape[-1])


def trajectory_mse_thirds(pred, target, mask=None):
    """Per-trajectory MSE for the three contiguous thirds of a full chunk.

    Padding is excluded. A trajectory with no real steps in a third does not
    vote in that third's mean; all-empty thirds are omitted.
    """
    horizon = pred.shape[1]
    boundaries = (0, horizon // 3, 2 * horizon // 3, horizon)
    parts = {}
    for label, start, stop in zip(
        ("early", "mid", "late"), boundaries[:-1], boundaries[1:]
    ):
        if start == stop:
            continue
        section_mask = None if mask is None else mask[:, start:stop]
        if section_mask is not None:
            if section_mask.ndim == 3:
                section_mask = section_mask.squeeze(-1)
            rows = section_mask.to(dtype=torch.bool).any(dim=1)
            if not rows.any():
                continue
            section_mask = section_mask[rows]
            section_pred = pred[rows, start:stop]
            section_target = target[rows, start:stop]
        else:
            section_pred = pred[:, start:stop]
            section_target = target[:, start:stop]
        parts[label] = trajectory_mse(section_pred, section_target, section_mask).mean()
    return parts


class SampledTrajectoryMSE(EvalVideo):
    """Unnormalized full-chunk MSE, averaged over sampled trajectory starts."""

    def compute_metrics_and_viz(self, batch, do_viz=False):
        predictions = self.model.forward_eval(batch)
        metrics = {}

        def record(key, pred, target, mask):
            metrics[key] = trajectory_mse(pred, target, mask).mean()
            for label, value in trajectory_mse_thirds(pred, target, mask).items():
                metrics[f"{key}/{label}"] = value

        for embodiment_id, sample in batch.items():
            sample = self.model.norm_stats.unnormalize(sample, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            key = self.model.ac_keys[embodiment_id]
            pred_key = f"{name}_{key}"
            pred, target = predictions[pred_key], sample[key]
            mask = sample.get("pad_mask")
            record("Valid/Action_MSE", pred, target, mask)
            record(f"Valid/{pred_key}_paired_mse_avg", pred, target, mask)
            layout = bimanual_cartesian_layout(pred.shape[-1])
            if layout is not None:
                for label, columns in (
                    ("XYZ_MSE", layout["xyz"]),
                    ("Rotation_MSE", layout["rot"]),
                ):
                    record(
                        f"Valid/{label}",
                        pred[..., list(columns)],
                        target[..., list(columns)],
                        mask,
                    )
            for aux in self.model.auxiliary_ac_keys.get(name, []):
                aux_pred = f"{name}_{aux}"
                if aux_pred in predictions:
                    record(
                        f"Valid/{aux_pred}_paired_mse_avg",
                        predictions[aux_pred],
                        sample[aux],
                        mask,
                    )
        return metrics, {}
