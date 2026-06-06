"""
Validation evaluator for the stage-based H-Net policy.

Drops the HPT-specific machinery (auxiliary heads, shared action key,
reverse-KL, transform-list cam-frame projection) and supports packed
validation batches: the algo's ``forward_eval`` runs an autoregressive
rollout per episode in the pack and returns ``(B, T_max, action_dim)``
padded predictions plus the per-episode ``seq_lens``. We compute MSE only on
the valid positions of each episode (no padding contribution), then hand a
single per-episode frame to the configured ``viz_func`` for the validation
video.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch

from egomimic.eval.core.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _masked_mse(
    pred: torch.Tensor, gt: torch.Tensor, seq_lens: torch.Tensor
) -> torch.Tensor:
    """Mean squared error averaged only over valid (non-padded) positions.

    ``pred`` / ``gt``: ``(B, T_max, D)``. ``seq_lens``: ``(B,)`` long, gives
    the number of valid time-steps per row.
    """
    B, T_max, _ = pred.shape
    # (B, T_max) bool mask of valid positions.
    arange = torch.arange(T_max, device=pred.device)[None, :]
    mask = arange < seq_lens.to(pred.device)[:, None]  # (B, T_max)
    err = ((pred - gt) ** 2).mean(dim=-1)  # (B, T_max)
    valid = mask.float()
    return (err * valid).sum() / valid.sum().clamp(min=1)


def _unpack_to_padded(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """``(T_total, ...) -> (B, T_max, ...)`` zero-padded past each ep's length."""
    B = int(seq_lens.shape[0])
    T_max = int(seq_lens.max().item())
    tail = packed.shape[1:]
    out = torch.zeros((B, T_max) + tail, device=packed.device, dtype=packed.dtype)
    for b in range(B):
        s = int(cu_seqlens[b].item())
        e = int(cu_seqlens[b + 1].item())
        out[b, : e - s] = packed[s:e]
    return out


class HNetEvalVideo(EvalVideo):
    """H-Net validation eval. Computes paired-MSE on packed (or padded)
    validation batches and renders per-episode trajectory frames.
    """

    def compute_metrics_and_viz(
        self, batch: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, np.ndarray]]:
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics: dict = {}
        images_dict: dict = {}
        total_loss = None
        n_loss = 0

        for emb_id, _batch in batch.items():
            embodiment_name = get_embodiment(emb_id).lower()
            ac_key = algo.ac_keys[embodiment_name]
            is_packed = _batch.get("_packed", False)

            # Always unnormalize GT to raw scale; the algo's forward_eval
            # already unnormalizes preds.
            _batch_unnorm = algo.norm_stats.unnormalize(_batch, emb_id)

            pred_key = f"emb{emb_id}_{ac_key}"
            pred = preds.get(pred_key)
            if pred is None:
                continue

            if is_packed:
                cu = _batch["cu_seqlens"]
                seq_lens = preds.get(f"emb{emb_id}_seq_lens", _batch["seq_lens"])
                gt = _unpack_to_padded(_batch_unnorm[ac_key], cu, seq_lens)
                mse_val = _masked_mse(pred, gt, seq_lens)
                metrics[f"Valid/emb{emb_id}_{ac_key}_paired_mse"] = mse_val.detach()
                # Final-position MSE = per-episode last valid index.
                last_idx = (seq_lens - 1).clamp(min=0)
                B = pred.shape[0]
                idx = last_idx.to(pred.device)
                final_pred = pred[torch.arange(B, device=pred.device), idx]
                final_gt = gt[torch.arange(B, device=pred.device), idx]
                metrics[f"Valid/emb{emb_id}_{ac_key}_final_mse"] = (
                    ((final_pred - final_gt) ** 2).mean().detach()
                )

                # Build the viz batch: pass the FULL per-episode video
                # (B, T_max, C, H, W) instead of just frame 0. The
                # viz_func (pushshapes.viz_gt_preds) then renders one
                # output frame per real timestep — saved video shows
                # each episode playing back with the GT/pred trajectory
                # overlay so you can verify alignment over time.
                per_frame_img = None
                cam_keys = algo.camera_keys.get(emb_id, [])
                if cam_keys:
                    img_key = cam_keys[0]
                    if img_key in _batch_unnorm:
                        packed_img = _batch_unnorm[img_key]
                        per_frame_img = _unpack_to_padded(packed_img, cu, seq_lens)

                viz_batch = {
                    ac_key: gt,
                    "embodiment": _batch["embodiment"],
                    "seq_lens": seq_lens,
                }
                if per_frame_img is not None:
                    viz_batch[cam_keys[0]] = per_frame_img
                viz_preds = {f"{embodiment_name}_{ac_key}": pred}
            else:
                # Padded mode (legacy).
                gt = _batch_unnorm[ac_key]
                mse_val = ((pred - gt) ** 2).mean()
                metrics[f"Valid/emb{emb_id}_{ac_key}_paired_mse"] = mse_val.detach()
                metrics[f"Valid/emb{emb_id}_{ac_key}_final_mse"] = (
                    ((pred[:, -1] - gt[:, -1]) ** 2).mean().detach()
                )
                viz_batch = copy.copy(_batch_unnorm)
                viz_preds = {f"{embodiment_name}_{ac_key}": pred}

            if total_loss is None:
                total_loss = torch.zeros_like(mse_val)
            total_loss = total_loss + mse_val
            n_loss += 1

            if self.viz_func is not None and embodiment_name in self.viz_func:
                ims = self.viz_func[embodiment_name](
                    viz_preds, viz_batch, max_videos=self.max_videos
                )
                images_dict[emb_id] = ims

        if total_loss is not None and n_loss > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss

        return metrics, images_dict
