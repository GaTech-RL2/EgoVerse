"""Eval for arc-tokenized action policies.

The trained model outputs a ``(B, M+1, 8)`` tensor per sample: M waypoints
followed by 1 velocity token, each 8-dim
``[Lx, Ly, Lz, L_grip, Rx, Ry, Rz, R_grip]``. Training loss and validation
MSE run in that arc-token space directly (matched targets from the same
tokenizer that produced them).

Val videos are the piece that need special handling: the canonical
``_viz_traj`` overlay expects a time-parameterized ``(H, 14)`` chunk and
projects every row as a dot. So before handing predictions and GT to the
viz path, we DETOKENIZE — reconstruct a stream of time-parameterized
setpoints at the control rate using the predicted velocity token to set
the timing, then zero-pad the missing rotation columns to match the
canonical 14-dim layout. Output shape into viz: ``(B, H, 14)``.

Rotation columns land as zeros because the arc-token variant doesn't
supervise orientation; the projection just needs xyz so this is harmless.
Gripper column continues to carry a real value (either the eva command or
the human zero-pad).
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_TOK_BIMANUAL_DIM,
    TokenizeBimanualArcLengthCartesian,
)


class ArcTokEvalVideo(HPTEvalVideo):
    """HPT evaluator specialized for the arc-tokenized action head.

    Adds a detokenization pass over the model output (and matched GT) before
    the viz function runs, so val videos overlay the *reconstructed*
    time-parameterized trajectory — the exact thing an open-loop rollout
    would send to the controller — rather than the raw arc waypoints.

    Config knobs (via yaml):
      * ``min_distance_unit``: D (meters) — must match the tokenizer used
        by the data pipeline. Determines the reconstruction speed via
        duration = D / ||vel||.
      * ``resampled_vector_length``: M — number of waypoints (also must
        match the data pipeline). Model output must be ``(B, M+1, 8)``.
      * ``rollout_horizon``: H (frames) — how many control-period steps
        to reconstruct per sample. Default 100 to match the non-arc
        cotrain val-video length. If ``duration < H*dt``, the tail is
        clamped to the last waypoint (matches how a real rollout would
        stop emitting once the chunk is exhausted).
      * ``dt``: control period in seconds (default 1/30).
    """

    def __init__(
        self,
        limit_val_batches: int = 400,
        viz_func: dict = None,
        transform_lists: dict | None = None,
        min_distance_unit: float = 0.60,
        resampled_vector_length: int = 20,
        rollout_horizon: int = 100,
        dt: float = 1.0 / 30.0,
    ):
        super().__init__(
            limit_val_batches=limit_val_batches,
            viz_func=viz_func,
            transform_lists=transform_lists,
        )
        # Reuse the tokenizer's ``detokenize`` — same reconstruction algebra
        # as an open-loop rollout, so val videos are faithful to deploy.
        self._detokenizer = TokenizeBimanualArcLengthCartesian(
            action_key="actions_cartesian",
            output_action_key="actions_cartesian",
            min_distance_unit=float(min_distance_unit),
            resampled_vector_length=int(resampled_vector_length),
            dt=float(dt),
        )
        self.rollout_horizon = int(rollout_horizon)
        self._M = int(resampled_vector_length)

    def _detokenize_batch(self, arc_tensor: torch.Tensor) -> torch.Tensor:
        """(B, M+1, 8) arc tokens -> (B, H, 14) time-parameterized chunks
        with zero-padded rotation columns.

        The canonical ``_viz_traj`` reads xyz from dims 0:3 and 7:10 (see
        ``viz_utils.py:150``). The detokenized 8-dim layout is
        ``[Lx, Ly, Lz, L_grip, Rx, Ry, Rz, R_grip]`` — after padding
        rotation slots at 3:6 and 10:13 it becomes the canonical 14-dim
        ``[L xyz ypr grip | R xyz ypr grip]`` and viz projects xyz through
        K exactly like it does for time-based models.
        """
        if not isinstance(arc_tensor, torch.Tensor):
            arc_tensor = torch.as_tensor(arc_tensor)
        arc_np = arc_tensor.detach().cpu().numpy().astype(np.float64)
        if arc_np.ndim != 3 or arc_np.shape[-1] != ARC_TOK_BIMANUAL_DIM:
            raise ValueError(
                f"ArcTokEvalVideo expects arc actions shaped (B, M+1, "
                f"{ARC_TOK_BIMANUAL_DIM}), got {arc_np.shape}"
            )
        if arc_np.shape[1] != self._M + 1:
            raise ValueError(
                f"ArcTokEvalVideo configured for M={self._M} (M+1={self._M+1} "
                f"tokens), got {arc_np.shape[1]} tokens in the model output"
            )
        B = arc_np.shape[0]
        H = self.rollout_horizon
        out = np.zeros((B, H, 14), dtype=np.float64)
        for b in range(B):
            det = self._detokenizer.detokenize(arc_np[b], action_horizon=H)  # (H, 8)
            # Splice into canonical 14-dim layout with zero rotation.
            out[b, :, 0:3] = det[:, 0:3]  # L xyz
            out[b, :, 6:7] = det[:, 3:4]  # L gripper
            out[b, :, 7:10] = det[:, 4:7]  # R xyz
            out[b, :, 13:14] = det[:, 7:8]  # R gripper
            # rotation columns [3:6] and [10:13] stay zero — the arc-token
            # variant doesn't carry orientation, and viz never reads these
            # dims for mode='traj'.
        return torch.from_numpy(out).to(arc_tensor.device, dtype=arc_tensor.dtype)

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()

        algo = self.model
        ac_key = algo.ac_keys[embodiment_id]
        main_pred_key = f"{embodiment_name}_{ac_key}"

        # Detokenize both GT and prediction so the viz path sees a
        # time-parameterized (B, H, 14) chunk it can project as dots.
        preds_viz = dict(predictions)
        batch_viz = dict(batch)
        if main_pred_key in predictions and predictions[main_pred_key] is not None:
            preds_viz[main_pred_key] = self._detokenize_batch(
                predictions[main_pred_key]
            )
        if ac_key in batch and batch[ac_key] is not None:
            batch_viz[ac_key] = self._detokenize_batch(batch[ac_key])

        return self.viz_func[embodiment_name](preds_viz, batch_viz)

    def compute_metrics_and_viz(self, batch):
        """Extends the base HPTEvalVideo pass with DETOKENIZED-space MSE
        metrics — the arc-token-space MSE inherited from HPTEvalVideo
        (see eval_hpt.py:50-56) is the training loss space, but it's in
        arc-token units (meters/second for vel, meters for waypoints) and
        hard to compare against the non-arc cotrain baseline directly.

        We add, per embodiment:
          * ``Valid/{emb}_actions_cartesian_detok_paired_mse_avg`` —
            MSE over the full (H, 8) detokenized chunk (waypoint-derived
            xyz + gripper reconstructed via the predicted vel token).
          * ``Valid/{emb}_actions_cartesian_detok_final_mse_avg`` — MSE
            at the last reconstructed frame (t = H-1).
          * ``Valid/{emb}_actions_cartesian_detok_xyz_mse_avg`` — MSE
            over just the 6 xyz dims (Lxyz + Rxyz), dropping grippers.
          * ``Valid/{emb}_actions_cartesian_detok_gripper_mse_avg`` —
            MSE over just the 2 gripper dims (L_grip + R_grip).

        The detokenized MSE directly measures "how close is the
        reconstructed trajectory to the target" in physical units, which
        is the metric that transfers to open-loop deployment quality.
        """
        metrics, images_dict = super().compute_metrics_and_viz(batch)

        algo = self.model
        # Reuse the algo's forward pass output; it was already unnormalized
        # by super().compute_metrics_and_viz upstream. We re-run
        # forward_eval to get preds again (super doesn't hand them back),
        # and unnormalize the batch the same way super did.
        preds = algo.forward_eval(batch)
        mse = MeanSquaredError()
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            if pred_key not in preds or ac_key not in _batch:
                continue
            pred_arc = preds[pred_key]
            gt_arc = _batch[ac_key]
            if pred_arc is None or gt_arc is None:
                continue
            # Detokenize to (B, H, 14) then take only the xyz + gripper
            # slots the arc variant actually supervises (rotation is
            # zero-padded so including it would inflate MSE with pure
            # zeros — mislead the metric).
            pred_det = self._detokenize_batch(pred_arc).cpu().contiguous()
            gt_det = self._detokenize_batch(gt_arc).cpu().contiguous()
            xyzg = [0, 1, 2, 6, 7, 8, 9, 13]  # Lxyz, L_grip, Rxyz, R_grip
            # ``.contiguous()`` after every advanced-index slice: the
            # ``.contiguous()`` on pred_det/gt_det above does not survive
            # ``[..., xyzg]``, and torchmetrics' MSE calls ``.view()``
            # internally, which raises "view size is not compatible with input
            # tensor's size and stride" on a non-contiguous tensor.
            pred_slice = pred_det[..., xyzg].contiguous()
            gt_slice = gt_det[..., xyzg].contiguous()

            metrics[f"Valid/{pred_key}_detok_paired_mse_avg"] = mse(
                pred_slice, gt_slice
            )
            metrics[f"Valid/{pred_key}_detok_final_mse_avg"] = mse(
                pred_slice[:, -1].contiguous(), gt_slice[:, -1].contiguous()
            )
            # After the ``xyzg`` slice above, the 8 columns are laid out
            # as ``[Lx, Ly, Lz, L_grip, Rx, Ry, Rz, R_grip]``. Splitting
            # into xyz vs gripper lets us tell a "position drift" story
            # apart from a "gripper-timing" story in wandb.
            xyz_slots = [0, 1, 2, 4, 5, 6]
            grip_slots = [3, 7]
            metrics[f"Valid/{pred_key}_detok_xyz_mse_avg"] = mse(
                pred_slice[..., xyz_slots].contiguous(),
                gt_slice[..., xyz_slots].contiguous(),
            )
            metrics[f"Valid/{pred_key}_detok_gripper_mse_avg"] = mse(
                pred_slice[..., grip_slots].contiguous(),
                gt_slice[..., grip_slots].contiguous(),
            )

        return metrics, images_dict
