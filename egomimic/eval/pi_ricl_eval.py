"""PIRiclEval: compare retrieval-conditioned pi0.5 vs the zero-context floor. (P4)

For each validation batch this runs the model twice on the *same* eva query frames:
  - retrieval: the full batch (with ``ricl_*`` keys) -> PIRicl injects k retrieved
    in-context demos,
  - floor: the same batch with ``ricl_*`` keys stripped -> PIRicl == base pi0.5
    (k=0, zero-context).
It reports flow val loss + Cartesian paired/final MSE + L1 + gripper accuracy for
both, and the deltas (``RICL/delta_*`` and ``RICL/retrieval_helps``). Retrieval
"works" when loss/MSE drop vs the floor. For the within-embodiment **oracle**
(D0, bank = eva), point ``data.bank_*`` at an eva bank; the same metrics then
bound the ceiling. Mirrors ``ricl_openpi/scripts/eval_cross_embodiment.py``.

Self-contained (one ``forward_eval`` per condition); the cam-frame revert MSE of
the base :class:`PIEvalVideo` is intentionally omitted here — native-frame MSE,
loss and gripper accuracy are the headline numbers.
"""

from __future__ import annotations

from egomimic.eval.eval_pi import PIEvalVideo
from egomimic.ricl import metrics as M
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PIRiclEval(PIEvalVideo):
    def __init__(
        self,
        *args,
        compute_floor: bool = True,
        gripper_threshold: float = 0.0,
        gripper_indices=(6, 13),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_floor = compute_floor
        self.gripper_threshold = gripper_threshold
        self.gripper_indices = tuple(gripper_indices)

    def _eval_condition(self, batch, make_viz: bool):
        algo = self.model
        preds = algo.forward_eval(batch)
        metrics, viz = {}, {}
        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{name}_{ac_key}"
            loss_key = f"{name}_loss"

            if loss_key in preds:
                metrics[f"{name}_loss"] = float(preds[loss_key])
            if pred_key in preds:
                p = preds[pred_key].cpu()
                g = _batch[ac_key].cpu()
                metrics[f"{name}_paired_mse"] = M.cartesian_mse(p, g)
                metrics[f"{name}_final_mse"] = M.cartesian_mse(p[:, -1], g[:, -1])
                metrics[f"{name}_paired_l1"] = M.cartesian_l1(p, g)
                ga = M.gripper_accuracy(
                    p, g, self.gripper_indices, self.gripper_threshold
                )
                if ga == ga:  # not NaN (i.e. the 14-D bimanual layout)
                    metrics[f"{name}_gripper_acc"] = ga
                if make_viz:
                    viz[embodiment_id] = self._visualize_preds(preds, _batch)
        return metrics, viz

    def compute_metrics_and_viz(self, batch):
        ret_metrics, viz = self._eval_condition(batch, make_viz=True)
        out = {f"RICL/retrieval_{k}": v for k, v in ret_metrics.items()}

        if self.compute_floor:
            floor_batch = M.strip_ricl_keys(batch)
            floor_metrics, _ = self._eval_condition(floor_batch, make_viz=False)
            out.update({f"RICL/floor_{k}": v for k, v in floor_metrics.items()})
            cmp = M.compare_to_floor(ret_metrics, floor_metrics)
            out.update({f"RICL/{k}": v for k, v in cmp.items()})

        # Surface a scalar the trainer logs as the primary validation number.
        for k, v in ret_metrics.items():
            if k.endswith("_loss"):
                out["Valid/action_loss"] = v
                break
        return out, viz
