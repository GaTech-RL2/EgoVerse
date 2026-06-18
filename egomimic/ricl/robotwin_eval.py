"""Validation for the RoboTwin RICL run — the RoboTwin analog of ``DroidRiclEval``.

Reuses the proven DROID evaluator wholesale (retrieval / random-within /
random-bank / clean zero-context floor on the *raw* batch, paired-seed flow loss
averaged over ``n_flow_samples``, optional sampled-action MSE/L1/gripper-accuracy,
and distance-weighted interpolation #2). The ONLY embodiment-specific change is
``_sampled_metrics``: RoboTwin is 16-D joint-space with **two** grippers (slots 7
and 15), vs DROID's 8-D single gripper at slot 7 — so the metric is restricted to
the real 16 dims and gripper accuracy is scored at both slots.

Mirrors ``egomimic/ricl/droid_eval.py`` line-for-line except the dim/gripper
constants — keep the two in sync if the DROID eval changes.
"""

from __future__ import annotations

import torch

from egomimic.ricl import metrics as M
from egomimic.ricl import robotwin_data as R
from egomimic.ricl.droid_eval import DroidRiclEval, DroidRiclModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment


class RoboTwinRiclEval(DroidRiclEval):
    """DROID's retrieval-vs-floor flow-loss eval, parametrized for RoboTwin's
    16-D joint-space action with two grippers."""

    def __init__(
        self, *args, gripper_indices=R.GRIPPER_SLOTS, raw_dim=R.RAW_DIM, **kwargs
    ):
        super().__init__(*args, gripper_indices=gripper_indices, **kwargs)
        # RoboTwin's real qpos dim (detect per-corpus: 14 for aloha-agilex, 16 for
        # 7-DOF). Slice the sampled-action error to these dims; the 32-D slot-fill
        # pad dims are zero in the target and would only dilute the metric.
        self._raw_dim = int(raw_dim)
        self._gripper_indices = tuple(gripper_indices)

    def _sampled_metrics(self, processed_batch, seed: int, *, dist_max: float = 1.0):
        """Sampled-action MSE/L1/gripper-accuracy (mirrors DroidRiclEval but over
        RoboTwin's real 16 dims and both gripper slots)."""
        algo = self.model
        base: dict[str, float] = {}
        sweep: dict[str, float] = {}
        d = self._raw_dim
        torch.manual_seed(seed)
        preds = algo.forward_eval(processed_batch)
        for emb_id, _b in processed_batch.items():
            name = get_embodiment(emb_id).lower()
            ac_key = algo.ac_keys[emb_id]
            pred_key = f"{name}_{ac_key}"
            if pred_key not in preds:
                continue
            p = preds[pred_key].detach().cpu()[..., :d]
            g = _b[ac_key].detach().cpu()[..., :d]
            base[f"{name}_sampled_mse"] = M.cartesian_mse(p, g)
            base[f"{name}_sampled_l1"] = M.cartesian_l1(p, g)
            ga = M.gripper_accuracy(p, g, self._gripper_indices, 0.0)
            if ga == ga:  # not NaN
                base[f"{name}_sampled_gripper_acc"] = ga
            for lam in self.interp_lamdas:
                p_l, wmean = self._interpolate_toward_neighbor(p, _b, dist_max, lam)
                if p_l is None:
                    continue
                sweep[f"{name}_sampled_mse_l{lam:g}"] = M.cartesian_mse(p_l, g)
                sweep[f"{name}_interp_wmean_l{lam:g}"] = wmean
        return base, sweep


class RoboTwinRiclModelWrapper(DroidRiclModelWrapper):
    """Identical to DroidRiclModelWrapper: pass the RAW val batch to the evaluator
    (for a clean floor) and keep the fit/validation barriers single-device-safe."""
