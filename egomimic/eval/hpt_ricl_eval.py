"""HptRiclEval: compare retrieval-conditioned HPT vs the zero-context floor.

The HPT analog of :class:`egomimic.eval.pi_ricl_eval.PIRiclEval`. For each
validation batch it runs the model on the *same* eva query frames twice:
  - retrieval: the full batch (with ``ricl_*`` keys) -> HptRicl encodes k retrieved
    in-context demos through the retrieved stems,
  - floor: the same batch with ``ricl_*`` stripped -> HptRicl == plain HPT (k=0).
It reports sampled-action Cartesian MSE / L1 / gripper accuracy + BC loss for both
conditions and the deltas (``RICL/delta_*``, ``RICL/retrieval_helps``). Retrieval
"works" when loss / MSE drop vs the floor; for the eva->eva oracle this bounds the
ceiling.

**Clean floor (strip before processing).** This evaluator receives the *raw* batch
(``wants_raw_batch = True``) and builds each condition itself. The floor strips
``ricl_*`` from the **raw** batch and *then* processes it, so
``process_batch_for_training`` carries no retrieved tensors and
``_robomimic_to_hpt_data`` adds no retrieved stems -> a genuine k=0 floor (stripping
after processing would leave the retrieved tokens in the model input).

v1 is intentionally lean (retrieval vs floor). The model-agnostic random-demo
control (``compute_random``, via :func:`shuffle_ricl_keys`) and the cheap
paired-seed flow-loss proxy (``compute_flow_loss``) are wired but default OFF;
flip them on to match the fuller PIRiclEval ablation set.
"""

from __future__ import annotations

import torch

from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.ricl import metrics as M
from egomimic.rldb.embodiment.embodiment import get_embodiment


class HptRiclEval(HPTEvalVideo):
    # Receive the raw (un-processed) batch so the floor is built by stripping
    # ricl_* before process_batch_for_training (see module docstring).
    wants_raw_batch = True

    def __init__(
        self,
        *args,
        compute_floor: bool = True,
        compute_flow_loss: bool = False,
        compute_random: bool = False,
        n_flow_samples: int = 4,
        seed_base: int = 1234,
        gripper_threshold: float = 0.0,
        gripper_indices=(6, 13),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_floor = compute_floor
        self.compute_flow_loss = compute_flow_loss
        self.compute_random = compute_random
        self.n_flow_samples = max(1, int(n_flow_samples))
        self.seed_base = seed_base
        self.gripper_threshold = gripper_threshold
        self.gripper_indices = tuple(gripper_indices)
        self._batch_idx = 0

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Shared per-batch seed so the optional flow-loss conditions see identical
        # noise/time (the delta then isolates the conditioning, not RNG).
        self._batch_idx = batch_idx
        super().on_validation_step(batch, batch_idx, dataloader_idx)

    # ------------------------------------------------------------------
    # One condition: sampled forward_eval + native-space scalar metrics.
    # (HPT.forward_eval already unnormalizes its predictions; unnormalize the GT.)
    # ------------------------------------------------------------------
    def _eval_condition(self, proc_batch, make_viz: bool):
        algo = self.model
        preds = algo.forward_eval(proc_batch)
        metrics, viz = {}, {}
        for embodiment_id, _batch in proc_batch.items():
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
                if make_viz and self.viz_func is not None:
                    # Native-frame viz (the base's cam-frame revert is omitted here,
                    # mirroring PIRiclEval — the scalar deltas are the headline).
                    viz[embodiment_id] = self._visualize_preds(preds, _batch)
        return metrics, viz

    def _flow_loss(self, proc_batch, seed: int) -> dict:
        """Paired-seed flow-matching (training) loss, averaged over n draws. Cheap
        (no sampling) and, scored under shared seeds, isolates the conditioning."""
        algo = self.model
        accum: dict[str, float] = {}
        with torch.no_grad():
            for s in range(self.n_flow_samples):
                torch.manual_seed(seed + s * 100003)
                preds = algo.forward_training(proc_batch)
                for k, v in preds.items():
                    if k.endswith("_loss"):
                        nk = k.replace("_loss", "_flow_loss")
                        accum[nk] = accum.get(nk, 0.0) + float(v)
        return {k: v / self.n_flow_samples for k, v in accum.items()}

    def compute_metrics_and_viz(self, raw_batch):
        algo = self.model
        seed = self.seed_base + self._batch_idx

        # retrieval (sampled): process the full raw batch -> retrieved demos encoded.
        proc_ret = algo.process_batch_for_training(raw_batch)
        ret_metrics, viz = self._eval_condition(proc_ret, make_viz=True)
        out = {f"RICL/retrieval_{k}": v for k, v in ret_metrics.items()}

        floor_metrics = None
        if self.compute_floor:
            # strip ricl_* from the RAW batch, then process -> genuine k=0 floor.
            proc_flr = algo.process_batch_for_training(M.strip_ricl_keys(raw_batch))
            floor_metrics, _ = self._eval_condition(proc_flr, make_viz=False)
            out.update({f"RICL/floor_{k}": v for k, v in floor_metrics.items()})
            out.update(
                {
                    f"RICL/{k}": v
                    for k, v in M.compare_to_floor(ret_metrics, floor_metrics).items()
                }
            )

        # Optional cheap paired-seed flow-loss proxy (off by default).
        if self.compute_flow_loss:
            ret_flow = self._flow_loss(proc_ret, seed)
            out.update({f"RICL/retrieval_{k}": v for k, v in ret_flow.items()})
            if self.compute_floor:
                floor_flow = self._flow_loss(
                    algo.process_batch_for_training(M.strip_ricl_keys(raw_batch)), seed
                )
                out.update({f"RICL/floor_{k}": v for k, v in floor_flow.items()})
                out.update(
                    {
                        f"RICL/flow_{k}": v
                        for k, v in M.compare_to_floor(ret_flow, floor_flow).items()
                    }
                )

        # Optional random-demo control (off by default): each query gets another
        # query's k demos; retrieval < random asks whether *similarity* matters.
        if self.compute_random:
            shuffled = M.shuffle_ricl_keys(raw_batch, seed)
            if shuffled is not None:
                proc_rnd = algo.process_batch_for_training(shuffled)
                rnd_metrics, _ = self._eval_condition(proc_rnd, make_viz=False)
                out.update({f"RICL/random_{k}": v for k, v in rnd_metrics.items()})
                cmp_r = M.compare_to_floor(ret_metrics, rnd_metrics)
                out["RICL/beats_random"] = float(cmp_r["retrieval_helps"])
                out["RICL/improvement_vs_random"] = cmp_r["mean_improvement"]

        # Primary scalar the trainer logs.
        for k, v in ret_metrics.items():
            if k.endswith("_loss"):
                out["Valid/action_loss"] = v
                break
        return out, viz
