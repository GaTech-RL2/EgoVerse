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

**Clean floor (strip before tokenize).** This evaluator receives the *raw* batch
(``wants_raw_batch = True`` -> :class:`ModelWrapper.validation_step` skips its
``process_batch_for_training``) and builds each condition itself. The floor must
be built by stripping ``ricl_*`` from the **raw** batch and *then* processing it:
``process_batch_for_training`` is what splices the retrieved demos' state/action
into the prompt and tokenizes it, so stripping *after* processing (the old path)
left the demo *text* baked into ``tokenized_prompt`` — the "floor" still saw the
demos and the measured retrieval gain was understated. Stripping first makes
``_build_prompts`` see no demos and emit a plain prompt: a genuine k=0 floor.

**Paired-seed flow loss (#3, ``compute_flow_loss``).** Alongside the sampled
``forward_eval`` metrics, each condition is also scored by the flow-matching
*training* loss under a single per-batch seed (``seed_base + batch_idx``). Sharing
the seed gives every condition identical noise/time, so ``RICL/flow_delta_*``
isolates the conditioning rather than the sampling RNG — and it skips the slow
``torch.compile`` action-sampling path, so it's cheap to log every validation.

**Random-retrieval control (#2, ``compute_random``).** A third condition derranges
the per-query ``ricl_retrieved_*`` blocks across the batch (``shuffle_ricl_keys``):
every query is spliced with *another* query's k real demos. ``floor < random``
(``RICL/random_helps``) asks "does *any* in-context demo help?"; ``retrieval <
random`` (``RICL/beats_random``) asks "does it matter the demos are visually
*similar*?" — the actual RICL claim. Mirrors ``DroidRiclEval`` in
``egomimic/ricl/droid_eval.py``, where this ablation was first validated.
"""

from __future__ import annotations

import torch

from egomimic.eval.eval_pi import PIEvalVideo
from egomimic.ricl import metrics as M
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PIRiclEval(PIEvalVideo):
    # Receive the raw (un-processed) batch so the floor can be built by stripping
    # ricl_* *before* the prompt is tokenized (see module docstring).
    wants_raw_batch = True

    def __init__(
        self,
        *args,
        compute_floor: bool = True,
        compute_flow_loss: bool = True,
        compute_random: bool = True,
        compute_sampled_random: bool = True,
        n_flow_samples: int = 8,
        seed_base: int = 1234,
        gripper_threshold: float = 0.0,
        gripper_indices=(6, 13),
        interp_lamda: float = 0.0,
        interp_lamdas=(0.5, 1.0, 2.0, 5.0, 10.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compute_floor = compute_floor
        # Paired-seed flow loss (#3) + random-retrieval control (#2); see the
        # ``_flow_loss`` / ``compute_metrics_and_viz`` docstrings.
        self.compute_flow_loss = compute_flow_loss
        self.compute_random = compute_random
        # Also score the *sampled* actions for the random control (forward_eval —
        # the slow torch.compile path) so ``beats_random`` can be judged on
        # sampled-action MSE, not just the flow proxy. The DROID verification found
        # the flow loss HIDES the retrieval-vs-random gap (retrieval≈random there)
        # while sampled MSE reveals it — so this is the headline diagnostic here.
        self.compute_sampled_random = compute_sampled_random
        # Average the paired-seed flow loss over this many noise/time draws to cut
        # the Monte-Carlo variance that otherwise swamps the small retrieval deltas.
        self.n_flow_samples = max(1, int(n_flow_samples))
        self.seed_base = seed_base
        self.gripper_threshold = gripper_threshold
        self.gripper_indices = tuple(gripper_indices)
        # Distance-weighted action interpolation sweep (inference-only): blend the
        # sampled action toward the NEAREST retrieved demo's (native-space) chunk
        # with w=exp(-lamda*dist/dist_max). On DROID λ≈1 was the single biggest
        # lever (−62% MSE vs floor). Empty -> no interpolation. Needs the cache to
        # store real ``ricl_retrieved_dist``.
        if interp_lamdas:
            self.interp_lamdas = tuple(float(x) for x in interp_lamdas if float(x) > 0)
        elif interp_lamda and float(interp_lamda) > 0:
            self.interp_lamdas = (float(interp_lamda),)
        else:
            self.interp_lamdas = ()
        self._batch_idx = 0

    def _eval_condition(self, batch, make_viz: bool, return_preds: bool = False):
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
        if return_preds:
            return metrics, viz, preds
        return metrics, viz

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Record the batch index so the flow-loss conditions share one per-batch
        # seed -> identical noise/time across retrieval/floor/random, so the delta
        # isolates the conditioning rather than RNG (see ``_flow_loss``).
        self._batch_idx = batch_idx
        super().on_validation_step(batch, batch_idx, dataloader_idx)

    def _flow_loss(self, processed_batch, seed: int) -> dict:
        """Flow-matching training loss under *seeded* noise/time. (#3)

        Cheaper than ``forward_eval``'s sampled-action path (no slow torch.compile
        action sampling) and, because every condition is scored under the *same*
        seeds, the retrieval/floor/random deltas isolate the conditioning, not the
        sampling RNG. Averaged over ``n_flow_samples`` draws (each with a shared
        per-draw seed ``seed + s*P``) so the Monte-Carlo variance of a single
        noise/time draw doesn't swamp the small retrieval deltas. Renames the
        ``{name}_loss`` keys to ``{name}_flow_loss`` so they don't clobber the
        sampled-action ``{name}_loss`` of ``_eval_condition``.
        """
        algo = self.model
        n = self.n_flow_samples
        accum: dict[str, float] = {}
        with torch.no_grad():
            for s in range(n):
                torch.manual_seed(seed + s * 100003)
                preds = algo.forward_training(processed_batch)
                for k, v in preds.items():
                    if k.endswith("_loss"):
                        accum[k.replace("_loss", "_flow_loss")] = accum.get(
                            k.replace("_loss", "_flow_loss"), 0.0
                        ) + float(v)
        return {k: v / n for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Distance-weighted action interpolation (inference-only; #2 from DROID).
    # Ported/generalized from egomimic/ricl/droid_eval.py to eva's native action
    # width: the neighbor action is unnormalized to physical units so it shares
    # the space of forward_eval's predictions before blending.
    # ------------------------------------------------------------------
    def _interp_scale(self, proc) -> float | None:
        """Per-batch normalization scale for the interp weight: the max finite kNN
        neighbor distance across the processed batch (mirrors ricl_openpi's global
        ``max_distance``, computed per-batch from the *retrieval* condition so all
        conditions share one scale). Returns ``None`` when no positive distance is
        present (e.g. a cache that didn't store distances) so the sweep is skipped
        rather than degenerating to w≡1 (pure neighbor replay)."""
        best = 0.0
        for _b in proc.values():
            d = _b.get("ricl_retrieved_dist")
            if d is not None:
                df = d[torch.isfinite(d)]
                if df.numel():
                    best = max(best, float(df.max()))
        return best if best > 0 else None

    def _neighbor_native_action(self, _b, emb_id, ac_key, width):
        """Nearest retrieved demo's action chunk, mapped to NATIVE physical units.

        ``ricl_retrieved_action`` is stored quantile-normalized in the query
        convention (width = data ``action_dim``); forward_eval's predictions are
        ``from32``-converted + unnormalized. To blend the two we put the neighbor
        in the same native space: (optionally ``from32`` if slot-filled to 32-D)
        then ``norm_stats.unnormalize`` under ``ac_key``. Returns ``(B, Ha, width)``
        on CPU, or ``None`` if no neighbor action is present."""
        a_nn = _b.get("ricl_retrieved_action")
        if a_nn is None:
            return None
        a0 = a_nn[:, 0].detach().float()  # (B, Ha, Dw) nearest neighbor, normalized
        if a0.shape[-1] != width:
            if a0.shape[-1] == 32:
                a0 = self.model.action_registry.get(emb_id, ac_key).from32(a0)
            else:
                a0 = a0[..., :width]
        a0 = self.model.norm_stats.unnormalize({ac_key: a0}, emb_id)[ac_key]
        return a0.detach().cpu()

    def _interpolate_toward_neighbor(self, p, a0, dist, mask, dist_max, lamda):
        """``p <- w*a0 + (1-w)*p`` over the overlapping chunk, with
        ``w=exp(-lamda*dist/dist_max)`` (masked, finite-guarded). ``p`` and ``a0``
        must already share a space (native). Returns ``(blended_p, mean_w)``."""
        d0 = dist[:, 0].detach().cpu().float()
        m0 = (
            mask[:, 0].detach().cpu().float()
            if mask is not None
            else torch.ones_like(d0)
        )
        finite = torch.isfinite(d0)
        w = (
            torch.where(finite, torch.exp(-lamda * d0 / dist_max), torch.zeros_like(d0))
            * m0
        )  # (B,)
        H = min(p.shape[1], a0.shape[1])
        out = p.clone()
        wv = w[:, None, None]
        out[:, :H] = wv * a0[:, :H] + (1.0 - wv) * out[:, :H]
        return out, float(w.mean())

    def _interp_sweep(self, preds, proc, dist_max) -> dict:
        """For each λ, blend the retrieval sampled action toward the nearest
        neighbor's native chunk and report the resulting full-chunk MSE (directly
        comparable to the no-interp ``*_paired_mse``). Reuses ``preds`` (no extra
        sampling). Empty if ``dist_max`` is None or no λ configured."""
        out: dict[str, float] = {}
        if dist_max is None or not self.interp_lamdas:
            return out
        algo = self.model
        for emb_id, _b in proc.items():
            name = get_embodiment(emb_id).lower()
            ac_key = algo.ac_keys[emb_id]
            pred_key = f"{name}_{ac_key}"
            if pred_key not in preds:
                continue
            p = preds[pred_key].detach().cpu()
            # GT is normalized in ``proc``; unnormalize a copy to native (preds
            # are already native, and _eval_condition does not mutate ``proc``).
            gt = algo.norm_stats.unnormalize(dict(_b), emb_id)[ac_key].detach().cpu()
            a0 = self._neighbor_native_action(_b, emb_id, ac_key, p.shape[-1])
            if a0 is None:
                continue
            dist = _b.get("ricl_retrieved_dist")
            mask = _b.get("ricl_retrieved_mask")
            if dist is None:
                continue
            for lam in self.interp_lamdas:
                p_l, wmean = self._interpolate_toward_neighbor(
                    p, a0, dist, mask, dist_max, lam
                )
                out[f"{name}_sampled_mse_l{lam:g}"] = M.cartesian_mse(p_l, gt)
                out[f"{name}_interp_wmean_l{lam:g}"] = wmean
        return out

    @staticmethod
    def _mse_only(metrics: dict) -> dict:
        return {k: v for k, v in metrics.items() if "mse" in k or "l1" in k}

    def compute_metrics_and_viz(self, raw_batch):
        algo = self.model
        seed = self.seed_base + self._batch_idx

        # retrieval (sampled): process the full raw batch -> demos spliced + tokenized.
        proc_ret = algo.process_batch_for_training(raw_batch)
        ret_metrics, viz, ret_preds = self._eval_condition(
            proc_ret, make_viz=True, return_preds=True
        )
        out = {f"RICL/retrieval_{k}": v for k, v in ret_metrics.items()}

        proc_flr = None
        floor_metrics = None
        floor_preds = None
        if self.compute_floor:
            # floor: strip ricl_* from the RAW batch, *then* process, so the prompt
            # is built/tokenized with no demo text -> a genuine k=0 zero-context floor.
            proc_flr = algo.process_batch_for_training(M.strip_ricl_keys(raw_batch))
            floor_metrics, _, floor_preds = self._eval_condition(
                proc_flr, make_viz=False, return_preds=True
            )
            out.update({f"RICL/floor_{k}": v for k, v in floor_metrics.items()})
            cmp = M.compare_to_floor(ret_metrics, floor_metrics)
            out.update({f"RICL/{k}": v for k, v in cmp.items()})

        # (#2) Distance-weighted interpolation sweep (inference-only; reuses the
        # sampled preds, no extra sampling). Each λ emits ``*_sampled_mse_l{λ}``
        # directly comparable to the no-interp ``*_paired_mse``. On DROID λ≈1 was the
        # biggest lever. Run it on BOTH the retrieval prediction AND the floor
        # prediction: the floor sweep is the fair "clean policy + kNN-action blend"
        # control (the query frames — hence the neighbors in ``proc_ret`` — are
        # identical for floor and retrieval), so it isolates whether interp helps
        # *because of* the trained demo-reading or is a generic post-hoc trick that
        # would lift a plain finetune just as much.
        dist_max = self._interp_scale(proc_ret)
        out.update(
            {
                f"RICL/retrieval_{k}": v
                for k, v in self._interp_sweep(ret_preds, proc_ret, dist_max).items()
            }
        )
        if floor_preds is not None:
            out.update(
                {
                    f"RICL/floor_{k}": v
                    for k, v in self._interp_sweep(
                        floor_preds, proc_ret, dist_max
                    ).items()
                }
            )

        # Random-retrieval control built ONCE and reused for both the sampled-MSE
        # judging (#2 headline) and the flow-loss judging (#3): each query keeps its
        # target but gets another query's k demos (same splice path, not kNN-matched).
        proc_rnd = None
        if self.compute_random:
            shuffled = M.shuffle_ricl_keys(raw_batch, seed)
            if shuffled is not None:
                proc_rnd = algo.process_batch_for_training(shuffled)

        # (#2 headline) sampled-action MSE judging of retrieval vs random vs floor.
        # The DROID verification showed the flow loss HIDES this gap (retrieval≈random
        # there) while sampled MSE reveals it -> this is the diagnostic that decides
        # whether retrieval is discriminative, so it drives ``beats_random_sampled``.
        if self.compute_sampled_random and proc_rnd is not None:
            rnd_metrics, _ = self._eval_condition(proc_rnd, make_viz=False)
            out.update({f"RICL/random_{k}": v for k, v in rnd_metrics.items()})
            cmp_s = M.compare_to_floor(
                self._mse_only(ret_metrics), self._mse_only(rnd_metrics)
            )
            out["RICL/beats_random_sampled"] = float(cmp_s["retrieval_helps"])
            out["RICL/improvement_vs_random_sampled"] = cmp_s["mean_improvement"]
            out.update(
                {
                    f"RICL/vsrandom_sampled_{k}": v
                    for k, v in cmp_s.items()
                    if k.startswith("delta_")
                }
            )
            if floor_metrics is not None:
                cmp_sf = M.compare_to_floor(
                    self._mse_only(rnd_metrics), self._mse_only(floor_metrics)
                )
                out["RICL/random_helps_sampled"] = float(cmp_sf["retrieval_helps"])

        # (#3) paired-seed flow loss under one shared seed so the deltas isolate the
        # conditioning. Cheap; logged every validation as a low-variance signal.
        if self.compute_flow_loss:
            ret_flow = self._flow_loss(proc_ret, seed)
            out.update({f"RICL/retrieval_{k}": v for k, v in ret_flow.items()})

            floor_flow = None
            if proc_flr is not None:
                floor_flow = self._flow_loss(proc_flr, seed)
                out.update({f"RICL/floor_{k}": v for k, v in floor_flow.items()})
                out.update(
                    {
                        f"RICL/flow_{k}": v
                        for k, v in M.compare_to_floor(ret_flow, floor_flow).items()
                    }
                )

            if proc_rnd is not None:
                rnd_flow = self._flow_loss(proc_rnd, seed)
                out.update({f"RICL/random_{k}": v for k, v in rnd_flow.items()})
                # does *similarity* matter? retrieval (kNN) vs random demos.
                cmp_r = M.compare_to_floor(ret_flow, rnd_flow)
                out["RICL/beats_random"] = float(cmp_r["retrieval_helps"])
                out["RICL/improvement_vs_random"] = cmp_r["mean_improvement"]
                out.update(
                    {
                        f"RICL/vsrandom_{k}": v
                        for k, v in cmp_r.items()
                        if k.startswith("delta_")
                    }
                )
                # does *any* in-context demo help? random vs the k=0 floor.
                if floor_flow is not None:
                    cmp_rf = M.compare_to_floor(rnd_flow, floor_flow)
                    out["RICL/random_helps"] = float(cmp_rf["retrieval_helps"])
                    out["RICL/random_improvement"] = cmp_rf["mean_improvement"]

        # Surface a scalar the trainer logs as the primary validation number.
        for k, v in ret_metrics.items():
            if k.endswith("_loss"):
                out["Valid/action_loss"] = v
                break
        return out, viz
