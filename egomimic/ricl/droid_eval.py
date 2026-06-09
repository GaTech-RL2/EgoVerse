"""Validation for the DROID RICL training run: retrieval vs a true zero-context floor.

The stock :class:`egomimic.eval.pi_ricl_eval.PIRiclEval` receives the *already
processed* batch (the prompt has the retrieved demos already spliced in), so its
``strip_ricl_keys`` floor still carries the demo *text* — not a clean zero-context
baseline. Here we instead hand the evaluator the **raw** batch (see
:class:`DroidRiclModelWrapper`) and process each condition separately:

  - retrieval : process the full raw batch  -> demos spliced + retrieved images
  - random    : process the raw batch after derangement-permuting the per-query
                ``ricl_retrieved_*`` neighbor blocks across the batch -> every
                query is spliced with *another* query's k real demos (same splice
                path, neighbors no longer kNN-matched). Isolates whether retrieval
                *quality* (similarity) matters, not just the presence of context.
  - floor     : process the raw batch with ``ricl_*`` keys removed -> base pi0.5,
                no demo text, no retrieved images (a genuine k=0 floor)

So floor < random checks "does any in-context demo help?"; retrieval < random
checks "does it matter that the demos are visually *similar*?" -- the real RICL
claim. All conditions are scored by the flow-matching loss under *identical* sampled
noise/time (seeded per batch), so the delta isolates the conditioning. We use the
flow loss rather than ``forward_eval``'s sampled actions to skip the (very slow)
``torch.compile`` action-sampling path during routine validation — lower loss
with retrieval is the headline "retrieval helps" signal, matching
``RICL/retrieval_helps`` / ``RICL/delta_*`` from :mod:`egomimic.ricl.metrics`.
"""

from __future__ import annotations

import torch

from egomimic.eval.pi_ricl_eval import PIRiclEval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.ricl import droid_data as D
from egomimic.ricl import metrics as M
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _strip_ricl_raw(raw_batch: dict) -> dict:
    """Drop ``ricl_*`` keys from a raw ``{emb_name: {...}}`` batch (true floor)."""
    out = {}
    for emb_name, sub in raw_batch.items():
        if isinstance(sub, dict):
            out[emb_name] = {k: v for k, v in sub.items() if not k.startswith("ricl_")}
        else:
            out[emb_name] = sub
    return out


def _shuffle_ricl_raw(raw_batch: dict, seed: int):
    """Derangement-permute the per-query ``ricl_retrieved_*`` blocks across the batch.

    Each query keeps its own image/state/action *target* but receives **another**
    query's k retrieved demos -> a random-retrieval control: identical splice path,
    neighbors no longer kNN-matched to the query. Returns ``None`` if any embodiment
    has batch size < 2 (a derangement is impossible, so the control is undefined).

    The shift is a cyclic roll (offset in ``[1, B-1]``), which is a guaranteed
    derangement -- no query ever keeps its own neighbors -- and is deterministic
    given ``seed`` so the comparison is reproducible.
    """
    out = {}
    for emb_name, sub in raw_batch.items():
        if not isinstance(sub, dict):
            out[emb_name] = sub
            continue
        ricl_keys = [k for k in sub if k.startswith("ricl_")]
        if not ricl_keys:
            out[emb_name] = dict(sub)
            continue
        ref = sub[ricl_keys[0]]
        B = ref.shape[0]
        if B < 2:
            return None
        offset = 1 + (seed % (B - 1))
        perm = torch.roll(torch.arange(B, device=ref.device), shifts=offset)
        new = dict(sub)
        for k in ricl_keys:
            new[k] = sub[k].index_select(0, perm)
        out[emb_name] = new
    return out


_RICL_FIELDS = ("images", "state", "action", "mask", "dist")


def _has_variant(raw_batch: dict, prefix: str) -> bool:
    return any(
        isinstance(sub, dict) and f"{prefix}images" in sub for sub in raw_batch.values()
    )


def _use_ricl_variant(raw_batch: dict, source_prefix: str) -> dict:
    """Return a batch whose canonical ``ricl_retrieved_*`` keys are taken from
    ``source_prefix`` (e.g. ``ricl_randwithin_retrieved_``), all other ``ricl_*``
    keys dropped. So one set of retrieved blocks is active and the model (which
    only reads ``ricl_retrieved_*``) sees exactly that condition's demos.
    """
    out = {}
    for emb_name, sub in raw_batch.items():
        if not isinstance(sub, dict):
            out[emb_name] = sub
            continue
        new = {k: v for k, v in sub.items() if not k.startswith("ricl_")}
        for f in _RICL_FIELDS:
            src = sub.get(f"{source_prefix}{f}")
            if src is not None:
                new[f"ricl_retrieved_{f}"] = src
        out[emb_name] = new
    return out


class DroidRiclEval(PIRiclEval):
    """Retrieval-vs-floor flow-loss eval over the *raw* batch."""

    def __init__(
        self,
        *args,
        seed_base: int = 1234,
        compute_random: bool = True,
        n_flow_samples: int = 8,
        compute_sampled: bool = False,
        interp_lamda: float = 0.0,
        interp_lamdas=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed_base = seed_base
        self.compute_random = compute_random
        # (#1) average the flow loss over this many noise/time draws to cut the
        # Monte-Carlo variance that otherwise swamps the small retrieval deltas.
        self.n_flow_samples = max(1, int(n_flow_samples))
        # (#2) also score the *sampled* actions (the slow torch.compile path) —
        # the real action-prediction error, not just the flow-matching proxy.
        self.compute_sampled = compute_sampled
        # (#2, action interpolation) blend the sampled action toward the NEAREST
        # retrieved demo's action chunk, weighted by w=exp(-lamda*dist/dist_max)
        # (the continuous-flow analog of ricl_openpi's logit interpolation). This
        # is a SWEEP: each lambda emits its own ``sampled_mse_l{lamda}`` so we can
        # see the MSE-vs-lambda curve in one sampling pass. The headline
        # ``sampled_mse`` is always the model's own (no-interp) prediction, so
        # ``beats_random`` reflects what the trained model does, not the blend.
        # Empty -> no interpolation. Needs the slow sampling path (--compute-sampled).
        if interp_lamdas:
            self.interp_lamdas = tuple(float(x) for x in interp_lamdas if float(x) > 0)
        elif interp_lamda and float(interp_lamda) > 0:
            self.interp_lamdas = (float(interp_lamda),)
        else:
            self.interp_lamdas = ()
        self._batch_idx = 0

    def _flow_loss(self, processed_batch, seed: int) -> dict:
        """Flow-matching loss averaged over ``n_flow_samples`` noise/time draws.

        A single draw of the flow noise/time is a high-variance Monte-Carlo
        estimate of the expected flow loss, and that variance is large relative
        to the ~0.005 retrieval-vs-random deltas we care about. Averaging N
        independent draws cuts the variance ~1/N. Every condition reuses the
        *same* per-draw seeds (``seed + s*P``), so averaging tightens the
        estimate without decoupling the conditions — the retrieval/floor/random
        deltas still isolate the conditioning, not the sampling RNG.
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
                        accum[k] = accum.get(k, 0.0) + float(v)
        return {k: v / n for k, v in accum.items()}

    def _interp_scale(self, proc) -> float:
        """Normalization scale for the interpolation weight: the max finite kNN
        neighbor distance in this batch (mirrors ricl_openpi's global
        ``max_distance`` normalization, computed per-batch from the *retrieval*
        condition so all conditions — incl. the larger-distance random ones —
        share one scale)."""
        for _b in proc.values():
            d = _b.get("ricl_retrieved_dist")
            if d is not None:
                df = d[torch.isfinite(d)]
                if df.numel():
                    return max(float(df.max()), 1e-6)
        return 1.0

    def _sampled_metrics(self, processed_batch, seed: int, *, dist_max: float = 1.0):
        """Sampled-action error: run the flow sampler (``forward_eval`` — the slow
        ``torch.compile`` path) to get *predicted actions*, then MSE / L1 /
        gripper-accuracy vs ground truth. (#2)

        Unlike the flow loss (a proxy on the velocity field), this is the actual
        action-prediction error the original RICL repo's offline eval reports.
        Restricted to DROID's real dims ``0..RAW_DIM-1``: the 24 slot-fill pad
        dims are zero in the target and would only dilute the error. Seeded so
        every condition samples from the same initial noise — the delta isolates
        the conditioning.

        Returns ``(base, sweep)``: ``base`` holds the model's own (no-interp)
        ``sampled_mse``/``l1``/``gripper_acc`` — this is what ``beats_random`` is
        computed from, so the headline reflects the trained model, not a hand-coded
        blend. ``sweep`` holds, for each interpolation ``lambda``, the MSE after
        blending the sampled action toward the NEAREST demo's chunk
        (``w=exp(-lambda*dist/dist_max)``) plus the mean blend weight — used to
        chart whether distance-weighted interpolation adds anything on top.
        """
        algo = self.model
        base: dict[str, float] = {}
        sweep: dict[str, float] = {}
        d = D.RAW_DIM
        torch.manual_seed(seed)
        preds = algo.forward_eval(processed_batch)  # no_grad inside; sampled actions
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
            ga = M.gripper_accuracy(p, g, (D.GRIPPER_SLOT,), 0.0)
            if ga == ga:  # not NaN
                base[f"{name}_sampled_gripper_acc"] = ga
            for lam in self.interp_lamdas:
                p_l, wmean = self._interpolate_toward_neighbor(p, _b, dist_max, lam)
                if p_l is None:
                    continue
                sweep[f"{name}_sampled_mse_l{lam:g}"] = M.cartesian_mse(p_l, g)
                sweep[f"{name}_interp_wmean_l{lam:g}"] = wmean
        return base, sweep

    def _interpolate_toward_neighbor(self, p, _b, dist_max: float, lamda: float):
        """``p <- w*a_nn + (1-w)*p`` with the nearest retrieved demo's action chunk
        ``a_nn`` (8-D, step-aligned) and ``w=exp(-lamda*dist/dist_max)`` (masked).
        Returns ``(blended_p, mean_w)`` or ``(None, 0.0)`` if no neighbors."""
        a_nn = _b.get("ricl_retrieved_action")
        dist = _b.get("ricl_retrieved_dist")
        if a_nn is None or dist is None:
            return None, 0.0
        d = p.shape[-1]
        a0 = a_nn[:, 0].detach().cpu().float()[..., :d]  # (B, H, d)
        d0 = dist[:, 0].detach().cpu().float()  # (B,)
        msk = _b.get("ricl_retrieved_mask")
        m0 = (
            msk[:, 0].detach().cpu().float() if msk is not None else torch.ones_like(d0)
        )
        finite = torch.isfinite(d0)
        w = (
            torch.where(finite, torch.exp(-lamda * d0 / dist_max), torch.zeros_like(d0))
            * m0
        )  # (B,)
        H = min(p.shape[1], a0.shape[1])
        wv = w[:, None, None]
        out = p.clone()
        out[:, :H] = wv * a0[:, :H] + (1.0 - wv) * out[:, :H]
        return out, float(w.mean())

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._batch_idx = batch_idx
        super().on_validation_step(batch, batch_idx, dataloader_idx)

    def compute_metrics_and_viz(self, raw_batch):
        algo = self.model
        seed = self.seed_base + self._batch_idx

        # retrieval: canonical kNN demos (drop any random-variant keys first so the
        # model sees only the kNN block). Score both the (low-variance, #1) flow
        # loss and, if enabled, the sampled-action error (#2).
        proc_ret = algo.process_batch_for_training(
            _use_ricl_variant(raw_batch, "ricl_retrieved_")
        )
        ret = self._flow_loss(proc_ret, seed)
        out = {f"RICL/retrieval_{k}": v for k, v in ret.items()}
        # (#2) distance-weighted interpolation scale: shared across all conditions,
        # computed from the kNN (retrieval) neighbor distances in this batch.
        interp = bool(self.interp_lamdas)
        dmax = self._interp_scale(proc_ret) if interp else 1.0
        ret_s = None
        if self.compute_sampled:
            # base = model's own (no-interp) sampled metrics -> drives beats_random;
            # sweep = per-lambda interpolated MSE -> charts whether interp adds value.
            ret_s, ret_sw = self._sampled_metrics(proc_ret, seed, dist_max=dmax)
            out.update({f"RICL/retrieval_{k}": v for k, v in ret_s.items()})
            out.update({f"RICL/retrieval_{k}": v for k, v in ret_sw.items()})

        flr = flr_s = None
        if self.compute_floor:
            proc_flr = algo.process_batch_for_training(_strip_ricl_raw(raw_batch))
            flr = self._flow_loss(proc_flr, seed)
            out.update({f"RICL/floor_{k}": v for k, v in flr.items()})
            out.update(
                {f"RICL/{k}": v for k, v in M.compare_to_floor(ret, flr).items()}
            )
            if self.compute_sampled:
                # floor batch has no ricl_* keys -> sweep is empty (no neighbors)
                flr_s, flr_sw = self._sampled_metrics(proc_flr, seed, dist_max=dmax)
                out.update({f"RICL/floor_{k}": v for k, v in flr_s.items()})
                out.update({f"RICL/floor_{k}": v for k, v in flr_sw.items()})
                out.update(
                    {
                        f"RICL/sampled_{k}": v
                        for k, v in M.compare_to_floor(ret_s, flr_s).items()
                    }
                )

        def _score_random(tag: str, proc):
            rnd = self._flow_loss(proc, seed)
            out.update({f"RICL/random_{tag}_{k}": v for k, v in rnd.items()})
            # Does *similarity/ranking* matter? kNN retrieval vs this random set.
            cmp_r = M.compare_to_floor(ret, rnd)
            out[f"RICL/beats_random_{tag}"] = float(cmp_r["retrieval_helps"])
            out[f"RICL/improvement_vs_random_{tag}"] = cmp_r["mean_improvement"]
            # Does *this* random set help at all vs the k=0 floor?
            if flr is not None:
                cmp_rf = M.compare_to_floor(rnd, flr)
                out[f"RICL/random_{tag}_helps"] = float(cmp_rf["retrieval_helps"])
            # Same comparisons on the sampled-action error (#2).
            if self.compute_sampled and ret_s is not None:
                rnd_s, rnd_sw = self._sampled_metrics(proc, seed, dist_max=dmax)
                out.update({f"RICL/random_{tag}_{k}": v for k, v in rnd_s.items()})
                out.update({f"RICL/random_{tag}_{k}": v for k, v in rnd_sw.items()})
                cmp_rs = M.compare_to_floor(ret_s, rnd_s)
                out[f"RICL/sampled_beats_random_{tag}"] = float(
                    cmp_rs["retrieval_helps"]
                )
                out[f"RICL/sampled_improvement_vs_random_{tag}"] = cmp_rs[
                    "mean_improvement"
                ]

        if self.compute_random:
            # Proper random controls when the collate provides them (eval-root runs):
            #   within = random demos from the query's OWN new-task group
            #   bank   = random demos from ALL new tasks pooled
            ran = False
            for tag, prefix in (
                ("within", "ricl_randwithin_retrieved_"),
                ("bank", "ricl_randbank_retrieved_"),
            ):
                if _has_variant(raw_batch, prefix):
                    ran = True
                    proc = algo.process_batch_for_training(
                        _use_ricl_variant(raw_batch, prefix)
                    )
                    _score_random(tag, proc)
            # Fallback for same-corpus runs (no variant keys): in-batch permutation.
            if not ran:
                shuffled = _shuffle_ricl_raw(raw_batch, seed)
                if shuffled is not None:
                    _score_random("perm", algo.process_batch_for_training(shuffled))

        # primary scalar the trainer surfaces (the low-variance flow loss)
        for k, v in ret.items():
            if k.endswith("_loss"):
                out["Valid/action_loss"] = v
                break
        return out, {}  # no viz -> EvalVideo writes no videos


class DroidRiclModelWrapper(ModelWrapper):
    """Pass the RAW val batch to the evaluator (so it can build a clean floor) and
    make the fit/validation barriers safe under single-device (non-DDP) runs."""

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.evaluator is None:
            return
        self.evaluator.on_validation_step(batch, batch_idx, dataloader_idx)

    def _maybe_barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_fit_start(self):
        self.model.device = self.device
        self._maybe_barrier()

    def on_validation_end(self):
        if self.evaluator is not None:
            self.evaluator.on_validation_end()
        self._maybe_barrier()
