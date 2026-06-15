"""Validation-loss equivalence between the Pi and HPT evaluators.

Both flow-matching configs (``pi0.5_bc_eva``, ``hpt_bc_flow_eva``) compute the *same*
validation loss: flow-matching velocity MSE, i.e. the training loss under ``no_grad``.
HPT's ``bc_flow`` head is an ``FMPolicy`` (``hpt_bc_flow_eva.yaml``), so its
``compute_loss`` matches Pi's ``forward``. The loss-*value* equivalence is proven in
``egomimic/models/test_fm_loss_equivalence.py`` (the shared flow-matching formula).

This module pins the *validation* side. ``PIEvalVideo`` and ``HPTEvalVideo``
(``egomimic/eval/eval_pi.py`` and ``eval_hpt.py``) carry a duplicated per-embodiment
loss-aggregation convention:

    metrics["Valid/{emb}_loss"]  = preds["{emb}_loss"]          # per embodiment
    metrics["Valid/action_loss"] = mean of the per-embodiment losses

Given the same algo predictions, both evaluators must emit identical ``Valid/*loss``
metrics. We drive the *real* ``compute_metrics_and_viz`` of both with a mocked algo
that returns only the loss keys (no sampled-action keys), which exercises exactly the
loss-aggregation path and skips the per-key MSE / Fréchet / reverse-KL / viz machinery.
"""

import types

import torch

from egomimic.eval.eval_hpt import HPTEvalVideo
from egomimic.eval.eval_pi import PIEvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id

EVA = get_embodiment_id("eva_bimanual")
ARIA = get_embodiment_id("aria_bimanual")


def _mock_algo(losses_by_emb):
    """Algo stand-in whose ``forward_eval`` returns only ``{emb}_loss`` keys.

    Without sampled-action prediction keys, every ``if pred_key in preds`` branch in
    both evaluators is skipped, so only the loss aggregation runs. The HPT-only
    attributes are set so its optional (aux / shared / reverse-KL) branches no-op.
    """
    preds = {
        f"{get_embodiment(eid).lower()}_loss": torch.tensor(v, dtype=torch.float32)
        for eid, v in losses_by_emb.items()
    }
    return types.SimpleNamespace(
        forward_eval=lambda batch: preds,
        norm_stats=types.SimpleNamespace(unnormalize=lambda b, eid: b),
        ac_keys={eid: "actions_cartesian" for eid in losses_by_emb},
        shared_ac_key=None,
        auxiliary_ac_keys={},
        rkl_samples=1,
        device=torch.device("cpu"),
    )


def _loss_metrics(evaluator_cls, losses_by_emb):
    evaluator = evaluator_cls()
    evaluator.model = _mock_algo(losses_by_emb)
    # _batch contents are unused once the sampled-action keys are absent.
    batch = {eid: {} for eid in losses_by_emb}
    metrics, _ = evaluator.compute_metrics_and_viz(batch, do_viz=False)
    return {k: v for k, v in metrics.items() if k.endswith("loss")}


def _assert_metrics_equal(a, b):
    assert set(a) == set(b), (set(a), set(b))
    for k in a:
        torch.testing.assert_close(torch.as_tensor(a[k]), torch.as_tensor(b[k]))


def test_pi_hpt_val_loss_metrics_match_single_embodiment():
    losses = {EVA: 0.7}
    pi = _loss_metrics(PIEvalVideo, losses)
    hpt = _loss_metrics(HPTEvalVideo, losses)

    # Same keys + values across the two evaluators ...
    _assert_metrics_equal(pi, hpt)
    # ... and the expected convention: per-embodiment loss + the aggregate.
    assert set(pi) == {"Valid/eva_bimanual_loss", "Valid/action_loss"}
    torch.testing.assert_close(pi["Valid/eva_bimanual_loss"], torch.tensor(0.7))
    torch.testing.assert_close(pi["Valid/action_loss"], torch.tensor(0.7))


def test_pi_hpt_val_loss_metrics_match_multi_embodiment():
    losses = {EVA: 0.4, ARIA: 1.0}
    pi = _loss_metrics(PIEvalVideo, losses)
    hpt = _loss_metrics(HPTEvalVideo, losses)

    _assert_metrics_equal(pi, hpt)
    assert set(pi) == {
        "Valid/eva_bimanual_loss",
        "Valid/aria_bimanual_loss",
        "Valid/action_loss",
    }
    # Valid/action_loss is the mean over embodiments.
    torch.testing.assert_close(pi["Valid/action_loss"], torch.tensor((0.4 + 1.0) / 2))


def test_no_action_loss_when_no_per_embodiment_loss():
    """If forward_eval emits no ``{emb}_loss``, neither evaluator emits Valid/action_loss."""

    def _no_loss_algo():
        algo = _mock_algo({EVA: 0.0})
        algo.forward_eval = lambda batch: {}  # drop the loss key
        return algo

    for cls in (PIEvalVideo, HPTEvalVideo):
        evaluator = cls()
        evaluator.model = _no_loss_algo()
        metrics, _ = evaluator.compute_metrics_and_viz({EVA: {}}, do_viz=False)
        assert "Valid/action_loss" not in metrics
