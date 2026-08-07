"""Shared cam-frame MSE + viz-batch helper for the wrist-frame policy evaluators.

Hoisted in dedup Combine C: :class:`eval.zoo.eval_hpt.HPTEvalVideo` and
:class:`eval.zoo.eval_pi.PIEvalVideo` each carried a byte-for-byte equivalent
"apply the revert transform once, then reuse it for both the cam-frame paired/
final MSE and the viz video" block (HPT named the prediction key
``main_pred_key``, PI named it ``pred_key`` — same ``f"{embodiment}_{ac_key}"``
value). They now both delegate here.

The helper is the cam-frame-transform + paired/final MSE viz sequence only; the
per-evaluator native-frame metrics (Frechet, Reverse-KL, auxiliary/shared heads)
stay in each evaluator.
"""

import copy

from egomimic.rldb.embodiment.embodiment import Embodiment


def cam_frame_mse_and_viz_batches(
    *,
    transform_list,
    pred_key,
    ac_key,
    _batch,
    preds,
    metrics,
    mse,
):
    """Revert wrist-frame actions to cam frame once; emit cam MSE + viz batches.

    Mutates ``metrics`` in place (adds ``Valid/{pred_key}_cam_paired_mse_avg``
    and ``Valid/{pred_key}_cam_final_mse_avg`` when a transform is configured and
    ``pred_key`` is present in ``preds``) and returns the GT / prediction batches
    to hand to ``_visualize_preds``.

    Args:
        transform_list: ``list[Transform] | None`` for this embodiment
            (``self.transform_lists.get(embodiment_name)``).
        pred_key (str): prediction key ``f"{embodiment_name}_{ac_key}"``.
        ac_key (str): action key into ``_batch`` / ``preds``.
        _batch (dict): the (already-unnormalized) GT batch for this embodiment.
        preds (dict): the model's eval predictions.
        metrics (dict): mutated in place with the cam-frame MSE entries.
        mse: a ``torchmetrics.MeanSquaredError`` instance.

    Returns:
        (gt_batch_viz, preds_for_viz): the batches for ``_visualize_preds``. When
        no transform applies these are exactly ``_batch`` and ``preds``.
    """
    gt_batch_viz = _batch
    preds_for_viz = preds
    if transform_list is not None and pred_key in preds:
        pred_batch = copy.deepcopy(_batch)
        pred_batch[ac_key] = preds[pred_key]
        gt_t = Embodiment.apply_transform(_batch, transform_list)
        pred_t = Embodiment.apply_transform(pred_batch, transform_list)
        # apply_transform drops keys whose shape[0] != batch_size
        # (e.g. ``embodiment``, ``annotations``). Merge to preserve them.
        gt_batch_viz = {**_batch, **gt_t}
        pred_batch_viz = {**_batch, **pred_t}

        # ``.contiguous()`` because ``apply_transform`` returns CPU tensors,
        # so ``.cpu()`` here is a no-op and ``[:, -1]`` leaves a non-contiguous
        # view that torchmetrics' MSE doesn't accept.
        metrics[f"Valid/{pred_key}_cam_paired_mse_avg"] = mse(
            pred_batch_viz[ac_key].cpu().contiguous(),
            gt_batch_viz[ac_key].cpu().contiguous(),
        )
        metrics[f"Valid/{pred_key}_cam_final_mse_avg"] = mse(
            pred_batch_viz[ac_key][:, -1].cpu().contiguous(),
            gt_batch_viz[ac_key][:, -1].cpu().contiguous(),
        )

        preds_for_viz = dict(preds)
        preds_for_viz[pred_key] = pred_batch_viz[ac_key]

    return gt_batch_viz, preds_for_viz
