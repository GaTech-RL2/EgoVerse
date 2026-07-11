import copy

import torch
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import Embodiment, get_embodiment


def _normalize_text(s: str) -> str:
    return " ".join(str(s).lower().split())


def _token_f1(pred: str, gt: str) -> float:
    """Whitespace-token multiset F1 (SQuAD-style) between two strings."""
    p, g = _normalize_text(pred).split(), _normalize_text(gt).split()
    if not p or not g:
        return float(p == g)
    common = 0
    g_counts = {}
    for tok in g:
        g_counts[tok] = g_counts.get(tok, 0) + 1
    for tok in p:
        if g_counts.get(tok, 0) > 0:
            g_counts[tok] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_subtask_decodes(pred_texts, gt_lists):
    """Score autoregressively-decoded subtasks against GT candidate lists.

    Per item, the GT is a LIST of valid paraphrases (every annotation span
    active at that frame); the decode counts as correct if it matches ANY of
    them (exact match, whitespace/case-normalized) and token-F1 takes the max
    over candidates. Items with no GT candidates are excluded (the model has
    nothing to hit) — their prevalence shows in the returned ``scored_frac``.

    Returns ``(exact_match, token_f1, scored_frac)``; the first two are 0.0
    when no item was scorable.
    """
    n_scored = 0
    em_sum = 0.0
    f1_sum = 0.0
    for pred, cands in zip(pred_texts, gt_lists):
        if not cands:
            continue
        n_scored += 1
        norm_pred = _normalize_text(pred)
        em_sum += float(any(norm_pred == _normalize_text(c) for c in cands))
        f1_sum += max(_token_f1(pred, c) for c in cands)
    if n_scored == 0:
        return 0.0, 0.0, 0.0
    return em_sum / n_scored, f1_sum / n_scored, n_scored / max(len(pred_texts), 1)


class PIEvalVideo(EvalVideo):
    """
    Eval class for PI models. Per embodiment, computes:
      - val loss (flow-matching loss, same as training; also aggregated as ``Valid/action_loss``)
      - paired/final MSE in the model's native wrist frame
      - paired/final MSE in cam frame, when a ``transform_lists`` entry is configured
    The revert transform is applied once and reused for both the cam-frame MSE
    and the viz video.
    """

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss_embodiments = 0

        for embodiment_id, _batch in batch.items():
            _batch = algo.norm_stats.unnormalize(_batch, embodiment_id)
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            pred_key = f"{embodiment_name}_{ac_key}"
            loss_key = f"{embodiment_name}_loss"

            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                if total_loss is None:
                    total_loss = torch.zeros_like(loss_val)
                total_loss = total_loss + loss_val
                n_loss_embodiments += 1

            # Subtask-prediction (hierarchical) val signals — CE loss + teacher-
            # forced token accuracy. Computed by forward_eval; surfaced here.
            sub_loss_key = f"{embodiment_name}_subtask_loss"
            if sub_loss_key in preds:
                metrics[f"Valid/{sub_loss_key}"] = preds[sub_loss_key]
            sub_acc_key = f"{embodiment_name}_subtask_acc"
            if sub_acc_key in preds:
                metrics[f"Valid/{sub_acc_key}"] = preds[sub_acc_key]

            # Free-running decode vs GT: compare the autoregressively-decoded
            # subtask text against the frame's GT candidate list (any-match).
            # Complements the teacher-forced CE/acc above with the metric that
            # actually reflects inference behavior.
            sub_pred = preds.get(f"{embodiment_name}_subtask_pred")
            gt_lists = _batch.get("gt_subtask_all")
            if sub_pred is not None and gt_lists is not None:
                em, f1, frac = score_subtask_decodes(sub_pred, gt_lists)
                metrics[f"Valid/{embodiment_name}_subtask_decode_exact_match"] = (
                    torch.tensor(em)
                )
                metrics[f"Valid/{embodiment_name}_subtask_decode_token_f1"] = (
                    torch.tensor(f1)
                )
                metrics[f"Valid/{embodiment_name}_subtask_decode_scored_frac"] = (
                    torch.tensor(frac)
                )

            if pred_key in preds:
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )
                metrics[f"Valid/{pred_key}_final_mse_avg"] = mse(
                    preds[pred_key][:, -1].cpu(), _batch[ac_key][:, -1].cpu()
                )

            transform_list = self.transform_lists.get(embodiment_name)
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

            ims = self._visualize_preds(preds_for_viz, gt_batch_viz)
            images_dict[embodiment_id] = ims

        if total_loss is not None and n_loss_embodiments > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss_embodiments

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        embodiment_id = batch["embodiment"][0].item()
        embodiment_name = get_embodiment(embodiment_id).lower()
        return self.viz_func[embodiment_name](predictions, batch)
