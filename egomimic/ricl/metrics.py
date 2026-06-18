"""RICL eval metrics: retrieval-conditioned vs zero-context floor. (P4)

Mirrors ``ricl_openpi/scripts/eval_cross_embodiment.py``: Cartesian action
MSE/L1, gripper accuracy, and a bank-vs-floor comparison (retrieval helps when
loss/MSE drop relative to the k=0 zero-context floor; an optional within-
embodiment run is the oracle ceiling). Import-light (numpy, optional torch) so
the math is unit-testable on a CPU box without the model/data.

eva ``actions_cartesian`` is 14-D bimanual: per ``PI._extract_xyz`` the layout is
right [xyz(3) rot(3) grip(1)] then left [xyz(3) rot(3) grip(1)], so the gripper
dims are indices (6, 13).
"""

from __future__ import annotations

import numpy as np

EVA_GRIPPER_INDICES = (6, 13)


def _np(x) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _is_num(x) -> bool:
    try:
        float(_np(x).reshape(-1)[0]) if _np(x).ndim else float(x)
        return _np(x).size == 1
    except Exception:
        return False


def cartesian_mse(pred, gt) -> float:
    p, g = _np(pred), _np(gt)
    return float(np.mean((p - g) ** 2))


def cartesian_l1(pred, gt) -> float:
    p, g = _np(pred), _np(gt)
    return float(np.mean(np.abs(p - g)))


def gripper_accuracy(
    pred, gt, gripper_indices=EVA_GRIPPER_INDICES, threshold: float = 0.0
) -> float:
    """Binarized agreement on the gripper dims (open/closed) at ``threshold``."""
    p, g = _np(pred), _np(gt)
    idx = list(gripper_indices)
    if p.shape[-1] <= max(idx):
        return float("nan")  # not the 14-D bimanual layout; skip
    return float(np.mean((p[..., idx] > threshold) == (g[..., idx] > threshold)))


def compare_to_floor(
    retrieval: dict,
    floor: dict,
    lower_is_better=("mse", "loss", "l1"),
) -> dict:
    """Per-key deltas (retrieval - floor) + a headline 'retrieval_helps' summary.

    For metrics where lower is better, a negative delta means retrieval improved
    over the zero-context floor. ``mean_improvement`` is the average drop (positive
    = better) across those metrics.
    """
    out = {}
    improvements = []
    for k, rv in retrieval.items():
        if k in floor and _is_num(rv) and _is_num(floor[k]):
            d = float(_np(rv).reshape(-1)[0]) - float(_np(floor[k]).reshape(-1)[0])
            out[f"delta_{k}"] = d
            if any(s in k.lower() for s in lower_is_better):
                improvements.append(-d)  # positive = retrieval better
    out["retrieval_helps"] = bool(improvements) and float(np.mean(improvements)) > 0.0
    out["mean_improvement"] = float(np.mean(improvements)) if improvements else 0.0
    return out


def strip_ricl_keys(batch: dict) -> dict:
    """Return a copy of a processed batch with ``ricl_*`` keys removed (k=0 floor).

    Works on the per-embodiment processed batch ``{embodiment_id: {key: tensor}}``.
    PIRicl with no ``ricl_*`` keys behaves exactly like the base PI (zero-context).
    """
    out = {}
    for emb_id, sub in batch.items():
        if isinstance(sub, dict):
            out[emb_id] = {k: v for k, v in sub.items() if not k.startswith("ricl_")}
        else:
            out[emb_id] = sub
    return out


def shuffle_ricl_keys(batch: dict, seed: int):
    """Derangement-permute the per-query ``ricl_retrieved_*`` blocks across the batch.

    A random-retrieval *control*: each query keeps its own target but receives
    **another** query's k retrieved demos -> the same splice path, neighbors no
    longer kNN-matched to the query. Comparing retrieval (kNN) vs random isolates
    whether retrieval *quality* (similarity) matters, not just the presence of
    in-context context.

    Call this on the **raw** ``{emb_name: {key: tensor}}`` batch *before*
    ``process_batch_for_training`` (like :func:`strip_ricl_keys`), so the shuffled
    demos get spliced/tokenized into the prompt. Returns ``None`` if any embodiment
    has batch size < 2 (a derangement is impossible, so the control is undefined).

    The shift is a cyclic roll (offset in ``[1, B-1]``) -- a guaranteed derangement
    (no query keeps its own neighbors) and deterministic given ``seed`` so the
    comparison is reproducible.
    """
    import torch

    out = {}
    for emb_name, sub in batch.items():
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
