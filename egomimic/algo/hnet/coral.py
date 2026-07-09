"""Deep-CORAL alignment loss for the hybrid dual-stream H-Net.

Used to align the embodiment-AGNOSTIC trunk representation ``A_top`` across
embodiments so the shared agnostic stream learns embodiment-invariant structure.
Computed ONLY on the agnostic trunk's output (never on the specific H-Nets).

For a set of embodiments ``e`` each with agnostic features ``A_e`` (N_e tokens x
d), CORAL aligns the second-order (and optionally first-order) feature
statistics pairwise:

    L = mean_{i<j} [ ||Cov(A_i) - Cov(A_j)||_F^2 / (4 d^2)
                     ( + ||mu_i - mu_j||^2  if include_mean ) ]

The ``1/(4 d^2)`` normaliser is the standard Deep-CORAL scaling (Sun & Saenko,
2016). The optional mean term aligns first-order statistics too.
"""

import itertools
from typing import Dict, Optional

import torch


def coral_loss(
    feats_by_emb: Dict[object, Optional[torch.Tensor]],
    include_mean: bool = True,
) -> torch.Tensor:
    """Pairwise Deep-CORAL loss across embodiments.

    Args:
        feats_by_emb: ``{embodiment -> features}`` where each ``features`` is a
            ``(N_e, d)`` (or ``(..., d)``, flattened to ``(N_e, d)``) tensor of
            agnostic representations carrying grad. ``None`` / fewer-than-2-sample
            entries are skipped.
        include_mean: also align first-order means (``+ ||mu_i - mu_j||^2``).

    Returns:
        Scalar tensor (mean over embodiment pairs). Zero (grad-free) when fewer
        than two usable embodiments are present.
    """
    usable = [
        (e, f)
        for e, f in feats_by_emb.items()
        if f is not None and f.reshape(-1, f.shape[-1]).shape[0] >= 2
    ]
    if len(usable) < 2:
        ref = next((f for f in feats_by_emb.values() if f is not None), None)
        dev = ref.device if ref is not None else None
        return torch.zeros((), device=dev)

    cov: Dict[object, torch.Tensor] = {}
    mean: Dict[object, torch.Tensor] = {}
    d = None
    for e, f in usable:
        f = f.reshape(-1, f.shape[-1]).float()  # (N, d), fp32 for stable cov
        d = f.shape[-1]
        n = f.shape[0]
        mu = f.mean(dim=0)  # (d,)
        fc = f - mu
        cov[e] = (fc.t() @ fc) / (n - 1)  # (d, d) unbiased covariance
        mean[e] = mu

    total = None
    npairs = 0
    for a, b in itertools.combinations([e for e, _ in usable], 2):
        term = ((cov[a] - cov[b]) ** 2).sum() / (4.0 * d * d)
        if include_mean:
            term = term + ((mean[a] - mean[b]) ** 2).sum()
        total = term if total is None else total + term
        npairs += 1
    return total / npairs
