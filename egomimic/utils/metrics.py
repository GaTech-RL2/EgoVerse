"""Distribution-distance metrics used by evaluation.

Moved out of egomimicUtils; code unchanged.
"""

import math

import torch

# ---- moved from egomimicUtils.py (code unchanged) ----

def reverse_kl_from_samples(pred_samples, targets):
    M, B, T, D = pred_samples.shape

    TD = T * D
    const = -0.5 * TD * math.log(2.0 * math.pi)

    A = pred_samples.permute(1, 0, 2, 3).reshape(B, M, TD)  # (B,M,TD)
    MU = targets.reshape(B, 1, TD)  # (B,1,TD)

    d2 = torch.cdist(A, A).pow(2)  # (B,M,M)
    log_q_each = torch.logsumexp(const - 0.5 * d2, dim=-1) - math.log(M)  # (B,M)

    d2p = ((A - MU) ** 2).sum(dim=-1)  # (B,M)
    log_p_each = const - 0.5 * d2p  # (B,M)

    rkl_each = (log_q_each - log_p_each).mean(dim=-1)  # (B,)
    return rkl_each.mean()

def frechet_gaussian_over_time(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    squared: bool = False,
    return_stats: bool = False,
    eps: float = 1e-6,
):
    """
    Gaussian Fréchet (Bures / 2-Wasserstein) distance between the empirical
    time-distributions of pred and tgt, computed per sample.

    Args:
        pred: Tensor of shape (B, T, D) or (B, T, ...).
        tgt : Tensor of shape (B, T, D) or (B, T, ...).
        squared: If True, return W2^2 instead of W2.
        return_stats: If True, also return {'avg','min','max'} over batch.
        eps: Small jitter for numerical stability (eigenvalue clamp & cov reg).

    Returns:
        dist: Tensor (B,) of per-sample distances (W2 or W2^2).
        stats (optional): {'avg': float, 'min': float, 'max': float}
    """
    assert pred.shape[:2] == tgt.shape[:2], "pred/tgt must match in (B,T)"
    B, T = pred.shape[:2]

    pred = pred.to(torch.float32)
    tgt = tgt.to(torch.float32)
    if pred.ndim > 3:
        D = int(torch.tensor(pred.shape[2:], device=pred.device).prod().item())
        X = pred.reshape(B, T, D)
        Y = tgt.reshape(B, T, D)
    else:
        _, _, D = pred.shape
        X, Y = pred, tgt

    # Means
    m1 = X.mean(dim=1)  # (B,D)
    m2 = Y.mean(dim=1)  # (B,D)

    # Covariances
    if T <= 1:
        C1 = torch.zeros(B, D, D, device=X.device, dtype=X.dtype)
        C2 = torch.zeros(B, D, D, device=X.device, dtype=X.dtype)
    else:
        Xc = X - m1.unsqueeze(1)
        Yc = Y - m2.unsqueeze(1)
        C1 = (Xc.transpose(1, 2) @ Xc) / (T - 1)
        C2 = (Yc.transpose(1, 2) @ Yc) / (T - 1)

    eye = torch.eye(D, device=X.device, dtype=X.dtype).expand(B, D, D)
    C1 = C1 + eps * eye
    C2 = C2 + eps * eye

    # Symmetric sqrt via eigendecomp
    w2, V2 = torch.linalg.eigh(C2)
    w2 = torch.clamp(w2, min=eps)
    C2_sqrt = V2 @ torch.diag_embed(torch.sqrt(w2)) @ V2.transpose(-1, -2)

    inner = C2_sqrt @ C1 @ C2_sqrt
    w_inner, V_inner = torch.linalg.eigh(inner.to(torch.float32))
    w_inner = torch.clamp(w_inner, min=eps)
    inner_sqrt = (
        V_inner @ torch.diag_embed(torch.sqrt(w_inner)) @ V_inner.transpose(-1, -2)
    )

    # Fréchet formula
    mean_term = (m1 - m2).pow(2).sum(dim=1)  # (B,)
    trace_term = (
        C1.diagonal(dim1=-2, dim2=-1).sum(-1)
        + C2.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 2.0 * inner_sqrt.diagonal(dim1=-2, dim2=-1).sum(-1)
    )
    w2_sq = torch.clamp(mean_term + trace_term, min=0.0)
    dist = w2_sq if squared else torch.sqrt(w2_sq)

    if return_stats:
        return dist, {
            "avg": dist.mean().item(),
            "min": dist.min().item(),
            "max": dist.max().item(),
        }
    return dist
