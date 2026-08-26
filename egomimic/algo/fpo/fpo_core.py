"""Convention-independent FPO + DiPOD math (arXiv 2507.21053, 2606.13795).

The CFM per-sample loss lives in fpo_policy.py (it must match FMPolicy's exact
rectified-flow convention x_t = time*noise + (1-time)*a, target u_t = noise - a).
Here we keep only the pieces that are independent of that convention.

Verified numerically (numpy reference) before deploy:
  - FPO ratio (Eq.6): r = exp(L_old - L_new)  [sign: old - new]
  - PPO-clip both advantage signs; GAE recursion.
"""
from __future__ import annotations
import torch


def fpo_ratio(l_cfm_old: torch.Tensor, l_cfm_new: torch.Tensor) -> torch.Tensor:
    """r = exp(L_CFM^old - L_CFM^new), per action. Same stored (time,noise) for both."""
    return torch.exp(l_cfm_old - l_cfm_new)


def fpo_clip_loss(ratio, advantage, eps_clip: float = 0.05):
    """PPO-clip objective to MINIMIZE: -E[min(r*A, clip(r,1±eps)*A)]."""
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def fpo_ratio_persample(l_old_pp, l_new_pp, clamp: float = 10.0):
    """Per-sample FPO++ ratio (FPO++ Eq.10): exp(l_old_i - l_new_i) PER (tau,eps)
    pair (sum stays outside exp). Shapes [B,Nmc] -> [B,Nmc]. Clamp the exponent
    for numerical safety."""
    return torch.exp((l_old_pp - l_new_pp).clamp(-clamp, clamp))


def aspo_loss(rho_pp, advantage, eps_clip: float = 0.01):
    """FPO++ asymmetric trust region (ASPO, Eq.11-13) to MINIMIZE.
    rho_pp: [B,Nmc] per-sample ratios; advantage: [B].
      Â>=0:  PPO-clip   min(rho*A, clip(rho,1±eps)*A)
      Â<0:   SPO        rho*A - (|A|/(2eps))*(rho-1)^2   (smooth, always pulls rho->1;
                        no zero-gradient dead zone -> prevents the erosion/collapse)
    Sum over the Nmc pairs (Eq.13), mean over the batch."""
    A = advantage.unsqueeze(1)                                    # [B,1]
    pos = torch.minimum(rho_pp * A, torch.clamp(rho_pp, 1.0 - eps_clip, 1.0 + eps_clip) * A)
    neg = rho_pp * A - (A.abs() / (2.0 * eps_clip)) * (rho_pp - 1.0) ** 2
    surr = torch.where(A >= 0, pos, neg)
    return -surr.sum(dim=1).mean()


def dipod_reg_loss(l_cfm_new, beta: float = 0.05):
    """DiPOD on-policy ELBO-tightening reg (Eq.7): +beta * mean(L_CFM) on rollout actions."""
    return beta * l_cfm_new.mean()


def value_loss(values, returns, coef: float = 0.25):
    return coef * torch.nn.functional.mse_loss(values, returns)


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """Standard GAE-lambda. rewards/values/dones: [T,N]; last_values: [N].
    Returns (advantages [T,N], returns [T,N])."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    next_value = last_values
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        next_value = values[t]
    return adv, adv + values


def normalize_adv(adv, eps: float = 1e-8):
    return (adv - adv.mean()) / (adv.std() + eps)
