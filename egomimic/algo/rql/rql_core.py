"""Reversal Q-Learning (RQL, arXiv 2606.17551) core math — convention-agnostic
pieces + the flow-reversal mapped to OUR FMPolicy convention.

FMPolicy flow convention (verified from egomimic/models/heads/fm_policy.py):
  - at flow-time t in [0,1]:  x_t = t*noise + (1-t)*action   (t=1 -> noise, t=0 -> action)
  - velocity field  v(x_t, t, cond) = model(x_t, t, cond) ~= (noise - action)
  - sampling (noise->action) integrates t: 1 -> 0 with  x_{t-dt} = x_t - dt*v
RQL's expanded MDP uses flow-step index f=0..F. We map f -> t_f = f/F is NOT used;
instead we keep FMPolicy's t and discretize into F steps. RQL "f=0 (x_0=noise)"
== our t=1; RQL "f=F (action)" == our t=0.

Reversal (RQL Eq.8): given the data action `a`, integrate the flow BACKWARD to
recover the noise + intermediate states the current policy would pass through:
  start x(t=0)=a, step up t: x_{t+dt} = x_t + dt*v(x_t, t, cond),  dt=1/F  -> x(t=1)=noise.
"""
from __future__ import annotations
import torch


# --------------------------------------------------------------------------- #
# Expectile regression (IQL-style implicit max) — RQL Eq.14/17
# --------------------------------------------------------------------------- #
def expectile_loss(diff, kappa: float = 0.7):
    """ℓ_2^κ(diff) = |κ - 1[diff<0]| * diff^2.  diff = pred - target.
    κ=0.5 -> MSE (SARSA); κ->1 -> implicit max."""
    w = torch.where(diff < 0, (1.0 - kappa), kappa)
    return (w * diff.pow(2)).mean()


# --------------------------------------------------------------------------- #
# Flow reversal + forward (FMPolicy convention). `vfn(x, t, cond) -> velocity`.
# t is a [B] tensor in [0,1]; x is [B, H, D]; cond is the head's global_cond.
# --------------------------------------------------------------------------- #
def reverse_flow(vfn, cond, a, F: int):
    """Given action chunk `a` (=x at t=0), integrate up to t=1 (noise).
    Returns xs: list length F+1, xs[k] = x at t=k/F (xs[0]=a, xs[F]=noise)."""
    dt = 1.0 / F
    x = a
    xs = [x]
    B = a.shape[0]
    for k in range(F):
        t = torch.full((B,), k * dt, device=a.device, dtype=a.dtype)
        x = x + dt * vfn(x, t, cond)
        xs.append(x)
    return xs


def forward_flow(vfn, cond, noise, F: int):
    """Sampling: from t=1 (noise) down to t=0 (action). Returns xs[k] at t=1-k/F."""
    dt = 1.0 / F
    x = noise
    xs = [x]
    B = noise.shape[0]
    for k in range(F):
        t = torch.full((B,), 1.0 - k * dt, device=noise.device, dtype=noise.dtype)
        x = x - dt * vfn(x, t, cond)
        xs.append(x)
    return xs  # xs[F] = action


# --------------------------------------------------------------------------- #
# Pessimistic ensemble target (RQL Eq.17): r + γ ( mean_j V̄_j(s') - ρ std_j V̄_j(s') )
# --------------------------------------------------------------------------- #
def pessimistic_target(reward, gamma, next_v_stack, rho: float, nonterminal):
    """next_v_stack: [K, B] target-V ensemble at the real next state.
    Returns target [B]."""
    mean = next_v_stack.mean(dim=0)
    std = next_v_stack.std(dim=0)
    return reward + gamma * nonterminal * (mean - rho * std)


if __name__ == "__main__":
    # reversal-consistency self-test: forward(reverse(a)) ~= a for a linear vfn
    torch.manual_seed(0)
    B, H, D, F = 4, 8, 2, 10
    a = torch.randn(B, H, D)
    cond = torch.randn(B, 16)
    # rectified-flow velocity is constant = (noise - action); emulate with a fixed field
    fixed = torch.randn(B, H, D)
    def vfn(x, t, c):  # constant velocity (rectified flow) -> reversal must be exact
        return fixed
    xs = reverse_flow(vfn, cond, a, F)
    noise = xs[-1]
    back = forward_flow(vfn, cond, noise, F)[-1]
    err = (back - a).abs().max().item()
    print(f"reversal round-trip max_err={err:.2e} (should be ~0 for constant v)")
    assert err < 1e-4, "reversal/forward inconsistent"
    print("ok")
