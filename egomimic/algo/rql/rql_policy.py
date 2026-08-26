"""RQLPolicy — Reversal Q-Learning wrapper around a loaded HPT FMPolicy.

Reuses FPOPolicy for obs->features (frozen HPT trunk) and the FMPolicy velocity
net as the flow policy `v`. Adds an expanded-state V-ensemble critic over
(pooled_features, partial-action x, flow-time t) and the RQL critic/actor losses.

Flow convention = FMPolicy's (see rql_core): t in [0,1], t=1 noise, t=0 action.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from egomimic.algo.fpo.fpo_policy import FPOPolicy
from egomimic.algo.rql import rql_core as R


class VEnsemble(nn.Module):
    """K MLPs over [pooled_feat, x_flat, t] -> scalar V (expanded-state value)."""
    def __init__(self, feat_dim, act_flat, K=10, hidden=512):
        super().__init__()
        self.K = K
        din = feat_dim + act_flat + 1
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, 1)) for _ in range(K)
        ])

    def forward(self, pooled, x_flat, t):  # -> [K, B]
        inp = torch.cat([pooled, x_flat, t.reshape(-1, 1)], dim=-1)
        return torch.stack([n(inp).squeeze(-1) for n in self.nets], dim=0)


class RQLPolicy:
    def __init__(self, algo, emb_name="pushshapes_sim", device="cuda", F=10, K=10, vhidden=512):
        self.fpo = FPOPolicy(algo, emb_name=emb_name, device=device)
        self.algo = algo; self.device = torch.device(device)
        self.head = self.fpo.head; self.chunk = self.fpo.chunk; self.action_dim = self.fpo.action_dim
        self.F = F; self.K = K; self.vhidden = vhidden
        self.value = None; self.value_tgt = None  # built lazily (need feat dim)

    # ---- obs -> features (reuse FPOPolicy) ----
    def features(self, obs_zarr):
        return self.fpo._features(self.fpo.build_data(obs_zarr))   # [B,T,d]

    def cond_for_head(self, feats):
        # match FMPolicy preprocess pooling
        return feats.mean(dim=1) if self.fpo.pooling == "mean" else feats

    def pooled(self, feats):
        return feats.mean(dim=1) if feats.dim() == 3 else feats     # critic input [B,d]

    def _ensure_value(self, feats):
        if self.value is None:
            fd = self.pooled(feats).shape[-1]
            af = self.chunk * self.action_dim
            self.value = VEnsemble(fd, af, self.K, self.vhidden).to(self.device)
            import copy
            self.value_tgt = copy.deepcopy(self.value).to(self.device)
            for p in self.value_tgt.parameters(): p.requires_grad_(False)

    def vfn(self, cond):
        """Returns a velocity closure v(x,t)->velocity using the FMPolicy net."""
        def f(x, t, _c=None):
            return self.head.model(x, t, cond)
        return f

    def V(self, feats, x, t, target=False):
        net = self.value_tgt if target else self.value
        return net(self.pooled(feats), x.reshape(x.shape[0], -1), t)   # [K,B]

    # ---- RQL critic loss (Eq.14/17): expectile, pessimistic-ensemble target ----
    def critic_loss(self, obs, a_norm, reward, next_obs, nonterminal, gamma, kappa, rho):
        feats = self.features(obs)
        cond = self.cond_for_head(feats)
        B = a_norm.shape[0]
        # reverse the flow from the data action to get the intra-flow trajectory
        with torch.no_grad():
            xs = R.reverse_flow(self.vfn(cond), cond, a_norm, self.F)  # list F+1, xs[k] at t=k/F
        # sample one flow step k per sample
        k = torch.randint(0, self.F, (B,), device=self.device)
        x_k = torch.stack([xs[k[i]][i] for i in range(B)], dim=0)
        t_k = k.float() / self.F
        self._ensure_value(feats)
        pred = self.V(feats, x_k, t_k)                                 # [K,B]
        with torch.no_grad():
            nfeats = self.features(next_obs)
            noise = torch.randn(B, self.chunk, self.action_dim, device=self.device)
            t1 = torch.ones(B, device=self.device)                     # reset-flow state t=1
            nv = self.V(nfeats, noise, t1, target=True)                # [K,B]
            tgt = R.pessimistic_target(reward, gamma, nv, rho, nonterminal)  # [B]
        # expectile per-ensemble-member
        loss = R.expectile_loss(pred - tgt.unsqueeze(0), kappa)
        return loss, float(pred.mean())

    # ---- RQL actor loss (Eq.15): maximize V one flow-step ahead + alpha*BC ----
    def actor_loss(self, obs, a_norm, alpha, n_mc=4):
        feats = self.features(obs)
        cond = self.cond_for_head(feats)
        B = a_norm.shape[0]
        with torch.no_grad():
            xs = R.reverse_flow(self.vfn(cond), cond, a_norm, self.F)
        k = torch.randint(0, self.F, (B,), device=self.device)
        x_k = torch.stack([xs[k[i]][i].detach() for i in range(B)], dim=0)
        t_k = k.float() / self.F
        v = self.head.model(x_k, t_k, cond)                # policy velocity (grad flows to head)
        x_next = x_k - (1.0 / self.F) * v                  # one step toward action (t decreases)
        t_next = (t_k - 1.0 / self.F).clamp(min=0.0)
        self._ensure_value(feats)
        q = self.V(feats, x_next, t_next).mean(dim=0)      # [B] (value frozen for actor)
        # behavior-cloning anchor (reuse FMPolicy CFM loss on the data action)
        data = self.fpo.build_data(obs); data = dict(data); data["action"] = a_norm
        time = torch.rand(B, n_mc, device=self.device) * 0.999 + 0.001
        noise = torch.randn(B, n_mc, self.chunk, self.action_dim, device=self.device)
        l_bc, _ = self.fpo.cfm_per_sample_loss(self.fpo.build_data(obs), a_norm, time, noise)
        return (-q.mean()) + alpha * l_bc.mean(), float(q.mean())

    def soft_update(self, tau=0.005):
        with torch.no_grad():
            for p, pt in zip(self.value.parameters(), self.value_tgt.parameters()):
                pt.mul_(1 - tau).add_(tau * p)

    def trainable_policy(self):
        return [q for q in self.head.parameters()]
