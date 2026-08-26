"""FPOPolicy: thin RL wrapper around a loaded HPT FMPolicy algo (composition,
no edits to existing code). Owns: obs->data building, action-chunk sampling,
the FPO per-sample CFM loss with injected (time, noise), and a value head.

Reuses, verbatim from the existing model:
  - algo.norm_stats (normalize / unnormalize)
  - algo.nets["policy"].forward_features(emb, data)  -> trunk tokens (global_cond)
  - head.preprocess_compute_loss(features, data)      -> (actions, pooled_cond)
  - head.model(x_t, time, cond)                        -> velocity (CFM net)
  - head(features) / head.inference(noise, cond)       -> sampled chunk
  - algo._robomimic_to_hpt_data(...)                   -> HPT data dict
"""
from __future__ import annotations
import torch
import torch.nn as nn

from egomimic.rldb.embodiment.embodiment import get_embodiment_id, get_embodiment


class ValueHead(nn.Module):
    """Critic on pooled trunk features (FPO uses a standard separate value net)."""
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled):  # [B, in_dim] -> [B]
        return self.net(pooled).squeeze(-1)


class FPOPolicy:
    def __init__(self, algo, emb_name="pushshapes_sim", device="cuda", value_hidden=256):
        self.algo = algo
        self.device = torch.device(device)
        self.emb_name = emb_name
        self.emb_id = get_embodiment_id(emb_name)
        self.policy = algo.nets["policy"]            # HPTModel
        self.head = self.policy.heads[emb_name]      # FMPolicy
        self.chunk = int(self.head.action_horizon)
        self.ac_key = algo.ac_keys[emb_name]
        self.cam_keys = algo.camera_keys[self.emb_id]
        self.proprio_keys = algo.proprio_keys[self.emb_id]
        # action_dim (FMPolicy exposes infer_ac_dims)
        if hasattr(self.head, "infer_ac_dims"):
            self.action_dim = int(self.head.infer_ac_dims[emb_name])
        else:
            self.action_dim = int(getattr(self.head, "output_dim", 2))
        self.pooling = getattr(self.head, "pooling", "mean")
        self.value = None  # built lazily once we know feature dim

    # ---- obs -> HPT data dict (mirrors HPTAlgo.inference_step replan branch) ----
    def build_data(self, obs_zarr):
        """obs_zarr: dict of [B,...] tensors (B inferred). Returns HPT data dict."""
        obs_norm = self.algo.norm_stats.normalize(obs_zarr, self.emb_id)
        B = next(iter(obs_norm.values())).shape[0]
        robo = dict(obs_norm)
        robo[self.ac_key] = torch.zeros(B, self.chunk, self.action_dim, device=self.device)
        robo["pad_mask"] = torch.ones(B, self.chunk, 1, device=self.device)
        robo["embodiment"] = torch.tensor([self.emb_id] * B, device=self.device, dtype=torch.int64)
        return self.algo._robomimic_to_hpt_data(
            robo, self.cam_keys, self.proprio_keys, [], self.ac_key, [],
        )

    def _features(self, data):
        feats, _ = self.policy.forward_features(self.emb_name, data)
        return feats

    def _pool(self, feats):
        # critic pooling: collapse the token dim to a fixed-size vector
        # (independent of the policy head's own cond pooling, which the CFM
        # loss handles via head.preprocess_compute_loss).
        return feats.mean(dim=1) if feats.dim() == 3 else feats

    def _ensure_value(self, pooled):
        if self.value is None:
            self.value = ValueHead(pooled.shape[-1]).to(self.device)

    @torch.no_grad()
    def sample(self, data, explore_std=0.0):
        """Sample a normalized action chunk + value at rollout time.
        explore_std>0 adds Gaussian exploration noise to the (normalized) chunk —
        the EXECUTED+RECORDED action, so FPO trains on-policy on the explored
        action. Lets the policy discover solutions on hard inits it deterministically
        fails. Use explore_std=0 for eval.
        Returns (a_norm [1,H,D], value scalar, chunk_world [H,D] np)."""
        feats = self._features(data)
        pooled = self._pool(feats)
        self._ensure_value(pooled)
        value = self.value(pooled)
        # HPTModel wraps (features, domain) for diffusion heads before calling them
        a_norm = self.head((feats, self.emb_name))      # samples via inference()
        a_norm = a_norm[:, : self.chunk, : self.action_dim]
        if explore_std > 0:
            a_norm = a_norm + explore_std * torch.randn_like(a_norm)
        chunk_world = self.algo.norm_stats.unnormalize(
            {self.ac_key: a_norm.squeeze(0)}, self.emb_id
        )[self.ac_key]
        return a_norm.detach(), float(value.item()), chunk_world.detach().cpu().numpy()

    @torch.no_grad()
    def value_of(self, data):
        feats = self._features(data)
        pooled = self._pool(feats)
        self._ensure_value(pooled)
        return float(self.value(pooled).item())

    # ---- FPO per-sample CFM loss with injected (time, noise) (FPO Eq.7-9) ----
    def cfm_per_sample_loss(self, data, a_norm, time, noise):
        """data: HPT data dict (B inputs). a_norm [B,H,D]; time [B,Nmc]; noise [B,Nmc,H,D].
        Returns PER-MC-PAIR CFM losses [B,Nmc] (for the FPO++ per-sample ratio) AND
        the value estimate [B] (from the same features pass, with grad). Average over
        dim=1 for the plain-FPO averaged loss."""
        feats = self._features(data)                    # [B,T,d]
        pooled = self._pool(feats)
        self._ensure_value(pooled)
        value = self.value(pooled)                      # [B]
        # reuse the head's exact action reshape + cond pooling
        data = dict(data)
        data["action"] = a_norm
        actions, cond = self.head.preprocess_compute_loss(feats, data)  # [B,H,D],[B,d]
        B, Nmc = time.shape
        H, D = actions.shape[1], actions.shape[2]
        a = actions.unsqueeze(1).expand(B, Nmc, H, D).reshape(B * Nmc, H, D)
        t = time.reshape(B * Nmc)
        eps = noise.reshape(B * Nmc, H, D)
        # cond may be [B, d] (pooled) or [B, T, d] (token seq) — expand rank-agnostically
        cond_e = cond.unsqueeze(1).expand(B, Nmc, *cond.shape[1:]).reshape(B * Nmc, *cond.shape[1:])
        t_exp = t.view(-1, 1, 1)
        x_t = t_exp * eps + (1.0 - t_exp) * a           # FMPolicy convention
        u_t = eps - a
        v_t = self.head.model(x_t, t, cond_e)
        # MEAN over chunk*dim (matches the model's training MSE reduction). Using
        # sum() inflates the loss ~16x -> exp(l_old-l_new) explodes (rho>>1). The
        # FPO ratio must be on the model's native (mean) likelihood scale.
        per = ((v_t - u_t) ** 2).reshape(B, Nmc, -1).mean(dim=-1)  # [B,Nmc] per-pair CFM loss
        return per, value

    def trainable_parameters(self):
        params = list(self.policy.parameters())
        if self.value is not None:
            params += list(self.value.parameters())
        return params
