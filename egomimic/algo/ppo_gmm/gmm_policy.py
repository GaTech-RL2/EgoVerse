"""GMMPolicy: thin RL wrapper around a loaded BC-RNN GMM algo for EXACT-log-prob
PPO (composition; no edits to existing model code).

Why this exists: the flow-matching RL family (FPO/FPO++/DiPOD) was empirically
insufficient on this task (peaks at baseline then erodes) and the off-policy RQL
diverged. A GMM action head gives EXACT log-probs, so a standard, well-understood
PPO ratio exp(logp_new - logp_old) works directly -- no Monte-Carlo CFM surrogate
and no off-policy value bootstrap.

The BC-RNN policy is NOT Markov: its action at obs-step k is conditioned on a
transformer over the rolling window of strided obs since the last rnn_horizon
re-init (TransformerCore.forward(window)[:, -1] == sequential .step over the
window -- verified). So we:
  (a) FREEZE the obs encoder (perception) and capture its (E,)-dim per-frame
      embedding at rollout (cheap to store, exact since the encoder is frozen);
  (b) reconstruct each decision's window from the stored per-episode embeddings
      and re-run the TRAINABLE core + GMM head at update time to recompute an
      exact log-prob of the stored action chunk -> exact PPO ratio.

BatchNorm: the whole net stays in eval() (per-step B=1 rollouts would corrupt
running BN stats -- documented bug). To still get exploration variance + grads
from the GMM (eval mode otherwise forces a hard 1e-4 std via low_noise_eval), we
set gmm_head.low_noise_eval=False so the softplus-std path engages in eval mode.
Eval coverage temporarily restores low_noise_eval=True (the deployable BC
inference behaviour) for a near-deterministic, reproducible measurement.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.algo.fpo.fpo_policy import ValueHead


class GMMPolicy:
    def __init__(self, algo, emb_name="pushshapes_sim", device="cuda",
                 freeze_encoder=True, freeze_core=False, value_hidden=256):
        self.algo = algo
        self.device = torch.device(device)
        self.emb_name = emb_name
        self.emb_id = get_embodiment_id(emb_name)
        self.policy = algo.nets["policy"]            # BCRNNPolicy
        self.encoder = self.policy.obs_encoder       # frozen perception
        self.core = self.policy.lstm                 # TransformerCore (attr named .lstm)
        self.actor_mlp = self.policy.actor_mlp       # Identity for paper-exact
        self.head = self.policy.gmm_head             # GMMActionHead
        self.rnn_horizon = int(self.policy.rnn_horizon)
        self.obs_stride = int(self.policy.obs_stride)
        self.chunk = int(self.policy.chunk_len)
        self.action_dim = int(self.policy.action_dim)
        # action key for (un)normalization -- mirror BCRNN.inference_step exactly
        emb_lower = get_embodiment(self.emb_id).lower()
        self.ac_key = (algo.ac_keys[emb_lower] if emb_lower in algo.ac_keys
                       else algo.ac_keys[self.emb_id])
        # force the softplus-std GMM even in eval mode (frozen BN) -> real
        # exploration variance + gradients during PPO.
        self.head.low_noise_eval = False
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_core = bool(freeze_core)
        if not self.freeze_encoder:
            raise NotImplementedError(
                "GMMPolicy caches frozen-encoder embeddings during rollout; "
                "tuning the encoder would need raw-obs storage + re-encode. "
                "Use freeze_encoder=True (the stable, efficient default)."
            )
        for q in self.encoder.parameters():
            q.requires_grad_(False)
        if self.freeze_core:
            for q in self.core.parameters():
                q.requires_grad_(False)
        self.value = ValueHead(int(self.core.hidden_dim), hidden=value_hidden).to(self.device)

    # ---- obs -> normalized obs dict (single frame, B inferred) ----
    def normalize_obs(self, obs_zarr):
        return self.algo.norm_stats.normalize(obs_zarr, self.emb_id)

    @torch.no_grad()
    def encode_frame(self, obs_norm):
        """normalized single-frame obs dict -> (B, E) frame embedding (frozen)."""
        emb = self.encoder(obs_norm)        # (B, 1, E)
        return emb[:, 0]                    # (B, E)

    def features_from_window(self, window_emb):
        """window_emb (B, L, E) -> h_t (B, hidden) via core + actor_mlp.
        Matches BCRNNPolicy.step exactly: core.forward(window)[:, -1]."""
        out, _ = self.core(window_emb)      # (B, L, hidden)
        return self.actor_mlp(out[:, -1])   # (B, hidden)

    def _dist(self, h_t):
        raw = self.head(h_t)                # (B, C*per_step)
        return self.head._make_dist(raw)    # MixtureSameFamily, batch shape (B, C)

    @torch.no_grad()
    def act(self, window_emb):
        """Rollout: sample a chunk + exact log-prob + value.
        Returns (a_norm (B,C,D), logp (B,), value (B,))."""
        h_t = self.features_from_window(window_emb)
        value = self.value(h_t)                     # (B,)
        dist = self._dist(h_t)
        a = dist.sample()                           # (B, C, D)
        logp = dist.log_prob(a).mean(-1)            # (B,) MEAN per-chunk-position log-prob
        return a, logp, value

    @torch.no_grad()
    def value_of(self, window_emb):
        return self.value(self.features_from_window(window_emb))   # (B,)

    @torch.no_grad()
    def logp_given(self, window_emb, a_norm):
        """Log-prob of a GIVEN action chunk under this policy (frozen-reference use
        for the KL-to-BC anchor). MEAN per-chunk-position to match act()/the ratio."""
        return self._dist(self.features_from_window(window_emb)).log_prob(a_norm).mean(-1)  # (B,)

    def logp_value_entropy(self, window_emb, a_norm):
        """Update: recompute logp_new + value + a differentiable entropy surrogate
        WITH grad. window_emb (B,L,E), a_norm (B,C,D). Returns logp (B,),
        value (B,), entropy (B,). Entropy = mixture-weighted component entropy
        (an upper bound on the true mixture entropy; fully differentiable in the
        scale params -> a clean exploration regularizer)."""
        h_t = self.features_from_window(window_emb)
        value = self.value(h_t)
        dist = self._dist(h_t)
        # MEAN per-chunk-position log-prob (NOT sum): summing over the 8 chunk
        # positions makes the PPO ratio exp(Sum dlogp) hyper-sensitive (one gentle
        # Adam step -> kl~4), so the trust region can't be controlled. The mean is
        # ~8x gentler -> standard PPO target_kl becomes meaningful. MUST match act().
        logp = dist.log_prob(a_norm).mean(-1)                      # (B,)
        probs = dist.mixture_distribution.probs                    # (B, C, M)
        comp_ent = dist.component_distribution.entropy()           # (B, C, M)
        ent = (probs * comp_ent).sum(-1).sum(-1)                   # (B,) sum over M then C
        return logp, value, ent

    def unnormalize_chunk(self, a_norm):
        """a_norm (C, D) -> world actions (C, D) tensor."""
        return self.algo.norm_stats.unnormalize(
            {self.ac_key: a_norm}, self.emb_id
        )[self.ac_key]

    def policy_parameters(self):
        params = list(self.head.parameters())
        if not self.freeze_core:
            params += list(self.core.parameters())
        return [q for q in params if q.requires_grad]

    def state_dict(self):
        return {"policy": self.policy.state_dict(), "value": self.value.state_dict()}
