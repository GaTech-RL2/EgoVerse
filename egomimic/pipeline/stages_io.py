"""Batch-native I/O + head + loss stages (BATCHFLOW).

ObsEncoders / TargetBuilder / GMMHead / CVAEPosterior / CVAEHead / RatioLoss —
every one is ``stage(batch) -> batch``. The stem modules (ObsToken,
MultiEmbodimentCondEncoder, VisualCore) are reused unchanged (ctx-free).

Head convention: at TRAIN ("target" in batch) a head writes its loss/* and
log/* keys AND "pred_action" (decoded, (T, C, D)); at ROLLOUT (no "target")
it writes only "pred_action". Rollout consumers read decoded ACTIONS — the
old raw-params + head.decode() indirection is gone.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.hnet import ratio_loss_from_aux
from egomimic.pipeline.core import Stage
from egomimic.pipeline import packed


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class ObsEncoders(Stage):
    """A (shared/agnostic) + S (per-emb specific) obs encoders.

    Wraps the existing stem modules: ``agnostic`` = one InputModule (ObsToken)
    shared across embs; ``specific`` = list of InputModules whose contributions
    sum into S (per-emb routing handled inside MultiEmbodimentCondEncoder).
    Writes A, S, time_pos."""

    reads = ["obs/*", "cu_seqlens", "embodiment"]   # actions NOT required (rollout)
    writes = ["A", "S", "time_pos"]

    def __init__(self, agnostic: nn.Module, specific: List[nn.Module]):
        super().__init__()
        self.agnostic = agnostic
        self.specific = nn.ModuleList(specific)

    def forward(self, batch: dict) -> dict:
        obs_packed = {k.split("/", 1)[1]: v for k, v in batch.items()
                      if k.startswith("obs/")}
        ref = next(v for v in obs_packed.values() if torch.is_tensor(v))
        T, dev = ref.shape[0], ref.device
        dt = torch.float32 if not ref.is_floating_point() else ref.dtype
        # actions donor optional: ObsToken reads it only for T/device/dtype.
        actions = batch.get("actions")
        if actions is None:
            actions = torch.zeros((T, 1), device=dev, dtype=dt)
        cu = batch["cu_seqlens"].to(device=dev, dtype=torch.long)
        emb = batch["embodiment"]
        # episode-consistent crop support: stamp the frame-grid cu on any
        # VisualCore configured with crop_scope="episode" (duck-typed; the
        # attribute is consumed inside VisualCore._crop and revalidated by
        # length there, so stale stamps from other embs are harmless).
        for m in self.modules():
            if getattr(m, "crop_scope", None) == "episode":
                m._episode_cu = cu
        kw = dict(actions_packed=actions, obs_packed=obs_packed, cu_seqlens=cu,
                  T_total=T, device=dev, dtype=actions.dtype,
                  embodiment_id=emb)
        S = None
        for mod in self.specific:
            c = mod.forward_packed(**kw)
            S = c if S is None else S + c
        batch["A"] = self.agnostic.forward_packed(**kw)
        batch["S"] = S
        batch["time_pos"] = packed.frame_idx(cu)
        return batch


class SharedObsEncoders(Stage):
    """ONE shared obs encoder feeding BOTH streams (vs ``ObsEncoders``, which
    builds two fully independent encoders each with its own VisualCore).

    Layout::

        shared encoder (images only, 1 copy)
              |
              +-- proj_a   (SHARED)     -> A
              +-- proj_s   (per-emb)  \\
              +-- proprio  (per-emb)  /  + -> S

    A is vision-only and embodiment-agnostic. S stays embodiment-specific
    through a per-embodiment projection of the shared feature PLUS a per-
    embodiment proprio encoder; the two are summed at width ``d_s``.

    Args:
        shared:   InputModule (ObsToken) over IMAGES ONLY, width d_shared.
        proj_a:   Module d_shared -> d_a, shared across embodiments.
        proj_s:   dict[emb_name -> Module d_shared -> d_s], per embodiment.
        proprio:  optional InputModule (ObsToken wrapping a
                  MultiEmbodimentCondEncoder) over PROPRIO ONLY at width d_s.
                  None disables the proprio path (S = projected vision only).

    Writes the same keys as ``ObsEncoders`` so every downstream stage is
    unchanged and the two are drop-in swappable in YAML.
    """

    reads = ["obs/*", "cu_seqlens", "embodiment"]   # actions NOT required (rollout)
    writes = ["A", "S", "time_pos"]

    def __init__(
        self,
        shared: nn.Module,
        proj_a: nn.Module,
        proj_s: Dict[str, nn.Module],
        proprio: Optional[nn.Module] = None,
    ):
        super().__init__()
        if not proj_s:
            raise ValueError("SharedObsEncoders needs a non-empty proj_s dict "
                             "(one entry per embodiment).")
        self.shared = shared
        self.proj_a = proj_a
        self.proj_s = nn.ModuleDict(proj_s)
        self.proprio = proprio

    def forward(self, batch: dict) -> dict:
        obs_packed = {k.split("/", 1)[1]: v for k, v in batch.items()
                      if k.startswith("obs/")}
        ref = next(v for v in obs_packed.values() if torch.is_tensor(v))
        T, dev = ref.shape[0], ref.device
        dt = torch.float32 if not ref.is_floating_point() else ref.dtype
        actions = batch.get("actions")
        if actions is None:
            actions = torch.zeros((T, 1), device=dev, dtype=dt)
        cu = batch["cu_seqlens"].to(device=dev, dtype=torch.long)
        emb = batch["embodiment"]
        # episode-consistent crop support (identical to ObsEncoders)
        for m in self.modules():
            if getattr(m, "crop_scope", None) == "episode":
                m._episode_cu = cu
        kw = dict(actions_packed=actions, obs_packed=obs_packed, cu_seqlens=cu,
                  T_total=T, device=dev, dtype=actions.dtype,
                  embodiment_id=emb)

        feat = self.shared.forward_packed(**kw)          # (T, d_shared)

        if emb not in self.proj_s:
            raise KeyError(
                f"SharedObsEncoders has no proj_s entry for embodiment {emb!r}; "
                f"available: {list(self.proj_s.keys())}."
            )
        A = self.proj_a(feat)                            # (T, d_a)
        S = self.proj_s[emb](feat)                       # (T, d_s)
        if self.proprio is not None:
            p = self.proprio.forward_packed(**kw)        # (T, d_s)
            if p.shape != S.shape:
                raise ValueError(
                    f"SharedObsEncoders: proprio output {tuple(p.shape)} does not "
                    f"match projected-vision S {tuple(S.shape)}; set the proprio "
                    f"encoder's d_cond equal to proj_s out_features."
                )
            S = S + p

        batch["A"] = A
        batch["S"] = S
        batch["time_pos"] = packed.frame_idx(cu)
        return batch


class DeriveVelocity(Stage):
    """Finite-difference velocity as a DERIVED obs key (velocity-ablation arm 2).

    v[t] = x[t] - x[t-1] within an episode, v[0] = 0. Written as a new
    ``obs/<out_key>`` so the obs encoders can consume it like any other proprio
    input.

    WHY derived instead of a new zarr key: ``state_agent_obj`` is already
    emitted by BOTH the training keymap and the eval env->zarr converters
    (``_env_to_zarr_pushshapes`` / ``_usocket``), so deriving from it means
    train and eval cannot disagree about the key -- the failure mode the
    handoff doc warns about. It also needs no norm-stats recompute: this stage
    runs AFTER normalization, so v is a difference of normalized coordinates
    and is already on a sane scale.

    Place it AFTER ``TargetBuilder`` so the difference is taken between
    consecutive TOKENS (stride-decimated), which is the same grid the L0
    attention window sees -- a w=2 window and this key then carry the same
    information, which is the whole point of the ablation.

    Args:
        source:  obs key to difference (default ``state_agent_obj``).
        out_key: obs key to write (default ``agent_vel``).
        dims:    how many leading components of ``source`` to difference
                 (2 = pusher xy; 3 = u_socket pusher x,y,theta).
    """

    reads = ["obs/*", "cu_seqlens"]
    writes = []          # obs/<out_key>; declared dynamically in __init__

    def __init__(self, source: str = "state_agent_obj",
                 out_key: str = "agent_vel", dims: int = 2):
        super().__init__()
        self.source = str(source)
        self.out_key = str(out_key)
        self.dims = int(dims)
        self.reads = [f"obs/{self.source}", "cu_seqlens"]
        self.writes = [f"obs/{self.out_key}"]

    def forward(self, batch: dict) -> dict:
        src = batch.get(f"obs/{self.source}")
        if src is None:
            raise KeyError(
                f"DeriveVelocity: obs/{self.source} not in batch (have "
                f"{[k for k in batch if k.startswith('obs/')]})")
        x = src[..., : self.dims]
        cu = batch["cu_seqlens"].to(device=x.device, dtype=torch.long)
        v = torch.zeros_like(x)
        # packed layout: episodes are contiguous spans [cu[i], cu[i+1]).
        # Difference INSIDE each span so the first token of an episode is 0
        # and no velocity leaks across an episode boundary.
        v[1:] = x[1:] - x[:-1]
        starts = cu[:-1].clamp(min=0, max=max(x.shape[0] - 1, 0))
        v[starts] = 0.0
        batch[f"obs/{self.out_key}"] = v
        return batch


class TargetBuilder(Stage):
    """Stride + targets as DATA (kills the obs_stride dual code path).

    stride == 1: writes "target" (T, C, D) from actions (episode-clamped).
    stride > 1: decimates every packed per-frame key (obs/*, actions, A/S run
    AFTER this stage so only raw keys need decimation), rewrites cu_seqlens /
    max_seq_len / frame_idx to the kept-token space, and builds "target" from
    the ORIGINAL full-rate actions at the kept frames."""

    reads = ["actions", "cu_seqlens"]
    writes = ["target", "frame_idx", "cu_seqlens", "max_seq_len"]

    def __init__(self, chunk_len: int = 4, stride: int = 1):
        super().__init__()
        self.chunk_len, self.stride = int(chunk_len), int(stride)

    def forward(self, batch: dict) -> dict:
        actions = batch["actions"]
        cu = batch["cu_seqlens"].to(device=actions.device, dtype=torch.long)
        full_target = packed.chunk_targets(actions, cu, self.chunk_len)
        if self.stride == 1:
            batch["target"] = full_target
            batch["frame_idx"] = packed.frame_idx(cu)
            return batch
        kept, new_cu = packed.strided_view(cu, self.stride)
        for k in list(batch.keys()):
            if k.startswith("obs/") or k == "actions":
                v = batch[k]
                if torch.is_tensor(v) and v.shape[:1] == (int(cu[-1]),):
                    batch[k] = v[kept]
        batch["target"] = full_target[kept]          # full-rate C-chunks @ kept frames
        batch["frame_idx"] = kept - new_cu[:-1][packed.episode_ids(new_cu)]
        batch["cu_seqlens"] = new_cu
        batch["max_seq_len"] = int(packed.lengths(new_cu).max())
        return batch


# --------------------------------------------------------------------------- #
# GMM head (sym_w1 / prenet_decoder gate arm)
# --------------------------------------------------------------------------- #
class RegressionHead(Stage):
    """Deterministic action head — the GMM-ablation twin. Mirrors GMMHead's
    partition (shared A MLP on a_top + per-emb S MLP on s, additive fusion,
    tanh squash) with the mixture removed. Train: loss/mse. Rollout/eval:
    pred_action (T, C, D)."""

    reads = ["a_top", "s", "embodiment"]
    writes = ["pred_action", "loss/mse", "log/mse", "log/l1"]

    def __init__(self, d_a: int, d_s: int, action_dim: int, chunk_len: int,
                 embodiments: Optional[List[str]] = None,
                 head_hidden_dim: int = 512, head_n_layers: int = 3,
                 squash: bool = True):
        super().__init__()
        self.D, self.C = int(action_dim), int(chunk_len)
        self.squash = bool(squash)
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]
        C, D = self.C, self.D

        def _mlp(d_in, hidden, n, d_out):
            layers, cur = [], d_in
            for _ in range(int(n)):
                layers += [nn.Linear(cur, hidden), nn.GELU()]
                cur = hidden
            layers += [nn.Linear(cur, d_out)]
            return nn.Sequential(*layers)

        self.A_head = _mlp(d_a, head_hidden_dim, head_n_layers, C * D)
        self.S_head = nn.ModuleDict(
            {e: _mlp(d_s, head_hidden_dim, head_n_layers, C * D) for e in embs})

    def forward(self, batch: dict) -> dict:
        a_top, s, emb = batch["a_top"], batch["s"], batch["embodiment"]
        e = emb if emb in self.S_head else next(iter(self.S_head))
        lead = a_top.shape[:-1]
        raw = (self.A_head(a_top) + self.S_head[e](s)).reshape(*lead, self.C, self.D)
        pred = torch.tanh(raw) if self.squash else raw
        if "target" in batch:
            mse = F.mse_loss(pred, batch["target"])
            batch["loss/mse"] = mse
            batch["log/mse"] = float(mse)
            batch["log/l1"] = float(F.l1_loss(pred, batch["target"]))
        if not self.training:
            batch["pred_action"] = pred.detach()
        return batch


class GMMHead(Stage):
    """Partitioned GMM head as one stage: k_agnostic modes from a_top (shared)
    + k_specific modes from s (per-emb) + per-emb gate. Train: loss/nll.
    Rollout: pred_action = mixture sample (low-noise) decoded to (T, C, D)."""

    reads = ["a_top", "s", "embodiment"]
    writes = ["pred_action", "loss/nll", "log/nll"]

    def __init__(self, d_a: int, d_s: int, action_dim: int, chunk_len: int,
                 k_agnostic: int = 16, k_specific: int = 16,
                 embodiments: Optional[List[str]] = None,
                 head_hidden_dim: int = 512, head_n_layers: int = 3,
                 gate_n_layers: int = 3, min_std: float = 1e-4,
                 low_noise_eval: bool = True, squash_means: bool = True):
        super().__init__()
        self.D, self.C = int(action_dim), int(chunk_len)
        self.Ka, self.Ks = int(k_agnostic), int(k_specific)
        self.min_std = float(min_std)
        self.low_noise_eval = bool(low_noise_eval)
        self.squash_means = bool(squash_means)
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]
        C, D = self.C, self.D

        def _mlp(d_in, hidden, n, d_out):
            layers, cur = [], d_in
            for _ in range(int(n)):
                layers += [nn.Linear(cur, hidden), nn.GELU()]
                cur = hidden
            layers += [nn.Linear(cur, d_out)]
            return nn.Sequential(*layers)

        self.A_head = _mlp(d_a, head_hidden_dim, head_n_layers, C * self.Ka * 2 * D)
        self.S_head = nn.ModuleDict(
            {e: _mlp(d_s, head_hidden_dim, head_n_layers, C * self.Ks * 2 * D)
             for e in embs})
        self.gate = nn.ModuleDict(
            {e: _mlp(d_a + d_s, head_hidden_dim, gate_n_layers, C * (self.Ka + self.Ks))
             for e in embs})

    def _dist(self, a_top, s, emb, training: bool):
        C, D, Ka, Ks = self.C, self.D, self.Ka, self.Ks
        M = Ka + Ks
        lead = a_top.shape[:-1]
        e = emb if emb in self.S_head else next(iter(self.S_head))
        a_raw = self.A_head(a_top).reshape(*lead, C, Ka, 2, D)
        s_raw = self.S_head[e](s).reshape(*lead, C, Ks, 2, D)
        means = torch.cat([a_raw[..., 0, :], s_raw[..., 0, :]], dim=-2)
        scales = torch.cat([a_raw[..., 1, :], s_raw[..., 1, :]], dim=-2)
        logits = self.gate[e](torch.cat([a_top, s], -1)).reshape(*lead, C, M)
        if self.squash_means:
            means = torch.tanh(means)
        if self.low_noise_eval and not training:
            scales = torch.full_like(means, 1e-4)
        else:
            scales = F.softplus(scales) + self.min_std
        comp = torch.distributions.Independent(
            torch.distributions.Normal(means, scales), 1)
        mix = torch.distributions.Categorical(logits=logits)
        return torch.distributions.MixtureSameFamily(mix, comp), logits

    def forward(self, batch: dict) -> dict:
        a_top, s, emb = batch["a_top"], batch["s"], batch["embodiment"]
        dist, logits = self._dist(a_top, s, emb, self.training)
        if "target" in batch:
            lp = dist.log_prob(batch["target"])        # (T, C)
            nll = -(lp.sum(-1)).mean()                 # sum over C, mean over T
            batch["loss/nll"] = nll
            batch["log/nll"] = float(nll)
            with torch.no_grad():
                probs = torch.softmax(logits.float(), -1)
                batch["log/w_agnostic"] = float(probs[..., : self.Ka].sum(-1).mean())
        if not self.training:
            with torch.no_grad():
                batch["pred_action"] = dist.sample().clamp(-1.0, 1.0)  # (T, C, D)
        return batch


# --------------------------------------------------------------------------- #
# CVAE stages (v1 head arm + shared by v2 cvae_zs)
# --------------------------------------------------------------------------- #
def _rope_cache(positions, dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device).float() / dim))
    ang = positions.float()[..., None] * inv
    return ang.cos()[:, None], ang.sin()[:, None]


def _rope_rotate(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class _RoPEAttnLayer(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.h, self.dh = int(heads), d // int(heads)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, cos, sin, attn_mask):
        B, N, _ = x.shape
        q, k, v = (self.qkv(self.norm1(x)).reshape(B, N, 3, self.h, self.dh)
                   .permute(2, 0, 3, 1, 4))
        if cos is not None:
            q, k = _rope_rotate(q, cos, sin), _rope_rotate(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.out(o.transpose(1, 2).reshape(B, N, -1))
        return x + self.ffn(self.norm2(x))


class CVAEPosterior(Stage):
    """Trajectory-level posterior q(z | src, act): one z per EPISODE.

    src = the configured stream ("S" default per the design: emb-specific
    fused features; "A" or "h"(cat) as knobs). Train-only by contract (reads
    "target"), so Pipeline.plan() excludes it at rollout. Writes cvae_z (B,32),
    cvae_z_tok (T, 32) broadcast, loss/kl, log/kl_raw."""

    reads = ["S", "target", "cu_seqlens", "embodiment"]
    writes = ["cvae_z", "cvae_z_tok", "loss/kl", "log/kl_raw", "log/kl_dim_max"]

    def __init__(self, src_dim: int, action_dim: int = 2, chunk_len: int = 4,
                 src_key: str = "S", z_dim: int = 32, enc_d_model: int = 256,
                 enc_n_layers: int = 10, enc_n_heads: int = 4, enc_ffn_dim: int = 1024,
                 beta: float = 10.0, free_bits: float = 0.5,
                 embodiments: Optional[List[str]] = None, logvar_clamp: float = 8.0,
                 zero_z_prob: float = 0.0):
        super().__init__()
        E = int(enc_d_model)
        self.src_key = str(src_key)
        self.reads = [self.src_key, "target", "cu_seqlens", "embodiment"]
        self.z_dim, self.beta, self.free_bits = int(z_dim), float(beta), float(free_bits)
        self.zero_z_prob = float(zero_z_prob)
        self.logvar_clamp = float(logvar_clamp)
        self.enc_n_heads = int(enc_n_heads)
        C, D = int(chunk_len), int(action_dim)
        self.obs_proj = nn.Linear(int(src_dim), E)
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]
        self.act_proj = nn.ModuleDict({e: nn.Linear(C * D, E) for e in embs})
        self.type_emb = nn.Parameter(torch.zeros(2, E))
        self.readout = nn.Parameter(torch.zeros(1, E))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)
        self.layers = nn.ModuleList(
            _RoPEAttnLayer(E, enc_n_heads, int(enc_ffn_dim)) for _ in range(int(enc_n_layers)))
        self.norm_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, 2 * self.z_dim)

    def forward(self, batch: dict) -> dict:
        src, target = batch[self.src_key], batch["target"]
        cu = batch["cu_seqlens"].to(device=src.device, dtype=torch.long)
        emb = str(batch["embodiment"])
        T = src.shape[0]
        lens = packed.lengths(cu)
        B, L = int(lens.numel()), int(lens.max())
        E = self.obs_proj.out_features

        e = emb if emb in self.act_proj else next(iter(self.act_proj))
        obs_tok = self.obs_proj(src) + self.type_emb[0]
        act_tok = self.act_proj[e](target.reshape(T, -1)) + self.type_emb[1]

        N = 1 + 2 * L
        seq = src.new_zeros(B, N, E)
        pad = torch.ones(B, N, dtype=torch.bool, device=src.device)
        pos = torch.zeros(B, N, device=src.device)
        seq[:, 0] = self.readout
        pad[:, 0] = False
        for b in range(B):
            s0, e0 = int(cu[b]), int(cu[b + 1])
            n = e0 - s0
            seq[b, 1:1 + n] = obs_tok[s0:e0]
            seq[b, 1 + L:1 + L + n] = act_tok[s0:e0]
            pad[b, 1:1 + n] = False
            pad[b, 1 + L:1 + L + n] = False
            ts = torch.arange(1, n + 1, device=src.device, dtype=pos.dtype)
            pos[b, 1:1 + n] = ts
            pos[b, 1 + L:1 + L + n] = ts   # paired obs/act share the position
        cos, sin = _rope_cache(pos, E // self.enc_n_heads)
        attn_mask = (~pad)[:, None, None, :]
        x = seq
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        mu, logvar = self.head(self.norm_f(x)[:, 0]).chunk(2, -1)
        logvar = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        if self.training and self.zero_z_prob > 0:
            # z->0 sampling: train the canonical (z=0) path the rollout uses —
            # mitigates the trunk train/test z-shift (cvae_zs design).
            keep = (torch.rand(z.shape[0], device=z.device) >= self.zero_z_prob)
            z = z * keep.unsqueeze(-1).to(z.dtype)

        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = torch.clamp(kl_dim.mean(0), min=self.free_bits).sum()
        batch["cvae_z"] = z
        batch["cvae_z_tok"] = packed.broadcast_per_episode(z, cu)
        batch["loss/kl"] = self.beta * kl
        batch["log/kl_raw"] = float(kl_dim.sum(-1).mean())
        batch["log/kl_dim_max"] = float(kl_dim.mean(0).max())
        return batch


class CVAEHead(Stage):
    """Deployed CVAE decoder: broadcast-concat residual mixer + MLP.

    z source = batch["cvae_z_tok"] if present (train) else zeros (rollout
    canonical strategy). Train adds loss/recon + the distance-scaled BAND
    (decoder-only via detach; generous constants)."""

    reads = ["a_top", "s", "cu_seqlens"]
    writes = ["pred_action", "loss/recon", "loss/band", "log/recon", "log/gap",
              "log/zdist", "log/L_use", "log/L_smooth",
              "log/band_use_rate", "log/band_smooth_rate"]

    def __init__(self, d_a: int, d_s: int, action_dim: int = 2, chunk_len: int = 4,
                 z_dim: int = 32, mixer_hidden_dim: int = 1024, mixer_n_layers: int = 3,
                 dec_hidden_dim: int = 1024, dec_n_layers: int = 4,
                 band_weight: float = 1.0, band_c_lo: float = 0.05,
                 band_c_hi: float = 10.0, band_d_cap: float = 6.0,
                 band_stopgrad: bool = True, rollout_z_scale: float = 0.0):
        super().__init__()
        self.D, self.C, self.z_dim = int(action_dim), int(chunk_len), int(z_dim)
        self.d_cond = int(d_a) + int(d_s)
        self.band_weight, self.band_stopgrad = float(band_weight), bool(band_stopgrad)
        self.rollout_z_scale = float(rollout_z_scale)
        self.c_lo, self.c_hi, self.d_cap = float(band_c_lo), float(band_c_hi), float(band_d_cap)
        mh, mn = int(mixer_hidden_dim), max(1, int(mixer_n_layers))
        layers, cur = [], self.d_cond + self.z_dim
        for _ in range(mn):
            layers += [nn.Linear(cur, mh), nn.GELU()]
            cur = mh
        layers += [nn.Linear(cur, self.d_cond)]
        self.mixer = nn.Sequential(*layers)
        dec, cur = [], self.d_cond
        for _ in range(int(dec_n_layers)):
            dec += [nn.Linear(cur, int(dec_hidden_dim)), nn.GELU()]
            cur = int(dec_hidden_dim)
        dec += [nn.Linear(cur, self.C * self.D)]
        self.decoder = nn.Sequential(*dec)

    def _decode(self, h, z_tok):
        mixed = h + self.mixer(torch.cat([h, z_tok], -1))
        out = self.decoder(mixed)
        return out.reshape(*out.shape[:-1], self.C, self.D)

    @staticmethod
    def _l1(pred, target):
        return (pred - target).abs().sum(dim=(-1, -2))  # (T,)

    def forward(self, batch: dict) -> dict:
        h = torch.cat([batch["a_top"], batch["s"]], -1)
        z_tok = batch.get("cvae_z_tok")
        if z_tok is None:
            if not self.training and self.rollout_z_scale > 0:
                # prior-sample ablation: one z ~ N(0, scale^2 I) per forward
                # (rollout is B=1 recompute-over-prefix -> fresh z per replan),
                # broadcast across tokens. Tests whether the z=0 conditional
                # mean is what kills closed-loop coverage. Default 0.0 = z=0.
                z1 = h.new_empty(1, self.z_dim).normal_() * self.rollout_z_scale
                z_tok = z1.expand(*h.shape[:-1], self.z_dim)
            else:
                z_tok = h.new_zeros(*h.shape[:-1], self.z_dim)  # canonical strategy
        pred = self._decode(h, z_tok)
        batch["pred_action"] = pred.clamp(-1.0, 1.0)
        if "target" not in batch:
            return batch

        target = batch["target"]
        l1 = self._l1(pred, target)
        recon = l1.mean()
        batch["loss/recon"] = recon
        batch["log/recon"] = float(recon)

        cu = batch["cu_seqlens"].to(device=h.device, dtype=torch.long)
        B = int(cu.numel()) - 1
        z = batch.get("cvae_z")
        if self.training and self.band_weight > 0 and z is not None and B >= 2:
            zb = z.detach() if self.band_stopgrad else z
            hb = h.detach() if self.band_stopgrad else h
            perm = torch.roll(torch.arange(B, device=h.device), 1)
            d = (zb - zb[perm]).norm(dim=-1)
            with torch.no_grad():
                r_true = packed.seg_mean(
                    self._l1(self._decode(hb, packed.broadcast_per_episode(zb, cu)), target), cu)
            r_perm = packed.seg_mean(
                self._l1(self._decode(hb, packed.broadcast_per_episode(zb[perm], cu)), target), cu)
            gap = r_perm - r_true
            u_pair = F.relu(self.c_lo * d.clamp(max=self.d_cap) - gap)
            s_pair = F.relu(gap - self.c_hi * d)
            l_use, l_sm = u_pair.mean(), s_pair.mean()
            batch["loss/band"] = self.band_weight * (l_use + l_sm)
            # fraction of pairs with an active hinge -> should trend to 0 as
            # the posterior calibrates (guardrail firing = exception, not norm)
            batch["log/band_use_rate"] = float((u_pair > 0).float().mean())
            batch["log/band_smooth_rate"] = float((s_pair > 0).float().mean())
            batch["log/gap"] = float(gap.mean())
            batch["log/zdist"] = float(d.mean())
            batch["log/L_use"] = float(l_use)
            batch["log/L_smooth"] = float(l_sm)
        return batch


# --------------------------------------------------------------------------- #
# Regularizer stage
# --------------------------------------------------------------------------- #
#: emb id (Embodiment enum) <-> config-friendly name, for per-emb loss targets
_EMB_ALIAS = {"eva_bimanual": "8", "human_bimanual": "18"}


class RatioLoss(Stage):
    """H-Net chunker ratio loss from the accumulated aux/chunker records.

    The compression TARGET is loss policy and lives HERE, not on the chunker:
    the chunker only routes and publishes b_t/p_t. ``target_ratio`` accepts

      * ``None``  -- use the value the chunker recorded (legacy default, exactly
        reproduces every pre-existing run),
      * a float   -- one override for all levels,
      * a dict    -- per embodiment, keyed by id ("8"/"18") or name
        ("eva_bimanual"/"human_bimanual"), falling back to the recorded value
        for any embodiment not listed.

    Per-emb targets exist because the streams run at very different speeds: a
    human fold is 156 frames (5.2 s) and the same robot fold is 1151 frames
    (38.4 s) at a shared 30 fps, so one ratio gives 17 apex tokens for human and
    128 for eva -- a 7.4x mismatch on the axis DTW alignment compares.
    """

    reads = ["aux/chunker"]
    writes = ["loss/ratio", "loss/dec", "loss/seam_recon", "loss/cons",
              "loss/topk", "loss/softrate", "loss/confidence", "log/*"]

    def __init__(self, target_ratio=None, target_ratio_s=None):
        super().__init__()
        self.target_ratio = target_ratio
        self.target_ratio_s = target_ratio_s

    @staticmethod
    def _is_seq(x):
        """list / tuple / omegaconf ListConfig -- NOT str, NOT a scalar.

        hydra passes ListConfig and DictConfig, for which isinstance against the
        native types is False; duck-type instead of enumerating classes."""
        return (not isinstance(x, (str, bytes, int, float))
                and hasattr(x, "__len__") and hasattr(x, "__getitem__")
                and not hasattr(x, "keys"))

    @classmethod
    def _pick_level(cls, val, level, recorded):
        """A per-emb entry may be a scalar (all levels) or a per-level sequence."""
        if cls._is_seq(val):
            if level is None or level >= len(val):
                return recorded
            return float(val[level])
        return float(val)

    def _target(self, cfg, recorded, batch, level=None):
        """Resolve the target for this record: loss config wins, else the value
        the chunker recorded (so an unconfigured stage is byte-identical).

        ``level`` is the chunker's index in aux/chunker (0 = first stage). It
        matters because the two stages COMPOUND -- 3.0 twice is 9x, not 3x."""
        if cfg is None:
            # TODO(remove-chunker-target): DEPRECATED fallback to the chunker-recorded target. Once
            # every model yaml sets RatioLoss(target_ratio=...), drop this
            # branch and make the argument required -- a silent fallback is how
            # a config that MEANT to set a per-emb target but mistyped the key
            # would quietly train on the old shared 3.0 instead.
            return recorded
        if isinstance(cfg, (int, float)):
            return float(cfg)
        if self._is_seq(cfg):
            return self._pick_level(cfg, level, recorded)
        key = str(batch.get("embodiment", ""))
        if key in cfg:
            return self._pick_level(cfg[key], level, recorded)
        for name, idx in _EMB_ALIAS.items():
            if idx == key and name in cfg:
                return self._pick_level(cfg[name], level, recorded)
        return recorded

    def forward(self, batch: dict) -> dict:
        aux = [{"bpred": _B(rec),
                "target_ratio": self._target(self.target_ratio,
                                             rec["target_ratio"], batch, _i),
                "weight": rec["ratio_weight"], "cu_seqlens": rec["cu_seqlens"]}
               for _i, rec in enumerate(batch.get("aux/chunker", []))]
        dev = None
        for rec in batch.get("aux/chunker", []):
            dev = rec["boundary_prob"].device
            break
        batch["loss/ratio"] = ratio_loss_from_aux(aux, device=dev)
        # PER-STREAM BOUNDARIES: the SPECIFIC stream has its own boundary, so it
        # needs its own compression regulariser. Only levels that actually enabled
        # separate_boundaries contribute; otherwise this key is never written and
        # the loss dict is byte-identical to before.
        aux_s = [{"bpred": _BS(rec),
                  "target_ratio": self._target(self.target_ratio_s,
                                               rec["target_ratio_s"], batch, _i),
                  # S boundaries live on S's grid; fall back to A's cu for old
                  # records (pre-fix) where the streams were never split.
                  "weight": rec["ratio_weight_s"],
                  "cu_seqlens": rec.get("cu_seqlens_s", rec["cu_seqlens"])}
                 for rec in batch.get("aux/chunker", [])
                 if rec.get("separate_boundaries", False)]
        if aux_s:
            batch["loss/ratio_s"] = ratio_loss_from_aux(aux_s, device=dev)
        # Decisiveness term: L_dec = mean(1-(2p-1)^2) is 1.0 at p=0.5, 0 at
        # p in {0,1}. Closes the ratio loss's threshold-riding loophole (a
        # point-mass at p=0.5 scores 0.75 < the honest optimum 1.0); with
        # near-binary probs G~F and the honest F=1/N becomes the true minimum.
        dec_total = None
        rec_total = None
        cons_total = None
        tk_total = None
        srate_total = None
        confidence_total = None
        _rates = []
        for i, rec in enumerate(batch.get("aux/chunker", [])):
            # Deep-supervision recon term (computed inside DualChunkerLevel —
            # dense dL/dp through the p-gated dechunk EMA). Assembled FIRST,
            # before the empty-valid `continue` below: the recon loss covers
            # ALL positions, so it must stay active even when a level has zero
            # unforced tokens (e.g. the seam at init, when the bottom router
            # still fires only at forced episode starts).
            rl_rec = rec.get("recon_loss", None)
            w_rec = float(rec.get("recon_weight", 0.0))
            if rl_rec is not None and w_rec > 0:
                batch[f"log/L{i}_recon"] = float(rl_rec)
                rec_total = (w_rec * rl_rec if rec_total is None
                             else rec_total + w_rec * rl_rec)
            # Consistency-under-noise (computed inside the chunker, like recon
            # — and like recon it covers ALL positions, so assemble before the
            # empty-valid continue).
            rl_cons = rec.get("cons_loss", None)
            w_cons = float(rec.get("cons_weight", 0.0))
            if rl_cons is not None and w_cons > 0:
                batch[f"log/L{i}_cons"] = float(rl_cons)
                cons_total = (w_cons * rl_cons if cons_total is None
                              else cons_total + w_cons * rl_cons)
            # Confidence-side predictive supervision is computed inside the
            # chunker so its graph remains attached to the selected-confidence
            # STE. Assemble it before the empty-valid continue, as short
            # episodes can have no unforced tokens while still producing a
            # well-defined auxiliary result.
            confidence_losses = rec.get("confidence_ssl_losses")
            confidence_weights = rec.get("confidence_ssl_weights", {})
            if confidence_losses is not None:
                batch[f"log/L{i}_conf_chunks"] = float(
                    confidence_losses["n_chunks"])
                batch[f"log/L{i}_conf_pairs"] = float(
                    confidence_losses["n_pairs"])
                for name in ("pred", "id", "vic"):
                    value = confidence_losses[name]
                    weight = float(confidence_weights.get(name, 0.0))
                    if weight <= 0:
                        continue
                    batch[f"log/L{i}_conf_{name}"] = float(value)
                    weighted = weight * value
                    confidence_total = (
                        weighted if confidence_total is None
                        else confidence_total + weighted
                    )
            p_b = rec["boundary_prob"][..., -1]
            valid = torch.ones_like(p_b, dtype=torch.bool)
            valid[rec["cu_seqlens"][:-1]] = False  # exclude forced subseq starts
            pv = p_b[valid]
            if pv.numel() == 0:
                continue
            # Chunk stats, old-repo semantics (chunk_stats_from_aux in
            # EgoVerse-gmm hnet.py): hard boundary mask with forced subseq
            # starts counted as boundaries; avg_chunk_len = n_tok / max(n_b, 1).
            bm = (p_b > 0.5).clone()
            bm[rec["cu_seqlens"][:-1]] = True
            n_tok = max(bm.numel(), 1)
            n_b = int(bm.sum().item())
            f_hard = n_b / n_tok
            batch[f"log/L{i}_boundary_rate"] = f_hard
            batch[f"log/L{i}_avg_chunk_len"] = n_tok / max(n_b, 1)
            _rates.append(f_hard)
            # --- FENCING diagnostics (prob distribution, not the rate) ------
            # A fenced router and an honest one have the SAME boundary rate;
            # only the spread of p distinguishes them. Cheap to log, and the
            # failure is otherwise invisible until a run is wasted.
            if pv.numel() > 1:
                # PRIMARY fencing signal (user, 2026-08-01): mean |p - 0.5|.
                # Distance from the fence, not spread and not a thresholded
                # count -- a router pinned at 0.5 has absdev ~0 no matter what
                # boundary RATE it produces, and rate alone cannot see it.
                batch[f"log/L{i}_prob_absdev"] = float((pv - 0.5).abs().mean())
                batch[f"log/L{i}_prob_mean"] = float(pv.mean())
                batch[f"log/L{i}_prob_std"] = float(pv.std())
                batch[f"log/L{i}_prob_range"] = float(pv.max() - pv.min())
                batch[f"log/L{i}_fenced_frac"] = float(
                    ((pv - 0.5).abs() < 0.02).float().mean())
            # --- SPECIFIC stream: same semantics, own grid ---------------
            # With separate_boundaries the S router chunks on its OWN grid, so
            # the A stats above say nothing about it. Without these, a
            # collapsed or runaway S router is silent until it crashes (the
            # ratio-loss index OOB) or quietly wastes the run.
            p_bs = rec.get("boundary_prob_s")
            if p_bs is not None:
                p_bs = p_bs[..., -1]
                cu_s = rec.get("cu_seqlens_s")
                if cu_s is None:
                    cu_s = rec["cu_seqlens"]
                bms = (p_bs > 0.5).clone()
                if int(cu_s[:-1].max()) < bms.numel():
                    bms[cu_s[:-1]] = True
                n_tok_s = max(bms.numel(), 1)
                n_b_s = int(bms.sum().item())
                batch[f"log/L{i}_boundary_rate_s"] = n_b_s / n_tok_s
                batch[f"log/L{i}_avg_chunk_len_s"] = n_tok_s / max(n_b_s, 1)
                valid_s = torch.ones_like(p_bs, dtype=torch.bool)
                if int(cu_s[:-1].max()) < valid_s.numel():
                    valid_s[cu_s[:-1]] = False
                pvs = p_bs[valid_s]
                if pvs.numel() > 1:
                    batch[f"log/L{i}_prob_absdev_s"] = float((pvs - 0.5).abs().mean())
                    batch[f"log/L{i}_prob_mean_s"] = float(pvs.mean())
                    batch[f"log/L{i}_prob_std_s"] = float(pvs.std())
                    batch[f"log/L{i}_fenced_frac_s"] = float(
                        ((pvs - 0.5).abs() < 0.02).float().mean())
                # 1.0 => S boundaries identical to A's (streams not independent)
                if bms.shape == bm.shape:
                    batch[f"log/L{i}_as_overlap"] = float(
                        (bms == bm).float().mean())
            w = float(rec.get("dec_weight", 0.0))
            if w > 0:
                dec_i = (1.0 - (2.0 * pv - 1.0).pow(2)).mean()
                batch[f"log/L{i}_dec"] = float(dec_i)
                dec_total = w * dec_i if dec_total is None else dec_total + w * dec_i
            # Distribution-level alternative: variance floor. Var(p) <= G(1-G)
            # with equality iff mass sits at {0,1}; requiring alpha of that
            # maximum kills the p=0.5 point-mass without taxing individual
            # ambiguous frames (unlike the frame-wise dec term).
            w_s = float(rec.get("spread_weight", 0.0))
            if w_s > 0:
                g_d = pv.mean().detach()
                target = float(rec.get("spread_alpha", 0.5)) * g_d * (1.0 - g_d)
                spread_i = F.relu(target - pv.var(unbiased=False))
                batch[f"log/L{i}_pvar"] = float(pv.var(unbiased=False))
                dec_total = w_s * spread_i if dec_total is None else dec_total + w_s * spread_i
            # Top-K hinge (targeted ratio replacement, fence-combat 2026-07-18):
            # rank the valid probs; the top K=M/N are asked to clear 0.5+m
            # (up-hinge), the rest to clear 0.5-m (down-hinge); tokens already
            # clear get ZERO gradient (free to sharpen under the task). Rate is
            # enforced by RANK (structural), so there is no flat valley at
            # F=1/N and no uniform push. Anti-merge is built in (the up-hinge
            # always demands K boundaries). Pair with ratio_loss_weight: 0.0.
            w_tk = float(rec.get("topk_weight", 0.0))
            if w_tk > 0 and pv.numel() > 0:
                m_tk = float(rec.get("topk_margin", 0.1))
                K = max(1, int(round(pv.numel() / float(rec["target_ratio"]))))
                order = torch.argsort(pv, descending=True)
                l_up = F.relu((0.5 + m_tk) - pv[order[:K]]).sum()
                l_dn = F.relu(pv[order[K:]] - (0.5 - m_tk)).sum()
                tk_i = w_tk * (l_up + l_dn) / max(pv.numel(), 1)
                batch[f"log/L{i}_topk"] = float(tk_i)
                tk_total = tk_i if tk_total is None else tk_total + tk_i
            # Soft-rate + indecision reg (user-designed dynamic ratio-loss
            # replacement, 2026-07-18). Term 1: rate via DIFFERENTIABLE soft
            # count — the correction's per-token magnitude carries sigma'
            # ((p-0.5)/tau), so it lands on undecided tokens and exempts
            # confident ones (and releases recruits once past the threshold
            # region instead of parking them at 0.5). Term 2: dead-zoned
            # indecision tax — only tokens with p(1-p) > c pay, pushed toward
            # their nearest extreme with force |1-2p|; direction at the exact
            # 0.5 tie is supplied by term 1's rate need. Pair with
            # ratio_loss_weight: 0.0 on the same level (this REPLACES it).
            # Softrate v2 (2026-07-18 final design): endpoint-calibrated votes
            # (targets exactly feasible, live gradient on all of [0,1]),
            # PER-EPISODE thermostats (episode mean preserves the 1/M force
            # scale), both terms share one weight, NO indecision tax. Replaces
            # v1 when softrate_vote == "endpoint".
            if str(rec.get("softrate_vote", "sigmoid")) == "endpoint":
                w_sr2v = float(rec.get("softrate_weight", 0.0))
                if w_sr2v > 0 and pv.numel() > 0:
                    tau_v2 = float(rec.get("softrate_tau", 0.12))
                    c1v = float(rec.get("softrate_center", 0.7))
                    c2v = float(rec.get("softrate2_center", 0.3))
                    inv_n = 1.0 / float(rec["target_ratio"])
                    _s = lambda x: 1.0 / (1.0 + __import__("math").exp(-x))
                    lo1v, hi1v = _s(-c1v / tau_v2), _s((1 - c1v) / tau_v2)
                    lo2v, hi2v = _s((c2v - 1) / tau_v2), _s(c2v / tau_v2)
                    cu_v2 = rec["cu_seqlens"]
                    ep_terms, sf_l, sb_l = [], [], []
                    for _b in range(len(cu_v2) - 1):
                        seg = p_b[cu_v2[_b]:cu_v2[_b + 1]]
                        vseg = valid[cu_v2[_b]:cu_v2[_b + 1]]
                        pe = seg[vseg]
                        if pe.numel() == 0:
                            continue
                        u_v = (torch.sigmoid((pe - c1v) / tau_v2) - lo1v) / (hi1v - lo1v)
                        d_v = (torch.sigmoid((c2v - pe) / tau_v2) - lo2v) / (hi2v - lo2v)
                        sf, sb = u_v.mean(), d_v.mean()
                        ep_terms.append((sf - inv_n) ** 2 + (sb - (1.0 - inv_n)) ** 2)
                        sf_l.append(float(sf)); sb_l.append(float(sb))
                    if ep_terms:
                        sr_v2 = w_sr2v * torch.stack(ep_terms).mean()
                        mean_sf = sum(sf_l) / len(sf_l)
                        batch[f"log/L{i}_softF"] = mean_sf
                        batch[f"log/L{i}_softB2"] = sum(sb_l) / len(sb_l)
                        batch[f"log/L{i}_srdiv"] = abs(f_hard - mean_sf)
                        _pc = pv.clamp(1e-6, 1 - 1e-6)
                        _al = (_pc / (1 - _pc)).log().abs()
                        batch[f"log/L{i}_abslogit_mean"] = float(_al.mean())
                        batch[f"log/L{i}_abslogit_p99"] = float(_al.quantile(0.99))
                        srate_total = (sr_v2 if srate_total is None
                                       else srate_total + sr_v2)
            w_sr = float(rec.get("softrate_weight", 0.0))
            w_ind = float(rec.get("indecision_weight", 0.0))
            if (w_sr > 0 or w_ind > 0) and pv.numel() > 0 and str(rec.get("softrate_vote", "sigmoid")) != "endpoint":
                tau_sr = float(rec.get("softrate_tau", 0.15))
                ctr_sr = float(rec.get("softrate_center", 0.5))
                c_dz = float(rec.get("indecision_deadzone", 0.21))
                inv_n = 1.0 / float(rec["target_ratio"])
                soft_f = torch.sigmoid((pv - ctr_sr) / tau_sr).mean()
                sr_i = w_sr * (soft_f - inv_n) ** 2
                # Second thermostat (user term, 2026-07-18): demand (N-1)/N of
                # tokens confidently BELOW center2 — softB = mean sigma((c2-p)/tau),
                # loss (softB - (N-1)/N)^2. With term 1 this pins BOTH tails and
                # empties the (center2, center) band: a forbidden zone under the
                # decision line. Identity: == (softF_c2 - 1/N)^2 by 1-s(x)=s(-x).
                w_sr2 = float(rec.get("softrate2_weight", 0.0))
                if w_sr2 > 0:
                    c2 = float(rec.get("softrate2_center", 0.3))
                    soft_b2 = torch.sigmoid((c2 - pv) / tau_sr).mean()
                    sr_i = sr_i + w_sr2 * (soft_b2 - (1.0 - inv_n)) ** 2
                    batch[f"log/L{i}_softB2"] = float(soft_b2)
                ind_i = w_ind * F.relu(pv * (1.0 - pv) - c_dz).mean()
                batch[f"log/L{i}_softF"] = float(soft_f)
                batch[f"log/L{i}_indec"] = float(ind_i)
                sr_total_i = sr_i + ind_i
                srate_total = (sr_total_i if srate_total is None
                               else srate_total + sr_total_i)
            # Hard-band reg (ratio-loss replacement): constrain the REALIZED
            # boundary decisions — the statistic eval actually uses — via STE.
            # (a) rate band [lo, hi] on the hard boundary rate: zero force
            #     inside the band (flat feasible basin, task owns the router);
            # (b) >= 1 hard boundary per `window` tokens within each episode
            #     (bans all-merge locally). Hedging at p~0.5 makes the realized
            #     stats flap -> violations kick probs ACROSS the threshold, so
            #     the 0.5 fence is unstable here rather than optimal.
            w_hb = float(rec.get("hb_weight", 0.0))
            if w_hb > 0:
                p_all = rec["boundary_prob"][..., -1]
                b_ste = (p_all > 0.5).float() + (p_all - p_all.detach())
                cu_hb = rec["cu_seqlens"]
                Fr = b_ste[valid].mean()  # forced starts excluded from the rate
                lo, hi = float(rec.get("hb_lo", 0.25)), float(rec.get("hb_hi", 0.45))
                l_rate = F.relu(Fr - hi) + F.relu(lo - Fr)
                W = max(2, int(rec.get("hb_window", 12)))
                l_win = p_all.new_zeros(())
                nwin = 0
                for _e in range(len(cu_hb) - 1):
                    seg = b_ste[int(cu_hb[_e]):int(cu_hb[_e + 1])]
                    k = seg.numel() // W
                    if k > 0:
                        wsum = seg[: k * W].reshape(k, W).sum(-1)
                        l_win = l_win + F.relu(1.0 - wsum).sum()
                        nwin += k
                if nwin:
                    l_win = l_win / nwin
                hb_i = w_hb * (l_rate + l_win)
                batch[f"log/L{i}_hb_rate"] = float(Fr)
                batch[f"log/L{i}_hb_winviol"] = float(l_win)
                dec_total = hb_i if dec_total is None else dec_total + hb_i
        if _rates:
            batch["log/boundary_rate"] = sum(_rates) / len(_rates)
        if dec_total is not None:
            batch["loss/dec"] = dec_total
        if rec_total is not None:
            batch["loss/seam_recon"] = rec_total
        if cons_total is not None:
            batch["loss/cons"] = cons_total
        if tk_total is not None:
            batch["loss/topk"] = tk_total
        if srate_total is not None:
            batch["loss/softrate"] = srate_total
        if confidence_total is not None:
            batch["loss/confidence"] = confidence_total
        return batch


class _B:
    """Adapter: aux/chunker record -> the bpred duck-type ratio_loss expects."""

    def __init__(self, rec):
        self.boundary_mask = rec["boundary_mask"]
        self.boundary_prob = rec["boundary_prob"]
        self.selected_probs = rec["selected_probs"]


class _BS:
    """Same adapter for the SPECIFIC stream's own boundary (separate_boundaries)."""

    def __init__(self, rec):
        self.boundary_mask = rec["boundary_mask_s"]
        self.boundary_prob = rec["boundary_prob_s"]
        self.selected_probs = rec["selected_probs_s"]
