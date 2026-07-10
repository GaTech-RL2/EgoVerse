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

from typing import List, Optional

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
                 band_stopgrad: bool = True):
        super().__init__()
        self.D, self.C, self.z_dim = int(action_dim), int(chunk_len), int(z_dim)
        self.d_cond = int(d_a) + int(d_s)
        self.band_weight, self.band_stopgrad = float(band_weight), bool(band_stopgrad)
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
class RatioLoss(Stage):
    """H-Net chunker ratio loss from the accumulated aux/chunker records."""

    reads = ["aux/chunker"]
    writes = ["loss/ratio"]

    def forward(self, batch: dict) -> dict:
        aux = [{"bpred": _B(rec), "target_ratio": rec["target_ratio"],
                "weight": rec["ratio_weight"], "cu_seqlens": rec["cu_seqlens"]}
               for rec in batch.get("aux/chunker", [])]
        dev = None
        for rec in batch.get("aux/chunker", []):
            dev = rec["boundary_prob"].device
            break
        batch["loss/ratio"] = ratio_loss_from_aux(aux, device=dev)
        return batch


class _B:
    """Adapter: aux/chunker record -> the bpred duck-type ratio_loss expects."""

    def __init__(self, rec):
        self.boundary_mask = rec["boundary_mask"]
        self.boundary_prob = rec["boundary_prob"]
        self.selected_probs = rec["selected_probs"]
