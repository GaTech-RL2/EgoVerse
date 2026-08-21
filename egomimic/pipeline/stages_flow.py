"""Dualstream diffusion/flow action heads (batchflow stages).

Two heads share one two-stream denoiser (the diffusion analog of the
partitioned GMMHead — A-stream conditioned only on ``a_top``, S-stream on
``s``, trunk-style asym mask, additive velocity/eps partition):

  * FlowHead — single-chunk flow matching (FMPolicy recipe).
  * SDPHead  — Streaming Diffusion Policy (TEDi): a K-chunk action BUFFER
    with a per-chunk noise LADDER. Training: per-position noise levels
    (chunk-wise diagonal primary regime), eps-prediction. Rollout: the
    buffer persists across token steps — each step denoises under the
    CURRENT h_t until the head chunk is clean, pops it for execution,
    shifts, appends fresh noise. Conditioning = the dualstream token
    (a_top, s) instead of raw obs; the buffer is refined under
    progressively newer observations (receding-horizon for H-Net).

SDP reference: github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy
(tedi_unet_hybrid_image_policy + schedulers), mapped to chunk granularity:
their Ta = our chunk C, their horizon = K*C, one control step = one token.
"""
from typing import List, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.moe_ffn import MoEFFN
from egomimic.models.diffusion.denoising_nets import (
    ConditionalUnet1D,
    SinusoidalPosEmb,
)
from egomimic.pipeline.core import Stage


class _ResidualMLPBlock(nn.Module):
    """Pre-norm residual hidden block for deeper per-embodiment adapters."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(width))
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, x):
        return x + self.proj(F.silu(self.norm(x)))


class _DualStreamBlock(nn.Module):
    """One denoiser block: joint masked attention over [A-tokens | S-tokens]
    (separate per-stream QKV/out to a shared attention dim, trunk idiom) +
    per-stream FFN."""

    def __init__(self, d_a: int, d_s: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.h = int(n_heads)
        self.dm = int(d_a)                      # shared attention dim
        assert self.dm % self.h == 0
        self.norm_a1, self.norm_s1 = nn.LayerNorm(d_a), nn.LayerNorm(d_s)
        self.qkv_a = nn.Linear(d_a, 3 * self.dm)
        self.qkv_s = nn.Linear(d_s, 3 * self.dm)
        self.out_a = nn.Linear(self.dm, d_a)
        self.out_s = nn.Linear(self.dm, d_s)
        self.norm_a2, self.norm_s2 = nn.LayerNorm(d_a), nn.LayerNorm(d_s)
        self.ffn_a = nn.Sequential(nn.Linear(d_a, ffn_mult * d_a), nn.GELU(),
                                   nn.Linear(ffn_mult * d_a, d_a))
        self.ffn_s = nn.Sequential(nn.Linear(d_s, ffn_mult * d_s), nn.GELU(),
                                   nn.Linear(ffn_mult * d_s, d_s))

    def forward(self, A, S, allow):            # A (T,L,dA)  S (T,L,dS)
        T, L, _ = A.shape
        hd = self.dm // self.h

        def _heads(x):
            q, k, v = x.chunk(3, -1)
            return [t.reshape(T, -1, self.h, hd).transpose(1, 2)
                    for t in (q, k, v)]

        qa, ka, va = _heads(self.qkv_a(self.norm_a1(A)))
        qs, ks, vs = _heads(self.qkv_s(self.norm_s1(S)))
        q = torch.cat([qa, qs], dim=2)          # (T,h,2L,hd)
        k = torch.cat([ka, ks], dim=2)
        v = torch.cat([va, vs], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        o = o.transpose(1, 2).reshape(T, 2 * L, self.dm)
        A = A + self.out_a(o[:, :L])
        S = S + self.out_s(o[:, L:])
        A = A + self.ffn_a(self.norm_a2(A))
        S = S + self.ffn_s(self.norm_s2(S))
        return A, S


class DualStreamDenoiser(nn.Module):
    """eps/v(x_t, t | a_top, s, emb) over an (L, D) action window per token.
    ``t`` may be (T,) — one level for the window (FlowHead) — or (T, L) —
    per-position levels (SDPHead's ladder)."""

    def __init__(self, d_a_in: int, d_s_in: int, action_dim: int, chunk_len: int,
                 embodiments: List[str], d_model_a: int = 256, d_model_s: int = 128,
                 n_layers: int = 4, n_heads: int = 4, ffn_mult: int = 4,
                 mask_mode: str = "sym", n_positions: Optional[int] = None):
        super().__init__()
        C, D = int(chunk_len), int(action_dim)
        L = int(n_positions) if n_positions else C
        dA, dS = int(d_model_a), int(d_model_s)
        self.C, self.D, self.L = C, D, L
        self.mask_mode = str(mask_mode)
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]

        # A path: everything shared (embodiment-blind end-to-end)
        self.in_a = nn.Linear(D, dA)
        self.cond_a = nn.Linear(int(d_a_in), dA)
        self.temb_a = nn.Sequential(SinusoidalPosEmb(dA),
                                    nn.Linear(dA, dA), nn.GELU(), nn.Linear(dA, dA))
        self.pos_a = nn.Parameter(torch.zeros(L, dA))
        self.vout_a = nn.Linear(dA, D)
        # S path: per-emb in/cond/out projections, SHARED stream weights
        self.in_s = nn.ModuleDict({e: nn.Linear(D, dS) for e in embs})
        self.cond_s = nn.ModuleDict({e: nn.Linear(int(d_s_in), dS) for e in embs})
        self.temb_s = nn.Sequential(SinusoidalPosEmb(dS),
                                    nn.Linear(dS, dS), nn.GELU(), nn.Linear(dS, dS))
        self.pos_s = nn.Parameter(torch.zeros(L, dS))
        self.vout_s = nn.ModuleDict({e: nn.Linear(dS, D) for e in embs})
        nn.init.trunc_normal_(self.pos_a, std=0.02)
        nn.init.trunc_normal_(self.pos_s, std=0.02)

        self.blocks = nn.ModuleList(
            _DualStreamBlock(dA, dS, n_heads, ffn_mult) for _ in range(int(n_layers)))
        self.norm_fa, self.norm_fs = nn.LayerNorm(dA), nn.LayerNorm(dS)

    def _allow(self, L, device):
        allow = torch.ones(2 * L, 2 * L, dtype=torch.bool, device=device)
        if self.mask_mode == "asym":            # A queries see A keys only
            allow[:L, L:] = False
        return allow

    def _temb(self, mlp, t, T, L):
        """t (T,) or (T,L) -> (T, L, d) position-wise time embedding."""
        if t.dim() == 1:
            return mlp(t)[:, None, :].expand(T, L, -1)
        return mlp(t.reshape(-1)).reshape(T, L, -1)

    def forward(self, x_t, t, a_top, s, emb: str):
        # x_t (T,L,D)  t (T,)|(T,L)  a_top (T,dA_in)  s (T,dS_in)
        T, L, _ = x_t.shape
        e = emb if emb in self.in_s else next(iter(self.in_s))
        A = (self.in_a(x_t) + self.cond_a(a_top)[:, None, :]
             + self._temb(self.temb_a, t, T, L) + self.pos_a[None, :L])
        S = (self.in_s[e](x_t) + self.cond_s[e](s)[:, None, :]
             + self._temb(self.temb_s, t, T, L) + self.pos_s[None, :L])
        allow = self._allow(L, x_t.device)
        for blk in self.blocks:
            A, S = blk(A, S, allow)
        v_a = self.vout_a(self.norm_fa(A))
        v_s = self.vout_s[e](self.norm_fs(S))
        return v_a + v_s, v_a, v_s


class FlowHead(Stage):
    """Dualstream flow-matching head (single chunk). Train: loss/flow
    (velocity MSE, FMPolicy recipe). Rollout: Euler ODE -> pred_action."""

    reads = ["a_top", "s", "embodiment"]
    writes = ["pred_action", "loss/flow", "log/flow", "log/vA_frac"]

    def __init__(self, d_a: int, d_s: int, action_dim: int, chunk_len: int,
                 embodiments: Optional[List[str]] = None,
                 d_model_a: int = 256, d_model_s: int = 128,
                 n_layers: int = 4, n_heads: int = 4, ffn_mult: int = 4,
                 mask_mode: str = "sym", num_inference_steps: int = 20,
                 time_dist: str = "beta", denoiser_arch: str = "additive"):
        super().__init__()
        self.C, self.D = int(chunk_len), int(action_dim)
        self.N = int(num_inference_steps)
        self.time_dist = str(time_dist)
        _cls = (DualStreamDenoiserV2 if str(denoiser_arch) == "adaln"
                else DualStreamDenoiser)
        self.net = _cls(
            d_a_in=d_a, d_s_in=d_s, action_dim=action_dim, chunk_len=chunk_len,
            embodiments=list(embodiments) if embodiments else ["shared"],
            d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
            n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode)

    def _sample_t(self, T, device):
        if self.time_dist == "beta":            # FMPolicy default (a=1.5, b=1)
            t = torch.distributions.Beta(1.5, 1.0).sample((T,)).to(device)
        else:
            t = torch.rand(T, device=device)
        return t * 0.999 + 0.001

    def forward(self, batch: dict) -> dict:
        a_top, s, emb = batch["a_top"], batch["s"], str(batch["embodiment"])
        if "target" in batch:
            x0 = batch["target"]                              # (T,C,D)
            noise = torch.randn_like(x0)
            t = self._sample_t(x0.shape[0], x0.device)
            te = t[:, None, None]
            x_t = te * noise + (1.0 - te) * x0                # FMPolicy convention
            v, v_a, v_s = self.net(x_t, t, a_top, s, emb)
            loss = F.mse_loss(v, noise - x0)
            batch["loss/flow"] = loss
            batch["log/flow"] = float(loss)
            with torch.no_grad():
                na = v_a.norm(dim=-1).mean()
                ns = v_s.norm(dim=-1).mean()
                batch["log/vA_frac"] = float(na / (na + ns + 1e-8))
        if not self.training:
            with torch.no_grad():
                T = a_top.shape[0]
                # ROLLOUT FAST PATH (2026-08-13). algo.py:375 consumes only
                # pred_action[T-1], but this loop denoised ALL T rows at full
                # sampler depth -> cost quadratic in episode length (~199x
                # wasted on a 397-step episode; one episode cost ~66 min and no
                # dn_* sim eval ever completed a single episode). T is a pure
                # batch dim in these denoisers -- blocks attend over the chunk
                # axis, never across T -- so row -1 is identical whether or not
                # the other rows are computed. Mirrors SDPHead's rollout_t
                # branch (stages_flow.py:453-455 slice, :495-497 scatter).
                _stream = "rollout_t" in batch
                _a = a_top[-1:] if _stream else a_top
                _s = (s[-1:] if s is not None else None) if _stream else s
                Tc = _a.shape[0]
                x = torch.randn(Tc, self.C, self.D, device=a_top.device,
                                dtype=a_top.dtype)
                dt = 1.0 / self.N
                for i in range(self.N):                        # t: 1 -> 0
                    tt = torch.full((Tc,), 1.0 - i * dt, device=x.device,
                                    dtype=x.dtype)
                    v, _, _ = self.net(x, tt, _a, _s, emb)
                    x = x - dt * v
                x = x.clamp(-1.0, 1.0)
                if _stream:
                    _out = torch.zeros(T, self.C, self.D,
                                       device=a_top.device, dtype=a_top.dtype)
                    _out[-1] = x[0]
                    x = _out
                batch["pred_action"] = x
        return batch


# --------------------------------------------------------------------------- #
# Streaming Diffusion Policy (TEDi) head
# --------------------------------------------------------------------------- #
def _cosine_alphas_cumprod(N: int, s: float = 0.008):
    """squaredcos_cap_v2 cumulative alphas, index 0..N-1 (level = index)."""
    steps = torch.arange(N + 1, dtype=torch.float64)
    f = torch.cos(((steps / N) + s) / (1 + s) * math.pi / 2) ** 2
    ac = (f / f[0]).clamp(1e-8, 1.0)
    return ac[1:].float()                       # (N,) abar at level i


class SDPHead(Stage):
    """Streaming Diffusion Policy head over a K-chunk buffer, conditioned on
    the dualstream token (a_top, s).

    Train: per-token buffer targets (chunks t..t+K-1, episode-masked); noise
    levels per CHUNK from {chunk-wise diagonal (primary) | constant | random};
    eps-prediction MSE -> loss/sdp.

    Rollout (batch carries ``rollout_t`` from algo.step): persistent buffer;
    per token-step DDIM-decrement rounds under the CURRENT h until the head
    chunk is clean, pop -> pred_action[T-1], shift, append fresh noise.

    TF-val (no ``rollout_t``): per-token full DDIM denoise of a fresh buffer
    at a constant ladder; head chunk -> pred_action (overlay viz only).
    """

    reads = ["a_top", "s", "embodiment", "cu_seqlens"]
    writes = ["pred_action", "loss/sdp", "log/sdp", "log/vA_frac"]

    def __init__(self, d_a: int, d_s: int, action_dim: int, chunk_len: int,
                 embodiments: Optional[List[str]] = None,
                 buffer_chunks: int = 4, num_train_timesteps: int = 100,
                 num_inference_steps: int = 16,
                 regime_weights: Optional[List[float]] = None,  # [chunkwise, constant, random]
                 d_model_a: int = 256, d_model_s: int = 128,
                 n_layers: int = 4, n_heads: int = 4, ffn_mult: int = 4,
                 mask_mode: str = "sym", end_mode: str = "masked",
                 denoiser_arch: str = "additive",
                 detach_offregime: bool = False,
                 action_dims: Optional[dict] = None, latent_dim: Optional[int] = None,
                 dual_stream: bool = True, enc_hidden=256, enc_layers=3,
                 enc_residual=False, head_adapter: str = "latent",
                 adapter_layers: int = 4, adapter_hidden: Optional[int] = None,
                 enc_per_stream: bool = False):
        super().__init__()
        self.C, self.D, self.K = int(chunk_len), int(action_dim), int(buffer_chunks)
        self.L = self.K * self.C
        self.N = int(num_train_timesteps)
        self.S = int(num_inference_steps)
        assert self.S % self.K == 0, "num_inference_steps must divide by buffer_chunks"
        self.rw = list(regime_weights) if regime_weights else [0.7, 0.15, 0.15]
        self.end_mode = str(end_mode)
        self.detach_offregime = bool(detach_offregime)
        self.register_buffer("abar", _cosine_alphas_cumprod(self.N))
        # inference sub-ladder: index j in [0, S) -> train level (descending)
        self.register_buffer(
            "inf_levels",
            torch.linspace(self.N - 1, 0, self.S).round().long())
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]
        # per-embodiment action dims (hetero-D robot-human). Homogeneous case:
        # every emb maps to the scalar action_dim (identical to old behaviour).
        self.Dmap = {e: int((action_dims or {}).get(e, action_dim)) for e in embs}
        self.dual_stream = bool(dual_stream)
        self.rh = action_dims is not None or latent_dim is not None
        if self.rh and str(head_adapter) == "direct":
            # DIRECT hetero adapters: per-emb MLPs straight to/from the trunk
            # width, no common action-width waist. See DualStreamDenoiserHetero.
            assert self.dual_stream, "head_adapter=direct requires dual_stream"
            assert str(denoiser_arch) == "adaln", \
                "head_adapter=direct is implemented on the adaln core only"
            self.net = DualStreamDenoiserHetero(
                d_a_in=d_a, d_s_in=d_s, action_dims=self.Dmap,
                chunk_len=chunk_len, embodiments=embs,
                d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode,
                n_positions=self.L, adapter_layers=adapter_layers,
                adapter_hidden=adapter_hidden)
        elif self.rh:
            # hetero-action-dim: noise/x_t/loss stay in the ORIGINAL per-emb dim;
            # per-emb 3-layer SiLU E_e (dim_e -> latent) / D_e (latent -> dim_e)
            # wrap a SHARED denoiser core that runs entirely in the latent.
            L_lat = int(latent_dim if latent_dim is not None else max(self.Dmap.values()))
            self.net = LatentRHDenoiser(
                d_a_in=d_a, d_s_in=d_s, action_dims=self.Dmap, latent_dim=L_lat,
                chunk_len=chunk_len, embodiments=embs,
                d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode,
                n_positions=self.L, dual_stream=self.dual_stream,
                enc_hidden=enc_hidden, enc_layers=enc_layers,
                enc_residual=enc_residual, enc_per_stream=enc_per_stream,
                dual_arch=("adaln" if str(denoiser_arch) == "adaln" else "v1"))
        else:
            net_cls = (DualStreamDenoiserV2 if str(denoiser_arch) == "adaln"
                       else DualStreamDenoiser)
            self.net = net_cls(
                d_a_in=d_a, d_s_in=d_s, action_dim=action_dim, chunk_len=chunk_len,
                embodiments=embs,
                d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode,
                n_positions=self.L)
        self._stream = None                      # rollout-only buffer state

    def _D(self, emb) -> int:
        """Per-embodiment action dim (== the scalar D in the homogeneous case)."""
        return self.Dmap.get(str(emb), self.D)

    # ---------------- level sampling (train) ---------------- #
    def _sample_levels(self, T, device):
        """Per-token, per-CHUNK train levels (T, K) — TEDi regimes."""
        K, N = self.K, self.N
        pick = torch.multinomial(
            torch.tensor(self.rw, device=device, dtype=torch.float), T,
            replacement=True)                                   # (T,)
        # chunk-wise diagonal: floor(N*(k+1)/K) - 1 - j,  j ~ U[0, N/K)
        kk = torch.arange(K, device=device)[None, :]
        j = torch.randint(0, max(1, N // K), (T, 1), device=device)
        diag = (torch.div(N * (kk + 1), K, rounding_mode="floor") - 1 - j)
        const = torch.randint(0, N, (T, 1), device=device).expand(T, K)
        rand = torch.randint(0, N, (T, K), device=device)
        lv = torch.where(pick[:, None] == 0, diag,
                         torch.where(pick[:, None] == 1, const, rand))
        return lv.clamp(0, N - 1), pick                          # (T,K),(T,)

    # ---------------- packed multi-chunk targets ---------------- #
    def _buffer_targets(self, target, cu, state=None):
        """target (T,C,D), cu (E+1,) -> (T,K,C,D) buffer targets + (T,K) valid.

        end_mode="masked" (legacy): batch-clamped gather; overrun positions
        masked from the loss (content may be foreign-episode junk in packs).
        end_mode="pusher_hold": per-episode clamp (no cross-episode content)
        and beyond-end targets = HOLD at the episode's final pusher pose. The
        first D state coordinates must match the D-dimensional action pose
        (e.g. xy for a circle pusher, xy-theta for a U-socket); tail positions
        are TRAINED (explicit endgame semantics)."""
        T = target.shape[0]
        Dd = target.shape[-1]
        dev = target.device
        ep_end = torch.empty(T, dtype=torch.long, device=dev)
        for b in range(len(cu) - 1):
            ep_end[cu[b]:cu[b + 1]] = cu[b + 1]
        idx = torch.arange(T, device=dev)[:, None] + torch.arange(self.K, device=dev)[None, :]
        valid = idx < ep_end[:, None]                           # (T,K)
        if self.end_mode == "pusher_hold" and state is not None:
            if state.shape[-1] < Dd:
                raise ValueError(
                    f"pusher_hold needs at least {Dd} state coordinates for "
                    f"a {Dd}D action, got state shape {tuple(state.shape)}"
                )
            idx_c = torch.minimum(idx, (ep_end - 1)[:, None])
            tgt = target[idx_c]                                 # same-episode only
            pad = state[(ep_end - 1), :Dd].to(tgt.dtype)        # (T,D) final pusher pose
            pad = pad[:, None, None, :].expand(T, self.K, self.C, Dd)
            tgt = torch.where(valid[..., None, None], tgt, pad)
            valid = torch.ones_like(valid)                      # tail trained
        else:
            tgt = target[idx.clamp(max=T - 1)]                  # (T,K,C,D)
        return tgt, valid

    # ---------------- ddim update at per-position levels ---------------- #
    def _ddim_round(self, x, lv_idx, a_top, s, emb, stop=None):
        """One decrement round: positions at sub-ladder index j move to j+1
        (toward clean); chunks at/past their ``stop`` index (default S=clean)
        hold. x (T,K,C,D), lv_idx (T,K) in [0..S] where S == clean."""
        T = x.shape[0]
        Dd = x.shape[-1]                                        # per-emb action dim
        act = lv_idx < (self.S if stop is None else stop)       # active chunks
        if not act.any():
            return x, lv_idx
        tl = self.inf_levels[lv_idx.clamp(max=self.S - 1)]      # train levels (T,K)
        t_pos = tl.repeat_interleave(self.C, dim=1).float()     # (T,L)
        eps, _, _ = self.net(x.reshape(T, self.L, Dd), t_pos, a_top, s, emb)
        eps = eps.reshape(T, self.K, self.C, Dd)
        ab_t = self.abar[tl][..., None, None]                   # (T,K,1,1)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0 = x0.clamp(-1.0, 1.0)
        nxt = lv_idx + 1
        done = nxt >= self.S
        tl_n = self.inf_levels[nxt.clamp(max=self.S - 1)]
        ab_n = self.abar[tl_n][..., None, None]
        x_new = torch.where(
            done[..., None, None], x0,
            ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps)
        x = torch.where(act[..., None, None], x_new, x)
        lv_idx = torch.where(act, nxt.clamp(max=self.S), lv_idx)
        return x, lv_idx

    # ---------------- forward ---------------- #
    def forward(self, batch: dict) -> dict:
        a_top, s, emb = batch["a_top"], batch["s"], str(batch["embodiment"])
        T = a_top.shape[0]
        dev = a_top.device
        De = self._D(emb)                                       # per-emb action dim

        if "target" in batch:                                   # ---- train loss
            cu = batch["cu_seqlens"].to(device=dev, dtype=torch.long)
            # New PushShapes datasets expose the commanded pusher pose, which
            # is in the exact same coordinate system as actions. Prefer it for
            # tail HOLD padding; legacy datasets fall back to actual state.
            hold_pose = batch.get(
                "obs/pusher_cmd_pose", batch.get("obs/state_agent_obj")
            )
            tgt, valid = self._buffer_targets(
                batch["target"], cu, hold_pose)                  # (T,K,C,D)
            Dd = tgt.shape[-1]                                  # per-emb action dim
            lv, pick = self._sample_levels(T, dev)              # (T,K),(T,)
            noise = torch.randn_like(tgt)
            ab = self.abar[lv][..., None, None]                 # (T,K,1,1)
            x_t = ab.sqrt() * tgt + (1 - ab).sqrt() * noise
            t_pos = lv.repeat_interleave(self.C, dim=1).float() # (T,L)
            if self.detach_offregime:
                infm = (pick == 0).float()[:, None]          # (T,1) inference-regime mask
                a_cond = a_top * infm + a_top.detach() * (1 - infm)
                s_cond = s * infm + s.detach() * (1 - infm)
            else:
                a_cond, s_cond = a_top, s
            eps, e_a, e_s = self.net(x_t.reshape(T, self.L, Dd),
                                     t_pos, a_cond, s_cond, emb)
            eps = eps.reshape(T, self.K, self.C, Dd)
            m = valid[..., None, None].float()
            loss = ((eps - noise).pow(2) * m).sum() / (
                m.sum().clamp(min=1) * self.C * Dd)
            batch["loss/sdp"] = loss
            batch["log/sdp"] = float(loss)
            with torch.no_grad():
                if e_a is not None:                             # dual variant only
                    na = e_a.norm(dim=-1).mean()
                    ns = e_s.norm(dim=-1).mean()
                    batch["log/vA_frac"] = float(na / (na + ns + 1e-8))

        if not self.training:
            with torch.no_grad():
                if "rollout_t" in batch:                        # ---- streaming
                    t_env = int(batch["rollout_t"])
                    h_a, h_s = a_top[-1:], s[-1:]
                    if t_env == 0 or self._stream is None:
                        # TEDi denoise-init: (1) burn the fresh buffer
                        # UNIFORMLY to clean — every model call is a
                        # constant-regime state (15% of training covers the
                        # all-equal ladder at every level), never a mixed
                        # staircase; (2) re-noise chunks 1..K-1 with EXACT
                        # forward q-sampling (no model) to the steady
                        # diagonal j_k = S - k*S/K, head stays clean -> the
                        # pop loop below no-ops and post-pop state is steady.
                        x = torch.randn(1, self.K, self.C, De, device=dev,
                                        dtype=a_top.dtype)
                        lv = torch.zeros(1, self.K, dtype=torch.long, device=dev)
                        while bool((lv < self.S).any()):
                            x, lv = self._ddim_round(x, lv, h_a, h_s, emb)
                        stp = self.S // self.K
                        for k in range(1, self.K):
                            jk = self.S - k * stp
                            ab = self.abar[int(self.inf_levels[jk])]
                            x[:, k] = (ab.sqrt() * x[:, k]
                                       + (1 - ab).sqrt() * torch.randn_like(x[:, k]))
                            lv[0, k] = jk
                        self._stream = {"x": x, "lv": lv}
                    st = self._stream
                    while int(st["lv"][0, 0]) < self.S:         # clean the head
                        st["x"], st["lv"] = self._ddim_round(
                            st["x"], st["lv"], h_a, h_s, emb)
                    head = st["x"][:, 0]                        # (1,C,D) clean
                    # shift + fresh noise at max level
                    st["x"] = torch.cat(
                        [st["x"][:, 1:],
                         torch.randn(1, 1, self.C, De, device=dev,
                                     dtype=a_top.dtype)], dim=1)
                    st["lv"] = torch.cat(
                        [st["lv"][:, 1:] ,
                         torch.zeros(1, 1, dtype=torch.long, device=dev)], dim=1)
                    out = torch.zeros(T, self.C, De, device=dev,
                                      dtype=a_top.dtype)
                    out[-1] = head[0].clamp(-1.0, 1.0)
                    batch["pred_action"] = out
                else:                                           # ---- TF-val
                    x = torch.randn(T, self.K, self.C, De, device=dev,
                                    dtype=a_top.dtype)
                    lv = torch.zeros(T, self.K, dtype=torch.long, device=dev)
                    for _ in range(self.S):
                        x, lv = self._ddim_round(x, lv, a_top, s, emb)
                    batch["pred_action"] = x[:, 0].clamp(-1.0, 1.0)
        return batch


# --------------------------------------------------------------------------- #
# Standard (DP-style) diffusion head — one-shot DDPM over the single chunk
# --------------------------------------------------------------------------- #
class DiffusionHead(Stage):
    """Standard diffusion action head on the dualstream denoiser: one-shot
    DDPM eps-prediction over the C-action chunk per token, DDIM (eta=0)
    sampling on an S-level sub-ladder. Objective twin of FlowHead (same
    denoiser; DDPM vs rectified flow) and one-shot twin of SDPHead (same
    objective; no streaming buffer). Stateless -> TF-val and the rollout
    step() path are the same computation."""

    reads = ["a_top", "s", "embodiment"]      # narrowed in __init__ when d_s is None
    writes = ["pred_action", "loss/ddpm", "log/ddpm", "log/vA_frac",
              "loss/moe_lb", "log/*"]

    def __init__(self, d_a: int, d_s: int, action_dim: int, chunk_len: int,
                 embodiments: Optional[List[str]] = None,
                 num_train_timesteps: int = 100, num_inference_steps: int = 16,
                 d_model_a: int = 256, d_model_s: int = 128,
                 n_layers: int = 4, n_heads: int = 4, ffn_mult: int = 4,
                 mask_mode: str = "sym",
                 denoiser: str = "dual",
                 denoiser_arch: str = "adaln",
                 moe_experts: int = 0, moe_top_k: int = 4,
                 moe_d_expert: Optional[int] = None,
                 moe_aux_weight: float = 0.01,
                 action_dims: Optional[dict] = None,
                 latent_dim: Optional[int] = None,
                 enc_hidden=256, enc_layers=3, enc_residual=False,
                 enc_per_stream: bool = False,
                 emit_loss: bool = True,
                 loss_space: str = "eps"):
        super().__init__()
        self.emit_loss = bool(emit_loss)
        # loss_space (OBJECTIVE ABLATION, user 2026-08-18): "eps" (default,
        # unchanged) scores MSE in NOISE space -- the DDPM eps objective.
        # "action" recovers x0 from eps and scores MSE in ACTION space:
        #   x0 = (x_t - sqrt(1-abar)*eps)/sqrt(abar), clamped to [-1,1] to match
        #   the sampler's clip_sample and bound the 1/sqrt(abar) blow-up at high
        #   noise. Published under the SAME aux keys so MaskedActionLoss /
        #   emit_loss score it unchanged. Isolates the objective (pooled
        #   conditioning + head unchanged).
        if str(loss_space) not in ("eps", "action"):
            raise ValueError(f"loss_space must be eps|action, got {loss_space!r}")
        self.loss_space = str(loss_space)
        self.C, self.D = int(chunk_len), int(action_dim)
        self.N, self.S = int(num_train_timesteps), int(num_inference_steps)
        self.register_buffer("abar", _cosine_alphas_cumprod(self.N))
        self.register_buffer(
            "inf_levels", torch.linspace(self.N - 1, 0, self.S).round().long())
        if denoiser not in ("dual", "single"):
            raise ValueError(f"denoiser must be dual|single, got {denoiser!r}")
        self.denoiser_kind = str(denoiser)
        # HETERO action dims (robot-human): the SAME construction SDPHead uses
        # -- per-emb E_e/D_e MLP codec wrapped around a SHARED denoiser core
        # that runs entirely in a common latent, so eva(14) and human(132) can
        # cotrain on one core. Configs that pass neither action_dims nor
        # latent_dim keep the homogeneous path byte-for-byte.
        embs_ = [str(e) for e in embodiments] if embodiments else ["shared"]
        self.Dmap = {e: int((action_dims or {}).get(e, action_dim)) for e in embs_}
        self.rh = action_dims is not None or latent_dim is not None
        if not d_s:
            # single-stream on a replica-style obs path: no S key exists
            self.reads = ["a_top", "embodiment"]
        if self.rh:
            L_lat = int(latent_dim if latent_dim is not None
                        else max(self.Dmap.values()))
            self.net = LatentRHDenoiser(
                d_a_in=d_a, d_s_in=d_s, action_dims=self.Dmap, latent_dim=L_lat,
                chunk_len=chunk_len, embodiments=embs_,
                d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode,
                n_positions=int(chunk_len), dual_stream=(denoiser == "dual"),
                enc_hidden=enc_hidden, enc_layers=enc_layers,
                enc_residual=enc_residual, enc_per_stream=enc_per_stream,
                dual_arch="adaln",
                moe_experts=moe_experts, moe_top_k=moe_top_k,
                moe_d_expert=moe_d_expert, moe_aux_weight=moe_aux_weight)
        elif denoiser == "single":
            # One denoiser stream; A/S survive only as conditioning projections.
            # MoE (when moe_experts > 0) swaps this block's FFN for experts.
            self.net = SingleStreamDenoiserV2(
                d_a_in=d_a, d_s_in=d_s, action_dim=action_dim,
                chunk_len=chunk_len, d_model=d_model_a, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, n_positions=int(chunk_len),
                moe_experts=moe_experts, moe_top_k=moe_top_k,
                moe_d_expert=moe_d_expert, moe_aux_weight=moe_aux_weight)
        else:
            # adaLN-Zero by default: the v1 core injects conditioning once, and
            # every OTHER cell in this fleet is adaLN. Leaving v1 as the
            # homogeneous default made "no latent_dim" silently mean "older
            # architecture" and confounded dual-vs-MoE.
            _cls = (DualStreamDenoiserV2 if str(denoiser_arch) == "adaln"
                    else DualStreamDenoiser)
            self.net = _cls(
                d_a_in=d_a, d_s_in=d_s, action_dim=action_dim, chunk_len=chunk_len,
                embodiments=list(embodiments) if embodiments else ["shared"],
                d_model_a=d_model_a, d_model_s=d_model_s, n_layers=n_layers,
                n_heads=n_heads, ffn_mult=ffn_mult, mask_mode=mask_mode,
                n_positions=int(chunk_len))

    def _D(self, emb) -> int:
        """Per-embodiment action dim (== the scalar D when homogeneous)."""
        return self.Dmap.get(str(emb), self.D)

    def forward(self, batch: dict) -> dict:
        a_top = batch["a_top"]
        s = batch.get("s")
        emb = str(batch["embodiment"])
        if "target" in batch:
            x0 = batch["target"]                                # (T,C,D)
            T = x0.shape[0]
            t = torch.randint(0, self.N, (T,), device=x0.device)
            ab = self.abar[t][:, None, None]
            noise = torch.randn_like(x0)
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            eps, e_a, e_s = self.net(x_t, t.float(), a_top, s, emb)
            # Publish the eps prediction/target so a downstream loss STAGE can
            # score them (e.g. MaskedActionLoss, which drops the gripper dims
            # for an embodiment that has no gripper). emit_loss=False hands the
            # objective entirely to that stage.
            if self.loss_space == "action":
                x0_pred = ((x_t - (1 - ab).sqrt() * eps)
                           / ab.sqrt().clamp_min(1e-4)).clamp(-1.0, 1.0)
                batch["aux/eps_pred"] = x0_pred
                batch["aux/eps_target"] = x0
            else:
                batch["aux/eps_pred"] = eps
                batch["aux/eps_target"] = noise
            # t is needed to bin the loss by noise level: the uniform-t
            # average hides whether the model solves the LOW-t regime,
            # which is the one that decides action fidelity.
            # (DiffusionDiagnosticEval)
            batch["aux/ddpm_t"] = t
            if self.emit_loss:
                loss = F.mse_loss(batch["aux/eps_pred"], batch["aux/eps_target"])
                batch["loss/ddpm"] = loss
                batch["log/ddpm"] = float(loss)
            if e_a is not None and e_s is not None:
                with torch.no_grad():
                    na = e_a.norm(dim=-1).mean()
                    ns = e_s.norm(dim=-1).mean()
                    batch["log/vA_frac"] = float(na / (na + ns + 1e-8))
        if not self.training:
            with torch.no_grad():
                T = a_top.shape[0]
                # ROLLOUT FAST PATH (2026-08-13). algo.py:375 consumes only
                # pred_action[T-1], but this loop denoised ALL T rows at full
                # sampler depth -> cost quadratic in episode length (~199x
                # wasted on a 397-step episode; one episode cost ~66 min and no
                # dn_* sim eval ever completed a single episode). T is a pure
                # batch dim in these denoisers -- blocks attend over the chunk
                # axis, never across T -- so row -1 is identical whether or not
                # the other rows are computed. Mirrors SDPHead's rollout_t
                # branch (stages_flow.py:453-455 slice, :495-497 scatter).
                _stream = "rollout_t" in batch
                _a = a_top[-1:] if _stream else a_top
                _s = (s[-1:] if s is not None else None) if _stream else s
                Tc = _a.shape[0]
                x = torch.randn(Tc, self.C, self._D(emb), device=a_top.device,
                                dtype=a_top.dtype)
                for j in range(self.S):                          # DDIM eta=0
                    tl = int(self.inf_levels[j])
                    tt = torch.full((Tc,), float(tl), device=x.device,
                                    dtype=x.dtype)
                    eps, _, _ = self.net(x, tt, _a, _s, emb)
                    ab_t = self.abar[tl]
                    x0p = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt())
                    x0p = x0p.clamp(-1.0, 1.0)
                    if j + 1 < self.S:
                        ab_n = self.abar[int(self.inf_levels[j + 1])]
                        x = ab_n.sqrt() * x0p + (1 - ab_n).sqrt() * eps
                    else:
                        x = x0p
                x = x.clamp(-1.0, 1.0)
                if _stream:
                    _out = torch.zeros(T, self.C, self._D(emb),
                                       device=a_top.device, dtype=a_top.dtype)
                    _out[-1] = x[0]
                    x = _out
                batch["pred_action"] = x
        moes = [m for m in self.net.modules() if isinstance(m, MoEFFN)]
        aux = [m.last_aux_loss for m in moes if m.last_aux_loss is not None]
        if aux:
            batch["loss/moe_lb"] = torch.stack(aux).sum()
            e = str(batch["embodiment"])
            f = torch.stack([m.last_expert_frac for m in moes
                             if m.last_expert_frac is not None]).mean(0)
            for i_ in range(f.numel()):
                batch[f"log/moe_expert_frac_{e}_e{i_}"] = f[i_]
            batch[f"log/moe_gate_entropy_{e}"] = torch.stack(
                [m.last_gate_entropy for m in moes
                 if m.last_gate_entropy is not None]).mean()
        return batch


# --------------------------------------------------------------------------- #
# Cursor/prev-action proprio (copycat-vs-smoothness experiment, user 2026-07-18)
# --------------------------------------------------------------------------- #

def _dp_alphas_cumprod(N: int, max_beta: float = 0.999, s: float = 0.008):
    """diffusers' squaredcos_cap_v2 alphas_cumprod, EXACTLY.

    betas_for_alpha_bar caps each beta at 0.999 before the cumprod; forming
    abar = f(t)/f(0) directly (what _cosine_alphas_cumprod does) skips that cap
    and bottoms out at the clamp instead. Identical in the middle, ~24x apart at
    the last level, which matters because x0_hat divides by sqrt(abar).
    """
    ab = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
    betas = [min(1.0 - ab((i + 1) / N) / ab(i / N), max_beta) for i in range(N)]
    alphas = 1.0 - torch.tensor(betas, dtype=torch.float64)
    return torch.cumprod(alphas, dim=0).float()


class DPUNetHead(Stage):
    """Diffusion Policy's denoiser as a pipeline head.

    Wraps :class:`ConditionalUnet1D` in ``dp_exact`` mode -- proven bit-identical
    to stock Diffusion Policy's UNet by ``unet_equiv.py`` (same param count, same
    state_dict, 0.0 output difference). DDPM eps-prediction on the cosine
    (``squaredcos_cap_v2``) ladder, which matches diffusers' schedule to 2.3e-07.

    Conditioning is whatever keys ``cond_keys`` names, concatenated:

      ``[a_top]``       -- the DP replica: one global conditioning vector
      ``[a_top, s]``    -- dual-stream DP: the specific stream conditions too

    so the dual-stream and MoE variants are a one-line config change rather than
    a different head. One conditioning vector per token, and the UNet denoises
    that token's ``chunk_len`` action window, so batch = tokens.
    """

    writes = ["pred_action", "loss/dp", "log/dp"]

    def __init__(self, cond_keys: List[str], cond_dims: List[int],
                 action_dim: int, chunk_len: int,
                 num_train_timesteps: int = 100, num_inference_steps: int = 100,
                 down_dims: Optional[List[int]] = None, kernel_size: int = 5,
                 n_groups: int = 8, diffusion_step_embed_dim: int = 128,
                 cond_predict_scale: bool = True, clip_sample: bool = True,
                 sampler: str = "ddpm",
                 embodiments: Optional[List[str]] = None,
                 emit_loss: bool = True):
        super().__init__()
        self.emit_loss = bool(emit_loss)
        if len(cond_keys) != len(cond_dims):
            raise ValueError("DPUNetHead: one cond_dim per cond_key.")
        self.cond_keys = [str(k) for k in cond_keys]
        self.reads = list(self.cond_keys) + ["embodiment"]
        self.C, self.D = int(chunk_len), int(action_dim)
        self.N, self.S = int(num_train_timesteps), int(num_inference_steps)
        self.clip_sample = bool(clip_sample)
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(f"sampler must be ddpm|ddim, got {sampler!r}")
        self.sampler = str(sampler)
        G = int(sum(cond_dims))
        self.net = ConditionalUnet1D(
            input_dim=self.D, cond_dim=G,
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            down_dims=list(down_dims or [512, 1024, 2048]),
            kernel_size=int(kernel_size), n_groups=int(n_groups),
            cond_predict_scale=bool(cond_predict_scale),
            dp_exact=True)
        self.register_buffer("abar", _dp_alphas_cumprod(self.N))
        self.register_buffer(
            "inf_levels", torch.linspace(self.N - 1, 0, self.S).round().long())

    def _cond(self, batch: dict) -> torch.Tensor:
        return torch.cat([batch[k] for k in self.cond_keys], dim=-1)   # (T, G)

    def forward(self, batch: dict) -> dict:
        g = self._cond(batch)
        T, dev = g.shape[0], g.device

        if "target" in batch:
            x0 = batch["target"]                                  # (T,C,D)
            t = torch.randint(0, self.N, (T,), device=dev)
            ab = self.abar[t][:, None, None]
            noise = torch.randn_like(x0)
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            eps = self.net(x_t, t, global_cond=g)
            batch["aux/eps_pred"] = eps
            batch["aux/eps_target"] = noise
            # t is needed to bin the loss by noise level: the uniform-t
            # average hides whether the model solves the LOW-t regime,
            # which is the one that decides action fidelity.
            # (DiffusionDiagnosticEval)
            batch["aux/ddpm_t"] = t
            if self.emit_loss:
                loss = F.mse_loss(eps, noise)
                batch["loss/dp"] = loss
                batch["log/dp"] = loss.detach()

        # Sampling is for ROLLOUT/EVAL only. Without this guard the loop ran on
        # every training step -- 100 extra UNet forwards, all discarded (2.16s
        # of a 2.27s step). DiffusionHead has had this guard all along.
        if self.training:
            return batch
        with torch.no_grad():
            x = torch.randn(T, self.C, self.D, device=dev, dtype=g.dtype)
            for i in range(self.S):
                lv = self.inf_levels[i]
                tt = torch.full((T,), int(lv), device=dev, dtype=torch.long)
                eps = self.net(x, tt, global_cond=g)
                ab = self.abar[lv]
                x0_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
                if self.clip_sample:                    # DP sets clip_sample=True
                    x0_hat = x0_hat.clamp(-1.0, 1.0)
                if i + 1 >= self.S:
                    x = x0_hat
                elif self.sampler == "ddim":
                    ab_prev = self.abar[self.inf_levels[i + 1]]
                    x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
                else:
                    # DDPM ancestral, variance_type="fixed_small" -- what DP
                    # actually samples with. posterior mean from (x0_hat, x_t)
                    # plus noise scaled by the posterior variance.
                    ab_prev = self.abar[self.inf_levels[i + 1]]
                    beta = 1.0 - ab / ab_prev
                    coef_x0 = ab_prev.sqrt() * beta / (1.0 - ab)
                    coef_xt = (ab / ab_prev).sqrt() * (1.0 - ab_prev) / (1.0 - ab)
                    mean = coef_x0 * x0_hat + coef_xt * x
                    var = beta * (1.0 - ab_prev) / (1.0 - ab)
                    x = mean + var.clamp_min(1e-20).sqrt() * torch.randn_like(x)
            batch["pred_action"] = x
        return batch

class _DualStreamBlockAdaLN(_DualStreamBlock):
    """AdaLN-Zero variant of _DualStreamBlock: identical joint masked attention
    and per-stream FFNs, but each branch input is modulated per-position by
    (1+scale)*LN(x)+shift and each branch output gated, with (shift, scale,
    gate) x2 derived from the conditioning stream c. Zero-init modulation =>
    every block is the identity at init (DiT adaLN-Zero)."""

    def __init__(self, d_a: int, d_s: int, n_heads: int, ffn_mult: int = 4):
        super().__init__(d_a, d_s, n_heads, ffn_mult)
        self.mod_a = nn.Sequential(nn.SiLU(), nn.Linear(d_a, 6 * d_a))
        self.mod_s = nn.Sequential(nn.SiLU(), nn.Linear(d_s, 6 * d_s))
        for m in (self.mod_a[1], self.mod_s[1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, A, S, allow, cA=None, cS=None):
        T, L, _ = A.shape
        hd = self.dm // self.h
        sh_a1, sc_a1, g_a1, sh_a2, sc_a2, g_a2 = self.mod_a(cA).chunk(6, -1)
        sh_s1, sc_s1, g_s1, sh_s2, sc_s2, g_s2 = self.mod_s(cS).chunk(6, -1)

        def _heads(x):
            q, k, v = x.chunk(3, -1)
            return [y.reshape(T, -1, self.h, hd).transpose(1, 2)
                    for y in (q, k, v)]

        qa, ka, va = _heads(self.qkv_a(self.norm_a1(A) * (1 + sc_a1) + sh_a1))
        qs, ks, vs = _heads(self.qkv_s(self.norm_s1(S) * (1 + sc_s1) + sh_s1))
        q = torch.cat([qa, qs], dim=2)
        k = torch.cat([ka, ks], dim=2)
        v = torch.cat([va, vs], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        o = o.transpose(1, 2).reshape(T, 2 * L, self.dm)
        A = A + g_a1 * self.out_a(o[:, :L])
        S = S + g_s1 * self.out_s(o[:, L:])
        A = A + g_a2 * self.ffn_a(self.norm_a2(A) * (1 + sc_a2) + sh_a2)
        S = S + g_s2 * self.ffn_s(self.norm_s2(S) * (1 + sc_s2) + sh_s2)
        return A, S


class DualStreamDenoiserV2(DualStreamDenoiser):
    """AdaLN-Zero denoiser: token = in(x_t) + pos only (content + slot id);
    c = temb(t) + cond_proj(a_top | s) is re-injected at EVERY block as
    per-position scale/shift/gate, plus a modulated final LN before vout.
    Same in/cond/temb/pos/vout modules and attention dims as v1 — only the
    injection point and interaction type change."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        dA = self.pos_a.shape[1]
        dS = self.pos_s.shape[1]
        n_layers = len(self.blocks)
        n_heads = self.blocks[0].h
        ffn_mult = self.blocks[0].ffn_a[0].out_features // dA
        self.blocks = nn.ModuleList(
            _DualStreamBlockAdaLN(dA, dS, n_heads, ffn_mult)
            for _ in range(n_layers))
        self.fmod_a = nn.Sequential(nn.SiLU(), nn.Linear(dA, 2 * dA))
        self.fmod_s = nn.Sequential(nn.SiLU(), nn.Linear(dS, 2 * dS))
        for m in (self.fmod_a[1], self.fmod_s[1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x_t, t, a_top, s, emb: str, x_s=None):
        # x_s: optional SEPARATE input for the S stream (per-stream codec).
        # None -> both streams read x_t, the original shared-codec behaviour.
        T, L, _ = x_t.shape
        e = emb if emb in self.in_s else next(iter(self.in_s))
        cA = self.cond_a(a_top)[:, None, :] + self._temb(self.temb_a, t, T, L)
        cS = self.cond_s[e](s)[:, None, :] + self._temb(self.temb_s, t, T, L)
        A = self.in_a(x_t) + self.pos_a[None, :L]
        S = self.in_s[e](x_t if x_s is None else x_s) + self.pos_s[None, :L]
        allow = self._allow(L, x_t.device)
        for blk in self.blocks:
            A, S = blk(A, S, allow, cA, cS)
        sh, sc = self.fmod_a(cA).chunk(2, -1)
        v_a = self.vout_a(self.norm_fa(A) * (1 + sc) + sh)
        sh, sc = self.fmod_s(cS).chunk(2, -1)
        v_s = self.vout_s[e](self.norm_fs(S) * (1 + sc) + sh)
        return v_a + v_s, v_a, v_s


class DualStreamDenoiserHetero(DualStreamDenoiserV2):
    """Hetero-action-dim denoiser with DIRECT per-embodiment projections.

    Alternative to LatentRHDenoiser. That wrapper keeps the core homogeneous by
    squeezing every embodiment through a common action-width waist L:

        act_dim_e -> [E_e: 64] -> L -> [core in: d_model] -> L -> [D_e: 64] -> act_dim_e

    with L pinned to max(action_dims), so the adapters' hidden width cannot
    carry more than L dimensions of information and the per-emb capacity sits
    behind a choke. Here the adapters instead map STRAIGHT to the trunk width:

        act_dim_e -> [in_a[e] / in_s[e]: MLP] -> d_model_a / d_model_s
        d_model_* -> [vout_a[e] / vout_s[e]: MLP] -> act_dim_e

    so per-emb capacity is spent against the 256/128-wide trunk and there is no
    waist. Consequence to be aware of: `in_a` and `vout_a` become PER-EMB (a
    single Linear cannot span different input widths), so the A stream's input
    and output projections are no longer shared across embodiments -- only the
    transformer blocks are. The latent variant kept those two shared.
    """

    def __init__(self, *args, action_dims: dict, adapter_layers: int = 4,
                 adapter_hidden: Optional[int] = None, **kw):
        Dmap = {str(e): int(d) for e, d in dict(action_dims).items()}
        # build the parent at the widest dim so every non-projection shape is
        # valid, then replace all four projection banks with per-emb MLPs
        super().__init__(*args, action_dim=max(Dmap.values()), **kw)
        dA, dS = self.pos_a.shape[1], self.pos_s.shape[1]
        self.Dmap = Dmap

        def mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
            n = max(1, int(adapter_layers))
            if n == 1:
                return nn.Sequential(nn.Linear(int(d_in), int(d_out)))
            layers = [nn.Linear(int(d_in), hidden), nn.SiLU()]
            for _ in range(n - 2):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers.append(nn.Linear(hidden, int(d_out)))
            return nn.Sequential(*layers)

        hA = int(adapter_hidden) if adapter_hidden else dA
        hS = int(adapter_hidden) if adapter_hidden else dS
        self.in_a = nn.ModuleDict({e: mlp(Dmap[e], dA, hA) for e in Dmap})
        self.in_s = nn.ModuleDict({e: mlp(Dmap[e], dS, hS) for e in Dmap})
        self.vout_a = nn.ModuleDict({e: mlp(dA, Dmap[e], hA) for e in Dmap})
        self.vout_s = nn.ModuleDict({e: mlp(dS, Dmap[e], hS) for e in Dmap})

    def forward(self, x_t, t, a_top, s, emb: str):
        T, L, _ = x_t.shape
        e = emb if emb in self.in_s else next(iter(self.in_s))
        cA = self.cond_a(a_top)[:, None, :] + self._temb(self.temb_a, t, T, L)
        cS = self.cond_s[e](s)[:, None, :] + self._temb(self.temb_s, t, T, L)
        A = self.in_a[e](x_t) + self.pos_a[None, :L]
        S = self.in_s[e](x_t) + self.pos_s[None, :L]
        allow = self._allow(L, x_t.device)
        for blk in self.blocks:
            A, S = blk(A, S, allow, cA, cS)
        sh, sc = self.fmod_a(cA).chunk(2, -1)
        v_a = self.vout_a[e](self.norm_fa(A) * (1 + sc) + sh)
        sh, sc = self.fmod_s(cS).chunk(2, -1)
        v_s = self.vout_s[e](self.norm_fs(S) * (1 + sc) + sh)
        return v_a + v_s, v_a, v_s


# --------------------------------------------------------------------------- #
# Hetero-action-dim robot-human denoiser (E_e / core-in-latent / D_e)
# --------------------------------------------------------------------------- #
class _SingleStreamBlockAdaLN(nn.Module):
    """AdaLN-Zero DiT block: full self-attention over the L buffer positions +
    FFN, each modulated by (1+scale)*LN(x)+shift and output-gated from the
    conditioning c (shift, scale, gate) x2. Zero-init modulation => identity at
    init."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4,
                 moe_experts: int = 0, moe_top_k: int = 4,
                 moe_d_expert: Optional[int] = None,
                 moe_aux_weight: float = 0.01):
        super().__init__()
        self.h, self.d = int(n_heads), int(d)
        assert self.d % self.h == 0
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        # MoE swaps ONLY the FFN -- attention, adaLN modulation and the
        # residual structure are untouched, which is the standard way experts
        # enter a transformer block.
        if moe_experts:
            self.ffn = MoEFFN(d, int(moe_d_expert or ffn_mult * d),
                              num_experts=int(moe_experts),
                              top_k=int(moe_top_k),
                              aux_weight=float(moe_aux_weight))
        else:
            self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                     nn.Linear(ffn_mult * d, d))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, X, c):                     # X (T,L,d)  c (T,L,d)
        T, L, _ = X.shape
        hd = self.d // self.h
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(c).chunk(6, -1)
        q, k, v = self.qkv(self.norm1(X) * (1 + sc1) + sh1).chunk(3, -1)
        q, k, v = [y.reshape(T, L, self.h, hd).transpose(1, 2) for y in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(T, L, self.d)
        X = X + g1 * self.out(o)
        X = X + g2 * self.ffn(self.norm2(X) * (1 + sc2) + sh2)
        return X


class SingleStreamDenoiserV2(nn.Module):
    """Single-stream adaLN-Zero denoiser in the latent (no-dual RH variant):
    token = in(z_t) + pos; c = temb(t) + cond_a(a_top) + cond_s(s) re-injected
    at every block as per-position scale/shift/gate, plus a modulated final LN.
    A/S survive ONLY as the two conditioning projections — there is no S
    denoiser stream and no per-emb in/out (E_e/D_e own the per-emb mapping).
    Returns (v, None, None): no v_a/v_s partition => no vA_frac probe."""

    def __init__(self, d_a_in: int, d_s_in: Optional[int], action_dim: int,
                 chunk_len: int,
                 d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 ffn_mult: int = 4, n_positions: Optional[int] = None,
                 moe_experts: int = 0, moe_top_k: int = 4,
                 moe_d_expert: Optional[int] = None,
                 moe_aux_weight: float = 0.01):
        super().__init__()
        C, D = int(chunk_len), int(action_dim)
        L = int(n_positions) if n_positions else C
        d = int(d_model)
        self.C, self.D, self.L = C, D, L
        self.in_x = nn.Linear(D, d)
        self.cond_a = nn.Linear(int(d_a_in), d)
        # d_s_in None -> no S conditioning at all (the DP-replica obs path
        # produces a_top only). Otherwise A/S enter as two summed projections.
        self.cond_s = nn.Linear(int(d_s_in), d) if d_s_in else None
        self.temb = nn.Sequential(SinusoidalPosEmb(d),
                                  nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(L, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.vout = nn.Linear(d, D)
        self.blocks = nn.ModuleList(
            _SingleStreamBlockAdaLN(d, n_heads, ffn_mult,
                                    moe_experts=moe_experts,
                                    moe_top_k=moe_top_k,
                                    moe_d_expert=moe_d_expert,
                                    moe_aux_weight=moe_aux_weight)
            for _ in range(int(n_layers)))
        self.norm_f = nn.LayerNorm(d)
        self.fmod = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.fmod[1].weight)
        nn.init.zeros_(self.fmod[1].bias)

    def _temb(self, t, T, L):
        if t.dim() == 1:
            return self.temb(t)[:, None, :].expand(T, L, -1)
        return self.temb(t.reshape(-1)).reshape(T, L, -1)

    def forward(self, x_t, t, a_top, s, emb: Optional[str] = None):
        T, L, _ = x_t.shape
        c = self.cond_a(a_top)[:, None, :] + self._temb(t, T, L)
        if self.cond_s is not None and s is not None:
            c = c + self.cond_s(s)[:, None, :]
        X = self.in_x(x_t) + self.pos[None, :L]
        for blk in self.blocks:
            X = blk(X, c)
        sh, sc = self.fmod(c).chunk(2, -1)
        v = self.vout(self.norm_f(X) * (1 + sc) + sh)
        return v, None, None


class LatentRHDenoiser(nn.Module):
    """Hetero-action-dim wrapper: per-embodiment ENCODER E_e
    (action_dim_e -> latent L) + DECODER D_e (latent L -> action_dim_e) around
    a SHARED denoiser core that runs ENTIRELY in the latent. Adapter width,
    depth, and residual mode may be scalars (legacy/same for every embodiment)
    or mappings keyed by embodiment. Noise, x_t and the loss stay in the
    ORIGINAL per-emb dim (the head owns that); the core only ever sees the
    common latent L (RAE: L must be >= the max per-emb noise dim, else the core
    cannot denoise the largest embodiment).

    dual_stream=True  : core = DualStreamDenoiserV2 (A shared / S per-emb, additive
        v = v_a + v_s in the latent, vA_frac probe). E_e feeds in_a AND in_s[e];
        D_e maps the SUMMED latent velocity -> action_dim_e.
    dual_stream=False : core = SingleStreamDenoiserV2 (one shared stream; A/S only
        as a_top vs s conditioning). E_e/D_e are the ONLY per-emb parts.

    Interface identical to the denoiser core: forward(x_t, t, a_top, s, emb) ->
    (v, v_a, v_s) with v in action_dim_e and v_a/v_s the latent partition
    (None,None for the single-stream variant)."""

    def __init__(self, d_a_in: int, d_s_in: int, action_dims: dict, latent_dim: int,
                 chunk_len: int, embodiments: List[str],
                 d_model_a: int = 256, d_model_s: int = 192,
                 n_layers: int = 4, n_heads: int = 4, ffn_mult: int = 4,
                 mask_mode: str = "sym", n_positions: Optional[int] = None,
                 dual_stream: bool = True, enc_hidden=256,
                 dual_arch: str = "adaln", enc_layers=3,
                 enc_residual=False, enc_per_stream: bool = False,
                 moe_experts: int = 0, moe_top_k: int = 4,
                 moe_d_expert: Optional[int] = None,
                 moe_aux_weight: float = 0.01):
        super().__init__()
        self.dual_stream = bool(dual_stream)
        self.enc_per_stream = bool(enc_per_stream)
        self.latent = int(latent_dim)
        embs = [str(e) for e in embodiments] if embodiments else ["shared"]
        Dmap = {str(e): int(d) for e, d in dict(action_dims).items()}
        assert self.latent >= max(Dmap.values()), (
            f"latent_dim {self.latent} < max action_dim {max(Dmap.values())} "
            f"(RAE: latent must cover the largest per-emb noise dim)")
        L = self.latent

        def _emb_value(value, emb, name, cast):
            if hasattr(value, "items"):
                default = value.get("default") if hasattr(value, "get") else None
                picked = value.get(emb, default)
                if picked is None:
                    raise ValueError(f"{name} has no value for embodiment {emb!r}")
                return cast(picked)
            return cast(value)

        def _mlp(emb, d_in, d_out):
            H = _emb_value(enc_hidden, emb, "enc_hidden", int)
            n_mlp = max(2, _emb_value(enc_layers, emb, "enc_layers", int))
            residual = _emb_value(enc_residual, emb, "enc_residual", bool)
            layers = [nn.Linear(int(d_in), H), nn.SiLU()]
            for _ in range(n_mlp - 2):
                if residual:
                    layers.append(_ResidualMLPBlock(H))
                else:
                    layers += [nn.Linear(H, H), nn.SiLU()]
            layers.append(nn.Linear(H, int(d_out)))
            return nn.Sequential(*layers)

        self.enc = nn.ModuleDict({e: _mlp(e, Dmap[e], L) for e in embs})  # E_e
        self.dec = nn.ModuleDict({e: _mlp(e, L, Dmap[e]) for e in embs})  # D_e
        # PER-STREAM codec: each stream gets its own view of the action instead
        # of sharing one z. enc/dec above are then used for the A stream only.
        if self.enc_per_stream:
            if not self.dual_stream:
                raise ValueError("enc_per_stream requires dual_stream=True")
            self.enc_s_codec = nn.ModuleDict({e: _mlp(e, Dmap[e], L) for e in embs})
            self.dec_s_codec = nn.ModuleDict({e: _mlp(e, L, Dmap[e]) for e in embs})

        if self.dual_stream:
            core_cls = (DualStreamDenoiserV2 if str(dual_arch) == "adaln"
                        else DualStreamDenoiser)
            self.core = core_cls(
                d_a_in=d_a_in, d_s_in=d_s_in, action_dim=L, chunk_len=chunk_len,
                embodiments=embs, d_model_a=d_model_a, d_model_s=d_model_s,
                n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult,
                mask_mode=mask_mode, n_positions=n_positions)
        else:
            self.core = SingleStreamDenoiserV2(
                d_a_in=d_a_in, d_s_in=d_s_in, action_dim=L, chunk_len=chunk_len,
                d_model=d_model_a, n_layers=n_layers, n_heads=n_heads,
                ffn_mult=ffn_mult, n_positions=n_positions,
                moe_experts=moe_experts, moe_top_k=moe_top_k,
                moe_d_expert=moe_d_expert, moe_aux_weight=moe_aux_weight)

    def forward(self, x_t, t, a_top, s, emb: str):
        e = emb if emb in self.enc else next(iter(self.enc))
        if self.enc_per_stream:
            z_a = self.enc[e](x_t)                     # A-stream view
            z_s = self.enc_s_codec[e](x_t)             # S-stream view
            _, v_a, v_s = self.core(z_a, t, a_top, s, emb, x_s=z_s)
            # decode EACH stream with its own decoder, sum in ACTION space
            d_a = self.dec[e](v_a)
            d_s = self.dec_s_codec[e](v_s)
            return d_a + d_s, d_a, d_s
        z = self.enc[e](x_t)                     # (T, L_pos, latent)
        v_lat, v_a, v_s = self.core(z, t, a_top, s, emb)
        v = self.dec[e](v_lat)                   # (T, L_pos, action_dim_e)
        return v, v_a, v_s


class MaskedActionLoss(Stage):
    """DDPM eps-MSE with per-embodiment action dims EXCLUDED from the score.

    Reads ``aux/eps_pred`` / ``aux/eps_target`` (published by DiffusionHead when
    a loss stage owns the objective) and writes ``loss/<name>``.

    ``exclude_dims`` maps an embodiment to the action-dim indices that must not
    be scored. The motivating case: with the gripper included the action is
    20-D per the layout [L xyz(3) rot6d(6) grip(1), R xyz(3) rot6d(6) grip(1)],
    so the gripper slots are dims 9 and 19. eva has both; Aria has neither, and
    scoring its unpopulated gripper columns would train the model to regress a
    constant and pollute the reported MSE. The action space stays MATCHED at 20
    -- this is a masked loss, not a hetero head.

    Excluding a dim for EVERY embodiment would silently make it untrained while
    still being emitted at rollout, so that is refused.
    """

    reads = ["aux/eps_pred", "aux/eps_target", "embodiment"]

    def __init__(self, exclude_dims: Optional[dict] = None,
                 name: str = "ddpm", embodiments: Optional[List[str]] = None,
                 weights: Optional[dict] = None):
        super().__init__()
        self.name = str(name)
        # Per-embodiment loss weight. Scales that stream's gradient into the
        # SHARED core as well as its own encoders -- which per-param-group LRs
        # cannot do, since the shared trunk is one tensor set. Default 1.0
        # leaves every existing run byte-identical.
        self.weights = {str(k): float(v) for k, v in (weights or {}).items()}
        self.writes = [f"loss/{self.name}", f"log/{self.name}"]
        self.exclude = {str(k): [int(i) for i in v]
                        for k, v in (exclude_dims or {}).items()}
        embs = [str(e) for e in (embodiments or [])]
        if embs and self.exclude:
            common = set.intersection(*[set(self.exclude.get(e, [])) for e in embs]) \
                if all(e in self.exclude for e in embs) else set()
            if common:
                raise ValueError(
                    f"MaskedActionLoss: dims {sorted(common)} are excluded for "
                    f"EVERY embodiment {embs} -- they would never be trained yet "
                    f"still be emitted at rollout. Drop them from the action "
                    f"space instead.")

    def forward(self, batch: dict) -> dict:
        pred = batch.get("aux/eps_pred")
        if pred is None:                      # rollout / eval: nothing to score
            return batch
        tgt = batch["aux/eps_target"]
        emb = str(batch["embodiment"])
        drop = self.exclude.get(emb)
        if drop:
            D = pred.shape[-1]
            bad = [d for d in drop if d >= D]
            if bad:
                raise IndexError(
                    f"MaskedActionLoss: exclude_dims {bad} out of range for "
                    f"embodiment {emb!r} with action_dim {D}. The gripper dims "
                    f"are only valid on the 20-D cartesian layout.")
            keep = torch.ones(D, dtype=torch.bool, device=pred.device)
            keep[torch.tensor(drop, device=pred.device)] = False
            pred, tgt = pred[..., keep], tgt[..., keep]
        loss = F.mse_loss(pred, tgt)
        w = self.weights.get(emb, 1.0)
        batch[f"loss/{self.name}"] = loss * w
        # LOG THE UNWEIGHTED loss on purpose: the weight changes the gradient,
        # not the quantity we compare across cells. Logging loss*w would make
        # a down-weighted run look better than an unweighted one for free.
        batch[f"log/{self.name}"] = float(loss)
        batch[f"log/{self.name}_weight"] = float(w)
        batch[f"log/{self.name}_dims_scored"] = float(pred.shape[-1])
        return batch


# ------------------------------------------------------------------------- #
# TOKEN-CONDITIONING ABLATION (user, 2026-08-18)
# The DP failure survives every head (UNet/single/dual/MoE) and every encoder
# (VisualCore/HPTVisualEncoder), and cond_gain ~= 0.3%: the model ignores its
# observation. Common factor = the obs is POOLED into one a_top/s vector and
# injected via AdaLN. HPT instead keeps per-modality TOKENS and ATTENDS to
# them. This head is the single-variable test: same DDPM eps head, but the
# action chunk CROSS-ATTENDS to obs_tokens (from ObsEncoders.expose_tokens)
# instead of AdaLN on the pooled vector.
# ------------------------------------------------------------------------- #
class _XAttnBlock(nn.Module):
    """DiT-ish block: self-attn over the chunk positions + cross-attn to obs
    tokens + FFN. Self-contained MHA so shapes are unambiguous."""
    def __init__(self, d: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.d = int(n_heads), int(d)
        self.n1 = nn.LayerNorm(d); self.sq = nn.Linear(d, d); self.sk = nn.Linear(d, d)
        self.sv = nn.Linear(d, d); self.so = nn.Linear(d, d)
        self.nc = nn.LayerNorm(d); self.cq = nn.Linear(d, d); self.ck = nn.Linear(d, d)
        self.cv = nn.Linear(d, d); self.co = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(),
                                nn.Linear(d * ffn_mult, d))
        self.drop = nn.Dropout(dropout)

    def _mha(self, q, k, v):                       # (T,Nq,d),(T,Nk,d),(T,Nk,d)
        T, Nq, d = q.shape; h = self.h; hd = d // h
        q = q.view(T, Nq, h, hd).transpose(1, 2)
        k = k.view(T, -1, h, hd).transpose(1, 2)
        v = v.view(T, -1, h, hd).transpose(1, 2)
        a = (q @ k.transpose(-1, -2)) * (hd ** -0.5)
        a = a.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(T, Nq, d)
        return o

    def forward(self, x, ctx):                     # x (T,C,d)  ctx (T,K,d)
        s = self.n1(x); x = x + self.so(self._mha(self.sq(s), self.sk(s), self.sv(s)))
        q = self.nc(x); x = x + self.co(self._mha(self.cq(q), self.ck(ctx), self.cv(ctx)))
        x = x + self.ff(self.n2(x))
        return x


class XAttnDiffusionHead(Stage):
    """DDPM eps head whose action chunk cross-attends to obs_tokens.

    reads obs_tokens (T,K,d_token) + target (train). No a_top/s AdaLN -- the
    ONLY conditioning is cross-attention to the per-modality tokens. This is
    the pooled-vs-token ablation; everything else (DDPM ladder, chunk, DDIM
    sampling) mirrors DiffusionHead.
    """
    reads = ["obs_tokens", "embodiment"]
    writes = ["pred_action", "aux/eps_pred", "aux/eps_target", "loss/ddpm"]

    def __init__(self, action_dim: int, chunk_len: int, d_token: int,
                 embodiments=None, d_model: int = 384, n_layers: int = 6,
                 n_heads: int = 6, ffn_mult: int = 4, dropout: float = 0.1,
                 num_train_timesteps: int = 100, num_inference_steps: int = 16,
                 emit_loss: bool = True):
        super().__init__()
        self.C, self.D = int(chunk_len), int(action_dim)
        self.N, self.S = int(num_train_timesteps), int(num_inference_steps)
        self.emit_loss = bool(emit_loss)
        self.register_buffer("abar", _cosine_alphas_cumprod(self.N))
        self.register_buffer("inf_levels",
                             torch.linspace(self.N - 1, 0, self.S).round().long())
        self.in_proj = nn.Linear(self.D, d_model)
        self.ctx_proj = nn.Linear(int(d_token), d_model)
        self.pos = nn.Parameter(torch.zeros(self.C, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.temb = nn.Sequential(SinusoidalPosEmb(d_model), nn.Linear(d_model, d_model),
                                  nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(
            _XAttnBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(int(n_layers)))
        self.norm_f = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, self.D)

    def _denoise(self, x_t, t, ctx):               # (T,C,D),(T,),(T,K,d)
        h = self.in_proj(x_t) + self.pos[None]
        h = h + self.temb(t.float())[:, None, :]
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.out(self.norm_f(h))

    def forward(self, batch: dict) -> dict:
        ot = batch.get("obs_tokens")
        if ot is None:
            raise KeyError("XAttnDiffusionHead needs batch['obs_tokens'] -- set "
                           "ObsEncoders.expose_tokens=True and the specific "
                           "obs_encoder per_obs_keys=True.")
        ctx = self.ctx_proj(ot)                    # (T,K,d_model)
        if "target" in batch:
            x0 = batch["target"]                   # (T,C,D)
            T = x0.shape[0]
            t = torch.randint(0, self.N, (T,), device=x0.device)
            ab = self.abar[t][:, None, None]
            noise = torch.randn_like(x0)
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            eps = self._denoise(x_t, t, ctx)
            batch["aux/eps_pred"] = eps
            batch["aux/eps_target"] = noise
            batch["aux/ddpm_t"] = t
            if self.emit_loss:
                loss = F.mse_loss(eps, noise)
                batch["loss/ddpm"] = loss
                batch["log/ddpm"] = float(loss)
        if not self.training:
            with torch.no_grad():
                T = ctx.shape[0]
                x = torch.randn(T, self.C, self.D, device=ctx.device, dtype=ctx.dtype)
                for j in range(self.S):
                    tl = int(self.inf_levels[j])
                    tt = torch.full((T,), float(tl), device=x.device, dtype=x.dtype)
                    eps = self._denoise(x, tt, ctx)
                    ab_t = self.abar[tl]
                    x0p = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1.0, 1.0)
                    if j + 1 < self.S:
                        ab_n = self.abar[int(self.inf_levels[j + 1])]
                        x = ab_n.sqrt() * x0p + (1 - ab_n).sqrt() * eps
                    else:
                        x = x0p
                batch["pred_action"] = x.clamp(-1.0, 1.0)
        return batch
