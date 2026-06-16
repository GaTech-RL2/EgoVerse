"""Rung-0 transformer baseline: per-frame obs -> action chunk (HPT-structural).

This is the *obs-only, chunk-predicting* transformer to build up toward H-Net
from. It is deliberately structured like HPT and deliberately NOT like H-Net:

* **No action tokens** — the model never sees a previous action, so the
  copy-cheat is structurally impossible (like HPT).
* **Spatial image tokens** — the conv feature map is emitted as a grid of
  tokens (not pooled to one vector), and a (non-causal) transformer attends
  over ``[img_tokens..., proprio_token]`` so the policy can read *where* the
  T / goal / pusher are. This is the spatial conditioning HPT has and the
  H-Net global-pool encoder lacked.
* **Chunk head + temporal ensemble** — each forward predicts a ``chunk_k``
  action chunk; the rollout re-encodes the current obs every env step and
  temporally-ensembles overlapping chunks (the existing ``chunk_te`` rollout
  in ``HNetSimEval``). This is the "commit to a trajectory" mechanism that
  single-step AR lacks.
* **Markovian** — the chunk depends only on the current frame's obs, so
  training (per-frame chunk targets) and the chunk_te rollout (re-encode each
  step) match exactly; no KV cache, no exposure-bias gap.

The build-up ladder from here: add prev-action tokens (rung 1) -> single-step
AR (rung 2) -> dynamic chunking (rung 3, = full H-Net), and watch which rung
drops coverage from ~HPT to ~0.

Reuses the H-Net Isotropic backbone, the H-Net packed data pipeline, and the
``HNetFused`` algo plumbing (``process_batch_for_training`` / ``_build_obs`` /
``compute_losses`` / ``log_info``) so the morph toward H-Net is continuous.
"""

import math
import re
from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn


def _split_arch_layout(arch_layout: str):
    """Split a single-block-type arch_layout like ``"T4"`` into three
    sub-layouts ``(enc, inner, dec)`` whose counts sum to the original, split
    roughly evenly with the middle (inner) group getting the remainder.

    ``"T4" -> ("T1", "T2", "T1")``; ``"T3" -> ("T1", "T1", "T1")``;
    ``"T2" -> ("T1", "T1", "T0"?)`` — we guarantee >=1 in enc and inner and put
    the rest in dec (which may be 0, handled by the caller skipping empty groups).
    """
    m = re.fullmatch(r"\s*([mMtT])(\d+)\s*", str(arch_layout))
    if not m:
        raise ValueError(
            f"hnet_chunked backbone needs a single-token arch_layout like 'T4', got {arch_layout!r}"
        )
    letter, n = m.group(1), int(m.group(2))
    if n < 2:
        raise ValueError(
            f"hnet_chunked backbone needs arch_layout count >= 2 (to span enc/inner/dec), got {arch_layout!r}"
        )
    enc = max(1, n // 4)
    dec = max(1, n // 4)
    inner = n - enc - dec
    if inner < 1:
        # Re-balance so inner gets at least 1 (shrink dec, then enc).
        inner = 1
        rem = n - inner
        enc = max(1, rem // 2)
        dec = rem - enc
    return f"{letter}{enc}", f"{letter}{inner}", f"{letter}{dec}"


class _SinusoidalTime(nn.Module):
    """Sinusoidal embedding of a scalar flow time t in [0,1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        a = t[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)  # (B, dim)


class FlowHead(nn.Module):
    """Minimal flow-matching action-chunk head (conditional rectified flow).

    A residual MLP denoiser predicts the velocity ``noise - actions`` along the
    linear path ``x_t = t*noise + (1-t)*actions``, conditioned on a global
    feature. ``sample`` integrates that velocity ODE from noise -> action chunk,
    so it COMMITS to one mode (e.g. a rotation direction) instead of MSE-averaging
    multimodal actions into "no rotation". Same masked-loss interface as the
    regression head (returns pred/target velocities for the algo to MSE+mask).
    """

    def __init__(self, chunk_k, action_dim, cond_dim, hidden=512, time_dim=128, n_steps=10):
        super().__init__()
        self.chunk_k = int(chunk_k)
        self.action_dim = int(action_dim)
        self.n_steps = int(n_steps)
        self.flat = self.chunk_k * self.action_dim
        self.time_emb = _SinusoidalTime(time_dim)
        self.net = nn.Sequential(
            nn.Linear(self.flat + time_dim + cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.flat),
        )

    def _vel(self, x, t, cond):  # x:(B,K,A) t:(B,) cond:(B,Dc) -> (B,K,A)
        B = x.shape[0]
        inp = torch.cat([x.reshape(B, -1), self.time_emb(t), cond], dim=-1)
        return self.net(inp).reshape(B, self.chunk_k, self.action_dim)

    def loss(self, cond, actions):
        """Returns (pred_velocity, target_velocity), both (B,K,A); algo applies
        the (masked) MSE between them."""
        noise = torch.randn_like(actions)
        t = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(actions.device)
        t = t * 0.999 + 0.001
        te = t[:, None, None]
        x_t = te * noise + (1.0 - te) * actions
        target = noise - actions
        pred = self._vel(x_t, t, cond)
        return pred, target

    @torch.no_grad()
    def sample(self, cond):  # cond:(B,Dc) -> (B,K,A)
        B = cond.shape[0]
        x = torch.randn(B, self.chunk_k, self.action_dim, device=cond.device)
        dt = -1.0 / self.n_steps
        t = torch.ones(B, device=cond.device)
        for _ in range(self.n_steps):
            x = x + dt * self._vel(x, t, cond)
            t = t + dt
        return x

from egomimic.algo.algo import Algo
from egomimic.algo.hnet.fused import HNetFused, resolve_embodiment_keys
from egomimic.models.cores.hpt_transformer import CrossAttention


def _build_chunk_targets_packed(actions, cu_seqlens, K):
    """Packed actions ``(T_total, A)`` -> (target ``(T_total, K, A)``, mask
    ``(T_total, K)``). ``target[i, k] = actions[i + k]`` if ``i + k`` is within
    frame ``i``'s episode, else 0; ``mask`` marks the valid (non-tail) entries.
    """
    T_total, A = actions.shape
    device = actions.device
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    pos = torch.arange(T_total, device=device)
    seq_idx = (pos[:, None] >= cu[None, 1:]).sum(dim=-1)  # episode index per frame
    e_i = cu[seq_idx + 1]                                 # episode end (exclusive)
    ks = torch.arange(K, device=device)
    idx = pos[:, None] + ks[None, :]                      # (T_total, K) target frames
    valid = idx < e_i[:, None]                            # within-episode mask
    idx_c = torch.minimum(idx, e_i[:, None] - 1)          # clamp tail to last valid
    target = actions[idx_c]                               # (T_total, K, A)
    target = target * valid[..., None].to(target.dtype)
    return target, valid.to(actions.dtype)


def _build_chunk_targets_padded(actions, seq_lens, K):
    """Padded actions ``(B, T, A)`` -> (target ``(B, T, K, A)``, mask
    ``(B, T, K)``). Fallback path; packed is the trained-on format."""
    B, T, A = actions.shape
    device = actions.device
    ks = torch.arange(K, device=device)
    pos = torch.arange(T, device=device)
    idx = pos[:, None] + ks[None, :]            # (T, K)
    idx_c = idx.clamp(max=T - 1)
    target = actions[:, idx_c]                  # (B, T, K, A)
    if seq_lens is not None:
        e = seq_lens.to(device).view(B, 1, 1)
        valid = (idx[None] < e)                 # (B, T, K)
    else:
        valid = (idx < T)[None].expand(B, -1, -1)
    target = target * valid[..., None].to(target.dtype)
    return target, valid.to(actions.dtype)


class ChunkTokenPolicy(nn.Module):
    """Image -> spatial tokens + proprio token -> transformer -> pool -> chunk.

    Markovian, obs-only. ``forward``/``forward_packed`` ignore the ``actions``
    argument (kept for a uniform policy interface) and predict a ``chunk_k``
    chunk from each frame's obs. ``generate`` returns a single one-shot chunk
    for the ``chunk_te`` rollout.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_k: int,
        d_model: int,
        proprio_dim: int = 2,
        input_slice: Sequence[int] = (0, 2),
        img_key: str = "front_img_1",
        proprio_key: str = "state_agent_obj",
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        stride: int = 2,
        image_size: int = 96,
        norm_groups: int = 8,
        arch_layout: str = "T4",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        action_horizon: int = 1024,
        readout: str = "mean_pool",
        stem: str = "none",
        stem_latents: int = 16,
        encoder: str = "conv",
        resnet_model: str = "resnet18",
        pretrained: bool = True,
        head: str = "regression",
        flow_steps: int = 10,
        backbone: str = "flat",
        chunk_compress_ratio: float = 4.0,
        ratio_loss_weight: float = 0.03,
    ):
        super().__init__()
        from egomimic.models.hnet.isotropic_builder import build_isotropic

        self.action_dim = action_dim
        self.chunk_k = int(chunk_k)
        self.d_model = d_model
        self.img_key = img_key
        self.proprio_key = proprio_key
        # readout: "mean_pool" (avg obs tokens -> one vector -> Linear to K*A) or
        # "action_token" (HPT-style: append K learnable query tokens, the
        # transformer reads each out -> per-action head). action_token avoids
        # the spatial-averaging of mean-pool.
        self.readout = str(readout)
        self.islice = (int(input_slice[0]), int(input_slice[1]))
        # action_horizon is only consulted by the evaluator as a rollout-length
        # cap; set it large so the chunk_te rollout runs the full episode.
        self.action_horizon = int(action_horizon)

        # Image encoder -> spatial tokens (N, M, d_model).
        #   "conv":   from-scratch GroupNorm convnet (no global pool).
        #   "resnet": HPT's (optionally ImageNet-pretrained) ResNet, reused
        #             verbatim so this is the exact HPT vision backbone.
        self.encoder = str(encoder)
        if self.encoder == "resnet":
            from egomimic.models.stems.hpt_stems import ResNet
            self.img_encoder = ResNet(
                output_dim=d_model,
                resnet_model=resnet_model,
                weights="DEFAULT" if pretrained else None,
            )
            self.img_proj = nn.Identity()                # ResNet already proj's to d_model
            with torch.no_grad():
                _f = self.img_encoder(torch.zeros(1, in_channels, image_size, image_size))
                self._n_img_tok = int(_f.shape[1])       # (1, M, d_model)
        else:
            convs = []
            c = in_channels
            for c_out in channels:
                convs += [
                    nn.Conv2d(c, c_out, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.GroupNorm(min(norm_groups, c_out), c_out),
                    nn.GELU(),
                ]
                c = c_out
            self.img_conv = nn.Sequential(*convs)
            with torch.no_grad():
                _d = torch.zeros(1, in_channels, image_size, image_size)
                _f = self.img_conv(_d)                   # (1, C', h, w)
                self._n_img_tok = int(_f.shape[2] * _f.shape[3])
                _c_last = int(_f.shape[1])
            self.img_proj = nn.Linear(_c_last, d_model)
        self.proprio_proj = nn.Linear(self.islice[1] - self.islice[0], d_model)
        # Sequence = [img tokens, proprio token] (+ K action-query tokens in
        # action_token mode). pos_emb covers the full sequence.
        # Optional HPT-style cross-attention stem: pool the obs tokens into
        # ``stem_latents`` learned latents before the trunk (a strong learnable
        # bottleneck that extracts task-relevant, goal-relative features).
        self.stem = str(stem)
        if self.stem == "cross_attn":
            self.stem_q = nn.Parameter(torch.zeros(1, int(stem_latents), d_model))
            nn.init.normal_(self.stem_q, std=0.02)
            self.stem_attn = CrossAttention(
                d_model, heads=num_heads, dim_head=d_model // num_heads, dropout=dropout
            )
            n_pre = int(stem_latents)            # trunk sees pooled latents
        else:
            n_pre = self._n_img_tok + 1          # trunk sees raw obs tokens

        # SLOT 4: action head ----  options: regression | flow (flow-matching)
        self.head = str(head)
        if self.head == "flow":
            # flow head consumes a single pooled global cond; no query tokens.
            self.flow = FlowHead(self.chunk_k, action_dim, cond_dim=d_model, n_steps=flow_steps)
            pos_len = n_pre
        elif self.readout == "action_token":
            self.action_queries = nn.Parameter(torch.zeros(1, self.chunk_k, d_model))
            nn.init.normal_(self.action_queries, std=0.02)
            self.chunk_head = nn.Linear(d_model, action_dim)   # per query token
            pos_len = n_pre + self.chunk_k
        else:
            self.chunk_head = nn.Linear(d_model, self.chunk_k * action_dim)
            pos_len = n_pre
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Backbone: "flat" (single non-causal transformer over the obs tokens,
        # current/default behaviour — MUST stay byte-identical) or
        # "hnet_chunked" (the same transformer with H-Net's dynamic chunking
        # layer spliced into the middle:
        #   toks -> [enc] -> route -> chunk(merge) -> [inner] -> dechunk(expand)
        #        -> [dec] -> chunk head).
        self.backbone_type = str(backbone)
        self.chunk_compress_ratio = float(chunk_compress_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self._last_ratio_loss = None

        def _mk_blocks(layout):
            return build_isotropic(
                {
                    "arch_layout": layout,
                    "d_model": d_model,
                    "d_intermediate": d_intermediate,
                    "num_heads": num_heads,
                    "cond": False,
                    "dropout": dropout,
                    "resid_dropout": resid_dropout,
                },
                d_cond=0,
                causal=False,
            )

        if self.backbone_type == "hnet_chunked":
            from egomimic.models.hnet.routing import (
                ChunkLayer,
                DeChunkLayer,
                RoutingModule,
            )

            enc_l, inner_l, dec_l = _split_arch_layout(arch_layout)
            self.enc_blocks = _mk_blocks(enc_l)
            self.inner_blocks = _mk_blocks(inner_l)
            self.dec_blocks = _mk_blocks(dec_l)
            self.router = RoutingModule(d_model)
            self.chunker = ChunkLayer()
            self.dechunker = DeChunkLayer(d_model)
        else:
            # Non-causal transformer over the obs tokens (within-frame attention).
            self.backbone = build_isotropic(
                {
                    "arch_layout": arch_layout,
                    "d_model": d_model,
                    "d_intermediate": d_intermediate,
                    "num_heads": num_heads,
                    "cond": False,
                    "dropout": dropout,
                    "resid_dropout": resid_dropout,
                },
                d_cond=0,
                causal=False,
            )

    def _run_backbone(self, toks: torch.Tensor) -> torch.Tensor:
        """Run the configured backbone on obs tokens ``(N, L, d_model)`` and
        return ``(N, L, d_model)`` (shape preserved either way).

        flat:         ``self.backbone(toks)`` (byte-identical to old behaviour).
        hnet_chunked: enc -> route -> chunk(merge) -> inner -> dechunk(expand)
                      -> dec, with an H-Net ratio loss stashed on
                      ``self._last_ratio_loss``. Obs tokens are never padded, so
                      the routing mask is all-ones ``(N, L)``.
        """
        if self.backbone_type != "hnet_chunked":
            return self.backbone(toks)

        N, L, _ = toks.shape
        mask = torch.ones(N, L, dtype=torch.bool, device=toks.device)

        # 1. Encoder transformer blocks (full resolution).
        h = self.enc_blocks(toks, mask=mask)

        # 2. Routing: decide chunk boundaries from consecutive-token cos-sim.
        bpred = self.router(h, mask=mask)

        # 3. Merge: keep only the boundary tokens -> (N, M, d), padded with next_mask.
        merged, _, _, next_mask = self.chunker(
            h, bpred.boundary_mask, mask=mask
        )

        # 4. Inner transformer blocks operate on the compressed sequence.
        merged = self.inner_blocks(merged, mask=next_mask)

        # 5. Expand back to full length via the DeChunkLayer EMA. It casts to
        #    bf16 internally, so feed bf16 and cast the result back to the
        #    policy dtype.
        in_dtype = h.dtype
        full = self.dechunker(
            merged.to(self.dechunker.dtype),
            bpred.boundary_mask,
            bpred.boundary_prob.to(self.dechunker.dtype),
            mask=mask,
        ).to(in_dtype)

        # 6. Decoder transformer blocks (full resolution).
        out = self.dec_blocks(full, mask=mask)

        # H-Net ratio loss so the chunker actually compresses toward 1/ratio.
        # Exclude the forced first-token boundary (position 0 of every row),
        # which the routing module hard-sets to prob 1.0 — counting it inflates
        # F/G away from the real target (matches ratio_loss_from_aux's intent).
        from egomimic.models.hnet.hnet import ratio_loss_from_aux

        ratio_valid = mask.clone()
        ratio_valid[:, 0] = False
        self._last_ratio_loss = ratio_loss_from_aux(
            [
                {
                    "bpred": bpred,
                    "target_ratio": self.chunk_compress_ratio,
                    "weight": 1.0,
                    "valid_mask_padded": ratio_valid,
                }
            ],
            device=out.device,
        )
        return out

    def _obs_tokens(self, obs: dict):
        """obs -> (tokens (N, L, d) after encoder+stem, lead dims, N). Shared by
        the regression chunk head and the flow head's global cond."""
        img = obs[self.img_key]
        lead = img.shape[:-3]
        img = img.reshape(-1, *img.shape[-3:])                 # (N, C, H, W)
        N = img.shape[0]
        if self.encoder == "resnet":
            img_tok = self.img_encoder(img)                    # (N, M, d_model)
        else:
            fmap = self.img_conv(img)                          # (N, C', h, w)
            img_tok = self.img_proj(fmap.flatten(2).transpose(1, 2))   # (N, M, d)
        prop = obs[self.proprio_key]
        prop = prop.reshape(-1, prop.shape[-1])[:, self.islice[0]:self.islice[1]]
        prop_tok = self.proprio_proj(prop).unsqueeze(1)        # (N, 1, d)
        toks = torch.cat([img_tok, prop_tok], dim=1)           # (N, M+1, d)
        if self.stem == "cross_attn":
            toks = self.stem_attn(self.stem_q.expand(N, -1, -1), toks)  # (N, L, d)
        return toks, lead, N

    def _global_cond(self, obs: dict):
        """Pooled global conditioning for the flow head -> (feat (N, d), lead)."""
        toks, lead, N = self._obs_tokens(obs)
        toks = toks + self.pos_emb[:, : toks.shape[1]].to(toks.dtype)
        out = self._run_backbone(toks)
        return out.mean(dim=1), lead

    def _predict_chunk(self, obs: dict) -> torch.Tensor:
        """obs -> regression chunk ``(..., chunk_k, action_dim)``."""
        toks, lead, N = self._obs_tokens(obs)
        if self.readout == "action_token":
            q = self.action_queries.expand(N, -1, -1)          # (N, K, d)
            seq = torch.cat([toks, q], dim=1)                  # (N, M+1+K, d)
            seq = seq + self.pos_emb[:, : seq.shape[1]].to(seq.dtype)
            out = self._run_backbone(seq)
            act = out[:, -self.chunk_k:]                       # (N, K, d) query reads
            pred = self.chunk_head(act)                        # (N, K, A)
        else:
            toks = toks + self.pos_emb[:, : toks.shape[1]].to(toks.dtype)
            out = self._run_backbone(toks)                     # (N, M+1, d)
            feat = out.mean(dim=1)                             # (N, d) mean-pool
            pred = self.chunk_head(feat).reshape(N, self.chunk_k, self.action_dim)
        return pred.reshape(*lead, self.chunk_k, self.action_dim)

    def forward(self, actions, obs):
        return self._predict_chunk(obs), []

    def forward_packed(self, actions, obs, cu_seqlens, max_seqlen):
        # Obs frames are independent (Markovian), so packing is irrelevant to
        # the encode; just run the per-frame forward over the (T_total, ...) obs.
        return self.forward(actions, obs)

    @torch.no_grad()
    def generate(self, obs, batch_size, device, T: Optional[int] = None):
        """One-shot chunk from the current obs, ``(batch_size, T, action_dim)``.
        Matches the signature the ``chunk_te`` rollout calls."""
        if self.head == "flow":
            cond, _ = self._global_cond(obs)
            pred = self.flow.sample(cond).reshape(batch_size, self.chunk_k, self.action_dim)
        else:
            pred = self._predict_chunk(obs).reshape(batch_size, self.chunk_k, self.action_dim)
        if T is not None:
            pred = pred[:, :T]
        return pred


class HNetChunkToken(HNetFused):
    """Algo wrapper for :class:`ChunkTokenPolicy`.

    Reuses ``HNetFused``'s ``process_batch_for_training`` / ``_build_obs`` /
    ``compute_losses`` / ``log_info`` and overrides only construction, the
    (chunk) training loss, and the (first-action) teacher-forced eval. Rollout
    is the existing ``chunk_te`` path in ``HNetSimEval`` (set
    ``evaluator.rollout_mode=chunk_te``).
    """

    def __init__(
        self,
        action_dim: int,
        chunk_k: int,
        d_model: int,
        norm_stats,
        domains: list = None,
        ac_keys: dict = None,
        device=None,
        proprio_dim: int = 2,
        input_slice: Sequence[int] = (0, 2),
        img_key: str = "front_img_1",
        proprio_key: str = "state_agent_obj",
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        stride: int = 2,
        image_size: int = 96,
        norm_groups: int = 8,
        arch_layout: str = "T4",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
        action_horizon: int = 1024,
        readout: str = "mean_pool",
        stem: str = "none",
        stem_latents: int = 16,
        encoder: str = "conv",
        resnet_model: str = "resnet18",
        pretrained: bool = True,
        head: str = "regression",
        flow_steps: int = 10,
        backbone: str = "flat",
        chunk_compress_ratio: float = 4.0,
        ratio_loss_weight: float = 0.03,
        **kwargs,
    ):
        Algo.__init__(self)
        self.norm_stats = norm_stats
        self.domains = list(domains or [])
        self.ac_keys = dict(ac_keys or {})
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.chunk_k = int(chunk_k)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Training-recipe knobs that HNetFused/HNet read; harmless defaults
        # here (this baseline has no chunker hierarchy / parameter groups).
        self.use_parameter_groups = False
        self.lr_multipliers = None
        self.weight_decay = 0.0
        self._hnet_core = None
        # Backbone selection + chunker aux-loss knobs (only active when
        # backbone == "hnet_chunked"; "flat" keeps the baseline byte-identical).
        self.backbone_type = str(backbone)
        self.ratio_loss_weight = float(ratio_loss_weight)

        policy = ChunkTokenPolicy(
            action_dim=action_dim,
            chunk_k=chunk_k,
            d_model=d_model,
            proprio_dim=proprio_dim,
            input_slice=input_slice,
            img_key=img_key,
            proprio_key=proprio_key,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            image_size=image_size,
            norm_groups=norm_groups,
            arch_layout=arch_layout,
            num_heads=num_heads,
            d_intermediate=d_intermediate,
            dropout=dropout,
            resid_dropout=resid_dropout,
            action_horizon=action_horizon,
            readout=readout,
            stem=stem,
            stem_latents=stem_latents,
            encoder=encoder,
            resnet_model=resnet_model,
            pretrained=pretrained,
            head=head,
            flow_steps=flow_steps,
            backbone=backbone,
            chunk_compress_ratio=chunk_compress_ratio,
            ratio_loss_weight=ratio_loss_weight,
        )
        self.nets = nn.ModuleDict({"policy": policy})
        self.nets = self.nets.float().to(self.device)
        resolve_embodiment_keys(self, norm_stats)

    def forward_training(self, batch):
        predictions = OrderedDict()
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            actions = _batch[ac_key]
            obs = self._build_obs(_batch, emb_id)
            is_packed = _batch.get("_packed", False)
            if is_packed:
                cu = _batch["cu_seqlens"]
                target, mask = _build_chunk_targets_packed(actions, cu, policy.chunk_k)
            else:
                target, mask = _build_chunk_targets_padded(
                    actions, _batch.get("seq_lens"), policy.chunk_k
                )
            if getattr(policy, "head", "regression") == "flow":
                # Flow-matching velocity loss on the (masked) chunk targets.
                cond, _ = policy._global_cond(obs)                    # (Nflat, d)
                Nflat = cond.shape[0]
                tgt = target.reshape(Nflat, policy.chunk_k, policy.action_dim)
                msk = mask.reshape(Nflat, policy.chunk_k)
                pred_v, target_v = policy.flow.loss(cond, tgt)
                diff = (pred_v - target_v) ** 2 * msk[..., None]
                aloss = diff.sum() / (msk.sum() * policy.action_dim + 1e-8)
                pred = None
            else:
                if is_packed:
                    pred, _ = policy.forward_packed(
                        actions, obs, cu, int(_batch["max_seq_len"])
                    )
                else:
                    pred, _ = policy(actions, obs)
                diff = (pred - target) ** 2 * mask[..., None]
                aloss = diff.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
            # H-Net chunker ratio loss: the chunked backbone stashes its ratio
            # loss on policy._last_ratio_loss during the forward above. Add it
            # (weighted) to the action loss so the chunker learns to compress.
            if (
                getattr(policy, "backbone_type", "flat") == "hnet_chunked"
                and getattr(policy, "_last_ratio_loss", None) is not None
            ):
                rloss = policy._last_ratio_loss
                aloss = aloss + self.ratio_loss_weight * rloss
            else:
                rloss = torch.tensor(0.0, device=aloss.device)
            predictions[f"{emb_id}_pred"] = pred
            predictions[f"{emb_id}_action_loss"] = aloss
            predictions[f"{emb_id}_ratio_loss"] = rloss.detach()
        return predictions

    @torch.no_grad()
    def forward_eval(self, batch):
        """Teacher-forced eval metric: use each chunk's first action as the
        per-frame prediction, un-normalized + padded like the H-Net path."""
        unnorm = {}
        policy = self.nets["policy"]
        for emb_id, _batch in batch.items():
            ac_key = self.resolved_ac_keys[emb_id]
            obs = self._build_obs(_batch, emb_id)
            if _batch.get("_packed", False):
                cu = _batch["cu_seqlens"]
                max_s = int(_batch["max_seq_len"])
                seq_lens = _batch["seq_lens"].clone()
                if policy.head == "flow":
                    cond, _ = policy._global_cond(obs)
                    pred = policy.flow.sample(cond)                    # (T_total, K, A)
                else:
                    pred, _ = policy.forward_packed(_batch[ac_key], obs, cu, max_s)
                first = pred[:, 0, :]                                  # (T_total, A)
                B = int(seq_lens.shape[0])
                T_max = int(seq_lens.max().item())
                padded = torch.zeros(
                    B, T_max, policy.action_dim, device=first.device, dtype=first.dtype
                )
                for b in range(B):
                    s = int(cu[b].item())
                    e = int(cu[b + 1].item())
                    padded[b, : e - s] = first[s:e]
                un = self.norm_stats.unnormalize(OrderedDict({ac_key: padded}), emb_id)
                for k, v in un.items():
                    unnorm[f"emb{emb_id}_{k}"] = v
                unnorm[f"emb{emb_id}_seq_lens"] = seq_lens
            else:
                pred, _ = policy(_batch[ac_key], obs)                  # (B,T,K,A)
                first = pred[:, :, 0, :]
                un = self.norm_stats.unnormalize(OrderedDict({ac_key: first}), emb_id)
                for k, v in un.items():
                    unnorm[f"emb{emb_id}_{k}"] = v
        return unnorm
