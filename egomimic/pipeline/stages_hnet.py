"""Batch-native dual-stream H-Net stages (BATCHFLOW port of the v2 pyramid).

Every class here follows the one convention: ``forward(batch: dict) -> dict``.
The tensor-level modules (MultiStreamTrunk, DualStreamRouter, ChunkLayer,
DeChunkLayer, residual mixers, per_emb, Isotropic) are reused UNCHANGED from
the legacy tree — they already take explicit args and never touched ctx.

What died in the port (vs the ctx lineage):
  * HNetContext / StreamBundle — streams travel as batch keys "A"/"S".
  * The chunker's ctx.cu_seqlens save/swap/restore try/finally — replaced by
    ``sub_batch`` shallow copies per recursion level (nothing to restore).
  * ctx.register_aux — chunkers append level records to batch["aux/chunker"]
    (a shared list that survives sub_batch); the trunk flattens them into
    "chunk/L{i}/*" + "log/L{i}/*" keys after the forward.
  * forward hooks for probes — the apex writes "apex/tokens" directly.

Level nesting (trunk -> chunker -> trunk -> chunker(seam) -> apex) is the
algorithm's intrinsic recursion and stays INSIDE the single DualstreamTrunk
pipeline stage; the top-level pipeline stays flat.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.isotropic_builder import build_isotropic
from egomimic.models.hnet.multi_stream_trunk import MultiStreamTrunk
from egomimic.models.hnet.per_emb import per_emb, pick
from egomimic.models.hnet.residual_mixer import build_residual_mixer
from egomimic.models.hnet.routing import ChunkLayer, DeChunkLayer
from egomimic.models.hnet.stages import _init_isotropic_linears, _ste
from egomimic.models.hnet.stream_bundle import build_same_episode_causal_mask
from egomimic.models.hnet.dual_stream_chunker import DualStreamRouter
from egomimic.pipeline.core import Stage, sub_batch
from egomimic.pipeline import packed


def _within_episode_time_pos(cu: torch.Tensor, T: int, device) -> torch.Tensor:
    cu = cu.to(device=device, dtype=torch.long)
    seg = packed.episode_ids(cu)
    return torch.arange(T, device=device, dtype=torch.long) - cu[:-1][seg]


# --------------------------------------------------------------------------- #
# Levels — nested inside DualstreamTrunk. Each is a Stage (same convention),
# with `.inner` wired by DualstreamTrunk.
# --------------------------------------------------------------------------- #
class DualTrunkLevel(Stage):
    """Dual-stream compute level: MultiStreamTrunk encode -> inner ->
    optional decode-side MultiStreamTrunk (the prenet_decoder split)."""

    reads = ["A", "S", "cu_seqlens", "time_pos", "embodiment"]
    writes = ["A", "S"]

    def __init__(self, streams_cfg, adjacency=None, n_layers=6, rotary_emb_dim=0,
                 mask_mode="asym", causal=True, dropout=0.0,
                 embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False, decoder_layout=None,
                 adaln_cond: Optional[str] = None, adaln_dim: Optional[int] = None
                 attn_dropout: float = 0.0, ffn_dropout: float = 0.0,):
        super().__init__()
        streams_cfg = [dict(s) for s in streams_cfg]
        N = len(streams_cfg)
        if adjacency is None:
            adjacency = ([[j for j in range(N) if j != i] for i in range(N)]
                         if mask_mode == "sym"
                         else [[] if i == 0 else [0] for i in range(N)])
        self.causal = bool(causal)
        self.adaln_cond = str(adaln_cond) if adaln_cond else None
        ad = int(adaln_dim) if (adaln_cond and adaln_dim) else None
        self.trunk = MultiStreamTrunk(streams_cfg, adjacency, n_layers, rotary_emb_dim,
                                      dropout=attn_dropout, ffn_dropout=ffn_dropout,
                                      dropout, embodiments=embodiments,
                                      allow_agnostic_cross=allow_agnostic_cross,
                                      adaln_dim=ad)
        dec_n = self._dec_layers(decoder_layout)
        self.decoder_trunk = (MultiStreamTrunk(streams_cfg, adjacency, dec_n,
                                               dropout=attn_dropout, ffn_dropout=ffn_dropout,
                                               rotary_emb_dim, dropout,
                                               embodiments=embodiments,
                                               allow_agnostic_cross=allow_agnostic_cross,
                                               adaln_dim=ad)
                              if dec_n is not None else None)
        object.__setattr__(self, "inner", None)  # NON-registered ref (wired by DualstreamTrunk)

    @staticmethod
    def _dec_layers(layout):
        if layout is None:
            return None
        if isinstance(layout, int):
            return int(layout)
        s = str(layout).strip()
        return int(s[1:] if s[:1] in ("T", "t") else s)

    def forward(self, batch: dict) -> dict:
        A, S = batch["A"], batch["S"]
        cu, tp, emb = batch["cu_seqlens"], batch["time_pos"], batch["embodiment"]
        mask = build_same_episode_causal_mask(A.shape[0], cu, tp, self.causal, A.device)
        rope = tp.to(device=A.device, dtype=torch.long)
        cond = None
        if self.adaln_cond and self.adaln_cond in batch:
            # per-episode z -> per-token at THIS level's resolution (works at
            # every pyramid level: z is episode-constant, cu is this level's).
            cond = packed.broadcast_per_episode(batch[self.adaln_cond], cu)
        tops = self.trunk([A, S], mask, rope, emb, cond=cond)
        batch["A"], batch["S"] = tops[0], tops[1]
        if self.inner is not None:
            batch = self.inner(batch)
        if self.decoder_trunk is not None:
            # decode-side trunk on the chunker's (already residual-mixed) output;
            # same resolution => same mask/rope. No extra residual (the single
            # STE-gated skip lives inside the chunker).
            douts = self.decoder_trunk([batch["A"], batch["S"]], mask, rope, emb, cond=cond)
            batch["A"], batch["S"] = douts[0], douts[1]
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        n = parent_residuals + self.trunk.height
        if self.decoder_trunk is not None:
            n += self.decoder_trunk.height
        scaled = rng / max(n, 1) ** 0.5
        for trunk in [self.trunk] + ([self.decoder_trunk] if self.decoder_trunk else []):
            for name, m in trunk.named_modules():
                if not isinstance(m, nn.Linear) or getattr(m.weight, "_no_reinit", False):
                    continue
                std = scaled if (name.endswith("out") or "w2" in name or "fc2" in name
                                 or "down" in name) else rng
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.inner is not None:
            n = self.inner._init_weights(rng, n)
        return n


class DualChunkerLevel(Stage):
    """Dual-stream chunker: shared boundary from a fused A+S router; chunk both
    streams; recurse into `.inner` at chunk resolution via ``sub_batch``;
    dechunk; STE-gated residual mixers. Appends its level record to
    batch["aux/chunker"]."""

    reads = ["A", "S", "cu_seqlens", "max_seq_len", "embodiment", "aux/chunker"]
    writes = ["A", "S", "aux/chunker"]

    def __init__(self, d_a: int, apex_a: int, d_s: Optional[int] = None,
                 apex_s: Optional[int] = None, dual_inner: bool = False,
                 target_compression_ratio: float = 8.0, ratio_loss_weight: float = 0.03,
                 decisiveness_loss_weight: float = 0.0,
                 spread_loss_weight: float = 0.0, spread_alpha: float = 0.5,
                 grab_prev_end: bool = True, residual_mixer: str = "mlp",
                 residual_mixer_kwargs: Optional[dict] = None,
                 embodiments: Optional[List[str]] = None, router_per_emb: bool = True,
                 router_fusion: str = "residual_mlp", d_router: Optional[int] = None,
                 router_pre_layout: Optional[str] = None, router_pre_detach: bool = False,
                 router_core: str = "cossim",
                 router_mixer_n_layers: int = 4, router_hidden_mult: float = 4.0):
        super().__init__()
        d_s = int(d_s) if d_s is not None else int(d_a)
        apex_s = int(apex_s) if apex_s is not None else d_s
        self.d_a, self.d_s, self.apex_a, self.apex_s = int(d_a), d_s, int(apex_a), apex_s
        self.dual_inner = bool(dual_inner)
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.decisiveness_loss_weight = float(decisiveness_loss_weight)
        self.spread_loss_weight = float(spread_loss_weight)
        self.spread_alpha = float(spread_alpha)
        self.grab_prev_end = bool(grab_prev_end)
        self._embs = list(embodiments) if embodiments else None
        object.__setattr__(self, "inner", None)  # NON-registered ref

        router_embs = self._embs if router_per_emb else None
        self._router = per_emb(
            lambda: DualStreamRouter(d_a, d_s, d_router=d_router,
                                     router_core=router_core,
                                     router_pre_layout=router_pre_layout,
                                     router_pre_detach=router_pre_detach,
                                     n_layers=router_mixer_n_layers,
                                     fusion=router_fusion,
                                     hidden_mult=router_hidden_mult),
            router_embs)
        self.chunk_layer = ChunkLayer()
        self.dechunk_A = DeChunkLayer(d_a)
        self.dechunk_S = DeChunkLayer(d_s)

        mk = residual_mixer_kwargs or {}

        def _combine(d_in, d_out):
            return nn.Sequential(nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                                 nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                                 nn.Linear(2 * d_in, d_out))

        def _res_proj(d):
            rp = nn.Linear(d, d)
            nn.init.zeros_(rp.weight); nn.init.zeros_(rp.bias)
            rp.weight._no_reinit = True
            return rp

        # A path (shared)
        if self.grab_prev_end:
            self.prev_end_combine_A = _combine(d_a, self.apex_a)
            self.no_prev_A = nn.Parameter(torch.randn(d_a) * 0.02)
            self.proj_in_A = None
        else:
            self.proj_in_A = nn.Linear(d_a, self.apex_a) if d_a != self.apex_a else None
        self.proj_out_A = nn.Linear(self.apex_a, d_a) if self.apex_a != d_a else None

        # S path (per-emb)
        s_target = self.apex_s if self.dual_inner else d_s
        if self.grab_prev_end:
            self.prev_end_combine_S = per_emb(lambda: _combine(d_s, s_target), self._embs)
            self.no_prev_S = per_emb(lambda: nn.Parameter(torch.randn(d_s) * 0.02), self._embs)
            self.proj_in_S = None
        else:
            self.proj_in_S = (per_emb(lambda: nn.Linear(d_s, s_target), self._embs)
                              if d_s != s_target else None)
        self.proj_out_S = (per_emb(lambda: nn.Linear(self.apex_s, d_s), self._embs)
                           if (self.dual_inner and self.apex_s != d_s) else None)

        self.residual_proj_A = _res_proj(d_a)
        self.residual_proj_S = per_emb(lambda: _res_proj(d_s), self._embs)
        self.residual_mixer_A = build_residual_mixer(residual_mixer, d_a, **mk)
        self.residual_mixer_S = per_emb(
            lambda: build_residual_mixer(residual_mixer, d_s, **mk), self._embs)

    def _combine_prev_end(self, chunked, x_stream, bmask, cu, combine_mlp, no_prev):
        pos = bmask.nonzero(as_tuple=True)[0]
        prev_end = x_stream[(pos - 1).clamp(min=0)]
        is_start = torch.isin(pos, cu[:-1])
        prev_end = torch.where(is_start.unsqueeze(-1), no_prev.to(prev_end.dtype), prev_end)
        return combine_mlp(torch.cat([chunked, prev_end], -1).float()).to(chunked.dtype)

    def forward(self, batch: dict) -> dict:
        A, S = batch["A"], batch["S"]
        cu = batch["cu_seqlens"].to(device=A.device, dtype=torch.long)
        emb = batch["embodiment"]

        bpred = pick(self._router, emb)(A, S, cu_seqlens=cu,
                                        max_seqlen=batch["max_seq_len"])
        bmask, bprob, sprob = bpred.boundary_mask, bpred.boundary_prob, bpred.selected_probs

        resA = self.residual_proj_A(A.float())
        resS = pick(self.residual_proj_S, emb)(S.float())

        A_ch, next_cu, next_max, _ = self.chunk_layer(A, bmask, cu_seqlens=cu)
        S_ch, _, _, _ = self.chunk_layer(S, bmask, cu_seqlens=cu)

        if self.grab_prev_end:
            A_in = self._combine_prev_end(A_ch, A, bmask, cu, self.prev_end_combine_A, self.no_prev_A)
            S_in = self._combine_prev_end(S_ch, S, bmask, cu,
                                          pick(self.prev_end_combine_S, emb),
                                          pick(self.no_prev_S, emb))
        else:
            A_in = self.proj_in_A(A_ch) if self.proj_in_A is not None else A_ch
            pS = pick(self.proj_in_S, emb) if self.proj_in_S is not None else None
            S_in = pS(S_ch) if pS is not None else S_ch

        # ---- recursion: fresh sub_batch at chunk resolution (nothing to restore) ----
        if self.inner is not None:
            N = A_in.shape[0]
            ctp = _within_episode_time_pos(next_cu, N, A.device)
            if self.dual_inner:
                ib = sub_batch(batch, A=A_in, S=S_in, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp)
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["A"]) if self.proj_out_A is not None else ib["A"]
                pOutS = pick(self.proj_out_S, emb) if self.proj_out_S is not None else None
                S_inner = pOutS(ib["S"]) if pOutS is not None else ib["S"]
            else:  # AGNOSTIC SEAM: only A enters (apex); S bypasses
                ib = sub_batch(batch, x=A_in, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp)
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["x"]) if self.proj_out_A is not None else ib["x"]
                S_inner = S_in
        else:
            A_inner, S_inner = A_in, S_in

        A_dech = self.dechunk_A(A_inner, bmask, bprob, cu_seqlens=next_cu)
        S_dech = self.dechunk_S(S_inner if self.dual_inner else S_in, bmask, bprob,
                                cu_seqlens=next_cu)

        batch["A"] = self.residual_mixer_A(A_dech.float() * _ste(sprob), resA).to(A.dtype)
        batch["S"] = pick(self.residual_mixer_S, emb)(
            S_dech.float() * _ste(sprob), resS).to(S.dtype)

        batch["aux/chunker"].append({
            "boundary_mask": bmask,
            "boundary_prob": bprob,
            "selected_probs": sprob,
            "cu_seqlens": cu,            # THIS level's input token space
            "chunk_cu_seqlens": next_cu,  # chunk space
            "target_ratio": self.target_compression_ratio,
            "ratio_weight": self.ratio_loss_weight,
            "dec_weight": self.decisiveness_loss_weight,
            "spread_weight": self.spread_loss_weight,
            "spread_alpha": self.spread_alpha,
            "tokens": A_ch,               # chunkviz PCA reads A tokens ONLY
        })
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        def _members(obj):
            if obj is None:
                return []
            table = getattr(obj, "table", None)
            return list(table.values()) if table is not None else [obj]

        lins: List[nn.Linear] = []
        for attr in ("prev_end_combine_A", "prev_end_combine_S"):
            for seq in _members(getattr(self, attr, None)):
                lins += [m for m in seq if isinstance(m, nn.Linear)]
        for attr in ("proj_in_A", "proj_out_A", "proj_in_S", "proj_out_S"):
            for m in _members(getattr(self, attr, None)):
                if isinstance(m, nn.Linear):
                    lins.append(m)
        for m in lins:
            if not getattr(m.weight, "_no_reinit", False):
                nn.init.normal_(m.weight, mean=0.0, std=rng)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for r in _members(self._router):
            rp = getattr(r, "router_pre", None)
            if rp is not None:
                _init_isotropic_linears(rp, rng, rp.height)
        if self.inner is not None:
            return self.inner._init_weights(rng, parent_residuals)
        return parent_residuals


class ApexLevel(Stage):
    """Agnostic-only apex: one Isotropic stack on the bare A tokens ("x").
    Publishes the apex tokens as "apex/tokens" — the DTW eval / probes read
    the key; the forward-hook machinery is dead."""

    reads = ["x", "cu_seqlens", "max_seq_len"]
    writes = ["x", "apex/tokens"]

    def __init__(self, main_network: dict, causal: bool = True):
        super().__init__()
        self.main_network = build_isotropic(dict(main_network), d_cond=0, causal=causal)

    def forward(self, batch: dict) -> dict:
        y = self.main_network(batch["x"], cu_seqlens=batch["cu_seqlens"],
                              max_seqlen=batch["max_seq_len"])
        batch["x"] = y
        batch["apex/tokens"] = y
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        n = parent_residuals + self.main_network.height
        _init_isotropic_linears(self.main_network, rng, n)
        return n


# --------------------------------------------------------------------------- #
# The pipeline stage: owns the level chain (intrinsic hierarchy stays inside).
# --------------------------------------------------------------------------- #
class DualstreamTrunk(Stage):
    """The H-Net dual-stream pyramid as ONE pipeline stage.

    ``levels`` = hydra-instantiated list [DualTrunkLevel, DualChunkerLevel, ...,
    ApexLevel]; wired here into the nested chain (level i's `.inner` = level
    i+1). After the forward, flattens batch["aux/chunker"] into flat
    "chunk/L{i}/*" keys + "log/L{i}/boundary_rate"."""

    reads = ["A", "S", "cu_seqlens", "max_seq_len", "time_pos", "embodiment"]
    writes = ["a_top", "s", "chunk/*", "apex/*", "log/*", "aux/chunker"]

    def __init__(self, levels: List[Stage], init_range: Optional[float] = 0.02):
        super().__init__()
        self.levels = nn.ModuleList(levels)   # the ONLY registration (canonical keys)
        for i in range(len(levels) - 1):
            # non-registered reference: the recursion chain must NOT duplicate
            # every level's params under levels.{i}.inner.* (the old code's
            # nested double-registration pathology — kill it for real).
            object.__setattr__(levels[i], "inner", levels[i + 1])
        object.__setattr__(self, "root", levels[0])  # non-registered ref (levels owns them)
        if init_range:
            self.root._init_weights(float(init_range), 0)

    def forward(self, batch: dict) -> dict:
        batch.setdefault("aux/chunker", [])
        batch = self.root(batch)
        batch["a_top"], batch["s"] = batch["A"], batch["S"]
        for i, rec in enumerate(batch["aux/chunker"]):
            batch[f"chunk/L{i}/boundary_mask"] = rec["boundary_mask"]
            batch[f"chunk/L{i}/boundary_prob"] = rec["boundary_prob"]
            batch[f"chunk/L{i}/cu_seqlens"] = rec["chunk_cu_seqlens"]
            batch[f"chunk/L{i}/tokens"] = rec["tokens"]
            batch[f"log/L{i}/boundary_rate"] = float(rec["boundary_mask"].float().mean())
        return batch
