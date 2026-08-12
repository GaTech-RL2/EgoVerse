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
from egomimic.pipeline.confidence_ssl import ConfidenceSSLHead
from egomimic.pipeline.core import Stage, sub_batch
from egomimic.pipeline import packed


def _within_episode_time_pos(cu: torch.Tensor, T: int, device) -> torch.Tensor:
    cu = cu.to(device=device, dtype=torch.long)
    seg = packed.episode_ids(cu)
    return torch.arange(T, device=device, dtype=torch.long) - cu[:-1][seg]


def _ste_scaled(x: torch.Tensor, gain: float) -> torch.Tensor:
    """Paper STE confidence gate with a scalable backward.

    Forward multiplier is EXACTLY 1 for any gain (gain*x + (1-gain*x) == 1);
    backward grad w.r.t. x is `gain` instead of 1 — amplifies the task-driven
    decisiveness pressure on the router without touching activations or adding
    a loss term. gain=1.0 == the canonical `_ste`.
    """
    if gain == 1.0:
        return _ste(x)
    gx = gain * x
    return gx + (1.0 - gx).detach()


# --------------------------------------------------------------------------- #
# Levels — nested inside DualstreamTrunk. Each is a Stage (same convention),
# with `.inner` wired by DualstreamTrunk.
# --------------------------------------------------------------------------- #

def _norm_window(w, fallback, n_layers):
    """None -> fallback; int -> int; list/tuple -> list of n_layers ints (each
    entry may be None = full history for that block)."""
    if w is None:
        return fallback
    # NOTE: hydra hands this over as an omegaconf ListConfig, which is NOT a
    # list/tuple -- accept any non-string sequence so the yaml path works.
    if isinstance(w, (list, tuple)) or (
            hasattr(w, "__iter__") and not isinstance(w, (str, bytes))):
        w = [None if e is None else int(e) for e in w]
        if len(w) != int(n_layers):
            raise ValueError(
                "attn_window list has %d entries but the level has %d layers."
                % (len(w), int(n_layers)))
        return w
    return int(w)


def _masks_for(window, n_tokens, cu, tp, causal, device, n_layers):
    """One mask, or a per-block list when `window` is a list. Returns
    (mask, layer_masks) where layer_masks is None in the scalar case."""
    if not isinstance(window, list):
        return build_same_episode_causal_mask(n_tokens, cu, tp, causal, device,
                                              window=window), None
    cache, per_layer = {}, []
    for w in window:
        if w not in cache:
            cache[w] = build_same_episode_causal_mask(n_tokens, cu, tp, causal,
                                                      device, window=w)
        per_layer.append(cache[w])
    return per_layer[0], per_layer

class DualTrunkLevel(Stage):
    """Dual-stream compute level: MultiStreamTrunk encode -> inner ->
    optional decode-side MultiStreamTrunk (the prenet_decoder split)."""

    reads = ["A", "S", "cu_seqlens", "time_pos", "embodiment"]
    writes = ["A", "S"]

    def __init__(self, streams_cfg, adjacency=None, n_layers=6, rotary_emb_dim=0,
                 mask_mode="asym", causal=True, dropout=0.0,
                 embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False, decoder_layout=None,
                 adaln_cond: Optional[str] = None, adaln_dim: Optional[int] = None,
                 ffn_dropout: float = 0.0, attn_window: Optional[int] = None,
                 attn_window_enc: Optional[int] = None,
                 attn_window_dec: Optional[int] = None,
                 stream_keys: Optional[List[str]] = None):
        super().__init__()
        streams_cfg = [dict(s) for s in streams_cfg]
        N = len(streams_cfg)
        # WHICH batch keys this level's streams are. Default (None) = the
        # historical ["A", "S"] two-stream level. A single key (e.g. ["A"])
        # runs the level on that stream alone, so the other stream can be
        # composed as its own level (see StreamMLP) instead of riding along.
        # NOTE: in single-stream mode the stream is index 0 = AGNOSTIC, i.e.
        # SHARED across embodiments even when `embodiments` is given -- that is
        # correct for ["A"]; a per-emb single stream wants StreamMLP instead.
        self.stream_keys = [str(k) for k in stream_keys] if stream_keys else ["A", "S"]
        if len(self.stream_keys) != N:
            raise ValueError(
                "DualTrunkLevel: stream_keys %r has %d entries but streams_cfg "
                "has %d -- one key per stream." % (self.stream_keys, len(self.stream_keys), N))
        self.reads = list(self.stream_keys) + ["cu_seqlens", "time_pos", "embodiment"]
        self.writes = list(self.stream_keys)
        if adjacency is None:
            adjacency = ([[j for j in range(N) if j != i] for i in range(N)]
                         if mask_mode == "sym"
                         else [[] if i == 0 else [0] for i in range(N)])
        self.causal = bool(causal)
        # attn_window: sliding causal window in THIS level's tokens for both
        # the encode and decode trunks (None = full within-episode history).
        # Apex stays full-history — set this only on non-apex levels.
        self.attn_window = int(attn_window) if attn_window is not None else None
        # split window: encode trunk vs decode trunk. Default (None) -> the
        # shared attn_window, so existing configs stay byte-identical.
        # Either an int (whole trunk) or a LIST of n_layers ints (per block),
        # so an ablation can window ONLY block 0 -- a 2-token receptive field --
        # instead of every block, whose windows compound with depth.
        self._n_layers = int(n_layers)
        self.attn_window_enc = _norm_window(attn_window_enc, self.attn_window, n_layers)
        self.attn_window_dec = _norm_window(attn_window_dec, self.attn_window, n_layers)
        self.adaln_cond = str(adaln_cond) if adaln_cond else None
        ad = int(adaln_dim) if (adaln_cond and adaln_dim) else None
        self.trunk = MultiStreamTrunk(streams_cfg, adjacency, n_layers, rotary_emb_dim,
                                      dropout, ffn_dropout=ffn_dropout,
                                      embodiments=embodiments,
                                      allow_agnostic_cross=allow_agnostic_cross,
                                      adaln_dim=ad)
        dec_n = self._dec_layers(decoder_layout)
        self._n_dec_layers = int(dec_n or 0)
        self.decoder_trunk = (MultiStreamTrunk(streams_cfg, adjacency, dec_n,
                                               rotary_emb_dim, dropout,
                                               ffn_dropout=ffn_dropout,
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

    def _forward_single(self, batch: dict) -> dict:
        """One stream only (stream_keys has a single entry).

        Same encode -> inner -> decode structure as the two-stream path, minus
        every per-stream branch that does not apply: no S grid (cu_seqlens_s /
        time_pos_s), no cross-stream adjacency to honour.
        """
        k = self.stream_keys[0]
        x = batch[k]
        cu, tp, emb = batch["cu_seqlens"], batch["time_pos"], batch["embodiment"]
        mask, lmasks = _masks_for(self.attn_window_enc, x.shape[0], cu, tp,
                                  self.causal, x.device, self._n_layers)
        rope = tp.to(device=x.device, dtype=torch.long)
        cond = None
        if self.adaln_cond and self.adaln_cond in batch:
            cond = packed.broadcast_per_episode(batch[self.adaln_cond], cu)
        out = self.trunk([x], mask, rope, emb, cond=cond, layer_masks=lmasks)[0]
        batch[k] = out
        # chunkviz probe: (level, encoder tokens, None) -- the S slot is None
        # because this level HAS no second stream. A consumer that assumes a
        # tensor there fails loudly rather than silently mislabelling A as S.
        if "aux/trunk_enc" in batch:
            batch["aux/trunk_enc"].append(
                (getattr(self, "viz_level_idx", 0), out.detach(), None))
        if self.inner is not None:
            batch = self.inner(batch)
        if self.decoder_trunk is not None:
            if self.attn_window_dec == self.attn_window_enc:
                dmask, dlmasks = mask, lmasks
            else:
                dmask, dlmasks = _masks_for(self.attn_window_dec, x.shape[0], cu, tp,
                                            self.causal, x.device, self._n_dec_layers)
            dout = self.decoder_trunk([batch[k]], dmask, rope, emb, cond=cond,
                                      layer_masks=dlmasks)[0]
            batch[k] = dout
            if "aux/trunk_dec" in batch:
                batch["aux/trunk_dec"].append(
                    (getattr(self, "viz_level_idx", 0), dout.detach(), None))
        return batch

    def forward(self, batch: dict) -> dict:
        if len(self.stream_keys) == 1:
            return self._forward_single(batch)
        A, S = batch["A"], batch["S"]
        cu, tp, emb = batch["cu_seqlens"], batch["time_pos"], batch["embodiment"]
        mask, lmasks = _masks_for(self.attn_window_enc, A.shape[0], cu, tp,
                                  self.causal, A.device, self._n_layers)
        rope = tp.to(device=A.device, dtype=torch.long)
        # PER-STREAM BOUNDARIES: when the chunker below gave S its own grid, the
        # two streams have different lengths, so each needs its OWN mask + rope.
        # Absent (legacy) => one shared mask/rope, byte-identical to before.
        cu_s, tp_s = batch.get("cu_seqlens_s"), batch.get("time_pos_s")
        if cu_s is not None and tp_s is not None:
            if isinstance(self.attn_window_enc, list):
                raise NotImplementedError(
                    "per-block attn_window is not supported together with "
                    "per-stream chunk boundaries (separate_boundaries).")
            mask_s = build_same_episode_causal_mask(S.shape[0], cu_s, tp_s, self.causal,
                                                   S.device, window=self.attn_window_enc)
            rope_s = tp_s.to(device=S.device, dtype=torch.long)
            mask, rope = [mask, mask_s], [rope, rope_s]
        cond = None
        if self.adaln_cond and self.adaln_cond in batch:
            # per-episode z -> per-token at THIS level's resolution (works at
            # every pyramid level: z is episode-constant, cu is this level's).
            cond = packed.broadcast_per_episode(batch[self.adaln_cond], cu)
        tops = self.trunk([A, S], mask, rope, emb, cond=cond, layer_masks=lmasks)
        a_out, s_out = tops[0], tops[1]
        batch["A"], batch["S"] = a_out, s_out
        # chunkviz probe: the ENCODER-side tokens of this trunk level, i.e. the
        # representation handed to the chunker/router. Appended in pyramid order
        # so index 0 is the LOWEST level. Inert unless a caller seeds the list
        # (only collect_chunkviz does), so training/eval pay nothing.
        if "aux/trunk_enc" in batch:
            batch["aux/trunk_enc"].append(
                (getattr(self, "viz_level_idx", 0), a_out.detach(), s_out.detach()))
        if self.inner is not None:
            batch = self.inner(batch)
        if self.decoder_trunk is not None:
            # decode-side trunk on the chunker's (already residual-mixed) output;
            # same resolution => same mask/rope. No extra residual (the single
            # STE-gated skip lives inside the chunker).
            if self.attn_window_dec == self.attn_window_enc:
                dmask = mask
            elif isinstance(mask, list):
                dmask = [build_same_episode_causal_mask(A.shape[0], cu, tp, self.causal,
                                                        A.device, window=self.attn_window_dec),
                         build_same_episode_causal_mask(S.shape[0], cu_s, tp_s, self.causal,
                                                        S.device, window=self.attn_window_dec)]
            else:
                dmask = build_same_episode_causal_mask(A.shape[0], cu, tp, self.causal,
                                                       A.device, window=self.attn_window_dec)
            douts = self.decoder_trunk([batch["A"], batch["S"]], dmask, rope, emb, cond=cond)
            batch["A"], batch["S"] = douts[0], douts[1]
            if "aux/trunk_dec" in batch:
                batch["aux/trunk_dec"].append(
                    (getattr(self, "viz_level_idx", 0),
                     douts[0].detach(), douts[1].detach()))
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
    res_scale: float = 1.0  # class default: unpickled pre-knob instances stay inert
    residual_rank: int = 0  # class default: 0 = full-rank residual_proj (byte-identical)
    res_anneal_epochs: int = 0  # class default: 0 = no anneal (byte-identical)
    res_anneal_to: float = 0.05
    res_anneal_fwd_per_epoch: int = 200
    res_dropout: bool = True
    res_grad_scale: float = 1.0
    # PER-STREAM BOUNDARIES: class default keeps unpickled pre-knob instances inert.
    separate_boundaries: bool = False
    ratio_loss_weight_s: float = 0.0
    target_compression_ratio_s: float = 0.0

    def __init__(self, d_a: int, apex_a: int, d_s: Optional[int] = None,
                 apex_s: Optional[int] = None, dual_inner: bool = False,
                 route_on: Optional[str] = None,
                 flat_s: bool = False,
                 target_compression_ratio: float = 8.0, ratio_loss_weight: float = 0.03,
                 decisiveness_loss_weight: float = 0.0,
                 spread_loss_weight: float = 0.0, spread_alpha: float = 0.5,
                 grab_prev_end: bool = True, residual_mixer: str = "mlp",
                 residual_mixer_kwargs: Optional[dict] = None,
                 embodiments: Optional[List[str]] = None, router_per_emb: bool = True,
                 router_fusion: str = "residual_mlp", d_router: Optional[int] = None,
                 router_pre_layout: Optional[str] = None, router_pre_detach: bool = False,
                 router_core: str = "cossim",
                 router_mixer_n_layers: int = 4, router_hidden_mult: float = 4.0,
                 ste_gain: float = 1.0,
                 hard_band_weight: float = 0.0, hard_band_lo: float = 0.25,
                 hard_band_hi: float = 0.45, hard_band_window: int = 12,
                 recon_loss_weight: float = 0.0, recon_hidden_mult: float = 2.0,
                 confidence_ssl_enable: bool = False,
                 confidence_ssl_pred_weight: float = 0.0,
                 confidence_ssl_id_weight: float = 0.0,
                 confidence_ssl_vic_weight: float = 0.0,
                 confidence_ssl_proj_dim: int = 256,
                 confidence_ssl_hidden_dim: Optional[int] = None,
                 confidence_ssl_ema_tau: float = 0.99,
                 confidence_ssl_temperature: float = 0.1,
                 confidence_ssl_vic_target_std: float = 1.0,
                 confidence_ssl_router_only: bool = False,
                 router_temp: float = 1.0, router_sample: bool = False,
                 cons_loss_weight: float = 0.0, cons_noise_std: float = 0.1,
                 topk_loss_weight: float = 0.0, topk_margin: float = 0.1,
                 softrate_weight: float = 0.0, softrate_tau: float = 0.15,
                 softrate_vote: str = "sigmoid",
                 softrate_center: float = 0.5,
                 softrate2_weight: float = 0.0, softrate2_center: float = 0.3,
                 indecision_weight: float = 0.0, indecision_deadzone: float = 0.21,
                 res_scale: float = 1.0, residual_rank: int = 0,
                 res_anneal_epochs: int = 0, res_anneal_to: float = 0.05,
                 res_anneal_fwd_per_epoch: int = 200,
                 res_dropout: bool = True,
                 res_grad_scale: float = 1.0,
                 separate_boundaries: bool = False,
                 ratio_loss_weight_s: float = 0.0,
                 target_compression_ratio_s: float = 0.0,
                 inner_s=None):
        super().__init__()
        d_s = int(d_s) if d_s is not None else int(d_a)
        apex_s = int(apex_s) if apex_s is not None else d_s
        self.d_a, self.d_s, self.apex_a, self.apex_s = int(d_a), d_s, int(apex_a), apex_s
        self.dual_inner = bool(dual_inner)
        # flat_s: S skips this chunker entirely (no router/chunk/inner/
        # dechunk) and stays on the frame-rate grid. A is untouched.
        self.flat_s = bool(flat_s)
        # flat_s and dual_inner are ORTHOGONAL: flat_s says S is not chunked
        # here, dual_inner says the inner chain reads the A/S stream keys
        # rather than the apex's bare x. Combined, A is chunked into the
        # inner as "A" while S rides along unchunked as "S".
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.decisiveness_loss_weight = float(decisiveness_loss_weight)
        self.spread_loss_weight = float(spread_loss_weight)
        self.spread_alpha = float(spread_alpha)
        self.ste_gain = float(ste_gain)
        # hard-band reg: constrains REALIZED (hard) boundary decisions — rate
        # band + min-one-boundary-per-window — instead of prob densities.
        # Replaces the ratio loss (set ratio_loss_weight: 0.0 alongside).
        self.hard_band_weight = float(hard_band_weight)
        self.hard_band_lo = float(hard_band_lo)
        self.hard_band_hi = float(hard_band_hi)
        self.hard_band_window = int(hard_band_window)
        self.grab_prev_end = bool(grab_prev_end)
        self._embs = list(embodiments) if embodiments else None
        object.__setattr__(self, "inner", None)  # NON-registered ref

        # Deep supervision at this level (fence-combat, 2026-07-18): dechunk
        # the RAW selected chunk tokens back to this level's input grid via the
        # p-gated EMA (dense grad on boundary_prob) and reconstruct the level
        # INPUT with a small MLP. One summary token must then explain every
        # token in its span — boundary placement becomes consequential
        # (rate-distortion pressure the action loss doesn't supply at the
        # seam). Weight 0.0 (default) = no module, byte-identical.
        self.recon_loss_weight = float(recon_loss_weight)
        if self.recon_loss_weight > 0:
            _rh = int(int(d_a) * float(recon_hidden_mult))
            self.recon_head = nn.Sequential(
                nn.Linear(int(d_a), _rh), nn.GELU(), nn.Linear(_rh, int(d_a)))
        else:
            self.recon_head = None
        # Optional confidence-side predictive supervision. With the master
        # switch off, no module or parameters are constructed and forward does
        # not add an auxiliary record or loss.
        self.confidence_ssl_weights = {
            "pred": float(confidence_ssl_pred_weight),
            "id": float(confidence_ssl_id_weight),
            "vic": float(confidence_ssl_vic_weight),
        }
        if any(weight < 0 for weight in self.confidence_ssl_weights.values()):
            raise ValueError("confidence SSL weights must be non-negative")
        self.confidence_ssl_router_only = bool(confidence_ssl_router_only)
        self.confidence_ssl = (
            ConfidenceSSLHead(
                d_model=int(d_a),
                proj_dim=int(confidence_ssl_proj_dim),
                hidden_dim=confidence_ssl_hidden_dim,
                ema_tau=float(confidence_ssl_ema_tau),
                temperature=float(confidence_ssl_temperature),
                vic_target_std=float(confidence_ssl_vic_target_std),
                enable_pred=self.confidence_ssl_weights["pred"] > 0,
                enable_id=self.confidence_ssl_weights["id"] > 0,
                enable_vic=self.confidence_ssl_weights["vic"] > 0,
            )
            if confidence_ssl_enable
            else None
        )
        # Consistency-under-noise (fence-combat, 2026-07-18): second router
        # pass on input-noised A/S; loss = mean (p1-p2)^2. Near-0.5 probs flip
        # under noise -> penalized; margined probs don't -> free. Manufactures
        # margin exactly where decisions are unstable. ⚠️ minimized by a
        # data-independent router (constant p / clamp-saturated all-low) — keep
        # the ratio loss ON alongside (its F<1/N branch opposes all-merge).
        self.cons_loss_weight = float(cons_loss_weight)
        self.cons_noise_std = float(cons_noise_std)
        # Top-K hinge (targeted ratio replacement) params ride the aux record;
        # the loss itself is computed in RatioLoss where valid-masking lives.
        self.topk_loss_weight = float(topk_loss_weight)
        self.topk_margin = float(topk_margin)
        # Soft-rate + indecision reg (user-designed, 2026-07-18) — a DYNAMIC
        # ratio-loss replacement: rate via a differentiable soft count
        # (correction concentrated on undecided tokens via sigma', confident
        # tokens exempt) + a dead-zoned per-token indecision tax pushing
        # fence-sitters toward their nearest extreme. Params ride the aux
        # record; computed in RatioLoss. Defaults 0.0 = byte-identical.
        self.softrate_weight = float(softrate_weight)
        self.softrate_tau = float(softrate_tau)
        self.softrate_center = float(softrate_center)
        # Bypass narrowing (earn arms, 2026-07-19): scale this level's residual
        # lanes (A and S) by res_scale before the mixer. <1 makes the bypass
        # insufficient alone -> the chunk path must carry the output -> seam
        # boundary quality finally matters to the loss. Default 1.0 = byte-
        # identical (safe for every existing run on requeue).
        self.res_scale = float(res_scale)
        self.residual_rank = int(residual_rank)
        self.res_anneal_epochs = int(res_anneal_epochs)
        self.res_anneal_to = float(res_anneal_to)
        self.res_anneal_fwd_per_epoch = int(res_anneal_fwd_per_epoch)
        self.res_dropout = bool(res_dropout)
        self.res_grad_scale = float(res_grad_scale)
        self.register_buffer("_res_fwd", torch.zeros((), dtype=torch.long), persistent=False)
        self.softrate2_weight = float(softrate2_weight)
        self.softrate_vote = str(softrate_vote)
        self.softrate2_center = float(softrate2_center)
        self.indecision_weight = float(indecision_weight)
        self.indecision_deadzone = float(indecision_deadzone)

        # --- per-stream boundaries -------------------------------------------
        self.separate_boundaries = bool(separate_boundaries)
        # S's own ratio loss; fall back to A's setting when unset (0.0).
        self.ratio_loss_weight_s = (
            float(ratio_loss_weight_s) if ratio_loss_weight_s else float(ratio_loss_weight))
        self.target_compression_ratio_s = (
            float(target_compression_ratio_s) if target_compression_ratio_s
            else float(target_compression_ratio))

        # PER-STREAM BOUNDARIES layout:
        #   AGNOSTIC boundary -> ONE router SHARED across embodiments (route_on="a",
        #     reads the shared A stream only, so sharing introduces no contamination)
        #   SPECIFIC boundary -> ONE router PER EMBODIMENT (route_on="s")
        # => 1 + n_emb boundary heads per level (NOT n_emb x 2).
        # Legacy (separate_boundaries=False): a single fused router, per-emb or not.
        router_embs = None if separate_boundaries else (self._embs if router_per_emb else None)
        # ROUTE-ON: what the boundary router reads. Default preserves the
        # historical derivation. Set explicitly to "a" on any chunker whose two
        # streams can be on DIFFERENT grids -- e.g. a flat_s chunker below
        # another chunker, where A arrives chunked and S is still at frame rate.
        # Fusion is undefined across grids, and the router's own fallback needs
        # solo_a to have been built (it only is when route_on is "a"/"s").
        _route_on = str(route_on) if route_on else ("a" if separate_boundaries else "fused")
        self._router = per_emb(
            lambda: DualStreamRouter(d_a, d_s, d_router=d_router,
                                     router_core=router_core,
                                     router_pre_layout=router_pre_layout,
                                     router_pre_detach=router_pre_detach,
                                     n_layers=router_mixer_n_layers,
                                     fusion=router_fusion,
                                     hidden_mult=router_hidden_mult,
                                     router_temp=router_temp,
                                     router_sample=router_sample,
                                     route_on=_route_on),
            router_embs)
        # SPECIFIC-stream router: one per embodiment, routes on S.
        self._router_s = None
        if separate_boundaries:
            self._router_s = per_emb(
                lambda: DualStreamRouter(d_a, d_s, d_router=d_router,
                                         router_core=router_core,
                                         router_pre_layout=router_pre_layout,
                                         router_pre_detach=router_pre_detach,
                                         n_layers=router_mixer_n_layers,
                                         fusion=router_fusion,
                                         hidden_mult=router_hidden_mult,
                                         router_temp=router_temp,
                                         router_sample=router_sample,
                                         route_on="s"),
                self._embs)
        self.chunk_layer = ChunkLayer()
        self.dechunk_A = DeChunkLayer(d_a)
        self.dechunk_S = DeChunkLayer(d_s)

        mk = residual_mixer_kwargs or {}

        def _combine(d_in, d_out):
            return nn.Sequential(nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                                 nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                                 nn.Linear(2 * d_in, d_out))

        def _res_proj(d):
            if self.residual_rank and 0 < self.residual_rank < d:
                k = int(self.residual_rank)
                m = nn.Sequential(nn.Linear(d, k), nn.Linear(k, d))
                nn.init.zeros_(m[1].weight); nn.init.zeros_(m[1].bias)  # start closed
                m[0].weight._no_reinit = True; m[1].weight._no_reinit = True
                return m
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
        # S goes DEEP when it enters the shared inner (dual_inner) OR when it has
        # its own apex (inner_s). Either way it is widened to apex_s and projected
        # back on the way out.
        _s_deep = bool(dual_inner) or (inner_s is not None)
        s_target = self.apex_s if _s_deep else d_s
        if self.grab_prev_end:
            self.prev_end_combine_S = per_emb(lambda: _combine(d_s, s_target), self._embs)
            self.no_prev_S = per_emb(lambda: nn.Parameter(torch.randn(d_s) * 0.02), self._embs)
            self.proj_in_S = None
        else:
            self.proj_in_S = (per_emb(lambda: nn.Linear(d_s, s_target), self._embs)
                              if d_s != s_target else None)
        self.proj_out_S = (per_emb(lambda: nn.Linear(self.apex_s, d_s), self._embs)
                           if (_s_deep and self.apex_s != d_s) else None)

        # --- S-APEX: S's OWN deep stack, per-embodiment ------------------------
        # Hydra instantiates ONE stage from the config; we deep-copy it per
        # embodiment so each specific stream gets its own weights (S is specific).
        # None => legacy behaviour (S bypasses the apex at the seam).
        self.inner_s = None
        if inner_s is not None:
            import copy as _copy
            self.inner_s = per_emb(lambda: _copy.deepcopy(inner_s), self._embs)

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

    def _forward_flat_s(self, batch: dict, A, S, cu, emb) -> dict:
        """A takes the full hierarchy; S is left untouched at frame rate."""
        _cu_s = batch.get("cu_seqlens_s")
        cu_s_in = (_cu_s.to(device=S.device, dtype=torch.long)
                   if _cu_s is not None else cu)

        bpred = pick(self._router, emb)(A, S, cu_seqlens=cu,
                                        max_seqlen=batch["max_seq_len"],
                                        cu_seqlens_s=cu_s_in)
        bmask, bprob, sprob = (bpred.boundary_mask, bpred.boundary_prob,
                               bpred.selected_probs)

        resA = A
        if self.res_scale is not None:
            resA = resA * self.res_scale

        A_ch, next_cu, next_max, _ = self.chunk_layer(A, bmask, cu_seqlens=cu)
        if self.grab_prev_end:
            A_in = self._combine_prev_end(A_ch, A, bmask, cu,
                                          self.prev_end_combine_A, self.no_prev_A)
        else:
            A_in = self.proj_in_A(A_ch) if self.proj_in_A is not None else A_ch

        S_out = S
        if self.inner is not None:
            ctp = _within_episode_time_pos(next_cu, A_in.shape[0], A.device)
            if self.dual_inner:
                # A enters chunked as "A"; S rides along UNCHUNKED as "S" with
                # its own grid, so an inner stream level (trunk or StreamMLP)
                # can process it at frame rate. No S projection exists on this
                # path -- flat_s means S's width is untouched here.
                ms_s = int(batch.get("max_seq_len_s", batch["max_seq_len"]))
                ib = sub_batch(batch, A=A_in, S=S, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp,
                               cu_seqlens_s=cu_s_in, max_seq_len_s=ms_s,
                               time_pos_s=_within_episode_time_pos(
                                   cu_s_in, S.shape[0], S.device))
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["A"]) if self.proj_out_A is not None else ib["A"]
                S_out = ib["S"]
            else:  # AGNOSTIC SEAM: only A enters the shared apex
                ib = sub_batch(batch, x=A_in, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp)
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["x"]) if self.proj_out_A is not None else ib["x"]
        else:
            A_inner = A_in

        A_dech = self.dechunk_A(A_inner, bmask, bprob, cu_seqlens=next_cu)
        batch["A"] = self.residual_mixer_A(
            A_dech.float() * _ste_scaled(sprob, self.ste_gain), resA).to(A.dtype)
        # S never passed through a chunker here; its grid is explicitly carried
        # so downstream levels do not mistake A's chunk grid for S's.
        batch["S"] = S_out
        batch["cu_seqlens_s"] = cu_s_in
        batch["max_seq_len_s"] = int(batch["max_seq_len"]) if _cu_s is None else int(
            batch.get("max_seq_len_s", batch["max_seq_len"]))
        # cu_seqlens/max_seq_len stay at the INPUT token space: A was dechunked
        # back to it just above (resA is the input A), exactly like the non-flat
        # path, which also leaves them untouched. Advertising the CHUNK grid here
        # made every downstream consumer index a frame-rate tensor with
        # chunk-space offsets -> CUDA index-out-of-bounds in the head.

        batch["aux/chunker"].append({
            "boundary_mask": bmask,
            "boundary_prob": bprob,
            "selected_probs": sprob,
            "cu_seqlens": cu,
            "chunk_cu_seqlens": next_cu,
            # TODO(remove-chunker-target): the compression TARGET is loss policy and now lives on
            # stages_io.RatioLoss(target_ratio=...). This record field is a
            # DEPRECATED fallback kept only so existing model yamls (which all
            # set target_compression_ratio here) keep working. Delete this line
            # and the constructor arg once every yaml sets it on RatioLoss.
            "target_ratio": self.target_compression_ratio,
            "ratio_weight": self.ratio_loss_weight,
            "flat_s": True,          # no S boundary terms exist at this level
            "separate_boundaries": False,
            "tokens": A_ch,
        })
        return batch


    def forward(self, batch: dict) -> dict:
        A, S = batch["A"], batch["S"]
        cu = batch["cu_seqlens"].to(device=A.device, dtype=torch.long)
        emb = batch["embodiment"]
        if getattr(self, "flat_s", False):
            # ---- FLAT S: chunk A only; S passes straight through ------------
            # S keeps the incoming (frame-rate) grid, so downstream trunk levels
            # build S's mask/rope from cu_seqlens_s while A uses its chunk grid.
            return self._forward_flat_s(batch, A, S, cu, emb)
        # S's INPUT grid: its own when an earlier chunker split the streams,
        # otherwise identical to A's (legacy).
        _cu_s = batch.get("cu_seqlens_s")
        cu_s_in = (_cu_s.to(device=S.device, dtype=torch.long)
                   if _cu_s is not None else cu)

        bpred = pick(self._router, emb)(A, S, cu_seqlens=cu,
                                        max_seqlen=batch["max_seq_len"],
                                        cu_seqlens_s=cu_s_in)
        if self.separate_boundaries:
            # SHARED agnostic router (above) + PER-EMBODIMENT specific router.
            bpred_s = pick(self._router_s, emb)(A, S, cu_seqlens=cu,
                                                max_seqlen=batch["max_seq_len"],
                                                cu_seqlens_s=cu_s_in)
        else:
            bpred_s = bpred                     # same object => identical grids
        bmask, bprob, sprob = bpred.boundary_mask, bpred.boundary_prob, bpred.selected_probs
        bmask_s = bpred_s.boundary_mask
        bprob_s = bpred_s.boundary_prob
        sprob_s = bpred_s.selected_probs

        # Consistency-under-noise (see __init__): agreement between this pass
        # and a second pass on input-noised streams. Forced starts are 1.0 in
        # both -> zero contribution; no self.training gate so the gradient
        # probe (eval mode) can measure the force.
        cons_loss = None
        if self.cons_loss_weight > 0:
            _nA = (A.float() + self.cons_noise_std * A.float().std()
                   * torch.randn_like(A.float())).to(A.dtype)
            _nS = (S.float() + self.cons_noise_std * S.float().std()
                   * torch.randn_like(S.float())).to(S.dtype)
            _bp2 = pick(self._router, emb)(_nA, _nS, cu_seqlens=cu,
                                           max_seqlen=batch["max_seq_len"])
            cons_loss = ((bprob[..., -1] - _bp2.boundary_prob[..., -1]) ** 2).mean()

        resA = self.residual_proj_A(A.float())
        resS = pick(self.residual_proj_S, emb)(S.float())
        if self.res_grad_scale != 1.0:
            _b = self.res_grad_scale
            resA = _b * resA + (1.0 - _b) * resA.detach()
            resS = _b * resS + (1.0 - _b) * resS.detach()
        rs = self.res_scale
        if self.res_anneal_epochs > 0:
            if self.training:
                self._res_fwd += 1
            _frac = min(1.0, (float(self._res_fwd) / max(1, self.res_anneal_fwd_per_epoch)) / self.res_anneal_epochs)
            rs = 1.0 + (self.res_anneal_to - 1.0) * _frac
        if rs != 1.0:
            # Bypass narrowing as RES-LANE DROPOUT (panel fix, 2026-07-19): a
            # multiplicative scale on a learned lane is absorbable (proj/mixer
            # regrow 1/s — empirically verified). A stochastic blackout is not:
            # train-time, each token's residual lane survives with prob s ==
            # res_scale (NO inverted rescale — absence is the point), so the
            # chunk path MUST carry the output on (1-s) of steps. Eval scales
            # by s to match the train-time expectation deterministically.
            if self.training and self.res_dropout:
                keep = (torch.rand(resA.shape[0], 1, device=resA.device)
                        < rs).float()
                resA = resA * keep
                resS = resS * keep
            else:
                resA = resA * rs
                resS = resS * rs

        A_ch, next_cu, next_max, _ = self.chunk_layer(A, bmask, cu_seqlens=cu)
        # S chunks on ITS OWN boundary when separate_boundaries; otherwise bmask_s
        # IS bmask, so this is the same call as before and next_cu_s == next_cu.
        S_ch, next_cu_s, next_max_s, _ = self.chunk_layer(S, bmask_s, cu_seqlens=cu_s_in)

        if self.grab_prev_end:
            A_in = self._combine_prev_end(A_ch, A, bmask, cu, self.prev_end_combine_A, self.no_prev_A)
            S_in = self._combine_prev_end(S_ch, S, bmask_s, cu_s_in,
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
                extra = {}
                if self.separate_boundaries:
                    # S travels on its own grid; the inner trunk builds a second
                    # mask/rope from these. Streams stay INDEPENDENT while chunked
                    # (requires adjacency=[[], []]) and rejoin after dechunk.
                    extra = dict(
                        cu_seqlens_s=next_cu_s,
                        max_seq_len_s=int(next_max_s),
                        time_pos_s=_within_episode_time_pos(
                            next_cu_s, S_in.shape[0], S.device),
                    )
                ib = sub_batch(batch, A=A_in, S=S_in, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp, **extra)
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["A"]) if self.proj_out_A is not None else ib["A"]
                pOutS = pick(self.proj_out_S, emb) if self.proj_out_S is not None else None
                S_inner = pOutS(ib["S"]) if pOutS is not None else ib["S"]
            else:  # AGNOSTIC SEAM: only A enters the shared apex
                ib = sub_batch(batch, x=A_in, cu_seqlens=next_cu,
                               max_seq_len=int(next_max), time_pos=ctp)
                ib = self.inner(ib)
                A_inner = self.proj_out_A(ib["x"]) if self.proj_out_A is not None else ib["x"]
                if self.inner_s is not None:
                    # S runs its OWN apex on ITS OWN grid -> a genuinely parallel
                    # hierarchy, instead of a compress/decompress round trip.
                    ctp_s = _within_episode_time_pos(next_cu_s, S_in.shape[0], S.device)
                    ib_s = sub_batch(batch, x=S_in, cu_seqlens=next_cu_s,
                                     max_seq_len=int(next_max_s), time_pos=ctp_s)
                    ib_s = pick(self.inner_s, emb)(ib_s)
                    _poS = pick(self.proj_out_S, emb) if self.proj_out_S is not None else None
                    S_inner = _poS(ib_s["x"]) if _poS is not None else ib_s["x"]
                else:
                    S_inner = S_in
        else:
            A_inner, S_inner = A_in, S_in

        A_dech = self.dechunk_A(A_inner, bmask, bprob, cu_seqlens=next_cu)
        # S dechunks on its OWN grid -> lands back at level-0 resolution, where the
        # two streams REJOIN in the residual mixer below. This is why per-stream
        # boundaries need no realignment machinery downstream.
        S_dech = self.dechunk_S(S_inner, bmask_s, bprob_s,
                                cu_seqlens=next_cu_s)
        confidence_weighted_A = (
            A_dech.float() * _ste_scaled(sprob, self.ste_gain)
        )

        confidence_ssl_losses = None
        confidence_ssl = getattr(self, "confidence_ssl", None)
        if confidence_ssl is not None:
            ssl_input = confidence_weighted_A
            if getattr(self, "confidence_ssl_router_only", False):
                # Recompute p from detached level inputs, then use the original
                # hard assignment for the signed selected confidence. This
                # freezes the encoder path while preserving router gradients.
                ssl_bpred = pick(self._router, emb)(
                    A.detach(), S.detach(), cu_seqlens=cu,
                    max_seqlen=batch["max_seq_len"])
                ssl_p = ssl_bpred.boundary_prob[..., -1]
                ssl_selected = torch.where(bmask, ssl_p, 1.0 - ssl_p).unsqueeze(-1)
                ssl_input = (
                    A_dech.detach().float()
                    * _ste_scaled(ssl_selected, self.ste_gain)
                )
            confidence_ssl_losses = confidence_ssl(
                confidence_frames=ssl_input,
                latent_frames=A_dech,
                raw_frames=A,
                boundary_mask=bmask,
                chunk_cu_seqlens=next_cu,
            )

        # Deep supervision (see __init__): reconstruct the level INPUT from the
        # RAW chunked tokens dechunked back to the input grid. Uses the same
        # p-gated EMA dechunk as the main path — dense dL/dp on every token —
        # but bypasses the inner/apex so nothing can compensate for bad
        # grouping: recon quality is a direct function of boundary placement.
        recon_loss = None
        if self.recon_head is not None:
            _rec = self.dechunk_A(A_ch, bmask, bprob, cu_seqlens=next_cu)
            recon_loss = nn.functional.mse_loss(
                self.recon_head(_rec.float()), A.float().detach())

        batch["A"] = self.residual_mixer_A(
            confidence_weighted_A, resA).to(A.dtype)
        batch["S"] = pick(self.residual_mixer_S, emb)(
            S_dech.float() * _ste_scaled(sprob_s, self.ste_gain), resS).to(S.dtype)

        aux_rec = {
            "boundary_mask": bmask,
            "boundary_prob": bprob,
            "selected_probs": sprob,
            "cu_seqlens": cu,            # THIS level's input token space
            "chunk_cu_seqlens": next_cu,  # chunk space
            # TODO(remove-chunker-target): the compression TARGET is loss policy and now lives on
            # stages_io.RatioLoss(target_ratio=...). This record field is a
            # DEPRECATED fallback kept only so existing model yamls (which all
            # set target_compression_ratio here) keep working. Delete this line
            # and the constructor arg once every yaml sets it on RatioLoss.
            "target_ratio": self.target_compression_ratio,
            "ratio_weight": self.ratio_loss_weight,
            # --- per-stream boundaries: S's own boundary + its own ratio target.
            # When separate_boundaries is off these mirror the A entries exactly,
            # and separate_boundaries=False tells consumers to ignore them.
            "separate_boundaries": self.separate_boundaries,
            "boundary_mask_s": bmask_s,
            "boundary_prob_s": bprob_s,
            "selected_probs_s": sprob_s,
            "chunk_cu_seqlens_s": next_cu_s,
            # S's OWN input grid (differs from cu once separate_boundaries has
            # split the streams upstream). RatioLoss must mask S entries with
            # THIS, not A's cu -- A's indices can exceed the S tensor length
            # (crash, job 3622575) or silently mask wrong positions.
            "cu_seqlens_s": cu_s_in,
            # TODO(remove-chunker-target): same deprecation as target_ratio above (S stream).
            "target_ratio_s": self.target_compression_ratio_s,
            "ratio_weight_s": self.ratio_loss_weight_s,
            "dec_weight": self.decisiveness_loss_weight,
            "spread_weight": self.spread_loss_weight,
            "spread_alpha": self.spread_alpha,
            "hb_weight": self.hard_band_weight,
            "hb_lo": self.hard_band_lo,
            "hb_hi": self.hard_band_hi,
            "hb_window": self.hard_band_window,
            "recon_loss": recon_loss,
            "recon_weight": self.recon_loss_weight,
            "cons_loss": cons_loss,
            "cons_weight": self.cons_loss_weight,
            "topk_weight": self.topk_loss_weight,
            "topk_margin": self.topk_margin,
            "softrate_weight": self.softrate_weight,
            "softrate_vote": self.softrate_vote,
            "softrate_tau": self.softrate_tau,
            "softrate_center": self.softrate_center,
            "softrate2_weight": self.softrate2_weight,
            "softrate2_center": self.softrate2_center,
            "indecision_weight": self.indecision_weight,
            "indecision_deadzone": self.indecision_deadzone,
            "tokens": A_ch,               # chunkviz PCA reads A tokens ONLY
        }
        if confidence_ssl_losses is not None:
            aux_rec["confidence_ssl_losses"] = confidence_ssl_losses
            aux_rec["confidence_ssl_weights"] = self.confidence_ssl_weights
        batch["aux/chunker"].append(aux_rec)
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
    the key; the forward-hook machinery is dead.

    ``self_only=True`` removes every temporal edge at the apex by presenting
    each packed token to the unchanged Isotropic stack as its own length-one
    sequence. ``attn_window=N`` instead gives every Transformer layer a causal
    sliding window of N apex tokens, including the current token. The default
    is the original full within-episode history path.
    """

    reads = ["x", "cu_seqlens", "max_seq_len"]
    writes = ["x", "apex/tokens"]

    def __init__(
        self,
        main_network: dict,
        causal: bool = True,
        self_only: bool = False,
        attn_window: Optional[int] = None,
    ):
        super().__init__()
        if self_only and attn_window is not None:
            raise ValueError(
                "ApexLevel self_only and attn_window are mutually exclusive"
            )
        if attn_window is not None and int(attn_window) < 1:
            raise ValueError("ApexLevel attn_window must be >= 1")
        if attn_window is not None and not causal:
            raise ValueError("ApexLevel attn_window requires causal=True")
        self.causal = bool(causal)
        self.self_only = False
        self.attn_window: Optional[int] = None
        self.main_network = build_isotropic(
            dict(main_network),
            d_cond=0,
            causal=causal,
        )
        initial_mode = (
            "self" if self_only else "windowed" if attn_window is not None else "full"
        )
        self.set_attention_mode(initial_mode, window=attn_window)

    @property
    def attention_mode(self) -> str:
        if self.self_only:
            return "self"
        if self.attn_window is not None:
            return "windowed"
        return "full"

    def set_attention_mode(
        self,
        mode: str,
        *,
        window: Optional[int] = None,
    ) -> None:
        """Switch apex context without changing parameters or state-dict keys.

        ``full`` uses the complete causal history, ``self`` presents every
        token as a length-one packed sequence, and ``windowed`` gives each
        Transformer layer the current token plus ``window - 1`` predecessors.
        The method is deliberately runtime-safe so the same checkpoint can
        switch regimes per training step and at rollout.
        """
        mode = str(mode).lower().replace("-", "_")
        if mode in {"self_only", "token_local"}:
            mode = "self"
        if mode in {"window", "local"}:
            mode = "windowed"
        if mode not in {"full", "self", "windowed"}:
            raise ValueError(
                f"unknown apex attention mode {mode!r}; "
                "expected full, self, or windowed"
            )
        if mode == "windowed":
            if window is None or int(window) < 1:
                raise ValueError("windowed apex attention requires window >= 1")
            if not self.causal:
                raise ValueError("windowed apex attention requires causal=True")
            resolved_window = int(window)
        else:
            if window is not None:
                raise ValueError(f"{mode} apex attention does not accept a window")
            resolved_window = None

        for block in self.main_network.layers:
            mixer = getattr(block, "mixer", None)
            if mixer is None or not hasattr(mixer, "attn_window"):
                if mode == "windowed":
                    raise ValueError(
                        "windowed apex attention requires a transformer-only "
                        "main_network"
                    )
                continue
            mixer.attn_window = resolved_window

        self.self_only = mode == "self"
        self.attn_window = resolved_window

    def forward(self, batch: dict) -> dict:
        x = batch["x"]
        cu_seqlens = batch["cu_seqlens"]
        max_seqlen = batch["max_seq_len"]
        if self.self_only:
            # One token per packed subsequence makes attention (or any other
            # sequence mixer) strictly token-local while leaving the apex
            # weights, residual blocks, FFNs, and output shape unchanged.
            cu_seqlens = torch.arange(
                x.shape[0] + 1,
                device=x.device,
                dtype=cu_seqlens.dtype,
            )
            max_seqlen = 1
        y = self.main_network(
            x,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        batch["x"] = y
        batch["apex/tokens"] = y
        if "aux/apex" in batch:
            batch["aux/apex"].append(y)
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        n = parent_residuals + self.main_network.height
        _init_isotropic_linears(self.main_network, rng, n)
        return n


# --------------------------------------------------------------------------- #
# The pipeline stage: owns the level chain (intrinsic hierarchy stays inside).
# --------------------------------------------------------------------------- #

class StreamMLP(Stage):
    """Per-timestep MLP on ONE stream -- a pyramid level with NO temporal mixing.

    Written for the "history through S hurts" hypothesis: the S stream should
    carry the current frame's embodiment-specific content and nothing else, so
    the policy cannot lean on S to smuggle in history. This stage is
    structurally incapable of it -- it never reads ``cu_seqlens`` or
    ``time_pos``, so no token can reach another token's information. Contrast
    with a transformer level pinned to ``attn_window=1``: that reaches the same
    function class through a degenerate one-element attention (qkv projections
    and a softmax over a single logit), i.e. identical semantics at the cost of
    the qkv/out parameters and the mask bookkeeping.

    Structure mirrors ``DualTrunkLevel`` so the two compose in one ``levels``
    chain: ``n_layers`` encoder blocks -> ``self.inner(batch)`` (the rest of the
    pyramid) -> ``decoder_layers`` decoder blocks. Each block is pre-norm
    residual ``x <- x + fc2(act(fc1(norm(x))))``.

    Per-embodiment when ``embodiments`` is given (the S stream is the SPECIFIC
    stream, so this is the normal case); a bare instantiation shares one MLP.

    Args:
        d_model: stream width. Unchanged across the level (this stage never
            projects) -- so the chunker above it must leave S's width alone
            (``flat_s: true`` / no ``apex_s``).
        n_layers: encoder-side blocks.
        decoder_layers: decoder-side blocks, applied after ``inner``. None or 0
            puts every block on the encoder side.
        d_hidden: block hidden width. Default 4*d_model, matching the
            ``d_intermediate`` a transformer level of this width would use --
            note the resulting level is still CHEAPER than that transformer,
            which also pays qkv+out (~ d_model^2 * 4 per block).
        stream: which batch key to transform ("S" by default).
    """

    def __init__(self, d_model: int, n_layers: int = 4,
                 decoder_layers: Optional[int] = None,
                 d_hidden: Optional[int] = None,
                 stream: str = "S",
                 embodiments: Optional[List[str]] = None,
                 activation: str = "gelu",
                 dropout: float = 0.0,
                 final_norm: bool = True):
        super().__init__()
        self.stream = str(stream)
        self.reads = [self.stream, "embodiment"]
        self.writes = [self.stream]
        d_model = int(d_model)
        d_hidden = int(d_hidden) if d_hidden else 4 * d_model
        n_enc = int(n_layers)
        n_dec = int(decoder_layers) if decoder_layers else 0
        if n_enc < 0 or n_dec < 0:
            raise ValueError("StreamMLP layer counts must be >= 0")
        if n_enc + n_dec == 0:
            raise ValueError("StreamMLP needs at least one block")
        acts = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
        if activation not in acts:
            raise ValueError("StreamMLP activation %r not in %s"
                             % (activation, sorted(acts)))
        act = acts[activation]

        def _stack(n):
            if n == 0:
                return None
            return per_emb(lambda: _MLPBlocks(d_model, d_hidden, n, act,
                                              dropout, final_norm),
                           embodiments)

        self.encoder = _stack(n_enc)
        self.decoder = _stack(n_dec)
        self.n_blocks = n_enc + n_dec
        object.__setattr__(self, "inner", None)  # wired by DualstreamTrunk

    def forward(self, batch: dict) -> dict:
        emb = batch["embodiment"]
        x = batch[self.stream]
        if self.encoder is not None:
            x = pick(self.encoder, emb)(x)
        batch[self.stream] = x
        if self.inner is not None:
            batch = self.inner(batch)
        if self.decoder is not None:
            batch[self.stream] = pick(self.decoder, emb)(batch[self.stream])
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        n = parent_residuals + self.n_blocks
        scaled = rng / max(n, 1) ** 0.5
        for name, m in self.named_modules():
            if not isinstance(m, nn.Linear) or getattr(m.weight, "_no_reinit", False):
                continue
            # down-projection of each block gets the depth-scaled std, matching
            # DualTrunkLevel._init_weights ("fc2" is the block's output linear).
            std = scaled if "fc2" in name else rng
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.inner is not None:
            n = self.inner._init_weights(rng, n)
        return n


class _MLPBlocks(nn.Module):
    """n pre-norm residual MLP blocks at constant width. Purely per-timestep."""

    def __init__(self, d_model, d_hidden, n, act, dropout, final_norm):
        super().__init__()
        def block():
            return nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "fc1": nn.Linear(d_model, d_hidden),
                "act": act(),
                "fc2": nn.Linear(d_hidden, d_model),
            })
        self.blocks = nn.ModuleList([block() for _ in range(n)])
        self.drop = nn.Dropout(dropout) if dropout else None
        self.final = nn.LayerNorm(d_model) if final_norm else None

    def forward(self, x):
        for b in self.blocks:
            h = b["fc2"](b["act"](b["fc1"](b["norm"](x))))
            if self.drop is not None:
                h = self.drop(h)
            x = x + h
        return self.final(x) if self.final is not None else x

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
        for i, _lv in enumerate(levels):
            object.__setattr__(_lv, "viz_level_idx", i)
        object.__setattr__(self, "root", levels[0])  # non-registered ref (levels owns them)
        if init_range:
            self.root._init_weights(float(init_range), 0)

    def forward(self, batch: dict) -> dict:
        batch.setdefault("aux/chunker", [])
        batch.setdefault("aux/apex", [])
        batch = self.root(batch)
        batch["a_top"], batch["s"] = batch["A"], batch["S"]
        if batch.get("aux/apex"):
            batch["apex/tokens"] = batch["aux/apex"][-1]
        for i, rec in enumerate(batch["aux/chunker"]):
            batch[f"chunk/L{i}/boundary_mask"] = rec["boundary_mask"]
            batch[f"chunk/L{i}/boundary_prob"] = rec["boundary_prob"]
            batch[f"chunk/L{i}/cu_seqlens"] = rec["chunk_cu_seqlens"]
            batch[f"chunk/L{i}/tokens"] = rec["tokens"]
            batch[f"log/L{i}/boundary_rate"] = float(rec["boundary_mask"].float().mean())
        return batch
