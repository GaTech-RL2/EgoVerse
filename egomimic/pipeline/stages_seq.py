"""Sequential (flat) pipeline stages -- the un-nested form of the H-Net pyramid.

``stages_hnet.py`` expresses the pyramid as a NESTED chain: ``DualstreamTrunk``
wires level *i*'s ``.inner`` to level *i+1*, and ``DualChunkerLevel`` calls
``self.inner(...)`` in the middle of its own forward (chunk -> recurse ->
dechunk). That nesting is the only reason a container stage exists at all --
``PipelineAlgo`` already runs a flat list.

Here the same computation is a FLAT list. Two changes make that possible:

1. The chunker is split into :class:`Chunk` (router + compress, publishes a
   route record) and :class:`Dechunk` (expand + residual mix, consumes it), so
   what used to be the return half of one forward is now its own stage.
2. Every stage names the batch keys it reads and writes (``A_L0``, ``S_L0``,
   ``A_L1``, ...), instead of mutating fixed ``"A"``/``"S"`` slots.

Consequences worth stating, because they remove three flags:

* ``stream_keys`` is gone -- a stage operates on exactly the keys it is given,
  so "A only" is expressed by naming one key.
* ``flat_s`` is gone -- "S votes in the router but is not chunked" is just
  ``Chunk`` reading ``s_key`` and not writing an S output.
* ``dual_inner`` is gone -- there is no inner; the next stage in the list is
  whatever you put there.

The trunk's encoder/decoder split (``decoder_layout``) also becomes explicit:
an encoder :class:`StreamTrunk` before the ``Chunk`` and a decoder one after
the matching ``Dechunk``.

Model components are REUSED UNCHANGED from ``egomimic.models.hnet``
(``MultiStreamTrunk``, ``ChunkLayer``/``DeChunkLayer``, ``DualStreamRouter``,
``build_residual_mixer``, ``build_isotropic``). This module is wiring, not new
math.

NOT carried over from ``DualChunkerLevel`` (deliberately -- these stay
available on the nested path, and none is used by the yolo arms):
``separate_boundaries``, ``confidence_ssl``, ``recon_head`` deep supervision,
``cons_loss``/``topk``/``softrate``/``indecision`` regularizers, residual
annealing and res-lane dropout, ``inner_s`` (S running its own apex).
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
from egomimic.models.hnet.stages import _init_isotropic_linears
from egomimic.models.hnet.stream_bundle import build_same_episode_causal_mask
from egomimic.models.hnet.dual_stream_chunker import DualStreamRouter
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_io import RatioLoss as _RatioLoss
from egomimic.pipeline.stages_hnet import (
    _within_episode_time_pos,
    _ste_scaled,
    _masks_for,
    _norm_window,
    _MLPBlocks,
)


def _grid_keys(key: str):
    """Per-level grid key names carried alongside a stream key.

    ``A_L1`` -> ``("cu_seqlens@A_L1", "max_seq_len@A_L1", "time_pos@A_L1")``.
    Grids belong to a RESOLUTION, not a stream, but keying them off the stream
    name keeps every stage self-describing from its config alone.
    """
    return f"cu_seqlens@{key}", f"max_seq_len@{key}", f"time_pos@{key}"


def _read_grid(batch: dict, key: str, ref: torch.Tensor):
    """Grid for ``key``, falling back to the batch-level grid (level 0)."""
    ck, mk, tk = _grid_keys(key)
    cu = batch.get(ck, batch.get("cu_seqlens"))
    mx = batch.get(mk, batch.get("max_seq_len"))
    tp = batch.get(tk, batch.get("time_pos"))
    cu = cu.to(device=ref.device, dtype=torch.long)
    return cu, int(mx), tp


def _write_grid(batch: dict, key: str, cu, mx, tp):
    ck, mk, tk = _grid_keys(key)
    batch[ck], batch[mk], batch[tk] = cu, int(mx), tp


def per_emb_module(factory_or_mod, use_per_emb: bool, embodiments):
    """per_emb=True -> one copy per embodiment (PerEmb); else a single shared one.
    Accepts either a factory or an already-instantiated module (hydra gives the
    latter for nested _target_ blocks, which cannot be duplicated per emb)."""
    if not use_per_emb or not embodiments:
        return factory_or_mod() if callable(factory_or_mod) and not isinstance(
            factory_or_mod, nn.Module) else factory_or_mod
    if isinstance(factory_or_mod, nn.Module):
        raise ValueError(
            "per_emb=True needs a factory, not an instantiated module: give each "
            "embodiment its own stage, or pass a _partial_ target.")
    return per_emb(factory_or_mod, embodiments)   # module-level helper, NOT the flag


class StreamTrunk(Stage):
    """MultiStreamTrunk over one or more named stream keys, at one resolution.

    Replaces ``DualTrunkLevel`` with explicit keys. ``in_keys``/``out_keys``
    are positionally matched to ``streams_cfg``; a single key is the A-only
    case that used to need ``stream_keys: [A]``.
    """

    def __init__(self, in_keys: List[str], streams_cfg: List[dict],
                 out_keys: Optional[List[str]] = None,
                 adjacency=None, n_layers: int = 4, rotary_emb_dim: int = 0,
                 attn_window=None, causal: bool = True,
                 mask_mode: str = "sym", embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False,
                 dropout: float = 0.0, ffn_dropout: float = 0.0,
                 adaln_cond: Optional[str] = None,
                 adaln_dim: Optional[int] = None):
        super().__init__()
        streams_cfg = [dict(s) for s in streams_cfg]
        N = len(streams_cfg)
        self.in_keys = [str(k) for k in in_keys]
        self.out_keys = [str(k) for k in (out_keys or in_keys)]
        if len(self.in_keys) != N:
            raise ValueError(
                f"StreamTrunk: {len(self.in_keys)} in_keys but "
                f"{N} streams_cfg entries -- one key per stream.")
        if len(self.out_keys) != len(self.in_keys):
            raise ValueError("StreamTrunk: out_keys must match in_keys in length.")
        # _read_grid consumes the level grid; declare it so the routing graph
        # and the lint see the real dependency.
        self.reads = list(self.in_keys) + ["embodiment", "cu_seqlens", "max_seq_len", "time_pos"]
        self.writes = list(self.out_keys)
        # Same derivation as DualTrunkLevel: "sym" = every stream sees every
        # other; "asym" = specific streams read the agnostic one, not vice versa.
        if adjacency is None:
            adjacency = ([[j for j in range(N) if j != i] for i in range(N)]
                         if mask_mode == "sym"
                         else [[] if i == 0 else [0] for i in range(N)])
        self.causal = bool(causal)
        self._n_layers = int(n_layers)
        self.attn_window = _norm_window(attn_window, None, n_layers)
        self.adaln_cond = str(adaln_cond) if adaln_cond else None
        ad = int(adaln_dim) if (adaln_cond and adaln_dim) else None
        self.trunk = MultiStreamTrunk(
            streams_cfg, adjacency, int(n_layers), int(rotary_emb_dim),
            float(dropout), ffn_dropout=float(ffn_dropout),
            embodiments=embodiments,
            allow_agnostic_cross=bool(allow_agnostic_cross), adaln_dim=ad)

    def forward(self, batch: dict) -> dict:
        xs = [batch[k] for k in self.in_keys]
        ref = xs[0]
        cu, mx, tp = _read_grid(batch, self.in_keys[0], ref)
        mask, lmasks = _masks_for(self.attn_window, ref.shape[0], cu, tp,
                                  self.causal, ref.device, self._n_layers)
        rope = tp.to(device=ref.device, dtype=torch.long)
        cond = batch.get(self.adaln_cond) if self.adaln_cond else None
        outs = self.trunk(xs, mask, rope, batch["embodiment"],
                          cond=cond, layer_masks=lmasks)
        for k, v in zip(self.out_keys, outs):
            batch[k] = v
            _write_grid(batch, k, cu, mx, tp)
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        return _init_isotropic_linears(self.trunk, rng, parent_residuals)


class Framewise(Stage):
    """Per-frame MLP on one or more streams -- NO temporal mixing.

    Generalises ``StreamMLP`` to several streams at once. Like it, this stage
    never reads ``cu_seqlens`` or ``time_pos``, so it is *structurally*
    incapable of moving information between timesteps -- that is the point,
    not an optimisation.

    Streams listed in ``per_emb_keys`` get one MLP per embodiment (the usual
    choice for the specific stream); the rest share one.
    """

    def __init__(self, in_keys: List[str], dims: List[int],
                 out_keys: Optional[List[str]] = None,
                 n_layers: int = 4, d_hidden: Optional[List[int]] = None,
                 per_emb_keys: Optional[List[str]] = None,
                 embodiments: Optional[List[str]] = None,
                 activation: str = "gelu", dropout: float = 0.0,
                 final_norm: bool = True):
        super().__init__()
        self.in_keys = [str(k) for k in in_keys]
        self.out_keys = [str(k) for k in (out_keys or in_keys)]
        if len(self.in_keys) != len(dims):
            raise ValueError("Framewise: one dim per in_key.")
        self.reads = list(self.in_keys) + ["embodiment"]
        self.writes = list(self.out_keys)
        acts = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
        if activation not in acts:
            raise ValueError("Framewise activation %r not in %s"
                             % (activation, sorted(acts)))
        act = acts[activation]
        pe = set(per_emb_keys or [])
        hid = list(d_hidden) if d_hidden else [4 * int(d) for d in dims]
        blocks = {}
        for k, d, h in zip(self.in_keys, dims, hid):
            mk = lambda d=d, h=h: _MLPBlocks(int(d), int(h), int(n_layers),
                                             act, float(dropout),
                                             bool(final_norm))
            blocks[k] = per_emb(mk, embodiments) if k in pe else mk()
        self.blocks = nn.ModuleDict(
            {k.replace(".", "_"): v for k, v in blocks.items()})
        self._key_map = {k: k.replace(".", "_") for k in self.in_keys}

    def forward(self, batch: dict) -> dict:
        emb = batch["embodiment"]
        for ik, ok in zip(self.in_keys, self.out_keys):
            # pick() passes shared (non-PerEmb) modules through unchanged.
            mod = pick(self.blocks[self._key_map[ik]], emb)
            batch[ok] = mod(batch[ik])
            if ok != ik:
                ck, mk, tk = _grid_keys(ik)
                if ck in batch:
                    _write_grid(batch, ok, batch[ck], batch[mk], batch[tk])
        return batch


class Chunk(Stage):
    """Router + compress, over a LIST of streams sharing one boundary.

    ``in_keys[0]`` is the AGNOSTIC stream; it always drives the compression.
    Which streams are listed decides the behaviour that used to need flags:

    * ``in_keys: [A_L0, S_L0] -> out_keys: [A_L1, S_L1]`` -- both streams
      compressed on the shared boundary (the old dual chunker).
    * ``in_keys: [A_L0] -> out_keys: [A_L1]`` with ``route_on: a`` -- only A is
      compressed and the router scores A alone; S is simply not named here, so
      it stays on the frame grid (the old ``flat_s``).

    ``route_on`` picks what enters the router's cossim core: ``"fused"`` (A+S),
    ``"a"``, or ``"s"``. A single-stream ``in_keys`` requires ``"a"`` -- there
    is no S to fuse.
    """

    def __init__(self, in_keys: List[str], out_keys: List[str], route_key: str,
                 dims: List[int], apex_dims: Optional[List[int]] = None,
                 route_on: str = "fused", level: Optional[int] = None,
                 target_compression_ratio: float = 8.0,
                 ratio_loss_weight: float = 0.03,
                 ratio_loss_weight_s: float = 0.0,
                 grab_prev_end: bool = True,
                 embodiments: Optional[List[str]] = None,
                 per_emb_keys: Optional[List[str]] = None,
                 router_per_emb: bool = True,
                 router_fusion: str = "residual_mlp",
                 router_core: str = "cossim",
                 d_router: Optional[int] = None,
                 router_mixer_n_layers: int = 4,
                 router_hidden_mult: float = 4.0,
                 router_temp: float = 1.0, router_sample: bool = False,
                 res_scale: Optional[float] = None):
        super().__init__()
        self.in_keys = [str(k) for k in in_keys]
        self.out_keys = [str(k) for k in out_keys]
        self.route_key = str(route_key)
        self.dims = [int(d) for d in dims]
        self.apex_dims = [int(d) for d in (apex_dims or dims)]
        n = len(self.in_keys)
        if not (len(self.out_keys) == len(self.dims) == len(self.apex_dims) == n):
            raise ValueError(
                "Chunk: in_keys/out_keys/dims/apex_dims must be the same length.")
        self.route_on = str(route_on)
        # Explicit pyramid level. The NESTED form appends aux/chunker records
        # innermost-first (each chunker appends after self.inner returns),
        # while a flat list appends in execution order -- so list POSITION
        # means opposite things in the two forms. RatioLoss indexes per-level
        # target_ratio by position, so stamping the level makes per-level
        # targets/schedules unambiguous instead of order-dependent.
        self.level = None if level is None else int(level)
        if n == 1 and self.route_on != "a":
            raise ValueError(
                "Chunk: a single-stream in_keys has no S to fuse; "
                "set route_on='a'.")
        self.reads = list(self.in_keys) + ["embodiment", "cu_seqlens", "max_seq_len", "time_pos"]
        self.writes = (list(self.out_keys) + [f"{k}@res" for k in self.in_keys]
                       + [self.route_key, "aux/chunker"])
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        # MATCHES DualChunkerLevel (stages_hnet.py:471): an unset `_s` weight
        # falls back to the A weight. With one shared boundary that applies the
        # ratio term TWICE on the same b_t -- i.e. the effective weight is 2x
        # the configured value. That looks accidental upstream, but every
        # existing baseline trained under it, so reproducing it is required for
        # the sequential baseline to be comparable. Pass ratio_loss_weight_s=0
        # explicitly to opt out.
        self.ratio_loss_weight_s = (float(ratio_loss_weight_s)
                                    if ratio_loss_weight_s
                                    else float(ratio_loss_weight))
        self.grab_prev_end = bool(grab_prev_end)
        self.res_scale = res_scale
        self._embs = list(embodiments) if embodiments else None
        pe = set(per_emb_keys or [])
        self._per_emb = [k in pe for k in self.in_keys]

        d_a = self.dims[0]
        d_s = self.dims[1] if n > 1 else d_a
        mk_router = lambda: DualStreamRouter(
            d_a, d_s, d_router=d_router, router_core=str(router_core),
            n_layers=int(router_mixer_n_layers), fusion=str(router_fusion),
            hidden_mult=float(router_hidden_mult),
            router_temp=float(router_temp),
            router_sample=bool(router_sample), route_on=self.route_on)
        self._router = per_emb(mk_router, self._embs) if router_per_emb else mk_router()

        self.chunk_layer = ChunkLayer()

        def _mk(fac, i):
            return per_emb(fac, self._embs) if self._per_emb[i] else fac()

        self.residual_proj = nn.ModuleList([
            _mk((lambda d=d: nn.Identity() if a == d else nn.Linear(d, d)), i)
            for i, (d, a) in enumerate(zip(self.dims, self.apex_dims))])
        if self.grab_prev_end:
            self.prev_end_combine = nn.ModuleList([
                _mk((lambda d=d, a=a: nn.Linear(2 * d, a)), i)
                for i, (d, a) in enumerate(zip(self.dims, self.apex_dims))])
            # no_prev must be PER-EMBODIMENT for specific streams, matching
            # DualChunkerLevel's per_emb no_prev_S. A bare Parameter and a
            # PerEmb (which wraps a ParameterDict) cannot share one container,
            # so per-emb entries live in a ModuleList and shared ones are
            # registered directly; _no_prev() resolves either.
            self.no_prev_pe = nn.ModuleList()
            self._no_prev_ref = []
            for i, d in enumerate(self.dims):
                if self._per_emb[i] and self._embs:
                    self.no_prev_pe.append(
                        per_emb(lambda d=d: nn.Parameter(torch.zeros(d)), self._embs))
                    self._no_prev_ref.append(("pe", len(self.no_prev_pe) - 1))
                else:
                    self.register_parameter(f"no_prev_{i}",
                                            nn.Parameter(torch.zeros(d)))
                    self._no_prev_ref.append(("p", i))
            self.proj_in = None
        else:
            self.prev_end_combine = None
            self.no_prev_pe = None
            self._no_prev_ref = []
            self.proj_in = nn.ModuleList([
                _mk((lambda d=d, a=a: nn.Identity() if a == d else nn.Linear(d, a)), i)
                for i, (d, a) in enumerate(zip(self.dims, self.apex_dims))])

    def _no_prev(self, i: int, emb):
        kind, j = self._no_prev_ref[i]
        return (pick(self.no_prev_pe[j], emb) if kind == "pe"
                else getattr(self, f"no_prev_{j}"))

    def _combine_prev_end(self, chunked, x, bmask, cu, combine, no_prev, d):
        prev = torch.zeros_like(chunked[:, :d]) + no_prev
        idx = torch.nonzero(bmask, as_tuple=False).flatten()
        if idx.numel():
            first = torch.isin(idx, cu[:-1])
            taken = x[(idx - 1).clamp(min=0)]
            prev = torch.where(first.unsqueeze(-1), prev, taken)
        return combine(torch.cat([chunked, prev], -1).float()).to(chunked.dtype)

    def forward(self, batch: dict) -> dict:
        emb = batch["embodiment"]
        xs = [batch[k] for k in self.in_keys]
        A = xs[0]
        cu, mx, tp = _read_grid(batch, self.in_keys[0], A)
        S = xs[1] if len(xs) > 1 else A
        cu_s = (_read_grid(batch, self.in_keys[1], S)[0] if len(xs) > 1 else cu)

        bpred = pick(self._router, emb)(A, S, cu_seqlens=cu, max_seqlen=mx,
                                        cu_seqlens_s=cu_s)
        bmask, bprob, sprob = (bpred.boundary_mask, bpred.boundary_prob,
                               bpred.selected_probs)

        residuals, next_cu, next_max = [], None, None
        for i, (ik, ok, d) in enumerate(zip(self.in_keys, self.out_keys, self.dims)):
            x = xs[i]
            res = pick(self.residual_proj[i], emb)(x.float())
            if self.res_scale is not None:
                res = res * float(self.res_scale)
            residuals.append(res)
            x_ch, ncu, nmx, _ = self.chunk_layer(x, bmask, cu_seqlens=cu)
            if self.grab_prev_end:
                x_in = self._combine_prev_end(
                    x_ch, x, bmask, cu, pick(self.prev_end_combine[i], emb),
                    self._no_prev(i, emb), d)
            else:
                x_in = pick(self.proj_in[i], emb)(x_ch)
            next_cu, next_max = ncu, nmx
            batch[ok] = x_in

        ctp = _within_episode_time_pos(next_cu, batch[self.out_keys[0]].shape[0],
                                       A.device)
        for ok in self.out_keys:
            _write_grid(batch, ok, next_cu, int(next_max), ctp)
        # The residual (this level's input, projected) is the U-net SKIP. Kept
        # inside the route dict it is invisible to any consumer that reasons
        # about dataflow -- the routing graph drew Chunk->Dechunk as a single
        # "route_LN" edge with no sign that the pre-chunk activations flow
        # across too. Publishing it as a named key makes the skip a real edge
        # and leaves `route` as pure metadata.
        for ik, res in zip(self.in_keys, residuals):
            batch[f"{ik}@res"] = res
        route = {
            "boundary_mask": bmask,
            "boundary_prob": bprob,
            "selected_probs": sprob,
            "cu_seqlens": cu,               # this level's INPUT grid
            "chunk_cu_seqlens": next_cu,    # chunk grid
            "residual_keys": [f"{k}@res" for k in self.in_keys],
            "in_keys": list(self.in_keys),
            "target_ratio": self.target_compression_ratio,
            "ratio_weight": self.ratio_loss_weight,
        }
        batch[self.route_key] = route
        # Also publish in the legacy aux/chunker shape so stages_io.RatioLoss
        # works UNCHANGED -- it reads that list, and its loss math is exactly
        # what we want. Optional-regularizer fields are emitted inert (0/None)
        # because this stage deliberately does not implement them; see the
        # module docstring for the full list.
        if "aux/chunker" in batch and len(self.in_keys) == 1:
            # A-ONLY chunker: mirrors DualChunkerLevel._forward_flat_s, which
            # emits a MINIMAL record with NO _s keys -- there is no S boundary
            # at this level, so an S ratio term would double-count the SAME
            # boundary and silently double the effective ratio weight.
            batch["aux/chunker"].append({
                "boundary_mask": bmask,
                "boundary_prob": bprob,
                "selected_probs": sprob,
                "cu_seqlens": cu,
                "chunk_cu_seqlens": next_cu,
                "target_ratio": self.target_compression_ratio,
                "ratio_weight": self.ratio_loss_weight,
                "flat_s": True,
                "level": self.level,
                "separate_boundaries": False,
                "tokens": batch[self.out_keys[0]],
            })
        elif "aux/chunker" in batch:
            batch["aux/chunker"].append({
                **{k: route[k] for k in
                   ("boundary_mask", "boundary_prob", "selected_probs",
                    "cu_seqlens", "chunk_cu_seqlens", "target_ratio",
                    "ratio_weight")},
                # no separate per-stream boundaries here: S (when chunked) rides
                # A's mask, so the S entries mirror A and carry zero weight.
                "level": self.level,
                "separate_boundaries": False,
                "boundary_mask_s": bmask,
                "boundary_prob_s": bprob,
                "selected_probs_s": sprob,
                "chunk_cu_seqlens_s": next_cu,
                "cu_seqlens_s": cu,
                "target_ratio_s": self.target_compression_ratio,
                "ratio_weight_s": float(self.ratio_loss_weight_s),
                "dec_weight": 0.0, "spread_weight": 0.0, "spread_alpha": 0.5,
                "hb_weight": 0.0, "hb_lo": 0.25, "hb_hi": 0.45, "hb_window": 12,
                "recon_loss": None, "recon_weight": 0.0,
                "cons_loss": None, "cons_weight": 0.0,
                "topk_weight": 0.0, "topk_margin": 0.1,
                "softrate_weight": 0.0, "softrate_vote": "sigmoid",
                "softrate_tau": 0.15, "softrate_center": 0.5,
                "softrate2_weight": 0.0, "softrate2_center": 0.3,
                "indecision_weight": 0.0, "indecision_deadzone": 0.21,
                "tokens": batch[self.out_keys[0]],   # chunkviz reads A tokens
            })
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        return parent_residuals


class Dechunk(Stage):
    """Expand back to the route's input grid and residual-mix.

    The second half of what ``DualChunkerLevel.forward`` used to do after its
    ``self.inner(...)`` call.
    """

    def __init__(self, in_keys: List[str], route_key: str, out_keys: List[str],
                 dims: List[int], apex_dims: Optional[List[int]] = None,
                 residual_keys: Optional[List[str]] = None,
                 residual_mixer: str = "mlp",
                 residual_mixer_kwargs: Optional[dict] = None,
                 per_emb_keys: Optional[List[str]] = None,
                 embodiments: Optional[List[str]] = None,
                 ste_gain: float = 1.0):
        super().__init__()
        self.in_keys = [str(k) for k in in_keys]
        self.out_keys = [str(k) for k in out_keys]
        self.route_key = str(route_key)
        self.dims = [int(d) for d in dims]
        self.apex_dims = [int(d) for d in (apex_dims or dims)]
        n = len(self.in_keys)
        if not (len(self.out_keys) == len(self.dims) == len(self.apex_dims) == n):
            raise ValueError(
                "Dechunk: in_keys/out_keys/dims/apex_dims must be the same length.")
        # Which residual each stream mixes with -- name them so the skip shows
        # up in the routing graph. Falls back to the route record's list.
        self.residual_keys = ([str(k) for k in residual_keys] if residual_keys
                              else None)
        self.reads = (list(self.in_keys) + [self.route_key, "embodiment"]
                      + (self.residual_keys or []))
        self.writes = list(self.out_keys)
        self.ste_gain = float(ste_gain)
        pe = set(per_emb_keys or [])
        self._per_emb = [k in pe for k in self.out_keys]
        embs = list(embodiments) if embodiments else None

        def _mk(fac, i):
            return per_emb(fac, embs) if self._per_emb[i] else fac()

        self.proj_out = nn.ModuleList([
            _mk((lambda d=d, a=a: nn.Identity() if a == d else nn.Linear(a, d)), i)
            for i, (d, a) in enumerate(zip(self.dims, self.apex_dims))])
        self.dechunk = nn.ModuleList([DeChunkLayer(d) for d in self.dims])
        self.mixers = nn.ModuleList([
            _mk((lambda d=d: build_residual_mixer(
                residual_mixer, d, **(residual_mixer_kwargs or {}))), i)
            for i, d in enumerate(self.dims)])

    def forward(self, batch: dict) -> dict:
        r = batch[self.route_key]
        emb = batch["embodiment"]
        res_keys = self.residual_keys or r["residual_keys"]
        for i, (ik, ok) in enumerate(zip(self.in_keys, self.out_keys)):
            x = pick(self.proj_out[i], emb)(batch[ik])
            dech = self.dechunk[i](x, r["boundary_mask"], r["boundary_prob"],
                                   cu_seqlens=r["chunk_cu_seqlens"])
            gated = dech.float() * _ste_scaled(r["selected_probs"], self.ste_gain)
            rk = res_keys[i]
            if rk not in batch:
                raise KeyError(f"Dechunk: residual {rk!r} not in batch -- the "
                               f"matching Chunk must run first and publish it.")
            out = pick(self.mixers[i], emb)(gated, batch[rk])
            batch[ok] = out.to(dech.dtype)
            _write_grid(batch, ok, r["cu_seqlens"],
                        int(batch.get("max_seq_len", 0)),
                        _within_episode_time_pos(r["cu_seqlens"],
                                                 out.shape[0], out.device))
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        return parent_residuals + 1


class Apex(Stage):
    """Isotropic stack on ONE stream key (the old ``ApexLevel``, un-nested)."""

    def __init__(self, in_key: str, main_network: dict,
                 out_key: Optional[str] = None, causal: bool = True):
        super().__init__()
        self.in_key = str(in_key)
        self.out_key = str(out_key or in_key)
        self.reads = [self.in_key, "cu_seqlens", "max_seq_len", "time_pos"]
        self.writes = [self.out_key, "apex/tokens"]
        self.causal = bool(causal)
        self.net = build_isotropic(dict(main_network), d_cond=0, causal=causal)

    def forward(self, batch: dict) -> dict:
        x = batch[self.in_key]
        cu, mx, tp = _read_grid(batch, self.in_key, x)
        out = self.net(x, cu_seqlens=cu, max_seqlen=mx)
        batch[self.out_key] = out
        batch["apex/tokens"] = out
        _write_grid(batch, self.out_key, cu, mx, tp)
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        return _init_isotropic_linears(self.net, rng, parent_residuals)


class Mix(Stage):
    """Recombine two same-resolution streams through a residual mixer.

    Used for ``A_agn = mixer(A_from_pyramid, A_framewise)`` -- the per-frame
    branch rejoining the chunked path.
    """

    def __init__(self, in_keys: List[str], out_key: str, dim: int,
                 mixer: str = "mlp", mixer_kwargs: Optional[dict] = None):
        super().__init__()
        if len(in_keys) != 2:
            raise ValueError("Mix takes exactly two in_keys (main, residual).")
        self.in_keys = [str(k) for k in in_keys]
        self.out_key = str(out_key)
        self.reads = list(self.in_keys)
        self.writes = [self.out_key]
        self.mixer = build_residual_mixer(mixer, int(dim), **(mixer_kwargs or {}))

    def forward(self, batch: dict) -> dict:
        main, res = batch[self.in_keys[0]], batch[self.in_keys[1]]
        out = self.mixer(main.float(), res.float()).to(main.dtype)
        batch[self.out_key] = out
        ck, mk, tk = _grid_keys(self.in_keys[0])
        if ck in batch:
            _write_grid(batch, self.out_key, batch[ck], batch[mk], batch[tk])
        return batch

    def _init_weights(self, rng: float, parent_residuals: int) -> int:
        return parent_residuals + 1


class Rename(Stage):
    """Alias batch keys (e.g. ``S_out -> s``) so heads keep their own names."""

    def __init__(self, mapping: dict):
        super().__init__()
        self.mapping = {str(k): str(v) for k, v in dict(mapping).items()}
        self.reads = list(self.mapping)
        self.writes = list(self.mapping.values())

    def forward(self, batch: dict) -> dict:
        for src, dst in self.mapping.items():
            batch[dst] = batch[src]
            ck, mk, tk = _grid_keys(src)
            if ck in batch:
                _write_grid(batch, dst, batch[ck], batch[mk], batch[tk])
        return batch


class ScheduledRatioLoss(_RatioLoss):
    """:class:`stages_io.RatioLoss` with a SCHEDULABLE compression target.

    The base class resolves ``target_ratio`` as a scalar / per-level list /
    per-embodiment dict, all constant in time. Curriculum on the chunker's
    target (e.g. start loose at 8x and tighten to 3x) needs it to vary with
    training progress, so this subclass accepts two extra forms:

      * a CALLABLE ``f(progress) -> value`` (hydra ``_partial_: true``), or
      * a spec dict ``{start, end, over_steps, shape: linear|cosine}``.

    ``progress`` is in [0, 1]. It is read from ``batch["global_step"]`` when the
    wrapper supplies it; otherwise this stage counts its own TRAIN forwards --
    the same fallback ``DualChunkerLevel`` uses for its residual anneal, since
    Lightning's ``global_step`` is not currently placed in the batch.

    Everything the base class does is unchanged: only the resolved target is
    swapped before ``forward``. ``ratio_weight`` still comes from the Chunk
    stage (the base class has no weight argument).
    """

    def __init__(self, *args, schedule=None, over_steps: int = 0,
                 log_key: str = "log/ratio_target", **kw):
        super().__init__(*args, **kw)
        self.schedule = schedule
        self.over_steps = int(over_steps)
        self.log_key = str(log_key)
        self._fwd = 0

    def _progress(self, batch: dict) -> float:
        step = batch.get("global_step")
        if step is None:
            if self.training:
                self._fwd += 1
            step = self._fwd
        if self.over_steps <= 0:
            return 0.0
        return max(0.0, min(1.0, float(step) / float(self.over_steps)))

    def _resolve(self, p: float):
        s = self.schedule
        if callable(s):
            return float(s(p))
        lo, hi = float(s["start"]), float(s["end"])
        if str(s.get("shape", "linear")) == "cosine":
            import math
            p = 0.5 * (1.0 - math.cos(math.pi * p))
        return lo + (hi - lo) * p

    def forward(self, batch: dict) -> dict:
        recs = batch.get("aux/chunker")
        if recs and all(r.get("level") is not None for r in recs):
            # index by CONFIG level, not append order (see Chunk.level)
            batch["aux/chunker"] = sorted(recs, key=lambda r: r["level"])
        if self.schedule is not None:
            p = self._progress(batch)
            tgt = self._resolve(p)
            self.target_ratio = tgt          # base _target() picks this up
            batch[self.log_key] = float(tgt)
        return super().forward(batch)


class ObsNoise(Stage):
    """Observation augmentation as an explicit STAGE.

    DESIGN RULE for this module: every stage computes the SAME function in
    training and inference. That keeps a rollout faithful to what was trained
    and means no stage silently changes behaviour when you switch to eval.

    This stage is the ONE deliberate exception, and it is declared rather than
    hidden: with ``train_only: true`` (the default) it is an exact pass-through
    at eval/rollout and only perturbs during training. Previously this lived on
    ``PipelineAlgo.train_obs_transforms``, where both the placement and the
    train-only gate were invisible from the stage list.

    Reuses the existing ``egomimic.pipeline.obs_transforms`` callables (e.g.
    ``GaussianImageNoise``) -- they are ``dict -> dict`` over the un-prefixed
    obs names, so this stage strips/restores the ``obs/`` prefix around them.

    Place it BEFORE the obs encoders. Because it is a stage you can now put it
    anywhere in the list -- e.g. perturb the image but not the proprio, or
    inject on one stream's input only.
    """

    def __init__(self, transforms: List[object], keys: Optional[List[str]] = None,
                 prefix: str = "obs/", train_only: bool = True):
        super().__init__()
        self.transforms = list(transforms)
        self.keys = [str(k) for k in keys] if keys else None
        self.prefix = str(prefix)
        self.train_only = bool(train_only)
        self.reads = [f"{self.prefix}*"]
        self.writes = [f"{self.prefix}*"]

    def forward(self, batch: dict) -> dict:
        if self.train_only and not self.training:
            return batch                      # exact identity at inference
        sel = (self.keys if self.keys is not None
               else [k[len(self.prefix):] for k in batch
                     if k.startswith(self.prefix)])
        obs = {k: batch[self.prefix + k] for k in sel if self.prefix + k in batch}
        if not obs:
            return batch
        for t in self.transforms:
            obs = t(obs)
        for k, v in obs.items():
            batch[self.prefix + k] = v
        return batch


# --------------------------------------------------------------------------- #
# OBS stages -- the encoder path, un-nested
#
# SharedObsEncoders / ObsEncoders hid the A/S routing INSIDE the class: what
# reached which stream depended on which class you picked and where a key was
# declared, not on anything visible in the config. That cost several silent
# bugs (a key that never arrived, an object pose that reached both streams).
# These three stages make every hop explicit and named, so "shared encoder"
# vs "separate encoders" is just how many VisualEncode stages you list.
# --------------------------------------------------------------------------- #


class VisualEncode(Stage):
    """One image key -> one named feature. Share it by pointing two Fuse
    stages at the same out_key; separate it by listing two of these."""

    def __init__(self, in_key: str, out_key: str, encoder,
                 per_emb: bool = False, embodiments: Optional[List[str]] = None):
        super().__init__()
        self.in_key, self.out_key = str(in_key), str(out_key)
        self.reads, self.writes = [self.in_key, "embodiment"], [self.out_key]
        self.enc = per_emb_module(encoder, per_emb, embodiments)

    def forward(self, batch: dict) -> dict:
        x = batch.get(self.in_key)
        if x is None:
            raise KeyError(f"VisualEncode: {self.in_key!r} not in batch "
                           f"(have: {sorted(k for k in batch if k.startswith('obs/'))}).")
        batch[self.out_key] = pick(self.enc, batch["embodiment"])(x)
        return batch


class ObsEmbed(Stage):
    """One proprio key -> one named embedding (Linear, optionally per-emb).

    Raises on a missing key: a silently-dropped proprio trains the model on
    vision alone with no error, which is exactly how object_x once ended up
    standing in for a rotation channel.
    """

    def __init__(self, in_key: str, out_key: str, input_dim, embed_dim: int,
                 per_emb: bool = False, embodiments: Optional[List[str]] = None):
        super().__init__()
        self.in_key, self.out_key = str(in_key), str(out_key)
        self.reads, self.writes = [self.in_key, "embodiment"], [self.out_key]
        # input_dim may be a scalar OR a per-embodiment mapping: pusher_pose is
        # 2-D for a circle pusher and 4-D for a rotvec u_socket, so one width
        # cannot serve a cotrain run.
        if hasattr(input_dim, "items"):
            if not embodiments:
                raise ValueError("ObsEmbed: per-emb input_dim needs embodiments.")
            self.input_dim = {str(e): int(d) for e, d in dict(input_dim).items()}
            self.emb = nn.ModuleDict({
                e: nn.Linear(self.input_dim[e], int(embed_dim)) for e in self.input_dim})
            self._per_emb_in = True
        else:
            self.input_dim = int(input_dim)
            self._per_emb_in = False
            mk = lambda: nn.Linear(int(input_dim), int(embed_dim))
            self.emb = per_emb_module(mk, per_emb, embodiments)

    def forward(self, batch: dict) -> dict:
        x = batch.get(self.in_key)
        if x is None:
            raise KeyError(f"ObsEmbed: {self.in_key!r} not in batch "
                           f"(have: {sorted(k for k in batch if k.startswith('obs/'))}). "
                           f"Transforms/encoders see KEY_MAP ALIASES, not zarr paths.")
        emb = str(batch["embodiment"])
        if self._per_emb_in:
            if emb not in self.emb:
                raise KeyError(f"ObsEmbed: no entry for embodiment {emb!r} "
                               f"(have {list(self.emb)}).")
            want, mod = self.input_dim[emb], self.emb[emb]
        else:
            want, mod = self.input_dim, pick(self.emb, emb)
        if x.shape[-1] != want:
            raise ValueError(f"ObsEmbed: {self.in_key!r} width {x.shape[-1]} != "
                             f"declared input_dim {want} for embodiment {emb!r}.")
        batch[self.out_key] = mod(x.float())
        return batch


class Fuse(Stage):
    """Concat named features -> MLP -> one stream key at width ``d_out``.

    This is where a stream is actually composed, so what reaches A vs S is
    readable from the config instead of implied by an encoder class.
    """

    def __init__(self, in_keys: List[str], out_key: str, dims: List[int],
                 d_out: int, widths: Optional[List[int]] = None,
                 per_emb: bool = False, embodiments: Optional[List[str]] = None):
        super().__init__()
        self.in_keys = [str(k) for k in in_keys]
        self.out_key = str(out_key)
        self.reads = list(self.in_keys) + ["embodiment"]
        self.writes = [self.out_key]
        fused = int(sum(dims))
        ws = list(widths) if widths else [max(int(d_out), fused)]
        def mk():
            layers, prev = [], fused
            for w in ws:
                layers += [nn.Linear(prev, int(w)), nn.GELU()]
                prev = int(w)
            layers.append(nn.Linear(prev, int(d_out)))
            return nn.Sequential(*layers)
        self.proj = per_emb_module(mk, per_emb, embodiments)

    def forward(self, batch: dict) -> dict:
        feats = []
        for k in self.in_keys:
            if k not in batch:
                raise KeyError(f"Fuse: {k!r} not in batch (have: {sorted(batch)[:12]}).")
            feats.append(batch[k].float())
        out = pick(self.proj, batch["embodiment"])(torch.cat(feats, dim=-1))
        batch[self.out_key] = out
        return batch


class TimePos(Stage):
    """Publish ``time_pos`` -- the within-episode frame index from cu_seqlens.

    ``ObsEncoders``/``SharedObsEncoders`` used to emit this as a side effect of
    encoding. Splitting the encoder into explicit stages made that implicit
    write disappear, and the trunk's mask/RoPE need it -- so it becomes its own
    one-line stage rather than a hidden effect of something else.
    """

    reads = ["cu_seqlens"]
    writes = ["time_pos"]

    def forward(self, batch: dict) -> dict:
        from egomimic.pipeline import packed
        cu = batch["cu_seqlens"]
        batch["time_pos"] = packed.frame_idx(
            cu.to(dtype=torch.long) if torch.is_tensor(cu) else cu)
        return batch
