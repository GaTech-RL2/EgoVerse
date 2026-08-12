"""Modular multi-stream transformer trunk for the dual/multi-stream H-Net.

Generalizes TwoStreamTrunk to N TIME-ALIGNED streams (one *agnostic* + one per
embodiment / *specific*), each with its OWN ``d_model_i`` / ``num_heads_i`` /
``d_intermediate_i``. Every block is two decoupled sub-modules:

  * IntraStreamModule  (per stream): self-attention + SwiGLU MLP over that
    stream's own tokens, in its own ``d_model_i``. "Compute within the stream."
  * InterStreamComm    (per target stream): cross-attention where stream i queries
    the streams it is allowed to attend, projecting each source ``d_model_j ->
    d_model_i``. "Communication across streams." Which streams a stream may attend
    is a config ``adjacency`` (generalizes the old asym/sym rule:
        asym: agnostic attends [] (cross), specific attends [agnostic]
        sym:  every stream attends every other).

Per-embodiment SPECIFIC streams (``embodiments`` given):
  A dual-stream transformer is "the SAME agnostic compute (shared across
  embodiments) PLUS a per-embodiment specific compute (swapped per embodiment)".
  So stream 0 (the AGNOSTIC stream) is a single SHARED module set, while every
  stream with index >= 1 (a SPECIFIC stream) is materialized once PER EMBODIMENT
  (an ``nn.ModuleDict`` keyed by the domain name == ``ctx.embodiment_id``) and
  only the active embodiment's copy runs on a given forward. Consequences:
    * total params: the specific path is counted N times (once per embodiment).
    * active params: counted ONCE (only the active embodiment runs).
  The cross-stream comm A->S writes INTO a specific stream, so it too is per-emb;
  S->A (writing into the shared agnostic stream) is FORBIDDEN under per-emb (it
  would inject embodiment-specific info into the shared stream) -> use asym.
  When ``embodiments`` is None the trunk is fully SHARED (BC; identical param
  names to upstream) — used for single-embodiment runs / the legacy 2-trunk.

All streams share T_total / episode boundaries / within-episode time index (they
are the same obs sequence, different feature streams), so a single (T,T) bool
mask (same-episode & time-causal) drives BOTH the intra self-attention and the
inter cross-attention. PACKED path only. Reuses blocks.py primitives unchanged.

Per-block order (this build): intra (all streams) -> inter (all streams, from the
post-intra snapshot so comm is order-independent).
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.hnet.blocks import RMSNorm, SwiGLU, _apply_rotary, _RotaryCache
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.per_emb import PerEmb, per_emb, pick
from egomimic.models.hnet.stages import _BaseStage


def _sdpa(q, k, v, mask, dropout, training):
    """q:(Tq,H,Dh) k/v:(Tk,H,Dh) mask:(Tq,Tk) bool -> (Tq, H*Dh)."""
    Tq, H, Dh = q.shape
    q = q.permute(1, 0, 2)[None]  # (1,H,Tq,Dh)
    k = k.permute(1, 0, 2)[None]
    v = v.permute(1, 0, 2)[None]
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[None, None], dropout_p=dropout if training else 0.0
    )
    return o[0].permute(1, 0, 2).reshape(Tq, H * Dh)


# Per-embodiment dispatch lives in per_emb.py (the single reusable abstraction):
#   per_emb(factory, embodiments) -> PerEmb (per-emb) | bare module (shared, BC)
#   pick(x, emb_id)               -> the active embodiment's module/param
_pick = pick  # backward-compatible alias


def _per_stream(x, i):
    """Return the per-stream element when ``x`` is a list/tuple, else ``x`` itself.

    Lets ``mask``/``rope_positions`` be EITHER one shared tensor (legacy: every
    stream on the same token grid) OR one per stream (per-stream chunk
    boundaries, where A and S have different lengths).
    """
    return x[i] if isinstance(x, (list, tuple)) else x


class IntraStreamModule(nn.Module):
    """Self-attention + SwiGLU MLP within ONE stream (its own d_model)."""

    def __init__(self, d_model, num_heads, d_intermediate, rotary_emb_dim, dropout=0.0,
                 ffn_dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads, self.head_dim = d_model, num_heads, d_model // num_heads
        self.dropout = float(dropout)
        self.norm_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.rotary_emb_dim = int(rotary_emb_dim)
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be <= head_dim"
        self.rotary = _RotaryCache(self.rotary_emb_dim) if self.rotary_emb_dim > 0 else None
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_intermediate, dropout=ffn_dropout)

    def forward(self, x, mask, rope_positions):
        T, H, Dh = x.shape[0], self.num_heads, self.head_dim
        h = self.norm_attn(x)
        q, k, v = self.qkv(h).reshape(T, 3, H, Dh).unbind(1)
        if self.rotary is not None:
            cos, sin = self.rotary.get(rope_positions, q.dtype)
            cos, sin = cos[:, None, :], sin[:, None, :]
            q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        o = _sdpa(q, k, v, mask, self.dropout, self.training)
        x = x + self.out(o)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class InterStreamComm(nn.Module):
    """Cross-attention: target stream (query, d_model_q) attends a list of source
    streams, projecting each source d_model_j -> d_model_q. Time-aligned: source
    keys share the query's rope positions and the same (T,T) causal/episode mask.
    """

    def __init__(self, d_model_q, num_heads, src_dims: List[int], rotary_emb_dim, dropout=0.0):
        super().__init__()
        assert d_model_q % num_heads == 0
        self.d_model_q, self.num_heads, self.head_dim = d_model_q, num_heads, d_model_q // num_heads
        self.dropout = float(dropout)
        self.norm_q = RMSNorm(d_model_q)
        self.q_proj = nn.Linear(d_model_q, d_model_q, bias=True)
        self.norm_kv = nn.ModuleList([RMSNorm(dj) for dj in src_dims])
        self.k_proj = nn.ModuleList([nn.Linear(dj, d_model_q, bias=True) for dj in src_dims])
        self.v_proj = nn.ModuleList([nn.Linear(dj, d_model_q, bias=True) for dj in src_dims])
        self.out = nn.Linear(d_model_q, d_model_q, bias=True)
        self.rotary_emb_dim = int(rotary_emb_dim)
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be <= head_dim"
        self.rotary = _RotaryCache(self.rotary_emb_dim) if self.rotary_emb_dim > 0 else None

    def forward(self, x_q, src_xs: List[torch.Tensor], mask, rope_positions):
        T, H, Dh = x_q.shape[0], self.num_heads, self.head_dim
        q = self.q_proj(self.norm_q(x_q)).reshape(T, H, Dh)
        ks, vs = [], []
        for j, xj in enumerate(src_xs):
            hj = self.norm_kv[j](xj)
            ks.append(self.k_proj[j](hj).reshape(T, H, Dh))
            vs.append(self.v_proj[j](hj).reshape(T, H, Dh))
        if self.rotary is not None:
            cos, sin = self.rotary.get(rope_positions, q.dtype)
            cos, sin = cos[:, None, :], sin[:, None, :]
            q = _apply_rotary(q, cos, sin)
            ks = [_apply_rotary(kj, cos, sin) for kj in ks]
        k = torch.cat(ks, 0)  # (Nsrc*T, H, Dh)
        v = torch.cat(vs, 0)
        cross_mask = mask.repeat(1, len(src_xs))  # (T, Nsrc*T) — sources time-aligned
        o = _sdpa(q, k, v, cross_mask, self.dropout, self.training)
        return x_q + self.out(o)


class MultiStreamBlock(nn.Module):
    """One block: intra (all streams) then inter (all streams that have sources,
    from the post-intra snapshot so comm is order-independent).

    If ``embodiments`` is given, every SPECIFIC stream (index >= 1) is built
    per-embodiment (``nn.ModuleDict`` keyed by domain name); the AGNOSTIC stream
    (index 0) is shared. The cross-stream comm into a specific stream is also
    per-emb. ``forward`` selects the active embodiment's copy via ``emb_id``.
    """

    def __init__(self, streams_cfg: List[dict], adjacency: List[List[int]], rotary_emb_dim,
                 dropout=0.0, ffn_dropout=0.0, embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False):
        super().__init__()
        self.adjacency = [list(a) for a in adjacency]
        embs = list(embodiments) if embodiments else None
        self.embodiments = embs
        # By default the AGNOSTIC stream (idx 0) is forbidden from reading any
        # per-emb specific stream (that would contaminate the shared stream with
        # embodiment-specific info). `allow_agnostic_cross=True` is an explicit
        # opt-in for the sym ABLATION (contaminated-A vs uncontaminated-A): it
        # lets idx 0 cross-attend the specific stream(s); the agnostic cross-attn
        # itself stays SHARED (see inter dict below) so only A's *information*
        # is contaminated, not its parameter-sharing.
        if embs is not None and self.adjacency[0] and not allow_agnostic_cross:
            raise ValueError(
                "per-emb dual-stream forbids cross-stream sources on the AGNOSTIC "
                "stream (idx 0): it would inject embodiment-specific info into the "
                "shared agnostic stream. Use mask_mode='asym' (or set "
                "allow_agnostic_cross=True for the contaminated-A ablation)."
            )
        # RoPE dim is PER-STREAM: each stream's head_dim differs, so clamp the
        # global rotary_emb_dim to each stream's head_dim (or take an explicit
        # per-stream 'rotary_emb_dim' from its cfg).
        def _rot(s):
            hd = int(s["d_model"]) // int(s["num_heads"])
            return int(s.get("rotary_emb_dim", min(int(rotary_emb_dim), hd)))

        def _mk_intra(i):
            s = streams_cfg[i]
            return IntraStreamModule(s["d_model"], s["num_heads"], s["d_intermediate"], _rot(s), dropout,
                                      ffn_dropout=ffn_dropout)

        def _mk_inter(i, srcs):
            src_dims = [streams_cfg[j]["d_model"] for j in srcs]
            return InterStreamComm(
                streams_cfg[i]["d_model"], streams_cfg[i]["num_heads"], src_dims, _rot(streams_cfg[i]), dropout
            )

        # stream 0 = AGNOSTIC (shared); stream >=1 = SPECIFIC (per-emb). One path:
        # per_emb(factory, None) -> shared (BC); per_emb(factory, embs) -> per-emb.
        self.intra = nn.ModuleList([
            per_emb(lambda i=i: _mk_intra(i), embs if i >= 1 else None)
            for i in range(len(streams_cfg))
        ])
        # inter only exists for streams with sources (asym -> only specific i>=1;
        # sym/allow_agnostic_cross also gives idx 0 an inter). Keep idx 0's inter
        # SHARED (embs->None) so the agnostic stream gains no per-emb PARAMETERS;
        # specific streams (i>=1) stay per-emb.
        self.inter = nn.ModuleDict({
            str(i): per_emb(lambda i=i, srcs=srcs: _mk_inter(i, srcs), embs if i >= 1 else None)
            for i, srcs in enumerate(self.adjacency) if srcs
        })

    def forward(self, xs: List[torch.Tensor], mask, rope_positions, emb_id: Optional[str] = None):
        xs = [pick(self.intra[i], emb_id)(xs[i], _per_stream(mask, i),
                                          _per_stream(rope_positions, i))
              for i in range(len(xs))]
        post_intra = list(xs)  # snapshot -> order-independent comm
        for i, srcs in enumerate(self.adjacency):
            if srcs:
                # Cross-stream attention is only defined when the source and query
                # streams live on the SAME token grid. With per-stream chunk
                # boundaries (separate_boundaries) the grids differ, so the model
                # must be configured with no inter-stream comm at chunk resolution.
                if any(post_intra[j].shape[0] != post_intra[i].shape[0] for j in srcs):
                    raise ValueError(
                        "cross-stream attention needs equal stream lengths "
                        f"(stream {i}: {post_intra[i].shape[0]}, sources: "
                        f"{[post_intra[j].shape[0] for j in srcs]}). With per-stream "
                        "boundaries set adjacency=[[], []] so the streams stay "
                        "independent while chunked and rejoin after dechunk."
                    )
                inter = pick(self.inter[str(i)], emb_id)
                xs[i] = inter(post_intra[i], [post_intra[j] for j in srcs],
                              _per_stream(mask, i), _per_stream(rope_positions, i))
        return xs


class MultiStreamTrunk(nn.Module):
    """n_layers MultiStreamBlocks + per-stream final norm. Every stream fully
    independent in d_model / num_heads / d_intermediate; N streams; configurable
    inter-stream comm adjacency.

    ``embodiments`` (optional) makes every SPECIFIC stream (idx>=1) per-embodiment
    (the final norm too); the AGNOSTIC stream (idx 0) stays shared.
    """

    def __init__(self, streams_cfg: List[dict], adjacency: List[List[int]], n_layers,
                 rotary_emb_dim, dropout=0.0, ffn_dropout=0.0,
                 embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False, adaln_dim: Optional[int] = None):
        super().__init__()
        self.n_streams = len(streams_cfg)
        self.n_layers = int(n_layers)
        self.streams_cfg = streams_cfg
        self.embodiments = list(embodiments) if embodiments else None
        # AdaLN conditioning on the SPECIFIC streams only (batchflow cvae_zs):
        # per-block, per-S-stream Linear(adaln_dim -> 2*d_s) producing (gamma,
        # beta); applied BEFORE each block as x_s <- x_s*(1+gamma)+beta.
        # adaLN-zero init => identity at start; A (idx 0) NEVER modulated.
        self.adaln_dim = int(adaln_dim) if adaln_dim else None
        if self.adaln_dim:
            def _zero_lin(d_out):
                m = nn.Linear(self.adaln_dim, d_out)
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
                m.weight._no_reinit = True
                return m
            self.adaln = nn.ModuleList([
                nn.ModuleList([_zero_lin(2 * int(streams_cfg[i]["d_model"]))
                               for i in range(1, len(streams_cfg))])
                for _ in range(int(n_layers))
            ])
        self.blocks = nn.ModuleList([
            MultiStreamBlock(streams_cfg, adjacency, rotary_emb_dim, dropout, ffn_dropout=ffn_dropout,
                             embodiments=embodiments,
                             allow_agnostic_cross=allow_agnostic_cross)
            for _ in range(int(n_layers))
        ])
        # final norm: agnostic (idx0) shared, specific (idx>=1) per-emb.
        embs = self.embodiments
        self.norm_f = nn.ModuleList([
            per_emb(lambda s=s: RMSNorm(s["d_model"]), embs if i >= 1 else None)
            for i, s in enumerate(streams_cfg)
        ])

    @property
    def height(self) -> int:
        return 2 * self.n_layers

    def forward(self, xs: List[torch.Tensor], mask, rope_positions,
                emb_id: Optional[str] = None,
                cond: Optional[torch.Tensor] = None,
                layer_masks: Optional[List] = None) -> List[torch.Tensor]:
        """layer_masks: optional ONE MASK PER BLOCK (len == n_layers) so a level
        can give block 0 a narrow window and the rest full history. None
        (default) => every block uses `mask`, byte-identical to before. Each
        entry follows the same convention as `mask` (one tensor, or one per
        stream)."""
        if layer_masks is not None and len(layer_masks) != len(self.blocks):
            raise ValueError(
                "layer_masks has %d entries but the trunk has %d blocks -- "
                "one mask per block." % (len(layer_masks), len(self.blocks)))
        for li, blk in enumerate(self.blocks):
            if cond is not None and self.adaln_dim:
                for i in range(1, len(xs)):
                    g, b = self.adaln[li][i - 1](cond).chunk(2, -1)
                    xs = list(xs)
                    xs[i] = xs[i] * (1 + g) + b
            xs = blk(xs, mask if layer_masks is None else layer_masks[li],
                     rope_positions, emb_id)
        return [pick(self.norm_f[i], emb_id)(xs[i]) for i in range(len(xs))]


class MultiStreamComputeStage(_BaseStage):
    """HNet root stage wrapping MultiStreamTrunk. Reads the per-stream token list
    from ctx.extras['ms'] (set by MultiStreamOuterStage.encode), builds the (T,T)
    same-episode & time-causal mask, runs the trunk, returns the AGNOSTIC
    (stream 0) top, and stashes the rest in ctx.extras['ms_tops'] +
    ctx.extras['specific_tokens'] (= stream-1 top, for the 2-stream head).
    PACKED path only; closed-loop AR is recompute-over-prefix (no KV-cache).

    streams_cfg: list of {d_model, num_heads, d_intermediate[, rotary_emb_dim]}.
      stream 0 is the agnostic stream; its d_model MUST equal input_hidden_dim.
    adjacency: optional explicit per-stream source lists; if None it is derived
      from mask_mode (asym: agnostic->[], specific->[0]; sym: all-to-all).
    embodiments: optional list of domain names; when given, every SPECIFIC stream
      (idx>=1) is per-embodiment (the specific compute is swapped per embodiment),
      while the agnostic stream stays shared. Dispatched on ctx.embodiment_id.
    """

    def __init__(self, input_hidden_dim, output_hidden_dim, streams_cfg,
                 adjacency=None, n_layers=16, rotary_emb_dim=0, mask_mode="asym",
                 cond_key=None, d_cond=0, causal=True, depth_split_k=None, dropout=0.0,
                 embodiments: Optional[List[str]] = None):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if int(input_hidden_dim) != int(output_hidden_dim):
            raise ValueError("MultiStreamComputeStage requires input==output hidden dim.")
        streams_cfg = [dict(s) for s in streams_cfg]
        if int(streams_cfg[0]["d_model"]) != int(input_hidden_dim):
            raise ValueError(
                f"streams_cfg[0].d_model ({streams_cfg[0]['d_model']}) (agnostic) must "
                f"equal input_hidden_dim ({input_hidden_dim})."
            )
        if mask_mode not in ("asym", "sym"):
            raise ValueError(f"mask_mode must be 'asym'|'sym', got {mask_mode!r}")
        N = len(streams_cfg)
        if adjacency is None:
            if mask_mode == "sym":
                adjacency = [[j for j in range(N) if j != i] for i in range(N)]
            else:  # asym: agnostic (0) reads no cross; each specific reads agnostic
                adjacency = [[] if i == 0 else [0] for i in range(N)]
        self.mask_mode = mask_mode
        self.causal = bool(causal)
        self.streams_cfg = streams_cfg
        self.embodiments = list(embodiments) if embodiments else None
        self.trunk = MultiStreamTrunk(streams_cfg, adjacency, n_layers, rotary_emb_dim, dropout,
                                      embodiments=embodiments)

    def _build_mask(self, T_total, cu_seqlens, time_pos, device):
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        ep = torch.zeros(T_total, dtype=torch.long, device=device)
        for b in range(cu.shape[0] - 1):
            ep[cu[b]:cu[b + 1]] = b
        tp = time_pos.to(device=device, dtype=torch.long)
        same_ep = ep[:, None] == ep[None, :]
        if self.causal:
            time_ok = tp[None, :] <= tp[:, None]
        else:
            time_ok = torch.ones(T_total, T_total, dtype=torch.bool, device=device)
        allow = (same_ep & time_ok) | torch.eye(T_total, dtype=torch.bool, device=device)
        return allow

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError("MultiStreamComputeStage: PACKED path only.")
        if self.inner_stage is not None:
            raise NotImplementedError("MultiStreamComputeStage is terminal.")
        ms = ctx.extras.get("ms")
        if ms is None:
            raise RuntimeError(
                "MultiStreamComputeStage requires ctx.extras['ms'] "
                "(set by MultiStreamOuterStage.encode)."
            )
        streams = ms["streams"]
        T_total = int(ms["T_total"])
        device = streams[0].device
        mask = self._build_mask(T_total, ms["cu_seqlens"], ms["time_pos"], device)
        rope_positions = ms["time_pos"].to(device=device, dtype=torch.long)
        tops = self.trunk(streams, mask, rope_positions, ctx.embodiment_id)
        ctx.extras["ms_tops"] = tops
        ctx.extras["specific_tokens"] = tops[1] if len(tops) > 1 else None
        return tops[0]

    # ---- AR / framework stubs (recompute-over-prefix; mirror DualStream) ----
    def step(self, x, ctx, state):
        raise NotImplementedError(
            "MultiStreamComputeStage has no single-token step; AR is "
            "recompute-over-prefix (MultiStreamOuterStage.step calls forward)."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None

    def allocate_inference_cache(self, batch_size, max_seqlen=None, device=None, dtype=None):
        return None

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        n_residuals = parent_residuals + self.trunk.height
        scaled_std = initializer_range / max(n_residuals, 1) ** 0.5
        for name, m in self.trunk.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if getattr(m.weight, "_no_reinit", False):
                continue
            # residual-out projections (attn .out + SwiGLU down-proj) get scaled
            if name.endswith("out") or "w2" in name or "fc2" in name or "down" in name:
                nn.init.normal_(m.weight, mean=0.0, std=scaled_std)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        return n_residuals
