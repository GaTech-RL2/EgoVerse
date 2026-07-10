"""DualStreamChunkerStage — dual-stream chunker for the v2 H-Net pyramid.

Two inner modes:
  * ``dual_inner=False`` (AGNOSTIC SEAM, default): only A is lifted into the inner
    (the single-stream agnostic apex ``ComputeStage``); S is chunked + dechunked
    at d_s (no taper, bypasses the apex). This realizes "the main trunk consumes
    only the agnostic stream."
  * ``dual_inner=True``: BOTH A and S are lifted (proj_in_A->apex_a, S combine->
    apex_s) and passed as a StreamBundle to a DUAL inner compute
    (DualBundleComputeStage); both dechunk back. Used for the intermediate
    (non-apex) pyramid levels.

Boundary: a DUAL-STREAM router (``DualStreamRouter``) fuses A + S with a 4-layer
MLP mixing head, then runs the cossim core on the fusion to emit ONE shared
boundary (chunks A+S at the SAME positions; one ratio aux). per-stream duplicate
dechunk + STE + mlp residual mixer.

Per-embodiment vs shared (all via the single ``per_emb`` abstraction):
  * AGNOSTIC / SHARED (one copy): the A-path params (prev_end_combine_A, no_prev_A,
    proj_in/out_A, residual_proj_A, residual_mixer_A) and the parameterless
    chunk/dechunk ops.
  * SPECIFIC / PER-EMB (one copy per embodiment, keyed by ctx.embodiment_id): every
    parametric S-path piece (prev_end_combine_S, no_prev_S, proj_in/out_S,
    residual_proj_S, residual_mixer_S, optional A->S inject).
  * ROUTER: ``router_per_emb=True`` -> a full DualStreamRouter PER EMBODIMENT
    (boundaries are embodiment-specific); ``router_per_emb=False`` -> ONE shared
    "combined" router fusing A+S. Toggle to A/B the two.
When ``embodiments`` is None everything collapses to the SHARED layout (BC).
"""

from typing import List, Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.per_emb import PerEmb, per_emb, pick
from egomimic.models.hnet.residual_mixer import build_residual_mixer
from egomimic.models.hnet.routing import ChunkLayer, DeChunkLayer, RoutingModule
from egomimic.models.hnet.isotropic_builder import build_isotropic
from egomimic.models.hnet.stages import _BaseStage, _init_isotropic_linears, _ste
from egomimic.models.hnet.stream_bundle import StreamBundle


class DualStreamRouter(nn.Module):
    """Boundary router fusing the agnostic A and specific S streams, then the cossim
    ``RoutingModule`` on the fusion -> ONE boundary. WHAT ENTERS THE COSINE is the
    fused ``h`` (A+S), not A alone; the cossim core scores adjacent ``h`` tokens.

    Two fusion modes (``fusion``):
      * ``"residual_mlp"`` (default): ``h = proj_a(A) + proj_s(S) + MLP([proj_a(A),
        proj_s(S)])`` (the ``MLPMixer`` used by the dechunk mixers). The additive
        ``proj_a(A)+proj_s(S)`` base exists ONLY for a WARM START == route-on-A:
        proj_a identity-init, proj_s zero-init, MLP last layer zero-init -> at init
        h == A, so boundaries are byte-identical to the single-stream router and S's
        influence is learned. Training-stable; costs the 4x-hidden MLPMixer.
      * ``"mlp"``: a PURE n-layer MLP over ``concat[A, S]`` -> d_router (no additive
        base, NO warm start -> boundaries start random; simpler, and cheaper at
        hidden=d_router*hidden_mult with a thinner mult). Use to A/B whether the
        warm-start base helps.
    Both feed the same cossim core. ``DualStreamChunkerStage`` wraps this whole
    module shared ("combined") or per-embodiment (``router_per_emb``).
    """

    def __init__(self, d_a, d_s, d_router=None, n_layers=4, fusion="residual_mlp",
                 router_pre_layout=None, router_pre_detach=False,
                 hidden_mult=4.0, router_core="cossim"):
        super().__init__()
        d_router = int(d_router) if d_router else int(d_a)
        self.d_a, self.d_s, self.d_router = int(d_a), int(d_s), d_router
        self.fusion = str(fusion)

        if self.fusion == "residual_mlp":
            self.proj_a = nn.Linear(d_a, d_router, bias=False)
            if d_router == int(d_a):
                # exact warm start == route-on-A: identity, frozen from any reinit.
                with torch.no_grad():
                    self.proj_a.weight.copy_(torch.eye(d_router))
                self.proj_a.weight._no_reinit = True
            else:
                # d_router != d_a: no exact route-on-A warm start possible; give a
                # proper init (NOT left at default-and-frozen). proj_s zero-init still
                # makes S start neutral, so boundaries start from a fixed proj of A.
                nn.init.normal_(self.proj_a.weight, mean=0.0, std=0.02)
            self.proj_s = nn.Linear(d_s, d_router, bias=False)
            nn.init.zeros_(self.proj_s.weight)
            self.proj_s.weight._no_reinit = True
            self.mixer = build_residual_mixer("mlp", d_router, n_layers=int(n_layers),
                                              hidden_mult=float(hidden_mult))
        elif self.fusion == "mlp":
            # pure MLP: no warm start (boundaries random at init by design). Give it a
            # clean explicit init -- it lives outside the stage chain so init_weights
            # never reaches it; leaving it at PyTorch default would be unintentional.
            self.fuse = self._build_mlp(d_a + d_s, d_router, int(n_layers),
                                        int(d_router * hidden_mult))
            for m in self.fuse:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"router fusion must be 'residual_mlp'|'mlp', got {fusion!r}")
        # Router pre-net (ported from the single-stream ChunkerStage /
        # EgoVerse-gmm-rp): optional small CAUSAL transformer PRIVATE to the
        # boundary router, applied to the FUSED A+S representation before the
        # cossim core. DEFAULT None => no module, byte-identical.
        self.router_pre_detach = bool(router_pre_detach)
        self.router_pre = (
            build_isotropic(
                {
                    "arch_layout": router_pre_layout,
                    "d_model": d_router,
                    "d_intermediate": 3 * d_router,
                    "num_heads": max(1, d_router // 64),
                    "cond": False,
                    "rotary_emb_dim": 64,
                },
                d_cond=0,
                causal=True,
            )
            if router_pre_layout is not None
            else None
        )
        if router_core == "cossim":
            self.router = RoutingModule(d_router)
        elif router_core == "sigmoid":
            from egomimic.models.hnet.routing import SigmoidRoutingModule
            self.router = SigmoidRoutingModule(d_router)
        else:
            raise ValueError(f"router_core must be 'cossim'|'sigmoid', got {router_core!r}")

    @staticmethod
    def _build_mlp(d_in, d_out, n_layers, hidden):
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, d_out)]
        return nn.Sequential(*layers)

    def forward(self, A, S, cu_seqlens=None, max_seqlen=None):
        if self.fusion == "residual_mlp":
            h = self.mixer(self.proj_a(A), self.proj_s(S))   # <- enters the cosine
        else:  # "mlp": pure MLP over concat[A, S]
            h = self.fuse(torch.cat([A, S], dim=-1))
        if self.router_pre is not None:
            # Private causal transform of the fused stream for the router only.
            if self.router_pre_detach:
                h = h.detach()
            h = self.router_pre(h, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return self.router(h, cu_seqlens=cu_seqlens)


class DualStreamChunkerStage(_BaseStage):
    def __init__(
        self,
        input_hidden_dim: int,              # d_a (agnostic in)
        output_hidden_dim: int,             # apex_a (agnostic inner/apex working dim)
        s_hidden_dim: Optional[int] = None,    # d_s (specific in); default d_a
        s_output_hidden_dim: Optional[int] = None,  # apex_s (specific inner dim; dual_inner only)
        dual_inner: bool = False,
        target_compression_ratio: float = 8.0,
        ratio_loss_weight: float = 0.03,
        down_interface: str = "first_token",
        up_interface: str = "duplicate",
        router_interface: str = "cossim",
        grab_prev_end: bool = True,
        residual_mixer: str = "mlp",
        residual_mixer_kwargs: Optional[dict] = None,
        a_to_s_inject: bool = False,
        cond_key: Optional[str] = None,
        embodiments: Optional[List[str]] = None,
        dual_router: bool = True,
        router_per_emb: bool = True,
        router_fusion: str = "residual_mlp",
        d_router: Optional[int] = None,
        # Router pre-net (see DualStreamRouter): "T1"/"T2" etc. or None (BC).
        router_pre_layout: Optional[str] = None,
        router_pre_detach: bool = False,
        router_mixer_n_layers: int = 4,
        router_hidden_mult: float = 4.0,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if down_interface != "first_token" or up_interface != "duplicate" or router_interface != "cossim":
            raise NotImplementedError(
                "DualStreamChunkerStage v1 supports first_token/duplicate/cossim only."
            )
        d_a = int(input_hidden_dim)
        d_s = int(s_hidden_dim) if s_hidden_dim is not None else d_a
        apex_a = int(output_hidden_dim)
        apex_s = int(s_output_hidden_dim) if s_output_hidden_dim is not None else d_s
        self.d_a, self.d_s, self.apex_a, self.apex_s = d_a, d_s, apex_a, apex_s
        self.dual_inner = bool(dual_inner)
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.grab_prev_end = bool(grab_prev_end)
        self.residual_scale: float = 1.0
        self.a_to_s_inject = bool(a_to_s_inject)
        self.inner_stage: Optional[_BaseStage] = None  # set by HNet
        self._embs = list(embodiments) if embodiments else None

        # --- ROUTER (dual-stream, fuse A+S). router_per_emb -> one router PER emb
        #     ("emb-specific"); else ONE shared router ("combined"). chunk/dechunk
        #     ops are parameterless + shared. ---
        self.dual_router = bool(dual_router)
        self.router_per_emb = bool(router_per_emb)
        if self.dual_router:
            router_embs = self._embs if self.router_per_emb else None
            self._router = per_emb(
                lambda: DualStreamRouter(d_a, d_s, d_router=d_router,
                                         router_pre_layout=router_pre_layout,
                                         router_pre_detach=router_pre_detach,
                                         n_layers=router_mixer_n_layers,
                                         fusion=router_fusion,
                                         hidden_mult=router_hidden_mult),
                router_embs,
            )
        else:
            self._router = RoutingModule(d_a)  # route-on-A fallback (ablation)
        self.chunk_layer = ChunkLayer()
        self.dechunk_A = DeChunkLayer(d_a)
        self.dechunk_S = DeChunkLayer(d_s)

        mk = residual_mixer_kwargs or {}

        def _combine(d_in, d_out):
            return nn.Sequential(
                nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                nn.Linear(2 * d_in, 2 * d_in), nn.GELU(),
                nn.Linear(2 * d_in, d_out),
            )

        def _mk_res_proj(d):
            rp = nn.Linear(d, d)
            nn.init.zeros_(rp.weight)
            nn.init.zeros_(rp.bias)
            rp.weight._no_reinit = True
            return rp

        # --- A path (SHARED; always enters the inner at apex_a) ---
        if self.grab_prev_end:
            self.prev_end_combine_A = _combine(d_a, apex_a)
            self.no_prev_A = nn.Parameter(torch.randn(d_a) * 0.02)
            self.proj_in_A = None
        else:
            self.proj_in_A = nn.Linear(d_a, apex_a) if d_a != apex_a else None
        self.proj_out_A = nn.Linear(apex_a, d_a) if apex_a != d_a else None

        # --- S path (PER-EMB via per_emb; shared if embodiments is None) ---
        s_target = apex_s if self.dual_inner else d_s
        if self.grab_prev_end:
            self.prev_end_combine_S = per_emb(lambda: _combine(d_s, s_target), self._embs)
            self.no_prev_S = per_emb(lambda: nn.Parameter(torch.randn(d_s) * 0.02), self._embs)
            self.proj_in_S = None
        else:
            self.proj_in_S = (per_emb(lambda: nn.Linear(d_s, s_target), self._embs)
                              if d_s != s_target else None)
        self.proj_out_S = (per_emb(lambda: nn.Linear(apex_s, d_s), self._embs)
                           if (self.dual_inner and apex_s != d_s) else None)

        self.residual_proj_A = _mk_res_proj(d_a)
        self.residual_proj_S = per_emb(lambda: _mk_res_proj(d_s), self._embs)
        self.residual_mixer_A = build_residual_mixer(residual_mixer, d_a, **mk)
        self.residual_mixer_S = per_emb(lambda: build_residual_mixer(residual_mixer, d_s, **mk), self._embs)

        if self.a_to_s_inject:
            self.inject_proj = per_emb(lambda: nn.Linear(d_a, s_target), self._embs)
            self.inject_gate = per_emb(lambda: nn.Parameter(torch.zeros(1)), self._embs)

    # ------------------------------------------------------------------
    @property
    def inner_working_dim(self) -> int:
        return self.apex_a

    @property
    def end_to_end_output_dim(self) -> int:
        return self.d_a

    @staticmethod
    def _within_episode_time_pos(cu: torch.Tensor, T: int, device) -> torch.Tensor:
        cu = cu.to(device=device, dtype=torch.long)
        t = torch.arange(T, device=device, dtype=torch.long)
        starts = torch.zeros(T, dtype=torch.long, device=device)
        for b in range(cu.shape[0] - 1):
            starts[cu[b]:cu[b + 1]] = cu[b]
        return t - starts

    def _combine_prev_end(self, chunked, x_stream, bmask, cu, combine_mlp, no_prev):
        pos = bmask.nonzero(as_tuple=True)[0]
        prev_end = x_stream[(pos - 1).clamp(min=0)]
        is_start = torch.isin(pos, cu[:-1])
        prev_end = torch.where(is_start.unsqueeze(-1), no_prev.to(prev_end.dtype), prev_end)
        cat = torch.cat([chunked, prev_end], dim=-1).float()
        return combine_mlp(cat).to(chunked.dtype)

    def forward(self, x, ctx: HNetContext) -> StreamBundle:  # type: ignore[override]
        if not isinstance(x, StreamBundle):
            raise TypeError("DualStreamChunkerStage requires a StreamBundle.")
        if not ctx.packed:
            raise NotImplementedError("DualStreamChunkerStage supports the PACKED path only.")
        A, S = x.A, x.S
        cu = ctx.cu_seqlens
        emb = ctx.embodiment_id

        # active embodiment's SPECIFIC-path modules (shared pass-through if no embs)
        residual_proj_S = pick(self.residual_proj_S, emb)
        residual_mixer_S = pick(self.residual_mixer_S, emb)

        # ---- boundary (dual-stream router fuses A+S; or route-on-A fallback) ----
        if self.dual_router:
            bpred = pick(self._router, emb)(
                A, S, cu_seqlens=cu, max_seqlen=ctx.max_seqlen
            )
        else:
            bpred = self._router(A, cu_seqlens=cu)
        bmask, bprob, sprob = bpred.boundary_mask, bpred.boundary_prob, bpred.selected_probs

        resA = self.residual_scale * self.residual_proj_A(A.float())
        resS = self.residual_scale * residual_proj_S(S.float())

        A_ch, next_cu, next_max, _ = self.chunk_layer(A, bmask, cu_seqlens=cu)
        S_ch, _, _, _ = self.chunk_layer(S, bmask, cu_seqlens=cu)

        if self.grab_prev_end:
            A_in = self._combine_prev_end(A_ch, A, bmask, cu, self.prev_end_combine_A, self.no_prev_A)
            S_in = self._combine_prev_end(S_ch, S, bmask, cu,
                                          pick(self.prev_end_combine_S, emb), pick(self.no_prev_S, emb))
        else:
            A_in = self.proj_in_A(A_ch) if self.proj_in_A is not None else A_ch
            proj_in_S = pick(self.proj_in_S, emb) if self.proj_in_S is not None else None
            S_in = proj_in_S(S_ch) if proj_in_S is not None else S_ch

        if self.a_to_s_inject:
            inject_proj, inject_gate = pick(self.inject_proj, emb), pick(self.inject_gate, emb)
            S_in = S_in + (inject_gate * inject_proj(A_ch.float())).to(S_in.dtype)

        saved_cu, saved_max = ctx.cu_seqlens, ctx.max_seqlen
        if self.dual_inner:
            # Both streams through a DUAL inner compute (StreamBundle in/out).
            N = A_in.shape[0]
            ctp = self._within_episode_time_pos(next_cu, N, A.device)
            bundle_in = StreamBundle(A=A_in, S=S_in, T_total=N, cu_seqlens=next_cu, time_pos=ctp)
            ctx.cu_seqlens, ctx.max_seqlen = next_cu, next_max
            try:
                bundle_out = self.inner_stage(bundle_in, ctx) if self.inner_stage is not None else bundle_in
            finally:
                ctx.cu_seqlens, ctx.max_seqlen = saved_cu, saved_max
            A_inner = self.proj_out_A(bundle_out.A) if self.proj_out_A is not None else bundle_out.A
            proj_out_S = pick(self.proj_out_S, emb) if self.proj_out_S is not None else None
            S_inner = proj_out_S(bundle_out.S) if proj_out_S is not None else bundle_out.S
            A_dech = self.dechunk_A(A_inner, bmask, bprob, cu_seqlens=next_cu)
            S_dech = self.dechunk_S(S_inner, bmask, bprob, cu_seqlens=next_cu)
        else:
            # AGNOSTIC SEAM: only A enters the inner (single-stream apex); S bypasses.
            ctx.cu_seqlens, ctx.max_seqlen = next_cu, next_max
            try:
                A_inner = self.inner_stage(A_in, ctx) if self.inner_stage is not None else A_in
            finally:
                ctx.cu_seqlens, ctx.max_seqlen = saved_cu, saved_max
            A_inner = self.proj_out_A(A_inner) if self.proj_out_A is not None else A_inner
            A_dech = self.dechunk_A(A_inner, bmask, bprob, cu_seqlens=next_cu)
            S_dech = self.dechunk_S(S_in, bmask, bprob, cu_seqlens=next_cu)

        A_out = self.residual_mixer_A(A_dech.float() * _ste(sprob), resA).to(A.dtype)
        S_out = residual_mixer_S(S_dech.float() * _ste(sprob), resS).to(S.dtype)

        ctx.register_aux({
            "bpred": bpred,
            "target_ratio": self.target_compression_ratio,
            "weight": self.ratio_loss_weight,
            "cu_seqlens": cu,
            # Chunkviz interface: PCA computes on the AGNOSTIC (A) chunk
            # tokens ONLY (user rule 2026-07-04) — S never enters the PCA.
            "viz_tokens": A_ch,
        })
        return x.replace(A=A_out, S=S_out)

    def step(self, x, ctx, state):
        raise NotImplementedError(
            "DualStreamChunkerStage has no incremental step; AR is recompute-over-prefix."
        )

    def allocate_inference_cache(self, batch_size, dtype, device):
        return None

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        # Collect every Linear in the combine/proj/inject paths (A shared + S
        # per-emb), skipping zero-init residual projections (_no_reinit). The
        # router self-inits (proj_a identity, proj_s zero, mixer/cossim _no_reinit).
        def _members(obj):
            if obj is None:
                return []
            if isinstance(obj, PerEmb):
                return list(obj.table.values())
            return [obj]

        lins: List[nn.Linear] = []
        for attr in ("prev_end_combine_A", "prev_end_combine_S"):
            for seq in _members(getattr(self, attr, None)):
                lins += [m for m in seq if isinstance(m, nn.Linear)]
        for attr in ("proj_in_A", "proj_out_A", "proj_in_S", "proj_out_S"):
            for m in _members(getattr(self, attr, None)):
                if isinstance(m, nn.Linear):
                    lins.append(m)
        if self.a_to_s_inject:
            for m in _members(getattr(self, "inject_proj", None)):
                if isinstance(m, nn.Linear):
                    lins.append(m)

        for m in lins:
            if not getattr(m.weight, "_no_reinit", False):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Router pre-net(s): private residual stream feeding only the boundary
        # router — init like any Isotropic stack scaled by its OWN depth.
        for r in _members(getattr(self, "_router", None)):
            rp = getattr(r, "router_pre", None)
            if rp is not None:
                _init_isotropic_linears(rp, initializer_range, rp.height)
        # The chunker adds no counted residual-stream depth (matches single-stream
        # ChunkerStage default); chain init into the inner stage so the rest of the
        # chain (compute levels + apex) is initialized, not left at PyTorch default.
        if self.inner_stage is not None:
            return self.inner_stage._init_weights(initializer_range, parent_residuals)
        return parent_residuals
