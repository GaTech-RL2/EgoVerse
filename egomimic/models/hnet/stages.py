"""
Stage classes for the stage-based H-Net.

A *stage* is a self-contained nn.Module that owns its own sub-modules
(encoders / chunkers / main networks) and exposes a uniform interface:

    forward(x, ctx) -> y
    step(x_step, ctx, state) -> (y_step, state)
    allocate_inference_cache(batch_size, dtype, device) -> state

Stages nest recursively: each stage carries an optional ``inner_stage``. The
top-level ``HNet`` wrapper wires the flat YAML list into a chain by setting
``stages[i].inner_stage = stages[i+1]``. The last stage's ``inner_stage``
stays None (terminal stage).

Three stage types are provided:

* ``EncoderDecoderStage`` — encoder Isotropic + (optional) inner_stage +
  decoder Isotropic with an encoder→decoder residual.
* ``ChunkerStage`` — RoutingModule + ChunkLayer + (optional) inner_stage +
  DeChunkLayer, with STE residual gating. Registers boundary predictions
  into ``ctx.aux`` for the ratio loss.
* ``ComputeStage`` — a single Isotropic main_network. Terminal by default but
  may also carry an inner_stage if the user wants further nesting.

Dimension bridging between consecutive stages is **explicit**: each stage
declares its ``input_hidden_dim`` and ``output_hidden_dim``. When the inner
stage's expected input dim differs from this stage's working dim, the stage
inserts a minimal ``nn.Linear`` projection. No padding-based bridging.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.blocks import (
    IsotropicInferenceParams,
    KVCache,
    MambaCache,
)
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.isotropic_builder import build_isotropic
from egomimic.models.hnet.register_interface import (
    RegisterInterface,
    RegisterTrunk,
    RegisterTrunkState,
)
from egomimic.models.hnet.routing import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleState,
)
from egomimic.models.hnet.scan_interface import (
    ChunkScanEncoder,
    ChunkScanEncoderState,
    MSublatentDeChunker,
    ScanRouter,
    ScanRouterState,
    SplineUpsampler,
    SplineUpsamplerState,
)
from egomimic.models.hnet.transformer_router import (
    PhaseSummaryRouter,
    PhaseSummaryRouterState,
)

# --------------------------------------------------------------------------- #
# Inference-state dataclasses (one per stage type).
# --------------------------------------------------------------------------- #


@dataclass
class EncoderDecoderStageState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    inner_state: Optional[Any] = None
    decoder_state: Optional[IsotropicInferenceParams] = None
    # closure-pattern residual carry between encoder.step and decoder.step
    # within a single step() call is handled as a local variable; no field.


@dataclass
class ChunkerStageState:
    # routing_state is a RoutingModuleState (cossim router) or a
    # ScanRouterState (scan router); both are sliced/scattered generically.
    routing_state: Optional[Any] = None
    inner_state: Optional[Any] = None
    dechunk_state: Optional[DeChunkState] = None
    # Scan/spline interface state (None on the first_token/duplicate path).
    scan_enc_state: Optional[ChunkScanEncoderState] = None
    spline_state: Optional[SplineUpsamplerState] = None
    # Register masked-transformer interface state (None unless
    # chunk_interface == "register").
    reg_trunk_state: Optional[RegisterTrunkState] = None
    # grab_prev_end: previous step's input token (B, D) for the first_token
    # chunker's "first + prev-chunk-end" combine at AR rollout. None until 1st step.
    prev_x: Optional[torch.Tensor] = None


@dataclass
class ComputeStageState:
    main_state: Optional[IsotropicInferenceParams] = None
    inner_state: Optional[Any] = None


# --------------------------------------------------------------------------- #
# Straight-through estimator for the boundary mask (matches upstream HNet).
# --------------------------------------------------------------------------- #


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _ste(x):
    return _STE.apply(x)


def _build_padded_valid_mask(shape, device) -> torch.Tensor:
    """(B, T) mask that's False at position 0 of each row (the synthetic
    boundary forced by ``RoutingModule._forward_padded``), True elsewhere.

    Used by ``ratio_loss_from_aux`` (F4) so the F/G aggregates only see
    actual predicted boundaries.
    """
    B, T = shape
    m = torch.ones(B, T, dtype=torch.bool, device=device)
    if T > 0:
        m[:, 0] = False
    return m


# --------------------------------------------------------------------------- #
# Per-batch slice/scatter helpers for nested inference state during step().
# Reused (and slightly simplified) from the previous monolithic HNet.
# --------------------------------------------------------------------------- #


def _slice_iso_params(params: Optional[IsotropicInferenceParams], mask: torch.Tensor):
    if params is None:
        return None
    new = IsotropicInferenceParams(
        layer_caches={},
        max_seqlen=params.max_seqlen,
        batch_size=int(mask.sum().item()),
    )
    for k, c in params.layer_caches.items():
        if isinstance(c, KVCache):
            new.layer_caches[k] = KVCache(
                k=c.k[mask].clone(),
                v=c.v[mask].clone(),
                offsets=c.offsets[mask].clone(),
            )
        elif isinstance(c, MambaCache):
            new.layer_caches[k] = MambaCache(
                conv_state=c.conv_state[mask].clone(),
                ssm_state=c.ssm_state[mask].clone(),
            )
        else:
            raise TypeError(f"Unknown layer cache type: {type(c)}")
    return new


def _scatter_iso_params(
    params: Optional[IsotropicInferenceParams],
    sub: Optional[IsotropicInferenceParams],
    mask: torch.Tensor,
):
    if params is None or sub is None:
        return
    for k, c in params.layer_caches.items():
        s = sub.layer_caches[k]
        if isinstance(c, KVCache):
            c.k[mask] = s.k.to(c.k.dtype)
            c.v[mask] = s.v.to(c.v.dtype)
            c.offsets[mask] = s.offsets
        elif isinstance(c, MambaCache):
            c.conv_state[mask] = s.conv_state.to(c.conv_state.dtype)
            c.ssm_state[mask] = s.ssm_state.to(c.ssm_state.dtype)
        else:
            raise TypeError(f"Unknown layer cache type: {type(c)}")


def _slice_routing_state(rs: Any, mask: torch.Tensor):
    """Slice a RoutingModuleState (cossim) or ScanRouterState (scan)."""
    if rs is None:
        return None
    if isinstance(rs, RoutingModuleState):
        return RoutingModuleState(
            has_seen_tokens=rs.has_seen_tokens[mask].clone(),
            last_hidden_state=rs.last_hidden_state[mask].clone(),
        )
    if isinstance(rs, ScanRouterState):
        return ScanRouterState(
            has_seen_tokens=rs.has_seen_tokens[mask].clone(),
            gru_h=(rs.gru_h[:, mask].clone() if rs.gru_h is not None else None),
            mamba_cache=_slice_mamba_cache(rs.mamba_cache, mask),
            h_prev=(rs.h_prev[mask].clone() if rs.h_prev is not None else None),
        )
    if isinstance(rs, PhaseSummaryRouterState):
        return PhaseSummaryRouterState(
            has_seen_tokens=rs.has_seen_tokens[mask].clone(),
            kv=[
                KVCache(
                    k=c.k[mask].clone(),
                    v=c.v[mask].clone(),
                    offsets=c.offsets[mask].clone(),
                )
                for c in rs.kv
            ],
            last_cut_pos=rs.last_cut_pos[mask].clone(),
            t_idx=rs.t_idx[mask].clone(),
            seg_sum=rs.seg_sum[mask].clone(),
            seg_count=rs.seg_count[mask].clone(),
        )
    raise TypeError(f"Unknown routing state type: {type(rs)}")


def _slice_mamba_cache(cache, mask: torch.Tensor):
    if cache is None:
        return None
    # MambaCache: conv_state / ssm_state batched on dim 0.
    return MambaCache(
        conv_state=cache.conv_state[mask].clone(),
        ssm_state=cache.ssm_state[mask].clone(),
    )


def _slice_scan_enc_state(s: Optional[ChunkScanEncoderState], mask: torch.Tensor):
    if s is None:
        return None
    return ChunkScanEncoderState(
        gru_h=s.gru_h[:, mask].clone(),
        offset=s.offset[mask].clone(),
        buf=s.buf[mask].clone(),
    )


def _slice_spline_state(s: Optional[SplineUpsamplerState], mask: torch.Tensor):
    if s is None:
        return None
    return SplineUpsamplerState(
        prev_inner_sub=s.prev_inner_sub[mask].clone(),
        have_prev=s.have_prev[mask].clone(),
        offset=s.offset[mask].clone(),
    )


def _slice_reg_trunk_state(s: Optional[RegisterTrunkState], mask: torch.Tensor):
    if s is None:
        return None
    return RegisterTrunkState(
        reg_k=[t[mask].clone() for t in s.reg_k],
        reg_v=[t[mask].clone() for t in s.reg_v],
        reg_fill=s.reg_fill[mask].clone(),
        mem_k=[t[mask].clone() for t in s.mem_k],
        mem_v=[t[mask].clone() for t in s.mem_v],
        mem_fill=s.mem_fill[mask].clone(),
        mem_x=s.mem_x[mask].clone(),
        offset=s.offset[mask].clone(),
        has_seen=s.has_seen[mask].clone(),
    )


def _scatter_reg_trunk_state(s: Optional[RegisterTrunkState], sub, mask: torch.Tensor):
    if s is None or sub is None:
        return
    for c, d in zip(s.reg_k, sub.reg_k):
        c[mask] = d.to(c.dtype)
    for c, d in zip(s.reg_v, sub.reg_v):
        c[mask] = d.to(c.dtype)
    s.reg_fill[mask] = sub.reg_fill
    for c, d in zip(s.mem_k, sub.mem_k):
        c[mask] = d.to(c.dtype)
    for c, d in zip(s.mem_v, sub.mem_v):
        c[mask] = d.to(c.dtype)
    s.mem_fill[mask] = sub.mem_fill
    s.mem_x[mask] = sub.mem_x.to(s.mem_x.dtype)
    s.offset[mask] = sub.offset
    s.has_seen[mask] = sub.has_seen


def _slice_state(state: Any, mask: torch.Tensor):
    """Generic batch-dim slice for any stage state dataclass."""
    if state is None:
        return None
    if isinstance(state, IsotropicInferenceParams):
        return _slice_iso_params(state, mask)
    if isinstance(state, EncoderDecoderStageState):
        return EncoderDecoderStageState(
            encoder_state=_slice_iso_params(state.encoder_state, mask),
            inner_state=_slice_state(state.inner_state, mask),
            decoder_state=_slice_iso_params(state.decoder_state, mask),
        )
    if isinstance(state, ChunkerStageState):
        ds = state.dechunk_state
        if ds is not None:
            ds = DeChunkState(last_value=ds.last_value[mask].clone())
        return ChunkerStageState(
            routing_state=_slice_routing_state(state.routing_state, mask),
            inner_state=_slice_state(state.inner_state, mask),
            dechunk_state=ds,
            scan_enc_state=_slice_scan_enc_state(state.scan_enc_state, mask),
            spline_state=_slice_spline_state(state.spline_state, mask),
            reg_trunk_state=_slice_reg_trunk_state(state.reg_trunk_state, mask),
            prev_x=(state.prev_x[mask].clone() if state.prev_x is not None else None),
        )
    if isinstance(state, ComputeStageState):
        return ComputeStageState(
            main_state=_slice_iso_params(state.main_state, mask),
            inner_state=_slice_state(state.inner_state, mask),
        )
    if isinstance(state, dict):
        # PerEmbodimentStage holds per-embodiment sub-states in a dict keyed by
        # embodiment_id. Slice each independently — all sub-states share the
        # same batch dim, so the boundary mask applies elementwise. (None subs
        # are handled by the early return above.)
        return {k: _slice_state(v, mask) for k, v in state.items()}
    raise TypeError(f"Unknown stage state type: {type(state)}")


def _scatter_routing_state(rs: Any, sub: Any, mask: torch.Tensor):
    if rs is None or sub is None:
        return
    if isinstance(rs, RoutingModuleState):
        rs.has_seen_tokens[mask] = sub.has_seen_tokens
        rs.last_hidden_state[mask] = sub.last_hidden_state.to(
            rs.last_hidden_state.dtype
        )
        return
    if isinstance(rs, ScanRouterState):
        rs.has_seen_tokens[mask] = sub.has_seen_tokens
        if rs.gru_h is not None and sub.gru_h is not None:
            rs.gru_h[:, mask] = sub.gru_h.to(rs.gru_h.dtype)
        if rs.h_prev is not None and sub.h_prev is not None:
            rs.h_prev[mask] = sub.h_prev.to(rs.h_prev.dtype)
        if rs.mamba_cache is not None and sub.mamba_cache is not None:
            rs.mamba_cache.conv_state[mask] = sub.mamba_cache.conv_state.to(
                rs.mamba_cache.conv_state.dtype
            )
            rs.mamba_cache.ssm_state[mask] = sub.mamba_cache.ssm_state.to(
                rs.mamba_cache.ssm_state.dtype
            )
        return
    if isinstance(rs, PhaseSummaryRouterState):
        rs.has_seen_tokens[mask] = sub.has_seen_tokens
        for c, s in zip(rs.kv, sub.kv):
            c.k[mask] = s.k.to(c.k.dtype)
            c.v[mask] = s.v.to(c.v.dtype)
            c.offsets[mask] = s.offsets
        rs.last_cut_pos[mask] = sub.last_cut_pos
        rs.t_idx[mask] = sub.t_idx
        rs.seg_sum[mask] = sub.seg_sum.to(rs.seg_sum.dtype)
        rs.seg_count[mask] = sub.seg_count
        return
    raise TypeError(f"Unknown routing state type: {type(rs)}")


def _scatter_scan_enc_state(s: Any, sub: Any, mask: torch.Tensor):
    if s is None or sub is None:
        return
    s.gru_h[:, mask] = sub.gru_h.to(s.gru_h.dtype)
    s.offset[mask] = sub.offset
    s.buf[mask] = sub.buf.to(s.buf.dtype)


def _scatter_spline_state(s: Any, sub: Any, mask: torch.Tensor):
    if s is None or sub is None:
        return
    s.prev_inner_sub[mask] = sub.prev_inner_sub.to(s.prev_inner_sub.dtype)
    s.have_prev[mask] = sub.have_prev
    s.offset[mask] = sub.offset


def _scatter_state(state: Any, sub: Any, mask: torch.Tensor):
    if state is None or sub is None:
        return
    if isinstance(state, IsotropicInferenceParams):
        _scatter_iso_params(state, sub, mask)
        return
    if isinstance(state, EncoderDecoderStageState):
        _scatter_iso_params(state.encoder_state, sub.encoder_state, mask)
        _scatter_state(state.inner_state, sub.inner_state, mask)
        _scatter_iso_params(state.decoder_state, sub.decoder_state, mask)
        return
    if isinstance(state, ChunkerStageState):
        _scatter_routing_state(state.routing_state, sub.routing_state, mask)
        _scatter_state(state.inner_state, sub.inner_state, mask)
        if state.dechunk_state is not None and sub.dechunk_state is not None:
            state.dechunk_state.last_value[mask] = sub.dechunk_state.last_value.to(
                state.dechunk_state.last_value.dtype
            )
        _scatter_scan_enc_state(state.scan_enc_state, sub.scan_enc_state, mask)
        _scatter_spline_state(state.spline_state, sub.spline_state, mask)
        _scatter_reg_trunk_state(state.reg_trunk_state, sub.reg_trunk_state, mask)
        if state.prev_x is not None and sub.prev_x is not None:
            state.prev_x[mask] = sub.prev_x.to(state.prev_x.dtype)
        return
    if isinstance(state, ComputeStageState):
        _scatter_iso_params(state.main_state, sub.main_state, mask)
        _scatter_state(state.inner_state, sub.inner_state, mask)
        return
    if isinstance(state, dict):
        # PerEmbodimentStage per-embodiment sub-states (mirror of _slice_state).
        for k in state:
            _scatter_state(state[k], sub.get(k) if isinstance(sub, dict) else sub, mask)
        return
    raise TypeError(f"Unknown stage state type: {type(state)}")


# --------------------------------------------------------------------------- #
# Shared base — handles inner_stage wiring assertion and the optional
# input/output projections triggered only when input_hidden_dim != working dim.
# --------------------------------------------------------------------------- #


class _BaseStage(nn.Module):
    """Common scaffolding for stage classes."""

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        cond_key: str = "fused_cond",
    ):
        super().__init__()
        self.input_hidden_dim = int(input_hidden_dim)
        self.output_hidden_dim = int(output_hidden_dim)
        self.cond_key = cond_key
        # Filled in by HNet.__init__ when chaining stages.
        self.inner_stage: Optional["_BaseStage"] = None

    def _get_cond(self, ctx: HNetContext) -> Optional[torch.Tensor]:
        if self.cond_key is None:
            return None
        return ctx.cond_dict.get(self.cond_key)

    @property
    def inner_working_dim(self) -> int:
        """Hidden dim at which this stage's inner_stage operates.

        Default: ``input_hidden_dim``. ``ChunkerStage`` overrides to
        ``output_hidden_dim`` (its inner trunk runs in the chunked space).
        ``PerEmbodimentStage`` delegates to its wrapped sub-stage. Used by
        ``HNet`` chain assembly for the dim-handoff check (no isinstance).
        """
        return self.input_hidden_dim

    @property
    def end_to_end_output_dim(self) -> int:
        """Hidden dim this stage's ``forward`` ACTUALLY returns at (end to end).

        Default = ``output_hidden_dim``. ``ChunkerStage`` overrides to
        ``input_hidden_dim`` because it DECHUNKS back to the outer (input) dim
        before returning — its ``output_hidden_dim`` is the INNER working dim,
        not the returned dim. ``PerEmbodimentStage`` delegates. Read by
        ``HNet.output_hidden_dim`` so a chunker-rooted chain reports its true
        end-to-end output dim. Backward-compatible: every existing stage-0 is
        dim-preserving (input==output) so this changes no committed config.
        """
        return self.output_hidden_dim

    # Subclasses implement these.
    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def step(self, x: torch.Tensor, ctx: HNetContext, state: Any):
        raise NotImplementedError

    def allocate_inference_cache(self, batch_size: int, dtype, device):
        raise NotImplementedError

    # ----- Training recipe hooks (mirror upstream hnet/models/hnet.py) ----- #

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        """Initialize Linear weights in this stage using residual-stream-aware
        scaling. Returns the cumulative ``n_residuals`` after this stage's
        contribution (used by the parent's recursion into sibling stages).

        Default behavior is no-op: subclasses override to scale their inner
        Isotropic stacks' ``out_proj`` / ``fc2`` weights by
        ``initializer_range / sqrt(n_residuals)`` and recurse into
        ``inner_stage``.
        """
        return parent_residuals

    def _apply_lr_multiplier(self, lr_multipliers, stage_idx: int) -> None:
        """Stamp every parameter in this stage (and ``inner_stage``) with the
        corresponding LR multiplier via ``apply_optimization_params``. The
        outer stage gets ``lr_multipliers[stage_idx]``; ``inner_stage`` gets
        ``stage_idx + 1``. Indexing past the list is a config error.
        """
        from egomimic.models.hnet.hnet import apply_optimization_params

        if stage_idx >= len(lr_multipliers):
            raise IndexError(
                f"lr_multipliers list has {len(lr_multipliers)} entries but "
                f"stage tree wants index {stage_idx}; provide one multiplier "
                f"per stage in the flat list passed to ``HNet([...])``."
            )
        mul = float(lr_multipliers[stage_idx])
        for p in self.parameters(recurse=False):
            apply_optimization_params(p, lr_multiplier=mul)
        # Recurse into our own owned sub-modules (encoder/decoder/main_network),
        # but skip ``inner_stage`` here — that's a sibling in the flat list and
        # handled by its own _apply_lr_multiplier call at stage_idx+1.
        for name, child in self.named_children():
            if name == "inner_stage":
                continue
            for p in child.parameters():
                apply_optimization_params(p, lr_multiplier=mul)
        if self.inner_stage is not None:
            self.inner_stage._apply_lr_multiplier(lr_multipliers, stage_idx + 1)


# --------------------------------------------------------------------------- #
# EncoderDecoderStage
# --------------------------------------------------------------------------- #


class EncoderDecoderStage(_BaseStage):
    """encoder → (optional inner_stage) → +encoder-residual → decoder.

    The residual is a *local* variable inside forward / step (option C closure
    pattern): no ``self.residual`` slot is mutated across calls.

    Hidden-dim contract:
        x_in.shape[-1] == input_hidden_dim == output_hidden_dim
    (EncoderDecoderStages do not change hidden dim themselves; if the inner
    stage uses a different working dim, the inner stage handles its own
    projections.)
    """

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
        cond_key: str = "fused_cond",
        d_cond: int = 0,
        causal: bool = True,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if input_hidden_dim != output_hidden_dim:
            raise ValueError(
                f"EncoderDecoderStage requires input_hidden_dim==output_hidden_dim "
                f"(got {input_hidden_dim} vs {output_hidden_dim}). Use a "
                f"ChunkerStage or ComputeStage to change hidden dim."
            )
        if int(encoder["d_model"]) != input_hidden_dim:
            raise ValueError(
                f"encoder.d_model ({encoder['d_model']}) must equal stage "
                f"input_hidden_dim ({input_hidden_dim})."
            )
        if int(decoder["d_model"]) != output_hidden_dim:
            raise ValueError(
                f"decoder.d_model ({decoder['d_model']}) must equal stage "
                f"output_hidden_dim ({output_hidden_dim})."
            )

        self.causal = causal
        self.encoder = build_isotropic(encoder, d_cond=d_cond, causal=causal)
        self.decoder = build_isotropic(decoder, d_cond=d_cond, causal=causal)

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        # F1 (upstream parity): there is exactly ONE residual around the
        # chunker, and it lives INSIDE the ChunkerStage as the STE-gated
        # ``residual_proj(encoder_output)`` path (matches upstream
        # ``residual_func = out * ste_func(p) + residual`` in
        # ``hnet/models/hnet.py``). We must NOT add a second un-gated
        # encoder→decoder residual at this outer level; doing so would feed
        # the encoder output around the chunker twice. The inner_stage
        # (chunker) returns the already-residual-mixed tensor.
        cond = self._get_cond(ctx)
        if ctx.packed:
            x = self.encoder(
                x, cu_seqlens=ctx.cu_seqlens, max_seqlen=ctx.max_seqlen, cond=cond
            )
            if self.inner_stage is not None:
                x = self.inner_stage(x, ctx)
            x = self.decoder(
                x, cu_seqlens=ctx.cu_seqlens, max_seqlen=ctx.max_seqlen, cond=cond
            )
        else:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            x = self.encoder(x, mask=mask, cond=cond)
            if self.inner_stage is not None:
                x = self.inner_stage(x, ctx)
            x = self.decoder(x, mask=mask, cond=cond)
        return x

    def step(self, x: torch.Tensor, ctx: HNetContext, state: EncoderDecoderStageState):
        # F1 (upstream parity): single residual lives inside the chunker; no
        # outer encoder→decoder residual here.
        cond = self._get_cond(ctx)
        x = self.encoder.step(x, state.encoder_state, cond=cond)
        if self.inner_stage is not None:
            x = self.inner_stage.step(x, ctx, state.inner_state)
        x = self.decoder.step(x, state.decoder_state, cond=cond)
        return x

    def allocate_inference_cache(self, batch_size, dtype, device):
        # The algo will call HNet.allocate_inference_cache(batch_size, max_seqlen, ...)
        # which forwards max_seqlen through; for simplicity we keep a separate path.
        raise NotImplementedError(
            "Use _allocate(batch_size, max_seqlen, dtype, device)."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return EncoderDecoderStageState(
            encoder_state=self.encoder.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            ),
            inner_state=(
                self.inner_stage._allocate(batch_size, max_seqlen, dtype, device)
                if self.inner_stage is not None
                else None
            ),
            decoder_state=self.decoder.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            ),
        )

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        # Encoder + decoder both contribute to the residual stream of this
        # stage. Inner stage runs at the same outer dim and contributes
        # additional residuals which are passed through to its own init.
        n_residuals = parent_residuals + self.encoder.height + self.decoder.height
        _init_isotropic_linears(self.encoder, initializer_range, n_residuals)
        _init_isotropic_linears(self.decoder, initializer_range, n_residuals)
        if self.inner_stage is not None:
            n_residuals = self.inner_stage._init_weights(initializer_range, n_residuals)
        return n_residuals


# --------------------------------------------------------------------------- #
# ChunkerStage
# --------------------------------------------------------------------------- #


class ChunkerStage(_BaseStage):
    """RoutingModule → ChunkLayer → (optional inner_stage) → DeChunkLayer.

    Includes the STE residual gating from the upstream H-Net so the dechunked
    signal can fall back to a learned residual when no boundary fires.

    Hidden-dim contract:
        Working dim outside the chunker = ``input_hidden_dim``.
        Working dim of the inner stage  = ``output_hidden_dim``.
        If they differ, the stage inserts an explicit Linear pre-chunk and a
        matching Linear post-chunk so dechunk operates back at the outer dim.
    """

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        target_compression_ratio: float = 2.0,
        ratio_loss_weight: float = 0.03,
        cond_key: str = "fused_cond",
        down_interface: str = "first_token",
        up_interface: str = "duplicate",
        router_interface: str = "cossim",
        m_sublatents: int = 4,
        n_phase_freqs: int = 4,
        # --- Scan-interface stability/flexibility knobs (vault config-gated
        #     stability/flexibility task). ALL DEFAULT OFF so existing
        #     scan/cossim/PSER/register configs are byte-identical (no new
        #     params created, live-job requeue safe). Used only on the scan
        #     down-encoder / scan router respectively. ---
        encoder_layernorm: bool = False,  # ChunkScanEncoder post-GRU LayerNorm
        router_n_layer: int = 1,  # ScanRouter MambaBlock-stack depth
        router_use_block: bool = False,  # ScanRouter: wrap scan in MambaBlock(s)
        router_has_mlp: bool = False,  # ScanRouter MambaBlock: add SwiGLU MLP
        router_mlp_dim: Optional[int] = None,  # SwiGLU width (None -> d_model)
        # --- PhaseSummaryRouter ("transformer" / PSER) knobs. Used ONLY when
        #     router_interface == "transformer"; ignored otherwise so existing
        #     scan/cossim configs are byte-identical. ---
        pser_depth: int = 3,
        pser_n_heads: int = 4,
        pser_ff_mult: int = 4,
        pser_L_min: int = 2,
        pser_T_temp: float = 1.0,
        pser_causal: bool = True,
        pser_variant: str = "pser",
        pser_rotary_emb_dim: Optional[int] = None,
        pser_w_l: float = 1.0,
        pser_dwell_sigma: float = 0.5,
        # --- Register masked-transformer chunk interface knobs. Used ONLY when
        #     chunk_interface == "register"; ignored otherwise (default
        #     "scan_spline") so existing scan/cossim/PSER configs are byte-
        #     identical (no register modules created). The register interface is
        #     a FUSED design: ONE masked transformer over the augmented
        #     (members+registers) stream realizes down+inner+up, so down/up/
        #     router selectors above still pick the BOUNDARY router but the
        #     scan-down / spline-up modules are bypassed. ---
        chunk_interface: str = "scan_spline",
        reg_m: int = 4,
        reg_trunk_depth: int = 2,
        reg_trunk_n_heads: int = 4,
        reg_trunk_ff_mult: int = 4,
        reg_trunk_rotary_emb_dim: int = 0,
        # Gated encoder-end RMSNorm on the register encoder's chunk-rate output
        # (the paper's "network normalization" at the component/encoder end).
        # Used by BOTH the fused chunk_interface=="register" path (no-op there:
        # the trunk emits member outputs, end_norm is only applied on the
        # register-DOWN gather) and the down_interface=="register" path. DEFAULT
        # OFF -> no new param, byte-identical to existing register configs.
        component_end_norm: bool = False,
        # FIX #7: residual-aware init for the RegisterTrunk's attn out_proj / MLP
        # fc2 (scaled by initializer_range/sqrt(n_residuals), counting 2*depth
        # residual adds). DEFAULT OFF so the existing fused register config's
        # init is byte-unchanged (BC); turn ON for the new arch. Param COUNT is
        # identical either way — only init VALUES differ.
        reg_trunk_residual_init: bool = False,
        # --- Transformer-arch fixes (vault 30_Projects/RL2/Transformer HNet/Findings).
        #     Used on down=register / up=msublatent; default OFF = current arch. ---
        # FIX 1 (capacity grab): run the inner trunk at CHUNK rate, not sub-latent
        #     rate -- concat each chunk's m register sub-latents into ONE chunk-token
        #     before the trunk (proj_in: m*D->d_model) and split after (proj_out:
        #     d_model->m*D), so the trunk sees N tokens not N*m and chunk_len 1 no
        #     longer buys m-fold trunk capacity.
        trunk_chunk_rate: bool = False,
        # FIX 2 (residual bypass): swappable mixer for out = mix(up, residual),
        #     replacing additive `up + residual`. "additive" (default = current)
        #     | "mlp" | "attn"; mlp/attn start ~= additive. See residual_mixer.py.
        residual_mixer: str = "additive",
        # extra kwargs forwarded to the mixer ctor (e.g. {n_layers: 4,
        # hidden_mult: 4.0} for mlp). Empty = the mixer's own defaults.
        residual_mixer_kwargs: dict | None = None,
        # first_token chunker: also fold in the PREVIOUS chunk's end token (the
        # token just before each boundary) via a zero-init proj (starts as
        # first-token-only). Causal; gives the high-level each chunk's start+end.
        grab_prev_end: bool = False,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)
        self.reg_trunk_residual_init = bool(reg_trunk_residual_init)
        self.trunk_chunk_rate = bool(trunk_chunk_rate)

        # Inter-stage interface selectors (vault "Expressive Chunk
        # Downsampler-Upsampler"). Defaults = upstream first-token / duplicate
        # so existing configs are byte-identical (no new params created).
        #   down_interface: "first_token" (ChunkLayer) | "scan" (ChunkScanEncoder, A)
        #                 | "register" (RegisterInterface+RegisterTrunk DOWN-only:
        #                   the masked register encoder used as a chunker, emitting
        #                   chunk-rate register tokens (N*m, D) consumed by the
        #                   inner_stage + up_interface, exactly like the scan path)
        #   up_interface:   "duplicate"   (DeChunkLayer) | "spline" (SplineUpsampler, B')
        if down_interface not in ("first_token", "scan", "register"):
            raise ValueError(
                f"down_interface must be first_token|scan|register, got {down_interface}"
            )
        if up_interface not in ("duplicate", "spline", "msublatent"):
            raise ValueError(
                f"up_interface must be duplicate|spline|msublatent, got {up_interface}"
            )
        if up_interface == "msublatent" and down_interface != "register":
            # The m-sublatents dechunker consumes the chunk's m inner-processed
            # register tokens, which only the register down-encoder emits. Scope
            # it to the register-down path (the scan path has no m-register
            # reshape; spline already covers that family).
            raise ValueError(
                "up_interface='msublatent' requires down_interface='register' "
                "(it reads the register down-encoder's m chunk tokens)."
            )
        if router_interface not in ("cossim", "scan", "transformer"):
            raise ValueError(
                f"router_interface must be cossim|scan|transformer, got {router_interface}"
            )
        if chunk_interface not in ("scan_spline", "register"):
            raise ValueError(
                f"chunk_interface must be scan_spline|register, got {chunk_interface}"
            )
        self.down_interface = down_interface
        self.up_interface = up_interface
        self.router_interface = router_interface
        self.chunk_interface = chunk_interface
        self.m_sublatents = int(m_sublatents)

        # Routing / chunk / dechunk operate at the *outer* (input) hidden dim.
        # router_interface "cossim" = upstream pairwise cos-sim(h_{t-1}, h_t);
        # "scan" = ScanRouter (C): p_t = sigmoid(MLP([running-chunk-state, x_t]))
        # decided from the whole chunk so far (Mamba2 packed scan on GPU).
        # Create ONLY the selected router -- an unused cossim RoutingModule
        # leaves its q/k proj params untouched by the loss (DDP unused-params error).
        self.routing_module = (
            RoutingModule(self.input_hidden_dim)
            if router_interface == "cossim"
            else None
        )
        self.scan_router = (
            ScanRouter(
                d_model=self.input_hidden_dim,
                n_layer=router_n_layer,
                use_block=router_use_block,
                has_mlp=router_has_mlp,
                mlp_dim=router_mlp_dim,
            )
            if router_interface == "scan"
            else None
        )
        # PhaseSummaryRouter (PSER): causal-transformer boundary head, RoutingModule
        # -output compatible. target_rate derives from the stage compression ratio
        # so the renewal-init bias is calibrated to the same chunk-rate target.
        self.transformer_router = (
            PhaseSummaryRouter(
                d_model=self.input_hidden_dim,
                depth=pser_depth,
                n_heads=pser_n_heads,
                ff_mult=pser_ff_mult,
                L_min=pser_L_min,
                target_rate=1.0 / max(self.target_compression_ratio, 1.0),
                T_temp=pser_T_temp,
                causal=pser_causal,
                variant=pser_variant,
                rotary_emb_dim=pser_rotary_emb_dim,
                w_l=pser_w_l,
                dwell_sigma=pser_dwell_sigma,
            )
            if router_interface == "transformer"
            else None
        )
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(self.input_hidden_dim)
        # grab_prev_end: first_token chunker also folds in the previous chunk's
        # END token. The combine (concat -> MLP) is built in the proj_in section
        # below -- it REPLACES proj_in (2*input -> trunk dim), preserving both
        # boundary tokens instead of summing them (no info-lossy add).
        self.grab_prev_end = bool(grab_prev_end)
        if self.grab_prev_end and down_interface != "first_token":
            raise ValueError("grab_prev_end requires down_interface='first_token'")

        # The implemented SplineUpsampler (B') reads the scan encoder's per-token
        # running state + m sub-latents, so the encoder must run whenever the
        # DOWN side is scan OR the UP side is spline. The register interface is
        # fused (own trunk) and bypasses scan/spline entirely.
        self._needs_scan_encoder = chunk_interface != "register" and (
            (down_interface == "scan") or (up_interface == "spline")
        )
        self.scan_down = (
            ChunkScanEncoder(
                d_model=self.input_hidden_dim,
                d_out=self.input_hidden_dim,
                m_sublatents=self.m_sublatents,
                n_phase_freqs=n_phase_freqs,
                layernorm=encoder_layernorm,
            )
            if self._needs_scan_encoder
            else None
        )
        self.spline_up = (
            SplineUpsampler(
                d_inner=self.input_hidden_dim,
                d_state=self.input_hidden_dim,
                d_model=self.input_hidden_dim,
                m_sublatents=self.m_sublatents,
                n_phase_freqs=n_phase_freqs,
            )
            if (up_interface == "spline" and chunk_interface != "register")
            else None
        )
        # Bespoke m-sublatents phase-mix + EMA + c_t STE dechunker for the
        # register-DOWN path (replaces the placeholder duplicate). Built at the
        # outer (input) hidden dim: the m register tokens come back at
        # input_hidden_dim after proj_out, and the dechunker output must match
        # the outer residual dim. The optional decoder-end RMSNorm is gated by
        # ``component_end_norm`` (shared with the register encoder-end norm flag).
        self.msublatent_up = (
            MSublatentDeChunker(
                d_inner=self.input_hidden_dim,
                d_model=self.input_hidden_dim,
                m_sublatents=int(reg_m),
                n_phase_freqs=n_phase_freqs,
                component_end_norm=component_end_norm,
            )
            if up_interface == "msublatent"
            else None
        )

        # --- Register masked-transformer interface. ---
        # TWO disjoint uses of the SAME RegisterInterface + RegisterTrunk modules:
        #   (a) FUSED  : chunk_interface == "register" -> ONE masked trunk realizes
        #       down+inner+up; the inner_stage ctx-swap is unused (the trunk IS the
        #       chunk-rate inner network). Bypasses scan/spline. (Unchanged.)
        #   (b) DOWN-ONLY: down_interface == "register" -> the register encoder is
        #       a *chunker*: it emits chunk-rate register tokens (N*m, D) which a
        #       SEPARATE inner_stage + up_interface + router then consume, exactly
        #       like ChunkScanEncoder feeds the scan/spline path. Built at the
        #       INPUT hidden dim; proj_in lifts the chunk tokens to the inner dim.
        # Either use builds the modules. ``component_end_norm`` adds a gated
        # RMSNorm at the encoder OUTPUT (paper "network normalization"); off by
        # default so existing register configs stay byte-identical.
        self.reg_m = int(reg_m)
        self._register_chunker = (
            chunk_interface == "register" or down_interface == "register"
        )
        self.register_interface = (
            RegisterInterface(
                d_model=self.input_hidden_dim,
                m=self.reg_m,
                n_phase_freqs=n_phase_freqs,
                component_end_norm=component_end_norm,
            )
            if self._register_chunker
            else None
        )
        self.register_trunk = (
            RegisterTrunk(
                d_model=self.input_hidden_dim,
                depth=reg_trunk_depth,
                n_heads=reg_trunk_n_heads,
                ff_mult=reg_trunk_ff_mult,
                rotary_emb_dim=reg_trunk_rotary_emb_dim,
            )
            if self._register_chunker
            else None
        )

        # Explicit projections to/from the inner-stage working dim when it
        # differs from the outer dim. Created only when needed.
        if self.grab_prev_end:
            # concat [first; prev_end] (2*input) -> expressive MLP -> output dim
            # (= trunk dim). REPLACES proj_in: the combine IS the bridge, so both
            # boundary tokens are preserved (no info-lossy sum). proj_out stays.
            _h = 2 * self.input_hidden_dim
            self.prev_end_combine = nn.Sequential(
                nn.Linear(2 * self.input_hidden_dim, _h),
                nn.GELU(),
                nn.Linear(_h, _h),
                nn.GELU(),
                nn.Linear(_h, self.output_hidden_dim),
            )
            # Learned "no previous chunk" token: fed as the prev-end half for the
            # first chunk of a sequence (instead of zeros) -> unambiguous BOS-like
            # signal the MLP can key on.
            self.no_prev = nn.Parameter(torch.randn(self.input_hidden_dim) * 0.02)
            self.proj_in = None
            self.proj_out = (
                nn.Linear(self.output_hidden_dim, self.input_hidden_dim)
                if self.input_hidden_dim != self.output_hidden_dim
                else None
            )
        elif self.trunk_chunk_rate and self.register_interface is not None:
            # FIX 1: trunk runs at chunk-rate over the concat'd m sub-latents, so
            # proj_in/out bridge m*input_hidden_dim <-> output_hidden_dim.
            _md = self.register_interface.m * self.input_hidden_dim
            self.proj_in = nn.Linear(_md, self.output_hidden_dim)
            self.proj_out = nn.Linear(self.output_hidden_dim, _md)
        elif self.input_hidden_dim != self.output_hidden_dim:
            self.proj_in = nn.Linear(self.input_hidden_dim, self.output_hidden_dim)
            self.proj_out = nn.Linear(self.output_hidden_dim, self.input_hidden_dim)
        else:
            self.proj_in = None
            self.proj_out = None

        # STE residual mixer at the outer (input) dim. Init to zero so the
        # network starts off transparent.
        self.residual_proj = nn.Linear(self.input_hidden_dim, self.input_hidden_dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        self.residual_proj.weight._no_reinit = True
        # Anneal-able scalar gate on the skip path; default 1.0 = upstream behaviour.
        # A Lightning callback (ChunkerResidualScheduler) can drive this over training
        # steps to suppress the skip path early so the inner trunk must do real work.
        self.residual_scale: float = 1.0
        # FIX 2: swappable residual mixer (out = mix(up, residual)).
        from egomimic.models.hnet.residual_mixer import build_residual_mixer

        self.residual_mixer_kind = str(residual_mixer)
        self.residual_mixer = build_residual_mixer(
            residual_mixer, self.input_hidden_dim, **(residual_mixer_kwargs or {})
        )

    @property
    def inner_working_dim(self) -> int:
        # Chunker's inner trunk runs in the chunked space at output_hidden_dim.
        return self.output_hidden_dim

    @property
    def end_to_end_output_dim(self) -> int:
        # A ChunkerStage DECHUNKS back to the outer (input) dim before
        # returning (proj_out + DeChunkLayer(input_hidden_dim) / spline at
        # input dim), so its true end-to-end output is input_hidden_dim, NOT
        # output_hidden_dim (which is the INNER working dim). Lets a tapering
        # chunker (e.g. register-down 256->512) sit at the ROOT of the chain
        # while HNet.output_hidden_dim still reports d_model (256).
        return self.input_hidden_dim

    @property
    def _router(self) -> nn.Module:
        """The active boundary router module (cossim | scan | transformer).

        Centralizes router selection so the packed forward and both AR-step
        paths dispatch identically. All three return a RoutingModuleOutput and
        expose (forward, step, allocate_inference_cache)."""
        if self.router_interface == "scan":
            return self.scan_router
        if self.router_interface == "transformer":
            return self.transformer_router
        return self.routing_module

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if ctx.packed:
            return self._forward_packed(x, ctx)
        return self._forward_padded(x, ctx)

    def _forward_padded(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if (
            self.down_interface != "first_token"
            or self.up_interface != "duplicate"
            or self.router_interface != "cossim"
        ):
            raise NotImplementedError(
                "scan/spline/scan-router interfaces are implemented for PACKED "
                "training only; padded forward not supported yet."
            )
        # Padded mode (B, T, D_in).
        B, T, _ = x.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        bpred = self.routing_module(x, mask=mask)
        residual = self.residual_scale * self.residual_proj(x.float())

        chunked, _next_cu, next_max, next_mask = self.chunk_layer(
            x,
            bpred.boundary_mask,
            mask=mask,
        )

        # Project into inner-stage working dim if required.
        if self.proj_in is not None:
            chunked = self.proj_in(chunked)

        if self.inner_stage is not None:
            chunked = self.inner_stage(chunked, ctx)

        if self.proj_out is not None:
            chunked = self.proj_out(chunked)

        dechunked = self.dechunk_layer(
            chunked,
            bpred.boundary_mask,
            bpred.boundary_prob,
            mask=mask,
        )

        # STE residual gating (fp32 like upstream).
        out = (dechunked.float() * _ste(bpred.selected_probs) + residual).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
                # F4: ratio loss must exclude positions where boundary_prob
                # was *forced* to 1.0 by the routing module. In padded mode
                # that's only position 0 along each row (see
                # ``RoutingModule._forward_padded``: the leading
                # ``F.pad(..., 1.0)``). Encode this as a (B, T) bool mask of
                # *real* (non-forced) positions.
                "valid_mask_padded": _build_padded_valid_mask(
                    bpred.boundary_mask.shape, x.device
                ),
            }
        )
        return out

    def _forward_packed(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        # Packed mode: x is (T_total, D_in), routed/chunked/dechunked in 2D.
        # All sub-modules' packed paths assume the outer cu_seqlens from ctx;
        # the inner_stage runs with the chunked-space cu_seqlens (we substitute
        # and restore around the call).
        cu = ctx.cu_seqlens

        router = self._router
        bpred = router(x, cu_seqlens=cu)
        residual = self.residual_scale * self.residual_proj(x.float())

        # Test hook: run the scan/spline TF forward under the PREFIX-CAUSAL
        # phase so it is directly comparable to the AR ``step`` (which can only
        # use a causal phase). Off by default => training forward is unchanged.
        cp = getattr(self, "_ar_consistency_causal_phase", False)

        # --- REGISTER masked-transformer interface (fused down+inner+up). ------
        if self.chunk_interface == "register":
            return self._forward_packed_register(x, ctx, cu, bpred, residual, cp)

        # --- REGISTER DOWN-ONLY encoder (chunker) + separate up/inner/router. --
        if self.down_interface == "register":
            return self._forward_packed_register_down(x, ctx, cu, bpred, residual, cp)

        # --- DOWN: produce the inner-stage input + (optionally) scan features. --
        h_tok = sub_latents = None
        if self._needs_scan_encoder:
            h_tok, sub_latents, chunk_summary, scan_cu, scan_max = self.scan_down(
                x, bpred.boundary_mask, cu, causal_phase=cp
            )
        if self.down_interface == "scan":
            chunked, next_cu, next_max = chunk_summary, scan_cu, scan_max
        else:  # first_token (upstream ChunkLayer)
            chunked, next_cu, next_max, _ = self.chunk_layer(
                x, bpred.boundary_mask, cu_seqlens=cu
            )
            if self.grab_prev_end:
                chunked = self._combine_prev_end_tf(chunked, x, bpred.boundary_mask, cu)

        if self.proj_in is not None:
            chunked = self.proj_in(chunked)

        if self.inner_stage is not None:
            saved_cu, saved_max = ctx.cu_seqlens, ctx.max_seqlen
            ctx.cu_seqlens, ctx.max_seqlen = next_cu, next_max
            try:
                chunked = self.inner_stage(chunked, ctx)
            finally:
                ctx.cu_seqlens, ctx.max_seqlen = saved_cu, saved_max

        if self.proj_out is not None:
            chunked = self.proj_out(chunked)
        inner_out = chunked  # (N, D_in)

        # --- UP: spline (B') or duplicate-broadcast (upstream DeChunkLayer). ----
        if self.up_interface == "spline":
            # m inner-sub-latents = encoder's m raw sub-latents with the LAST
            # slot replaced by the inner-trunk-processed summary (keeps the
            # inner stage at 1x chunk rate, dims identical to taper baseline).
            inner_sub = sub_latents.clone()
            inner_sub[:, -1] = inner_out
            up = self.spline_up(
                inner_sub,  # (N, m, D_in)
                h_tok,  # (T, D_in)
                x,  # (T, D_in) outer query
                bpred.boundary_mask,
                bpred.boundary_prob,
                cu,  # TOKEN-space cu_seqlens
                causal_phase=cp,
            )
            # SplineUpsampler already applies the confidence gate c_t internally
            # (preserves the (2b_t-1) p_t learning path), so NO outer STE multiply.
            out = (up.float() + residual).to(x.dtype)
        else:
            # DeChunkLayer in packed mode takes the **chunked-space** cu_seqlens.
            dechunked = self.dechunk_layer(
                inner_out,
                bpred.boundary_mask,
                bpred.boundary_prob,
                cu_seqlens=next_cu,
            )
            # STE residual gating (fp32 like upstream). Shapes in packed mode:
            #   dechunked: (T_total, D_in)
            #   selected_probs: (T_total, 1)  → STE returns ones_like → broadcasts.
            gated = dechunked.float() * _ste(bpred.selected_probs)
            out = self.residual_mixer(gated, residual).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
                # F4: in packed mode boundary_prob is forced to 1.0 at every
                # subseq start (``boundary_prob[cu_seqlens[:-1]] = 1.0`` in
                # ``RoutingModule._forward_packed``). Pass cu_seqlens through
                # so the ratio loss can exclude those synthetic positions.
                "cu_seqlens": cu,
            }
        )
        return out

    def _forward_packed_register(self, x, ctx, cu, bpred, residual, cp):
        """TF packed forward for the FUSED register masked-transformer interface.

        router -> RegisterInterface(stream + (L,L) mask) -> RegisterTrunk ->
        gather member outputs at ``member_idx`` -> confidence gate -> residual.
        ONE masked transformer realizes down (regs<-own members) + inner
        (regs<-earlier regs) + up (members<-prev regs + own causal prefix); the
        mask enforces causality + subseq isolation, so the whole packed stream
        runs as a single (L,L) attention. ``cp`` selects the prefix-causal member
        phase so this forward is directly comparable to the AR ``step`` (GATE-1).
        """
        stream = self.register_interface(x, bpred.boundary_mask, cu, causal_phase=cp)
        trunk_out = self.register_trunk(stream)  # (L, D_in)
        up = trunk_out[stream.member_idx]  # (T, D_in) member outputs, token order
        # Confidence gate (mirrors SplineUpsampler / the plan): keeps the p_t
        # learning path. c = p where boundary fires else 1-p.
        p = bpred.boundary_prob[..., -1]
        c = torch.where(bpred.boundary_mask, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)
        out = (up.float() * c.unsqueeze(-1) + residual).to(x.dtype)
        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
                "cu_seqlens": cu,
            }
        )
        return out

    def _forward_packed_register_down(self, x, ctx, cu, bpred, residual, cp):
        """TF packed forward for the DOWN-ONLY register encoder (chunker).

        router -> RegisterInterface+RegisterTrunk.encode_down -> chunk-rate
        register tokens (N*m, D) -> proj_in -> inner_stage (at REGISTER-rate
        cu_seqlens) -> proj_out -> up_interface (placeholder dechunker) ->
        STE residual gate. The register encoder REPLACES scan_down as the
        downsampler; the trunk's own member ("up") outputs are discarded.

        Placeholder up_interface for this MVP is ``duplicate`` (DeChunkLayer):
        the bespoke m-sublatent + EMA dechunker is the NEXT build step. Because
        DeChunkLayer maps exactly ONE chunk token per chunk back to tokens, we
        reduce the m register tokens of each chunk to a single CHUNK SUMMARY
        token (the LAST register, register m-1 — the scan-path analogue is
        ``sub_latents[:, -1]``) before dechunking. The inner_stage still runs
        over all N*m register tokens, so the m>1 capacity is exercised; only the
        final summary-slice collapses to chunk rate for the placeholder up. (A
        real dechunker would consume all m register tokens — see report note.)
        """
        reg_tokens, reg_cu, reg_max, stream = self.register_trunk.encode_down(
            self.register_interface, x, bpred.boundary_mask, cu, causal_phase=cp
        )  # reg_tokens (N*m, D_in); reg_cu/reg_max register-rate.

        m = self.register_interface.m
        chunked = reg_tokens  # (N*m, D_in)
        # FIX 1: run the inner trunk at CHUNK rate -- concat each chunk's m
        # sub-latents into one token (N*m, D) -> (N, m*D), so the trunk attends
        # over N chunk-tokens not N*m. reg_cu // m maps register-rate -> chunk-rate.
        if self.trunk_chunk_rate:
            chunked = chunked.reshape(-1, m * chunked.shape[-1])  # (N, m*D)
            inner_cu, inner_max = reg_cu // m, reg_max // m
        else:
            inner_cu, inner_max = reg_cu, reg_max

        if self.proj_in is not None:
            chunked = self.proj_in(chunked)

        if self.inner_stage is not None:
            saved_cu, saved_max = ctx.cu_seqlens, ctx.max_seqlen
            ctx.cu_seqlens, ctx.max_seqlen = inner_cu, inner_max
            try:
                chunked = self.inner_stage(chunked, ctx)
            finally:
                ctx.cu_seqlens, ctx.max_seqlen = saved_cu, saved_max

        if self.proj_out is not None:
            chunked = self.proj_out(chunked)
        if self.trunk_chunk_rate:
            chunked = chunked.reshape(
                -1, chunked.shape[-1] // m
            )  # (N, m*D) -> (N*m, D)
        inner_out = chunked  # (N*m, D_in)
        if self.up_interface == "spline":
            raise NotImplementedError(
                "down_interface='register' does not wire up_interface='spline' "
                "(spline reads the scan encoder's sub_latents/h_tok, absent on "
                "the register path). Use up_interface='msublatent' (the bespoke "
                "m-sublatent dechunker) or 'duplicate' (placeholder)."
            )
        if self.up_interface == "msublatent":
            # --- BESPOKE m-sublatents dechunker: per fine token, phase-mix the
            #     chunk's m inner-processed register tokens (NO collapse to a
            #     single summary), then the SAME EMA + c_t STE the duplicate
            #     uses. reg_tokens are chunk-major (chunk c => slots
            #     [c*m .. c*m+m)), so reshape (N*m, D_in) -> (N, m, D_in). The
            #     dechunker applies the c_t gate internally, so NO outer STE. ---
            inner_sub = inner_out.view(-1, m, inner_out.shape[-1])  # (N, m, D_in)
            # FIX #1: the dechunker now runs the chunk-rate EMA over the m
            # sub-latents BEFORE the phase m-mix (mirroring the canonical
            # DeChunkLayer._forward_packed which EMAs at chunk-rate). That EMA
            # needs the CHUNK-space cu_seqlens (size B+1, in chunk units). The
            # register encoder produces register-rate cu (N*m); chunk-rate =
            # reg_cu // m (m registers per chunk). Token-space ``cu`` is still
            # passed for the per-token phase machinery.
            chunk_cu = reg_cu // m  # register-rate -> chunk-rate cu_seqlens
            up = self.msublatent_up(
                inner_sub,
                bpred.boundary_mask,
                bpred.boundary_prob,
                cu,  # TOKEN-space cu_seqlens (phase machinery)
                chunk_cu_seqlens=chunk_cu,  # CHUNK-space cu_seqlens (chunk-rate EMA)
                causal_phase=cp,
            )
            out = self.residual_mixer(up.float(), residual).to(x.dtype)
            ctx.register_aux(
                {
                    "bpred": bpred,
                    "target_ratio": self.target_compression_ratio,
                    "weight": self.ratio_loss_weight,
                    "cu_seqlens": cu,
                }
            )
            return out
        # --- placeholder UP: collapse m registers -> 1 chunk summary, DeChunk. --
        # reg_tokens are chunk-major (chunk c => slots [c*m .. c*m+m)); the last
        # register of each chunk is the chunk summary. chunk-rate cu = reg_cu / m.
        summary = inner_out.view(-1, m, inner_out.shape[-1])[:, -1]  # (N, D_in)
        chunk_cu = reg_cu // m  # register-rate -> chunk-rate (m regs per chunk)
        dechunked = self.dechunk_layer(
            summary,
            bpred.boundary_mask,
            bpred.boundary_prob,
            cu_seqlens=chunk_cu,
        )
        out = (dechunked.float() * _ste(bpred.selected_probs) + residual).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
                "cu_seqlens": cu,
            }
        )
        return out

    def step(self, x: torch.Tensor, ctx: HNetContext, state: ChunkerStageState):
        # The register fused interface owns its own chunk-incremental AR path.
        if self.chunk_interface == "register":
            return self._step_register(x, ctx, state)
        # DOWN-ONLY register chunker: the chunk-incremental AR step belongs with
        # the bespoke dechunker (NEXT build step) — its shift-by-one timing and
        # the register-rep emission at chunk completion are co-designed with the
        # real up-interface. This foundational build wires the TF (packed)
        # forward only; closed-loop AR rollout is deferred. Fail loudly rather
        # than ship a subtly-wrong placeholder AR path.
        if self.down_interface == "register":
            return self._step_register_down(x, ctx, state)
        # The scan/spline AR orchestration is required iff the DOWN or UP
        # interface is scan/spline. The router choice (cossim|scan|transformer)
        # is orthogonal — any router can drive either down/up path.
        is_scan = (
            self.down_interface == "scan"
            or self.up_interface == "spline"
            or self.router_interface == "scan"
        )
        if is_scan:
            return self._step_scan(x, ctx, state)
        return self._step_first_token(x, ctx, state)

    def _combine_prev_end_tf(self, chunked, x, boundary_mask, cu):
        """first_token + prev-chunk-end (TF): CONCAT the chunk's first token with
        the token just before each boundary (prev chunk's END), zeroed at subseq
        starts, then MLP -> trunk dim. Concat (not sum) preserves both boundary
        tokens. Causal -- pos-1 is always in the past."""
        pos = boundary_mask.nonzero(as_tuple=True)[0]
        prev_end = x[(pos - 1).clamp(min=0)]
        is_start = torch.isin(pos, cu[:-1])
        # first chunk of each sequence has no prev -> this stage's learned no_prev
        # token (per-emb: each embodiment's ChunkerStage owns its own no_prev).
        prev_end = torch.where(
            is_start.unsqueeze(-1), self.no_prev.to(prev_end.dtype), prev_end
        )
        cat = torch.cat([chunked, prev_end], dim=-1).float()
        return self.prev_end_combine(cat).to(chunked.dtype)

    def _step_first_token(self, x, ctx, state: ChunkerStageState):
        bpred = self._router.step(x, state.routing_state)
        residual = self.residual_scale * self.residual_proj(x.float())

        inner_in = self.chunk_layer.step(x, bpred.boundary_mask)
        if self.grab_prev_end:
            if inner_in.shape[0] > 0:
                prev_end = (
                    state.prev_x[bpred.boundary_mask]
                    if state.prev_x is not None
                    else self.no_prev.to(inner_in.dtype).expand_as(inner_in)
                )
                cat = torch.cat([inner_in, prev_end], dim=-1).float()
                inner_in = self.prev_end_combine(cat).to(x.dtype)  # concat -> trunk dim
            state.prev_x = x  # update for next step (every step, boundary or not)
        if inner_in.shape[0] > 0:
            if self.proj_in is not None:
                inner_in = self.proj_in(inner_in)
            if self.inner_stage is not None:
                # Slice nested state to active rows, recurse, then scatter back.
                sub_state = _slice_state(state.inner_state, bpred.boundary_mask)
                inner_out = self.inner_stage.step(inner_in, ctx, sub_state)
                _scatter_state(state.inner_state, sub_state, bpred.boundary_mask)
            else:
                inner_out = inner_in
            if self.proj_out is not None:
                inner_out = self.proj_out(inner_out)
        else:
            inner_out = inner_in  # empty tensor at outer dim already

        dechunked = self.dechunk_layer.step(
            inner_out,
            bpred.boundary_mask,
            bpred.boundary_prob,
            state.dechunk_state,
        )

        # selected_probs at step has shape (B, 1); STE expects something
        # broadcastable to dechunked (B, 1, D).
        gated = dechunked.float() * _ste(bpred.selected_probs.unsqueeze(1))
        out = self.residual_mixer(gated, residual).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out

    def _step_scan(self, x, ctx, state: ChunkerStageState):
        """AR step for the scan-down / spline-up / scan-router interface.

        Shift-by-one timing (mirrors the TF ``_forward_packed``):
          * The router decides ``boundary_t``.
          * The scan encoder consumes x_t -> running h_t; if a boundary fires
            and a chunk was open, that chunk just COMPLETED -> its m sub-latents
            are finalized. Its summary is projected + run through the inner
            stage (one inner step), and the result is cached as the basis for
            the UPCOMING chunk's tokens (``spline.update_prev``).
          * The spline upsampler decodes x_t using the cached PREVIOUS chunk's
            basis + own h_t + the prefix-causal phase; confidence-gated.
          * out = up_t + residual  (spline gates internally; no outer STE).

        ``x`` may be (B, 1, D) or (B, D); returns (B, 1, D).
        """
        x_t = x.squeeze(1) if x.dim() == 3 else x  # (B, D_in)
        B = x_t.shape[0]
        residual = self.residual_scale * self.residual_proj(x_t.float())  # (B, D_in)

        # --- router ---
        bpred = self._router.step(x_t, state.routing_state)
        boundary_t = bpred.boundary_mask  # (B,)
        boundary_prob_t = bpred.boundary_prob  # (B, 2)

        # --- scan encoder: running h_t + finalize completed chunk ---
        h_t, finished_sub, _finished_summary, finished_mask = self.scan_down.step(
            x_t, state.scan_enc_state, boundary_t
        )

        # --- inner stage on the just-completed chunk's summary (shift-by-one) ---
        if finished_mask.any():
            # The inner stage consumes the chunk SUMMARY (= sub-latent m-1).
            # Shape (k, 1, D_in): the inner Isotropic.step expects a seq dim.
            inner_in = finished_sub[finished_mask][:, -1].unsqueeze(1)  # (k, 1, D_in)
            if self.proj_in is not None:
                inner_in = self.proj_in(inner_in)
            if self.inner_stage is not None:
                sub_state = _slice_state(state.inner_state, finished_mask)
                inner_out = self.inner_stage.step(inner_in, ctx, sub_state)
                _scatter_state(state.inner_state, sub_state, finished_mask)
            else:
                inner_out = inner_in
            if self.proj_out is not None:
                inner_out = self.proj_out(inner_out)
            inner_out = inner_out.squeeze(1)  # (k, D_in)
            # Build the completed chunk's inner-sub basis (raw sub-latents with
            # slot m-1 replaced by the inner-trunk-processed summary), then
            # cache it as the upcoming chunk's basis. Scatter k rows back to B.
            finished_inner_sub = finished_sub.clone()  # (B, m, D_in)
            full_inner_out = finished_sub.new_zeros(B, finished_sub.shape[-1])
            full_inner_out[finished_mask] = inner_out.to(full_inner_out.dtype)
            finished_inner_sub[:, -1] = torch.where(
                finished_mask.unsqueeze(-1), full_inner_out, finished_inner_sub[:, -1]
            )
            self.spline_up.update_prev(
                state.spline_state, finished_inner_sub, finished_mask
            )

        # --- spline upsampler: decode x_t with the cached prev-chunk basis ---
        up = self.spline_up.step(
            x_t, h_t, boundary_t, boundary_prob_t, state.spline_state
        )  # (B, D_in)
        out = (up.float() + residual).to(x_t.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out.unsqueeze(1)

    def _step_register(self, x, ctx, state: ChunkerStageState):
        """AR step for the FUSED register masked-transformer interface.

        Shift-by-one timing (mirrors ``_step_scan`` / the TF mask):
          * router decides ``boundary_t``;
          * if it fires AND a chunk is open (its members already emitted), that
            chunk just COMPLETED -> finalize its m registers JOINTLY (they attend
            their members + all earlier registers) and append their KV to the
            register cache; reset the open-chunk member cache;
          * emit member x_t with the prefix-causal phase of its running offset
            (== open-chunk member slot), attending the PREVIOUS chunk's m
            registers (last m finalized slots) + its own causal prefix;
          * confidence gate (same c_t as TF/spline) + residual.

        ``x`` may be (B, 1, D) or (B, D); returns (B, 1, D)."""
        x_t = x.squeeze(1) if x.dim() == 3 else x  # (B, D_in)
        residual = self.residual_scale * self.residual_proj(x_t.float())  # (B, D_in)

        bpred = self._router.step(x_t, state.routing_state)
        boundary_t = bpred.boundary_mask  # (B,)
        boundary_prob_t = bpred.boundary_prob  # (B, 2)

        rstate = state.reg_trunk_state
        # A chunk COMPLETES when a boundary fires and the open chunk already has
        # >=1 member (so the episode's first forced boundary does NOT finalize an
        # empty chunk). Finalize those rows' registers before the new chunk opens.
        finished_mask = boundary_t & (rstate.mem_fill > 0)
        self.register_trunk.finalize_chunk(
            rstate, finished_mask, self.register_interface
        )

        # member offset within the (now-possibly-reset) open chunk == mem_fill.
        offset = rstate.mem_fill  # (B,)
        mem_emb = self.register_interface.member_embed_from_offset(x_t, offset)
        trunk_out = self.register_trunk.step_member(
            mem_emb, rstate, self.register_interface.m
        )  # (B, D_in)
        rstate.offset = rstate.mem_fill  # running offset == new fill (post-incr)
        rstate.has_seen.fill_(True)

        p = boundary_prob_t[..., -1]
        c = torch.where(boundary_t, p, 1.0 - p).clamp(1e-4, 1 - 1e-4)
        out = (trunk_out.float() * c.unsqueeze(-1) + residual).to(x_t.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out.unsqueeze(1)

    def _step_register_down(self, x, ctx, state: ChunkerStageState):
        """AR step for down_interface='register' + up_interface='msublatent'
        (DECHUNK-AR). Shift-by-one timing mirrors _step_scan / _step_register:
          * router decides boundary_t;
          * if it fires AND a chunk is open, that chunk COMPLETED -> finalize its
            m registers (finalize_chunk returns the post-trunk reps), push them
            through the inner stage (chunk-rate concat if trunk_chunk_rate, else
            the m registers one-by-one at register rate), chunk-rate-EMA + cache
            as the UPCOMING chunk's msublatent basis (update_prev);
          * emit x_t into the (reset) open chunk (step_member; the member's own
            trunk_out is discarded -- down-only);
          * msublatent dechunker decodes x_t against the cached PREVIOUS chunk's
            basis + the prefix-causal phase; out = residual_mixer(up, residual).
        x may be (B,1,D) or (B,D); returns (B,1,D)."""
        x_t = x.squeeze(1) if x.dim() == 3 else x  # (B, D_in)
        B = x_t.shape[0]
        residual = self.residual_scale * self.residual_proj(x_t.float())  # (B, D_in)
        m = self.register_interface.m
        rstate = state.reg_trunk_state
        mstate = state.msublatent_state

        bpred = self._router.step(x_t, state.routing_state)
        boundary_t = bpred.boundary_mask  # (B,)
        boundary_prob_t = bpred.boundary_prob  # (B, 2)

        # --- finalize the just-completed chunk (returns its post-trunk m reps) ---
        finished_mask = boundary_t & (rstate.mem_fill > 0)
        reg_reps = self.register_trunk.finalize_chunk(
            rstate, finished_mask, self.register_interface
        )  # (B, m, D_in)

        if bool(finished_mask.any()):
            reg_sub = reg_reps  # (B, m, D_in)
            D_in = reg_sub.shape[-1]
            if self.trunk_chunk_rate:
                # Fix-1: concat the m sub-latents -> one chunk token -> 1 inner step.
                inner_out = self._inner_step_rows(
                    reg_sub.reshape(B, m * D_in), ctx, state, finished_mask
                )  # (B, m*D_in)
                inner_sub = inner_out.reshape(B, m, D_in)
            else:
                # pre-Fix-1: inner stage at REGISTER rate -- the m registers run
                # one-by-one (causal over the register stream / inner KV cache).
                inner_sub = torch.stack(
                    [
                        self._inner_step_rows(reg_sub[:, j], ctx, state, finished_mask)
                        for j in range(m)
                    ],
                    dim=1,
                )  # (B, m, D_in)
            inner_sub = torch.where(
                finished_mask.view(B, 1, 1), inner_sub.to(reg_sub.dtype), reg_sub
            )
            # chunk-rate EMA + promote as the upcoming chunk's msublatent basis.
            self.msublatent_up.update_prev(
                mstate, inner_sub, boundary_prob_t, finished_mask
            )

        # --- DOWN: emit x_t into the (reset) open chunk; trunk_out discarded. ---
        offset = rstate.mem_fill  # (B,)
        mem_emb = self.register_interface.member_embed_from_offset(x_t, offset)
        _ = self.register_trunk.step_member(mem_emb, rstate, m)
        rstate.offset = rstate.mem_fill
        rstate.has_seen.fill_(True)

        # --- UP: msublatent dechunk against the cached prev-chunk basis. ---
        up = self.msublatent_up.step(boundary_t, boundary_prob_t, mstate)  # (B, D_in)
        out = self.residual_mixer(up.float(), residual).to(x_t.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out.unsqueeze(1)

    def _inner_step_rows(self, inner_in, ctx, state, finished_mask):
        """``proj_in -> inner_stage.step -> proj_out`` on the finished rows of a
        ``(B, W)`` input; returns ``(B, W_out)``. Non-finished rows are unused by
        the caller (overwritten with the raw register reps)."""
        out = inner_in
        if self.proj_in is not None:
            out = self.proj_in(out)
        if self.inner_stage is not None:
            sub_state = _slice_state(state.inner_state, finished_mask)
            sub_out = self.inner_stage.step(
                out[finished_mask].unsqueeze(1), ctx, sub_state
            ).squeeze(1)
            _scatter_state(state.inner_state, sub_state, finished_mask)
            full = out.clone()
            full[finished_mask] = sub_out.to(full.dtype)
            out = full
        if self.proj_out is not None:
            out = self.proj_out(out)
        return out

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        # Router state: cossim RoutingModule, ScanRouter, or PhaseSummaryRouter.
        if self.router_interface == "scan":
            routing_state = self.scan_router.allocate_inference_cache(
                batch_size, device, dtype
            )
        elif self.router_interface == "transformer":
            # PSER needs T_max for its per-layer KV caches (== max subseq length).
            routing_state = self.transformer_router.allocate_inference_cache(
                batch_size, device, dtype, T_max=max_seqlen
            )
        else:
            routing_state = self.routing_module.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            )
        # Scan encoder + spline upsampler states (only on the scan/spline path).
        scan_enc_state = None
        spline_state = None
        if self._needs_scan_encoder:
            # Buffer capacity must cover the longest possible chunk = the full
            # episode length (a single-chunk episode at init). max_seqlen is
            # exactly that bound.
            scan_enc_state = self.scan_down.allocate_inference_cache(
                batch_size, device, dtype, max_chunk_len=max_seqlen
            )
        if self.up_interface == "spline":
            spline_state = self.spline_up.allocate_inference_cache(
                batch_size, self.input_hidden_dim, device, dtype
            )
        # DeChunk state only used by the duplicate path; allocate harmlessly.
        dechunk_state = self.dechunk_layer.allocate_inference_cache(
            batch_size, max_seqlen, device, dtype
        )
        # Register trunk state (only on the fused register interface).
        reg_trunk_state = None
        if self.chunk_interface == "register":
            # cap_mem: longest open chunk == full episode (single-chunk init).
            # cap_reg: at most one chunk per token across the episode, each
            # contributing m registers -> max_seqlen * m (loose but exact bound).
            reg_trunk_state = self.register_trunk.allocate(
                batch_size,
                cap_reg=int(max_seqlen) * self.reg_m,
                cap_mem=int(max_seqlen),
                d_model=self.input_hidden_dim,
                device=device,
                dtype=dtype,
            )
        return ChunkerStageState(
            routing_state=routing_state,
            inner_state=(
                self.inner_stage._allocate(batch_size, max_seqlen, dtype, device)
                if (self.inner_stage is not None and self.chunk_interface != "register")
                else None
            ),
            dechunk_state=dechunk_state,
            scan_enc_state=scan_enc_state,
            spline_state=spline_state,
            reg_trunk_state=reg_trunk_state,
        )

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        # The routing q/k are identity-init (marked _no_reinit) and residual_proj
        # is zero-init (also _no_reinit). proj_in / proj_out are dim-bridge
        # Linears between outer and inner working dim; use the unscaled
        # ``initializer_range`` (they're not in a residual stream context).
        #
        # FIX #7: the RegisterTrunk (register-fused OR register-down paths) DOES
        # add 2*depth residual-stream blocks (attn + MLP per block) — the prior
        # comment claiming "ChunkerStage adds no residual-stream MLP blocks" was
        # stale. Its out_proj/fc2 get residual-aware init when
        # ``reg_trunk_residual_init`` is ON (GATED for BC: default OFF keeps the
        # existing fused register config's init byte-unchanged; param count is
        # identical either way). Counted into ``n_residuals`` so a downstream
        # inner_stage sees the cumulative depth.
        for m in (self.proj_in, self.proj_out):
            if m is not None and not getattr(m.weight, "_no_reinit", False):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Only fold the trunk's residual count + apply its scaled init when the
        # gate is ON. With it OFF, ``n_residuals`` is left at ``parent_residuals``
        # exactly as before, so the inner_stage's init scaling is BYTE-UNCHANGED
        # (BC) and the trunk keeps its module-default init.
        n_residuals = parent_residuals
        if self.register_trunk is not None and self.reg_trunk_residual_init:
            # Faithful to H-Net's encoder rule (parent + encoder.height +
            # decoder.height): the trunk is the DOWN (encoder) residual depth and
            # the register-path UP-interface (dechunker) is the DECODER depth.
            # The bare softmax ``MSublatentDeChunker`` exposes ``_residual_adds()``
            # == 0 (no residual-block stack), so the value here is unchanged for
            # this arch — but a future residual-block dechunker would correctly
            # deepen ``n_residuals``. ``getattr`` so a None / no-method up-iface
            # (e.g. duplicate DeChunkLayer) contributes 0.
            up_iface = getattr(self, "msublatent_up", None)
            dechunker_residual_adds = (
                up_iface._residual_adds()
                if up_iface is not None and hasattr(up_iface, "_residual_adds")
                else 0
            )
            n_residuals = (
                parent_residuals
                + self.register_trunk.n_residual_adds()
                + dechunker_residual_adds
            )
            self.register_trunk._init_weights(initializer_range, n_residuals)
        if self.inner_stage is not None:
            return self.inner_stage._init_weights(initializer_range, n_residuals)
        return n_residuals


# --------------------------------------------------------------------------- #
# ComputeStage
# --------------------------------------------------------------------------- #


class ComputeStage(_BaseStage):
    """A single Isotropic main_network. Terminal by default; an inner_stage
    may be chained after the main_network for further nesting."""

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        main_network: Dict[str, Any],
        cond_key: str = "fused_cond",
        d_cond: int = 0,
        causal: bool = True,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if input_hidden_dim != output_hidden_dim:
            raise ValueError(
                f"ComputeStage requires input_hidden_dim==output_hidden_dim "
                f"(got {input_hidden_dim} vs {output_hidden_dim})."
            )
        if int(main_network["d_model"]) != input_hidden_dim:
            raise ValueError(
                f"main_network.d_model ({main_network['d_model']}) must equal "
                f"stage input_hidden_dim ({input_hidden_dim})."
            )
        self.causal = causal
        self.main_network = build_isotropic(main_network, d_cond=d_cond, causal=causal)

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        cond = self._get_cond(ctx)
        if ctx.packed:
            x = self.main_network(
                x,
                cu_seqlens=ctx.cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                cond=cond,
            )
        else:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            x = self.main_network(x, mask=mask, cond=cond)
        if self.inner_stage is not None:
            x = self.inner_stage(x, ctx)
        return x

    def step(self, x: torch.Tensor, ctx: HNetContext, state: ComputeStageState):
        cond = self._get_cond(ctx)
        x = self.main_network.step(x, state.main_state, cond=cond)
        if self.inner_stage is not None:
            x = self.inner_stage.step(x, ctx, state.inner_state)
        return x

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return ComputeStageState(
            main_state=self.main_network.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            ),
            inner_state=(
                self.inner_stage._allocate(batch_size, max_seqlen, dtype, device)
                if self.inner_stage is not None
                else None
            ),
        )

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        n_residuals = parent_residuals + self.main_network.height
        _init_isotropic_linears(self.main_network, initializer_range, n_residuals)
        if self.inner_stage is not None:
            n_residuals = self.inner_stage._init_weights(initializer_range, n_residuals)
        return n_residuals


# --------------------------------------------------------------------------- #
# Shared init helper — mirrors upstream's named-module iteration.
# --------------------------------------------------------------------------- #


def _init_isotropic_linears(
    isotropic, initializer_range: float, n_residuals: int
) -> None:
    """Initialize Linear weights inside an Isotropic stack.

    Out-projection (attention ``out_proj``) and FFN second linear (``fc2``)
    contribute to the residual stream and get scaled by
    ``initializer_range / sqrt(n_residuals)`` so the residual norm stays
    bounded with depth. All other Linears use ``initializer_range``.
    Modules flagged ``_no_reinit`` (e.g. routing q/k = identity, residual
    projections = zero-init, AdaLN = zero-init) are skipped.
    """
    scaled_std = initializer_range / max(n_residuals, 1) ** 0.5
    for name, m in isotropic.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if getattr(m.weight, "_no_reinit", False):
            continue
        if "out_proj" in name or "fc2" in name:
            nn.init.normal_(m.weight, mean=0.0, std=scaled_std)
        else:
            nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
