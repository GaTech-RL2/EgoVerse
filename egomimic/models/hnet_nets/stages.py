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

from egomimic.models.hnet_nets.blocks import (
    IsotropicInferenceParams,
    KVCache,
    MambaCache,
)
from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.isotropic_builder import build_isotropic
from egomimic.models.hnet_nets.routing import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleState,
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
    routing_state: Optional[RoutingModuleState] = None
    inner_state: Optional[Any] = None
    dechunk_state: Optional[DeChunkState] = None


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
        rs = state.routing_state
        if rs is not None:
            rs = RoutingModuleState(
                has_seen_tokens=rs.has_seen_tokens[mask].clone(),
                last_hidden_state=rs.last_hidden_state[mask].clone(),
            )
        ds = state.dechunk_state
        if ds is not None:
            ds = DeChunkState(last_value=ds.last_value[mask].clone())
        return ChunkerStageState(
            routing_state=rs,
            inner_state=_slice_state(state.inner_state, mask),
            dechunk_state=ds,
        )
    if isinstance(state, ComputeStageState):
        return ComputeStageState(
            main_state=_slice_iso_params(state.main_state, mask),
            inner_state=_slice_state(state.inner_state, mask),
        )
    raise TypeError(f"Unknown stage state type: {type(state)}")


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
        if state.routing_state is not None and sub.routing_state is not None:
            state.routing_state.has_seen_tokens[mask] = (
                sub.routing_state.has_seen_tokens
            )
            state.routing_state.last_hidden_state[mask] = (
                sub.routing_state.last_hidden_state.to(
                    state.routing_state.last_hidden_state.dtype
                )
            )
        _scatter_state(state.inner_state, sub.inner_state, mask)
        if state.dechunk_state is not None and sub.dechunk_state is not None:
            state.dechunk_state.last_value[mask] = sub.dechunk_state.last_value.to(
                state.dechunk_state.last_value.dtype
            )
        return
    if isinstance(state, ComputeStageState):
        _scatter_iso_params(state.main_state, sub.main_state, mask)
        _scatter_state(state.inner_state, sub.inner_state, mask)
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
        """Hidden dim at which this stage's ``inner_stage`` is expected to
        operate. Default: ``input_hidden_dim`` (EncoderDecoderStage /
        ComputeStage pass through at the outer dim). Overridden by
        ``ChunkerStage`` to return ``output_hidden_dim`` (chunked space).
        Polymorphic wrapper stages (e.g. ``PerEmbodimentStage``) override
        this to delegate to the wrapped sub-stage so chain-wiring works
        without isinstance checks.
        """
        return self.input_hidden_dim

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
        from egomimic.models.hnet_nets.hnet import apply_optimization_params

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
        cond = self._get_cond(ctx)
        if ctx.packed:
            # Packed mode: x is (T_total, D); pass cu_seqlens/max_seqlen straight
            # through. Both encoder and decoder operate at the outer pack.
            x = self.encoder(
                x, cu_seqlens=ctx.cu_seqlens, max_seqlen=ctx.max_seqlen, cond=cond
            )
            residual = x
            if self.inner_stage is not None:
                x = self.inner_stage(x, ctx)
            x = x + residual
            x = self.decoder(
                x, cu_seqlens=ctx.cu_seqlens, max_seqlen=ctx.max_seqlen, cond=cond
            )
        else:
            # Padded mode: build an all-ones mask along T (causal already wired
            # into the Isotropic). This matches the algo's BOS-prefixed teacher-
            # forced input.
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            x = self.encoder(x, mask=mask, cond=cond)
            residual = x
            if self.inner_stage is not None:
                x = self.inner_stage(x, ctx)
            x = x + residual
            x = self.decoder(x, mask=mask, cond=cond)
        return x

    def step(self, x: torch.Tensor, ctx: HNetContext, state: EncoderDecoderStageState):
        cond = self._get_cond(ctx)
        x = self.encoder.step(x, state.encoder_state, cond=cond)
        residual = x
        if self.inner_stage is not None:
            x = self.inner_stage.step(x, ctx, state.inner_state)
        x = x + residual
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
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        self.target_compression_ratio = float(target_compression_ratio)
        self.ratio_loss_weight = float(ratio_loss_weight)

        # Routing / chunk / dechunk operate at the *outer* (input) hidden dim.
        # inner_working_dim override below so the chain wiring sees the
        # chunked-space dim handoff.

        self.routing_module = RoutingModule(self.input_hidden_dim)
        self.chunk_layer = ChunkLayer()
        self.dechunk_layer = DeChunkLayer(self.input_hidden_dim)

        # Explicit projections to/from the inner-stage working dim when it
        # differs from the outer dim. Created only when needed.
        if self.input_hidden_dim != self.output_hidden_dim:
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

    @property
    def inner_working_dim(self) -> int:
        """Inner stage runs in chunked space at ``output_hidden_dim`` (after
        ``proj_in``)."""
        return self.output_hidden_dim

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if ctx.packed:
            return self._forward_packed(x, ctx)
        return self._forward_padded(x, ctx)

    def _forward_padded(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
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
            }
        )
        return out

    def _forward_packed(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        # Packed mode: x is (T_total, D_in), routed/chunked/dechunked in 2D.
        # All sub-modules' packed paths assume the outer cu_seqlens from ctx;
        # the inner_stage runs with the chunked-space cu_seqlens (we substitute
        # and restore around the call).
        cu = ctx.cu_seqlens

        bpred = self.routing_module(x, cu_seqlens=cu)
        residual = self.residual_scale * self.residual_proj(x.float())

        chunked, next_cu, next_max, _ = self.chunk_layer(
            x,
            bpred.boundary_mask,
            cu_seqlens=cu,
        )

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

        # DeChunkLayer in packed mode takes the **chunked-space** cu_seqlens.
        dechunked = self.dechunk_layer(
            chunked,
            bpred.boundary_mask,
            bpred.boundary_prob,
            cu_seqlens=next_cu,
        )

        # STE residual gating (fp32 like upstream). Shapes in packed mode:
        #   dechunked: (T_total, D_in)
        #   selected_probs: (T_total, 1)  → STE returns ones_like → broadcasts.
        #   residual: (T_total, D_in)
        out = (dechunked.float() * _ste(bpred.selected_probs) + residual).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out

    def step(self, x: torch.Tensor, ctx: HNetContext, state: ChunkerStageState):
        bpred = self.routing_module.step(x, state.routing_state)
        residual = self.residual_scale * self.residual_proj(x.float())

        inner_in = self.chunk_layer.step(x, bpred.boundary_mask)
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
        out = (
            dechunked.float() * _ste(bpred.selected_probs.unsqueeze(1)) + residual
        ).to(x.dtype)

        ctx.register_aux(
            {
                "bpred": bpred,
                "target_ratio": self.target_compression_ratio,
                "weight": self.ratio_loss_weight,
            }
        )
        return out

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return ChunkerStageState(
            routing_state=self.routing_module.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            ),
            inner_state=(
                self.inner_stage._allocate(batch_size, max_seqlen, dtype, device)
                if self.inner_stage is not None
                else None
            ),
            dechunk_state=self.dechunk_layer.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype
            ),
        )

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        # ChunkerStage itself adds no transformer/MLP blocks to the residual
        # stream — the routing q/k are identity-init (marked _no_reinit) and
        # residual_proj is zero-init (also _no_reinit). proj_in / proj_out
        # are dim-bridge Linears between outer and inner working dim; use
        # the unscaled ``initializer_range`` (they're not in a residual
        # stream context).
        for m in (self.proj_in, self.proj_out):
            if m is not None and not getattr(m.weight, "_no_reinit", False):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.inner_stage is not None:
            return self.inner_stage._init_weights(initializer_range, parent_residuals)
        return parent_residuals


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
