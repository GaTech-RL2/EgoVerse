"""PerEmbodimentStage: cotrain helper that holds N parallel ``_BaseStage``
instances and dispatches forward / step by ``ctx.embodiment_id``.

Designed to wrap the *outer* (embodiment-specific) levels of an H-Net chain
— typically the per-emb EncoderDecoderStage and the per-emb ChunkerStage
that feed into a shared inner trunk.

Wiring contract (matches the standard HNet chain assembly):
    chain = [
        PerEmbodimentStage({"circle": EncDec_c, "stick": EncDec_s}),
        PerEmbodimentStage({"circle": Chunker_c, "stick": Chunker_s}),
        EncoderDecoderStage(...),        # shared inner
        ChunkerStage(...),                # shared inner
        ComputeStage(...),                # shared trunk
    ]

``HNet.__init__`` wires ``chain[i].inner_stage = chain[i+1]`` via the
standard loop. For PerEmbodimentStage, the ``inner_stage`` setter
propagates the assignment down to every per-emb sub-stage so they all
share the same inner-trunk pointer.

``ctx.embodiment_id`` (string domain name, e.g. ``"pushshapes_sim_stick"``)
must be set on HNetContext before forward / step is called. The algo-side
``HNet.forward_training`` is responsible for populating it.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.context import HNetContext

from .stages import _BaseStage  # noqa: PLC2701


class PerEmbodimentStage(_BaseStage):
    """Holds ``nn.ModuleDict[str, _BaseStage]`` of per-embodiment sub-stages
    and dispatches by ``ctx.embodiment_id``.

    All sub-stages must declare matching ``input_hidden_dim`` /
    ``output_hidden_dim`` (they're parallel parameter sets sharing the same
    I/O contract). The wrapper exposes those dims on itself so the standard
    HNet chain-wiring sees a single advertised dim.

    Cond key on the wrapper itself is ``None`` — sub-stages each carry
    their own ``cond_key`` and read it during their own forward.
    """

    def __init__(self, sub_stages: dict[str, _BaseStage]):
        if not sub_stages:
            raise ValueError("PerEmbodimentStage requires at least one sub-stage.")
        dims = {(ss.input_hidden_dim, ss.output_hidden_dim) for ss in sub_stages.values()}
        if len(dims) != 1:
            raise ValueError(
                f"All per-embodiment sub-stages must share input/output hidden "
                f"dims; got {dims}."
            )
        in_dim, out_dim = next(iter(dims))
        super().__init__(input_hidden_dim=in_dim, output_hidden_dim=out_dim, cond_key=None)
        self.sub_stages = nn.ModuleDict(sub_stages)

    # ------------------------------------------------------------------ #
    # inner_stage is a property that mirrors writes onto every sub-stage,
    # so the standard ``chain[i].inner_stage = chain[i+1]`` wiring loop
    # reaches the actual leaf stages that execute the forward.
    # ------------------------------------------------------------------ #

    @property
    def inner_stage(self) -> Optional[_BaseStage]:
        for ss in self.sub_stages.values():
            return ss.inner_stage
        return None

    @inner_stage.setter
    def inner_stage(self, value: Optional[_BaseStage]) -> None:
        # Called by ``_BaseStage.__init__`` before ``sub_stages`` exists; guard.
        if not hasattr(self, "sub_stages"):
            return
        for ss in self.sub_stages.values():
            ss.inner_stage = value

    @property
    def inner_working_dim(self) -> int:
        """Delegate to a representative sub-stage. All sub-stages are
        parallel and share the same handoff dim by construction."""
        sub = next(iter(self.sub_stages.values()))
        return sub.inner_working_dim

    @property
    def end_to_end_output_dim(self) -> int:
        """Delegate to a representative sub-stage (all parallel, same I/O)."""
        sub = next(iter(self.sub_stages.values()))
        return sub.end_to_end_output_dim

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def _active(self, ctx: HNetContext) -> _BaseStage:
        emb = getattr(ctx, "embodiment_id", None)
        if emb is None:
            raise RuntimeError(
                "PerEmbodimentStage requires ctx.embodiment_id to be set; "
                "algo-side code must populate it from the batch's embodiment."
            )
        if emb not in self.sub_stages:
            raise KeyError(
                f"PerEmbodimentStage has no sub-stage for embodiment {emb!r}; "
                f"available: {list(self.sub_stages.keys())}."
            )
        return self.sub_stages[emb]

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        return self._active(ctx)(x, ctx)

    def step(self, x: torch.Tensor, ctx: HNetContext, state: Any):
        emb = getattr(ctx, "embodiment_id", None)
        sub_state = state.get(emb) if isinstance(state, dict) else state
        return self._active(ctx).step(x, ctx, sub_state)

    def allocate_inference_cache(self, batch_size: int, dtype, device):
        # Allocate state for every sub-stage so AR inference can switch
        # embodiments mid-rollout if ever desired.
        return {
            emb: ss.allocate_inference_cache(batch_size, dtype, device)
            for emb, ss in self.sub_stages.items()
        }

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        """AR-cache allocation (pact-2 scheme; called by ``HNet.root._allocate``).

        Returns ``{emb: sub_stage_state}``. Each sub-stage's ``_allocate``
        recurses into the SHARED inner trunk, so every embodiment gets its own
        inner-cache copy — harmless (unlike weight init, these are just
        per-rollout buffers, and only the active embodiment's cache is ever
        advanced). ``step`` dispatches to ``state[ctx.embodiment_id]``.
        """
        return {
            emb: ss._allocate(batch_size, max_seqlen, dtype, device)
            for emb, ss in self.sub_stages.items()
        }

    # ------------------------------------------------------------------ #
    # Init / LR plumbing
    # ------------------------------------------------------------------ #

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        """Run each sub-stage's init independently, then recurse into the
        shared inner_stage ONCE.

        Each sub-stage's own ``_init_weights`` recurses into its
        ``inner_stage`` by default; since all sub-stages share the same
        inner pointer, we'd double-init. We null each sub-stage's
        inner_stage during its init call, then restore.
        """
        n_residuals = parent_residuals
        shared_inner = self.inner_stage
        for ss in self.sub_stages.values():
            saved = ss.inner_stage
            ss.inner_stage = None
            n_residuals = ss._init_weights(initializer_range, parent_residuals)
            ss.inner_stage = saved
        if shared_inner is not None:
            n_residuals = shared_inner._init_weights(initializer_range, n_residuals)
        return n_residuals

    # ``_apply_lr_multiplier`` inherited from ``_BaseStage`` works as-is:
    # the default impl iterates ``named_children`` (= sub_stages ModuleDict),
    # stamps every param with the multiplier, then recurses into
    # self.inner_stage once. Per-emb sub-stages get the same multiplier
    # (they're parallel at the same depth).
