"""
Top-level H-Net wrapper for the stage-based architecture.

The YAML provides a *flat* list of stages. At construction time we walk the
list and set ``stages[i].inner_stage = stages[i+1]`` so the runtime tree is
recursive even though the config is flat. The terminal stage's ``inner_stage``
remains None.

The wrapper itself owns no learned modules — all parameters live inside the
individual stages. It exists to:

  * thread the ``HNetContext`` into the root stage on forward / step;
  * collect per-stage inference-state objects for autoregressive rollout;
  * expose the ratio_loss helper at the module level for the algo.

action_in / action_out / BOS / pos_emb / cond encoders live on the algo
class (``HNetPolicy``), not here.
"""
from typing import List

import torch
import torch.nn as nn

from egomimic.models.hnet_nets.context import HNetContext
from egomimic.models.hnet_nets.stages import _BaseStage


class HNet(nn.Module):
    def __init__(self, stages: List[_BaseStage]):
        super().__init__()
        if not stages:
            raise ValueError("HNet requires at least one stage.")
        # The flat list is wired into a chain. Assert each stage was constructed
        # with inner_stage=None so the user can't accidentally double-specify.
        for i, st in enumerate(stages):
            if not isinstance(st, _BaseStage):
                raise TypeError(
                    f"stages[{i}] is not a Stage instance (got {type(st).__name__})."
                )
            if st.inner_stage is not None:
                raise ValueError(
                    f"stages[{i}] already has inner_stage set; pass a flat list "
                    f"and let HNet wire the chain."
                )
        for i in range(len(stages) - 1):
            # Sanity-check dim handoff into the nested inner_stage. For
            # EncoderDecoderStage/ComputeStage the inner stage runs at the
            # outer working dim (== input_hidden_dim). For ChunkerStage the
            # inner stage runs in the chunked space at output_hidden_dim
            # (after the explicit proj_in).
            from egomimic.models.hnet_nets.stages import ChunkerStage as _CS
            expected = (
                stages[i].output_hidden_dim
                if isinstance(stages[i], _CS)
                else stages[i].input_hidden_dim
            )
            if stages[i + 1].input_hidden_dim != expected:
                raise ValueError(
                    f"Hidden-dim mismatch: stages[{i+1}].input_hidden_dim "
                    f"({stages[i + 1].input_hidden_dim}) does not match the "
                    f"inner working dim of stages[{i}] ({expected})."
                )
            stages[i].inner_stage = stages[i + 1]

        self.stages = nn.ModuleList(stages)
        # The root stage is stages[0]; deeper stages are only referenced via
        # inner_stage. We still keep them in the ModuleList so nn.Module sees
        # all params (the inner_stage attribute is set above which also makes
        # them children of the parent stage, but ModuleList is the canonical
        # registration). To avoid double-registration we detach inner_stage
        # from the parent's submodules and rely on ModuleList alone — but
        # detach is awkward; instead we leave both registered (PyTorch
        # deduplicates by parameter identity, but module-level it would double
        # count submodule names). Simplest robust choice: do NOT register
        # inner_stage via setattr; use object.__setattr__ to bypass the
        # automatic child registration in nn.Module.
        for i in range(len(stages) - 1):
            # Re-bind inner_stage as a *non-submodule* attribute to avoid
            # duplicate registration; ModuleList keeps the actual parameters.
            object.__setattr__(stages[i], "inner_stage", stages[i + 1])

    @property
    def root(self) -> _BaseStage:
        return self.stages[0]

    @property
    def input_hidden_dim(self) -> int:
        return self.stages[0].input_hidden_dim

    @property
    def output_hidden_dim(self) -> int:
        # Chain is recursive (stages[i].inner_stage = stages[i+1]) so the
        # end-to-end output dim is the outermost stage's output dim, not the
        # innermost. EncoderDecoder/Compute/Chunker stages each guarantee that
        # they return to their own output_hidden_dim before their forward ends.
        return self.stages[0].output_hidden_dim

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        return self.root(x, ctx)

    def step(self, x: torch.Tensor, ctx: HNetContext):
        """Single-token step. ``ctx.inference_params[0]`` is the root state."""
        assert ctx.inference_params is not None, "Call allocate_inference_cache first."
        return self.root.step(x, ctx, ctx.inference_params[0])

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, device, dtype=torch.float32):
        """Returns a list with a single entry: the root stage's nested state."""
        return [self.root._allocate(batch_size, max_seqlen, dtype, device)]


def ratio_loss_from_aux(aux: List[dict], device=None) -> torch.Tensor:
    """Compute the H-Net auxiliary ratio loss from a list of aux entries.

    Each entry is::

        {"bpred": RoutingModuleOutput, "target_ratio": float, "weight": float}

    Loss (per chunker, summed):
        N = target_ratio
        F = fraction of tokens selected as boundaries
        G = mean boundary probability
        L = weight * N/(N-1) * ((N-1)*F*G + (1-F)*(1-G))
    """
    if not aux:
        return torch.zeros((), device=device or "cpu")
    losses = []
    for entry in aux:
        bpred = entry["bpred"]
        N = float(entry["target_ratio"])
        w = float(entry["weight"])
        bm = bpred.boundary_mask.float()
        bp = bpred.boundary_prob[..., -1]
        # We treat all positions as valid (no padding mask).
        denom = max(bm.numel(), 1)
        F_ = bm.sum() / denom
        G_ = bp.sum() / denom
        loss_i = w * (N / (N - 1)) * ((N - 1) * F_ * G_ + (1 - F_) * (1 - G_))
        losses.append(loss_i)
    return torch.stack(losses).sum()
