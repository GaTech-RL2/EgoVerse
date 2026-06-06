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

from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.stages import _BaseStage


def apply_optimization_params(param: torch.Tensor, **kwargs) -> None:
    """Annotate a parameter with optimizer kwargs (lr_multiplier, weight_decay).

    Mirrors the upstream H-Net helper. ``HNet.parameter_groups`` later reads
    each parameter's ``_optim`` dict to build AdamW param groups.
    """
    if hasattr(param, "_optim"):
        param._optim.update(kwargs)
    else:
        param._optim = dict(kwargs)


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
            from egomimic.models.hnet.stages import ChunkerStage as _CS

            expected = (
                stages[i].output_hidden_dim
                if isinstance(stages[i], _CS)
                else stages[i].input_hidden_dim
            )
            if stages[i + 1].input_hidden_dim != expected:
                raise ValueError(
                    f"Hidden-dim mismatch: stages[{i + 1}].input_hidden_dim "
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

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, device, dtype=torch.float32
    ):
        """Returns a list with a single entry: the root stage's nested state."""
        return [self.root._allocate(batch_size, max_seqlen, dtype, device)]

    # ----- Training recipe (mirrors upstream hnet/models/hnet.py) ----- #

    def init_weights(self, initializer_range: float = 0.02) -> None:
        """Walk the stage chain and apply residual-stream-aware Linear init.

        ``out_proj`` and ``fc2`` weights are scaled by
        ``initializer_range / sqrt(n_residuals)`` where ``n_residuals`` is
        the cumulative residual depth through the hierarchy. Modules flagged
        ``_no_reinit`` (routing q/k, residual_proj, AdaLN) are skipped.
        """
        self.root._init_weights(initializer_range, parent_residuals=0)

    def apply_lr_multiplier(self, lr_multipliers) -> None:
        """Stamp every parameter with a per-stage ``lr_multiplier`` annotation.

        ``lr_multipliers`` is a list with one entry per stage in the flat
        list (outer first). Stage i's parameters get ``lr_multipliers[i]``.
        Read by ``parameter_groups`` when building optimizer param groups.
        """
        self.root._apply_lr_multiplier(lr_multipliers, stage_idx=0)

    def parameter_groups(self, weight_decay: float = 0.0) -> list:
        """Build AdamW parameter groups from per-parameter ``_optim`` annotations.

        Groups params by their ``_optim`` tuple. Sets ``weight_decay=0`` on
        biases and norm weights regardless of any other annotation.
        Returns ``list[dict]`` suitable for ``torch.optim.AdamW(params=...)``.

        If ``apply_lr_multiplier`` was never called, params have no
        ``_optim`` and end up in one group with ``lr_multiplier=1.0`` and the
        passed ``weight_decay`` (bias/norm still get WD=0).
        """
        # First pass: stamp bias / norm params with weight_decay=0. Other
        # params inherit the caller's ``weight_decay`` arg via the default
        # group below.
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".bias") or ".norm" in name or "rmsnorm" in name.lower():
                apply_optimization_params(p, weight_decay=0.0)

        # Build group dict keyed by sorted-tuple of (key, value) items from _optim.
        groups: dict = {}
        for p in self.parameters():
            if not p.requires_grad:
                continue
            optim = getattr(p, "_optim", {})
            key = tuple(sorted(optim.items()))
            entry = groups.setdefault(
                key, {"params": [], "weight_decay": weight_decay, "lr_multiplier": 1.0}
            )
            entry["params"].append(p)
            for k, v in optim.items():
                entry[k] = v
        return list(groups.values())


def chunk_stats_from_aux(aux: List[dict]) -> dict:
    """Per-chunker boundary stats for logging.

    For each entry in ``aux`` (one per chunker stage), report:
      - boundary_rate (F): fraction of tokens marked as boundaries.
      - avg_chunk_len: average length of a chunk in tokens. Defined as
        ``num_tokens / max(num_boundaries, 1)``; equals ``1 / F`` when F > 0
        and ``num_tokens`` (one chunk covering everything) when F == 0.

    Returns a dict with keys per chunker (``avg_chunk_len_0``,
    ``boundary_rate_0``, ...) plus aggregate ``avg_chunk_len`` /
    ``boundary_rate`` averaged across chunkers. All values are Python
    floats — these are pure logging stats, not loss terms.
    """
    out: dict = {}
    if not aux:
        return out
    rates = []
    lens = []
    for i, entry in enumerate(aux):
        bm = entry["bpred"].boundary_mask
        n = max(bm.numel(), 1)
        n_b = int(bm.sum().item())
        f = n_b / n
        avg_len = n / max(n_b, 1)
        out[f"boundary_rate_{i}"] = f
        out[f"avg_chunk_len_{i}"] = avg_len
        rates.append(f)
        lens.append(avg_len)
    out["boundary_rate"] = sum(rates) / len(rates)
    out["avg_chunk_len"] = sum(lens) / len(lens)
    return out


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
        # F4 (upstream parity): exclude positions where ``boundary_prob`` was
        # *forced* to 1.0 by the routing module. Padded mode: position 0 of
        # each row (the leading ``F.pad(..., 1.0)``). Packed mode:
        # ``cu_seqlens[:-1]`` (every subseq start). These are synthetic
        # boundaries, not learned predictions — counting them in F (fraction
        # selected) and G (mean boundary probability) inflates both toward
        # 1.0 and pulls the ratio loss away from its real target.
        cu = entry.get("cu_seqlens", None)
        if cu is not None:
            # Packed: bm/bp are (T_total,). Mask out subseq-start indices.
            valid = torch.ones_like(bm, dtype=torch.bool)
            valid[cu[:-1]] = False
            denom = max(int(valid.sum().item()), 1)
            vf = valid.to(bm.dtype)
            F_ = (bm * vf).sum() / denom
            G_ = (bp * vf).sum() / denom
        else:
            valid = entry.get("valid_mask_padded", None)
            if valid is not None:
                vf = valid.to(bm.dtype)
                denom = max(int(valid.sum().item()), 1)
                F_ = (bm * vf).sum() / denom
                G_ = (bp * vf).sum() / denom
            else:
                # Legacy / no mask info — preserve old behaviour.
                denom = max(bm.numel(), 1)
                F_ = bm.sum() / denom
                G_ = bp.sum() / denom
        loss_i = w * (N / (N - 1)) * ((N - 1) * F_ * G_ + (1 - F_) * (1 - G_))
        losses.append(loss_i)
    return torch.stack(losses).sum()
