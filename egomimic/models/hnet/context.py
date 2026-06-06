"""
Conditioning + aux context object threaded through all H-Net stages.

Built by the algo (HNetPolicy) at the top of forward / generate, then handed
to the stage tree. Stages read named tensors from ``cond_dict`` and append
auxiliary information (e.g. boundary predictions for the ratio loss) via
``register_aux``. ``inference_params`` carries per-stage step-time state when
running autoregressively; it is left None during teacher-forced forward.

Packed mode (FlashAttention-style varlen):
    Set ``cu_seqlens`` (LongTensor (B+1,), starts at 0) and ``max_seqlen``
    (int) on the context. Stages will then drive their sub-modules in packed
    mode: hidden states shaped ``(T_total, D)``, attention via varlen / SDPA
    block-diagonal mask, routing/chunk/dechunk via their packed paths.
    When ``cu_seqlens`` is None the stages fall back to padded mode with an
    all-ones mask of shape ``(B, L)``.

    The ``ChunkerStage`` saves and restores ``cu_seqlens`` / ``max_seqlen``
    around the call into its inner stage (substituting the chunked-space
    cu_seqlens for the recursion), so callers only ever set the *outer*
    pack info on the context and don't manage it per-level.

    In packed mode, ``cond_dict[cond_key]`` (when used) must be shaped
    ``(T_total, d_cond)`` to align with the per-token packed stream.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class HNetContext:
    """Per-call context for a stage tree.

    Attributes:
        cond_dict: named conditioning tensors. The default "fused_cond" entry
            (per-token cond of shape (B, T, d_cond) padded / (T_total, d_cond)
            packed) is produced by ``CondEncoderModule``; user-defined keys
            may be added alongside.
        aux: each chunker stage's forward / step appends a dict here with at
            least ``{"bpred": RoutingModuleOutput, "target_ratio": float,
            "weight": float}``. The algo iterates this to compute the ratio
            loss after forward returns.
        inference_params: per-stage state list during step inference, or None
            during teacher-forced forward.
        extras: free-form scratch dict for callers that want to pass extra
            information down (and back up) through the stage tree.
        cu_seqlens: optional (B+1,) long tensor with cumulative subseq lengths
            (starts at 0). Setting this switches stages into packed mode.
        max_seqlen: optional int — longest subseq length, required when
            cu_seqlens is set (attention kernels need it).
    """

    cond_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: List[Dict[str, Any]] = field(default_factory=list)
    inference_params: Optional[List[Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @property
    def packed(self) -> bool:
        return self.cu_seqlens is not None

    def register_aux(self, item: Dict[str, Any]) -> None:
        self.aux.append(item)
