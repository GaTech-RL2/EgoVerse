"""StreamBundle: the dual-stream currency flowing through dual-stream H-Net stages.

A :class:`StreamBundle` carries the emb-AGNOSTIC stream ``A`` (weight-shared
across embodiments) and the per-embodiment SPECIFIC stream ``S`` together with
the packed-sequence metadata (episode boundaries + within-episode time index)
that every dual-stream module needs to build its attention masks identically.

Design (see vault: Dual-Stream Shared/Specific Arch — v2):
  * ``_BaseStage.forward/step`` dispatch on ``isinstance(x, StreamBundle)``:
    single-stream stages keep the legacy plain-Tensor path untouched; dual-stream
    stages require a StreamBundle. NO ctx flag is the source of truth (avoids a
    forgotten-flag class of bugs); ``ctx.extras['dual_mode']`` is set only for
    cheap asserts/logging.
  * The apex (main) trunk consumes ONLY ``A`` (collapsed to a bare Tensor at the
    seam in ``DualStreamChunkerStage``); ``S`` bypasses the apex and rejoins at
    dechunk + the partitioned GMM head.
  * ``A`` and ``S`` share ``T_total`` / ``cu_seqlens`` / ``time_pos`` (they are
    the same obs sequence, different features) until a chunker rewrites them
    atomically for the chunked-rate recursion.

This replaces the v1 concat-tensor + scattered ``ctx.extras['dual'/'ms']`` keys
with one explicit, composable object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class StreamBundle:
    """A pair of packed token streams (agnostic ``A`` + specific ``S``) + metadata."""

    A: torch.Tensor            # agnostic stream, packed (T_total, d_a)
    S: torch.Tensor            # specific (per-emb) stream, packed (T_total, d_s)
    T_total: int               # tokens per stream (A and S share this)
    cu_seqlens: torch.Tensor   # (B+1,) episode boundaries in this stream's token space
    time_pos: torch.Tensor     # (T_total,) within-episode time index, shared by A_t and S_t

    @property
    def d_a(self) -> int:
        return int(self.A.shape[-1])

    @property
    def d_s(self) -> int:
        return int(self.S.shape[-1])

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "StreamBundle":
        """Apply ``fn`` to both A and S; metadata copied unchanged."""
        return _dc_replace(self, A=fn(self.A), S=fn(self.S))

    def replace(
        self,
        A: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None,
        **md,
    ) -> "StreamBundle":
        """Return a copy with ``A``/``S`` and/or metadata fields overridden."""
        kw = {}
        if A is not None:
            kw["A"] = A
        if S is not None:
            kw["S"] = S
        kw.update(md)
        return _dc_replace(self, **kw)


def build_same_episode_causal_mask(
    T_total: int,
    cu_seqlens: torch.Tensor,
    time_pos: torch.Tensor,
    causal: bool,
    device,
    window: "int | None" = None,
) -> torch.Tensor:
    """``(T_total, T_total)`` bool allow-mask: same-episode AND (optionally)
    time-causal, with the diagonal always allowed.

    This is the per-stream attention mask shared by ``A`` and ``S`` (they share
    episode boundaries + within-episode time index). Factored verbatim from
    ``MultiStreamComputeStage._build_mask`` so every dual-stream stage builds the
    mask identically (single source of truth).

    ``window``: sliding causal window in THIS grid's tokens — token i may
    attend only to j with ``tp[i] - tp[j] < window`` (plus same-episode +
    causal). ``None`` (default) = full within-episode history, byte-identical
    to before this knob. Only meaningful with ``causal=True``.
    """
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    ep = torch.zeros(T_total, dtype=torch.long, device=device)
    for b in range(cu.shape[0] - 1):
        ep[cu[b]:cu[b + 1]] = b
    tp = time_pos.to(device=device, dtype=torch.long)
    same_ep = ep[:, None] == ep[None, :]
    if causal:
        time_ok = tp[None, :] <= tp[:, None]
        if window is not None:
            time_ok = time_ok & ((tp[:, None] - tp[None, :]) < int(window))
    else:
        time_ok = torch.ones(T_total, T_total, dtype=torch.bool, device=device)
    allow = (same_ep & time_ok) | torch.eye(T_total, dtype=torch.bool, device=device)
    return allow
