"""Batchflow core — the entire framework.

Convention (BATCHFLOW.md): every module is ``stage(batch: dict) -> dict``.
ONE carrier (the batch dict), ONE interface. Stages read keys, compute, write
keys ("keep adding keys as you compute"). No context classes, no bundles, no
attribute stashes, no side channels.

Key namespaces:
    inputs   : "actions", "__obs", "embodiment", "cu_seqlens", "max_seq_len", ...
    features : "x", "A", "S", "a_top", "s", "cvae_z", ...
    outputs  : "pred_action"
    loss/*   : scalar loss terms — the runner sums them into the training loss
    log/*    : scalar diagnostics — the runner logs them (per-emb prefixed)
    aux/*    : accumulator lists (regularizer entries, probe payloads). Lists
               are SHARED through sub_batch shallow copies, so inner recursion
               levels append to the same accumulator.

Recursion rule: a stage that recurses into an inner stage at a different
temporal resolution hands it ``sub_batch(batch, x=..., cu_seqlens=next_cu, ...)``
— a SHALLOW copy with overridden view keys. Per-level state stays in local
variables; nothing is mutated-and-restored. (This deletes the old
HNetContext save/swap/restore dance and makes the mutate-without-restore
bug class unrepresentable.)
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

import torch.nn as nn


def sub_batch(batch: dict, **overrides) -> dict:
    """Shallow copy of ``batch`` with overridden view keys.

    Scalars/tensors override per-level; accumulator LISTS (aux/*) remain the
    same shared objects, so appends made inside the inner level are visible
    to the outer level. This is the one recursion primitive.
    """
    out = dict(batch)
    out.update(overrides)
    return out


def sum_losses(batch: dict):
    """Sum every ``loss/*`` entry into the total training loss."""
    total = None
    for k, v in batch.items():
        if k.startswith("loss/"):
            total = v if total is None else total + v
    if total is None:
        raise RuntimeError("no loss/* keys in batch — no loss stage ran")
    return total


class Stage(nn.Module):
    """Base module of the batchflow convention.

    Subclasses declare ``reads`` / ``writes`` (key contracts) and implement
    ``forward(batch) -> batch``. A stage whose training and rollout behavior
    differs can override either contract through ``reads_by_mode`` and
    ``writes_by_mode``. Contracts are used by :meth:`Pipeline.plan` for
    build-time validation and mode resolution — a stage whose reads cannot be
    satisfied is EXCLUDED at plan time with a loud report, never silently
    skipped at runtime.

    Wildcard suffix "*" in a contract entry matches any key with that prefix
    (e.g. writes = ["loss/*"]).
    """

    reads: Sequence[str] = ()
    writes: Sequence[str] = ()
    reads_by_mode: Mapping[str, Sequence[str]] = {}
    writes_by_mode: Mapping[str, Sequence[str]] = {}

    def contract(self, mode: str = "train") -> Tuple[Sequence[str], Sequence[str]]:
        """Return the exact read/write declaration for ``mode``.

        Plain stages retain the historical ``reads`` / ``writes`` behavior.
        Mode-aware stages override only the modes that differ, keeping this a
        backward-compatible extension for existing configs and graph tools.
        """
        if mode not in {"train", "rollout"}:
            raise ValueError(f"Stage contract mode must be train|rollout, got {mode!r}")
        reads = self.reads_by_mode.get(mode, self.reads)
        writes = self.writes_by_mode.get(mode, self.writes)
        return tuple(reads or ()), tuple(writes or ())

    def forward(self, batch: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def _matches(key: str, patterns: Iterable[str]) -> bool:
    for p in patterns:
        if p.endswith("*"):
            if key.startswith(p[:-1]):
                return True
        elif key == p:
            return True
    return False


class Pipeline(Stage):
    """An ordered list of stages; itself a Stage (pipelines nest)."""

    def __init__(self, stages: List[Stage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, batch: dict) -> dict:
        for stage in self.stages:
            batch = stage(batch)
        return batch

    # ---------------- plan-time dependency resolution ---------------- #
    def plan(
        self, seed_keys: Sequence[str], mode: str = "train"
    ) -> Tuple[List[Stage], List[Tuple[Stage, List[str]]]]:
        """Resolve which stages can run given ``seed_keys``.

        Returns (runnable, excluded) where excluded pairs each stage with the
        keys it is missing. Deterministic, computed once per mode — the
        rollout path calls this with obs-only seeds and loss/posterior stages
        drop out PROVABLY (reported, not silently skipped).
        """
        if mode not in {"train", "rollout"}:
            raise ValueError(f"Pipeline.plan mode must be train|rollout, got {mode!r}")
        have = set(seed_keys)
        runnable, excluded = [], []
        for stage in self.stages:
            if mode == "rollout" and getattr(stage, "train_only", False):
                excluded.append((stage, ["<train-only>"]))
                continue
            reads, writes = stage.contract(mode)
            missing = [r for r in reads if not _matches_any(r, have)]
            if missing:
                excluded.append((stage, missing))
            else:
                runnable.append(stage)
                for w in writes:
                    have.add(w)
        return runnable, excluded

    def explain(self, seed_keys: Sequence[str] = (), mode: str = "train") -> str:
        """Human-readable key-flow for an exact training or rollout graph."""
        if mode not in {"train", "rollout"}:
            raise ValueError(
                f"Pipeline.explain mode must be train|rollout, got {mode!r}"
            )
        lines = []
        have = set(seed_keys)
        for stage in self.stages:
            reads, writes = stage.contract(mode)
            if mode == "rollout" and getattr(stage, "train_only", False):
                lines.append(
                    f"{type(stage).__name__:28s} reads={list(reads)} "
                    f"writes={list(writes)} (EXCLUDED: train-only)"
                )
                continue
            missing = [r for r in reads if not _matches_any(r, have)]
            tag = " (EXCLUDED: missing %s)" % ",".join(missing) if missing else ""
            lines.append(
                f"{type(stage).__name__:28s} reads={list(reads)} "
                f"writes={list(writes)}{tag}"
            )
            if not missing:
                have.update(writes)
        return "\n".join(lines)


def _matches_any(read: str, have: set) -> bool:
    """A read is satisfied if an exact key, a wildcard-covered key, or a
    prefix-written family (writer declared 'foo/*') is present."""
    if read.endswith("*"):
        prefix = read[:-1]
        return any(k.startswith(prefix) for k in have)
    if read in have:
        return True
    return any(h.endswith("*") and read.startswith(h[:-1]) for h in have)
