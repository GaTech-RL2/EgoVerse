"""Small dependency-aware stage runner for configured Pipeline graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

_EXECUTION_MODES = frozenset({"train", "inference"})


def resolve_homogeneous_scalar(value, *, label: str = "selector"):
    """Return one Python scalar from a scalar or homogeneous 1-D collection.

    Batched metadata is commonly collated into a 1-D tensor, ndarray, or list.
    Stages that select a branch can use this helper without teaching the
    pipeline what the selector represents.
    """

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(f"{label} must be a scalar or non-empty 1-D value")
        first = value[0]
        if not bool(torch.all(value == first)):
            raise ValueError(f"{label} must be homogeneous within one batch")
        return first.item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.ndim != 1 or value.size == 0:
            raise ValueError(f"{label} must be a scalar or non-empty 1-D value")
        first = value[0]
        if not bool(np.all(value == first)):
            raise ValueError(f"{label} must be homogeneous within one batch")
        return first.item() if isinstance(first, np.generic) else first

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{label} must be non-empty")
        resolved = [resolve_homogeneous_scalar(item, label=label) for item in value]
        first = resolved[0]
        if any(item != first for item in resolved[1:]):
            raise ValueError(f"{label} must be homogeneous within one batch")
        return first

    if np.isscalar(value):
        return value.item() if isinstance(value, np.generic) else value
    raise TypeError(
        f"{label} must be a scalar, tensor, ndarray, list, or tuple; "
        f"got {type(value).__name__}"
    )


def sum_losses(batch: dict):
    """Sum every scalar written under the ``loss/`` namespace."""
    losses = [(key, value) for key, value in batch.items() if key.startswith("loss/")]
    if not losses:
        raise RuntimeError("no loss/* keys in batch -- no loss stage ran")
    for key, value in losses:
        if not torch.is_tensor(value) or value.ndim != 0:
            raise TypeError(
                f"{key} must be a scalar tensor, got {type(value).__name__} "
                f"with shape {getattr(value, 'shape', None)}"
            )
    values = [value for _, value in losses]
    return sum(values[1:], start=values[0])


class Stage(nn.Module):
    """A ``dict -> dict`` module with declarative read/write contracts.

    ``train_only`` and ``inference_only`` drop a stage from the other mode's
    graph. They exist because a stage that is merely missing a read is a
    CONFIGURATION ERROR here -- ``Pipeline.execute`` raises on a blocked stage
    rather than skipping it -- so mode-restricted stages have to say so rather
    than rely on their inputs happening to be absent.
    """

    reads: Sequence[str] = ()
    writes: Sequence[str] = ()
    reads_by_mode: Mapping[str, Sequence[str]] = {}
    writes_by_mode: Mapping[str, Sequence[str]] = {}

    def contract(self, mode: str = "train") -> tuple[tuple[str, ...], tuple[str, ...]]:
        if mode not in _EXECUTION_MODES:
            raise ValueError(
                f"Stage contract mode must be train|inference, got {mode!r}"
            )
        reads = self.reads_by_mode.get(mode, self.reads)
        writes = self.writes_by_mode.get(mode, self.writes)
        return tuple(reads or ()), tuple(writes or ())

    def execute(self, batch: dict, *, mode: str) -> dict:
        """Execute this stage for an already validated graph mode."""
        return self(batch)

    def forward(self, batch: dict) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def _read_is_available(read: str, available: set[str]) -> bool:
    if read.endswith("*"):
        return any(key.startswith(read[:-1]) for key in available)
    if read in available:
        return True
    return any(
        written.endswith("*") and read.startswith(written[:-1]) for written in available
    )


class Pipeline(Stage):
    """An ordered, registered list of stages."""

    def __init__(self, stages: Sequence[Stage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, batch: dict, mode: str = "train") -> dict:
        return self.execute(batch, mode=mode)

    def plan(
        self, seed_keys: Sequence[str], mode: str = "train"
    ) -> tuple[list[Stage], list[tuple[Stage, list[str]]]]:
        """Resolve runnable stages without executing the graph."""
        if mode not in _EXECUTION_MODES:
            raise ValueError(
                f"Pipeline.plan mode must be train|inference, got {mode!r}"
            )

        available = set(seed_keys)
        runnable: list[Stage] = []
        excluded: list[tuple[Stage, list[str]]] = []
        for stage in self.stages:
            if mode == "inference" and getattr(stage, "train_only", False):
                excluded.append((stage, ["<train-only>"]))
                continue
            if mode == "train" and getattr(stage, "inference_only", False):
                excluded.append((stage, ["<inference-only>"]))
                continue
            reads, writes = stage.contract(mode)
            missing = [key for key in reads if not _read_is_available(key, available)]
            if missing:
                excluded.append((stage, missing))
                continue
            runnable.append(stage)
            available.update(writes)
        return runnable, excluded

    def execute(self, batch: dict, *, mode: str) -> dict:
        """Run the selected subgraph from a flat mapping.

        The input is shallow-copied so stage mutation cannot alter evaluator or
        loader state. Ordinary metadata remains available to specialized stages;
        read/write contracts determine graph dependencies rather than filtering
        the shared dictionary.
        """

        if not isinstance(batch, Mapping):
            raise TypeError("Pipeline input must be a flat mapping")
        if any(not isinstance(key, str) for key in batch):
            raise TypeError("Pipeline input keys must be strings")

        runnable, excluded = self.plan(tuple(batch), mode=mode)
        blocked = [
            (type(stage).__name__, missing)
            for stage, missing in excluded
            if missing not in (["<train-only>"], ["<inference-only>"])
        ]
        if blocked:
            raise RuntimeError(f"Pipeline {mode} graph has blocked stages: {blocked}")

        result = dict(batch)
        for stage in runnable:
            result = stage.execute(result, mode=mode)
            if not isinstance(result, dict):
                raise TypeError(
                    f"{type(stage).__name__} returned {type(result).__name__}, "
                    "expected dict"
                )
        return result

    def explain(self, seed_keys: Sequence[str] = (), mode: str = "train") -> str:
        """Return a compact, human-readable dependency plan."""
        if mode not in _EXECUTION_MODES:
            raise ValueError(
                f"Pipeline.explain mode must be train|inference, got {mode!r}"
            )

        available = set(seed_keys)
        lines = []
        for stage in self.stages:
            reads, writes = stage.contract(mode)
            if mode == "inference" and getattr(stage, "train_only", False):
                reason = " (EXCLUDED: train-only)"
            elif mode == "train" and getattr(stage, "inference_only", False):
                reason = " (EXCLUDED: inference-only)"
            else:
                missing = [
                    key for key in reads if not _read_is_available(key, available)
                ]
                reason = f" (EXCLUDED: missing {','.join(missing)})" if missing else ""
                if not missing:
                    available.update(writes)
            lines.append(
                f"{type(stage).__name__:28s} reads={list(reads)} "
                f"writes={list(writes)}{reason}"
            )
        return "\n".join(lines)
