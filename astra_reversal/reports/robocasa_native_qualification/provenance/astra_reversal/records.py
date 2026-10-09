"""Append-only traces with separate, lossless array artifacts."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value) -> str:
    """Hash complete request contents, including pixels, rather than just IDs."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            result.update(b"array:")
            result.update(str((item.dtype.str, item.shape)).encode())
            result.update(np.ascontiguousarray(item).tobytes())
        elif is_dataclass(item):
            visit(asdict(item))
        elif isinstance(item, dict):
            result.update(b"{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            result.update(b"}")
        elif isinstance(item, (list, tuple)):
            result.update(b"[")
            for entry in item:
                visit(entry)
            result.update(b"]")
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
            result.update(b"\0")

    visit(value)
    return result.hexdigest()


@dataclass
class InversionResult:
    plan_id: str
    controller_actions: Any
    model_actions: Any
    inverse_condition_id: str
    noise: Any
    solver: str
    grid: list[float]
    checkpoint: dict
    velocity_evaluations: int
    latency_seconds: float
    reference: dict | None = None


@dataclass
class ControlStep:
    episode_id: str
    observation_step: int
    observation_id: str
    plan_id: str | None
    latent_id: str | None
    condition_id: str
    action: Any
    success: bool
    fallback_reason: str | None
    progress: dict
    timestamp: float


@dataclass
class RunManifest:
    config: dict
    versions: dict
    action_spec: dict
    checkpoint: dict
    task_manifest_sha256: str
    created_at: float


class Recorder:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "arrays").mkdir()
        self._sequence = 0
        self._array_sequence = 0

    def _serialize(self, value):
        if is_dataclass(value):
            return self._serialize(asdict(value))
        if isinstance(value, np.ndarray) or hasattr(value, "detach"):
            array = to_numpy(value)
            name = f"arrays/{self._array_sequence:08d}.npy"
            self._array_sequence += 1
            np.save(self.directory / name, array, allow_pickle=False)
            return {
                "array": name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": digest(array),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize(x) for x in value]
        return value

    def event(self, kind: str, **data):
        record = self._serialize(
            {"kind": kind, "sequence": self._sequence, "timestamp": time.time(), **data}
        )
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._sequence += 1

    def manifest(self, manifest: RunManifest):
        (self.directory / "manifest.json").write_text(
            json.dumps(self._serialize(manifest), indent=2, allow_nan=False) + "\n"
        )


class MemoryRecorder:
    """Small in-memory recorder for deterministic controller tests."""

    def __init__(self):
        self.events = []

    def event(self, kind, **data):
        self.events.append({"kind": kind, **data})
