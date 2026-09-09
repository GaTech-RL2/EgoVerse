#!/usr/bin/env python3
"""Capture an immutable Python, dependency, and source lock for an ICE run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Any

CORE_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "lightning",
    "hydra-core",
    "omegaconf",
    "torchdiffeq",
    "wandb",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def distribution_inventory() -> list[dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata.get("Name") or "").strip()
        version = str(dist.version).strip()
        if not name or not version:
            raise RuntimeError("installed distribution lacks a name or version")
        normalized = name.lower().replace("_", "-")
        rows[(normalized, version)] = {"name": normalized, "version": version}
    return [rows[key] for key in sorted(rows)]


def atomic_create(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime lock: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--lock-input",
        action="append",
        default=[],
        type=pathlib.Path,
        help="repeat for dependency inputs such as uv.lock",
    )
    parser.add_argument("--expected-head", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if git(repo, "rev-parse", "HEAD") != args.expected_head:
        raise RuntimeError("repository HEAD does not match --expected-head")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("repository must be exactly clean")

    executable = pathlib.Path(sys.executable)
    resolved_executable = executable.resolve()
    packages = distribution_inventory()
    package_bytes = canonical_bytes(packages)
    core_versions = {
        name: importlib.metadata.version(name) for name in CORE_DISTRIBUTIONS
    }

    import torch

    inputs = []
    for raw in args.lock_input:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    inputs.sort(key=lambda row: row["path"])

    payload = {
        "schema_version": 1,
        "status": "RUNTIME_LOCKED",
        "captured_at_unix_ns": time.time_ns(),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(executable),
            "resolved_executable": str(resolved_executable),
            "resolved_executable_size_bytes": resolved_executable.stat().st_size,
            "resolved_executable_sha256": sha256(resolved_executable),
        },
        "torch_runtime": {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available_on_capture_node": torch.cuda.is_available(),
        },
        "core_distribution_versions": core_versions,
        "installed_distributions": {
            "count": len(packages),
            "canonical_sha256": hashlib.sha256(package_bytes).hexdigest(),
            "items": packages,
        },
        "dependency_inputs": inputs,
        "source": {
            "repo": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "tree": git(repo, "rev-parse", "HEAD^{tree}"),
            "submodules": git(repo, "submodule", "status", "--recursive"),
            "dirty": False,
        },
    }
    encoded = canonical_bytes(payload)
    atomic_create(output, encoded)
    print(
        json.dumps(
            {
                "status": "RUNTIME_LOCKED",
                "path": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "package_count": len(packages),
                "package_inventory_sha256": payload["installed_distributions"][
                    "canonical_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
