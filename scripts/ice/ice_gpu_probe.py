#!/usr/bin/env python3
"""Fail-closed single-GPU ICE health and real-BF16 compute probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import time
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

_HEALTHY_FLAGS = {"0", "n/a", "no", "none", "not required", "not supported"}
_UNHEALTHY_FLAG_KEYS = (
    "pending page blacklist",
    "pending row remapping",
    "remapping failure occurred",
    "reset required",
    "repair pending",
)


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
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


def parse_ecc_health(report: str) -> dict[str, Any]:
    """Extract fail-closed uncorrectable-ECC and pending-repair status."""

    uncorrectable: list[dict[str, int | str]] = []
    unhealthy_flags: list[dict[str, str]] = []
    for raw_line in report.splitlines():
        if ":" not in raw_line:
            continue
        raw_key, raw_value = raw_line.split(":", 1)
        key = " ".join(raw_key.strip().lower().split())
        value = " ".join(raw_value.strip().lower().split())
        if "uncorrectable" in key:
            if value in {"n/a", "not supported"}:
                continue
            match = re.fullmatch(r"([0-9]+)(?:\s+.*)?", value)
            if match is None:
                raise RuntimeError(
                    f"could not interpret uncorrectable ECC counter {key!r}: {value!r}"
                )
            count = int(match.group(1))
            uncorrectable.append({"counter": key, "value": count})
        if any(marker in key for marker in _UNHEALTHY_FLAG_KEYS):
            if value not in _HEALTHY_FLAGS:
                unhealthy_flags.append({"field": key, "value": value})

    if not uncorrectable:
        raise RuntimeError("nvidia-smi did not expose any uncorrectable ECC counters")
    nonzero = [row for row in uncorrectable if row["value"] != 0]
    if nonzero or unhealthy_flags:
        raise RuntimeError(
            "GPU ECC/repair health check failed: "
            + json.dumps(
                {"nonzero_uncorrectable": nonzero, "unhealthy_flags": unhealthy_flags},
                sort_keys=True,
            )
        )
    return {
        "uncorrectable_ecc_counters": uncorrectable,
        "unhealthy_flags": unhealthy_flags,
    }


def _required_slurm_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is required")
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if result < 0:
        raise RuntimeError(f"{name} must be nonnegative")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    names = parser.add_mutually_exclusive_group(required=True)
    names.add_argument("--expected-gpu-name")
    names.add_argument("--allowed-gpu-name", action="append")
    parser.add_argument("--expected-world-size", type=int, choices=(1,), default=1)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix_size < 256:
        raise RuntimeError("--matrix-size must be at least 256 for a real BF16 smoke")

    rank = _required_slurm_integer("SLURM_PROCID")
    local_rank = _required_slurm_integer("SLURM_LOCALID")
    world_size = _required_slurm_integer("SLURM_NTASKS")
    if rank != 0 or local_rank != 0 or world_size != args.expected_world_size:
        raise RuntimeError(
            "single-GPU probe requires rank=0, local_rank=0, and world_size=1"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device; found {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    allowed_names = args.allowed_gpu_name or [args.expected_gpu_name]
    if not any(name.casefold() in gpu_name.casefold() for name in allowed_names):
        raise RuntimeError(
            f"allocated GPU {gpu_name!r} does not match any allowed name: "
            f"{allowed_names!r}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"allocated GPU does not support BF16: {gpu_name}")

    smi = subprocess.run(
        ["nvidia-smi", "-q", "-i", "0"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if smi.returncode != 0:
        raise RuntimeError(f"nvidia-smi health query failed: {smi.stderr.strip()}")
    health = parse_ecc_health(smi.stdout)

    generator = torch.Generator(device="cuda:0").manual_seed(420042)
    left = torch.randn(
        args.matrix_size,
        args.matrix_size,
        device="cuda:0",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    right = torch.randn(
        args.matrix_size,
        args.matrix_size,
        device="cuda:0",
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        product = left @ right
        loss = product.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    gradients = (left.grad, right.grad)
    if (
        product.dtype != torch.bfloat16
        or not bool(torch.isfinite(product).all())
        or any(gradient is None for gradient in gradients)
        or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
        or any(gradient.dtype != torch.bfloat16 for gradient in gradients)
    ):
        raise RuntimeError("real BF16 forward/backward was not finite BF16")

    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("the allocated PyTorch runtime does not provide NCCL")
    if dist.is_initialized():
        raise RuntimeError("distributed process group was initialized before the probe")
    collective_value: float | None = None
    with tempfile.TemporaryDirectory(prefix="egoverse-nccl-probe-") as directory:
        rendezvous = pathlib.Path(directory) / "rendezvous"
        try:
            dist.init_process_group(
                backend="nccl",
                init_method=f"file://{rendezvous}",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=60),
            )
            collective = torch.tensor(1.0, device="cuda:0", dtype=torch.float32)
            dist.all_reduce(collective)
            torch.cuda.synchronize()
            collective_value = float(collective.item())
            if collective_value != 1.0:
                raise RuntimeError(
                    f"one-rank NCCL all-reduce returned {collective_value}, expected 1.0"
                )
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    if dist.is_initialized():
        raise RuntimeError("NCCL process group remained initialized after destroy")

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    record = {
        "schema_version": 1,
        "status": "PASSED",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "gpu_name": gpu_name,
        "bf16_supported": True,
        "bf16_forward_backward": {
            "matrix_size": args.matrix_size,
            "dtype": str(product.dtype),
            "gradient_dtype": str(left.grad.dtype),
            "finite": True,
        },
        "nccl": {
            "backend": "nccl",
            "world_size": world_size,
            "all_reduce": collective_value,
            "destroyed": True,
        },
        "gpu_memory": {"free_bytes": free_bytes, "total_bytes": total_bytes},
        "ecc_health": health,
        "nvidia_smi_q_sha256": hashlib.sha256(smi.stdout.encode()).hexdigest(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if not output.is_absolute():
            raise RuntimeError("--output must resolve to an absolute path")
        _atomic_json(output, record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
