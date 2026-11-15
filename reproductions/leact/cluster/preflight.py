"""Fail before training when the pinned runtime or H100 allocation is invalid."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.gpus <= 8:
        raise ValueError("Only one H100 node with up to 8 GPUs is authorized here")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The dependency lock targets Python 3.12")
    shared_memory = os.statvfs("/dev/shm")
    shared_memory_free = shared_memory.f_bavail * shared_memory.f_frsize
    if shared_memory_free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free /dev/shm is required for 8-worker GPU IPC")

    import torch
    import pyspiel
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    from vllm import LLM, SamplingParams
    from llamafactory.train.sft.workflow import run_sft

    del AutoTokenizer, Qwen3ForCausalLM, LLM, SamplingParams, run_sft
    expected = {
        "torch": "2.8.0", "transformers": "4.55.4", "vllm": "0.10.2",
        "llamafactory": "0.9.4", "accelerate": "1.10.1", "numpy": "2.2.6",
        "open-spiel": "2.0.2", "tokenizers": "0.21.4",
    }
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in versions.items():
        if version.split("+")[0] != expected[name]:
            raise RuntimeError(f"Unexpected {name} version: {version}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != args.gpus:
        raise RuntimeError(f"Expected {args.gpus} visible CUDA GPUs")
    devices = []
    for index in range(args.gpus):
        props = torch.cuda.get_device_properties(index)
        if "H100" not in props.name or props.major != 9:
            raise RuntimeError(f"Expected H100, found {props.name}")
        if props.total_memory < 75 * 1024**3:
            raise RuntimeError("An H100 has less than 75 GiB usable device memory")
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bf16 is unavailable")
            matrix = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
            assert (matrix @ matrix).float().mean().item() == 64
        devices.append({"index": index, "name": props.name,
                        "total_memory": props.total_memory,
                        "compute_capability": [props.major, props.minor]})
    game = pyspiel.load_game("leduc_poker")
    if game.num_players() != 2:
        raise RuntimeError("OpenSpiel Leduc import failed")
    metadata = {
        "python": sys.version, "platform": platform.platform(),
        "versions": versions, "cuda": torch.version.cuda,
        "image": os.environ.get("LEACT_IMAGE"), "devices": devices,
        "workflow_id": os.environ.get("LEACT_WORKFLOW_ID"),
        "payload_sha256": os.environ.get("LEACT_PAYLOAD_SHA256"),
        "shared_memory_free": shared_memory_free,
    }
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
