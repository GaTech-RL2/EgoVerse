"""Prepare isolated Plus/PRO OSMO commissioning tasks; no inference credential."""

import argparse
import base64
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import yaml

IMAGE = "docker.io/library/python@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47"


def prepare(destination, pool="groot-l40-05", stage="commission"):
    if stage not in ("staging-check", "commission"):
        raise ValueError("live_model_launch_requires_completed_commissioning")
    root = Path(__file__).resolve().parents[3]
    paths = [
        "astra_reversal/__init__.py",
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "experiments/robocasa_hardware_interface",
        "experiments/liberomax_hardware_interface",
        "tests/unit/hardware_interface",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=root, text=True
    ).strip():
        raise ValueError("commit_tested_sources_before_packaging")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    files = (
        subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=root)
        .decode()
        .split("\0")
    )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    payload = destination / "payload.tar.gz"
    with tarfile.open(payload, "w:gz") as archive:
        for relative in sorted(filter(None, files)):
            if any(part in ("figures", "evidence") for part in Path(relative).parts):
                continue
            if (root / relative).is_symlink():
                raise ValueError("payload_symlink")
            archive.add(root / relative, arcname=relative, recursive=False)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    encoded = destination / "payload.tar.gz.b64"
    encoded.write_bytes(base64.b64encode(payload.read_bytes()))
    if encoded.stat().st_size > 950_000:
        raise ValueError("source_payload_exceeds_file_injection_limit")
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes(
        (
            root / "experiments/liberomax_hardware_interface/tools/bootstrap.sh"
        ).read_bytes()
    )
    tasks = []
    sources = ("plus",) if stage == "staging-check" else ("plus", "pro")
    for source in sources:
        task = {
            "name": source,
            "image": IMAGE,
            "command": ["bash"],
            "args": ["/tmp/hardware-bootstrap.sh"],
            "environment": {
                "PAYLOAD_SHA256": sha,
                "HARDWARE_SOURCE_COMMIT": head,
                "HARDWARE_WORKFLOW": "{{workflow_id}}",
                "HARDWARE_STAGE": stage,
                "MAX_SOURCE": source,
                "NVIDIA_DRIVER_CAPABILITIES": "all",
            },
            "files": [
                {"localpath": str(bootstrap), "path": "/tmp/hardware-bootstrap.sh"},
                {"localpath": str(encoded), "path": "/tmp/hardware-payload.tar.gz.b64"},
            ],
        }
        if stage == "commission":
            task["credentials"] = {
                "grabber-arc-r2-20260916": {
                    "R2_ACCESS_KEY_ID": "r2_access_key_id",
                    "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                    "R2_ENDPOINT_URL": "r2_endpoint_url",
                }
            }
        tasks.append(task)
    spec = {
        "workflow": {
            "name": "liberomax-hardware-interface-20261012-" + stage,
            "resources": {
                "default": {
                    "cpu": 1 if stage == "staging-check" else 4,
                    "gpu": 0 if stage == "staging-check" else 1,
                    "memory": "2Gi" if stage == "staging-check" else "16Gi",
                    "storage": "4Gi" if stage == "staging-check" else "80Gi",
                    "platform": {
                        "groot-l40-05": "ovx-l40",
                        "groot-l40s-01": "ovx-l40s",
                    }[pool],
                }
            },
            "timeout": {"queue_timeout": "4h"},
            "tasks": tasks,
        }
    }
    if stage == "staging-check":
        spec["workflow"]["timeout"]["exec_timeout"] = "5m"
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt = {
        "source_commit": head,
        "payload_sha256": sha,
        "image": IMAGE,
        "pool": pool,
        "stage": stage,
        "sources": list(sources),
        "model_calls": 0,
        "scored": False,
        "live_astra_ready": False,
        "workflow_file": str(destination / "workflow.yaml"),
    }
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument(
        "--pool", choices=("groot-l40-05", "groot-l40s-01"), default="groot-l40-05"
    )
    parser.add_argument(
        "--stage", choices=("staging-check", "commission"), default="commission"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.pool, args.stage)))
