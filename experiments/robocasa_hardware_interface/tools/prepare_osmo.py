"""Package committed RoboCasa study code for an owned L40S worker."""

import argparse
import base64
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import yaml


def prepare(destination, stage, pool="groot-l40s-01"):
    platform = {"groot-l40s-01": "ovx-l40s", "groot-l40-05": "ovx-l40"}[pool]
    root = Path(__file__).resolve().parents[3]
    paths = [
        "astra_reversal/__init__.py",
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "experiments/robocasa_hardware_interface",
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
            if (root / relative).is_symlink():
                raise ValueError("payload_symlink")
            archive.add(root / relative, arcname=relative, recursive=False)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    # Ship immutable source with the submission. Interactive rsync is not a
    # dependable bootstrap dependency on all OSMO backends.
    payload_text = destination / "payload.tar.gz.b64"
    payload_text.write_bytes(base64.b64encode(payload.read_bytes()))
    if payload_text.stat().st_size > 950_000:
        raise ValueError("source_payload_exceeds_file_injection_limit")
    study = root / "experiments/robocasa_hardware_interface"
    manifest = yaml.safe_load((study / "preregistration.yaml").read_text())
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes((study / "tools/bootstrap.sh").read_bytes())
    environment = {
        "PAYLOAD_SHA256": sha,
        "HARDWARE_SOURCE_COMMIT": head,
        "HARDWARE_WORKFLOW": "{{workflow_id}}",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "HARDWARE_STAGE": stage,
    }
    if stage != "commission":
        environment["HARDWARE_API_KEY_FILE"] = "/osmo/run/workspace/inference-api-key"
    staging_check = stage == "staging-check"
    credentials = (
        {}
        if staging_check
        else {
            "grabber-arc-r2-20260916": {
                "R2_ACCESS_KEY_ID": "r2_access_key_id",
                "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                "R2_ENDPOINT_URL": "r2_endpoint_url",
            }
        }
    )
    if stage != "commission":
        credentials["astra-hardware-inference-20261010"] = "/run/hardware-inference"
    spec = {
        "workflow": {
            "name": "robocasa-hardware-interface-20261009-" + stage,
            "resources": {
                "default": {
                    "cpu": 1 if staging_check else 8,
                    "gpu": 0 if staging_check else 1,
                    "memory": "2Gi" if staging_check else "32Gi",
                    "storage": "4Gi" if staging_check else "200Gi",
                    "platform": platform,
                }
            },
            "timeout": {
                "queue_timeout": "4h",
                **({"exec_timeout": "5m"} if staging_check else {}),
            },
            "tasks": [
                {
                    "name": "worker0",
                    "image": manifest["container_digest"],
                    "command": ["bash"],
                    "args": ["/tmp/hardware-bootstrap.sh"],
                    "environment": environment,
                    "credentials": credentials,
                    "files": [
                        {
                            "localpath": str(bootstrap),
                            "path": "/tmp/hardware-bootstrap.sh",
                        },
                        {
                            "localpath": str(payload_text),
                            "path": "/tmp/hardware-payload.tar.gz.b64",
                        },
                    ],
                }
            ],
        }
    }
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt = {
        "source_commit": head,
        "payload_sha256": sha,
        "pool": pool,
        "platform": platform,
        "stage": stage,
        "image": manifest["container_digest"],
        "pilot_tasks": 50,
        "pilot_trials": 150,
        "resource_caps": None,
        "source_transport": "OSMO file injection; base64 with SHA-256 verification",
        "inference_secret_transport": "OSMO generic secret mounted outside artifacts"
        if stage != "commission"
        else None,
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
        "--stage",
        choices=("staging-check", "commission", "smoke", "pilot"),
        default="pilot",
    )
    parser.add_argument(
        "--pool", choices=("groot-l40s-01", "groot-l40-05"), default="groot-l40s-01"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, args.stage, args.pool)))
