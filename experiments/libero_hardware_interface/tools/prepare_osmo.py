"""Package only committed study code and prepare a single owned L40S check."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import yaml


def prepare(destination, *, stage="commission"):
    if stage not in ("commission", "smoke"):
        raise ValueError("unknown_hardware_stage")
    root = Path(__file__).resolve().parents[3]
    study = root / "experiments/libero_hardware_interface"
    paths = [
        "astra_reversal/__init__.py",
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "tests/unit/hardware_interface",
    ]
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=root, text=True
    )
    if changed.strip():
        raise ValueError("Commit tested sources before packaging")
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
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("Regular study source files required")
            archive.add(path, arcname=relative, recursive=False)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    image = json.loads((study / "locks/container.json").read_text())["image"]
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes((study / "tools/bootstrap.sh").read_bytes())
    spec = {
        "workflow": {
            "name": "libero-hardware-interface-20261008-check",
            "resources": {
                "default": {
                    "cpu": 4,
                    "gpu": 1,
                    "memory": "16Gi",
                    "storage": "40Gi",
                    "platform": "ovx-l40s",
                }
            },
            "timeout": {"queue_timeout": "4h", "exec_timeout": "2h"},
            "tasks": [
                {
                    "name": "worker0",
                    "image": image,
                    "command": ["bash"],
                    "args": ["/tmp/hardware-bootstrap.sh"],
                    "environment": {
                        "PAYLOAD_SHA256": sha,
                        "HARDWARE_SOURCE_COMMIT": head,
                        "HARDWARE_WORKFLOW": "{{workflow_id}}",
                        "NVIDIA_DRIVER_CAPABILITIES": "all",
                        "HARDWARE_STAGE": stage,
                    },
                    "credentials": {
                        "grabber-arc-r2-20260916": {
                            "R2_ACCESS_KEY_ID": "r2_access_key_id",
                            "R2_SECRET_ACCESS_KEY": "r2_secret_access_key",
                            "R2_ENDPOINT_URL": "r2_endpoint_url",
                        }
                    },
                    "files": [
                        {
                            "localpath": str(bootstrap),
                            "path": "/tmp/hardware-bootstrap.sh",
                        }
                    ],
                }
            ],
        }
    }
    if stage == "smoke":
        spec["workflow"]["tasks"][0]["environment"]["HARDWARE_API_KEY_FILE"] = (
            "/osmo/run/workspace/inference-api-key"
        )
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt = {
        "source_commit": head,
        "payload_sha256": sha,
        "image": image,
        "pool": "groot-l40s-01",
        "workflow": str(destination / "workflow.yaml"),
        "scope": "unscored Linux simulator/interface/isolation commissioning"
        + (
            " plus live Astra transport and three-arm smoke"
            if stage == "smoke"
            else "; zero model requests"
        ),
        "stage": stage,
    }
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument(
        "--stage", choices=("commission", "smoke"), default="commission"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, stage=args.stage)))
