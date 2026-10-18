"""Package only committed study code and prepare a single owned L40S check."""

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml


def prepare(
    destination,
    *,
    stage="commission",
    pool="groot-l40s-01",
    adaptive=False,
    manifest_name=None,
    inference_credential="astra-hardware-inference-20261010",
):
    if stage not in ("staging-check", "commission", "smoke", "pilot"):
        raise ValueError("unknown_hardware_stage")
    platform = {"groot-l40s-01": "ovx-l40s", "groot-l40-05": "ovx-l40"}[pool]
    root = Path(__file__).resolve().parents[3]
    study = root / "experiments/libero_hardware_interface"
    paths = [
        "astra_reversal/__init__.py",
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "experiments/robocasa_hardware_interface",
        "experiments/liberomax_hardware_interface/tools/prepare_plus_assets.py",
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
            if any(part in ("figures", "evidence") for part in Path(relative).parts):
                continue
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("Regular study source files required")
            archive.add(path, arcname=relative, recursive=False)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    payload_text = destination / "payload.tar.gz.b64"
    payload_text.write_bytes(base64.b64encode(payload.read_bytes()))
    if payload_text.stat().st_size > 950_000:
        raise ValueError("source_payload_exceeds_file_injection_limit")
    image = json.loads((study / "locks/container.json").read_text())["image"]
    manifest_name = manifest_name or (
        "preregistration-adaptive.yaml" if adaptive else "preregistration.yaml"
    )
    if Path(manifest_name).name != manifest_name:
        raise ValueError("study_manifest_filename_required")
    manifest = yaml.safe_load((study / manifest_name).read_text())
    if (
        manifest.get("model_replication")
        and stage != "commission"
        and (
            not inference_credential
            or inference_credential == "astra-hardware-inference-20261010"
        )
    ):
        raise ValueError("explicit_deepseek_credential_required")
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes((study / "tools/bootstrap.sh").read_bytes())
    spec = {
        "workflow": {
            "name": "libero-hardware-interface-20261008-check",
            "resources": {
                "default": {
                    "cpu": 1 if stage == "staging-check" else 4,
                    "gpu": 0 if stage == "staging-check" else 1,
                    "memory": "2Gi" if stage == "staging-check" else "16Gi",
                    "storage": "4Gi" if stage == "staging-check" else "40Gi",
                    "platform": platform,
                }
            },
            "timeout": {
                "queue_timeout": "4h",
                "exec_timeout": "5m" if stage == "staging-check" else "2h",
            },
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
                        "HARDWARE_MANIFEST": "experiments/libero_hardware_interface/"
                        + manifest_name,
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
    if stage == "staging-check":
        spec["workflow"]["tasks"][0]["credentials"] = {}
        spec["workflow"]["name"] = "libero-hardware-interface-20261008-staging-check"
    if stage in ("staging-check", "smoke", "pilot"):
        spec["workflow"]["tasks"][0]["environment"]["HARDWARE_API_KEY_FILE"] = (
            "/osmo/run/workspace/inference-api-key"
        )
        spec["workflow"]["tasks"][0]["credentials"][inference_credential] = (
            "/run/hardware-inference"
        )
    if stage == "pilot":
        sys.path.insert(0, str(root))
        from astra_reversal.hardware_interface.protocol import RESOURCE_CAPS, validate

        validate(manifest)
        if any(manifest["limits"][key] is not None for key in RESOURCE_CAPS):
            raise ValueError("revised_pilot_requires_no_resource_budgets")
        spec["workflow"]["name"] = "libero-hardware-interface-20261008-uncapped"
        # No experiment runtime deadline. Any platform ceiling is infrastructure.
        spec["workflow"]["timeout"].pop("exec_timeout")
    if adaptive:
        spec["workflow"]["name"] = (
            "libero-hardware-interface-20261008-adaptive-" + stage
        )
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt = {
        "source_commit": head,
        "payload_sha256": sha,
        "image": image,
        "pool": pool,
        "platform": platform,
        "source_transport": "OSMO file injection; base64 with SHA-256 verification",
        "inference_secret_transport": "OSMO generic secret copied to private regular file"
        if stage != "commission"
        else None,
        "workflow": str(destination / "workflow.yaml"),
        "scope": "unscored Linux simulator/interface/isolation commissioning"
        + (
            " plus live model transport and %d-arm smoke" % len(manifest["conditions"])
            if stage in ("smoke", "pilot")
            else "; zero model requests"
        ),
        "stage": stage,
        "conditions": manifest["conditions"],
        "inference_credential_name": inference_credential
        if stage != "commission"
        else None,
        "manifest_sha256": hashlib.sha256(
            (study / manifest_name).read_bytes()
        ).hexdigest(),
    }
    if stage == "staging-check":
        receipt["scope"] = (
            "source and inference credential delivery only; zero simulator/model calls"
        )
    if stage == "pilot":
        count = (
            len(manifest["conditions"])
            * len(manifest["task_ids"])
            * len(manifest["pilot_indices"])
            * manifest["replicates"]
        )
        receipt["scope"] += (
            " then a separate %d-trial pilot without resource budgets" % count
        )
        receipt["limits"] = manifest["limits"]
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
        default="commission",
    )
    parser.add_argument(
        "--pool", choices=("groot-l40s-01", "groot-l40-05"), default="groot-l40s-01"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Separate matched F/F-adaptive cohort"
    )
    parser.add_argument("--manifest-name")
    parser.add_argument(
        "--inference-credential", default="astra-hardware-inference-20261010"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.destination,
                stage=args.stage,
                pool=args.pool,
                adaptive=args.adaptive,
                manifest_name=args.manifest_name,
                inference_credential=args.inference_credential,
            )
        )
    )
