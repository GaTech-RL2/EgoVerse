"""Prepare an immutable single-L40S pilot, reusing verified public study assets."""

import argparse
import json
import secrets
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import yaml

from astra_reversal.demo_segments import write_json
from astra_reversal.records import file_sha256

BASE_REVISION = "d4b2b690ac8992137f35af566429f6241c39afca"
ASSET_ROOTS = (
    "astra_reversal/.deps/tokenizers/",
    "astra_reversal/.deps/reference/",
    "astra_reversal/.deps/demo-skill-inputs/source_cache/",
)


def prepare(bundle, destination, *, activation, port=19966):
    root = Path(__file__).resolve().parents[2]
    bundle, destination, activation = map(
        lambda p: Path(p).resolve(), (bundle, destination, activation)
    )
    if not activation.is_file() or not 1024 <= port <= 65535:
        raise ValueError(
            "A working emimic activation and an unprivileged port are required"
        )
    identity = json.loads((bundle / "source_identity.json").read_text())
    if identity["source_revision"] != BASE_REVISION:
        raise ValueError("Assets must come from the exact recovered study revision")
    if file_sha256(bundle / "payload.tar.gz") != identity["payload_sha256"]:
        raise ValueError("Prior immutable payload checksum mismatch")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    paths = [
        "astra_reversal",
        "tests/unit/astra",
        "tests/integration/test_astra_lerobot_policy.py",
        "tests/integration/test_astra_interpolation_policy.py",
        "tests/fixtures/astra",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=root, text=True
    ):
        raise ValueError("Commit the tested experiment sources before packaging")
    tracked = (
        subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=root)
        .decode()
        .split("\0")
    )
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    payload = destination / "payload.tar.gz"
    assets = []
    with tarfile.open(payload, "w:gz") as output:
        for name in sorted(filter(None, tracked)):
            path = root / name
            if path.is_symlink():
                raise ValueError("Experiment source must be regular files")
            output.add(path, arcname=name, recursive=False)
        with tarfile.open(bundle / "payload.tar.gz", "r:gz") as previous:
            for member in previous:
                if not any(member.name.startswith(prefix) for prefix in ASSET_ROOTS):
                    continue
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe asset member in prior payload")
                if member.isfile():
                    output.addfile(member, previous.extractfile(member))
                    assets.append(member.name)
    if not any(name.endswith("source_cache/file_index.json") for name in assets):
        raise ValueError("Verified standard source cache is missing")
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes((root / "astra_reversal/osmo/bootstrap.sh").read_bytes())
    token = destination / "relay.token"
    with token.open("x") as stream:
        token.chmod(0o600)
        stream.write(secrets.token_urlsafe(32))
    sha = file_sha256(payload)
    spec = {
        "workflow": {
            "name": "astra-meta-harness-20261006-pilot",
            "resources": {
                "default": {
                    "cpu": 6,
                    "gpu": 1,
                    "memory": "48Gi",
                    "storage": "160Gi",
                    "platform": "ovx-l40s",
                }
            },
            "timeout": {"queue_timeout": "4h", "exec_timeout": "2h"},
            "tasks": [
                {
                    "name": "worker0",
                    "image": "nvcr.io/nvidia/pytorch:25.06-py3",
                    "command": ["bash"],
                    "args": ["/tmp/astra-bootstrap.sh"],
                    "environment": {
                        "PAYLOAD_SHA256": sha,
                        "ASTRA_SOURCE_REVISION": revision,
                        "ASTRA_RUN_ID": "{{workflow_id}}",
                        "ASTRA_ENTRY_MODULE": "astra_reversal.osmo.meta_harness",
                        "ASTRA_WORKER_INDEX": "0",
                        "ASTRA_INCLUDE_OOD": "1",
                        "ASTRA_CODEX_RELAY_PORT": "8769",
                        "ASTRA_CODEX_RELAY_HOST": "127.0.0.1",
                        "ASTRA_CODEX_RELAY_TOKEN_FILE": "/tmp/astra-relay.token",
                        "ASTRA_RETAIN_FAILED_WORKER": "0",
                        "NVIDIA_DRIVER_CAPABILITIES": "all",
                        "NVIDIA_TF32_OVERRIDE": "0",
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
                            "path": "/tmp/astra-bootstrap.sh",
                        },
                        {"localpath": str(token), "path": "/tmp/astra-relay.token"},
                    ],
                }
            ],
        }
    }
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    connect = f"""#!/usr/bin/env bash
set -Eeuo pipefail
source {shlex.quote(str(activation))}
workflow="${{1:?Pass the submitted workflow ID}}"
case "$workflow" in astra-meta-harness-20261006-pilot-*) ;; *) exit 2;; esac
cd {shlex.quote(str(root))}
osmo workflow port-forward "$workflow" worker0 --port {port}:8769 --connect-timeout 60 > {shlex.quote(str(destination / 'port-forward.log'))} 2>&1 &
forward_pid=$!
trap 'kill "$forward_pid" 2>/dev/null || true' EXIT
osmo workflow rsync "$workflow" worker0 {shlex.quote(str(payload) + ':/osmo/run/workspace')} --once --timeout 180
python -m astra_reversal.codex_relay --url http://127.0.0.1:{port} --token-file {shlex.quote(str(token))} --directory {shlex.quote(str(destination / 'jobs'))} --max-requests 32 --idle-timeout 7200
"""
    (destination / "connect.sh").write_text(connect)
    (destination / "connect.sh").chmod(0o700)
    result = {
        "status": "prepared_not_submitted",
        "source_revision": revision,
        "base_assets_revision": BASE_REVISION,
        "payload_sha256": sha,
        "bootstrap_sha256": file_sha256(bootstrap),
        "pool": "groot-l40s-01",
        "gpus": 1,
        "wall_limit_hours": 2,
        "max_physical_episodes": 6,
        "max_primitive_actions": 1800,
        "max_runtime_requests": 32,
        "scope": "latency/interface profile, not main-comparison results",
        "asset_files": len(assets),
        "workflow": str(destination / "workflow.yaml"),
    }
    write_json(destination / "launch_plan.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--activation", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.bundle, args.destination, activation=args.activation))
    )


if __name__ == "__main__":
    main()
