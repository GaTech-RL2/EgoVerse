"""Render a private, immutable two-worker OSMO launch; no submission on import."""

import argparse
import json
import secrets
import shlex
import subprocess
from pathlib import Path

import yaml

from astra_reversal.demo_segments import write_json
from astra_reversal.osmo.demo_skill_recovery import validate_manifest
from astra_reversal.records import file_sha256


def prepare(
    bundle, destination, *, pilot=False, port_base=19930, recovery_manifest=None
):
    repo = Path(__file__).resolve().parents[2]
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    identity = json.loads((bundle / "source_identity.json").read_text())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != identity["source_revision"]:
        raise ValueError("Payload must correspond to this committed checkout")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "astra_reversal", "tests/unit/astra"],
        cwd=repo,
        text=True,
    )
    if status:
        raise ValueError(
            "Commit all experiment source changes before preparing a launch"
        )
    payload = bundle / "payload.tar.gz"
    if file_sha256(payload) != identity["payload_sha256"]:
        raise ValueError("Immutable payload checksum changed")
    if not 1024 <= port_base <= 65534:
        raise ValueError("Choose two adjacent unprivileged relay ports")
    spec = yaml.safe_load(
        (repo / "astra_reversal/osmo/demo_skill_library_l40s.yaml").read_text()
    )
    if len(spec["workflow"]["tasks"]) != 2:
        raise ValueError("Expected one worker for each separate experiment arm")
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    if recovery_manifest is not None:
        recovery_manifest = validate_manifest(
            json.loads(Path(recovery_manifest).read_text())
        )
        write_json(destination / "recovery_manifest.json", recovery_manifest)
    spec["workflow"]["name"] += "-pilot" if pilot else "-full"
    workers = []
    for index, task in enumerate(spec["workflow"]["tasks"]):
        if task["name"] != f"worker{index}":
            raise ValueError("Unexpected task identity")
        directory = destination / task["name"]
        directory.mkdir(mode=0o700)
        token = directory / "relay.token"
        with token.open("x") as stream:
            token.chmod(0o600)
            stream.write(secrets.token_urlsafe(32))
        task["environment"].update(
            PAYLOAD_SHA256=identity["payload_sha256"],
            ASTRA_SOURCE_REVISION=revision,
            ASTRA_DEMO_PILOT="1" if pilot else "0",
        )
        for item in task["files"]:
            if item["path"] == "/tmp/astra-relay.token":
                item["localpath"] = str(token)
            elif item["path"] == "/tmp/astra-bootstrap.sh":
                item["localpath"] = str(bundle / "bootstrap.sh")
            else:
                raise ValueError("Unexpected task upload")
        if file_sha256(bundle / "bootstrap.sh") != identity["bootstrap_sha256"]:
            raise ValueError("Bootstrap differs from the immutable bundle")
        if recovery_manifest is not None:
            manifest_path = destination / "recovery_manifest.json"
            task["files"].append(
                {
                    "localpath": str(manifest_path),
                    "path": "/tmp/astra-demo-recovery.json",
                }
            )
            task["environment"].update(
                ASTRA_DEMO_RECOVERY_FILE="/tmp/astra-demo-recovery.json",
                ASTRA_DEMO_RECOVERY_SHA256=file_sha256(manifest_path),
            )
        workers.append(
            {
                "task": task["name"],
                "port": port_base + index,
                "token_file": str(token),
                "directory": str(directory),
            }
        )
    (destination / "workflow.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    write_json(destination / "source_identity.json", identity)
    write_json(
        destination / "launch_plan.json",
        {
            "pilot": pilot,
            "pool": "groot-l40s-01",
            "workers": workers,
            "payload": str(payload),
            "automatic_model_or_rollout_retries": False,
            "status": "prepared_not_submitted",
            "recovery_manifest": recovery_manifest,
        },
    )
    # One operator-owned terminal per worker. It connects only after OSMO reports
    # RUNNING and never retries a model job or submits another workflow.
    for row in workers:
        task, directory = row["task"], Path(row["directory"])
        script = f"""#!/usr/bin/env bash
set -euo pipefail
test "$#" = 1 || {{ echo "Usage: $0 EXACT_SUBMITTED_WORKFLOW_NAME" >&2; exit 2; }}
workflow="$1"
case "$workflow" in astra-pi05-input-skill-library-20260930-*) ;; *) exit 2;; esac
cd {shlex.quote(str(repo))}
osmo workflow port-forward "$workflow" {task} --port {row["port"]}:8769 --connect-timeout 60 > {shlex.quote(str(directory / "port-forward.log"))} 2>&1 &
forward_pid=$!
trap 'kill "$forward_pid" 2>/dev/null || true' EXIT
osmo workflow rsync "$workflow" {task} {shlex.quote(str(payload) + ":/osmo/run/workspace")} --once --timeout 180
python -m astra_reversal.codex_relay --url http://127.0.0.1:{row["port"]} --token-file {shlex.quote(str(directory / "relay.token"))} --directory {shlex.quote(str(directory / "jobs"))} --idle-timeout 86400
"""
        (directory / "connect.sh").write_text(script)
        (directory / "connect.sh").chmod(0o700)
    return {
        "directory": str(destination),
        "pilot": pilot,
        "status": "prepared_not_submitted",
        "source_revision": revision,
        "payload_sha256": identity["payload_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--port-base", type=int, default=19930)
    parser.add_argument("--recovery-manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.bundle,
                args.destination,
                pilot=args.pilot,
                port_base=args.port_base,
                recovery_manifest=args.recovery_manifest,
            )
        )
    )


if __name__ == "__main__":
    main()
