"""Package pinned experiment sources plus an explicit scheduling-only overlay."""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import yaml

from .prepare_osmo import prepare


def prepare_shard(
    destination,
    plan_path,
    index,
    *,
    adaptive=False,
    manifest_name=None,
    inference_credential="astra-hardware-inference-20261010",
    label=None,
):
    root = Path(__file__).resolve().parents[3]
    plan = json.loads(Path(plan_path).read_text())
    assignment = plan["shards"][index]
    if assignment["index"] != index or not assignment["trial_ids"]:
        raise ValueError("empty_or_invalid_shard")
    receipt = prepare(
        destination,
        stage="pilot",
        pool="groot-l40s-01",
        adaptive=adaptive,
        manifest_name=manifest_name,
        inference_credential=inference_credential,
    )
    destination = Path(destination)
    orchestration = receipt["source_commit"]
    source = plan["experiment_source_commit"]
    paths = [
        "astra_reversal/__init__.py",
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "experiments/robocasa_hardware_interface",
        "tests/unit/hardware_interface",
    ]
    extra = "experiments/liberomax_hardware_interface/tools/prepare_plus_assets.py"
    if (
        subprocess.run(
            ["git", "cat-file", "-e", source + ":" + extra],
            cwd=root,
            capture_output=True,
        ).returncode
        == 0
    ):
        paths.append(extra)
    raw = subprocess.check_output(["git", "archive", source, "--", *paths], cwd=root)
    overlays = [
        "experiments/libero_hardware_interface/tools/parallel_plan.py",
        "experiments/libero_hardware_interface/tools/parallel_worker.py",
        "experiments/libero_hardware_interface/tools/request_admission.py",
        "tests/unit/hardware_interface/test_parallel.py",
        "tests/unit/hardware_interface/test_request_admission.py",
    ]
    payload = destination / "payload.tar.gz"
    with (
        tarfile.open(fileobj=io.BytesIO(raw)) as base,
        tarfile.open(payload, "w:gz") as out,
    ):
        for member in base.getmembers():
            if not member.isfile() or member.name in overlays:
                continue
            if any(p in ("figures", "evidence") for p in Path(member.name).parts):
                continue
            out.addfile(member, base.extractfile(member))
        for relative in overlays:
            out.add(root / relative, arcname=relative, recursive=False)
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    encoded = destination / "payload.tar.gz.b64"
    encoded.write_bytes(base64.b64encode(payload.read_bytes()))
    if encoded.stat().st_size > 950000:
        raise ValueError("source_payload_exceeds_file_injection_limit")
    bootstrap = subprocess.check_output(
        [
            "git",
            "show",
            source + ":experiments/libero_hardware_interface/tools/bootstrap.sh",
        ],
        cwd=root,
    ).decode()
    original_entry = "python -m astra_reversal.hardware_interface.osmo_worker"
    if bootstrap.count(original_entry) != 1:
        raise ValueError("unknown_bootstrap_entry")
    (destination / "bootstrap.sh").write_text(
        bootstrap.replace(
            original_entry,
            "python -m experiments.libero_hardware_interface.tools.parallel_worker",
        )
    )
    plan_file = destination / "execution-plan.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    spec_path = destination / "workflow.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    label = label or ("adaptive" if adaptive else "original")
    spec["workflow"]["name"] = (
        "libero-hardware-interface-20261008-" + label + "-s" + str(index)
    )
    task = spec["workflow"]["tasks"][0]
    task["environment"].update(
        {
            "PAYLOAD_SHA256": sha,
            "HARDWARE_SOURCE_COMMIT": source,
            "HARDWARE_ORCHESTRATION_COMMIT": orchestration,
            "HARDWARE_EXECUTION_PLAN": "/tmp/hardware-execution-plan.json",
            "HARDWARE_SHARD_INDEX": str(index),
            "HARDWARE_REQUEST_SLOT_INDEX": str(2 * index + (0 if adaptive else 1)),
        }
    )
    task["files"].append(
        {
            "localpath": str(plan_file.resolve()),
            "path": "/tmp/hardware-execution-plan.json",
        }
    )
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt.update(
        {
            "source_commit": source,
            "orchestration_commit": orchestration,
            "payload_sha256": sha,
            "parent_workflow": plan["parent_workflow"],
            "execution_plan_sha256": hashlib.sha256(plan_file.read_bytes()).hexdigest(),
            "experiment_source_sha256": plan["experiment_source_sha256"],
            "shard_index": index,
            "trial_ids": assignment["trial_ids"],
            "scope": "Native and live/replay checks, then %d disjoint trials from the unchanged parent cohort"
            % len(assignment["trial_ids"]),
        }
    )
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("plan")
    parser.add_argument("index", type=int)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_shard(
                args.destination, args.plan, args.index, adaptive=args.adaptive
            )
        )
    )
