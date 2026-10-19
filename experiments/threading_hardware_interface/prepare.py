"""Package committed Threading commissioning or a standalone H100 serving smoke."""

import argparse
import json
from pathlib import Path

import yaml

from astra_reversal.hardware_interface.common import file_hash
from astra_reversal.hardware_interface.replication import VLLM_IMAGE
from astra_reversal.hardware_interface.threading_protocol import ARMS, STUDY
from experiments.libero_hardware_interface.tools.prepare_osmo import prepare
from experiments.libero_hardware_interface.tools.prepare_self_hosted import (
    prepare_group,
)


def prepare_native(destination):
    destination = Path(destination).resolve()
    receipt = prepare(destination)
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes(Path(__file__).with_name("bootstrap.sh").read_bytes())
    path = destination / "workflow.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["workflow"]["name"] = "threading-hardware-20261019-native"
    spec["workflow"]["timeout"].pop("exec_timeout", None)
    task = spec["workflow"]["tasks"][0]
    task["exitActions"] = {"COMPLETE": "0", "FAIL": "1-9999"}
    task["environment"].pop("HARDWARE_MANIFEST", None)
    receipt.update(
        scope="Native Threading_D0 renderer, ten paired reset seeds, F-adaptive CAP/direct routing and fresh-process replay; zero model calls",
        planned_trials=0,
        conditions=list(ARMS),
        manifest_sha256=file_hash(STUDY / "preregistration.yaml"),
    )
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def prepare_serving_smoke(destination):
    destination = Path(destination).resolve()
    receipt = prepare_group(destination)
    path = destination / "workflow.yaml"
    spec = yaml.safe_load(path.read_text())
    workflow = spec["workflow"]
    inference = workflow.pop("groups")[0]["tasks"][0]
    inference["lead"] = True
    inference["environment"]["HARDWARE_SERVING_SMOKE_ONLY"] = "1"
    inference["exitActions"] = {"COMPLETE": "0", "FAIL": "1-9999"}
    workflow["name"] = "threading-hardware-20261019-serving-smoke"
    workflow["tasks"] = [inference]
    workflow["resources"].pop("evaluation")
    bootstrap = destination / "server-bootstrap.sh"
    text = bootstrap.read_text()
    text = text.replace(
        "python3 -m experiments",
        'trap \'receipt_dir="$HARDWARE_SERVING_CACHE/receipts/$HARDWARE_WORKFLOW"; mkdir -p "$receipt_dir"; cp -a /tmp/serving-artifacts/. "$receipt_dir/"\' EXIT\npython3 -m experiments',
    )
    bootstrap.write_text(text)
    receipt.update(
        stage="serving_smoke",
        planned_trials=0,
        inference_gpus=8,
        evaluator_gpus=0,
        scope="Exact pinned DeepSeek startup and image/tool/history transport smoke only; release node afterwards; preserve weights and receipts on own shared cache",
        conditions=list(ARMS),
        image=receipt["serving_image"],
        manifest_sha256=file_hash(STUDY / "preregistration.yaml"),
    )
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def prepare_pilot(destination):
    destination = Path(destination).resolve()
    receipt = prepare_group(destination)
    path = destination / "workflow.yaml"
    spec = yaml.safe_load(path.read_text())
    workflow = spec["workflow"]
    inference, evaluation = workflow["groups"][0]["tasks"]
    workflow["name"] = "threading-hardware-20261019-pilot"
    workflow["groups"][0]["name"] = "threading-study"
    bootstrap = destination / "bootstrap.sh"
    bootstrap.write_bytes(Path(__file__).with_name("bootstrap.sh").read_bytes())
    for task in (inference, evaluation):
        task["exitActions"] = {"COMPLETE": "0", "FAIL": "1-9999"}
    evaluation["environment"].update(
        HARDWARE_STAGE="pilot",
        HARDWARE_API_KEY_FILE="/osmo/run/workspace/inference-api-key",
    )
    evaluation["environment"].pop("HARDWARE_MANIFEST", None)
    evaluation["credentials"]["astra-hardware-inference-20261010"] = (
        "/run/hardware-inference"
    )
    receipt.update(
        stage="threading_pilot",
        planned_trials=40,
        conditions=list(ARMS),
        manifest_sha256=file_hash(STUDY / "preregistration.yaml"),
        hosted_api_credentials=True,
        scope="Four-cell F-adaptive Threading pilot, ten matched starts, native and model transport gates, eight balanced-block workers and independent replay; no LIBERO trials",
    )
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def prepare_colocated(destination, *, native_only=False):
    destination = Path(destination).resolve()
    receipt = prepare_group(destination)
    path = destination / "workflow.yaml"
    spec = yaml.safe_load(path.read_text())
    workflow = spec["workflow"]
    inference, evaluation = workflow.pop("groups")[0]["tasks"]
    bootstrap = destination / "bootstrap-colocated.sh"
    bootstrap.write_bytes(
        Path(__file__).with_name("bootstrap-colocated.sh").read_bytes()
    )
    environment = {**evaluation["environment"], **inference["environment"]}
    environment.pop("HARDWARE_MANIFEST", None)
    environment.update(
        HARDWARE_STAGE="commission" if native_only else "pilot",
        HARDWARE_INFERENCE_HOST="{{host:study}}",
        HARDWARE_SERVER_IMAGE=VLLM_IMAGE,
    )
    credentials = evaluation["credentials"]
    if not native_only:
        credentials["astra-hardware-inference-20261010"] = "/run/hardware-inference"
        environment["HARDWARE_API_KEY_FILE"] = "/osmo/run/workspace/inference-api-key"
    workflow.update(
        name="threading-hardware-20261019-colocated-"
        + ("native" if native_only else "pilot"),
        resources={
            "study": {
                "gpu": 8,
                "cpu": 88,
                "memory": "896Gi",
                "storage": "96Gi",
                "platform": "dgx-h100",
            }
        },
        tasks=[
            {
                "name": "study",
                "resource": "study",
                "image": VLLM_IMAGE,
                "command": ["bash"],
                "args": ["/tmp/colocated-bootstrap.sh"],
                "environment": environment,
                "credentials": credentials,
                "exitActions": {"COMPLETE": "0", "FAIL": "1-9999"},
                "files": [
                    {
                        "localpath": str(bootstrap),
                        "path": "/tmp/colocated-bootstrap.sh",
                    },
                    {
                        "localpath": str(destination / "payload.tar.gz.b64"),
                        "path": "/tmp/hardware-payload.tar.gz.b64",
                    },
                ],
            }
        ],
    )
    receipt.update(
        stage="colocated_native" if native_only else "colocated_pilot",
        planned_trials=0 if native_only else 40,
        conditions=list(ARMS),
        allocated_gpus=8,
        inference_gpus=0 if native_only else 8,
        evaluator_gpus=0,
        evaluator_backend="osmesa",
        native_runtime="Original pinned Python 3.8 image unpacked in a separate chroot",
        hosted_api_credentials=not native_only,
        hosted_key_provided_to_vllm=False,
        persistent_artifacts="Own shared cache /colocated/<workflow_id>/; five-second evaluator snapshots and live serving logs",
        manifest_sha256=file_hash(STUDY / "preregistration.yaml"),
        scope="Single-node Threading native validation only; zero model calls"
        if native_only
        else "Four-cell F-adaptive pilot, same ten matched starts and model settings, CPU-rendered native simulator alongside TP8 DeepSeek on one allocated H100 node",
    )
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "native",
            "serving-smoke",
            "pilot",
            "colocated-native",
            "colocated-pilot",
        ),
    )
    parser.add_argument("destination")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "native": prepare_native,
                "serving-smoke": prepare_serving_smoke,
                "pilot": prepare_pilot,
                "colocated-native": lambda destination: prepare_colocated(
                    destination, native_only=True
                ),
                "colocated-pilot": prepare_colocated,
            }[args.stage](args.destination)
        )
    )
