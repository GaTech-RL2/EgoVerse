"""Prepare the pinned H100 server and gated 250-trial evaluator OSMO group."""

import argparse
import json
from pathlib import Path

import yaml

from astra_reversal.hardware_interface.replication import VLLM_IMAGE

from .prepare_osmo import prepare


def prepare_group(destination):
    destination = Path(destination).resolve()
    receipt = prepare(
        destination,
        stage="commission",
        manifest_name="preregistration-deepseek-selfhost-commission.yaml",
    )
    spec_path = destination / "workflow.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    workflow = spec["workflow"]
    evaluation = workflow.pop("tasks")[0]
    evaluation.update(name="evaluation", lead=True, resource="evaluation")
    evaluation["environment"]["HARDWARE_INFERENCE_HOST"] = "{{host:inference}}"
    # No hosted-model credential is mounted. Bootstrap stays in commission mode
    # until the coordinator verifies serving identity and enables policy gates.
    bootstrap = destination / "bootstrap.sh"
    text = bootstrap.read_text()
    entry = "python -m astra_reversal.hardware_interface.osmo_worker"
    if text.count(entry) != 1:
        raise ValueError("unknown_bootstrap_entry")
    bootstrap.write_text(
        text.replace(
            entry,
            "python -m experiments.libero_hardware_interface.tools.self_hosted_campaign",
        )
    )
    server_bootstrap = destination / "server-bootstrap.sh"
    server_bootstrap.write_text("""#!/usr/bin/env bash
set -Eeuo pipefail
set +x
mkdir -p /workspace/deepseek-serving /tmp/serving-artifacts
cd /workspace/deepseek-serving
base64 --decode /tmp/hardware-payload.tar.gz.b64 > /tmp/serving-payload.tar.gz
echo "$PAYLOAD_SHA256  /tmp/serving-payload.tar.gz" | sha256sum --check --status
tar xzf /tmp/serving-payload.tar.gz
ulimit -l unlimited 2>/dev/null || true
python -m experiments.libero_hardware_interface.tools.self_hosted_server
""")
    inference = {
        "name": "inference",
        "resource": "inference",
        "image": VLLM_IMAGE,
        "command": ["bash"],
        "args": ["/tmp/server-bootstrap.sh"],
        "environment": {
            "PAYLOAD_SHA256": receipt["payload_sha256"],
            "HARDWARE_WORKFLOW": "{{workflow_id}}",
            "HARDWARE_SERVING_CACHE": "/mnt/amlfs-01/shared/egoverse/rpunamiya/deepseek-v41-1017-20261019",
            "HARDWARE_SERVER_ARTIFACTS": "/tmp/serving-artifacts",
            "PYTHONUNBUFFERED": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "OMP_NUM_THREADS": "8",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "VLLM_CACHE_ROOT": "/mnt/amlfs-01/shared/egoverse/rpunamiya/deepseek-v41-1017-20261019/vllm-cache",
        },
        "files": [
            {"localpath": str(server_bootstrap), "path": "/tmp/server-bootstrap.sh"},
            {
                "localpath": str(destination / "payload.tar.gz.b64"),
                "path": "/tmp/hardware-payload.tar.gz.b64",
            },
        ],
    }
    workflow.update(
        name="libero-hardware-interface-20261008-deepseek-1017-h100",
        resources={
            "inference": {
                "gpu": 8,
                "cpu": 64,
                "memory": "768Gi",
                "storage": "60Gi",
                "platform": "dgx-h100",
            },
            "evaluation": {
                "gpu": 1,
                "cpu": 24,
                "memory": "64Gi",
                "storage": "40Gi",
                "platform": "dgx-h100",
            },
        },
        timeout={"queue_timeout": "4h"},
        groups=[
            {
                "name": "deepseek-study",
                "ignoreNonleadStatus": False,
                "tasks": [inference, evaluation],
            }
        ],
    )
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    receipt.update(
        pool="groot-h100-02",
        platform="dgx-h100",
        stage="self_hosted_pilot",
        scope="Pinned 8xH100 TP8 server plus separate 1xH100 EGL evaluator; native, multimodal/tool and independent replay gates, then 150 + 100 matched trials",
        serving_image=VLLM_IMAGE,
        hosted_api_credentials=False,
        planned_trials=250,
        inference_gpus=8,
        evaluator_gpus=1,
    )
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    print(json.dumps(prepare_group(parser.parse_args().destination)))
