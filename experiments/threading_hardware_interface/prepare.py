"""Package committed Threading commissioning or a standalone H100 serving smoke."""

import argparse
import json
from pathlib import Path

import yaml

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
    )
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    (destination / "launch-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("native", "serving-smoke"))
    parser.add_argument("destination")
    args = parser.parse_args()
    print(
        json.dumps(
            (prepare_native if args.stage == "native" else prepare_serving_smoke)(
                args.destination
            )
        )
    )
