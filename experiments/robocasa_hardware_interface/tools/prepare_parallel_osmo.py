"""Package separate, complete original/adaptive RoboCasa execution plans."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from astra_reversal.hardware_interface.common import file_hash, write_json
from astra_reversal.hardware_interface.preflight import source_hash

from .parallel_plan import assigned_rows, create_plan
from .prepare_osmo import prepare


def prepare_campaign(destination, *, shards=4, pool="groot-l40-05"):
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for adaptive, cohort in [(False, "original"), (True, "adaptive")]:
        name = "preregistration-adaptive.yaml" if adaptive else "preregistration.yaml"
        manifest = yaml.safe_load(
            (root / "experiments/robocasa_hardware_interface" / name).read_text()
        )
        plan = create_plan(manifest, head, source_hash(), shards)
        plan_path = destination / (cohort + "-execution-plan.json")
        write_json(plan_path, plan)
        for index in range(shards):
            rows = assigned_rows(plan, manifest, index, head, source_hash())
            worker = destination / (cohort + "-s" + str(index))
            receipt = prepare(worker, "pilot", pool, adaptive)
            spec_path = worker / "workflow.yaml"
            spec = yaml.safe_load(spec_path.read_text())
            spec["workflow"]["name"] = (
                "robocasa-hardware-interface-20261009-" + cohort + "-s" + str(index)
            )
            task = spec["workflow"]["tasks"][0]
            task["environment"].update(
                {
                    "HARDWARE_EXECUTION_PLAN": "/tmp/hardware-execution-plan.json",
                    "HARDWARE_SHARD_INDEX": str(index),
                    "HARDWARE_REQUEST_SLOT_INDEX": str(2 * index + int(adaptive)),
                    "HARDWARE_REQUEST_PARTICIPANTS": str(2 * shards),
                }
            )
            task["files"].append(
                {
                    "localpath": str(plan_path),
                    "path": "/tmp/hardware-execution-plan.json",
                }
            )
            spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
            receipt.update(
                {
                    "cohort": cohort,
                    "shard_index": index,
                    "cohort_planned_trials": receipt["pilot_trials"],
                    "assigned_trials": len(rows),
                    "trial_ids": [row["trial_id"] for row in rows],
                    "execution_plan_sha256": file_hash(plan_path),
                    "experiment_source_sha256": plan["experiment_source_sha256"],
                    "launch_gate": "All 50 native reset/action checks pass, then each worker must pass its native and live/replay commissioning before policy collection.",
                }
            )
            write_json(worker / "launch-receipt.json", receipt, replace=True)
            receipts.append(receipt)
    write_json(destination / "campaign.json", {"workers": receipts, "submitted": False})
    return {
        "workers": len(receipts),
        "planned_trials": sum(r["assigned_trials"] for r in receipts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument(
        "--pool", choices=("groot-l40-05", "groot-l40s-01"), default="groot-l40-05"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_campaign(args.destination, shards=args.shards, pool=args.pool)
        )
    )
