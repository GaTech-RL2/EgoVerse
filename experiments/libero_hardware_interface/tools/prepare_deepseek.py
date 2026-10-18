"""Prepare the separate 150 + 100 trial replication; never submit implicitly."""

import argparse
import json
import subprocess
from pathlib import Path

from astra_reversal.hardware_interface.common import file_hash, write_json
from astra_reversal.hardware_interface.preflight import source_hash
from astra_reversal.hardware_interface.protocol import (
    load_manifest,
    save_schedule,
    schedule,
)

from .parallel_plan import create_fresh_plan
from .prepare_parallel_osmo import prepare_shard


def prepare_campaign(destination, catalog_path, *, inference_credential=None):
    root = Path(__file__).resolve().parents[3]
    # Freeze code before deriving source identities or packaging workers.
    paths = [
        "astra_reversal/hardware_interface",
        "experiments/libero_hardware_interface",
        "tests/unit/hardware_interface",
    ]
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=root, text=True
    )
    if changed.strip():
        raise ValueError("commit_tested_sources_before_preparation")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    catalog = json.loads(Path(catalog_path).read_text())
    receipt = {
        "source_commit": source,
        "source_sha256": source_hash(),
        "catalog_sha256": file_hash(catalog_path),
        "status": "prepared_not_submitted"
        if inference_credential
        else "plans_only_awaiting_exact_model_access",
        "serving_endpoint": "https://api.deepseek.com/v1",
        "requested_model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "inference_credential_name": inference_credential,
        "cohorts": {},
        "launch_gates": [
            "Reverify official API alias maps to 1017 on launch date",
            "Verify configured secret authenticates to the declared DeepSeek endpoint",
            "Native reset/interface/isolation gates, live multimodal tool/observer transport and disjoint replay smoke must pass before each shard collects trials",
        ],
    }
    for label, suffix in (("original", ""), ("adaptive", "-adaptive")):
        manifest_name = "preregistration-deepseek" + suffix + ".yaml"
        manifest_path = root / "experiments/libero_hardware_interface" / manifest_name
        manifest = load_manifest(manifest_path)
        rows = schedule(manifest, catalog)
        out = destination / label
        out.mkdir()
        save_schedule(out / "schedule.csv", rows)
        name = "libero-hardware-interface-20261008-deepseek-1017-" + label
        plan = create_fresh_plan(
            manifest, rows, source=source, source_sha256=source_hash(), cohort=name
        )
        plan_path = out / "execution-plan.json"
        write_json(plan_path, plan)
        launches = []
        if inference_credential:
            for index in range(4):
                launches.append(
                    prepare_shard(
                        out / ("worker-%d" % index),
                        plan_path,
                        index,
                        adaptive=bool(suffix),
                        manifest_name=manifest_name,
                        inference_credential=inference_credential,
                        label="deepseek-1017-" + label,
                    )
                )
        receipt["cohorts"][label] = {
            "manifest_sha256": file_hash(manifest_path),
            "plan": str(plan_path),
            "planned_trials": sum(r["split"] == "pilot" for r in rows),
            "planned_gpus": 4,
            "trial_counts_per_worker": [len(s["trial_ids"]) for s in plan["shards"]],
            "launches": launches,
        }
    write_json(destination / "campaign.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    parser.add_argument("--catalog", required=True)
    parser.add_argument(
        "--inference-credential",
        help="OSMO generic credential containing inference_api_key for api.deepseek.com; omit for plans only",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_campaign(
                args.destination,
                args.catalog,
                inference_credential=args.inference_credential,
            )
        )
    )
