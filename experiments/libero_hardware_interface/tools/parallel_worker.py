"""Run a schedule shard with the exact parent cohort's simulator/actor sources.

Only scheduling functions are projected. Existing runtime checks, trial execution,
prompts, actions, independent replay and immutable archives run unchanged.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from astra_reversal.hardware_interface import launcher, osmo_worker
from astra_reversal.hardware_interface.analysis import audit
from astra_reversal.hardware_interface.common import strict_json, write_json
from astra_reversal.hardware_interface.preflight import source_hash
from astra_reversal.hardware_interface.protocol import save_schedule, schedule, validate
from astra_reversal.hardware_interface.reporting import export

from .parallel_plan import project, validate_plan


def collect_shard(root, manifest):
    validate(manifest, scored=True)
    plan = strict_json(Path(os.environ["HARDWARE_EXECUTION_PLAN"]).read_bytes())
    rows = schedule(
        manifest, strict_json((root / "prepared/catalog.json").read_bytes())
    )
    validate_plan(plan, manifest, rows)
    if source_hash() != plan["experiment_source_sha256"]:
        raise ValueError("parallel_experiment_source_changed")
    index = int(os.environ["HARDWARE_SHARD_INDEX"])
    assignment = plan["shards"][index]
    selected = set(assignment["trial_ids"])
    if not selected:
        raise ValueError("parallel_empty_assignment")
    write_json(root / "execution-plan.json", plan)
    write_json(
        root / "execution-receipt.json",
        {
            "index": index,
            "trial_ids": assignment["trial_ids"],
            "experiment_source_commit": plan["experiment_source_commit"],
            "experiment_source_sha256": source_hash(),
            "orchestration_commit": os.environ["HARDWARE_ORCHESTRATION_COMMIT"],
            "design_sha256": plan["design_sha256"],
            "schedule_sha256": plan["schedule_sha256"],
            "workflow": os.environ["HARDWARE_WORKFLOW"],
        },
    )
    path = root / "pilot-preregistration.yaml"
    with path.open("x") as stream:
        stream.write(yaml.safe_dump(manifest, sort_keys=False))
    save_schedule(root / "trials.csv", rows)
    args = SimpleNamespace(
        manifest=str(path),
        prepared=str(root / "prepared"),
        libero_root="upstream-libero",
        out=str(root / "pilot"),
        split="pilot",
        schedule=str(root / "trials.csv"),
        archive_trials=True,
    )
    original_schedule, original_rows = launcher.schedule, launcher.load_rows

    def selected_schedule(*args, **kwargs):
        return project(original_schedule(*args, **kwargs), rows, selected)

    def selected_rows(*args, **kwargs):
        return project(original_rows(*args, **kwargs), rows, selected)

    # The parent implementation's full source hashes stay byte-identical. This
    # explicit projection occurs only after validating the full frozen schedule
    # and complete campaign partition. No trial or action code is replaced.
    launcher.schedule, launcher.load_rows = selected_schedule, selected_rows
    print(json.dumps({"shard_started": index, "trials": len(selected)}), flush=True)
    try:
        result = launcher.run(args, manifest)
        if result["trials"] != len(selected):
            raise ValueError("parallel_shard_incomplete")
        write_json(root / "shard-completed.json", result)
    finally:
        launcher.schedule, launcher.load_rows = original_schedule, original_rows
        if (root / "pilot/runs").exists():
            outcomes = launcher.load_outcomes(root / "pilot/runs")
            write_json(root / "pilot-audit.json", audit(root / "pilot/runs"))
            export(outcomes, manifest, root / "results", split="pilot")


if __name__ == "__main__":
    osmo_worker.collect_pilot = collect_shard
    osmo_worker.main()
