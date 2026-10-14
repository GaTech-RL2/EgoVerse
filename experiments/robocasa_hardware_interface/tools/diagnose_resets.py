"""Test RoboCasa reset/replay determinism in fresh simulator processes.

This is native commissioning, with no Astra calls and no scored episodes. It
retains full metadata, model XML and state arrays for each probe; no comparison
uses a numerical tolerance or discards a differing field.
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

from astra_reversal.hardware_interface.common import Events, digest, write_json
from astra_reversal.hardware_interface.robocasa import (
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    Environment,
    check_source,
)
from astra_reversal.hardware_interface.robocasa_compat import verify_fixture_order_fix
from astra_reversal.hardware_interface.robocasa_protocol import limits_for, schedule
from astra_reversal.hardware_interface.robocasa_worker import (
    CASA,
    ROOT,
    STUDY,
    SUITE,
    archive_tree,
    make_proxy,
    record_reset_evidence,
)


def probe(request):
    row = request["row"]
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    events = Events(directory / "events", {"trial_id": directory.name})
    env = None
    try:
        env = Environment(
            row["name"], row["env_seed"], row["horizon"], request["image_size"]
        )
        proxy = make_proxy(
            env, events, limits_for(request["manifest"], row["horizon"]), "probe"
        )
        reset = record_reset_evidence(env, proxy, directory)
        write_json(directory / "episode-metadata.json", env.episode_metadata)
        states = [env.state_hash()]
        observations = [reset["observations_sha256"]]
        success = [bool(env.check_success())]
        action = env.neutral_action()
        for _ in range(8):
            env.step(action)
            states.append(env.state_hash())
            observations.append(
                digest(proxy.observe(sorted(proxy.native_keys))["observations"])
            )
            success.append(bool(env.check_success()))
        env.terminal_snapshot(directory / "terminal.npz")
        result = {
            "reset": reset,
            "controller_sha256": digest(env.controller),
            "states": states,
            "observations": observations,
            "success": success,
            "native_action": action,
            "native_steps": 8,
            "model_calls": 0,
            "scored": False,
        }
        write_json(directory / "probe.json", result)
        return result
    finally:
        if env is not None:
            env.close()
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-request", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--task-ids", nargs="+", type=int, default=[4, 11, 17, 25, 46])
    scope.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()
    if args.child_request:
        probe(json.loads(args.child_request.read_text()))
        return
    write_json(ROOT / "worker-entered.json", {"worker_entered": True})
    manifest = yaml.safe_load((STUDY / "preregistration.yaml").read_text())
    task_ids = manifest["task_ids"] if args.all_tasks else args.task_ids
    write_json(
        ROOT / "source-receipt.json",
        {
            "commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "container": manifest["container_digest"],
            "workflow": os.environ["HARDWARE_WORKFLOW"],
        },
    )
    checks = []
    try:
        check_source(CASA, ROBOCASA_COMMIT)
        check_source(SUITE, ROBOSUITE_COMMIT)
        native_fix = verify_fixture_order_fix(CASA)
        selected = {}
        for row in schedule(manifest):
            if row["task_id"] in task_ids:
                selected.setdefault(row["task_id"], row)
        if set(selected) != set(task_ids):
            raise ValueError("unknown_diagnostic_task")
        write_json(ROOT / "diagnostic-schedule.json", list(selected.values()))
        child_environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("R2_", "OPENAI_")) and k != "HARDWARE_API_KEY_FILE"
        }
        for task_id, row in selected.items():
            results = []
            for repeat in range(2):
                directory = ROOT / "reset-diagnostic" / f"t{task_id:02d}-r{repeat}"
                request_path = ROOT / "requests" / f"t{task_id:02d}-r{repeat}.json"
                write_json(
                    request_path,
                    {
                        "row": row,
                        "directory": str(directory),
                        "image_size": manifest["image_size"],
                        "manifest": manifest,
                    },
                )
                with request_path.with_suffix(".log").open("x") as stream:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "experiments.robocasa_hardware_interface.tools.diagnose_resets",
                            "--child-request",
                            str(request_path),
                        ],
                        env=child_environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                results.append(json.loads((directory / "probe.json").read_text()))
            result = {
                "task_id": task_id,
                "task": row["name"],
                "seed": row["env_seed"],
                "matched": results[0] == results[1],
                "different_fields": [
                    k for k in results[0] if results[0][k] != results[1][k]
                ],
            }
            checks.append(result)
            write_json(ROOT / "diagnostic-progress.json", checks, replace=True)
            print(json.dumps(result), flush=True)
        passed = all(row["matched"] for row in checks)
        write_json(
            ROOT / "reset-diagnostic-summary.json",
            {
                "passed": passed,
                "checks": checks,
                "model_calls": 0,
                "scored": False,
                "native_fixture_order_fix": native_fix,
                "task_count": len(selected),
                "repeats_per_task": 2,
                "native_steps_per_probe": 8,
                "interpretation": "Native reset and action-trace repeatability on the recorded task/seed list; not a live Astra policy result.",
            },
        )
        if not passed:
            raise ValueError("isolated_native_repeatability_failed")
    except BaseException as error:
        write_json(
            ROOT / "failure.json",
            {
                "error_class": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        archive_tree(ROOT, "final")


if __name__ == "__main__":
    main()
