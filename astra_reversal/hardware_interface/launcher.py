"""Validate, schedule, commission, run, independently replay, audit and analyze."""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

from .analysis import audit
from .common import Events, digest, file_hash, strict_json, write_json
from .isolation import Scratch
from .libero import Environment, asset_manifest, configure
from .preflight import probe, source_hash
from .protocol import load_manifest, save_schedule, schedule, validate
from .provider import HTTP
from .proxy import NATIVE_KEYS, Proxy
from .reporting import export, power_worksheet
from .runner import run_trial
from .sources import SourceView


def load_rows(path):
    with Path(path).open(newline="") as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for key in ("task_id", "init_state_index", "env_seed", "replicate", "order"):
            row[key] = int(row[key])
        # csv.DictWriter serializes None as an empty cell. Restore only the
        # nullable resource fields before comparing with the frozen schedule.
        for key in ("wall_limit_s", "token_limit", "tool_call_limit"):
            if row[key] == "":
                row[key] = None
    return result


def load_outcomes(runs):
    rows = []
    audits = {r["trial"]: r for r in audit(runs)["trials"]}
    for path in sorted(Path(runs).glob("*/outcome.json")):
        row = strict_json(path.read_bytes())
        row["independent_evaluation_passed"] = (
            audits.get(path.parent.name, {}).get("status") == "passed"
        )
        rows.append(row)
    return rows


def verify_runtime(manifest, prepared, libero_root):
    resolved = manifest["resolved"]
    if resolved["adapter_sha256"] != source_hash():
        raise ValueError("adapter_changed_since_preflight")
    if (
        platform.python_version() != manifest["python_version"]
        or sys.platform != "linux"
    ):
        raise ValueError("runtime_platform_changed")
    installed = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"]
    )
    if installed != (prepared / "pip-freeze.txt").read_bytes():
        raise ValueError("installed_dependencies_changed")
    checks = {
        "dependency_lock_sha256": file_hash(prepared / "pip-freeze.txt"),
        "source_allowlist_sha256": SourceView(prepared / "source-view").manifest[
            "sha256"
        ],
        "catalog_sha256": digest(strict_json((prepared / "catalog.json").read_bytes())),
        "asset_manifest_sha256": asset_manifest(libero_root)["sha256"],
        "observer_sha256": file_hash(Path(__file__).with_name("provider.py")),
    }
    for key, value in checks.items():
        if resolved[key] != value:
            raise ValueError("runtime_lock_mismatch_" + key)


def calibration(registry, args, manifest, suite, row, limits):
    """One evaluator-only, no-agent trace for every scheduled official state."""
    name = f'{suite}-t{row["task_id"]:02d}-i{row["init_state_index"]:02d}'
    directory = Path(args.out) / "calibrations" / name
    events = Events(directory, {"calibration": name})
    env = None
    try:
        env = Environment(
            registry,
            args.libero_root,
            suite,
            row["task_id"],
            row["init_state_index"],
            row["env_seed"],
            settling_steps=manifest["settling_steps"],
            horizon=limits.steps,
            image_size=manifest["image_size"],
        )
        if env.reset_receipt["initial_success"]:
            raise ValueError("initially_successful_calibration")
        proxy = Proxy(
            env,
            env.observation,
            episode_id="calibration",
            controller=env.controller,
            events=events,
            limits=limits,
        )
        observation = proxy.observe(sorted(NATIVE_KEYS))["observations"]
        rejected = proxy.step([2, 0, 0, 0, 0, 0, -1], 1, 0)
        if rejected["accepted"] or proxy.step_count != 0:
            raise RuntimeError("calibration_guard_bypass")
        applied = proxy.step([0, 0, 0, 0, 0, 0, -1], 1, 0)
        if not applied["accepted"] or proxy.step_count != 1:
            raise RuntimeError("calibration_step_failed")
        receipt = {
            "reset": env.reset_receipt,
            "initial_observations_sha256": digest(observation),
            "bounds_rejection_verified": True,
            "evaluator_checked_after_step": True,
            "success_after_step": proxy.success,
            "actor_trial": False,
        }
        env.terminal_snapshot(directory / "terminal.npz")
        write_json(directory / "calibration.json", receipt)
        return receipt
    finally:
        if env is not None:
            env.close()
        events.close()


def save_frames(env, directory, prefix):
    import numpy as np
    from PIL import Image

    observation = env.read_sensors()
    for key, label in (
        ("agentview_image", "front"),
        ("robot0_eye_in_hand_image", "wrist"),
    ):
        Image.fromarray(np.ascontiguousarray(observation[key][::-1])).save(
            directory / f"{prefix}-{label}.png"
        )


def run(args, manifest, *, smoke=False):
    limits = validate(
        manifest,
        scored=not smoke,
        confirmatory=args.split == "confirmatory" and not smoke,
    )
    prepared = Path(args.prepared).resolve()
    if manifest.get("resolved"):
        verify_runtime(manifest, prepared, args.libero_root)
    elif not smoke:
        raise ValueError("unresolved_runtime")
    # Study roots may contain the manifest and a previous disjoint split. Every
    # calibration, trial directory, and split summary still uses exclusive create.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    source = SourceView(prepared / "source-view")
    runtime = strict_json((prepared / "scratch-runtime.json").read_bytes())
    registry = configure(args.libero_root, Path(args.out) / "libero-config")
    if smoke:
        selected = [
            {
                "trial_id": "smoke-" + c,
                "split": "smoke",
                "condition": c,
                "task_id": manifest["smoke"]["task_id"],
                "init_state_index": manifest["smoke"]["init_state_index"],
                "env_seed": manifest["environment_seed"],
                "replicate": 0,
                "order": i,
            }
            for i, c in enumerate(manifest["conditions"])
        ]
    else:
        selected = [r for r in load_rows(args.schedule) if r["split"] == args.split]
        expected = [
            r
            for r in schedule(
                manifest, strict_json((prepared / "catalog.json").read_bytes())
            )
            if r["split"] == args.split
        ]
        if len(selected) != len(expected) or any(
            any(str(row[k]) != str(reference[k]) for k in reference)
            for row, reference in zip(selected, expected)
        ):
            raise ValueError("schedule_changed")
    # Auth data stays in the parent model transport, never in jail/process args.
    post = HTTP(manifest["model"]["base_url"])
    reset_groups = {}
    results = []
    for row in selected:
        suite = manifest["smoke"]["suite"] if smoke else manifest["suite"]
        block = (row["task_id"], row["init_state_index"])
        if block not in reset_groups:
            reset_groups[block] = calibration(
                registry, args, manifest, suite, row, limits
            )
        identity = {
            k: row[k]
            for k in (
                "trial_id",
                "split",
                "condition",
                "task_id",
                "init_state_index",
                "env_seed",
                "replicate",
            )
        }
        directory = Path(args.out) / "runs" / row["trial_id"]
        events = Events(directory, identity)
        env, proxy, actor_entered = None, None, False
        try:
            env = Environment(
                registry,
                args.libero_root,
                suite,
                row["task_id"],
                row["init_state_index"],
                row["env_seed"],
                settling_steps=manifest["settling_steps"],
                horizon=limits.steps,
                image_size=manifest["image_size"],
            )
            initial_hash = env.reset_receipt["official_state_sha256"]
            if not smoke and initial_hash != row["init_state_hash"]:
                raise ValueError("schedule_initial_state_mismatch")
            if env.reset_receipt != reset_groups[block]["reset"]:
                raise ValueError("paired_reset_mismatch")
            if env.reset_receipt["initial_success"]:
                raise ValueError("initially_successful_reset")
            if (
                manifest.get("resolved")
                and digest(env.controller)
                != manifest["resolved"]["controller_config_sha256"]
            ):
                raise ValueError("controller_changed")
            identity["init_state_hash"] = initial_hash
            events.emit("reset", receipt=env.reset_receipt)
            proxy = Proxy(
                env,
                env.observation,
                episode_id=uuid.uuid4().hex,
                controller=env.controller,
                events=events,
                limits=limits,
            )
            if (
                digest(proxy.observe(sorted(NATIVE_KEYS))["observations"])
                != reset_groups[block]["initial_observations_sha256"]
            ):
                raise ValueError("paired_initial_sensor_mismatch")
            save_frames(env, directory, "initial")
            scratch = Scratch(
                prepared / "scratch-runtime",
                runtime["executable"],
                Path(args.out) / "jails" / row["trial_id"],
                source.root,
                row["condition"],
            )
            write_json(
                directory / "trial.json",
                {
                    "row": row,
                    "suite": suite,
                    "manifest_sha256": digest(manifest),
                    "source_sha256": source_hash(),
                    "reset": env.reset_receipt,
                },
            )
            actor_entered = True
            outcome = run_trial(
                proxy,
                source,
                row["condition"],
                env.task.language,
                manifest["model"],
                scratch=scratch,
                post=post,
            )
            env.terminal_snapshot(directory / "terminal.npz")
            save_frames(env, directory, "terminal")
            write_json(
                directory / "terminal-sha256.json",
                {"sha256": file_hash(directory / "terminal.npz")},
            )
            results.append({**identity, **outcome})
        except BaseException as error:
            events.emit(
                "error", error_class=type(error).__name__, phase="setup_or_teardown"
            )
            if not (directory / "outcome.json").exists():
                failure = {
                    **identity,
                    "actor_started": actor_entered,
                    "success": False,
                    "terminal_reason": "EVALUATOR_ERROR"
                    if actor_entered
                    else "RESET_FAILURE",
                    "sim_steps": proxy.step_count if proxy else 0,
                    "wall_s": proxy.clock() - proxy.wall_start if proxy else 0,
                    "known_workflow_tokens": 0,
                    "unknown_usage_records": 1,
                    "estimated_cost_usd": None,
                    "censored_wall": False,
                    "invalid_actions": proxy.invalid_actions if proxy else 0,
                    "safety_attempts": proxy.safety_attempts if proxy else 0,
                    "applied_safety_violations": proxy.applied_violations
                    if proxy
                    else 0,
                    "error_class": type(error).__name__,
                }
                events.emit("trial_end", **failure)
                write_json(directory / "outcome.json", failure)
            raise
        finally:
            if env is not None:
                env.close()
            events.close()
        environment = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OPENAI_", "R2_")) and k != "HARDWARE_API_KEY_FILE"
        }
        subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                "verify-replay",
                "--manifest",
                args.manifest,
                "--libero-root",
                args.libero_root,
                "--trial",
                str(directory),
            ],
            check=True,
            env=environment,
        )
        if outcome["terminal_reason"] in (
            "MODEL_ERROR",
            "TOOL_ERROR",
            "PROTOCOL_DEVIATION",
        ):
            raise RuntimeError("collection_paused_on_infrastructure_failure")
    write_json(
        Path(args.out) / ("summary-" + ("smoke" if smoke else args.split) + ".json"),
        results,
    )
    return {
        "trials": len(results),
        "successes": sum(r["success"] for r in results),
        "split": "smoke" if smoke else args.split,
    }


def verify_replay(args, manifest):
    directory = Path(args.trial)
    trial = strict_json((directory / "trial.json").read_bytes())
    if (
        trial["manifest_sha256"] != digest(manifest)
        or trial["source_sha256"] != source_hash()
    ):
        raise ValueError("replay_source_changed")
    row = trial["row"]
    registry = configure(args.libero_root, directory / "replay-config")
    env = Environment(
        registry,
        args.libero_root,
        trial["suite"],
        row["task_id"],
        row["init_state_index"],
        row["env_seed"],
        settling_steps=manifest["settling_steps"],
        horizon=manifest["limits"]["steps"],
        image_size=manifest["image_size"],
    )
    try:
        if env.reset_receipt != trial["reset"]:
            raise ValueError("replay_reset_mismatch")
        events = [
            strict_json(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
        ]
        actions = [
            r["applied_action"] for r in events if r["event"] == "action_applied"
        ]
        first_success = None
        for step, action in enumerate(actions, 1):
            env.step(action)
            if env.check_success():
                first_success = step
                break
        outcome = strict_json((directory / "outcome.json").read_bytes())
        if outcome["success"] != (first_success is not None) or first_success not in (
            None,
            len(actions),
        ):
            raise ValueError("independent_success_disagreement")
        import numpy as np

        from .libero import array_hash

        with np.load(directory / "terminal.npz") as snapshot:
            matched = array_hash(env.env.get_sim_state()) == array_hash(
                snapshot["state"]
            )
        if not matched:
            raise ValueError("independent_terminal_state_mismatch")
        receipt = {
            "status": "passed",
            "success": first_success is not None,
            "first_success_step": first_success,
            "replayed_steps": len(actions),
            "terminal_state_matched": True,
        }
        write_json(directory / "independent-replay.json", receipt)
        return receipt
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate",
        "schedule",
        "probe",
        "smoke",
        "run",
        "audit",
        "analyze",
        "verify-replay",
        "power",
    ):
        p = commands.add_parser(name)
        if name != "audit":
            p.add_argument("--manifest", required=True)
        if name in ("schedule", "probe", "smoke", "run", "audit", "analyze", "power"):
            p.add_argument("--out", required=name not in ("smoke", "run"))
        if name in ("probe", "smoke", "run", "verify-replay"):
            p.add_argument("--libero-root")
        if name == "schedule":
            p.add_argument("--catalog")
        if name in ("smoke", "run"):
            p.add_argument("--prepared")
            p.add_argument("--schedule", required=name == "run")
            p.add_argument(
                "--split", choices=("pilot", "confirmatory"), default="pilot"
            )
        if name in ("audit", "analyze", "power"):
            p.add_argument("--runs", required=True)
        if name == "analyze":
            p.add_argument(
                "--split", choices=("pilot", "confirmatory"), default="pilot"
            )
        if name == "verify-replay":
            p.add_argument("--trial", required=True)
    args = parser.parse_args()
    if args.command != "audit":
        study = Path(args.manifest).resolve().parent
        prepared = study if (study / "catalog.json").is_file() else study / "prepared"
        for name, default in (
            ("libero_root", study / "src/libero"),
            ("prepared", prepared),
            ("catalog", prepared / "catalog.json"),
            ("out", study),
        ):
            if hasattr(args, name) and getattr(args, name) is None:
                setattr(args, name, str(default))
    manifest = load_manifest(args.manifest) if args.command != "audit" else None
    if args.command == "validate":
        validate(manifest, scored=True)
        result = {"status": "ready_for_pilot"}
    elif args.command == "schedule":
        rows = schedule(manifest, strict_json(Path(args.catalog).read_bytes()))
        save_schedule(args.out, rows)
        result = {"trials": len(rows)}
    elif args.command == "probe":
        resolved, gates = probe(args.libero_root, args.out, manifest)
        result = {"resolved": resolved, "readiness": gates}
    elif args.command in ("smoke", "run"):
        result = run(args, manifest, smoke=args.command == "smoke")
    elif args.command == "verify-replay":
        result = verify_replay(args, manifest)
    elif args.command == "audit":
        result = audit(args.runs)
        write_json(args.out, result)
    elif args.command == "power":
        result = power_worksheet(load_outcomes(args.runs), manifest)
        write_json(args.out, result)
    else:
        result = export(
            load_outcomes(args.runs),
            manifest,
            args.out,
            split=args.split,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
