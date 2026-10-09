"""Commission, run and audit a separate RoboCasa direct-Astra cohort on OSMO."""

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .common import Events, digest, file_hash, strict_json, write_json
from .isolation import Scratch, build_runtime
from .preflight import source_hash
from .provider import HTTP
from .proxy import Proxy
from .robocasa import (
    CAMERAS,
    ROBOCASA_COMMIT,
    ROBOSUITE_COMMIT,
    SENSORS,
    SHARED_PROMPT,
    Environment,
    array_hash,
    build_view,
    check_source,
    task_catalog,
)
from .robocasa_protocol import limits_for, paired_success_times, schedule, validate
from .runner import run_trial
from .schema import validate_description
from .sources import SourceView

ROOT = Path("artifacts")
STUDY = Path("experiments/robocasa_hardware_interface")
CASA = Path("upstream-robocasa")
SUITE = Path("upstream-robosuite")


def archive_tree(directory, label):
    """Immutable incremental own-workflow archives; never publish credentials."""
    import boto3

    workflow = os.environ["HARDWARE_WORKFLOW"]
    if not workflow.startswith("robocasa-hardware-interface-20261009-"):
        raise ValueError("archive_workflow_binding")
    prefix = f"experiments/robocasa-hardware-interface-20261009/{workflow}/{label}/"
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("archive_prefix_already_exists")
    files = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if (
            not path.is_file()
            or path.is_symlink()
            or any("jail" in p or p == "scratch-runtime" for p in relative.parts)
        ):
            continue
        if "api-key" in path.name or "catalog.token" in path.name:
            raise ValueError("unexpected_private_credential_in_artifacts")
        sha = file_hash(path)
        client.upload_file(str(path), "rldb", prefix + str(relative))
        files.append(
            {"path": str(relative), "sha256": sha, "bytes": path.stat().st_size}
        )
    receipt = {"workflow": workflow, "prefix": prefix, "files": files}
    client.put_object(
        Bucket="rldb",
        Key=prefix + "archive-receipt.json",
        Body=json.dumps(receipt).encode(),
    )
    print(
        json.dumps({"archived": label, "files": len(files), "prefix": prefix}),
        flush=True,
    )
    return receipt


def make_proxy(env, events, limits, episode_id):
    return Proxy(
        env,
        env.observation,
        episode_id=episode_id,
        controller=env.controller,
        events=events,
        limits=limits,
        sensors=SENSORS,
    )


def save_frames(env, directory, prefix):
    observations = env.read_sensors()
    for camera, label in zip(CAMERAS, ("left", "right", "wrist")):
        frame = np.ascontiguousarray(observations[camera + "_image"][::-1])
        Image.fromarray(frame).save(directory / f"{prefix}-{label}.png")


def commission(manifest):
    check_source(CASA, ROBOCASA_COMMIT)
    check_source(SUITE, ROBOSUITE_COMMIT)
    if (
        sys.platform != "linux"
        or platform.python_version() != manifest["python_version"]
    ):
        raise ValueError("pinned_linux_runtime_required")
    if task_catalog(CASA) != manifest["catalog"]:
        raise ValueError("task_catalog_changed")
    prepared = ROOT / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    allowed = build_view(CASA, SUITE, prepared / "source-view")
    source = SourceView(prepared / "source-view")
    runtime = build_runtime(prepared / "scratch-runtime")
    write_json(prepared / "scratch-runtime.json", runtime)
    scratch = Scratch(
        prepared / "scratch-runtime",
        runtime["executable"],
        prepared / "isolation-jail",
        source.root,
        "B0",
    )
    checks = {}
    probes = {
        "network_denied": "import socket\ntry:\n socket.socket()\n result=False\nexcept PermissionError:\n result=True",
        "hidden_files_absent": "from pathlib import Path\nresult=not Path('/workspace/hardware-study/upstream-robocasa').exists() and not Path('/proc').exists()",
        "benchmark_modules_absent": "import importlib.util\nresult=importlib.util.find_spec('robocasa') is None and importlib.util.find_spec('mujoco') is None",
        "credentials_absent": "import os\nresult=not any(k.startswith(('OPENAI_','R2_')) or k=='HARDWARE_API_KEY_FILE' for k in os.environ)",
        "source_readonly": "from pathlib import Path\ntry:\n Path('/sources/robosuite/controllers/parts/arm/osc.py').write_text('bad')\n result=False\nexcept PermissionError:\n result=True",
        "scratch_works": "from pathlib import Path\np=Path('/scratch/memo');p.write_text('ok');result=p.read_text()=='ok'",
    }
    for name, code in probes.items():
        checks[name] = scratch.execute(code, lambda *_: {"error": "not_allowed"}) == {
            "result": True
        }
    fresh = Scratch(
        prepared / "scratch-runtime",
        runtime["executable"],
        prepared / "fresh-jail",
        source.root,
        "B0",
    )
    checks["fresh_scratch"] = fresh.execute(
        "from pathlib import Path\nresult=not Path('/scratch/memo').exists()",
        lambda *_: None,
    ) == {"result": True}
    try:
        source.read("robocasa/environments/kitchen/kitchen.py")
        checks["source_isolation"] = False
    except ValueError:
        checks["source_isolation"] = True
    traces = []
    smoke = manifest["smoke"]
    for arm in manifest["conditions"]:
        events = Events(
            prepared / ("calibration-" + arm), {"trial_id": "calibration-" + arm}
        )
        env = Environment(
            smoke["name"], smoke["seed"], smoke["horizon"], manifest["image_size"]
        )
        try:
            proxy = make_proxy(
                env, events, limits_for(manifest, smoke["horizon"]), "calibration"
            )
            validate_description(proxy.describe())
            write_json(events.directory / "hardware.json", proxy.describe())
            save_frames(env, events.directory, "initial")
            hashes = [digest(proxy.observe(sorted(proxy.native_keys))["observations"])]
            bad = env.neutral_action()
            bad[0] = 2.0
            if proxy.step(bad, 1, 0)["accepted"] or proxy.step_count:
                raise ValueError("mobile_bounds_guard_failed")
            for field in ("arm_dz", "base_yaw", "torso"):
                action = env.neutral_action()
                action[env.controller["action_order"].index(field)] = 0.05
                if arm == "F":
                    result = proxy.act(
                        {
                            "channel": "robot.controller_command",
                            "value": action,
                            "duration_steps": 2,
                            "metadata": {
                                "mode": "configured_controller",
                                "units": "normalized_controller_input",
                                "episode_id": "calibration",
                                "observation_step": proxy.step_count,
                            },
                        }
                    )
                else:
                    result = proxy.step(action, 2, proxy.step_count)
                if not result["accepted"] or proxy.terminal:
                    raise ValueError("calibration_trace_terminated")
                hashes.append(
                    digest(proxy.observe(sorted(proxy.native_keys))["observations"])
                )
            traces.append(
                {
                    "reset": env.reset_receipt,
                    "observations": hashes,
                    "terminal": env.state_hash(),
                    "controller": digest(env.controller),
                }
            )
        finally:
            env.close()
            events.close()
    checks["native_interface_equivalence"] = traces[0] == traces[1] == traces[2]
    checks["initially_unsolved"] = not traces[0]["reset"]["initial_success"]
    resolved = {
        "source_sha256": source_hash(),
        "allowlist_sha256": allowed["sha256"],
        "controller_sha256": traces[0]["controller"],
        "catalog_sha256": digest(manifest["catalog"]),
        "pip_freeze_sha256": file_hash(ROOT / "runtime/pip-freeze.txt"),
        "asset_downloads_sha256": file_hash(ROOT / "runtime/asset-downloads.json"),
    }
    write_json(
        prepared / "commissioning.json",
        {"checks": checks, "resolved": resolved, "traces": traces},
    )
    print(json.dumps({"commissioning": checks, "resolved": resolved}), flush=True)
    if not all(checks.values()):
        raise ValueError("commissioning_failed")
    return resolved


def verify_replay(manifest, directory):
    trial = strict_json((directory / "trial.json").read_bytes())
    if trial["source_sha256"] != source_hash() or trial["manifest_sha256"] != digest(
        manifest
    ):
        raise ValueError("replay_source_changed")
    row = trial["row"]
    env = Environment(
        row["name"], row["env_seed"], row["horizon"], manifest["image_size"]
    )
    video = None
    try:
        if env.reset_receipt != trial["reset"]:
            raise ValueError("independent_reset_mismatch")
        if row["split"] == "pilot" and row["task_id"] in manifest["video_task_ids"]:
            import imageio

            video = imageio.get_writer(
                directory / "replay-cameras.mp4", fps=4, macro_block_size=1
            )
        steps, first_success = 0, None
        previous, sequence = "0" * 64, 0
        with (directory / "events.jsonl").open() as stream:
            for line in stream:
                event = strict_json(line)
                sha = event.pop("sha256")
                if (
                    event["previous_sha256"] != previous
                    or event["sequence"] != sequence
                    or digest(event) != sha
                ):
                    raise ValueError("event_chain_corrupt")
                previous, sequence = sha, sequence + 1
                if event["event"] != "action_applied":
                    continue
                if first_success is not None:
                    raise ValueError("action_after_first_success")
                if (
                    event["simulator_step_before"] != steps
                    or event["simulator_step_after"] != steps + 1
                ):
                    raise ValueError("replay_action_discontinuity")
                observation, _, _, _ = env.step(event["applied_action"])
                steps += 1
                if video is not None and steps % 5 == 0:
                    frame = np.concatenate(
                        [observation[c + "_image"][::-1] for c in CAMERAS], axis=1
                    )
                    video.append_data(np.ascontiguousarray(frame))
                if env.check_success():
                    first_success = steps
        outcome = strict_json((directory / "outcome.json").read_bytes())
        if (
            outcome["success"] != (first_success is not None)
            or outcome["sim_steps"] != steps
        ):
            raise ValueError("independent_success_disagreement")
        with np.load(directory / "terminal.npz") as snapshot:
            if array_hash(snapshot["state"]) != env.state_hash():
                raise ValueError("independent_terminal_state_mismatch")
        receipt = {
            "status": "passed",
            "success": outcome["success"],
            "replayed_steps": steps,
            "first_success_step": first_success,
            "terminal_state_matched": True,
            "event_chain_verified": True,
            "model_requests": 0,
        }
        write_json(directory / "independent-replay.json", receipt)
        return receipt
    finally:
        if video is not None:
            video.close()
        env.close()


def collect_trial(manifest, row, reset_groups):
    prepared = ROOT / "prepared"
    source = SourceView(prepared / "source-view")
    source.verify()
    if source_hash() != manifest["resolved"]["source_sha256"]:
        raise ValueError("adapter_changed_after_commissioning")
    runtime = strict_json((prepared / "scratch-runtime.json").read_bytes())
    identity = {
        k: row[k]
        for k in (
            "trial_id",
            "condition",
            "split",
            "task_id",
            "init_state_index",
            "env_seed",
            "replicate",
        )
    }
    identity["task_name"] = row["name"]
    directory = ROOT / row["split"] / "runs" / row["trial_id"]
    events = Events(directory, identity)
    env = None
    try:
        env = Environment(
            row["name"], row["env_seed"], row["horizon"], manifest["image_size"]
        )
        if env.reset_receipt["initial_success"]:
            raise ValueError("initially_solved_scenario_not_resampled")
        if digest(env.controller) != manifest["resolved"]["controller_sha256"]:
            raise ValueError("controller_changed_between_tasks")
        proxy = make_proxy(
            env, events, limits_for(manifest, row["horizon"]), row["trial_id"]
        )
        reset = {
            "receipt": env.reset_receipt,
            "observations_sha256": digest(
                proxy.observe(sorted(proxy.native_keys))["observations"]
            ),
        }
        block = (row["name"], row["env_seed"])
        if block in reset_groups and reset_groups[block] != reset:
            raise ValueError("paired_initial_state_or_observation_mismatch")
        reset_groups.setdefault(block, reset)
        identity["init_state_hash"] = digest(env.reset_receipt)
        events.emit("reset", receipt=env.reset_receipt)
        write_json(
            directory / "trial.json",
            {
                "row": row,
                "reset": env.reset_receipt,
                "source_sha256": source_hash(),
                "manifest_sha256": digest(manifest),
            },
        )
        # Evaluator-private metadata; only its language string goes to the actor.
        write_json(directory / "episode-metadata.json", env.episode_metadata)
        save_frames(env, directory, "initial")
        scratch = Scratch(
            prepared / "scratch-runtime",
            runtime["executable"],
            ROOT / "jails" / row["trial_id"],
            source.root,
            row["condition"],
        )
        outcome = run_trial(
            proxy,
            source,
            row["condition"],
            env.instruction,
            manifest["model"],
            scratch=scratch,
            post=HTTP(manifest["model"]["base_url"]),
            shared_prompt=SHARED_PROMPT,
            source_entry="robocasa/wrappers/gym_wrapper.py",
        )
        env.terminal_snapshot(directory / "terminal.npz")
        save_frames(env, directory, "terminal")
    finally:
        if env is not None:
            env.close()
        events.close()
    # An independent trusted process with no model-provider or R2 credentials.
    child_environment = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OPENAI_", "R2_")) and k != "HARDWARE_API_KEY_FILE"
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "astra_reversal.hardware_interface.robocasa_worker",
            "--verify-replay",
            str(directory),
            "--manifest",
            str(ROOT / "resolved-manifest.yaml"),
        ],
        env=child_environment,
        check=True,
    )
    outcome = {**identity, **outcome, "independent_evaluation_passed": True}
    print(
        json.dumps(
            {
                "trial_completed": row["trial_id"],
                "task": row["name"],
                **{
                    k: outcome[k]
                    for k in (
                        "condition",
                        "success",
                        "terminal_reason",
                        "sim_steps",
                        "wall_s",
                        "known_workflow_tokens",
                        "unknown_usage_records",
                    )
                },
                "independent_replay": "passed",
                "usage_by_role": outcome["usage_by_role"],
            }
        ),
        flush=True,
    )
    archive_tree(directory, "trials/" + row["trial_id"])
    if outcome["terminal_reason"] in (
        "MODEL_ERROR",
        "TOOL_ERROR",
        "PROTOCOL_DEVIATION",
    ):
        raise RuntimeError("collection_paused_on_infrastructure_failure")
    return outcome


def summarize(manifest):
    from .analysis import analyze

    outcomes = []
    for path in sorted((ROOT / "pilot/runs").glob("*/outcome.json")):
        row = strict_json(path.read_bytes())
        receipt_path = path.parent / "independent-replay.json"
        row["independent_evaluation_passed"] = (
            receipt_path.exists()
            and strict_json(receipt_path.read_bytes())["status"] == "passed"
        )
        outcomes.append(row)
    result = {
        "completed_outcomes": len(outcomes),
        "planned_trials": len(schedule(manifest)),
        "successes": sum(r["success"] for r in outcomes),
        "known_tokens": sum(r["known_workflow_tokens"] for r in outcomes),
        "unknown_usage_records": sum(r["unknown_usage_records"] for r in outcomes),
        "paired_time_to_success": paired_success_times(outcomes),
    }
    if outcomes:
        result["analysis"] = analyze(outcomes, manifest, split="pilot")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-replay", type=Path)
    parser.add_argument("--manifest", type=Path, default=STUDY / "preregistration.yaml")
    parser.add_argument("--archive-bootstrap-failure", action="store_true")
    args = parser.parse_args()
    if args.archive_bootstrap_failure:
        archive_tree(ROOT, "bootstrap-failure")
        return
    manifest = yaml.safe_load(args.manifest.read_text())
    validate(manifest)
    if args.verify_replay:
        print(json.dumps(verify_replay(manifest, args.verify_replay)), flush=True)
        return
    write_json(ROOT / "worker-entered.json", {"worker_entered": True})
    write_json(
        ROOT / "source-receipt.json",
        {
            "commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "container": manifest["container_digest"],
            "workflow": os.environ["HARDWARE_WORKFLOW"],
        },
    )
    stage = os.environ.get("HARDWARE_STAGE", "commission")
    try:
        manifest["resolved"] = commission(manifest)
        with (ROOT / "resolved-manifest.yaml").open("x") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
        rows = schedule(manifest)
        write_json(ROOT / "schedule.json", rows)
        archive_tree(ROOT, "commissioning")
        if stage == "commission":
            return
        # Live three-arm checks use a disjoint task and 20-step episodes only.
        smoke_groups = {}
        for arm in manifest["conditions"]:
            smoke = manifest["smoke"]
            row = {
                "trial_id": "smoke-" + arm,
                "condition": arm,
                "split": "smoke",
                "task_id": -1,
                "init_state_index": smoke["seed"],
                "env_seed": smoke["seed"],
                "replicate": 0,
                "name": smoke["name"],
                "horizon": smoke["horizon"],
            }
            collect_trial(manifest, row, smoke_groups)
        print(
            json.dumps(
                {
                    "all_live_smokes_and_replays_passed": True,
                    "planned_pilot_trials": len(rows),
                    "planned_tasks": len(manifest["task_ids"]),
                    "resource_caps": None,
                }
            ),
            flush=True,
        )
        if stage == "smoke":
            return
        if stage != "pilot":
            raise ValueError("unknown_stage")
        reset_groups = {}
        for row in rows:
            print(
                json.dumps(
                    {
                        "trial_starting": row["trial_id"],
                        "task": row["name"],
                        "horizon": row["horizon"],
                    }
                ),
                flush=True,
            )
            collect_trial(manifest, row, reset_groups)
            write_json(
                ROOT / "progress" / f"after-{row['order']:04d}.json",
                summarize(manifest),
            )
    except BaseException as error:
        write_json(
            ROOT / "failure.json",
            {
                "error_class": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            json.dumps({"worker_failure": type(error).__name__, "reason": str(error)}),
            flush=True,
        )
        raise
    finally:
        write_json(ROOT / "final-summary.json", summarize(manifest))
        archive_tree(ROOT, "final")


if __name__ == "__main__":
    main()
