"""Native commissioning and append-only Threading policy trials on OSMO."""

import argparse
import functools
import json
import os
import subprocess
import sys
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .common import Events, digest, file_hash, strict_json, write_json
from .isolation import Scratch, build_runtime
from .launcher import save_frames
from .policy_modes import INSTRUCTIONS, PolicyRouter
from .preflight import source_hash
from .provider import model_http
from .proxy import NATIVE_KEYS, Proxy
from .robosuite_benchmark import (
    INSTRUCTION,
    MIMICGEN_COMMIT,
    ROBOSUITE_COMMIT,
    SHARED_PROMPT,
    Environment,
    array_hash,
    build_view,
    check_source,
)
from .runner import run_trial
from .sources import SourceView
from .threading_protocol import load, schedule, validate


def environment(row, manifest):
    return Environment(
        row["env_seed"],
        image_size=manifest["image_size"],
        horizon=manifest["limits"]["steps"],
        render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
    )


def reset_identity(env):
    return {
        "reset": env.reset_receipt,
        "initial_sensor_sha256": digest(
            {
                key: np.asarray(env.observation[key]).tolist()
                for key in sorted(NATIVE_KEYS)
            }
        ),
        "controller_sha256": digest(env.controller),
    }


def record_reset_evidence(env, directory):
    """Keep native reset evidence before a pairing rejection, outside actor tools."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    initial = reset_identity(env)
    write_json(directory / "reset-identity.json", initial)
    (directory / "reset-model.xml").write_text(env.reset_model_xml)
    np.savez_compressed(
        directory / "reset-evidence.npz",
        state=env.env.sim.get_state().flatten(),
        body_pos=env.env.sim.model.body_pos,
        body_quat=env.env.sim.model.body_quat,
        **{key: np.asarray(env.observation[key]) for key in sorted(NATIVE_KEYS)},
    )
    return initial


def check_reset_pair(reference, actual, directory):
    if reference == actual:
        return True
    receipt = {
        "different_fields": sorted(
            key
            for key in set(reference) | set(actual)
            if reference.get(key) != actual.get(key)
        ),
        "different_reset_fields": sorted(
            key
            for key in set(reference["reset"]) | set(actual["reset"])
            if reference["reset"].get(key) != actual["reset"].get(key)
        ),
        "reference": reference,
        "actual": actual,
    }
    write_json(Path(directory) / "reset-mismatch.json", receipt)
    print(json.dumps({"threading_reset_mismatch": receipt}), flush=True)
    return False


def isolated_replay(directory, manifest_path):
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OPENAI_", "R2_")) and k != "HARDWARE_API_KEY_FILE"
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "astra_reversal.hardware_interface.threading_worker",
            "replay",
            "--manifest",
            str(manifest_path),
            "--trial",
            str(directory),
        ],
        check=True,
        env=clean,
    )


def video_writer(path, fps):
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path), fps=fps, macro_block_size=16, ffmpeg_params=["-threads", "2"]
    )


def replay(directory, manifest):
    directory = Path(directory)
    trial = strict_json((directory / "trial.json").read_bytes())
    if trial["source_sha256"] != source_hash() or trial["manifest_sha256"] != digest(
        manifest
    ):
        raise ValueError("threading_replay_source_changed")
    env = environment(trial["row"], manifest)
    writer = None
    try:
        if reset_identity(env) != trial["initial"]:
            raise ValueError("threading_replay_reset_mismatch")
        actions = [
            row["applied_action"]
            for row in map(
                strict_json, (directory / "events.jsonl").read_text().splitlines()
            )
            if row["event"] == "action_applied"
        ]
        # Every completed trial gets every native frame. Rendering and encoding
        # remain outside measured actor time and never shorten a live episode.
        writer = video_writer(directory / "replay.mp4", manifest["video"]["fps"])
        frame_count = 0

        def frame():
            nonlocal frame_count
            if writer is not None:
                obs = env.read_sensors()
                writer.append_data(
                    np.concatenate(
                        [
                            obs["agentview_image"][::-1],
                            obs["robot0_eye_in_hand_image"][::-1],
                        ],
                        axis=1,
                    )
                )
                frame_count += 1

        frame()
        first_success = None
        for step, action in enumerate(actions, 1):
            env.step(action)
            frame()
            if env.check_success():
                first_success = step
                if step != len(actions):
                    raise ValueError("threading_replay_early_success_disagreement")
        outcome = strict_json((directory / "outcome.json").read_bytes())
        if outcome["success"] != (first_success is not None) or outcome[
            "sim_steps"
        ] != len(actions):
            raise ValueError("threading_replay_success_disagreement")
        with np.load(directory / "terminal.npz") as snapshot:
            matched = (
                array_hash(env.env.sim.get_state().flatten())
                == array_hash(snapshot["state"])
                and array_hash(env.env.sim.model.body_pos)
                == array_hash(snapshot["body_pos"])
                and array_hash(env.env.sim.model.body_quat)
                == array_hash(snapshot["body_quat"])
            )
        if not matched:
            raise ValueError("threading_replay_terminal_mismatch")
        receipt = {
            "status": "passed",
            "success": first_success is not None,
            "first_success_step": first_success,
            "replayed_steps": len(actions),
            "terminal_state_matched": True,
            "fresh_process": True,
        }
        write_json(directory / "independent-replay.json", receipt)
        if writer is not None:
            write_json(
                directory / "replay-video.json",
                {
                    "native_steps": len(actions),
                    "frames": frame_count,
                    "fps": manifest["video"]["fps"],
                    "frame_stride": 1,
                    "initial_frame_included": True,
                    "last_native_step": len(actions),
                    "control_seconds": len(actions) / manifest["control_frequency"],
                    "video_seconds": frame_count / manifest["video"]["fps"],
                    "horizon_steps": manifest["limits"]["steps"],
                    "terminal_reason": outcome["terminal_reason"],
                    "actor_wall_seconds": outcome["wall_s"],
                    "native_replay_verified": True,
                    "model_reasoning_waits_included": False,
                },
            )
        return receipt
    finally:
        if writer is not None:
            writer.close()
        env.close()


def native_post(row, proxy, model):
    """Deterministic routing check, never a learned policy or task result."""
    actions = [
        ([0, 0, 0.1, 0, 0, 0, -1], 2),
        ([0.1, 0, 0, 0, 0, 0, -1], 3),
        ([-0.1, 0, 0, 0, 0, 0, -1], 3),
    ]
    count = [0]

    def envelope(action, repeat):
        return {
            "channel": "robot.controller_command",
            "value": action,
            "duration_steps": repeat,
            "metadata": {
                "mode": "configured_controller",
                "units": "normalized_controller_input",
                "episode_id": proxy.episode_id,
                "observation_step": proxy.step_count,
            },
        }

    def post(path, body, timeout):
        i, count[0] = count[0], count[0] + 1
        if row["policy_mode"] == "CAP" and i == 0:
            code = (
                "for action, repeat in " + repr(actions) + ":\n"
                " packet = read('robot.eef_position',1000)\n"
                " result = act({'channel':'robot.controller_command',"
                "'value':action,'duration_steps':repeat,'metadata':{"
                "'mode':'configured_controller','units':'normalized_controller_input',"
                "'episode_id':packet['metadata']['episode_id'],"
                "'observation_step':packet['metadata']['simulator_step']}})\n"
            )
            name, args = "scratch_execute", {"code": code}
        elif row["policy_mode"] == "direct" and i < len(actions):
            name, args = "act", {"envelope": envelope(*actions[i])}
        else:
            name, args = "finish", {}
        return {
            "model": model["identifier"],
            "status": "completed",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "output": [
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": json.dumps(args),
                    "call_id": "native-%d" % i,
                }
            ],
        }

    return post


def run_one(row, manifest, prepared, root, manifest_path, *, native=False):
    identity = {**row, "benchmark": "Threading_D0", "scored_result": not native}
    events = Events(root / "runs" / row["trial_id"], identity)
    env, proxy = None, None
    try:
        env = environment(row, manifest)
        initial = record_reset_evidence(env, events.directory)
        catalog = strict_json((prepared / "catalog.json").read_bytes())
        if not check_reset_pair(
            catalog[str(row["env_seed"])], initial, events.directory
        ):
            raise ValueError("threading_paired_initial_state_mismatch")
        identity["init_state_hash"] = digest(initial)
        events.emit("reset", receipt=env.reset_receipt)
        source = SourceView(prepared / "source-view")
        runtime = strict_json((prepared / "scratch-runtime.json").read_bytes())
        proxy = Proxy(
            env,
            env.observation,
            episode_id=uuid.uuid4().hex,
            controller=env.controller,
            events=events,
            limits=validate(manifest),
        )
        save_frames(env, events.directory, "initial")
        write_json(
            events.directory / "trial.json",
            {
                "row": row,
                "manifest_sha256": digest(manifest),
                "source_sha256": source_hash(),
                "initial": initial,
                "scored_result": not native,
            },
        )
        scratch = Scratch(
            prepared / "scratch-runtime",
            runtime["executable"],
            root / "jails" / row["trial_id"],
            source.root,
            "F_adaptive",
        )
        model = manifest["models"][row["model_key"]]
        outcome = run_trial(
            proxy,
            source,
            "F_adaptive",
            INSTRUCTION,
            model,
            scratch=scratch,
            post=native_post(row, proxy, model) if native else model_http(model),
            shared_prompt=SHARED_PROMPT,
            source_entry="robosuite/controllers/osc.py",
            router_factory=functools.partial(
                PolicyRouter, policy_mode=row["policy_mode"], allow_early_finish=native
            ),
            policy_instructions=INSTRUCTIONS[row["policy_mode"]],
        )
        if native and outcome["sim_steps"] != 8:
            raise RuntimeError("threading_native_control_route_failed")
        env.terminal_snapshot(events.directory / "terminal.npz")
        save_frames(env, events.directory, "terminal")
    except BaseException as error:
        events.emit(
            "error",
            phase="threading_setup_or_teardown",
            error_class=type(error).__name__,
        )
        write_json(
            events.directory / "infrastructure-failure.json",
            {
                "error_class": type(error).__name__,
                "reason": str(error),
                "actor_entered": (events.directory / "outcome.json").exists(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
        events.close()
    isolated_replay(events.directory, manifest_path)
    print(
        json.dumps(
            {
                "trial_completed": row["trial_id"],
                "scored_result": not native,
                "success": outcome["success"],
                "sim_steps": outcome["sim_steps"],
                "wall_seconds": outcome["wall_s"],
                "terminal_reason": outcome["terminal_reason"],
                "independent_replay": "passed",
            }
        ),
        flush=True,
    )
    if outcome["terminal_reason"] in (
        "MODEL_ERROR",
        "TOOL_ERROR",
        "PROTOCOL_DEVIATION",
    ):
        raise RuntimeError("threading_collection_stopped_on_infrastructure_failure")
    return outcome


def prepare(root, manifest, manifest_path):
    import robosuite

    check_source("upstream-mimicgen", MIMICGEN_COMMIT)
    check_source("upstream-robosuite", ROBOSUITE_COMMIT)
    prepared = root / "prepared"
    prepared.mkdir()
    build_view(Path(robosuite.__file__).parent, prepared / "source-view")
    runtime = build_runtime(prepared / "scratch-runtime")
    write_json(prepared / "scratch-runtime.json", runtime)
    source = SourceView(prepared / "source-view")
    jail = Scratch(
        prepared / "scratch-runtime",
        runtime["executable"],
        prepared / "isolation-jail",
        source.root,
        "F_adaptive",
    )
    probes = {
        "no_simulator": "import importlib.util\nresult = importlib.util.find_spec('robosuite') is None",
        "no_host_environment": "import os\nresult = not any(k in os.environ for k in ['R2_SECRET_ACCESS_KEY','HARDWARE_API_KEY_FILE','OPENAI_API_KEY'])",
        "no_network": "import socket\ntry:\n socket.socket()\n result=False\nexcept PermissionError:\n result=True",
        "no_parent_files": "from pathlib import Path\nresult = not Path('/proc/1/environ').exists()",
        "read_only_sources": "from pathlib import Path\ntry:\n Path('/sources/robosuite/controllers/osc.py').write_text('changed')\n result=False\nexcept PermissionError:\n result=True",
    }
    isolation = {
        name: jail.execute(code, lambda *_: {"error": "unexpected_rpc"})
        for name, code in probes.items()
    }
    if any(value != {"result": True} for value in isolation.values()):
        raise RuntimeError("threading_scratch_isolation_failed")
    write_json(prepared / "isolation.json", isolation)
    catalog = {}
    for seed in manifest["environment_seeds"]:
        env = environment({"env_seed": seed}, manifest)
        try:
            # Proxy construction enforces all native sensor shapes and bounds.
            events = Events(prepared / ("reset-%03d" % seed), {"commissioning": True})
            proxy = Proxy(
                env,
                env.observation,
                episode_id="commissioning",
                controller=env.controller,
                events=events,
                limits=validate(manifest),
            )
            rejected = proxy.step([2, 0, 0, 0, 0, 0, -1], 1, 0)
            if rejected["accepted"] or proxy.step_count:
                raise ValueError("threading_bounds_guard_failed")
            catalog[str(seed)] = record_reset_evidence(env, events.directory)
            if seed == 0:
                save_frames(env, prepared, "commissioning")
                write_json(prepared / "hardware.json", proxy.describe())
        finally:
            events.close()
            env.close()
    write_json(prepared / "catalog.json", catalog)
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=(prepared / "pip-freeze.txt").open("w"),
    )
    native = root / "native"
    for row in [r for r in schedule(manifest) if r["init_state_index"] == 0]:
        run_one(
            {**row, "split": "native"},
            manifest,
            prepared,
            native,
            manifest_path,
            native=True,
        )
    from .analysis import audit

    checks = audit(native / "runs")
    if checks["status"] != "passed" or len(checks["trials"]) != 4:
        raise RuntimeError("threading_native_replay_audit_failed")
    write_json(
        root / "commissioning.json",
        {
            "passed": True,
            "model_calls": 0,
            "scored_results": False,
            "native_trials": checks,
            "catalog_sha256": digest(catalog),
            "source_sha256": source_hash(),
        },
    )
    return prepared


def archive(root, workflow, label="final"):
    import boto3

    if (
        not workflow.startswith("threading-hardware-20261019-")
        or label == ""
        or ".." in Path(label).parts
    ):
        raise ValueError("threading_archive_binding")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    prefix = "experiments/threading-hardware-20261019/" + workflow + "/" + label + "/"
    if client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("threading_archive_prefix_exists")
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(
                "jail" in p or p in ("scratch-runtime", "__pycache__")
                for p in relative.parts
            )
        ):
            continue
        if "api-key" in path.name:
            raise ValueError("private_credential_in_artifacts")
        client.upload_file(str(path), "rldb", prefix + str(relative))
        files.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    client.put_object(
        Bucket="rldb",
        Key=prefix + "archive-receipt.json",
        Body=json.dumps(
            {"workflow": workflow, "prefix": prefix, "files": files}
        ).encode(),
    )
    print(json.dumps({"archive_prefix": prefix, "files": len(files)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("commission", "replay"))
    parser.add_argument(
        "--manifest",
        default="experiments/threading_hardware_interface/preregistration.yaml",
    )
    parser.add_argument("--trial")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load(manifest_path)
    if args.command == "replay":
        replay(args.trial, manifest)
        return
    root = Path("artifacts").resolve()
    write_json(
        root / "source-receipt.json",
        {
            "source_commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "workflow": os.environ["HARDWARE_WORKFLOW"],
            "manifest_sha256": digest(manifest),
            "limits": asdict(validate(manifest)),
        },
    )
    try:
        prepare(root, manifest, manifest_path)
    except BaseException as error:
        write_json(
            root / "failure.json",
            {"error_class": type(error).__name__, "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if Path("bootstrap.log").exists():
            (root / "bootstrap.log").write_bytes(Path("bootstrap.log").read_bytes())
        archive(root, os.environ["HARDWARE_WORKFLOW"])


if __name__ == "__main__":
    main()
