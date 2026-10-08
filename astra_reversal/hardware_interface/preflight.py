"""Real Linux renderer/reset/interface/isolation checks, separate from actor trials."""

import json
import platform
import subprocess
import sys
from pathlib import Path

from .common import Events, digest, file_hash, write_json
from .isolation import Scratch, build_runtime
from .libero import Environment, asset_manifest, catalog, configure
from .protocol import save_schedule, schedule
from .proxy import NATIVE_KEYS, Proxy
from .schema import validate_description
from .sources import SourceView, build_view


def source_hash():
    return digest(
        {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )


def probe(libero_root, destination, manifest):
    if (
        platform.python_version() != manifest["python_version"]
        or sys.platform != "linux"
    ):
        raise ValueError("probe_requires_pinned_Linux_Python")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    registry = configure(libero_root, destination / "libero-config")
    import robosuite

    allowed = build_view(
        libero_root, Path(robosuite.__file__).parent, destination / "source-view"
    )
    view = SourceView(destination / "source-view")
    runtime = build_runtime(destination / "scratch-runtime")
    write_json(destination / "scratch-runtime.json", runtime)
    jail = Scratch(
        destination / "scratch-runtime",
        runtime["executable"],
        destination / "isolation-jail",
        view.root,
        "B0",
    )
    sentinel = destination / "hidden-evaluator-sentinel"
    sentinel.write_text("this must not be actor-readable")
    checks = {}
    payloads = {
        "host_path_denied": f"from pathlib import Path\nresult = Path({str(sentinel)!r}).exists()",
        "absolute_traversal_denied": "from pathlib import Path\nresult = Path('/../../proc/1/environ').exists()",
        "privileged_modules_unavailable": "import importlib.util\nresult = importlib.util.find_spec('libero') is None",
        "network_denied": "import socket\ntry:\n socket.socket()\n result = False\nexcept PermissionError:\n result = True",
        "credentials_absent": "import os\nresult = not any(k in os.environ for k in ['OPENAI_API_KEY','R2_ACCESS_KEY_ID','R2_SECRET_ACCESS_KEY'])",
        "read_only_source": "from pathlib import Path\ntry:\n Path('/sources/robosuite/controllers/osc.py').write_text('changed')\n result = False\nexcept PermissionError:\n result = True",
        "scratch_writable": "from pathlib import Path\np=Path('/scratch/memo'); p.write_text('ok'); result=p.read_text()=='ok'",
        "prohibited_proxy_denied": "import sys,json\nprint(json.dumps({'tool':'check_success','arguments':{}}),flush=True)\nresult=None",
    }
    receipts = {}
    for name, code in payloads.items():
        result = jail.execute(code, lambda *_: {"error": "unexpected_rpc"})
        expected = (
            False if name in ("host_path_denied", "absolute_traversal_denied") else True
        )
        passed = result == {"result": expected}
        if name == "prohibited_proxy_denied":
            passed = result == {"error": "scratch_tool_not_allowed"}
        receipts[name] = {"passed": passed, "response": result}
    fresh = Scratch(
        destination / "scratch-runtime",
        runtime["executable"],
        destination / "fresh-isolation-jail",
        view.root,
        "B0",
    )
    result = fresh.execute(
        "from pathlib import Path\nresult=Path('/scratch/memo').exists()",
        lambda *_: None,
    )
    receipts["fresh_trial_scratch"] = {
        "passed": result == {"result": False},
        "response": result,
    }
    checks["scratch_isolation"] = all(v["passed"] for v in receipts.values())
    denials = []
    for path in (
        "../hidden-evaluator-sentinel",
        "/etc/passwd",
        "libero/libero/envs/bddl_base_domain.py/private",
        "libero/libero/benchmark/__init__.py",
        "demonstrations/demo.hdf5",
        "other_trial/events.jsonl",
    ):
        try:
            view.read(path)
            denials.append(False)
        except ValueError:
            denials.append(True)
    checks["source_isolation"] = all(denials)
    write_json(
        destination / "isolation.json",
        {"checks": checks, "receipts": receipts, "source_denials": denials},
    )

    indices = manifest["pilot_indices"] + manifest["confirmatory_indices"]
    tasks = catalog(libero_root, registry, indices)
    write_json(destination / "catalog.json", tasks)
    save_schedule(destination / "trials.csv", schedule(manifest, tasks))
    assets = asset_manifest(libero_root)
    write_json(destination / "asset-manifest.json", assets)
    outcomes = []
    action_trace = [
        ([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0], 2),
        ([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 3),
        ([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], 3),
    ]
    smoke = manifest["smoke"]
    for condition in manifest["conditions"]:
        events = Events(
            destination / ("calibration-" + condition),
            {"trial_id": "calibration-" + condition},
        )
        environment = Environment(
            registry,
            libero_root,
            smoke["suite"],
            smoke["task_id"],
            smoke["init_state_index"],
            manifest["environment_seed"],
        )
        try:
            events.emit("reset", receipt=environment.reset_receipt)
            proxy = Proxy(
                environment,
                environment.observation,
                episode_id="calibration",
                controller=environment.controller,
                events=events,
            )
            description = proxy.describe()
            validate_description(description)
            import jsonschema

            schema_file = (
                Path(__file__).resolve().parents[2]
                / "experiments/libero_hardware_interface/schemas/hardware.schema.json"
            )
            jsonschema.validate(description, json.loads(schema_file.read_text()))
            write_json(events.directory / "hardware.json", description)
            initial = proxy.observe(sorted(NATIVE_KEYS))["observations"]
            hashes = [digest(initial)]
            for action, repeat in action_trace:
                if condition == "F":
                    result = proxy.act(
                        {
                            "channel": "robot.controller_command",
                            "value": action,
                            "duration_steps": repeat,
                            "metadata": {
                                "mode": "configured_controller",
                                "units": "normalized_controller_input",
                                "episode_id": "calibration",
                                "observation_step": proxy.step_count,
                            },
                        }
                    )
                else:
                    result = proxy.step(action, repeat, proxy.step_count)
                if not result["accepted"] or proxy.terminal:
                    raise ValueError("calibration_terminated_unexpectedly")
                hashes.append(
                    digest(proxy.observe(sorted(NATIVE_KEYS))["observations"])
                )
            import base64

            for key, camera in (
                ("agentview_image", "front"),
                ("robot0_eye_in_hand_image", "wrist"),
            ):
                (events.directory / (camera + ".png")).write_bytes(
                    base64.b64decode(initial[key]["base64"])
                )
            outcomes.append(
                {
                    "condition": condition,
                    "reset": environment.reset_receipt,
                    "observation_sequence_sha256": hashes,
                    "controller_sha256": digest(environment.controller),
                }
            )
            environment.terminal_snapshot(events.directory / "terminal.npz")
        finally:
            environment.close()
            events.close()
    checks["reset_replay"] = all(r["reset"] == outcomes[0]["reset"] for r in outcomes)
    checks["interface_equivalence"] = all(
        r["observation_sequence_sha256"] == outcomes[0]["observation_sequence_sha256"]
        and r["controller_sha256"] == outcomes[0]["controller_sha256"]
        for r in outcomes
    )
    checks.update(
        success_per_step=False,
        budget_termination=False,
        model_smoke_F=False,
        model_smoke_B0=False,
        model_smoke_B=False,
    )
    # Unit receipts are added only by the bootstrap after its real test command.
    lock = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    )
    (destination / "pip-freeze.txt").write_text(lock)
    resolved = {
        "dependency_lock_sha256": file_hash(destination / "pip-freeze.txt"),
        "asset_manifest_sha256": assets["sha256"],
        "source_allowlist_sha256": allowed["sha256"],
        "adapter_sha256": source_hash(),
        "observer_sha256": file_hash(Path(__file__).with_name("provider.py")),
        "controller_config_sha256": outcomes[0]["controller_sha256"],
        "catalog_sha256": digest(tasks),
    }
    write_json(
        destination / "preflight.json",
        {
            "checks": checks,
            "resolved": resolved,
            "calibrations": outcomes,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "container": manifest["container_digest"],
            },
            "scored_collection_allowed": False,
            "model_access": "not yet commissioned",
        },
    )
    if not all(
        checks[k]
        for k in (
            "reset_replay",
            "interface_equivalence",
            "source_isolation",
            "scratch_isolation",
        )
    ):
        raise RuntimeError("preflight_gate_failed")
    return resolved, checks
