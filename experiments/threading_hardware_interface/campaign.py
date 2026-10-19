"""Gated four-cell Threading pilot; exact serving and native replay required."""

import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from astra_reversal.hardware_interface.analysis import analyze
from astra_reversal.hardware_interface.common import (
    Events,
    digest,
    strict_json,
    write_json,
)
from astra_reversal.hardware_interface.launcher import load_outcomes
from astra_reversal.hardware_interface.policy_modes import PolicyRouter
from astra_reversal.hardware_interface.provider import Meter, Session, model_http
from astra_reversal.hardware_interface.proxy import Proxy
from astra_reversal.hardware_interface.sources import SourceView
from astra_reversal.hardware_interface.speed import analyze_speed, plot_speed
from astra_reversal.hardware_interface.threading_protocol import (
    load,
    schedule,
    validate,
)
from astra_reversal.hardware_interface.threading_worker import (
    archive,
    environment,
    prepare,
    run_one,
)
from experiments.libero_hardware_interface.tools.self_hosted_campaign import (
    relay,
    serving_host,
    wait_for_server,
)


def transport(root, prepared, manifest):
    """Exercise each actual tool schema and image/history transport without motion."""
    rows = [r for r in schedule(manifest) if r["env_seed"] == 0]
    for row in rows:
        events = Events(
            root / "transport" / row["condition"],
            {
                "kind": "transport",
                "condition": row["condition"],
                "scored_result": False,
            },
        )
        env = environment(row, manifest)
        try:
            limits = validate(manifest)
            proxy = Proxy(
                env,
                env.observation,
                episode_id="transport",
                controller=env.controller,
                events=events,
                limits=limits,
            )
            router = PolicyRouter(
                proxy,
                SourceView(prepared / "source-view"),
                "F_adaptive",
                policy_mode=row["policy_mode"],
            )
            model = manifest["models"][row["model_key"]]
            session = Session(
                model, limits, Meter(None), events, post=model_http(model)
            )
            session.history = [
                {
                    "role": "user",
                    "content": "Transport check only. Call read_latest once; do not move the robot.",
                }
            ]
            response = session.request(
                "Call read_latest exactly once to inspect camera inputs. This is an unscored transport check. Do not actuate.",
                router.tools,
            )
            calls = [r for r in response["output"] if r.get("type") == "function_call"]
            if len(calls) != 1 or calls[0]["name"] != "read_latest":
                raise RuntimeError("threading_model_tool_schema_smoke_failed")
            session.tool_result(
                calls[0]["call_id"],
                router.dispatch_model(
                    "read_latest", strict_json(calls[0]["arguments"])
                ),
            )
            response = session.request(
                "The camera frames were supplied by the tool. Call revise_interface_notes exactly once; notes should describe only what is visible in the supplied images, without motion or task advice. Set reason to transport check. Do not call other tools.",
                router.tools,
            )
            calls = [r for r in response["output"] if r.get("type") == "function_call"]
            if len(calls) != 1 or calls[0]["name"] != "revise_interface_notes":
                raise RuntimeError("threading_model_image_history_smoke_failed")
            result = router.dispatch_model(
                calls[0]["name"], strict_json(calls[0]["arguments"])
            )
            if "error" in result or proxy.step_count:
                raise RuntimeError("threading_model_smoke_invalid")
            write_json(
                events.directory / "transport.json",
                {
                    "passed": True,
                    "model": model,
                    "policy_mode": row["policy_mode"],
                    "usage": session.meter.usage,
                    "native_steps": 0,
                    "scored_result": False,
                },
            )
            print(json.dumps({"transport_passed": row["condition"]}), flush=True)
        finally:
            env.close()
            events.close()


def shard():
    root = Path(os.environ["THREADING_ROOT"])
    manifest_path = Path(
        "experiments/threading_hardware_interface/preregistration.yaml"
    ).resolve()
    manifest = load(manifest_path)
    index = int(os.environ["THREADING_SHARD"])
    workers = int(os.environ["THREADING_WORKERS"])
    for row in schedule(manifest):
        if row["init_state_index"] % workers != index:
            continue
        if (root / "stop-after-current-trial").exists():
            break
        run_one(row, manifest, root / "prepared", root, manifest_path)
        archive(
            root / "runs" / row["trial_id"],
            os.environ["HARDWARE_WORKFLOW"],
            "trials/" + row["trial_id"],
        )


def collect(root, manifest):
    workers, jobs = 8, []
    for index in range(workers):
        worker = root / ("worker-%d" % index)
        worker.mkdir()
        (worker / "prepared").symlink_to(
            (root / "prepared").resolve(), target_is_directory=True
        )
        log = (worker / "worker.log").open("w")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.threading_hardware_interface.campaign",
                "shard",
            ],
            env={
                **os.environ,
                "THREADING_ROOT": str(worker),
                "THREADING_SHARD": str(index),
                "THREADING_WORKERS": str(workers),
                "MUJOCO_EGL_DEVICE_ID": str(index),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs.append((process, log, worker))
    write_json(
        root / "campaign-started.json",
        {
            "workflow": os.environ["HARDWARE_WORKFLOW"],
            "trials": 40,
            "workers": workers,
            "source_commit": os.environ["HARDWARE_SOURCE_COMMIT"],
            "manifest_sha256": digest(manifest),
        },
    )
    print(
        json.dumps(
            {"campaign_started": True, "planned_trials": 40, "workers": workers}
        ),
        flush=True,
    )
    try:
        while jobs:
            remaining = []
            for process, log, worker in jobs:
                code = process.poll()
                if code is None:
                    remaining.append((process, log, worker))
                else:
                    log.close()
                    print(
                        json.dumps({"worker_finished": worker.name, "exit_code": code}),
                        flush=True,
                    )
                    if code:
                        print((worker / "worker.log").read_text()[-6000:], flush=True)
                        raise RuntimeError("threading_worker_failed")
            jobs = remaining
            if jobs:
                time.sleep(5)
    finally:
        for _, _, worker in jobs:
            (worker / "stop-after-current-trial").touch(exist_ok=True)
        for process, log, _ in jobs:
            process.wait()
            log.close()


def results(root, manifest):
    outcomes = []
    for worker in sorted(root.glob("worker-*")):
        outcomes.extend(load_outcomes(worker / "runs"))
    write_json(root / "verified-outcomes.json", outcomes)
    analysis = analyze(outcomes, manifest, split="pilot")
    speed = analyze_speed(outcomes, manifest, split="pilot")
    destination = root / "results"
    destination.mkdir()
    write_json(destination / "analysis.json", analysis)
    write_json(destination / "speed.json", speed)
    plot_speed(speed, destination)
    print(
        json.dumps(
            {
                "pilot_verified_trials": len(outcomes),
                "arms": {
                    arm: {"n": values["n"], "successes": values["successes"]}
                    for arm, values in analysis["arms"].items()
                },
            }
        ),
        flush=True,
    )
    if len(outcomes) != 40 or any(
        r.get("independent_evaluation_passed") is not True for r in outcomes
    ):
        raise RuntimeError("threading_pilot_incomplete")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "shard":
        shard()
        return
    root = Path("artifacts").resolve()
    manifest_path = Path(
        "experiments/threading_hardware_interface/preregistration.yaml"
    ).resolve()
    manifest = load(manifest_path)
    host = serving_host(os.environ["HARDWARE_INFERENCE_HOST"])
    connection = relay(host)
    claim = (
        Path(
            "/mnt/amlfs-07/shared/egoverse/rpunamiya/threading-20261019/campaign-ledger"
        )
        / os.environ["HARDWARE_WORKFLOW"]
    )
    try:
        # A persistent exclusive claim prevents accidental trial reruns even if
        # infrastructure restarts the evaluator with an empty local workspace.
        claim.mkdir(parents=True, exist_ok=False)
        write_json(
            claim / "claim.json",
            {
                "workflow": os.environ["HARDWARE_WORKFLOW"],
                "source_commit": os.environ["HARDWARE_SOURCE_COMMIT"],
                "manifest_sha256": digest(manifest),
            },
        )
        prepared = prepare(root, manifest, manifest_path)
        wait_for_server(host, root / "serving-identity.json")
        transport(root, prepared, manifest)
        collect(root, manifest)
        results(root, manifest)
    except BaseException as error:
        write_json(
            root / "failure.json",
            {"error_class": type(error).__name__, "traceback": traceback.format_exc()},
        )
        raise
    finally:
        try:
            with urllib.request.urlopen(
                "http://" + host + ":8001/artifacts", timeout=120
            ) as response:
                (root / "serving-artifacts.tar.gz").write_bytes(response.read())
        except OSError:
            pass
        try:
            request = urllib.request.Request(
                "http://" + host + ":8001/finish",
                data=json.dumps({"workflow": os.environ["HARDWARE_WORKFLOW"]}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15):
                pass
        except OSError:
            pass
        connection.shutdown()
        if Path("bootstrap.log").exists():
            (root / "bootstrap.log").write_bytes(Path("bootstrap.log").read_bytes())
        archive(root, os.environ["HARDWARE_WORKFLOW"])


if __name__ == "__main__":
    main()
