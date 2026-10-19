"""Private serving relay and gated matched cohorts in one OSMO group."""

import copy
import ipaddress
import json
import os
import re
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

from astra_reversal.hardware_interface import launcher, osmo_worker
from astra_reversal.hardware_interface.common import strict_json, write_json
from astra_reversal.hardware_interface.preflight import source_hash
from astra_reversal.hardware_interface.protocol import load_manifest, schedule
from astra_reversal.hardware_interface.replication import (
    DEEPSEEK_MODEL,
    DEEPSEEK_REVISION,
    VLLM_IMAGE,
)
from astra_reversal.hardware_interface.reporting import export

from .parallel_plan import create_fresh_plan


def serving_host(value):
    try:
        addresses = [ipaddress.ip_address(value)]
    except ValueError:
        # OSMO's host template expands to a Kubernetes service name on H100.
        # Resolve it once and pin the private IPv4 address for all connections.
        label = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        if len(value) > 253 or not re.fullmatch(
            r"(?:" + label + r"\.)+svc\.cluster\.local", value
        ):
            raise ValueError("private_osmo_group_address_required") from None
        addresses = [
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(
                value, 8000, family=socket.AF_INET, type=socket.SOCK_STREAM
            )
        ]
    if not addresses or any(
        not address.is_private
        or address.is_loopback
        or address.is_unspecified
        or address.is_multicast
        or address.is_link_local
        or address.version != 4
        for address in addresses
    ):
        raise ValueError("private_osmo_group_address_required")
    return str(addresses[0])


def relay(host):
    host = serving_host(host)

    class Forward(socketserver.BaseRequestHandler):
        def handle(self):
            with socket.create_connection((host, 8000), timeout=30) as remote:
                streams = [self.request, remote]
                while True:
                    ready, _, _ = select.select(streams, [], [], 200)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (remote if source is self.request else self.request).sendall(
                            data
                        )

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    server = Server(("127.0.0.1", 18080), Forward)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wait_for_server(host, destination):
    previous = None
    while True:
        try:
            with urllib.request.urlopen(
                "http://" + host + ":8001/identity", timeout=10
            ) as response:
                identity = json.load(response)
            if identity["phase"] != previous:
                print(json.dumps({"serving_phase": identity["phase"]}), flush=True)
                previous = identity["phase"]
            if identity["phase"] == "failed":
                raise RuntimeError("self_hosted_server_failed")
            if identity["phase"] == "ready":
                if (
                    identity.get("model"),
                    identity.get("revision"),
                    identity.get("image"),
                ) != (DEEPSEEK_MODEL, DEEPSEEK_REVISION, VLLM_IMAGE):
                    raise RuntimeError("serving_identity_mismatch")
                if (
                    len(identity.get("gpus", [])) != 8
                    or len(identity.get("shards", [])) != 48
                ):
                    raise RuntimeError("serving_allocation_or_checkpoint_incomplete")
                write_json(destination, identity)
                return
        except OSError:
            pass
        time.sleep(10)


def collect_cohorts(root, commissioned):
    """Share verified native setup; each actor still gets a fresh isolated trial."""
    parent = os.environ["HARDWARE_WORKFLOW"]
    source = os.environ["HARDWARE_SOURCE_COMMIT"]
    jobs = []
    for label, suffix in (("original", ""), ("adaptive", "-adaptive")):
        path = Path(
            "experiments/libero_hardware_interface/preregistration-deepseek-selfhost"
            + suffix
            + ".yaml"
        )
        manifest = load_manifest(path)
        manifest["resolved"] = copy.deepcopy(commissioned["resolved"])
        manifest["readiness"] = copy.deepcopy(commissioned["readiness"])
        out = root / label
        out.mkdir()
        manifest_path = (out / "preregistration.yaml").resolve()
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        catalog = strict_json((root / "prepared/catalog.json").read_bytes())
        rows = schedule(manifest, catalog)
        plan = create_fresh_plan(
            manifest,
            rows,
            source=source,
            source_sha256=source_hash(),
            cohort=parent + "-" + label,
        )
        plan_path = (out / "execution-plan.json").resolve()
        write_json(plan_path, plan)
        for index in range(4):
            worker = out / ("worker-%d" % index)
            worker.mkdir()
            (worker / "prepared").symlink_to(
                (root / "prepared").resolve(), target_is_directory=True
            )
            environment = {
                **os.environ,
                "HARDWARE_WORKFLOW": parent + "-" + label + "-s" + str(index),
                "HARDWARE_PARENT_WORKFLOW": parent,
                "HARDWARE_MANIFEST": str(manifest_path),
                "HARDWARE_EXECUTION_PLAN": str(plan_path),
                "HARDWARE_SHARD_INDEX": str(index),
                "HARDWARE_ORCHESTRATION_COMMIT": source,
                "HARDWARE_REQUEST_SLOT_INDEX": str(2 * index + (0 if suffix else 1)),
                "MUJOCO_EGL_DEVICE_ID": str(2 * index + (0 if suffix else 1)),
                "HARDWARE_WORKER_ROOT": str(worker.resolve()),
            }
            log = (worker / "worker.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.libero_hardware_interface.tools.self_hosted_campaign",
                    "shard",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((process, log, worker))
    write_json(
        root / "campaign-started.json",
        {
            "workflow": parent,
            "workers": 8,
            "planned_trials": 250,
            "source_commit": source,
        },
    )
    print(
        json.dumps({"campaign_started": True, "workers": 8, "trials": 250}), flush=True
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
                        json.dumps({"worker_finished": str(worker), "exit_code": code}),
                        flush=True,
                    )
                    if code:
                        print((worker / "worker.log").read_text()[-6000:], flush=True)
                        raise RuntimeError("collection_paused_on_worker_failure")
            jobs = remaining
            if jobs:
                time.sleep(5)
    finally:
        # Never stop a peer mid-trial. On infrastructure failure, each peer
        # drains its active verified trial and halts at the next trial boundary.
        for _, _, worker in jobs:
            (worker / "stop-after-current-trial").touch(exist_ok=True)
        for process, log, _ in jobs:
            process.wait()
            log.close()
    for label, suffix in (("original", ""), ("adaptive", "-adaptive")):
        out = root / label
        manifest = load_manifest(out / "preregistration.yaml")
        outcomes = []
        for worker in sorted(out.glob("worker-*")):
            outcomes.extend(launcher.load_outcomes(worker / "pilot/runs"))
        write_json(out / "verified-outcomes.json", outcomes)
        expected = 100 if suffix else 150
        if len(outcomes) != expected or any(
            r.get("independent_evaluation_passed") is not True for r in outcomes
        ):
            raise RuntimeError("self_hosted_campaign_incomplete")
        export(outcomes, manifest, out / "results", split="pilot")
    write_json(
        root / "campaign-completed.json",
        {"trials": 250, "independent_replay": "passed"},
    )


def shard():
    from .parallel_worker import collect_shard

    root = Path(os.environ["HARDWARE_WORKER_ROOT"])
    # The launcher checks the marker at exclusive trial creation boundaries.
    os.environ["HARDWARE_STOP_AFTER_TRIAL_FILE"] = str(
        root / "stop-after-current-trial"
    )
    manifest = load_manifest(os.environ["HARDWARE_MANIFEST"])
    write_json(
        root / "request-admission.json",
        {
            "mode": "self_hosted_continuous_batching",
            "artificial_client_pacing": False,
            "concurrent_actor_workers": 8,
            "server_max_num_seqs": 8,
        },
    )
    try:
        collect_shard(root, manifest)
    finally:
        osmo_worker.archive(root, os.environ["HARDWARE_WORKFLOW"])


def main():
    root = Path("artifacts")
    root.mkdir(exist_ok=True)
    host = serving_host(os.environ["HARDWARE_INFERENCE_HOST"])
    server = relay(host)
    original_archive = osmo_worker.archive

    def archive_with_serving(directory, workflow, label="final"):
        if Path(directory).resolve() == root.resolve() and label == "final":
            try:
                with urllib.request.urlopen(
                    "http://" + host + ":8001/artifacts", timeout=60
                ) as response:
                    (root / "serving-artifacts.tar.gz").write_bytes(response.read())
            except OSError as error:
                write_json(
                    root / "serving-artifact-error.json",
                    {"error_class": type(error).__name__},
                )
        return original_archive(directory, workflow, label)

    osmo_worker.archive = archive_with_serving
    try:
        wait_for_server(host, root / "serving-identity.json")
        os.environ["HARDWARE_STAGE"] = "pilot"
        osmo_worker.collect_pilot = collect_cohorts
        osmo_worker.main()
    finally:
        try:
            request = urllib.request.Request(
                "http://" + host + ":8001/finish",
                data=json.dumps({"workflow": os.environ["HARDWARE_WORKFLOW"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10):
                pass
        except OSError:
            pass
        finally:
            server.shutdown()


if __name__ == "__main__":
    shard() if sys.argv[1:] == ["shard"] else main()
