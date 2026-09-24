"""Run one unchanged OOD shard on an independently scheduled L40S task.

Each workflow has worker0 through worker7. Upload the same verified payload to
/osmo/run/workspace/payload.tar.gz on EVERY task. ASTRA_WORKER_INDEX selects the
original eight-GPU job; ASTRA_SHARD_GROUP_INDEX/GROUPS retain the global partition.
Every task performs its own immutable-input, checkpoint and numerical preflight.
Archives use <workflow_id>/worker_<index>; they never share a writable prefix.
"""

import importlib.metadata
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

from astra_reversal.evaluate import paired_comparison, read_episodes, summarize
from astra_reversal.osmo import ood_steering as steering
from astra_reversal.osmo.experiment import RESULTS, Archive
from astra_reversal.osmo.runtime_probe import SOLVER_OPTIONS, TOLERANCES, write_json
from astra_reversal.records import digest, file_sha256


def assignment(worker_index, group):
    steering.require(
        type(worker_index) is int and 0 <= worker_index < 8,
        "ASTRA_WORKER_INDEX must be an integer in [0, 7]",
    )
    steering.require(
        1 <= group["count"] <= 25
        and 0 <= group["index"] < group["count"]
        and group["global_shards_per_suite"] == 4 * group["count"],
        "Invalid distributed OOD shard group",
    )
    return {
        "worker_index": worker_index,
        "suite": steering.SUITES[worker_index // 4],
        "local_shard": worker_index % 4,
        "shard": 4 * group["index"] + worker_index % 4,
        "num_shards": group["global_shards_per_suite"],
        "physical_gpu": 0,
    }


def archive_prefix(workflow, worker_index):
    steering.require(
        type(worker_index) is int and 0 <= worker_index < 8,
        "Invalid archive worker index",
    )
    return f"experiments/astra-reversal-20260924/{workflow}/worker_{worker_index}"


class WorkerArchive(Archive):
    """Reuse the established archival implementation with an isolated prefix."""

    def __init__(self, worker_index):
        import boto3

        self.prefix = archive_prefix(os.environ["ASTRA_RUN_ID"], worker_index)
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        if self.client.list_objects_v2(
            Bucket="rldb",
            Prefix=self.prefix + "/",
            MaxKeys=1,
        ).get("KeyCount"):
            raise FileExistsError("This worker's remote artifact prefix already exists")
        self.uploaded = {}


def freeze_worker_inputs(methods, report, steps, group, worker_index):
    """Use the original freezer verbatim, including byte-identical configurations."""
    target = assignment(worker_index, group)
    base, manifests, all_jobs, diagnostics_path = steering.freeze_inputs(
        methods, report, steps, group
    )
    jobs = []
    for job in all_jobs:
        if job["physical_gpu"] != worker_index:
            continue
        steering.require(
            all(
                job[key] == target[key]
                for key in ("suite", "local_shard", "shard", "num_shards")
            ),
            "Distributed assignment differs from the original group partition",
        )
        jobs.append({**job, "physical_gpu": 0, "worker_index": worker_index})
    steering.require(
        [job["method"] for job in jobs] == methods
        and len({tuple(job["episode_ids"]) for job in jobs}) == 1,
        "Each method must receive the same single assigned shard",
    )
    path = RESULTS / "frozen_plan.json"
    plan = json.loads(path.read_text())
    original_hash = plan.pop("plan_sha256")
    plan.update(
        execution_topology="independent_single_gpu_tasks",
        worker_index=worker_index,
        workers_per_group=8,
        group_plan_sha256=original_hash,
        group_assigned_episodes_per_method=plan["assigned_episodes_per_method"],
        assigned_episodes_per_method=len(jobs[0]["episode_ids"]),
        jobs=jobs,
    )
    plan["plan_sha256"] = digest(plan)
    write_json(path, plan)
    return base, manifests, jobs, diagnostics_path, plan


def run_command(job, diagnostics_path):
    command = [
        sys.executable,
        "-m",
        "astra_reversal.osmo.ood_steering",
        "--worker",
        "run",
        "--config",
        job["config"],
        "--manifest",
        job["manifest"],
        "--libero-root",
        steering.LIBERO,
        "--shard-index",
        str(job["shard"]),
        "--num-shards",
        str(job["num_shards"]),
        "--output",
        job["output"],
    ]
    if job["method"] == "reversal":
        agent = [
            sys.executable,
            "-m",
            "astra_reversal.astra_client",
            "--endpoint",
            steering.ENDPOINT,
            "--model",
            steering.MODEL,
            "--response-log",
            job["api_log"],
            "--reasoning-effort",
            "low",
            "--max-completion-tokens",
            "8192",
            "--timeout",
            "170",
        ]
        command.extend(
            [
                "--diagnostics",
                str(diagnostics_path),
                "--agent-command",
                shlex.join(agent),
            ]
        )
    return command


def child_environment(method):
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        MUJOCO_EGL_DEVICE_ID="0",
        NVIDIA_TF32_OVERRIDE="0",
    )
    if method != "reversal":
        environment.pop("NVIDIA_INFERENCE_API_KEY", None)
    return environment


def wait_process(process, archive, *, phase, worker_index, group, job=None):
    while True:
        code = process.poll()
        if code not in (None, 0):
            raise RuntimeError(f"{phase} failed; inspect the worker's archived log")
        progress = {
            "phase": phase,
            "worker_index": worker_index,
            "shard_group": group,
            "full_manifest_episodes": 200,
            "completed": 0,
        }
        if job is not None:
            rows = steering.progress_rows(Path(job["output"]) / "events.jsonl")
            progress.update(
                suite=job["suite"],
                shard=job["shard"],
                num_shards=job["num_shards"],
                assigned_total=len(job["episode_ids"]),
                completed=len(rows),
                successes=sum(row["success"] for row in rows),
                execution_errors=sum(row.get("failure") is not None for row in rows),
            )
        write_json(RESULTS / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        archive.sync()
        if code == 0:
            return
        time.sleep(30)


def summarize_worker_method(job, manifests, diagnostic, group):
    rows = read_episodes(job["output"])
    ids = [row["episode_id"] for row in rows]
    steering.require(
        len(ids) == len(set(ids)) == len(job["episode_ids"])
        and set(ids) == set(job["episode_ids"])
        and all(
            row["method"] == job["method"] and row["suite"] == job["suite"]
            for row in rows
        ),
        "Worker has incomplete, duplicate or incorrectly assigned episodes",
    )
    run_manifest = json.loads((Path(job["output"]) / "manifest.json").read_text())
    steering.require(
        run_manifest["checkpoint"] == diagnostic["checkpoint"]
        and run_manifest["action_spec"]["action_spec_id"]
        == diagnostic["action_spec_id"]
        and run_manifest["task_manifest_sha256"] == manifests[job["suite"]]["sha256"]
        and run_manifest["config"] == json.loads(Path(job["config"]).read_text()),
        "Evaluated worker provenance differs from the frozen configuration or gate",
    )
    suite_summary = {
        **summarize(rows),
        "successes": sum(row["success"] for row in rows),
        "execution_errors": sum(row.get("failure") is not None for row in rows),
        "task_manifest_sha256": manifests[job["suite"]]["sha256"],
        "expected_assigned_episodes": len(ids),
        "full_suite_episodes": 100,
        "complete_frozen_suite": len(ids) == 100,
        "costs": steering.aggregate_costs([job]),
    }
    method_summary = {
        **suite_summary,
        "method": job["method"],
        "worker_index": job["worker_index"],
        "shard_group": group,
        "suites": {job["suite"]: suite_summary},
        "full_manifest_episodes": 200,
        "complete_frozen_benchmark": False,
        "selected_solver": diagnostic["selected_solver"],
        "success_rate": suite_summary["successes"] / len(ids),
    }
    directory = RESULTS / job["method"]
    for path, value in (
        (directory / job["suite"] / "summary.json", suite_summary),
        (directory / job["suite"] / "episodes.json", rows),
        (directory / "summary.json", method_summary),
        (directory / "episodes.json", rows),
    ):
        write_json(path, value)
    return method_summary, rows


def main():
    import torch
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    worker_index = int(os.environ["ASTRA_WORKER_INDEX"])
    group = steering.shard_group()
    target = assignment(worker_index, group)
    methods = steering.selected_methods(
        os.environ.get("ASTRA_OOD_METHODS", ",".join(steering.METHODS))
    )
    steering.require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "Each distributed OOD task requires exactly one allocated L40S",
    )
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = WorkerArchive(worker_index)
    processes, streams, reports, outcomes = [], [], {}, {}
    identity = {
        "workflow": os.environ["ASTRA_RUN_ID"],
        "worker_index": worker_index,
        "execution_topology": "independent_single_gpu_tasks",
        "workers_per_group": 8,
        "shard_group": group,
        "full_manifest_episodes": 200,
    }
    try:
        diagnostic = json.loads(
            (steering.INPUTS / "runtime_diagnostics.json").read_text()
        )
        steps = steering.select_solver(diagnostic)
        base, manifests, jobs, diagnostics_path, plan = freeze_worker_inputs(
            methods,
            diagnostic,
            steps,
            group,
            worker_index,
        )
        identity.update(
            assigned_episodes_per_method=plan["assigned_episodes_per_method"],
            group_assigned_episodes_per_method=plan[
                "group_assigned_episodes_per_method"
            ],
        )
        assets = steering.verify_assets(base, diagnostic)
        runtime = {
            **identity,
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": 1,
            "physical_gpu": 0,
            "assignment": target,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "tf32": False,
            "methods": methods,
            "selected_solver": diagnostic["selected_solver"],
            "verified_asset_sha256": assets,
            "archive_prefix": archive.prefix,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "mujoco",
                    "robosuite",
                    "transformers",
                    "lerobot",
                )
            },
            "astra_endpoint": steering.ENDPOINT,
            "astra_model": steering.MODEL,
            "astra_cache": {"no-cache": True},
            "protocol_notes": [
                "Scheduling-only change: configuration bytes, frozen resets, agent prompt, model, solver, and noise schedule match the eight-GPU group protocol.",
                "This worker evaluates one declared global shard; its result is not a full-suite estimate.",
                "Runtime and genuine-proposal numerical gates are repeated on this worker before any assigned test episode.",
            ],
        }
        steering.require(
            runtime["packages"]["torch"] == diagnostic["checkpoint"]["torch_version"]
            and runtime["packages"]["transformers"]
            == diagnostic["checkpoint"]["transformers_version"],
            "Worker package versions differ from numerical validation",
        )
        write_json(RESULTS / "runtime.json", runtime)
        write_json(RESULTS / "status.json", {"status": "preflight", **identity})
        archive.sync()

        if "reversal" in methods:
            steering.require(
                bool(os.environ.get("NVIDIA_INFERENCE_API_KEY")),
                "Missing injected NVIDIA inference credential",
            )
            proposal_path = RESULTS / "astra_proposal_diagnostics.json"
            command = [
                sys.executable,
                "-m",
                "astra_reversal.osmo.proposal_probe",
                "--runtime-diagnostics",
                str(diagnostics_path),
                "--proposal-input",
                str(RESULTS / "frozen/astra-proposal-input"),
                "--device",
                "cuda",
                "--output",
                str(proposal_path),
            ]
            stream = (RESULTS / "astra_proposal_diagnostics.log").open("w")
            streams.append(stream)
            process = subprocess.Popen(
                command,
                env=child_environment("preflight"),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            wait_process(
                process,
                archive,
                phase="genuine_astra_development_preflight",
                worker_index=worker_index,
                group=group,
            )
            proposal = json.loads(proposal_path.read_text())
            steering.require(
                proposal["status"] == "passed"
                and proposal["actual_model"] == steering.MODEL
                and proposal["checkpoint"] == diagnostic["checkpoint"]
                and proposal["action_spec_id"] == diagnostic["action_spec_id"]
                and proposal["tolerances"] == TOLERANCES
                and proposal["solver"] == "rk4"
                and proposal["solver_options"] == SOLVER_OPTIONS
                and proposal["actual_controller_proposal"]["passed"] is True
                and 0
                <= proposal["actual_controller_proposal"]["reconstruction"]["max_abs"]
                <= TOLERANCES["action_atol"]
                and any(
                    row["steps"] == steps
                    and row["passed"] is True
                    and steering.metrics_pass(row)
                    for row in proposal["results"]
                ),
                "Worker's genuine Astra proposal failed unchanged numerical acceptance",
            )
            write_json(
                RESULTS / "accepted_gates.json",
                {
                    "runtime_diagnostics_sha256": file_sha256(diagnostics_path),
                    "astra_proposal_diagnostics_sha256": file_sha256(proposal_path),
                    "selected_steps": steps,
                    "tolerances": TOLERANCES,
                },
            )

        for job in jobs:
            command = run_command(job, diagnostics_path)
            write_json(
                Path(job["log"]).with_suffix(".command.json"),
                {
                    "argv": command,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "MUJOCO_EGL_DEVICE_ID": "0",
                    "tf32": False,
                    "worker_index": worker_index,
                },
            )
            stream = Path(job["log"]).open("w")
            streams.append(stream)
            process = subprocess.Popen(
                command,
                env=child_environment(job["method"]),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            wait_process(
                process,
                archive,
                phase=job["method"],
                worker_index=worker_index,
                group=group,
                job=job,
            )
            reports[job["method"]], outcomes[job["method"]] = summarize_worker_method(
                job, manifests, diagnostic, group
            )
            write_json(
                RESULTS / "summary.json",
                {**identity, "methods": reports, "completed_methods": list(reports)},
            )
            archive.sync()

        comparisons = {
            f"{right}_minus_{left}": paired_comparison(
                outcomes[left], outcomes[right], seed=7
            )
            for left, right in itertools.combinations(methods, 2)
        }
        paired = {
            episode_id: {
                method: next(
                    row for row in outcomes[method] if row["episode_id"] == episode_id
                )
                for method in methods
            }
            for episode_id in jobs[0]["episode_ids"]
        }
        write_json(RESULTS / "paired_outcomes.json", paired)
        result = {
            **identity,
            "methods": reports,
            "paired_comparisons": comparisons,
            "complete_frozen_benchmark": False,
            "complete_pairing_within_worker": True,
            "total_evaluated_episodes": plan["assigned_episodes_per_method"]
            * len(methods),
            "execution_errors": sum(
                report["execution_errors"] for report in reports.values()
            ),
        }
        write_json(RESULTS / "summary.json", result)
        write_json(
            RESULTS / "status.json",
            {
                "status": "complete"
                if not result["execution_errors"]
                else "completed_with_execution_errors",
                **result,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "complete",
                    **identity,
                    "total_evaluated_episodes": result["total_evaluated_episodes"],
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        steering.terminate(processes)
        write_json(
            RESULTS / "status.json",
            {
                "status": "failed",
                **identity,
                "completed_methods": list(reports),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        for stream in streams:
            stream.close()
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    main()
