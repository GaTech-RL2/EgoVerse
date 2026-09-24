"""Frozen, paired LIBERO-OOD controls and genuine Astra Stage 1 reversal.

Package --include-ood-inputs and the three public astra-proposal-input files
before submission. ASTRA_OOD_METHODS can select a comma-separated subset of
policy_fresh,policy_reused,reversal; execution always follows that order.
ASTRA_SHARD_GROUPS and ASTRA_SHARD_GROUP_INDEX divide the frozen manifests
between disjoint eight-GPU workflows; each report declares its assigned subset.
No test outcomes are used to select a solver, prompt, or method configuration.
"""

import importlib.metadata
import itertools
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.evaluate import (
    paired_comparison,
    read_episodes,
    read_events,
    summarize,
)
from astra_reversal.libero_runner import load_task_manifest, shard_entries
from astra_reversal.osmo.experiment import RESULTS, ROOT, Archive
from astra_reversal.osmo.runtime_probe import (
    EXPECTED_CONDITIONS,
    SOLVER_OPTIONS,
    TOLERANCES,
    write_json,
)
from astra_reversal.records import digest, file_sha256

SUITES = ("libero_goal_ood", "libero_spatial_ood")
METHODS = ("policy_fresh", "policy_reused", "reversal")
MODEL = "azure/openai/gpt-6-astra"
ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
LIBERO = "astra_reversal/.deps/libero-ood/third_party/modified_libero"
INPUTS = ROOT / "astra_reversal/.deps/ood-inputs"
PROPOSAL_INPUT = ROOT / "astra_reversal/.deps/astra-proposal-input"
SHARDS = 4


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_methods(value):
    requested = value.replace(",", " ").split()
    require(
        requested
        and len(requested) == len(set(requested))
        and set(requested) <= set(METHODS),
        "Invalid ASTRA_OOD_METHODS subset",
    )
    return [method for method in METHODS if method in requested]


def shard_group():
    count = int(os.environ.get("ASTRA_SHARD_GROUPS", "1"))
    index = int(os.environ.get("ASTRA_SHARD_GROUP_INDEX", "0"))
    require(1 <= count <= 25 and 0 <= index < count, "Invalid OOD shard group")
    return {"index": index, "count": count, "global_shards_per_suite": SHARDS * count}


def metrics_pass(row):
    checks = {
        "full_internal_reconstruction": "action_atol",
        "controller_reconstruction": "action_atol",
        "known_noise_recovery": "noise_atol",
        "reference_sampler_parity": "parity_atol",
    }
    return all(
        type(row[name]["max_abs"]) in (int, float)
        and math.isfinite(row[name]["max_abs"])
        and 0 <= row[name]["max_abs"] <= TOLERANCES[tolerance]
        for name, tolerance in checks.items()
    )


def select_solver(report):
    """Check all recorded conditions and fixed limits, not only an aggregate flag."""
    require(
        report["schema_version"] == "1.0"
        and report["status"] == "complete_passing_solver"
        and report["solver"] == "rk4"
        and report["solver_options"] == SOLVER_OPTIONS
        and report["tolerances"] == TOLERANCES
        and report["source"]["suite"] == "libero_10"
        and report["source"]["split"] == "development"
        and report["source"]["conditions"] == EXPECTED_CONDITIONS
        and report["checkpoint_matches_recorded_source"]["matched"] is True
        and report["runtime"]["device"] == "cuda"
        and report["runtime"]["tf32"] is False
        and "L40S" in report["runtime"]["gpu"],
        "A completed, unchanged development runtime diagnostic is required",
    )
    condition_ids = [row["condition_id"] for row in report["full_condition_order"]]
    require(
        len(condition_ids) == len(set(condition_ids)) == EXPECTED_CONDITIONS,
        "Runtime report must cover 14 distinct development conditions",
    )
    require(
        set(report["native_parity"]) == set(condition_ids)
        and all(
            row["passed"] is True
            and 0 <= row["error"]["max_abs"] <= TOLERANCES["parity_atol"]
            for row in report["native_parity"].values()
        ),
        "Every development condition must pass native sampler parity",
    )
    require(
        [row["steps"] for row in report["results"]] == [100, 200, 500],
        "Runtime candidates must be the frozen RK4 100/200/500 sweep",
    )
    passing = []
    for row in report["results"]:
        if row["passed"] is not True:
            continue
        conditions = row["conditions"]
        require(
            row["status"] == "complete"
            and row["conditions_tested"]
            == row["conditions_required"]
            == row["passing_conditions"]
            == EXPECTED_CONDITIONS
            and len(conditions) == EXPECTED_CONDITIONS
            and {item["condition_id"] for item in conditions} == set(condition_ids)
            and row["velocity_evaluations_per_solve"] == 4 * row["steps"]
            and metrics_pass(row)
            and all(
                item["passed"] is True
                and item["steps"] == row["steps"]
                and item["full_internal_shape"] == [1, 10, 32]
                and metrics_pass(item)
                for item in conditions
            ),
            "A claimed passing solver failed the 14-condition numerical gate",
        )
        passing.append(row["steps"])
    require(passing, "No solver passes every recorded development condition")
    selected = report["selected_solver"]
    require(
        selected["solver"] == "rk4"
        and selected["steps"] == min(passing)
        and selected["solver_options"] == SOLVER_OPTIONS
        and selected["conditions_passed"] == EXPECTED_CONDITIONS
        and selected["velocity_evaluations_per_solve"] == 4 * min(passing),
        "Selected solver differs from the frozen minimum-cost passing rule",
    )
    return selected["steps"]


def make_config(base, suite, method, steps):
    config = json.loads(json.dumps(base))
    config.update(
        seed=7, method=method, benchmark=asdict(BenchmarkConfig.preset(suite))
    )
    config["policy"]["device"] = "cuda"
    config["agent"].update(
        model_version=MODEL,
        sampling_settings={"reasoning_effort": "low", "max_completion_tokens": 8192},
        refresh_env_steps=20,
        invalid_response_retries=1,
        request_timeout_seconds=180.0,
    )
    config["flow"].update(
        integrator="rk4",
        inversion_steps=steps,
        generation_steps=steps,
        time_power=3.0,
        noise_mix_rho=0.0,
    )
    config["evaluation"].update(split="test", save_flow_traces=False)
    return asdict(RunConfig.from_dict(config))


def copy_frozen(source, destination, hashes):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    checksum = file_sha256(source)
    require(file_sha256(destination) == checksum, "Frozen artifact copy mismatch")
    hashes[str(destination.relative_to(RESULTS))] = checksum


def verify_assets(config, report):
    from huggingface_hub import snapshot_download

    verified = {}
    for inventory_name, location in (
        ("lerobot_pi05_libero_base.json", config["policy"]["checkpoint"]),
        ("paligemma_tokenizer.json", config["policy"]["tokenizer_path"]),
    ):
        inventory = json.loads(
            (ROOT / "astra_reversal/checkpoints" / inventory_name).read_text()
        )
        if inventory_name.startswith("lerobot"):
            snapshot_download(
                inventory["repo_id"],
                revision=inventory["revision"],
                local_dir=ROOT / location,
                token=False,
            )
        for item in inventory["files"]:
            path = ROOT / location / item["path"]
            checksum = file_sha256(path)
            require(
                checksum == item["sha256"], f"Pinned asset hash mismatch: {path.name}"
            )
            verified[str(path.relative_to(ROOT))] = checksum
    reference = json.loads(
        (
            ROOT / "astra_reversal/checkpoints/openpi_libero_input_assets.json"
        ).read_text()
    )
    for key, filename in (
        ("norm_stats", "norm_stats.json"),
        ("tokenizer", "paligemma_tokenizer.model"),
    ):
        path = ROOT / config["policy"]["reference_assets"] / filename
        require(
            file_sha256(path) == reference[key]["sha256"],
            "Reference input asset mismatch",
        )
        verified[str(path.relative_to(ROOT))] = reference[key]["sha256"]
    metadata = report["checkpoint"]
    require(
        metadata["requested_artifact"] == config["policy"]["checkpoint"]
        and metadata["artifact"]
        == str((ROOT / config["policy"]["checkpoint"]).resolve())
        and metadata["tokenizer_path"] == config["policy"]["tokenizer_path"]
        and metadata["device"] == "cuda"
        and metadata["frozen"] is True
        and metadata["horizon"] == 10
        and metadata["model_action_dim"] == 32
        and metadata["input_profile"] == "openpi_libero"
        and metadata["input_profile_assets"] == reference
        and config["policy"]["execute_steps"] == 5,
        "Policy paths, device, or input convention differs from runtime diagnostics",
    )
    for field, filename in (
        ("adapter_source_sha256", "lerobot_policy.py"),
        ("input_profile_source_sha256", "openpi_inputs.py"),
    ):
        require(
            metadata[field] == file_sha256(ROOT / "astra_reversal" / filename),
            "Policy source differs from numerical validation",
        )
    return verified


def freeze_inputs(methods, report, steps, group):
    """Resolve every source, manifest and method config before any test episode."""
    hashes, manifests, jobs = {}, {}, []
    frozen = RESULTS / "frozen"
    diagnostic = frozen / "runtime_diagnostics.json"
    copy_frozen(INPUTS / "runtime_diagnostics.json", diagnostic, hashes)
    package = ROOT / "astra_reversal"
    sources = [
        *package.glob("*.py"),
        *package.glob("requirements*.txt"),
        *package.glob("*.md"),
        *package.joinpath("osmo").glob("*.py"),
        *package.joinpath("osmo").glob("*.yaml"),
        package / "osmo/bootstrap.sh",
        *package.joinpath("configs").glob("*.json"),
        *package.joinpath("checkpoints").glob("*.json"),
    ]
    for path in sorted(sources):
        copy_frozen(path, frozen / "source" / path.relative_to(ROOT), hashes)
    if "reversal" in methods:
        for filename in ("request.json", "response.json", "provider.jsonl"):
            copy_frozen(
                PROPOSAL_INPUT / filename,
                frozen / "astra-proposal-input" / filename,
                hashes,
            )
    base = json.loads(
        (package / "configs/pi05_libero_openpi_inputs_baseline.json").read_text()
    )
    for suite_index, suite in enumerate(SUITES):
        manifest_path = frozen / f"{suite}_manifest.json"
        copy_frozen(INPUTS / manifest_path.name, manifest_path, hashes)
        config = make_config(base, suite, methods[0], steps)
        manifest = load_task_manifest(manifest_path, RunConfig.from_dict(config))
        episodes = manifest["episodes"]
        require(
            manifest["schema_version"] == "1.1"
            and manifest["seed"] == 7
            and len(episodes) == len({e["episode_id"] for e in episodes}) == 100
            and {(e["task_id"], e["initial_state_id"]) for e in episodes}
            == set(itertools.product(range(10), repeat=2))
            and all(
                e["suite"] == suite
                and e["seed"] == 7
                and e["initially_successful"] is False
                for e in episodes
            ),
            "Frozen OOD manifest must contain all ten trials of all ten tasks",
        )
        manifests[suite] = manifest
        for method in methods:
            config = make_config(base, suite, method, steps)
            for local_shard in range(SHARDS):
                shard = SHARDS * group["index"] + local_shard
                directory = RESULTS / method / suite
                config_path = directory / f"config_{shard}.json"
                write_json(config_path, config)
                hashes[str(config_path.relative_to(RESULTS))] = file_sha256(config_path)
                entries = shard_entries(
                    manifest, shard, group["global_shards_per_suite"]
                )
                jobs.append(
                    {
                        "method": method,
                        "suite": suite,
                        "shard": shard,
                        "local_shard": local_shard,
                        "num_shards": group["global_shards_per_suite"],
                        "physical_gpu": suite_index * SHARDS + local_shard,
                        "config": str(config_path),
                        "manifest": str(manifest_path),
                        "output": str(directory / f"shard_{shard}"),
                        "log": str(directory / f"shard_{shard}.log"),
                        "api_log": str(directory / f"shard_{shard}_astra.jsonl"),
                        "episode_ids": [entry["episode_id"] for entry in entries],
                    }
                )
    assigned = sum(
        len(job["episode_ids"]) for job in jobs if job["method"] == methods[0]
    )
    plan = {
        "methods": methods,
        "full_manifest_episodes": 200,
        "assigned_episodes_per_method": assigned,
        "shard_group": group,
        "selected_solver": report["selected_solver"],
        "jobs": jobs,
        "frozen_file_sha256": hashes,
        "manifest_sha256": {suite: item["sha256"] for suite, item in manifests.items()},
    }
    plan["plan_sha256"] = digest(plan)
    write_json(RESULTS / "frozen_plan.json", plan)
    return base, manifests, jobs, diagnostic


def progress_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended last line is seen on the next poll.
            if row["kind"] == "episode_end":
                rows.append(row)
    return rows


def wait_processes(processes, archive, *, phase, jobs=()):
    while True:
        codes = [process.poll() for process in processes]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f"{phase} subprocess failed; see archived worker logs")
        progress = {
            "phase": phase,
            "completed": 0,
            "suites": {},
            "full_manifest_episodes": 200,
        }
        for suite in SUITES:
            rows = [
                row
                for job in jobs
                if job["suite"] == suite
                for row in progress_rows(Path(job["output"]) / "events.jsonl")
            ]
            if jobs:
                progress["suites"][suite] = {
                    "completed": len(rows),
                    "total": sum(
                        len(job["episode_ids"]) for job in jobs if job["suite"] == suite
                    ),
                    "full_suite_episodes": 100,
                    "successes": sum(bool(row["success"]) for row in rows),
                    "execution_errors": sum(
                        row.get("failure") is not None for row in rows
                    ),
                }
                progress["completed"] += len(rows)
        progress["assigned_total"] = sum(len(job["episode_ids"]) for job in jobs)
        write_json(RESULTS / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        archive.sync()
        if all(code == 0 for code in codes):
            return
        time.sleep(30)


def api_usage(value, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from api_usage(item, name)
        elif type(item) in (int, float) and math.isfinite(item):
            yield name, item


def aggregate_costs(jobs):
    counts, fallbacks, tokens, models = Counter(), Counter(), Counter(), Counter()
    counts.update(
        {
            name: 0
            for name in (
                "agent_calls",
                "accepted_proposals",
                "invalid_proposals",
                "retry_calls",
                "fallbacks",
                "velocity_evaluations",
                "recorded_api_calls",
                "api_schema_accepted",
                "executed_actions",
                "fallback_actions",
                "latent_actions",
                "nonfallback_latent_actions",
            )
        }
    )
    latency = defaultdict(list)
    maximum_clip = 0.0
    for job in jobs:
        for row in read_events(job["output"]):
            kind = row["kind"]
            if kind == "agent_response":
                counts["agent_calls"] += 1
                counts[
                    "accepted_proposals" if row["accepted"] else "invalid_proposals"
                ] += 1
                counts["retry_calls"] += int(row["attempt"] > 0)
            elif kind == "flow":
                counts["velocity_evaluations"] += row["velocity_evaluations"]
                counts[f"{row['role']}_solves"] += 1
            elif kind == "fallback":
                counts["fallbacks"] += 1
                fallbacks[row["reason"].split(":", 1)[0]] += 1
            elif kind == "control_step":
                step = row["step"]
                fallback = step["fallback_reason"] is not None
                latent = step["latent_id"] is not None
                counts["executed_actions"] += 1
                counts["fallback_actions"] += int(fallback)
                counts["latent_actions"] += int(latent)
                counts["nonfallback_latent_actions"] += int(latent and not fallback)
            elif kind == "generated_actions":
                counts["clipped_action_values"] += row["clipping"]["count"]
                maximum_clip = max(maximum_clip, row["clipping"]["max_abs"])
            if "latency_seconds" in row:
                latency[kind].append(row["latency_seconds"])
        path = Path(job["api_log"])
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    counts["recorded_api_calls"] += 1
                    counts["api_schema_accepted"] += int(record["accepted"])
                    latency["api"].append(record["latency_seconds"])
                    response = record.get("response", {})
                    if response.get("model"):
                        models[response["model"]] += 1
                    tokens.update(dict(api_usage(response.get("usage") or {})))
    return {
        "counts": dict(counts),
        "fallback_categories": dict(fallbacks),
        "invalid_response_rate": counts["invalid_proposals"] / counts["agent_calls"]
        if counts["agent_calls"]
        else None,
        "actual_api_models": dict(models),
        "api_usage_totals": dict(tokens),
        "action_fractions": {
            name: counts[name] / counts["executed_actions"]
            if counts["executed_actions"]
            else None
            for name in (
                "fallback_actions",
                "latent_actions",
                "nonfallback_latent_actions",
            )
        },
        "max_action_clip": maximum_clip,
        "latency_seconds": {
            name: {
                "total": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for name, values in latency.items()
        },
        "currency_cost": None,
        "cost_note": "Measured tokens, latency and velocity evaluations; no billing rate was assumed.",
    }


def summarize_method(method, jobs, manifests, diagnostic):
    episodes, suite_reports = [], {}
    for suite in SUITES:
        suite_jobs = [job for job in jobs if job["suite"] == suite]
        rows = []
        for job in suite_jobs:
            shard = read_episodes(job["output"])
            ids = [row["episode_id"] for row in shard]
            require(
                len(ids) == len(set(ids)) == len(job["episode_ids"])
                and set(ids) == set(job["episode_ids"]),
                "Evaluation shard has incomplete or duplicate episode coverage",
            )
            manifest = json.loads((Path(job["output"]) / "manifest.json").read_text())
            require(
                manifest["checkpoint"] == diagnostic["checkpoint"]
                and manifest["action_spec"]["action_spec_id"]
                == diagnostic["action_spec_id"]
                and manifest["task_manifest_sha256"] == manifests[suite]["sha256"],
                "Evaluated policy/controller differs from the frozen numerical gate",
            )
            rows.extend(shard)
        expected = {
            episode_id for job in suite_jobs for episode_id in job["episode_ids"]
        }
        require(
            len(rows) == len(expected)
            and {row["episode_id"] for row in rows} == expected,
            "Suite must cover exactly its assigned frozen episode subset",
        )
        summary = {
            **summarize(rows),
            "successes": sum(row["success"] for row in rows),
            "execution_errors": sum(row.get("failure") is not None for row in rows),
            "expected_assigned_episodes": len(expected),
            "full_suite_episodes": 100,
            "complete_frozen_suite": len(expected) == 100,
            "task_manifest_sha256": manifests[suite]["sha256"],
            "costs": aggregate_costs(suite_jobs),
        }
        write_json(RESULTS / method / suite / "episodes.json", rows)
        write_json(RESULTS / method / suite / "summary.json", summary)
        suite_reports[suite] = summary
        episodes.extend(rows)
    expected_count = sum(len(job["episode_ids"]) for job in jobs)
    require(
        len(episodes) == len({row["episode_id"] for row in episodes}) == expected_count,
        "Method must cover exactly its assigned paired episode subset",
    )
    summary = {
        **summarize(episodes),
        "method": method,
        "suites": suite_reports,
        "successes": sum(row["success"] for row in episodes),
        "execution_errors": sum(row.get("failure") is not None for row in episodes),
        "expected_assigned_episodes": expected_count,
        "full_manifest_episodes": 200,
        "complete_frozen_benchmark": expected_count == 200,
        "costs": aggregate_costs(jobs),
        "selected_solver": diagnostic["selected_solver"],
    }
    summary["success_rate"] = summary["successes"] / expected_count
    write_json(RESULTS / method / "summary.json", summary)
    write_json(RESULTS / method / "episodes.json", episodes)
    return summary, episodes


def terminate(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def worker(argv):
    import torch

    from astra_reversal.__main__ import main as run

    require(
        torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0),
        "Each OOD worker requires one isolated L40S",
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(argv)


def main():
    import torch
    from lerobot.policies.pi05 import modeling_pi05  # noqa: F401

    methods = selected_methods(os.environ.get("ASTRA_OOD_METHODS", ",".join(METHODS)))
    group = shard_group()
    require(
        torch.cuda.device_count() == 8,
        "OOD steering requires exactly eight allocated GPUs",
    )
    names = [torch.cuda.get_device_name(index) for index in range(8)]
    require(
        all("L40S" in name for name in names), "OOD steering requires eight L40S GPUs"
    )
    RESULTS.mkdir(parents=True, exist_ok=False)
    archive = Archive()
    processes, streams, reports, outcomes = [], [], {}, {}
    try:
        report = json.loads((INPUTS / "runtime_diagnostics.json").read_text())
        steps = select_solver(report)
        base, manifests, jobs, diagnostics_path = freeze_inputs(
            methods, report, steps, group
        )
        assigned = sum(
            len(job["episode_ids"]) for job in jobs if job["method"] == methods[0]
        )
        assets = verify_assets(base, report)
        runtime = {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "payload_sha256": os.environ["PAYLOAD_SHA256"],
            "gpu_names": names,
            "python": platform.python_version(),
            "tf32": False,
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
            "methods": methods,
            "selected_solver": report["selected_solver"],
            "shard_group": group,
            "full_manifest_episodes": 200,
            "assigned_episodes_per_method": assigned,
            "verified_asset_sha256": assets,
            "astra_endpoint": ENDPOINT,
            "astra_model": MODEL,
            "astra_cache": {"no-cache": True},
            "protocol_notes": [
                "Frozen released OOD reset model and dynamic states; seed 7, ten trials per task, 300 actions and ten stabilization steps.",
                "All methods use the same development-selected RK4 solver and cubic time grid; five actions executed per ten-action chunk.",
                "Astra supplies all Stage 1 Hx7 proposal values; failed calls use only the controller's explicit logged baseline fallback.",
                "Lossless inversion inputs/noise are retained; full per-integration-step traces are disabled.",
                "No OOD success-based selection or prompt changes. Checkpoint training overlap is unknown.",
            ],
        }
        require(
            runtime["packages"]["torch"] == report["checkpoint"]["torch_version"]
            and runtime["packages"]["transformers"]
            == report["checkpoint"]["transformers_version"],
            "Runtime package versions differ from numerical validation",
        )
        write_json(RESULTS / "runtime.json", runtime)
        write_json(
            RESULTS / "status.json",
            {
                "status": "preflight",
                "methods": methods,
                "shard_group": group,
                "assigned_episodes_per_method": assigned,
            },
        )
        archive.sync()

        if "reversal" in methods:
            require(
                bool(os.environ.get("NVIDIA_INFERENCE_API_KEY")),
                "Missing injected NVIDIA inference credential",
            )
            proposal_report = RESULTS / "astra_proposal_diagnostics.json"
            stream = (RESULTS / "astra_proposal_diagnostics.log").open("w")
            streams.append(stream)
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
                str(proposal_report),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            wait_processes(
                processes, archive, phase="genuine_astra_development_preflight"
            )
            proposal = json.loads(proposal_report.read_text())
            require(
                proposal["status"] == "passed"
                and proposal["actual_model"] == MODEL
                and proposal["checkpoint"] == report["checkpoint"]
                and proposal["action_spec_id"] == report["action_spec_id"]
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
                    and metrics_pass(row)
                    for row in proposal["results"]
                ),
                "Genuine Astra development proposal failed unchanged numerical acceptance",
            )
            write_json(
                RESULTS / "accepted_gates.json",
                {
                    "runtime_diagnostics_sha256": file_sha256(diagnostics_path),
                    "astra_proposal_diagnostics_sha256": file_sha256(proposal_report),
                    "selected_steps": steps,
                    "tolerances": TOLERANCES,
                },
            )

        for method in methods:
            method_jobs = [job for job in jobs if job["method"] == method]
            processes = []
            for job in method_jobs:
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
                    LIBERO,
                    "--shard-index",
                    str(job["shard"]),
                    "--num-shards",
                    str(job["num_shards"]),
                    "--output",
                    job["output"],
                ]
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=str(job["physical_gpu"]),
                    MUJOCO_EGL_DEVICE_ID=str(job["physical_gpu"]),
                    NVIDIA_TF32_OVERRIDE="0",
                )
                if method == "reversal":
                    agent = [
                        sys.executable,
                        "-m",
                        "astra_reversal.astra_client",
                        "--endpoint",
                        ENDPOINT,
                        "--model",
                        MODEL,
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
                else:
                    environment.pop("NVIDIA_INFERENCE_API_KEY", None)
                write_json(
                    Path(job["log"]).with_suffix(".command.json"),
                    {
                        "argv": command,
                        "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
                        "MUJOCO_EGL_DEVICE_ID": environment["MUJOCO_EGL_DEVICE_ID"],
                        "tf32": False,
                    },
                )
                stream = Path(job["log"]).open("w")
                streams.append(stream)
                processes.append(
                    subprocess.Popen(
                        command,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            wait_processes(processes, archive, phase=method, jobs=method_jobs)
            reports[method], outcomes[method] = summarize_method(
                method, method_jobs, manifests, report
            )
            reports[method]["shard_group"] = group
            write_json(
                RESULTS / "summary.json",
                {"methods": reports, "completed_methods": list(reports)},
            )
            archive.sync()

        comparisons = {}
        for left, right in itertools.combinations(methods, 2):
            comparisons[f"{right}_minus_{left}"] = paired_comparison(
                outcomes[left], outcomes[right], seed=7
            )
        paired = defaultdict(dict)
        for method, rows in outcomes.items():
            for row in rows:
                paired[row["episode_id"]][method] = row
        write_json(RESULTS / "paired_outcomes.json", dict(paired))
        report = {
            "workflow": os.environ["ASTRA_RUN_ID"],
            "methods": reports,
            "paired_comparisons": comparisons,
            "full_manifest_episodes": 200,
            "assigned_episodes_per_method": assigned,
            "shard_group": group,
            "total_evaluated_episodes": assigned * len(methods),
            "complete_frozen_benchmark": assigned == 200,
            "complete_pairing_within_group": all(
                len(values) == len(methods) for values in paired.values()
            ),
            "execution_errors": sum(
                item["execution_errors"] for item in reports.values()
            ),
        }
        write_json(RESULTS / "summary.json", report)
        write_json(
            RESULTS / "status.json",
            {
                "status": "complete"
                if not report["execution_errors"]
                else "completed_with_execution_errors",
                **report,
            },
        )
        print(
            json.dumps(
                {
                    "phase": "complete",
                    "methods": methods,
                    "total_evaluated_episodes": report["total_evaluated_episodes"],
                }
            ),
            flush=True,
        )
    except BaseException as exc:
        terminate(processes)
        write_json(
            RESULTS / "status.json",
            {
                "status": "failed",
                "completed_methods": list(reports),
                "shard_group": group,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        for stream in streams:
            stream.close()
        archive.sync(include_arrays=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2:])
    else:
        main()
