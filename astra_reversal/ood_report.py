"""Validate and aggregate the frozen OOD baseline, full-shard repair and controls.

Run with ``python -m astra_reversal.ood_report --baseline DIR --repair DIR
--groups GROUP0 GROUP1 GROUP2 GROUP3 --output NEW_DIR``. Omit --groups for the
repaired baseline alone. Raw input files are read only; output must be new.
Each logical group can contain the original single-node reports or eight
``worker_0`` through ``worker_7`` directories from independent single-GPU tasks.
"""

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from .config import RunConfig
from .evaluate import paired_comparison
from .libero_runner import load_task_manifest
from .records import digest

SUITES = ("libero_goal_ood", "libero_spatial_ood")
METHODS = ("policy_fresh", "policy_reused", "reversal")
LABELS = {
    "baseline_euler10": "Baseline: Euler 10, fresh noise, TF32 on",
    "matched_fresh_noise": "Matched solver: fresh noise",
    "matched_reused_noise": "Matched solver: reused noise",
    "astra_reversal": "Astra reversal",
}
METHOD_KEYS = dict(zip(METHODS, tuple(LABELS)[1:]))
LATENCY_NOTE = (
    "The complete repaired Goal shard used CUDA_LAUNCH_BLOCKING=1. Its wall "
    "latency is not comparable to the original asynchronous baseline. Reported "
    "wall totals describe these executions, not a controlled timing comparison."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    """Read each JSON file once and retain its exact bytes and provenance."""

    def __init__(self):
        self.raw, self.values, self.sources = {}, {}, {}

    def read(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.values:
            raw = path.read_bytes()
            self.raw[key] = raw
            self.values[key] = json.loads(raw)
            self.sources[key] = {
                "path": key,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        return self.values[key]

    def source(self, path):
        self.read(path)
        return self.sources[str(Path(path).resolve())]


def _number(value, name, *, integer=False):
    require(
        type(value) in ((int,) if integer else (int, float))
        and math.isfinite(value)
        and value >= 0,
        f"Invalid nonnegative {'integer ' if integer else ''}metric: {name}",
    )
    return value


def _manifest(inputs, path, suite):
    raw = inputs.read(path)
    config = RunConfig.from_dict(
        {"seed": 7, "benchmark": {"suite": suite}, "evaluation": {"split": "test"}}
    )
    manifest = load_task_manifest(path, config)
    require(raw == manifest, f"Manifest changed while reading: {path}")
    entries = manifest["episodes"]
    require(
        manifest["schema_version"] == "1.1"
        and manifest["seed"] == 7
        and len(entries) == 100,
        f"{suite}: expected the complete seed-7 OOD scene manifest",
    )
    expected = list(itertools.product(range(10), repeat=2))
    require(
        [(e["task_id"], e["initial_state_id"]) for e in entries] == expected,
        f"{suite}: manifest must contain ten ordered trials of all ten tasks",
    )
    for entry in entries:
        require(
            entry["episode_id"]
            == f"{suite}:task{entry['task_id']}:state{entry['initial_state_id']}"
            and entry["suite"] == suite
            and entry["seed"] == 7
            and entry["initially_successful"] is False
            and isinstance(entry["instruction"], str)
            and bool(entry["instruction"]),
            f"Invalid frozen episode identity: {entry.get('episode_id')}",
        )
    for task in range(10):
        require(
            len({e["instruction"] for e in entries if e["task_id"] == task}) == 1,
            f"{suite} task {task}: inconsistent instructions",
        )
    return manifest


def _validate_rows(rows, expected, method, entries, label):
    require(isinstance(rows, list), f"{label}: episodes must be a list")
    ids = [row["episode_id"] for row in rows]
    require(
        len(ids) == len(set(ids)) == len(expected) and set(ids) == set(expected),
        f"{label}: incomplete, duplicated, or unassigned episode IDs",
    )
    for row in rows:
        entry = entries[row["episode_id"]]
        for name in (
            "suite",
            "task_id",
            "initial_state_id",
            "seed",
            "reset_state_sha256",
        ):
            require(row[name] == entry[name], f"{label}: episode differs in {name}")
        require(
            row["method"] == method
            and row["protocol"] == "released_modified_libero"
            and row["split"] == "test"
            and row["task_action_budget"] == 300
            and type(row["success"]) is bool
            and "failure" in row,
            f"{label}: invalid episode method, protocol, or outcome",
        )
        for name in ("actions", "velocity_evaluations", "fallbacks", "agent_calls"):
            _number(row[name], name, integer=True)
        _number(row["wall_seconds"], "wall_seconds")
        require(row["actions"] <= 300, f"{label}: action budget exceeded")
        require(
            not (row["success"] and row["failure"] is not None),
            f"{label}: success and execution failure are both recorded",
        )
    return {row["episode_id"]: row for row in rows}


def _stats(rows):
    successes = [row for row in rows if row["success"]]
    counts = {
        "episodes": len(rows),
        "successes": len(successes),
        "execution_errors": sum(row["failure"] is not None for row in rows),
        "zero_action_successes": sum(row["actions"] == 0 for row in successes),
        "zero_action_failures": sum(
            row["actions"] == 0 and not row["success"] for row in rows
        ),
        "total_actions": sum(row["actions"] for row in rows),
        "total_wall_seconds": sum(row["wall_seconds"] for row in rows),
        "velocity_evaluations": sum(row["velocity_evaluations"] for row in rows),
        "agent_calls": sum(row["agent_calls"] for row in rows),
        "fallbacks": sum(row["fallbacks"] for row in rows),
    }
    return {
        **counts,
        "success_rate": counts["successes"] / len(rows),
        "mean_actions": counts["total_actions"] / len(rows),
        "actions_to_success": sum(row["actions"] for row in successes) / len(successes)
        if successes
        else None,
        "mean_wall_seconds": counts["total_wall_seconds"] / len(rows),
    }


def _check_summary(summary, rows, label):
    calculated = _stats(rows)
    for name in ("episodes", "successes", "execution_errors"):
        require(summary[name] == calculated[name], f"{label}: stale {name} summary")
    if "success_rate" in summary:
        require(
            math.isclose(
                summary["success_rate"], calculated["success_rate"], abs_tol=1e-12
            ),
            f"{label}: stale success rate summary",
        )


def _cuda_device(value):
    require(
        isinstance(value, str)
        and (value == "cuda" or (value.startswith("cuda:") and value[5:].isdigit())),
        "Expected a CUDA policy device",
    )
    return "cuda"


def _run_identity(manifest):
    checkpoint = copy.deepcopy(manifest["checkpoint"])
    checkpoint["device"] = _cuda_device(checkpoint["device"])
    config = copy.deepcopy(manifest["config"])
    config["policy"]["device"] = _cuda_device(config["policy"]["device"])
    return {
        "checkpoint": checkpoint,
        "config": config,
        "action_spec": manifest["action_spec"],
    }


def _annotate(
    rows,
    inputs,
    path,
    workflow,
    role,
    manifests,
    entries,
    originals=None,
    *,
    worker_index=None,
):
    source = inputs.source(path)
    result = []
    for row in rows:
        provenance = {
            "workflow": workflow,
            "role": role,
            "source_file": source,
            "source_row_sha256": digest(row),
            "task_manifest_sha256": manifests[row["suite"]]["sha256"],
            "reset_model_sha256": entries[row["episode_id"]]["reset_model_sha256"],
            "repair_debugger_affects_wall_latency": role == "baseline_repair",
        }
        if originals is not None:
            provenance["replaces_original_row_sha256"] = digest(
                originals[row["episode_id"]]
            )
        if worker_index is not None:
            provenance["worker_index"] = worker_index
            provenance["execution_id"] = f"{workflow}/worker_{worker_index}"
        result.append({**copy.deepcopy(row), "analysis_provenance": provenance})
    return result


def _baseline(inputs, directory, repair):
    manifests = {
        suite: _manifest(inputs, directory / suite / "test_manifest.json", suite)
        for suite in SUITES
    }
    entries = {e["episode_id"]: e for m in manifests.values() for e in m["episodes"]}
    runtime, summary = (
        inputs.read(directory / "runtime.json"),
        inputs.read(directory / "summary.json"),
    )
    require(
        runtime["workflow"] == summary["workflow"],
        "Original baseline workflow mismatch",
    )
    original_path = directory / "episodes.json"
    original = inputs.read(original_path)
    original_index = _validate_rows(
        original, entries, "policy_fresh", entries, "Original baseline"
    )
    _check_summary(summary, original, "Original baseline")
    for suite, manifest in manifests.items():
        require(
            summary["suites"][suite]["task_manifest_sha256"] == manifest["sha256"],
            "Original baseline summary refers to a different reset manifest",
        )
        _check_summary(
            summary["suites"][suite],
            [r for r in original if r["suite"] == suite],
            suite,
        )

    repair_runtime, repair_summary = (
        inputs.read(repair / "runtime.json"),
        inputs.read(repair / "summary.json"),
    )
    require(
        repair_runtime["workflow"] == repair_summary["workflow"]
        and repair_runtime["workflow"] != runtime["workflow"]
        and repair_runtime["replaces_workflow"] == runtime["workflow"]
        and repair_runtime["replaces_suite"] == SUITES[0]
        and repair_runtime["replaces_shard"] == 3
        and repair_runtime["num_shards"] == 4,
        "Repair must replace the original Goal shard 3 of 4 in a distinct workflow",
    )
    goal_digest = manifests[SUITES[0]]["sha256"]
    for source in (repair_runtime, repair_summary):
        require(
            source["task_manifest_sha256"] == goal_digest,
            "Repair reset manifest differs",
        )
    repair_manifest_path = repair / "test_manifest.json"
    if repair_manifest_path.exists():
        require(
            _manifest(inputs, repair_manifest_path, SUITES[0]) == manifests[SUITES[0]],
            "Repair scenes differ",
        )
    baseline_run = inputs.read(directory / SUITES[0] / "shard_0/manifest.json")
    repair_run = inputs.read(repair / "repair/manifest.json")
    require(
        baseline_run["task_manifest_sha256"]
        == repair_run["task_manifest_sha256"]
        == goal_digest
        and _run_identity(baseline_run) == _run_identity(repair_run),
        "Repair policy/config/controller differs from baseline (only CUDA device index may differ)",
    )
    config = baseline_run["config"]
    require(
        config["method"] == "policy_fresh"
        and config["seed"] == 7
        and config["flow"]["integrator"] == "euler"
        and config["flow"]["generation_steps"] == 10
        and config["flow"].get("time_power", 1.0) == 1.0,
        "Original baseline is not the declared Euler-10 fresh-noise method",
    )
    replacement_ids = [e["episode_id"] for e in manifests[SUITES[0]]["episodes"][3::4]]
    repair_path = repair / "episodes.json"
    replacement = inputs.read(repair_path)
    _validate_rows(
        replacement, replacement_ids, "policy_fresh", entries, "Complete repair shard"
    )
    _check_summary(repair_summary, replacement, "Complete repair shard")
    annotated = _annotate(
        original,
        inputs,
        original_path,
        runtime["workflow"],
        "baseline_original",
        manifests,
        entries,
    )
    repaired = _annotate(
        replacement,
        inputs,
        repair_path,
        repair_runtime["workflow"],
        "baseline_repair",
        manifests,
        entries,
        original_index,
    )
    canonical = {row["episode_id"]: row for row in annotated}
    canonical.update({row["episode_id"]: row for row in repaired})
    rows = [canonical[key] for key in entries]
    require(
        _stats(rows)["execution_errors"] == 0,
        "Canonical baseline still has execution errors; no valid baseline claim",
    )
    require(
        _stats(rows)["zero_action_successes"] == 0,
        "Canonical baseline has zero-action successes; stabilization protocol requires review",
    )
    audit = {
        "original": {
            "workflow": runtime["workflow"],
            **_stats(original),
            "raw_records": "original_baseline_episodes.json",
            "runtime": runtime,
        },
        "repair": {
            "workflow": repair_runtime["workflow"],
            "runtime": repair_runtime,
            "replacement_rule": "Replace all Goal manifest episodes [3::4], irrespective of prior outcome",
            "replaced_episode_ids": replacement_ids,
            "original_shard_counts": _stats(
                [original_index[key] for key in replacement_ids]
            ),
            "replacement_counts": _stats(replacement),
            "raw_records": "repair_episodes.json",
            "latency_note": LATENCY_NOTE,
            "policy_identity_verified_ignoring_only_cuda_device_index": True,
        },
    }
    return manifests, entries, rows, audit, baseline_run


def _sum_costs(parts):
    totals = {
        name: Counter()
        for name in (
            "counts",
            "api_usage_totals",
            "actual_api_models",
            "fallback_categories",
        )
    }
    latency, latency_totals = {}, Counter()
    max_clip = 0.0
    for workflow, cost in parts:
        require(workflow not in latency, "Duplicate source execution in cost summary")
        require(
            {
                "executed_actions",
                "fallback_actions",
                "latent_actions",
                "nonfallback_latent_actions",
                "velocity_evaluations",
                "agent_calls",
            }
            <= set(cost["counts"]),
            "Missing measured stage cost counts",
        )
        for section, accumulator in totals.items():
            for name, value in cost[section].items():
                accumulator[name] += _number(value, f"{section}.{name}")
        latency[workflow] = cost["latency_seconds"]
        for name, values in cost["latency_seconds"].items():
            for metric in ("total", "mean", "p95"):
                _number(values[metric], f"latency.{name}.{metric}")
            latency_totals[name] += values["total"]
        max_clip = max(max_clip, _number(cost["max_action_clip"], "max_action_clip"))
    counts = totals["counts"]
    executed = counts["executed_actions"]
    fractions = {}
    for name in ("fallback_actions", "latent_actions", "nonfallback_latent_actions"):
        require(counts[name] <= executed, f"Cost {name} exceeds executed actions")
        fractions[name] = counts[name] / executed if executed else None
    return {
        **{name: dict(value) for name, value in totals.items()},
        "action_fractions": fractions,
        "latency_seconds_by_execution": latency,
        "latency_seconds_totals": dict(latency_totals),
        "latency_note": "Each source workflow or distributed worker retains its own mean and p95; no pooled p95 is inferred.",
        "max_action_clip": max_clip,
        "currency_cost": None,
    }


def _group_sources(directory):
    workers = sorted(path for path in directory.glob("worker_*") if path.is_dir())
    if not workers:
        return [(directory, None)]
    require(
        {path.name for path in workers} == {f"worker_{i}" for i in range(8)}
        and not (directory / "frozen_plan.json").exists(),
        "Distributed group needs exactly worker_0 through worker_7 and no ambiguous group plan",
    )
    return [(directory / f"worker_{i}", i) for i in range(8)]


def _execution(inputs, directory, worker, manifests, entries, baseline_run, identity):
    """Validate one actual execution: an eight-GPU task or one single-GPU worker."""
    plan = inputs.read(directory / "frozen_plan.json")
    require(
        digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        == plan["plan_sha256"],
        "Frozen group plan hash mismatch",
    )
    group = plan["shard_group"]
    index = group["index"]
    require(
        type(index) is int
        and index in range(4)
        and group["count"] == 4
        and group["global_shards_per_suite"] == 16,
        "Invalid steering shard group",
    )
    runtime = inputs.read(directory / "runtime.json")
    summary = inputs.read(directory / "summary.json")
    workflow = runtime["workflow"]
    require(
        summary["workflow"] == workflow
        and runtime["shard_group"] == summary["shard_group"] == group,
        "Steering workflow/group identity mismatch",
    )
    if worker is None:
        require(
            all(
                value.get("worker_index") is None for value in (plan, runtime, summary)
            ),
            "A distributed worker must be supplied through its complete group directory",
        )
        suites, shards = SUITES, range(index * 4, index * 4 + 4)
        execution_id = workflow
    else:
        require(
            all(
                type(value.get("worker_index")) is int
                and value["worker_index"] == worker
                for value in (plan, runtime, summary)
            ),
            "Worker directory and recorded worker index disagree",
        )
        suites = (SUITES[worker // 4],)
        shards = (index * 4 + worker % 4,)
        execution_id = f"{workflow}/worker_{worker}"
        require(
            all(
                value.get("workers_per_group") == 8
                for value in (plan, runtime, summary)
            )
            and runtime.get("visible_gpu_count") == 1,
            "Distributed runtime must declare eight independent single-GPU workers",
        )
        expected_assignment = {
            "worker_index": worker,
            "suite": suites[0],
            "local_shard": worker % 4,
            "shard": shards[0],
            "num_shards": 16,
            "physical_gpu": 0,
        }
        require(
            runtime.get("assignment") == expected_assignment,
            "Worker runtime assignment differs from the frozen partition",
        )
        group_assigned = sum(
            len(manifests[suite]["episodes"][shard::16])
            for suite in SUITES
            for shard in range(index * 4, index * 4 + 4)
        )
        require(
            all(
                value.get("group_assigned_episodes_per_method") == group_assigned
                for value in (plan, runtime, summary)
            ),
            "Distributed logical group assignment count mismatch",
        )
    require(
        plan["methods"] == runtime["methods"] == list(METHODS)
        and set(summary["methods"]) == set(METHODS),
        "Every execution must complete all three distinct steering methods",
    )
    selected = plan["selected_solver"]
    assets = runtime["verified_asset_sha256"]
    require(
        runtime["selected_solver"] == selected and runtime["tf32"] is False,
        "Runtime differs from frozen solver or TF32-off protocol",
    )
    if "solver" not in identity:
        identity.update(solver=selected, assets=assets, configs={})
    require(
        selected == identity["solver"] and assets == identity["assets"],
        "Steering executions use different selected solvers or policy assets",
    )
    require(
        selected["solver"] == "rk4"
        and type(selected["steps"]) is int
        and selected["steps"] > 0
        and selected["solver_options"] == {"time_power": 3.0},
        "Unexpected matched solver identity",
    )
    checkpoint = baseline_run["checkpoint"]
    for filename, checksum in checkpoint["artifact_sha256"].items():
        key = str(Path(checkpoint["requested_artifact"]) / filename)
        require(
            assets.get(key) == checksum,
            f"Steering checkpoint asset differs: {filename}",
        )
    for suite in SUITES:
        manifest = manifests[suite]
        require(
            plan["manifest_sha256"][suite] == manifest["sha256"],
            "Group frozen reset manifest differs from baseline",
        )
        relative = f"frozen/{suite}_manifest.json"
        baseline_source = next(
            item
            for path, item in inputs.sources.items()
            if path.endswith(f"/{suite}/test_manifest.json")
        )
        require(
            plan["frozen_file_sha256"][relative] == baseline_source["sha256"],
            "Group frozen scene file differs from verified baseline bytes",
        )
        if (directory / relative).exists():
            require(
                _manifest(inputs, directory / relative, suite) == manifest,
                "Downloaded group reset scenes differ",
            )
    expected_jobs = {
        (method, suite, shard)
        for method in METHODS
        for suite in suites
        for shard in shards
    }
    jobs = plan["jobs"]
    require(
        len(jobs) == len(expected_jobs)
        and {(j["method"], j["suite"], j["shard"]) for j in jobs} == expected_jobs,
        "Execution job plan has missing, duplicate or unassigned suite/shards",
    )
    assigned = []
    for job in jobs:
        expected = [
            entry["episode_id"]
            for entry in manifests[job["suite"]]["episodes"][job["shard"] :: 16]
        ]
        require(
            job["num_shards"] == 16 and job["episode_ids"] == expected,
            "Execution episode IDs differ from frozen strided assignment",
        )
        relative = f"{job['method']}/{job['suite']}/config_{job['shard']}.json"
        checksum = plan["frozen_file_sha256"][relative]
        require(
            isinstance(checksum, str)
            and len(checksum) == 64
            and all(c in "0123456789abcdef" for c in checksum),
            "Invalid frozen method config hash",
        )
        config_key = (job["method"], job["suite"])
        identity["configs"].setdefault(config_key, checksum)
        require(
            identity["configs"][config_key] == checksum,
            "Frozen method/suite config changed across source executions",
        )
        if (directory / relative).exists():
            require(
                inputs.source(directory / relative)["sha256"] == checksum,
                "Downloaded method config differs from frozen plan",
            )
        if job["method"] == METHODS[0]:
            assigned.extend(expected)
    require(
        len(assigned)
        == len(set(assigned))
        == plan["assigned_episodes_per_method"]
        == runtime["assigned_episodes_per_method"]
        == summary["assigned_episodes_per_method"],
        "Execution assignment count mismatch",
    )
    require(
        summary["total_evaluated_episodes"] == len(assigned) * 3,
        "Execution evaluated episode count mismatch",
    )
    rows, costs, suite_costs = {}, {}, {}
    for method in METHODS:
        path = directory / method / "episodes.json"
        episodes = inputs.read(path)
        _validate_rows(episodes, assigned, method, entries, f"{execution_id} {method}")
        method_summary = summary["methods"][method]
        _check_summary(method_summary, episodes, f"{execution_id} {method}")
        require(
            method_summary["selected_solver"] == selected
            and set(method_summary["suites"]) == set(suites),
            "Method summary solver or assigned suite differs from its execution",
        )
        suite_costs[method] = {}
        for suite in suites:
            part = method_summary["suites"][suite]
            subset = [row for row in episodes if row["suite"] == suite]
            _check_summary(part, subset, f"{execution_id} {method} {suite}")
            require(
                part["task_manifest_sha256"] == manifests[suite]["sha256"],
                "Method summary reset manifest differs",
            )
            suite_costs[method][suite] = (execution_id, part["costs"])
        rows[method] = _annotate(
            episodes,
            inputs,
            path,
            workflow,
            METHOD_KEYS[method],
            manifests,
            entries,
            worker_index=worker,
        )
        costs[method] = (execution_id, method_summary["costs"])
    require(
        summary["execution_errors"]
        == sum(summary["methods"][method]["execution_errors"] for method in METHODS),
        "Execution error count mismatch",
    )
    audit = {
        "workflow": workflow,
        "group": group,
        "plan_sha256": plan["plan_sha256"],
        "runtime": runtime,
    }
    if worker is not None:
        audit.update(worker_index=worker, execution_id=execution_id)
    return rows, costs, suite_costs, audit


def _groups(inputs, directories, manifests, entries, baseline_run):
    require(
        len(directories) == 4,
        "Provide all four completed steering groups, or omit --groups",
    )
    rows = {method: [] for method in METHODS}
    costs = {method: [] for method in METHODS}
    suite_costs = {method: {suite: [] for suite in SUITES} for method in METHODS}
    seen_groups, seen_workflows, identity, group_audit = set(), set(), {}, []
    for directory in directories:
        sources = _group_sources(directory)
        executions = []
        for path, worker in sources:
            part_rows, part_costs, part_suites, audit = _execution(
                inputs, path, worker, manifests, entries, baseline_run, identity
            )
            if executions:
                require(
                    audit["group"] == executions[0]["group"]
                    and audit["workflow"] == executions[0]["workflow"],
                    "Distributed workers belong to different workflows or logical groups",
                )
            executions.append(audit)
            for method in METHODS:
                rows[method].extend(part_rows[method])
                costs[method].append(part_costs[method])
                for suite, value in part_suites[method].items():
                    suite_costs[method][suite].append(value)
        group, workflow = executions[0]["group"], executions[0]["workflow"]
        require(
            group["index"] not in seen_groups and workflow not in seen_workflows,
            "Duplicate steering logical group or workflow",
        )
        seen_groups.add(group["index"])
        seen_workflows.add(workflow)
        if len(executions) == 1:
            group_audit.append({"execution_topology": "single_node", **executions[0]})
        else:
            group_audit.append(
                {
                    "workflow": workflow,
                    "group": group,
                    "execution_topology": "distributed_workers",
                    "workers": executions,
                }
            )
    result, combined_costs = {}, {}
    for method in METHODS:
        indexed = _validate_rows(
            rows[method], entries, method, entries, f"All groups {method}"
        )
        result[METHOD_KEYS[method]] = [indexed[key] for key in entries]
        combined = _sum_costs(costs[method])
        for metric, row_key in (
            ("executed_actions", "actions"),
            ("velocity_evaluations", "velocity_evaluations"),
            ("agent_calls", "agent_calls"),
        ):
            require(
                combined["counts"].get(metric, 0)
                == sum(row[row_key] for row in rows[method]),
                f"Aggregated {method} cost differs in {metric}",
            )
        combined_costs[METHOD_KEYS[method]] = {
            "all": combined,
            **{
                suite: _sum_costs(parts) for suite, parts in suite_costs[method].items()
            },
        }
    return (
        result,
        combined_costs,
        sorted(group_audit, key=lambda item: item["group"]["index"]),
        identity["solver"],
    )


def _episode_costs(rows):
    counts = {
        "executed_actions": sum(row["actions"] for row in rows),
        "velocity_evaluations": sum(row["velocity_evaluations"] for row in rows),
        "agent_calls": sum(row["agent_calls"] for row in rows),
        "fallbacks": sum(row["fallbacks"] for row in rows),
    }
    return {
        "counts": counts,
        "api_usage_totals": {},
        "currency_cost": None,
        "action_fractions": {
            "fallback_actions": 0.0 if not counts["fallbacks"] else None
        },
        "cost_note": "Episode summary totals; fallback action fraction is unknown if fallback events occurred.",
        "latency_note": LATENCY_NOTE,
    }


def aggregate(baseline, repair, groups=()):
    """Return validated analysis data without writing to any input directory."""
    inputs = Inputs()
    baseline, repair = Path(baseline).resolve(), Path(repair).resolve()
    manifests, entries, baseline_rows, audit, baseline_run = _baseline(
        inputs, baseline, repair
    )
    rows = {"baseline_euler10": baseline_rows}
    costs = {
        "baseline_euler10": {
            "all": _episode_costs(baseline_rows),
            **{
                suite: _episode_costs([r for r in baseline_rows if r["suite"] == suite])
                for suite in SUITES
            },
        }
    }
    group_audit, selected_solver = [], None
    if groups:
        matched, matched_costs, group_audit, selected_solver = _groups(
            inputs,
            [Path(p).resolve() for p in groups],
            manifests,
            entries,
            baseline_run,
        )
        rows.update(matched)
        costs.update(matched_costs)
    methods = {}
    for key, episodes in rows.items():
        task_rows = {}
        for suite, manifest in manifests.items():
            for task in range(10):
                subset = [
                    row
                    for row in episodes
                    if row["suite"] == suite and row["task_id"] == task
                ]
                task_rows[f"{suite}:{task}"] = {
                    "suite": suite,
                    "task_id": task,
                    "instruction": manifest["episodes"][task * 10]["instruction"],
                    **_stats(subset),
                }
        stats = _stats(episodes)
        methods[key] = {
            "label": LABELS[key],
            **stats,
            "valid_complete_evaluation": stats["execution_errors"]
            == stats["zero_action_successes"]
            == 0,
            "per_suite": {
                suite: {
                    **_stats([r for r in episodes if r["suite"] == suite]),
                    "costs": costs[key][suite],
                }
                for suite in SUITES
            },
            "per_task": task_rows,
            "costs": costs[key]["all"],
        }
    comparisons = {}
    if groups:
        for left, right in itertools.combinations(rows, 2):
            clean = (
                methods[left]["valid_complete_evaluation"]
                and methods[right]["valid_complete_evaluation"]
            )
            comparisons[f"{right}_minus_{left}"] = {
                "left": left,
                "right": right,
                "execution_errors_or_zero_action_successes_present": not clean,
                "scope": "Includes solver and TF32 runtime changes"
                if left == "baseline_euler10"
                else "Matched selected solver and TF32-off runtime",
                **paired_comparison(rows[left], rows[right], seed=7),
                "per_suite": {
                    suite: paired_comparison(
                        [r for r in rows[left] if r["suite"] == suite],
                        [r for r in rows[right] if r["suite"] == suite],
                        seed=7,
                    )
                    for suite in SUITES
                },
            }
    report = {
        "schema_version": "1.0",
        "status": "complete_paired_methods" if groups else "complete_repaired_baseline",
        "baseline_repair_audit": audit,
        "verified_task_manifest_sha256": {
            suite: m["sha256"] for suite, m in manifests.items()
        },
        "selected_matched_solver": selected_solver,
        "steering_groups": group_audit,
        "methods": methods,
        "paired_comparisons": comparisons,
        "provenance": list(inputs.sources.values()),
        "canonical_records": "canonical_episodes.json",
        "notes": [
            "Original records are retained unchanged. All 25 repaired shard outcomes replace the original shard; replacement does not depend on success.",
            "The baseline and matched fresh-noise control are separate methods and are never pooled.",
            "Baseline-versus-matched comparisons include Euler/RK4 and TF32-on/off changes. Only the three matched methods share the solver and runtime settings.",
            "Full scene manifest digests include dynamic state and model fixture poses and were verified before pairing.",
            LATENCY_NOTE,
            "A zero-action success is a stabilization outcome, not evidence of policy task completion; it is counted explicitly.",
            "No currency price or pooled latency p95 was inferred. This is a new pi05 evaluation; checkpoint training overlap is unknown.",
        ],
    }
    return report, rows, inputs


def _markdown(report):
    audit = report["baseline_repair_audit"]
    original, repair = audit["original"], audit["repair"]
    lines = [
        "# LIBERO-OOD evaluation",
        "",
        f"The original baseline recorded {original['successes']}/{original['episodes']} successes and {original['execution_errors']} execution errors. The full affected Goal shard ({len(repair['replaced_episode_ids'])} episodes) was replaced using the same frozen scenes.",
        "",
        "| Method | Goal | Spatial | Total | Errors | Zero-action successes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"].values():
        cells = [f"{method['per_suite'][s]['successes']}/100" for s in SUITES]
        lines.append(
            f"| {method['label']} | {' | '.join(cells)} | {method['successes']}/200 ({method['success_rate']:.1%}) | {method['execution_errors']} | {method['zero_action_successes']} |"
        )
    if report["selected_matched_solver"] is not None:
        solver = report["selected_matched_solver"]
        lines.extend(
            [
                "",
                f"Matched methods use {solver['solver'].upper()}, {solver['steps']} steps, with solver options `{json.dumps(solver['solver_options'], sort_keys=True)}` and TF32 disabled. Comparisons against the baseline include solver and runtime changes; the three matched methods isolate steering/noise reuse.",
                "",
                "| Paired comparison | Difference (percentage points) | 95% paired bootstrap interval |",
                "|---|---:|---:|",
            ]
        )
        for result in report["paired_comparisons"].values():
            lo, hi = result["paired_bootstrap_95_percent"]
            flag = (
                " (contains execution/protocol failures)"
                if result["execution_errors_or_zero_action_successes_present"]
                else ""
            )
            lines.append(
                f"| {LABELS[result['right']]} minus {LABELS[result['left']]}{flag} | {100 * result['difference_right_minus_left']:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
            )
    lines.extend(
        [
            "",
            "| Method | Actions | Velocity evaluations | Agent calls | Fallback action fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in report["methods"].values():
        fraction = method["costs"]["action_fractions"]["fallback_actions"]
        shown = "unknown" if fraction is None else f"{fraction:.1%}"
        lines.append(
            f"| {method['label']} | {method['total_actions']} | {method['velocity_evaluations']} | {method['agent_calls']} | {shown} |"
        )
    lines.extend(
        [
            "",
            LATENCY_NOTE,
            "",
            "Per-task names, counts and action metrics are in `tasks.csv`. `report.json` retains API usage and separate latency summaries for each source workflow or distributed worker; `canonical_episodes.json` records each row's source workflow, worker and hashes. Original and repair records are copied unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output, report, rows, inputs, baseline, repair):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("report.json", report), ("canonical_episodes.json", rows)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for directory, name in (
        (baseline, "original_baseline_episodes.json"),
        (repair, "repair_episodes.json"),
    ):
        (output / name).write_bytes(
            inputs.raw[str((Path(directory) / "episodes.json").resolve())]
        )
    (output / "report.md").write_text(_markdown(report))
    fields = [
        "method",
        "label",
        "suite",
        "episodes",
        "successes",
        "success_rate",
        "execution_errors",
        "zero_action_successes",
        "zero_action_failures",
        "total_actions",
        "mean_actions",
        "actions_to_success",
        "total_wall_seconds",
        "mean_wall_seconds",
        "velocity_evaluations",
        "agent_calls",
        "fallbacks",
        "fallback_action_fraction",
    ]
    with (output / "methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, method in report["methods"].items():
            for suite, summary in (("all", method), *method["per_suite"].items()):
                writer.writerow(
                    {
                        name: {
                            "method": key,
                            "label": method["label"],
                            "suite": suite,
                            "fallback_action_fraction": summary["costs"][
                                "action_fractions"
                            ]["fallback_actions"],
                            **summary,
                        }.get(name)
                        for name in fields
                    }
                )
    task_fields = ["method", "label", "suite", "task_id", "instruction", *fields[3:-1]]
    with (output / "tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=task_fields)
        writer.writeheader()
        for key, method in report["methods"].items():
            for task in method["per_task"].values():
                writer.writerow(
                    {
                        name: {"method": key, "label": method["label"], **task}.get(
                            name
                        )
                        for name in task_fields
                    }
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--groups", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report, rows, inputs = aggregate(args.baseline, args.repair, args.groups)
    write_outputs(args.output, report, rows, inputs, args.baseline, args.repair)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": report["status"],
                "methods": {
                    key: {
                        name: value[name]
                        for name in (
                            "episodes",
                            "successes",
                            "execution_errors",
                            "zero_action_successes",
                        )
                    }
                    for key, value in report["methods"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
