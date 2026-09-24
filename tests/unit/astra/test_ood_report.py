import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.ood_report import METHODS, SUITES, aggregate, main
from astra_reversal.records import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def rehash(value, key):
    value.pop(key, None)
    value[key] = digest(value)
    return value


def stats(rows):
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "execution_errors": sum(row["failure"] is not None for row in rows),
    }


def row(entry, method="policy_fresh", success=False):
    return {
        **{
            name: entry[name]
            for name in (
                "episode_id",
                "suite",
                "task_id",
                "initial_state_id",
                "seed",
                "reset_state_sha256",
            )
        },
        "method": method,
        "success": success,
        "failure": None,
        "actions": 10,
        "wall_seconds": 1.0,
        "velocity_evaluations": 20,
        "fallbacks": 0,
        "agent_calls": int(method == "reversal"),
        "protocol": "released_modified_libero",
        "split": "test",
        "task_action_budget": 300,
    }


@pytest.fixture
def experiment(tmp_path):
    baseline, repair = tmp_path / "baseline", tmp_path / "repair"
    manifests = {}
    for suite in SUITES:
        episodes = []
        for task in range(10):
            for trial in range(10):
                state = np.array([task, trial], dtype=np.float64)
                model = {
                    "body_pos": np.array([[0.0, task, trial]]),
                    "body_quat": np.array([[1.0, 0, 0, 0]]),
                }
                episodes.append(
                    {
                        "episode_id": f"{suite}:task{task}:state{trial}",
                        "suite": suite,
                        "task_id": task,
                        "initial_state_id": trial,
                        "seed": 7,
                        "instruction": f"task {task} in {suite}",
                        "initially_successful": False,
                        "reset_state": state.tolist(),
                        "reset_state_sha256": digest(state),
                        "reset_model": {k: v.tolist() for k, v in model.items()},
                        "reset_model_sha256": digest(model),
                        "bddl_sha256": "b" * 64,
                    }
                )
        manifest = rehash(
            {
                "schema_version": "1.1",
                "seed": 7,
                "split": "test",
                "benchmark": asdict(BenchmarkConfig.preset(suite)),
                "episodes": episodes,
            },
            "sha256",
        )
        manifests[suite] = manifest
        write(baseline / suite / "test_manifest.json", manifest)
    entries = [
        entry for manifest in manifests.values() for entry in manifest["episodes"]
    ]
    originals = [row(entry) for entry in entries]
    affected = manifests[SUITES[0]]["episodes"][3::4]
    original_index = {item["episode_id"]: item for item in originals}
    originals[0]["success"] = True
    original_index[affected[0]["episode_id"]]["success"] = True
    for entry in affected[-8:]:
        original_index[entry["episode_id"]].update(
            failure="CUDA launch failure", actions=0
        )
    runtime = {"workflow": "original", "payload_sha256": "o" * 64}
    write(baseline / "runtime.json", runtime)
    write(baseline / "episodes.json", originals)
    write(
        baseline / "summary.json",
        {
            "workflow": "original",
            **stats(originals),
            "suites": {
                suite: {
                    **stats([r for r in originals if r["suite"] == suite]),
                    "task_manifest_sha256": manifest["sha256"],
                }
                for suite, manifest in manifests.items()
            },
        },
    )
    config = asdict(
        RunConfig.from_dict(
            {
                "seed": 7,
                "method": "policy_fresh",
                "benchmark": {"suite": SUITES[0]},
                "policy": {"device": "cuda:0"},
                "evaluation": {"split": "test"},
            }
        )
    )
    run = {
        "config": config,
        "checkpoint": {
            "device": "cuda:0",
            "requested_artifact": "checkpoint",
            "artifact_sha256": {"model.safetensors": "a" * 64},
        },
        "action_spec": {"action_spec_id": "controller"},
        "task_manifest_sha256": manifests[SUITES[0]]["sha256"],
    }
    write(baseline / SUITES[0] / "shard_0/manifest.json", run)
    repair_run = copy.deepcopy(run)
    repair_run["config"]["policy"]["device"] = "cuda"
    repair_run["checkpoint"]["device"] = "cuda"
    write(repair / "repair/manifest.json", repair_run)
    replacement = [row(entry) for entry in affected]
    for item in replacement:
        item.update(actions=20, wall_seconds=5)
    replacement[1]["success"] = True
    repair_runtime = {
        "workflow": "replacement",
        "replaces_workflow": "original",
        "replaces_suite": SUITES[0],
        "replaces_shard": 3,
        "num_shards": 4,
        "task_manifest_sha256": manifests[SUITES[0]]["sha256"],
        "payload_sha256": "r" * 64,
    }
    write(repair / "runtime.json", repair_runtime)
    write(repair / "summary.json", {**repair_runtime, **stats(replacement)})
    write(repair / "episodes.json", replacement)
    return baseline, repair, manifests, originals, replacement


def test_complete_shard_replacement_keeps_raw_records_and_distinct_provenance(
    experiment, tmp_path
):
    baseline, repair, manifests, originals, replacement = experiment
    original_bytes = (baseline / "episodes.json").read_bytes()
    replacement_bytes = (repair / "episodes.json").read_bytes()
    output = tmp_path / "report"
    main(
        ["--baseline", str(baseline), "--repair", str(repair), "--output", str(output)]
    )
    report = json.loads((output / "report.json").read_text())
    canonical = json.loads((output / "canonical_episodes.json").read_text())[
        "baseline_euler10"
    ]
    indexed = {r["episode_id"]: r for r in canonical}
    first = replacement[0]["episode_id"]
    assert (
        indexed[first]["success"] is False
    )  # Prior success must not be cherry-picked.
    assert indexed[first]["actions"] == 20
    assert (
        len(
            [
                r
                for r in canonical
                if r["analysis_provenance"]["workflow"] == "replacement"
            ]
        )
        == 25
    )
    assert indexed[first]["analysis_provenance"][
        "replaces_original_row_sha256"
    ] == digest(next(r for r in originals if r["episode_id"] == first))
    assert report["methods"]["baseline_euler10"]["successes"] == 2
    assert report["methods"]["baseline_euler10"]["total_actions"] == 2250
    assert report["methods"]["baseline_euler10"]["execution_errors"] == 0
    assert report["baseline_repair_audit"]["original"]["execution_errors"] == 8
    assert report["baseline_repair_audit"]["original"]["zero_action_failures"] == 8
    assert (output / "original_baseline_episodes.json").read_bytes() == original_bytes
    assert (output / "repair_episodes.json").read_bytes() == replacement_bytes
    assert (baseline / "episodes.json").read_bytes() == original_bytes
    assert "CUDA_LAUNCH_BLOCKING=1" in (output / "report.md").read_text()
    assert "task 0 in libero_goal_ood" in (output / "tasks.csv").read_text()


def test_repair_rejects_replacing_only_failed_outcomes(experiment):
    baseline, repair, _, _, replacement = experiment
    write(repair / "episodes.json", replacement[-8:])
    with pytest.raises(ValueError, match="unassigned episode IDs"):
        aggregate(baseline, repair)


@pytest.mark.parametrize(
    "change", ["failure", "zero_action_success", "policy_identity", "fixture_pose"]
)
def test_baseline_claim_rejects_invalid_repair_or_frozen_scenes(experiment, change):
    baseline, repair, _, _, replacement = experiment
    if change in ("failure", "zero_action_success"):
        if change == "failure":
            replacement[0]["failure"] = "still failed"
        else:
            replacement[0].update(actions=0, success=True)
        write(repair / "episodes.json", replacement)
        summary = json.loads((repair / "summary.json").read_text())
        write(repair / "summary.json", {**summary, **stats(replacement)})
    elif change == "policy_identity":
        path = repair / "repair/manifest.json"
        changed = json.loads(path.read_text())
        changed["config"]["flow"]["generation_steps"] = 20
        write(path, changed)
    else:
        path = baseline / SUITES[0] / "test_manifest.json"
        changed = json.loads(path.read_text())
        changed["episodes"][0]["reset_model"]["body_pos"][0][0] += 0.02
        write(path, rehash(changed, "sha256"))
    with pytest.raises(ValueError):
        aggregate(baseline, repair)


def cost(rows, group):
    actions, calls = (
        sum(r["actions"] for r in rows),
        sum(r["agent_calls"] for r in rows),
    )
    return {
        "counts": {
            "executed_actions": actions,
            "velocity_evaluations": sum(r["velocity_evaluations"] for r in rows),
            "agent_calls": calls,
            "fallback_actions": actions // 10 if calls else 0,
            "latent_actions": actions if calls else 0,
            "nonfallback_latent_actions": actions - actions // 10 if calls else 0,
        },
        "api_usage_totals": {
            "prompt_tokens": 3 * calls,
            "completion_tokens": 2 * calls,
        },
        "actual_api_models": {"Astra": calls} if calls else {},
        "fallback_categories": {},
        "latency_seconds": {"api": {"total": calls * 2, "mean": 2, "p95": 10 + group}},
        "max_action_clip": 0,
    }


def steering_groups(experiment, tmp_path):
    baseline, _, manifests, _, _ = experiment
    selected = {
        "solver": "rk4",
        "steps": 100,
        "solver_options": {"time_power": 3.0},
        "conditions_passed": 14,
        "velocity_evaluations_per_solve": 400,
    }
    directories = []
    for index in range(4):
        directory = tmp_path / f"group{index}"
        directories.append(directory)
        group = {"index": index, "count": 4, "global_shards_per_suite": 16}
        jobs, assigned = [], []
        for method in METHODS:
            for suite in SUITES:
                for shard in range(index * 4, index * 4 + 4):
                    entries = manifests[suite]["episodes"][shard::16]
                    jobs.append(
                        {
                            "method": method,
                            "suite": suite,
                            "shard": shard,
                            "num_shards": 16,
                            "episode_ids": [e["episode_id"] for e in entries],
                        }
                    )
                    if method == METHODS[0]:
                        assigned.extend(entries)
        plan = rehash(
            {
                "methods": list(METHODS),
                "shard_group": group,
                "assigned_episodes_per_method": len(assigned),
                "selected_solver": selected,
                "jobs": jobs,
                "manifest_sha256": {s: m["sha256"] for s, m in manifests.items()},
                "frozen_file_sha256": {
                    f"frozen/{s}_manifest.json": hashlib.sha256(
                        (baseline / s / "test_manifest.json").read_bytes()
                    ).hexdigest()
                    for s in SUITES
                }
                | {
                    f"{job['method']}/{job['suite']}/config_{job['shard']}.json": hashlib.sha256(
                        f"{job['method']}/{job['suite']}".encode()
                    ).hexdigest()
                    for job in jobs
                },
            },
            "plan_sha256",
        )
        write(directory / "frozen_plan.json", plan)
        runtime = {
            "workflow": f"steering{index}",
            "methods": list(METHODS),
            "shard_group": group,
            "assigned_episodes_per_method": len(assigned),
            "selected_solver": selected,
            "verified_asset_sha256": {"checkpoint/model.safetensors": "a" * 64},
            "tf32": False,
        }
        write(directory / "runtime.json", runtime)
        reports = {}
        for method in METHODS:
            episodes = [
                row(
                    entry,
                    method,
                    entry["initial_state_id"] < 8
                    if method == "reversal"
                    else entry["initial_state_id"] % 2 == 0,
                )
                for entry in assigned
            ]
            write(directory / method / "episodes.json", episodes)
            reports[method] = {
                **stats(episodes),
                "selected_solver": selected,
                "costs": cost(episodes, index),
                "suites": {
                    s: {
                        **stats([r for r in episodes if r["suite"] == s]),
                        "task_manifest_sha256": manifests[s]["sha256"],
                        "costs": cost([r for r in episodes if r["suite"] == s], index),
                    }
                    for s in SUITES
                },
            }
        write(
            directory / "summary.json",
            {
                "workflow": runtime["workflow"],
                "shard_group": group,
                "assigned_episodes_per_method": len(assigned),
                "total_evaluated_episodes": len(assigned) * 3,
                "methods": reports,
                "execution_errors": 0,
            },
        )
    return directories


def test_four_groups_validate_strided_union_pairing_and_sum_measured_costs(
    experiment, tmp_path
):
    baseline, repair, *_ = experiment
    groups = steering_groups(experiment, tmp_path)
    report, rows, _ = aggregate(baseline, repair, groups[::-1])
    assert {k: len(v) for k, v in rows.items()} == {
        "baseline_euler10": 200,
        "matched_fresh_noise": 200,
        "matched_reused_noise": 200,
        "astra_reversal": 200,
    }
    assert report["methods"]["matched_fresh_noise"]["successes"] == 100
    assert report["methods"]["astra_reversal"]["successes"] == 160
    paired = report["paired_comparisons"]["astra_reversal_minus_matched_fresh_noise"]
    assert paired["difference_right_minus_left"] == pytest.approx(0.3)
    assert paired["paired_episodes"] == 200 and paired["tasks"] == 20
    costs = report["methods"]["astra_reversal"]["costs"]
    assert costs["counts"]["agent_calls"] == 200
    assert costs["api_usage_totals"] == {"prompt_tokens": 600, "completion_tokens": 400}
    assert costs["action_fractions"]["fallback_actions"] == 0.1
    assert {
        v["api"]["p95"] for v in costs["latency_seconds_by_execution"].values()
    } == {
        10,
        11,
        12,
        13,
    }
    assert "p95" not in costs and "latency_seconds" not in costs


@pytest.mark.parametrize(
    "change",
    [
        "missing_group",
        "duplicate_group",
        "missing_episode",
        "scene_digest",
        "solver",
        "missing_cost",
    ],
)
def test_group_integrity_gates(experiment, tmp_path, change):
    baseline, repair, *_ = experiment
    groups = steering_groups(experiment, tmp_path)
    if change == "missing_group":
        groups.pop()
    elif change == "duplicate_group":
        groups[-1] = groups[0]
    elif change == "missing_episode":
        path = groups[0] / "policy_fresh/episodes.json"
        write(path, json.loads(path.read_text())[1:])
    elif change == "missing_cost":
        path = groups[0] / "summary.json"
        summary = json.loads(path.read_text())
        del summary["methods"]["reversal"]["costs"]["counts"]["fallback_actions"]
        write(path, summary)
    else:
        path = groups[-1] / "frozen_plan.json"
        plan = json.loads(path.read_text())
        if change == "scene_digest":
            plan["manifest_sha256"][SUITES[0]] = "x" * 64
        else:
            plan["selected_solver"]["steps"] = 200
        write(path, rehash(plan, "plan_sha256"))
    with pytest.raises(ValueError):
        aggregate(baseline, repair, groups)


def distribute_group(directory, destination):
    plan = json.loads((directory / "frozen_plan.json").read_text())
    runtime = json.loads((directory / "runtime.json").read_text())
    original_summary = json.loads((directory / "summary.json").read_text())
    index = plan["shard_group"]["index"]
    for worker in range(8):
        target = destination / f"worker_{worker}"
        suite, shard = SUITES[worker // 4], index * 4 + worker % 4
        jobs = [j for j in plan["jobs"] if j["suite"] == suite and j["shard"] == shard]
        assigned = set(jobs[0]["episode_ids"])
        worker_plan = copy.deepcopy(plan)
        worker_plan.update(
            worker_index=worker,
            workers_per_group=8,
            execution_topology="distributed_workers",
            group_assigned_episodes_per_method=plan["assigned_episodes_per_method"],
            assigned_episodes_per_method=len(assigned),
            jobs=jobs,
        )
        write(target / "frozen_plan.json", rehash(worker_plan, "plan_sha256"))
        worker_runtime = {
            **runtime,
            "worker_index": worker,
            "workers_per_group": 8,
            "group_assigned_episodes_per_method": plan["assigned_episodes_per_method"],
            "visible_gpu_count": 1,
            "assignment": {
                "worker_index": worker,
                "suite": suite,
                "local_shard": worker % 4,
                "shard": shard,
                "num_shards": 16,
                "physical_gpu": 0,
            },
            "assigned_episodes_per_method": len(assigned),
            "hostname": f"actual_host_{index}_{worker}",
            "gpu_names": ["NVIDIA L40S"],
        }
        write(target / "runtime.json", worker_runtime)
        reports = {}
        for method in METHODS:
            episodes = [
                r
                for r in json.loads((directory / method / "episodes.json").read_text())
                if r["episode_id"] in assigned
            ]
            write(target / method / "episodes.json", episodes)
            reports[method] = {
                **stats(episodes),
                "selected_solver": plan["selected_solver"],
                "costs": cost(episodes, index * 8 + worker),
                "suites": {
                    suite: {
                        **stats(episodes),
                        "task_manifest_sha256": plan["manifest_sha256"][suite],
                        "costs": cost(episodes, index * 8 + worker),
                    }
                },
            }
        write(
            target / "summary.json",
            {
                **original_summary,
                "worker_index": worker,
                "workers_per_group": 8,
                "group_assigned_episodes_per_method": plan[
                    "assigned_episodes_per_method"
                ],
                "assigned_episodes_per_method": len(assigned),
                "total_evaluated_episodes": len(assigned) * 3,
                "methods": reports,
            },
        )
    return destination


def test_mixed_distributed_groups_preserve_worker_provenance_and_latencies(
    experiment, tmp_path
):
    baseline, repair, *_ = experiment
    groups = steering_groups(experiment, tmp_path)
    for index in (2, 3):
        groups[index] = distribute_group(
            groups[index], tmp_path / f"distributed{index}"
        )
    report, rows, _ = aggregate(baseline, repair, groups)
    assert report["methods"]["astra_reversal"]["successes"] == 160
    assert all(len(episodes) == 200 for episodes in rows.values())
    for group in report["steering_groups"][2:]:
        assert group["execution_topology"] == "distributed_workers"
        assert "runtime" not in group
        assert len(group["workers"]) == 8
        assert len({worker["runtime"]["hostname"] for worker in group["workers"]}) == 8
    for episode in rows["astra_reversal"]:
        provenance = episode["analysis_provenance"]
        if provenance["workflow"] not in ("steering2", "steering3"):
            continue
        group_index = int(provenance["workflow"][-1])
        global_shard = (episode["task_id"] * 10 + episode["initial_state_id"]) % 16
        expected_worker = (
            SUITES.index(episode["suite"]) * 4 + global_shard - 4 * group_index
        )
        assert provenance["worker_index"] == expected_worker
        assert (
            provenance["execution_id"]
            == f"steering{group_index}/worker_{expected_worker}"
        )
    costs = report["methods"]["astra_reversal"]["costs"]
    latencies = costs["latency_seconds_by_execution"]
    assert len(latencies) == 18  # Two single-node executions and sixteen workers.
    assert latencies["steering2/worker_0"]["api"]["p95"] == 26
    assert latencies["steering2/worker_7"]["api"]["p95"] == 33
    assert costs["api_usage_totals"] == {"prompt_tokens": 600, "completion_tokens": 400}
    assert costs["action_fractions"]["fallback_actions"] == 0.1
    assert report["paired_comparisons"]["astra_reversal_minus_matched_fresh_noise"][
        "difference_right_minus_left"
    ] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "change", ["missing_worker", "worker_index", "assignment", "workflow", "config"]
)
def test_distributed_worker_integrity_gates(experiment, tmp_path, change):
    baseline, repair, *_ = experiment
    groups = steering_groups(experiment, tmp_path)
    groups[3] = distribute_group(groups[3], tmp_path / "distributed3")
    worker = groups[3] / "worker_7"
    if change == "missing_worker":
        worker.rename(groups[3] / "missing")
    elif change == "workflow":
        for name in ("runtime.json", "summary.json"):
            value = json.loads((worker / name).read_text())
            value["workflow"] = "unrelated_run"
            write(worker / name, value)
    else:
        path = worker / "frozen_plan.json"
        plan = json.loads(path.read_text())
        if change == "worker_index":
            plan["worker_index"] = 6
        elif change == "assignment":
            plan["jobs"][0]["suite"] = SUITES[0]
        else:
            job = plan["jobs"][0]
            config_key = f"{job['method']}/{job['suite']}/config_{job['shard']}.json"
            plan["frozen_file_sha256"][config_key] = "0" * 64
        write(path, rehash(plan, "plan_sha256"))
    with pytest.raises(ValueError):
        aggregate(baseline, repair, groups)
