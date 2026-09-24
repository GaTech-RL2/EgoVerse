"""Scheduling/provenance tests; no policy, simulator, GPU or API calls."""

import json
import shlex
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from astra_reversal.config import BenchmarkConfig
from astra_reversal.libero_runner import shard_entries
from astra_reversal.osmo import ood_distributed as distributed
from astra_reversal.osmo import ood_steering as steering
from astra_reversal.records import digest, file_sha256


@pytest.mark.parametrize("groups", [1, 2, 4, 25])
def test_distributed_workers_cover_exactly_the_original_group_partition(groups):
    manifests = {
        suite: {
            "episodes": [{"episode_id": f"{suite}:{index}"} for index in range(100)]
        }
        for suite in steering.SUITES
    }
    all_ids = []
    for group_index in range(groups):
        group = {
            "count": groups,
            "index": group_index,
            "global_shards_per_suite": 4 * groups,
        }
        expected, actual = [], []
        for suite in steering.SUITES:
            for shard in range(4 * group_index, 4 * group_index + 4):
                expected.extend(
                    row["episode_id"]
                    for row in shard_entries(manifests[suite], shard, 4 * groups)
                )
        for worker_index in range(8):
            target = distributed.assignment(worker_index, group)
            assert target["physical_gpu"] == 0
            actual.extend(
                row["episode_id"]
                for row in shard_entries(
                    manifests[target["suite"]],
                    target["shard"],
                    target["num_shards"],
                )
            )
        assert len(actual) == len(set(actual))
        assert sorted(actual) == sorted(expected)
        if groups == 4:
            assert len(actual) == (56 if group_index == 0 else 48)
        all_ids.extend(actual)
    assert len(all_ids) == len(set(all_ids)) == 200


def write_fixture_inputs(path):
    path.mkdir()
    for suite in steering.SUITES:
        benchmark = BenchmarkConfig.preset(suite)
        episodes = []
        model = {
            "body_pos": np.zeros((1, 3), np.float64),
            "body_quat": np.asarray([[0, 0, 0, 1]], np.float64),
        }
        for task_id in range(10):
            for trial in range(10):
                state = np.asarray([task_id, trial], np.float64)
                episodes.append(
                    {
                        "episode_id": f"{suite}:task{task_id}:state{trial}",
                        "suite": suite,
                        "task_id": task_id,
                        "initial_state_id": trial,
                        "instruction": "synthetic scheduling fixture",
                        "seed": 7,
                        "reset_state": state.tolist(),
                        "reset_state_sha256": digest(state),
                        "reset_model": {
                            key: value.tolist() for key, value in model.items()
                        },
                        "reset_model_sha256": digest(model),
                        "bddl_sha256": "synthetic",
                        "initially_successful": False,
                    }
                )
        manifest = {
            "schema_version": "1.1",
            "seed": 7,
            "split": "test",
            "benchmark": asdict(benchmark),
            "libero_root": "synthetic-test-only",
            "reset_procedure": benchmark.reset_source
            + "_captured_before_stabilization",
            "episodes": episodes,
        }
        manifest["sha256"] = digest(manifest)
        (path / f"{suite}_manifest.json").write_text(json.dumps(manifest))
    report = {
        "selected_solver": {
            "solver": "rk4",
            "steps": 100,
            "solver_options": {"time_power": 3.0},
        }
    }
    (path / "runtime_diagnostics.json").write_text(json.dumps(report))
    proposal = path / "proposal"
    proposal.mkdir()
    for name in ("request.json", "response.json", "provider.jsonl"):
        (proposal / name).write_text('{"synthetic_freezer_fixture":true}\n')
    return report, proposal


@pytest.mark.parametrize("worker_index", [0, 3, 4, 7])
def test_frozen_worker_config_bytes_and_scene_hashes_match_monolithic_group(
    tmp_path,
    monkeypatch,
    worker_index,
):
    inputs = tmp_path / "inputs"
    report, proposal = write_fixture_inputs(inputs)
    monkeypatch.setattr(steering, "INPUTS", inputs)
    monkeypatch.setattr(steering, "PROPOSAL_INPUT", proposal)
    original = tmp_path / "monolithic"
    monkeypatch.setattr(steering, "RESULTS", original)
    group = {"count": 4, "index": 2, "global_shards_per_suite": 16}
    methods = list(steering.METHODS)
    _, _, old_jobs, _ = steering.freeze_inputs(methods, report, 100, group)
    old_plan = json.loads((original / "frozen_plan.json").read_text())
    expected = {
        job["method"]: job for job in old_jobs if job["physical_gpu"] == worker_index
    }
    worker_root = tmp_path / "distributed"
    monkeypatch.setattr(steering, "RESULTS", worker_root)
    monkeypatch.setattr(distributed, "RESULTS", worker_root)
    _, _, jobs, _, plan = distributed.freeze_worker_inputs(
        methods, report, 100, group, worker_index
    )
    assert plan["group_assigned_episodes_per_method"] == 48
    assert plan["assigned_episodes_per_method"] == 6
    assert plan["worker_index"] == worker_index
    assert plan["manifest_sha256"] == old_plan["manifest_sha256"]
    assert plan["plan_sha256"] == digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    for suite in steering.SUITES:
        key = f"frozen/{suite}_manifest.json"
        assert plan["frozen_file_sha256"][key] == old_plan["frozen_file_sha256"][key]
    for job in jobs:
        reference = expected[job["method"]]
        assert job["physical_gpu"] == 0
        assert job["episode_ids"] == reference["episode_ids"]
        assert file_sha256(job["config"]) == file_sha256(reference["config"])
        config = json.loads(Path(job["config"]).read_text())
        assert config["policy"]["device"] == "cuda"
        assert (
            config["flow"]["generation_steps"]
            == config["flow"]["inversion_steps"]
            == 100
        )
        assert config["agent"]["sampling_settings"] == {
            "reasoning_effort": "low",
            "max_completion_tokens": 8192,
        }


def test_worker_commands_retain_exact_agent_settings_and_global_shard(monkeypatch):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-test-credential")
    for method in steering.METHODS:
        job = {
            "method": method,
            "config": "config.json",
            "manifest": "manifest.json",
            "shard": 11,
            "num_shards": 16,
            "output": "output",
            "api_log": "api.jsonl",
        }
        command = distributed.run_command(job, "runtime.json")
        assert command[command.index("--shard-index") + 1] == "11"
        assert command[command.index("--num-shards") + 1] == "16"
        environment = distributed.child_environment(method)
        assert (
            environment["CUDA_VISIBLE_DEVICES"]
            == environment["MUJOCO_EGL_DEVICE_ID"]
            == "0"
        )
        assert environment["NVIDIA_TF32_OVERRIDE"] == "0"
        if method == "reversal":
            agent = shlex.split(command[command.index("--agent-command") + 1])
            assert agent[agent.index("--model") + 1] == steering.MODEL
            assert agent[agent.index("--endpoint") + 1] == steering.ENDPOINT
            assert agent[agent.index("--reasoning-effort") + 1] == "low"
            assert agent[agent.index("--max-completion-tokens") + 1] == "8192"
            assert agent[agent.index("--timeout") + 1] == "170"
            assert command[command.index("--diagnostics") + 1] == "runtime.json"
            assert (
                environment["NVIDIA_INFERENCE_API_KEY"] == "synthetic-test-credential"
            )
        else:
            assert "--agent-command" not in command and "--diagnostics" not in command
            assert "NVIDIA_INFERENCE_API_KEY" not in environment


def test_eight_workers_have_disjoint_archive_prefixes():
    prefixes = [
        distributed.archive_prefix("synthetic-workflow", index) for index in range(8)
    ]
    assert len(set(prefixes)) == 8
    assert prefixes[7].endswith("/synthetic-workflow/worker_7")
    for index in (-1, 8, True):
        with pytest.raises(ValueError):
            distributed.archive_prefix("synthetic-workflow", index)


def test_single_suite_worker_summary_rejects_duplicate_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, "RESULTS", tmp_path)
    suite, method = "libero_spatial_ood", "policy_fresh"
    root = tmp_path / method / suite / "shard_11"
    root.mkdir(parents=True)
    config_path = root.parent / "config_11.json"
    config = {"synthetic_config": True}
    config_path.write_text(json.dumps(config))
    diagnostic = {
        "checkpoint": {"synthetic": True},
        "action_spec_id": "synthetic-controller",
        "selected_solver": {"steps": 100},
    }
    manifests = {suite: {"sha256": "synthetic-manifest"}}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "config": config,
                "checkpoint": diagnostic["checkpoint"],
                "action_spec": {"action_spec_id": diagnostic["action_spec_id"]},
                "task_manifest_sha256": manifests[suite]["sha256"],
            }
        )
    )
    episodes = [
        dict(
            kind="episode_end",
            episode_id=f"synthetic-{index}",
            suite=suite,
            method=method,
            task_id=index,
            success=False,
            actions=300,
            wall_seconds=1.0,
            velocity_evaluations=400,
            fallbacks=0,
            failure=None,
        )
        for index in range(6)
    ]
    events = root / "events.jsonl"
    events.write_text("".join(json.dumps(row) + "\n" for row in episodes))
    job = {
        "output": str(root),
        "config": str(config_path),
        "api_log": str(root / "not-created.jsonl"),
        "suite": suite,
        "method": method,
        "worker_index": 7,
        "episode_ids": [row["episode_id"] for row in episodes],
    }
    summary, rows = distributed.summarize_worker_method(
        job, manifests, diagnostic, {"index": 2, "count": 4}
    )
    assert set(summary["suites"]) == {suite}
    assert summary["episodes"] == summary["expected_assigned_episodes"] == 6
    assert len(rows) == 6 and not summary["complete_frozen_benchmark"]
    with events.open("a") as stream:
        stream.write(json.dumps(episodes[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        distributed.summarize_worker_method(
            job, manifests, diagnostic, {"index": 2, "count": 4}
        )


def test_yaml_defines_eight_independent_single_gpu_tasks():
    path = Path(distributed.__file__).with_name("ood_distributed_l40s.yaml")
    workflow = yaml.safe_load(path.read_text())["workflow"]
    assert workflow["resources"]["default"] == {
        "cpu": 4,
        "gpu": 1,
        "memory": "48Gi",
        "storage": "60Gi",
        "platform": "ovx-l40s",
    }
    tasks = workflow["tasks"]
    assert [task["name"] for task in tasks] == [f"worker{index}" for index in range(8)]
    for index, task in enumerate(tasks):
        assert task["environment"]["ASTRA_WORKER_INDEX"] == str(index)
        assert task["environment"]["PAYLOAD_SHA256"] == "{{payload_sha256}}"
        assert (
            task["environment"]["ASTRA_ENTRY_MODULE"]
            == "astra_reversal.osmo.ood_distributed"
        )
        assert "inputs" not in task  # Independent tasks, with no scheduling dependency.
        assert task["credentials"]["astra-reversal-inference-20260924"] == {
            "NVIDIA_INFERENCE_API_KEY": "api_key"
        }
