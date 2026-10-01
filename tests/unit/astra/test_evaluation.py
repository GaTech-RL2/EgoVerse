import json
from dataclasses import asdict

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionAdapter
from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.diagnostics import diagnose, require_diagnostics
from astra_reversal.evaluate import paired_comparison, summarize
from astra_reversal.libero_runner import load_task_manifest, quat_to_axisangle
from astra_reversal.records import Recorder, digest

from .conftest import SyntheticEnvironment, SyntheticPolicy


def test_suite_presets_override_all_protocol_fields_together():
    config = RunConfig.from_dict({"benchmark": {"suite": "libero_goal_ood"}})
    assert config.benchmark.task_action_budget == 300
    assert config.benchmark.trials_per_task == 10
    assert config.benchmark.reset_source == "seeded_reset_stream"
    assert config.benchmark.protocol == "released_modified_libero"
    with pytest.raises(ValueError):
        RunConfig.from_dict(
            {"benchmark": {"suite": "libero_goal_ood", "task_action_budget": 520}}
        )
    with pytest.raises(ValueError):
        RunConfig.from_dict({"benchmark": {"suite": "custom"}})


def test_repeated_seed_bootstrap_clusters_states_and_pairs_all_keys():
    left, right = [], []
    for state in range(2):
        for seed in range(3):
            row = {
                "suite": "libero_10",
                "task_id": 0,
                "initial_state_id": state,
                "seed": seed,
                "success": False,
                "reset_state_sha256": f"state{state}",
                "protocol": "standard_libero_10",
                "task_action_budget": 520,
                "split": "test",
            }
            left.append(row)
            right.append({**row, "success": state == 0})
    result = paired_comparison(left, right, samples=5000, seed=1)
    assert result["difference_right_minus_left"] == 0.5
    assert result["paired_bootstrap_95_percent"] == [0, 1]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(left, right[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_comparison(left + [left[0]], right)
    bad = [{**r, "protocol": "strict_unmodified"} for r in right]
    with pytest.raises(ValueError, match="protocol"):
        paired_comparison(left, bad)


def test_macro_success_weights_tasks_not_number_of_trials():
    common = {
        "suite": "libero_10",
        "actions": 20,
        "wall_seconds": 1,
        "velocity_evaluations": 10,
        "fallbacks": 0,
    }
    rows = [{**common, "task_id": 0, "success": True} for _ in range(10)]
    rows.append({**common, "task_id": 1, "success": False})
    result = summarize(rows)
    assert result["macro_success"] == 0.5
    assert not result["per_suite"]["libero_10"]["complete_ten_task_suite"]


def test_recorder_preserves_arrays_and_rejects_overwrite(tmp_path):
    recorder = Recorder(tmp_path / "run")
    array = np.arange(320, dtype=np.float32).reshape(1, 10, 32)
    recorder.event("inversion", noise=array)
    item = json.loads((recorder.directory / "events.jsonl").read_text())
    np.testing.assert_array_equal(
        np.load(recorder.directory / item["noise"]["array"], allow_pickle=False), array
    )
    with pytest.raises(FileExistsError):
        Recorder(tmp_path / "run")


def test_frozen_reset_manifest_detects_edits_and_protocol_mismatch(tmp_path):
    config = RunConfig()
    state = np.arange(10, dtype=np.float64)
    manifest = {
        "schema_version": "1.0",
        "benchmark": asdict(BenchmarkConfig()),
        "split": "development",
        "episodes": [
            {"reset_state": state.tolist(), "reset_state_sha256": digest(state)}
        ],
    }
    manifest["sha256"] = digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_task_manifest(path, config)["sha256"] == manifest["sha256"]
    manifest["episodes"][0]["reset_state"][0] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed"):
        load_task_manifest(path, config)


def test_diagnostics_measure_known_noise_and_require_matching_artifacts(spec):
    policy = SyntheticPolicy()
    actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    report = diagnose(
        policy,
        actions,
        SyntheticEnvironment().observe(),
        "move cup",
        action_atol=1e-4,
        noise_atol=1e-4,
        parity_atol=1e-4,
    )
    assert all(row["passed"] for row in report["results"])
    require_diagnostics(report, policy, actions, RunConfig())
    report["checkpoint"] = {"artifact": "different"}
    with pytest.raises(ValueError, match="different"):
        require_diagnostics(report, policy, actions, RunConfig())


def test_axisangle_does_not_mutate_quaternion():
    quaternion = np.array([0.0, 0.0, 0.0, 1.0000001])
    original = quaternion.copy()
    np.testing.assert_array_equal(quat_to_axisangle(quaternion), 0)
    np.testing.assert_array_equal(quaternion, original)


def test_request_hash_preserves_nested_structure():
    assert digest({"a": {"b": 1}, "c": 2}) != digest({"a": {"b": 1, "c": 2}})


def test_recorded_episode_roundtrip_includes_latency_and_agent_costs(spec, tmp_path):
    from astra_reversal.agent import AstraAgent
    from astra_reversal.controller import Controller
    from astra_reversal.evaluate import read_episodes, trace_costs

    from .conftest import SyntheticBackend

    recorder = Recorder(tmp_path / "synthetic_run")
    config, policy = RunConfig(), SyntheticPolicy()
    agent = AstraAgent(SyntheticBackend(), config.agent, recorder)
    actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
    controller = Controller(config, policy, actions, recorder, agent)
    controller.run_episode(
        SyntheticEnvironment(stop_after=6),
        episode_id="synthetic",
        instruction="move cup",
        seed=1,
        metadata={"suite": "libero_10", "task_id": 0, "initial_state_id": 0, "seed": 1},
    )
    episodes = read_episodes(recorder.directory)
    assert len(episodes) == 1 and episodes[0]["agent_calls"] == 1
    costs = trace_costs(recorder.directory)
    assert costs["counts"]["agent_calls"] == 1
    assert (
        costs["counts"]["velocity_evaluations"] == episodes[0]["velocity_evaluations"]
    )
    assert "condition_preparation" in costs["latency_seconds"]
    assert summarize(episodes)["per_task"]["libero_10:0"]["mean_agent_calls"] == 1
