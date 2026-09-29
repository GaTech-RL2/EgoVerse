"""Recovery task routing is CPU-only and cannot change scientific inputs."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.frs_experiment import stream_rng
from astra_reversal.osmo import frs_policy_improvement, representation_steering
from astra_reversal.osmo.interpolation import assignment as representation_assignment
from astra_reversal.osmo.task_selection import (
    TASK_IDS_ENV,
    select_assignment,
    selected_entries,
    selected_manifest,
)
from astra_reversal.records import digest
from astra_reversal.representation_search import keyed_rng


@pytest.mark.parametrize(
    "assignment", [frs_policy_improvement.assignment, representation_assignment]
)
@pytest.mark.parametrize("worker", range(8))
def test_default_and_subset_routing(assignment, worker):
    original = assignment("evaluation", worker)
    before = copy.deepcopy(original)
    assert select_assignment("evaluation", original, None) is original
    expected = list(range(worker % 4, 10, 4))
    selected = select_assignment("evaluation", original, str(expected[-1]))
    assert selected["task_ids"] == expected[-1:]
    assert selected["suite"] == original["suite"]
    reordered = select_assignment(
        "evaluation", original, ",".join(map(str, expected[::-1]))
    )
    assert reordered["task_ids"] == expected
    assert original == before


@pytest.mark.parametrize(
    "requested",
    [
        "",
        " ",
        "0,",
        ",0",
        "0,,4",
        "0, 4",
        "-1",
        "+0",
        "0.0",
        "a",
        "0\n",
        "０",
        "0,0",
        "0,00",
        "1",
        "10",
        "0,1",
    ],
)
def test_reject_invalid_subset(requested):
    with pytest.raises(ValueError, match=TASK_IDS_ENV):
        select_assignment(
            "evaluation", frs_policy_improvement.assignment("evaluation", 0), requested
        )


@pytest.mark.parametrize(
    "assignment", [frs_policy_improvement.assignment, representation_assignment]
)
def test_development_override_rejected_even_if_matching(assignment):
    original = assignment("development", 0)
    assert select_assignment("development", original, None) is original
    with pytest.raises(ValueError, match="only allowed during evaluation"):
        select_assignment("development", original, "6")


def test_filter_original_shard_and_preserve_scientific_identity():
    entries = [
        {
            "task_id": task,
            "initial_state_id": 0,
            "seed": 47,
            "episode_id": f"libero_goal_ood:seed47:task{task}:state0",
            "reset_state_sha256": f"state-{task}",
            "reset_model_sha256": f"model-{task}",
        }
        for task in range(10)
    ]
    manifest = {
        "cases": [[task, 0] for task in range(10)],
        "episodes": entries,
        "seed": 47,
    }
    manifest["sha256"] = digest(manifest)
    before = copy.deepcopy(manifest)
    # Shard before filtering: task 5 is worker 1, and must not be re-sharded away.
    chosen = selected_entries(entries[1::4], [5, 9])
    assert [row["task_id"] for row in chosen] == [5, 9]
    result = selected_manifest(manifest, chosen)
    assert result["cases"] == [[5, 0], [9, 0]]
    assert result["episodes"] == [entries[5], entries[9]]
    assert result["sha256"] == digest(
        {k: v for k, v in result.items() if k != "sha256"}
    )
    assert manifest == before
    for row in result["episodes"]:
        original = entries[row["task_id"]]
        assert row is original
        np.testing.assert_array_equal(
            keyed_rng(row, 1, 25).normal(size=10),
            keyed_rng(original, 1, 25).normal(size=10),
        )
        np.testing.assert_array_equal(
            stream_rng(row, 25, 0).normal(size=10),
            stream_rng(original, 25, 0).normal(size=10),
        )
    with pytest.raises(ValueError, match="missing"):
        selected_entries(entries[1::4], [4])


@pytest.mark.parametrize(
    "module,phase_env",
    [
        (frs_policy_improvement, "ASTRA_FRS_PHASE"),
        (representation_steering, "ASTRA_REPRESENTATION_PHASE"),
    ],
)
def test_entrypoints_validate_before_artifact_or_gpu_work(
    monkeypatch, tmp_path, module, phase_env
):
    monkeypatch.setenv(phase_env, "evaluation")
    monkeypatch.setenv("ASTRA_WORKER_INDEX", "1")
    monkeypatch.setenv(TASK_IDS_ENV, "4")
    results = tmp_path / "must-not-create"
    monkeypatch.setattr(module, "RESULTS", results)
    with pytest.raises(ValueError, match="outside the original worker shard"):
        module.main()
    assert not results.exists()


def test_protocol_loaders_are_unaffected_by_routing_environment(monkeypatch):
    frs = frs_policy_improvement.load_protocol()
    representation = representation_steering.load_protocol()
    monkeypatch.setenv(TASK_IDS_ENV, "5")
    assert frs_policy_improvement.load_protocol() == frs
    assert representation_steering.load_protocol() == representation


@pytest.mark.parametrize(
    "module,phase_env",
    [
        (frs_policy_improvement, "ASTRA_FRS_PHASE"),
        (representation_steering, "ASTRA_REPRESENTATION_PHASE"),
    ],
)
@pytest.mark.parametrize("requested", [None, "9,5"])
def test_metadata_routing_keeps_original_capture_order(
    monkeypatch, tmp_path, module, phase_env, requested
):
    import torch

    from astra_reversal import intervention_rollout, libero_runner

    for name, value in {
        phase_env: "evaluation",
        "ASTRA_WORKER_INDEX": "1",
        "ASTRA_RUN_ID": "unit-test",
        "PAYLOAD_SHA256": "payload",
        "ASTRA_SOURCE_REVISION": "revision",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ASTRA_PROTOCOL_PATH", raising=False)
    if requested is None:
        monkeypatch.delenv(TASK_IDS_ENV, raising=False)
    else:
        monkeypatch.setenv(TASK_IDS_ENV, requested)
    results = tmp_path / "results"
    monkeypatch.setattr(module, "RESULTS", results)
    monkeypatch.setattr(
        module, "WorkerArchive", lambda _: SimpleNamespace(sync=lambda **_: None)
    )
    monkeypatch.setattr(module, "initialize_worker_backend", lambda _: None)
    monkeypatch.setattr(module, "native_preflight", lambda _: None)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock L40S")
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(libero_runner, "configure_libero", lambda *args: (None, None))
    captured = []

    def capture(*args, cases, output, seed, **kwargs):
        captured.extend(copy.deepcopy(cases))
        manifest = {
            "cases": cases,
            "seed": seed,
            "episodes": [
                {
                    "task_id": task,
                    "initial_state_id": state,
                    "seed": seed,
                    "episode_id": f"task{task}:state{state}",
                }
                for task, state in cases
            ],
        }
        manifest["sha256"] = digest(manifest)
        module.write_json(output, manifest)
        return manifest

    monkeypatch.setattr(intervention_rollout, "capture_reset_manifest", capture)
    if module is representation_steering:
        monkeypatch.setattr(
            module,
            "load_library",
            lambda _: SimpleNamespace(metadata=lambda: {}, library_id="library"),
        )

        def banks(*args):
            module.write_json(results / "bank_inventory.json", {})
            return {}

        monkeypatch.setattr(module, "load_banks", banks)

    class StopBeforeModel(Exception):
        pass

    def stop():
        raise StopBeforeModel

    monkeypatch.setattr(module, "load_frozen_policy", stop)
    with pytest.raises(StopBeforeModel):
        module.main()
    if module is frs_policy_improvement:
        assert captured == [[task, state] for task in [1, 5, 9] for state in range(11)]
    else:
        assert captured == module.load_protocol()["evaluation_cases"]
    manifest = json.loads((results / "reset_manifest.json").read_text())
    runtime = json.loads((results / "runtime.json").read_text())
    frozen = json.loads((results / "frozen_plan.json").read_text())
    assert json.loads((results / "protocol.json").read_text()) == module.load_protocol()
    expected = [1, 5, 9] if requested is None else [5, 9]
    assert {
        int(identity.split(":")[0][4:]) for identity in frozen["assigned_episodes"]
    } == set(expected)
    if requested is not None:
        assert runtime["assignment"]["task_ids"] == expected
        assert {row["task_id"] for row in manifest["episodes"]} == set(expected)
    else:
        assert manifest["cases"] == captured
