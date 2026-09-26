"""Preserve complete task evidence when later worker work is interrupted."""

import json

import pytest

from astra_reversal.osmo import frs_policy_improvement as worker
from astra_reversal.records import file_sha256


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "RESULTS", tmp_path)
    directory = tmp_path / "task_2"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps({"status": "complete", "task_id": 2})
    )
    (directory / "events.jsonl").write_text('{"kind":"synthetic"}\n')
    (directory / "provider.jsonl").write_text('{"provider_call":false}\n')
    for name in (
        "checkpoint",
        "protocol",
        "frozen_plan",
        "reset_manifest",
        "prompts",
        "frozen_weights_before",
    ):
        (tmp_path / f"{name}.json").write_text("{}")
    (tmp_path / "runtime.json").write_text(
        json.dumps({"workflow": "synthetic", "worker": 1})
    )
    before = {"sha256": "a" * 64, "tensors": {"synthetic": "frozen"}}
    monkeypatch.setattr(worker, "frozen_parameter_receipt", lambda policy: before)
    return directory, before


def test_seal_binds_exact_task_and_metadata_bytes(task):
    directory, before = task
    receipt = worker.seal_completed_task(None, 2, before)
    assert receipt["native_tensor_sha256"] == before["sha256"]
    for name, sha in receipt["task_files_sha256"].items():
        assert sha == file_sha256(directory / name)
    for name, sha in receipt["worker_metadata_sha256"].items():
        assert sha == file_sha256(directory.parent / name)
    assert json.loads((directory / "completion_receipt.json").read_text()) == receipt
    assert not (directory.parent / "frozen_weights_after.json").exists()


def test_changed_native_weights_cannot_seal(task):
    directory, _ = task
    with pytest.raises(RuntimeError, match="tensor bytes changed"):
        worker.seal_completed_task(None, 2, {"sha256": "b" * 64})
    assert not (directory / "completion_receipt.json").exists()


@pytest.mark.parametrize("status,task_id", [("running", 2), ("complete", 3)])
def test_partial_or_wrong_task_cannot_seal(task, status, task_id):
    directory, before = task
    (directory / "summary.json").write_text(
        json.dumps({"status": status, "task_id": task_id})
    )
    with pytest.raises(ValueError, match="expected completed task"):
        worker.seal_completed_task(None, 2, before)
    assert not (directory / "completion_receipt.json").exists()
