"""Durable rollout evidence publication with mocked storage and weight reads."""

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from astra_reversal.osmo.rollout_checkpoint import (
    _EVALUATION_FIELDS,
    _METADATA,
    RolloutCheckpoint,
    checkpoint_progress,
)
from astra_reversal.records import digest


class FakeArchive:
    def __init__(self):
        self.prefix = "synthetic/run/worker_0"
        self.client = self
        self.uploads = []
        self.syncs = 0
        self.fail_upload = None

    def sync(self):
        self.syncs += 1

    def upload_file(self, filename, bucket, key):
        if len(self.uploads) == self.fail_upload:
            raise OSError("synthetic storage interruption")
        assert bucket == "rldb"
        self.uploads.append((key, Path(filename).read_bytes()))


@pytest.fixture
def evidence(tmp_path):
    results = tmp_path / "results"
    task = results / "task_2"
    (task / "arrays").mkdir(parents=True)
    (task / "arrays/00000000.npy").write_bytes(b"synthetic immutable array")
    (task / "provider.jsonl").write_text('{"invocation_id":"synthetic"}\n')
    protocol = {
        "schema_version": "frs-codex-frozen-evaluation-1.0",
        "astra": {"backend": "codex_relay"},
    }
    tensors = {"synthetic_weight": {"sha256": "a" * 64}}
    before = {"tensors": tensors, "sha256": digest(tensors)}
    row = {
        "episode_id": "suite:seed43:task2:state1",
        "attempt_id": "native_state1",
        "method": "native",
        "round_index": 0,
        "success": False,
        "actions_executed": 300,
    }
    row["physical_run_id"] = f"{row['episode_id']}:{row['attempt_id']}"
    summary = {
        "task_id": 2,
        "status": "running",
        "protocol_sha256": digest(protocol),
        "physical_rollouts": [row],
        "evaluation": [{k: row[k] for k in _EVALUATION_FIELDS}],
    }
    metadata = {name: {} for name in _METADATA}
    metadata.update(
        {
            "runtime.json": {"workflow": "synthetic", "worker": 0},
            "protocol.json": protocol,
            "frozen_weights_before.json": before,
            "frozen_plan.json": {"assigned_episodes": [row["episode_id"]]},
            "reset_manifest.json": {
                "episodes": [{"episode_id": row["episode_id"], "task_id": 2}]
            },
        }
    )
    for name, data in metadata.items():
        (results / name).write_text(json.dumps(data))
    archive = FakeArchive()
    weights = []

    def read_weights():
        weights.append(True)
        return copy.deepcopy(before)

    value = SimpleNamespace(
        results=results,
        task=task,
        summary=summary,
        protocol=protocol,
        before=before,
        weights=weights,
        read_weights=read_weights,
        archive=archive,
    )
    publish_input(value)
    return value


def publish_input(value, *, open_attempt=None):
    (value.task / "summary.json").write_text(json.dumps(value.summary))
    events = []
    family = value.protocol["schema_version"].startswith("frs-")
    for row in value.summary["physical_rollouts"]:
        events.extend(
            [
                {
                    "kind": "rollout_start"
                    if family
                    else "representation_rollout_start",
                    "attempt_id": row["attempt_id"],
                },
                {
                    "kind": "rollout_end"
                    if family
                    else "representation_rollout_complete",
                    "result" if family else "attempt": row,
                },
            ]
        )
        (value.task / f"{row['attempt_id']}.mp4").write_bytes(b"closed synthetic video")
    if open_attempt:
        events.append({"kind": "rollout_start", "attempt_id": open_attempt})
    (value.task / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )


def callback(value):
    return checkpoint_progress(
        value.archive,
        value.results,
        2,
        value.protocol,
        value.before,
        value.read_weights,
    )


def unpack(raw):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as stream:
        return {
            member.name: stream.extractfile(member).read()
            for member in stream.getmembers()
        }


def test_archive_precedes_commit_and_binds_complete_evidence(evidence):
    value = evidence
    checkpoint = callback(value)
    receipt = checkpoint()
    assert len(value.weights) == 1
    archive_key, raw = value.archive.uploads[0]
    commit_key, encoded = value.archive.uploads[1]
    assert (
        archive_key.endswith(".tar.gz")
        and commit_key == archive_key.removesuffix(".tar.gz") + ".json"
    )
    assert json.loads(encoded) == receipt
    assert receipt["archive"] == {
        "key": archive_key,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    files = unpack(raw)
    manifest = json.loads(files["checkpoint/manifest.json"])
    assert (
        receipt["manifest"]["sha256"]
        == hashlib.sha256(files["checkpoint/manifest.json"]).hexdigest()
    )
    assert "results/task_2/arrays/00000000.npy" in files
    assert "results/task_2/provider.jsonl" in files
    assert "results/task_2/native_state1.mp4" in files
    assert all(f"results/{name}" in files for name in _METADATA)
    assert json.loads(files["checkpoint/frozen_weights_after.json"]) == value.before
    assert manifest["evaluation"] == value.summary["evaluation"]
    assert manifest["closed_rollout_ids"] == [
        value.summary["physical_rollouts"][0]["physical_run_id"]
    ]
    for name, identity in manifest["files"].items():
        assert identity == {
            "sha256": hashlib.sha256(files[name]).hexdigest(),
            "bytes": len(files[name]),
        }
    assert not (value.task / "completion_receipt.json").exists()
    assert not (value.task / "frozen_weights_after.json").exists()
    assert json.loads(files["results/task_2/summary.json"])["status"] == "running"


def test_unchanged_and_midrollout_saves_do_not_hash_weights_again(evidence):
    checkpoint = callback(evidence)
    checkpoint()
    evidence.summary["live"] = {"attempt_id": "astra_next", "observation_step": 10}
    publish_input(evidence, open_attempt="astra_next")
    assert checkpoint() is None
    assert len(evidence.weights) == 1 and len(evidence.archive.uploads) == 2


def test_frs_waits_for_evaluation_row_then_commits(evidence):
    checkpoint = callback(evidence)
    rows = evidence.summary.pop("evaluation")
    evidence.summary["evaluation"] = []
    publish_input(evidence)
    assert checkpoint() is None
    assert not evidence.weights and not evidence.archive.uploads
    evidence.summary["evaluation"] = rows
    publish_input(evidence)
    assert checkpoint()["closed_rollout_ids"]


@pytest.mark.parametrize("fail_upload,expected_uploads", [(0, 0), (1, 1)])
def test_interrupted_publication_never_commits_or_advances(
    evidence, fail_upload, expected_uploads
):
    evidence.archive.fail_upload = fail_upload
    checkpoint = callback(evidence)
    with pytest.raises(OSError, match="storage interruption"):
        checkpoint()
    assert checkpoint.closed_ids == ()
    assert len(evidence.archive.uploads) == expected_uploads
    assert all(key.endswith(".tar.gz") for key, _ in evidence.archive.uploads)
    with pytest.raises(RuntimeError, match="cannot advance"):
        checkpoint()
    assert len(evidence.weights) == 1


def test_weight_change_stops_before_any_upload(evidence):
    checkpoint = callback(evidence)
    checkpoint.weight_receipt = lambda: {"sha256": "different"}
    with pytest.raises(RuntimeError, match="tensor bytes changed"):
        checkpoint()
    assert not evidence.archive.uploads and checkpoint.closed_ids == ()


@pytest.mark.parametrize(
    "mutation", ["protocol", "reset", "evaluation", "open_rollout", "video"]
)
def test_mismatched_evidence_cannot_commit(evidence, mutation):
    if mutation == "protocol":
        (evidence.results / "protocol.json").write_text("{}")
    elif mutation == "reset":
        (evidence.results / "reset_manifest.json").write_text('{"episodes":[]}')
    elif mutation == "evaluation":
        evidence.summary["evaluation"][0]["success"] = True
        publish_input(evidence)
    elif mutation == "open_rollout":
        publish_input(evidence, open_attempt="unfinished")
    else:
        (evidence.task / "native_state1.mp4").unlink()
    with pytest.raises(ValueError):
        callback(evidence)()
    assert not evidence.weights and not evidence.archive.uploads


def test_second_checkpoint_is_cumulative_and_has_distinct_immutable_keys(evidence):
    checkpoint = callback(evidence)
    first = checkpoint()
    row = copy.deepcopy(evidence.summary["physical_rollouts"][0])
    row.update(attempt_id="astra_state1", method="astra")
    row["physical_run_id"] = f"{row['episode_id']}:{row['attempt_id']}"
    evidence.summary["physical_rollouts"].append(row)
    evidence.summary["evaluation"].append({k: row[k] for k in _EVALUATION_FIELDS})
    publish_input(evidence)
    second = checkpoint()
    assert len(second["closed_rollout_ids"]) == 2
    assert second["archive"]["key"] != first["archive"]["key"]
    files = unpack(evidence.archive.uploads[2][1])
    assert (
        "results/task_2/native_state1.mp4" in files
        and "results/task_2/astra_state1.mp4" in files
    )
    assert len(evidence.weights) == 2


def test_representation_checkpoint_includes_bank_and_library(evidence):
    evidence.protocol["schema_version"] = "vision-codex-representation-screen-1.0"
    evidence.summary["protocol_sha256"] = digest(evidence.protocol)
    evidence.summary.pop("evaluation")
    (evidence.results / "protocol.json").write_text(json.dumps(evidence.protocol))
    for name in ("bank_inventory.json", "image_library.json"):
        (evidence.results / name).write_text('{"synthetic":"bound"}')
    publish_input(evidence)
    checkpoint = callback(evidence)
    assert isinstance(checkpoint, RolloutCheckpoint)
    receipt = checkpoint()
    assert receipt["family"] == "representation"
    files = unpack(evidence.archive.uploads[0][1])
    assert (
        "results/bank_inventory.json" in files and "results/image_library.json" in files
    )


def test_http_progress_callback_is_unchanged(evidence):
    protocol = {"schema_version": "frs-frozen-evaluation-1.0"}
    progress = checkpoint_progress(
        evidence.archive, evidence.results, 2, protocol, {}, None
    )
    assert progress == evidence.archive.sync
    progress()
    assert evidence.archive.syncs == 1 and not evidence.archive.uploads
