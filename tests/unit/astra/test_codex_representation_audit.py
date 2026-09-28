"""Synthetic prefix/history/application corruption checks; no provider/GPU calls."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import pytest

from astra_reversal import codex_representation_audit as audit
from astra_reversal.records import digest
from astra_reversal.representation_agent import parse_proposal
from astra_reversal.representation_search import load_protocol

from .test_representation_agent import library, proposal_for, request_for, snapshot


def description(raw):
    return {
        key: {
            "array": f"arrays/{key.replace('/', '_')}.npy",
            "sha256": digest(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in raw.items()
    }


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    sources, donors, sheets = library()
    bank = SimpleNamespace(
        library_id=donors[0]["library_id"],
        catalog=lambda: donors,
        contact_sheets=lambda: sheets,
    )
    monkeypatch.setattr(audit, "donor_catalog", lambda: sources)
    protocol = load_protocol(
        Path(audit.__file__).with_name("configs")
        / "vision_codex_representation_screen_v1.json"
    )
    protocol["seed"] = protocol["development_seed"]
    worker, task = tmp_path / "worker", tmp_path / "task"
    worker.mkdir()
    task.mkdir()
    entry = {"episode_id": "synthetic-case", "instruction": "Put the cup in the bowl."}
    raw = snapshot()
    initial, final = snapshot(), snapshot(5)
    baseline = {
        "attempt_id": "native_revision0",
        "arm": "native",
        "revision": 0,
        "success": False,
        "initial_success": False,
        "terminated": True,
        "actions_executed": 5,
        "status": "terminated",
        "wall_seconds": 1,
        "decisions": [],
        "provider_records": [],
        "velocity_evaluations": 10,
        "accepted_decisions_executed": 0,
        "accepted_decision_actions": 0,
        "explicit_native_actions": 0,
        "nonzero_intervention_actions": 0,
        "provider_failure_fallback_actions": 0,
        "generations": [
            {
                "step": 0,
                "choice": None,
                "provider_failure_fallback": False,
                "has_effect": False,
            }
        ],
    }
    previous = {
        "feedback": audit.feedback(baseline),
        "decisions": [],
        "snapshots": [initial, final],
    }
    req = request_for(
        episode_id=entry["episode_id"],
        attempt_id="astra_vei_revision1",
        representation_mode="vei",
        observations=[raw],
        previous_attempt=previous,
        completed_rollout_feedback=[audit.feedback(baseline)],
        action_spec=None,
    )
    choice = parse_proposal(
        proposal_for(req, mode="native", language=None, vision=None), req
    )
    decision = {
        "decision_index": 1,
        "observation_step": 0,
        "accepted": True,
        "error": None,
        "proposal": choice,
    }

    def start(aid, arm, revision):
        return {
            "kind": "representation_rollout_start",
            "attempt_id": aid,
            "arm": arm,
            "revision": revision,
            "entry": entry,
            "action_spec": None,
        }

    def generation(aid, active):
        return {
            "kind": "representation_generation",
            "attempt_id": aid,
            "observation_step": 0,
            "observation": description(raw["observation"]),
            "active_intervention": active,
            "provenance": None,
            "velocity_evaluations": 10,
        }

    events = [
        start("native_revision0", "native", 0),
        generation("native_revision0", None),
        {
            "kind": "representation_rollout_complete",
            "attempt": baseline,
            "snapshots": [
                {**s, "observation": description(s["observation"])}
                for s in (initial, final)
            ],
        },
        start("astra_vei_revision1", "astra_vei", 1),
        {"kind": "representation_request", "request": req},
        {
            "kind": "representation_decision",
            "attempt_id": "astra_vei_revision1",
            "decision": decision,
        },
        generation("astra_vei_revision1", choice),
    ]
    summary = {
        "schema_version": protocol["schema_version"],
        "status": "running",
        "episode_id": entry["episode_id"],
        "task_id": 2,
        "instruction": entry["instruction"],
        "development": True,
        "protocol_sha256": digest(protocol),
        "image_library_id": bank.library_id,
        "reset_entry_sha256": digest(entry),
        "physical_rollouts": [baseline],
    }
    for path, value in (
        (worker / "protocol.json", protocol),
        (task / "summary.json", summary),
        (
            worker / "runtime.json",
            {
                "phase": "development",
                "workflow": "synthetic",
                "worker": 1,
                "packages": {"numpy": np.__version__, "pillow": PIL.__version__},
            },
        ),
    ):
        path.write_text(json.dumps(value))

    def write():
        for index, event in enumerate(events):
            event["sequence"] = index
        (task / "events.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")

    write()
    return task, worker, bank, events, write


def run(prefix, **kwargs):
    task, worker, bank, _, _ = prefix
    return audit.audit_case(task, worker, library=bank, **kwargs)


def test_prefix_never_certifies_missing_archive_or_provider(prefix):
    result = run(prefix)
    assert result["status"] == "preserved_prefix" and not result["complete"]
    assert not result["arrays_verified"] and not result["event_coverage_complete"]
    assert result["completed_attempts"][0]["simulator_success"] is False
    assert (
        result["incomplete_attempts"][0]["outcome"]
        == "unknown_no_completed_rollout_record"
    )
    assert result["incomplete_attempts"][0]["executed_actions_lower_bound"] == 0
    with pytest.raises(ValueError, match="Complete certification"):
        run(prefix, require_complete=True)


def test_stale_generation_rejected(prefix):
    prefix[3][-1]["active_intervention"] = None
    prefix[4]()
    with pytest.raises(ValueError, match="stale/different"):
        run(prefix)


def test_other_rollout_feedback_rejected(prefix):
    prefix[3][4]["request"]["completed_rollout_feedback"][0]["success"] = True
    prefix[4]()
    with pytest.raises(ValueError, match="cross-arm"):
        run(prefix)


def test_current_pixel_swap_rejected(prefix):
    prefix[3][-1]["observation"]["observation/image"]["sha256"] = "0" * 64
    prefix[4]()
    with pytest.raises(ValueError, match="raw camera"):
        run(prefix)


def test_donor_sheet_change_rejected(prefix):
    prefix[3][4]["request"]["contact_sheets"][0]["camera"] = "unexpected"
    prefix[4]()
    with pytest.raises(ValueError):
        run(prefix)


def test_premature_failed_rollout_rejected(prefix):
    prefix[3][2]["attempt"]["terminated"] = False
    prefix[4]()
    with pytest.raises(ValueError, match="Incomplete rollout"):
        run(prefix)


def test_generation_without_new_scheduled_decision_rejected(prefix):
    current = copy.deepcopy(prefix[3][-1])
    for step in (5, 10, 15, 20, 25):
        event = copy.deepcopy(current)
        event["observation_step"] = step
        prefix[3].append(event)
    prefix[4]()
    with pytest.raises(ValueError, match="scheduled fresh decision"):
        run(prefix)


def test_prefix_parser_handles_only_unterminated_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"sequence":0}\n{"sequence":')
    rows, scope = audit.read_prefix(path)
    assert len(rows) == 1 and not scope["entire_file"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Malformed complete"):
        audit.read_prefix(path)


def test_wrong_visual_alpha_rejected():
    choice = {
        "mode": "interpolate",
        "language": None,
        "vision": {"donor_id": "a", "alpha": 0.7},
    }
    provenance = {"operator": "vei", "alpha": 0.2}
    with pytest.raises(ValueError, match="operator/alpha"):
        audit.match_application(choice, provenance, "vei", "task", None)


def test_native_cannot_retain_donor_provenance():
    with pytest.raises(ValueError, match="retained intervention"):
        audit.match_application(
            {"mode": "native"}, {"operator": "vei"}, "vei", "task", None
        )


@pytest.fixture
def archive_input(tmp_path):
    from astra_reversal.records import file_sha256

    task, worker = tmp_path / "task", tmp_path / "worker"
    task.mkdir()
    worker.mkdir()
    summary = {
        "episode_id": "case",
        "protocol_sha256": "protocol",
        "image_library_id": "library",
    }
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"payload_sha256": "payload", "source_revision": "revision"})
    )
    metadata = {
        name: {}
        for name in (
            "protocol.json",
            "checkpoint.json",
            "reset_manifest.json",
            "frozen_plan.json",
            "frozen_weights_before.json",
        )
    }
    metadata["runtime.json"] = {
        "workflow": "workflow",
        "worker": 1,
        "payload_sha256": "payload",
    }
    for name, value in metadata.items():
        (worker / name).write_text(json.dumps(value))
    sources = {}
    for name in ("summary.json", "events.jsonl", "provider.jsonl"):
        (task / name).write_text("{}")
        sources[name] = {
            "sha256": file_sha256(task / name),
            "bytes": 2,
            "entire_file": True,
        }
    hashes = {name: file_sha256(worker / name) for name in metadata}
    seal = {
        "episode_id": "case",
        "workflow": "workflow",
        "worker": 1,
        "case_files": {
            name: {"sha256": row["sha256"]} for name, row in sources.items()
        },
        "worker_files": hashes,
    }
    (task / "completion_receipt.json").write_text(json.dumps(seal))
    receipt = {
        "schema_version": "representation-case-array-audit-1",
        "status": "passed",
        "episode_id": "case",
        "workflow": "workflow",
        "worker": 1,
        "input_file_sha256": {name: row["sha256"] for name, row in sources.items()},
        "worker_metadata_sha256": hashes,
        "counts": {"physical_rollouts": 0, "actions": 0},
        "identities": {
            "protocol_sha256": "protocol",
            "image_library_id": "library",
            "payload_sha256": "payload",
            "source_revision": "revision",
        },
        "checks": {
            "complete_sealed_case": True,
            "all_regular_members_verified": True,
            "all_npy_references_verified": True,
        },
        "archive": {"sha256": "archive"},
    }
    path = tmp_path / "archive.json"
    path.write_text(json.dumps(receipt))
    return path, task, worker, summary, sources, source, receipt


def test_exact_archive_source_seal_binding(archive_input):
    path, task, worker, summary, sources, source, _ = archive_input
    proof = audit.bind_archive(path, task, worker, summary, sources, source, [])
    assert proof["verified"] and proof["complete_sealed_case"]
    missing = audit.bind_archive(path, task, worker, summary, sources, None, [])
    assert not missing["verified"] and missing["reason"] == "source_identity_missing"


@pytest.mark.parametrize(
    "change",
    ["task_bytes", "source_revision", "worker", "npy_proof", "metadata_omitted"],
)
def test_archive_mismatch_cannot_certify(archive_input, change):
    path, task, worker, summary, sources, source, receipt = archive_input
    if change == "task_bytes":
        receipt["input_file_sha256"]["events.jsonl"] = "other"
    if change == "source_revision":
        receipt["identities"]["source_revision"] = "other"
    if change == "worker":
        receipt["worker"] = 2
    if change == "npy_proof":
        receipt["checks"]["all_npy_references_verified"] = False
    if change == "metadata_omitted":
        receipt["worker_metadata_sha256"] = {}
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.bind_archive(path, task, worker, summary, sources, source, [])
