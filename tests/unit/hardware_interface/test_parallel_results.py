"""An overnight report must expose missingness and verify archive provenance."""

import copy
import hashlib
import json

import pytest
from botocore.exceptions import ClientError

from astra_reversal.hardware_interface.protocol import design
from experiments.libero_hardware_interface.tools.parallel_results import (
    RECORDS,
    snapshot,
    source_for_owner,
    verify_record,
)


def test_source_amendment_is_scoped_to_one_owner_and_reset_validation_only():
    cohort = {
        "plan": {
            "experiment_source_sha256": "a" * 64,
            "experiment_source_commit": "b" * 40,
        },
        "source_amendments": {
            "reset-worker": {
                "base_source_sha256": "a" * 64,
                "base_source_commit": "b" * 40,
                "source_sha256": "c" * 64,
                "changed_runtime_files": ["libero.py", "reset_validation.py"],
                "native_validation": {"status": "passed"},
            }
        },
    }
    assert source_for_owner(cohort, "original-worker") == "a" * 64
    assert source_for_owner(cohort, "reset-worker") == "c" * 64
    for field, value in (
        ("base_source_sha256", "d" * 64),
        ("base_source_commit", "e" * 40),
        ("changed_runtime_files", ["provider.py"]),
        ("native_validation", {"status": "failed"}),
    ):
        bad = copy.deepcopy(cohort)
        bad["source_amendments"]["reset-worker"][field] = value
        with pytest.raises(ValueError, match="invalid_reset_source_amendment"):
            source_for_owner(bad, "reset-worker")


def records():
    expected = {
        "trial_id": "pilot-t00-i00-r0-F",
        "condition": "F",
        "task_id": 0,
        "init_state_index": 0,
        "replicate": 0,
        "env_seed": 137,
        "init_state_hash": "a" * 64,
        "split": "pilot",
    }
    contents = {
        "trial.json": json.dumps({"row": expected, "source_sha256": "b" * 64}).encode(),
        "outcome.json": json.dumps(
            {**expected, "sim_steps": 3, "success": False}
        ).encode(),
        "independent-replay.json": json.dumps(
            {
                "status": "passed",
                "success": False,
                "replayed_steps": 3,
                "terminal_state_matched": True,
            }
        ).encode(),
        "events.jsonl": b"example event bytes",
    }
    receipt = {
        "files": [
            {"path": name, "sha256": hashlib.sha256(contents[name]).hexdigest()}
            for name in RECORDS
        ]
    }
    return expected, contents, receipt


def test_archive_hash_and_frozen_identity_are_required():
    expected, contents, receipt = records()
    assert verify_record(receipt, contents, expected, "b" * 64)[
        "independent_evaluation_passed"
    ]
    with pytest.raises(ValueError, match="experiment_source"):
        verify_record(receipt, contents, expected, "c" * 64)
    with pytest.raises(ValueError, match="frozen_trial"):
        verify_record(receipt, contents, {**expected, "condition": "B"}, "b" * 64)
    contents["outcome.json"] += b" "
    with pytest.raises(ValueError, match="hash_mismatch"):
        verify_record(receipt, contents, expected, "b" * 64)


def test_no_archives_produces_explicit_partial_report(tmp_path):
    class Missing:
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    expected, _, _ = records()
    manifest = design("docker.io/library/python@sha256:" + "a" * 64)
    campaign = {
        "deadline_utc": "2026-10-13T15:00:00+00:00",
        "cohorts": [
            {
                "name": "original",
                "title": "Original cohort",
                "manifest": manifest,
                "schedule": [expected],
                "plan": {"experiment_source_sha256": "b" * 64},
                "owners": [{"workflow": "owned", "trial_ids": [expected["trial_id"]]}],
            }
        ],
    }
    result = snapshot(Missing(), campaign, tmp_path / "cache", tmp_path / "report")
    assert result["complete"] is False
    assert result["cohorts"]["original"]["verified_trials"] == 0
    assert result["cohorts"]["original"]["missing"] == [expected["trial_id"]]
    assert "partial" in (tmp_path / "report/index.html").read_text()
    assert (tmp_path / "report/original/success-by-task.png").exists()


def test_recovery_report_keeps_failed_first_attempt(tmp_path, monkeypatch):
    from experiments.libero_hardware_interface.tools import parallel_results as module

    expected, _, _ = records()
    manifest = design("docker.io/library/python@sha256:" + "a" * 64)
    common = {
        **expected,
        "actor_started": True,
        "independent_evaluation_passed": True,
        "wall_s": 100,
        "known_workflow_tokens": 1000,
        "unknown_usage_records": 0,
        "estimated_cost_usd": None,
        "censored_wall": False,
        "invalid_actions": 0,
        "safety_attempts": 0,
        "applied_safety_violations": 0,
    }
    recovered = {
        **common,
        "success": True,
        "sim_steps": 100,
        "terminal_reason": "SUCCESS",
    }
    interrupted = {
        **common,
        "success": False,
        "sim_steps": 20,
        "terminal_reason": "MODEL_ERROR",
    }
    monkeypatch.setattr(
        module,
        "collect_trial",
        lambda client, owner, *args: interrupted if owner == "earlier" else recovered,
    )
    monkeypatch.setattr(
        module,
        "runtime_receipts",
        lambda *args: {"pending": ["current"], "verified": [], "differences": {}},
    )
    cohort = {
        "name": "original",
        "title": "Original",
        "manifest": manifest,
        "schedule": [expected],
        "plan": {},
        "owners": [{"workflow": "current", "trial_ids": [expected["trial_id"]]}],
        "prior_attempts": [
            {
                "workflow": "earlier",
                "trial_id": expected["trial_id"],
                "reason": "transport interruption",
            }
        ],
    }
    module.snapshot(
        None,
        {"deadline_utc": "2026-10-13T15:00:00+00:00", "cohorts": [cohort]},
        tmp_path / "cache",
        tmp_path / "report",
    )
    first = json.loads(
        (tmp_path / "report/original/first-attempt-analysis.json").read_text()
    )
    final = json.loads((tmp_path / "report/original/analysis.json").read_text())
    assert first["arms"]["F"]["successes"] == 0
    assert final["arms"]["F"]["successes"] == 1
    assert (
        len(json.loads((tmp_path / "report/original/prior-attempts.json").read_text()))
        == 1
    )
