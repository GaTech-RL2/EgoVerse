"""An overnight report must expose missingness and verify archive provenance."""

import hashlib
import json

import pytest
from botocore.exceptions import ClientError

from astra_reversal.hardware_interface.protocol import design
from experiments.libero_hardware_interface.tools.parallel_results import (
    RECORDS,
    snapshot,
    verify_record,
)


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
