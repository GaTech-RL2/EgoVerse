"""Preserve matched-reset and source checks across fresh simulator processes."""

import copy
import json
import subprocess

import pytest

from astra_reversal.hardware_interface import robocasa_worker as worker
from astra_reversal.hardware_interface.common import digest, write_json


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "source_hash", lambda: "pinned")
    return {"resolved": {"source_sha256": "pinned"}}


def request(manifest):
    return {
        "mode": "collect",
        "row": {"name": "Task", "env_seed": 7, "trial_id": "trial", "split": "pilot"},
        "reference": {
            "receipt": {"state_sha256": "first"},
            "observations_sha256": "view",
        },
        "manifest_sha256": digest(manifest),
    }


def test_child_receives_exact_first_arm_reset_before_collection(isolated, monkeypatch):
    value = request(isolated)
    called = []
    monkeypatch.setattr(
        worker, "collect_trial", lambda m, r, groups: called.append(groups)
    )
    worker.dispatch_isolated(isolated, value)
    assert called == [{("Task", 7): value["reference"]}]
    value = {**value, "reference": None}
    worker.dispatch_isolated(isolated, value)
    assert called[-1] == {}


@pytest.mark.parametrize("changed", ["manifest", "source"])
def test_changed_source_or_manifest_never_starts_collection(
    isolated, monkeypatch, changed
):
    value = request(isolated)
    called = []
    monkeypatch.setattr(worker, "collect_trial", lambda *args: called.append(args))
    if changed == "manifest":
        value["manifest_sha256"] = "different"
    else:
        monkeypatch.setattr(worker, "source_hash", lambda: "different")
    with pytest.raises(ValueError, match="isolated_" + changed + "_changed"):
        worker.dispatch_isolated(isolated, value)
    assert not called


@pytest.mark.parametrize(
    "different",
    [None, "receipt", "observations_sha256", "controller", "initial_success"],
)
def test_full_reset_audit_retains_exact_parity_and_valid_start_checks(
    isolated, monkeypatch, tmp_path, different
):
    rows = [
        {"name": "Task", "env_seed": 7, "trial_id": arm, "condition": arm}
        for arm in ["F", "B0", "B"]
    ]
    reference = {"receipt": {"state_sha256": "first"}, "observations_sha256": "view"}
    calls = []

    def run(mode, manifest, row, reference=None):
        assert mode == "reset"
        calls.append(row["trial_id"])
        directory = tmp_path / "reset-check" / row["trial_id"]
        reset = {"receipt": {"state_sha256": "first"}, "observations_sha256": "view"}
        status = {
            "initially_unsolved": True,
            "controller_matched": True,
            "model_calls": 0,
        }
        if row["condition"] == "B0":
            if different == "receipt":
                reset["receipt"]["state_sha256"] = "different"
            elif different == "observations_sha256":
                reset["observations_sha256"] = "different"
            elif different == "controller":
                status["controller_matched"] = False
            elif different == "initial_success":
                status["initially_unsolved"] = False
        write_json(directory / "reset-evidence.json", reset)
        write_json(directory / "native-reset-status.json", status)

    monkeypatch.setattr(worker, "isolated_request", run)
    if different:
        with pytest.raises(ValueError, match="scheduled_reset_check_failed"):
            worker.reset_check(isolated, rows)
    else:
        worker.reset_check(isolated, rows)
    result = json.loads((tmp_path / "reset-check-summary.json").read_text())
    assert result["passed"] is (different is None)
    assert calls == ["F", "B0", "B"]
    assert len(result["checks"]) == 3 and result["model_calls"] == 0
    if different in ["receipt", "observations_sha256"]:
        mismatch = json.loads(
            (tmp_path / "reset-check/B0/reset-mismatch.json").read_text()
        )
        assert mismatch["reference"] == reference


def test_child_failure_keeps_existing_reference_and_propagates(isolated, monkeypatch):
    value = request(isolated)
    groups = {("Task", 7): value["reference"]}
    before = copy.deepcopy(groups)

    def fail(mode, manifest, row, reference=None):
        assert mode == "collect" and reference == value["reference"]
        raise subprocess.CalledProcessError(1, "isolated trial")

    monkeypatch.setattr(worker, "isolated_request", fail)
    with pytest.raises(subprocess.CalledProcessError):
        worker.collect_trial_isolated(isolated, value["row"], groups)
    assert groups == before
