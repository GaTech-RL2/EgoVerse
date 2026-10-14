"""A multi-task native diagnostic must not stop after its first progress receipt."""

import json
from pathlib import Path

from astra_reversal.hardware_interface.common import write_json
from experiments.robocasa_hardware_interface.tools import diagnose_resets as diagnostic


def test_progress_receipt_advances_across_multiple_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(diagnostic.sys, "argv", ["diagnostic", "--task-ids", "4", "11"])
    monkeypatch.setattr(diagnostic, "check_source", lambda *args: None)
    monkeypatch.setattr(
        diagnostic, "verify_fixture_order_fix", lambda *args: {"fixture": True}
    )
    for key in ["HARDWARE_SOURCE_COMMIT", "PAYLOAD_SHA256", "HARDWARE_WORKFLOW"]:
        monkeypatch.setenv(key, "fixture")
    calls, archives = [], []

    def child(args, **kwargs):
        request = json.loads(Path(args[-1]).read_text())
        calls.append(request["row"]["task_id"])
        write_json(Path(request["directory"]) / "probe.json", {"task": calls[-1]})

    monkeypatch.setattr(diagnostic.subprocess, "run", child)
    monkeypatch.setattr(
        diagnostic, "archive_tree", lambda root, label: archives.append(label)
    )
    diagnostic.main()
    summary = json.loads(
        (diagnostic.ROOT / "reset-diagnostic-summary.json").read_text()
    )
    progress = json.loads((diagnostic.ROOT / "diagnostic-progress.json").read_text())
    assert summary["passed"] and summary["model_calls"] == 0
    assert len(summary["checks"]) == 2 and progress == summary["checks"]
    assert sorted(calls) == [4, 4, 11, 11] and archives == ["final"]
