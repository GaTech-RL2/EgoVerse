"""Preflight completion must never silently release production training."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_moving_opscale_pipeline import scenario  # noqa: F401


@pytest.fixture
def modules(monkeypatch):
    folder = (
        Path(__file__).resolve().parents[2] / "experiments/fold_clothes_moving_opscale"
    )
    monkeypatch.syspath_prepend(str(folder))
    return importlib.import_module("launch_pace"), importlib.import_module("monitor")


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"allow_training": False},
        {"allow_training": "true"},
        [],
        {"allow_training": True, "scope": "preflight_only"},
    ],
)
def test_production_defaults_to_denied(modules, tmp_path, value):
    launch, _ = modules
    if value is not None:
        (tmp_path / "production_control.json").write_text(json.dumps(value))
    assert launch.production_enabled(tmp_path) is False


def test_malformed_control_denies_training(modules, tmp_path):
    launch, _ = modules
    (tmp_path / "production_control.json").write_text("{invalid")
    assert launch.production_enabled(tmp_path) is False


def test_submit_denied_before_gate_or_scheduler_access(modules, tmp_path, monkeypatch):
    launch, _ = modules
    gate = Mock(side_effect=AssertionError("No gate access before control check"))
    command = Mock(side_effect=AssertionError("No scheduler access"))
    monkeypatch.setattr(launch, "validate_gate", gate)
    monkeypatch.setattr(launch, "command", command)
    with pytest.raises(RuntimeError, match="Production training is disabled"):
        launch.submit(SimpleNamespace(run_root=tmp_path, submit=True))
    gate.assert_not_called()
    command.assert_not_called()
    assert not (tmp_path / "registry.json").exists()


def test_every_training_submission_rechecks_current_control(
    modules, tmp_path, monkeypatch
):
    launch, _ = modules
    control = tmp_path / "production_control.json"
    control.write_text('{"allow_training": true, "scope": "production"}')
    command = Mock(return_value="12345")
    monkeypatch.setattr(launch, "command", command)
    assert launch.submit_training_job(tmp_path, ["sbatch", "fixture"]) == "12345"
    control.write_text('{"allow_training": false, "scope": "preflight_only"}')
    with pytest.raises(RuntimeError, match="Production training is disabled"):
        launch.submit_training_job(tmp_path, ["sbatch", "fixture"])
    assert command.call_count == 1


def test_worker_and_timeout_continuation_also_require_opt_in(
    modules, tmp_path, monkeypatch
):
    launch, _ = modules
    command = Mock(side_effect=AssertionError("No scheduler access"))
    monkeypatch.setattr(launch, "command", command)
    registry = tmp_path / "registry.json"
    with pytest.raises(RuntimeError, match="Production training is disabled"):
        launch.execute(registry, "b24_op4")
    assert launch.continue_timed_out_run(registry, "b24_op4", "123", "TIMEOUT") is False
    command.assert_not_called()
    assert not (tmp_path / "submission.lock").exists()


def test_all_preflight_stages_finish_without_production(scenario, monkeypatch):  # noqa: F811
    s = scenario
    s.ready(4)
    s.ready(128)
    s.done["normalization"] = s.done["evaluation-smoke"] = True
    original_read = s.module.read_json
    monkeypatch.setattr(
        s.module,
        "read_json",
        lambda p: {"status": "passed"} if p.name == "gate.json" else original_read(p),
    )
    command = Mock(return_value="")
    production = Mock(side_effect=AssertionError("Production subprocess must not run"))
    monkeypatch.setattr(s.module, "command", command)
    monkeypatch.setattr(s.module.subprocess, "run", production)
    assert s.module.step(s.state, s.args) == "PREFLIGHT_COMPLETE"
    assert s.actions == ["final-data-check", "normalization", "evaluation-smoke"]
    assert s.evidence == [((24, 4), (24, 128)), None]
    assert "validate_setup.py" in command.call_args.args[0][1]
    production.assert_not_called()
    assert "production" not in s.state["actions"]
    report = json.loads((s.args.run_root / "preflight_complete.json").read_text())
    assert report["status"] == "passed" and report["scope"] == "preflight_only"
    assert len(report["normalization_sha256"]) == 36
    assert not (s.args.run_root / "registry.json").exists()


def test_monitor_distinguishes_verified_preflight_from_training_completion(
    modules, tmp_path
):
    launch, monitor = modules
    commit = "a" * 40
    pipeline = {
        "state": "PREFLIGHT_COMPLETE",
        "phase": "PREFLIGHT_COMPLETE",
        "commit": commit,
    }
    launch.write_json(tmp_path / "pipeline_status.json", pipeline)
    for name in ("readiness.json", "gate.json", "evaluation_smoke.json"):
        launch.write_json(tmp_path / name, {"status": "passed"})
    checksums = {}
    for b, n in launch.MATRIX:
        cell = f"b{b}_op{n}"
        path = tmp_path / "norm" / cell / "norm_complete.json"
        launch.write_json(path, {"status": "passed", "commit": commit})
        checksums[cell] = launch.digest(path)
    report = {
        "status": "passed",
        "scope": "preflight_only",
        "commit": commit,
        "readiness_sha256": launch.digest(tmp_path / "readiness.json"),
        "gate_sha256": launch.digest(tmp_path / "gate.json"),
        "smoke_sha256": launch.digest(tmp_path / "evaluation_smoke.json"),
        "normalization_sha256": checksums,
    }
    launch.write_json(tmp_path / "preflight_complete.json", report)
    value = monitor.snapshot(tmp_path)
    assert value["state"] == "PREFLIGHT_COMPLETE"
    assert value["runs"] == {}
    launch.write_json(
        tmp_path / "norm/b24_op4/norm_complete.json", {"status": "failed"}
    )
    assert monitor.snapshot(tmp_path)["state"] == "PREFLIGHT_COMPLETE_UNVERIFIED"
