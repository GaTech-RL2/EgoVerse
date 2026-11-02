"""Operational controllers retain separate immutable training source identities."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def launcher():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/fold_clothes_moving_opscale/launch_pace.py"
    )
    spec = importlib.util.spec_from_file_location("dual_source_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repository(path, files):
    path.mkdir()
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    git("init", "-q")
    git("config", "user.email", "unit-test@example.invalid")
    git("config", "user.name", "Unit Test")
    git("add", ".")
    git("commit", "-qm", "Synthetic immutable fixture")
    return git("rev-parse", "HEAD")


@pytest.fixture
def sources(tmp_path, launcher, monkeypatch):
    training, controller = tmp_path / "training", tmp_path / "controller"
    training_commit = _repository(training, {"application.py": "# training source\n"})
    controller_commit = _repository(
        controller,
        {name: "# operational controller\n" for name in launcher.CONTROLLER_SCRIPTS},
    )
    monkeypatch.setattr(launcher, "CONTROLLER_REPO", controller)
    monkeypatch.setattr(launcher, "REPO", controller)
    monkeypatch.setattr(
        launcher, "__file__", str(controller / launcher.CONTROLLER_SCRIPTS[0])
    )
    return training, training_commit, controller, controller_commit


def test_controller_and_training_are_independently_pinned(launcher, sources):
    training, training_commit, controller, controller_commit = sources
    proof = launcher.configure_training_source(
        training, controller_commit, training_commit
    )
    assert launcher.REPO == training
    assert launcher.CONTROLLER_REPO == controller
    assert proof["repository"] == str(controller)
    assert proof["commit"] == controller_commit
    assert proof["training_repository"] == str(training)
    assert proof["training_commit"] == training_commit
    assert set(proof["scripts"]) == set(launcher.CONTROLLER_SCRIPTS)
    assert all(
        proof["scripts"][name] == launcher.digest(controller / name)
        for name in proof["scripts"]
    )


@pytest.mark.parametrize("which", ["training", "controller"])
def test_dirty_either_source_is_rejected(launcher, sources, which):
    training, training_commit, controller, controller_commit = sources
    path = training if which == "training" else controller
    (path / "unexpected.py").write_text("# unreviewed\n")
    with pytest.raises(RuntimeError, match="dirty"):
        launcher.configure_training_source(training, controller_commit, training_commit)


@pytest.mark.parametrize("which", ["training", "controller"])
def test_wrong_full_source_commit_is_rejected(launcher, sources, which):
    training, training_commit, _, controller_commit = sources
    with pytest.raises(RuntimeError, match="commit changed"):
        launcher.configure_training_source(
            training,
            "0" * 40 if which == "controller" else controller_commit,
            "0" * 40 if which == "training" else training_commit,
        )


def test_paired_options_and_full_commit_are_required(launcher, sources):
    training, training_commit, _, controller_commit = sources
    with pytest.raises(RuntimeError, match="supplied together"):
        launcher.configure_training_source(training, None, training_commit)
    with pytest.raises(RuntimeError, match="Full source commit"):
        launcher.configure_training_source(
            training, controller_commit[:12], training_commit
        )


def _registry(launcher, sources):
    training, training_commit, _, controller_commit = sources
    proof = launcher.operational_source(training, training_commit, controller_commit)
    return {
        "repository": str(training),
        "commit": training_commit,
        "operational_controller": proof,
    }


def test_worker_restores_training_source_and_rejects_tampered_binding(
    launcher, sources
):
    training, _, controller, _ = sources
    registry = _registry(launcher, sources)
    assert launcher.REPO == controller
    assert (
        launcher.restore_registry_source(registry) == registry["operational_controller"]
    )
    assert launcher.REPO == training
    registry["operational_controller"]["scripts"][launcher.CONTROLLER_SCRIPTS[1]] = (
        "0" * 64
    )
    with pytest.raises(RuntimeError, match="checksums changed"):
        launcher.restore_registry_source(registry)


def test_worker_rejects_foreign_controller_before_gate_or_training(
    launcher, sources, monkeypatch, tmp_path
):
    registry = _registry(launcher, sources)
    registry["operational_controller"]["repository"] = str(
        tmp_path / "wrong-controller"
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    monkeypatch.setattr(launcher, "require_production_enabled", lambda root: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Gate/training must not run before controller validation")

    monkeypatch.setattr(launcher, "validate_gate", forbidden)
    with pytest.raises(RuntimeError, match="recorded operational controller"):
        launcher.execute(path, "b24_op4")


def test_legacy_registry_does_not_change_default_source(launcher):
    original = launcher.REPO
    assert launcher.restore_registry_source({"commit": "a" * 40}) is None
    assert launcher.REPO == original


def test_gate_default_uses_current_training_source(launcher, sources, monkeypatch):
    training, _, _, _ = sources
    monkeypatch.setattr(launcher, "REPO", training)
    gate = {
        "schema_version": 1,
        "status": "passed",
        "checks": {k: True for k in launcher.GATE_CHECKS},
    }
    ready = {
        "schema_version": 1,
        "status": "passed",
        "checks": {k: True for k in launcher.DATA_CHECKS},
    }
    monkeypatch.setattr(
        launcher, "read_json", lambda path: gate if path == "gate" else ready
    )

    class CheckedRuntimeRepo(Exception):
        pass

    def command(args, **kwargs):
        assert args[:4] == ["git", "-C", str(training), "rev-parse"]
        raise CheckedRuntimeRepo

    monkeypatch.setattr(launcher, "command", command)
    with pytest.raises(CheckedRuntimeRepo):
        launcher.validate_gate("gate", "ready")


def test_submission_records_controller_and_routes_worker_training_and_monitor(
    launcher, sources, monkeypatch, tmp_path
):
    training, training_commit, controller, controller_commit = sources
    root = tmp_path / "run"
    root.mkdir()
    gate_path = root / "gate.json"
    gate = {
        "commit": training_commit,
        "design_sha256": "fixed-design",
        "private_root": str(root / "private"),
    }
    gate_path.write_text(json.dumps(gate))
    (root / "production_control.json").write_text(
        json.dumps({"allow_training": True, "scope": "production"})
    )
    entry = {
        "budget_hours": 24,
        "operator_count": 4,
        "config_sha256": "config",
        "split_sha256": "split",
    }
    monkeypatch.setattr(
        launcher, "validate_gate", lambda *args, **kwargs: (gate, {}, {(24, 4): entry})
    )
    monkeypatch.setattr(
        launcher, "verified_norm", lambda *args: (root / "norm/stats.json", "stats")
    )
    monkeypatch.setattr(launcher, "wandb_auth_available", lambda: True)
    for name in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "synthetic-test-value")
    monitors, submissions = [], []
    original_popen = launcher.subprocess.Popen

    def fake_popen(args, **kwargs):
        if len(args) > 1 and str(args[1]).endswith("/monitor.py"):
            monitors.append(args)
            return SimpleNamespace(pid=123)
        return original_popen(args, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    def fake_submit(run_root, args, **kwargs):
        submissions.append(args)
        return "12345"

    monkeypatch.setattr(launcher, "submit_training_job", fake_submit)
    launcher.submit(
        SimpleNamespace(
            submit=True,
            gate=gate_path,
            readiness=root / "readiness.json",
            run_root=root,
            python=Path(sys.executable),
            gpu="h200",
            cells=["b24_op4"],
            training_repo=training,
            controller_commit=controller_commit,
        )
    )
    registry = json.loads((root / "registry.json").read_text())
    assert registry["repository"] == str(training)
    assert registry["commit"] == training_commit
    assert registry["operational_controller"]["commit"] == controller_commit
    assert monitors[0][1] == str(
        training / "experiments/fold_clothes_moving_opscale/monitor.py"
    )
    script = root / "b24_op4/job.sbatch"
    text = script.read_text()
    assert str(controller / launcher.CONTROLLER_SCRIPTS[0]) in text
    assert "_execute" in text and str(root / "registry.json") in text
    assert f"cd {training}" in text
    assert f"export PYTHONPATH={training}:{root}/python-deps\n" in text
    assert "${PYTHONPATH" not in text
    assert registry["runs"]["b24_op4"]["sbatch_command"] == submissions[0]
    assert submissions[0][-1] == str(script)
    assert registry["runs"]["b24_op4"]["job_id"] == "12345"
