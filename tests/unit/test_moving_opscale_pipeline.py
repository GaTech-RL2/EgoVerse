"""The smoke jobs depend only on their verified norm caches; production needs all."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    directory = (
        Path(__file__).resolve().parents[2] / "experiments/fold_clothes_moving_opscale"
    )
    monkeypatch.syspath_prepend(str(directory))
    spec = spec_from_file_location(
        "moving_pipeline_under_test", directory / "pipeline.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        run_root=tmp_path, commit="a" * 40, python=Path("/fixture/python"), gpu="h200"
    )
    state = {"design_sha256": "design", "actions": {}}
    actions, evidence = [], []
    done = {"final-data-check": True, "normalization": False, "evaluation-smoke": False}
    monkeypatch.setattr(module, "guard", lambda commit: None)
    monkeypatch.setattr(module, "digest", lambda path: "design")
    monkeypatch.setattr(module, "scheduler", lambda job: ("COMPLETED", {}))

    def read(path):
        if path.name == "staging.json":
            return {"job_id": "stage"}
        if path.name == "readiness.json":
            return {
                "status": "passed",
                "checks": {"complete": True},
                "checked_at": module.now(),
            }
        if path.name == "norm_complete.json":
            return json.loads(path.read_text())
        if path.name == "evaluation_smoke.json":
            return {
                "status": "passed",
                "commit": args.commit,
                "design_sha256": "design",
            }
        raise AssertionError(f"Unexpected report access: {path}")

    def ensure(state, root, key, body, **kwargs):
        actions.append(key)
        return {"key": key}

    monkeypatch.setattr(module, "read_json", read)
    monkeypatch.setattr(module, "ensure_job", ensure)
    monkeypatch.setattr(module, "await_action", lambda action: done[action["key"]])
    monkeypatch.setattr(
        module, "norm_evidence", lambda *args, cells=None: evidence.append(cells)
    )

    def ready(count, status="passed"):
        path = tmp_path / "norm" / f"b24_op{count}" / "norm_complete.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": status,
                    "commit": args.commit,
                    "design_sha256": "design",
                    "budget_hours": 24,
                    "operator_count": count,
                }
            )
        )

    return SimpleNamespace(
        module=module,
        args=args,
        state=state,
        actions=actions,
        evidence=evidence,
        done=done,
        ready=ready,
    )


def test_smoke_waits_for_both_reports(scenario):
    s = scenario
    s.ready(4)
    assert s.module.step(s.state, s.args) == "COMPUTING_NORMALIZATION"
    assert "evaluation-smoke" not in s.actions


def test_two_verified_norms_start_smoke_before_array_finishes(scenario):
    s = scenario
    s.ready(4)
    s.ready(128)
    assert s.module.step(s.state, s.args) == "NORMALIZATION_AND_GPU_SMOKE"
    assert s.evidence == [((24, 4), (24, 128))]
    assert "evaluation-smoke" in s.actions
    assert "production" not in s.state["actions"]


def test_invalid_smoke_norm_provenance_prevents_launch(scenario, monkeypatch):
    s = scenario
    s.ready(4)
    s.ready(128)

    def invalid(*args, **kwargs):
        raise RuntimeError("Normalization evidence mismatch")

    monkeypatch.setattr(s.module, "norm_evidence", invalid)
    with pytest.raises(RuntimeError, match="Normalization evidence mismatch"):
        s.module.step(s.state, s.args)
    assert "evaluation-smoke" not in s.actions


def test_smoke_success_does_not_release_production_before_all_norms(scenario):
    s = scenario
    s.ready(4)
    s.ready(128)
    s.done["evaluation-smoke"] = True
    assert s.module.step(s.state, s.args) == "WAITING_FOR_REMAINING_NORMALIZATION"
    assert "production" not in s.state["actions"]


def test_completed_array_requires_all_norm_evidence_while_smoke_runs(scenario):
    s = scenario
    s.ready(4)
    s.ready(128)
    s.done["normalization"] = True
    assert s.module.step(s.state, s.args) == "RUNNING_GPU_SMOKE"
    assert s.evidence == [((24, 4), (24, 128)), None]
    assert "production" not in s.state["actions"]


def test_failed_array_blocks_even_when_smoke_norms_exist(scenario, monkeypatch):
    s = scenario
    s.ready(4)
    s.ready(128)

    def fail(action):
        if action["key"] == "normalization":
            raise RuntimeError("Preflight job failed")
        return True

    monkeypatch.setattr(s.module, "await_action", fail)
    with pytest.raises(RuntimeError, match="Preflight job failed"):
        s.module.step(s.state, s.args)
    assert "evaluation-smoke" not in s.actions


def test_failed_array_element_is_detected_before_other_records_arrive(
    scenario, monkeypatch
):
    monkeypatch.setattr(
        scenario.module, "command", lambda *args, **kwargs: "10_0|FAILED|1:0\n"
    )
    # Restore the actual scheduler rather than the fixture's stage stub.
    directory = Path(scenario.module.__file__)
    namespace = {}
    source = directory.read_text()
    start = source.index("def scheduler(")
    end = source.index("\n\ndef guard(", start)
    namespace.update(command=scenario.module.command, TERMINAL=scenario.module.TERMINAL)
    exec(source[start:end], namespace)
    assert namespace["scheduler"]("10", 36)[0] == "FAILED"


def test_running_normalization_reports_wait_without_blocking(scenario):
    s = scenario
    s.ready(4, status="running")
    s.ready(128, status="running")
    assert s.module.step(s.state, s.args) == "COMPUTING_NORMALIZATION"
    assert "evaluation-smoke" not in s.actions
    assert not s.evidence


def test_one_passed_and_one_running_report_still_waits(scenario):
    s = scenario
    s.ready(4)
    s.ready(128, status="running")
    assert s.module.step(s.state, s.args) == "COMPUTING_NORMALIZATION"
    assert "evaluation-smoke" not in s.actions


def test_failed_normalization_report_blocks_without_launch(scenario):
    s = scenario
    s.ready(4)
    s.ready(128, status="failed")
    with pytest.raises(RuntimeError, match="failed or is malformed"):
        s.module.step(s.state, s.args)
    assert "evaluation-smoke" not in s.actions


def test_staging_transfer_acceptance_survives_provisional_final_report(
    scenario, monkeypatch
):
    s = scenario
    s.done["final-data-check"] = False
    monkeypatch.setattr(s.module, "scheduler", lambda job: ("FAILED", {}))
    reports = {
        "staging.json": {"job_id": "stage"},
        "readiness.json": {
            "status": "blocked",
            "design_sha256": "design",
            "verified_episodes": 1,
            "failures": [{"episode": "missing"}],
            "selected_episodes": 2,
        },
    }
    monkeypatch.setattr(s.module, "read_json", lambda path: reports[path.name])
    assert s.module.step(s.state, s.args) == "VERIFYING_FINAL_DATA"
    assert s.state["accepted_staging"]["transfer_reached_verification"] is True
    reports["readiness.json"] = {"status": "blocked", "checks": {}}
    assert s.module.step(s.state, s.args) == "VERIFYING_FINAL_DATA"
    assert "normalization" not in s.actions


def test_refresh_job_provisional_report_does_not_fail_prior_data_gate(
    scenario, monkeypatch
):
    s = scenario
    s.state["actions"]["refresh-readiness"] = {"key": "refresh-readiness"}
    s.done["refresh-readiness"] = False
    reports = {
        "staging.json": {"job_id": "stage"},
        "readiness.json": {"status": "blocked", "checks": {}},
    }
    monkeypatch.setattr(s.module, "read_json", lambda path: reports[path.name])
    assert s.module.step(s.state, s.args) == "REFRESHING_READINESS"
    assert "normalization" not in s.actions


def test_completed_with_nonzero_exit_is_failed(scenario, monkeypatch):
    monkeypatch.setattr(
        scenario.module, "command", lambda *args, **kwargs: "10|COMPLETED|1:0\n"
    )
    namespace = {
        "command": scenario.module.command,
        "TERMINAL": scenario.module.TERMINAL,
    }
    source = Path(scenario.module.__file__).read_text()
    exec(
        source[source.index("def scheduler(") : source.index("\n\ndef guard(")],
        namespace,
    )
    assert namespace["scheduler"]("10")[0] == "FAILED"
