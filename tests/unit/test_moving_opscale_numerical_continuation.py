"""Numerical anomalies must prevent automatic checkpoint continuation."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.fold_clothes_moving_opscale import launch_pace as launch


@pytest.fixture
def scenario(tmp_path):
    run = {
        "wandb_id": "fresh-run",
        "directory": str(tmp_path / "cell"),
        "job_id": "123",
    }
    Path(run["directory"]).mkdir()
    registry = {
        "commit": "a" * 40,
        "runs": {"cell": run},
        "automatic_continuation": {"states": ["TIMEOUT"], "max_attempts": 3},
    }
    return SimpleNamespace(root=tmp_path, run=run, registry=registry)


def latch(s, run_id="fresh-run"):
    commit = s.registry["commit"]
    launch.write_json(
        s.root / "operations/training_numerics.json",
        {
            "schema_version": 1,
            "incidents": {
                commit + ":" + run_id: {
                    "source_commit": commit,
                    "wandb_id": run_id,
                    "first_observed_epoch": 1.0,
                    "evidence": [
                        {
                            "metric": "raw_gradient",
                            "classification": "magnitude",
                            "value": 1000.0,
                            "nonfinite": None,
                            "step": 1,
                        }
                    ],
                }
            },
        },
    )


def test_same_identity_remains_blocked_when_job_changes_or_log_rotates(scenario):
    s = scenario
    latch(s)
    s.run["job_id"] = "456"
    with pytest.raises(RuntimeError, match="numerical incident"):
        launch.require_clean_training_numerics(s.root, s.registry, s.run)
    s.run["wandb_id"] = "new-fresh-run"
    launch.require_clean_training_numerics(s.root, s.registry, s.run)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"schema_version": 1, "incidents": []},
        {"schema_version": 1, "incidents": {"bad": {}}},
    ],
)
def test_malformed_receipt_fails_closed(scenario, value):
    s = scenario
    launch.write_json(s.root / "operations/training_numerics.json", value)
    with pytest.raises(RuntimeError):
        launch.require_clean_training_numerics(s.root, s.registry, s.run)


@pytest.mark.parametrize(
    "text",
    [
        "[GRAD_NORM_SPIKE] step=12 grad_norm=1000.0 median=1",
        "[GRAD_NORM_SPIKE] step=12 grad_norm=1.2e+07 median=1",
        "[GRAD_NORM_SPIKE] step=12 grad_norm=nan median=1",
        "[GRAD_NORM_SPIKE] step=12 grad_norm=-inf median=1",
        '"Train/Loss": 2e13',
        "Train/Loss=+Infinity",
    ],
)
def test_recent_extreme_log_blocks_before_observer_poll(scenario, text):
    s = scenario
    (Path(s.run["directory"]) / "training.log").write_text(text)
    with pytest.raises(RuntimeError, match="numerical anomaly"):
        launch.require_clean_training_numerics(s.root, s.registry, s.run)


def test_ordinary_mad_flags_do_not_block(scenario):
    s = scenario
    (Path(s.run["directory"]) / "training.log").write_text(
        "[GRAD_NORM_SPIKE] step=12 grad_norm=4.5 median=1 threshold=3\nTrain/Loss=0.4"
    )
    launch.require_clean_training_numerics(s.root, s.registry, s.run)


def test_continuation_blocks_before_source_checkpoint_or_scheduler(
    scenario, monkeypatch
):
    s = scenario
    latch(s)
    path = s.root / "registry.json"
    launch.write_json(path, s.registry)
    monkeypatch.setattr(launch, "production_enabled", lambda root: True)
    source = Mock(side_effect=AssertionError("No source/checkpoint access"))
    submit = Mock(side_effect=AssertionError("No allocation"))
    monkeypatch.setattr(launch, "restore_registry_source", source)
    monkeypatch.setattr(launch, "submit_training_job", submit)
    assert launch.continue_timed_out_run(path, "cell", "123", "TIMEOUT") is False
    source.assert_not_called()
    submit.assert_not_called()
    assert (
        json.loads(path.read_text())["runs"]["cell"]["state"] == "CONTINUATION_BLOCKED"
    )


@pytest.mark.parametrize(
    "change", ["timestamp", "empty", "fields", "metric", "small", "step"]
)
def test_malformed_unrelated_incident_also_blocks(scenario, change):
    s = scenario
    latch(s, "unrelated")
    path = s.root / "operations/training_numerics.json"
    value = json.loads(path.read_text())
    item = next(iter(value["incidents"].values()))
    if change == "timestamp":
        item["first_observed_epoch"] = "yesterday"
    elif change == "empty":
        item["evidence"] = []
    elif change == "fields":
        item.pop("first_observed_epoch")
    elif change == "metric":
        item["evidence"][0]["metric"] = "unexpected"
    elif change == "small":
        item["evidence"][0]["value"] = 1.0
    else:
        item["evidence"][0]["step"] = -1
    launch.write_json(path, value)
    with pytest.raises(RuntimeError, match="Malformed numerical"):
        launch.require_clean_training_numerics(s.root, s.registry, s.run)


@pytest.mark.parametrize(
    "text",
    [
        "val/Train/Loss=1e12",
        "otherTrain/Loss=nan",
        "[GRAD_NORM_SPIKE] grad_norm=1000abc",
        "[GRAD_NORM_SPIKE] grad_norm=nanomaly",
    ],
)
def test_lookalike_numeric_fields_are_not_evidence(scenario, text):
    s = scenario
    (Path(s.run["directory"]) / "training.log").write_text(text)
    launch.require_clean_training_numerics(s.root, s.registry, s.run)


def test_numeric_read_has_explicit_byte_limit(scenario, monkeypatch):
    import io

    from experiments.fold_clothes_moving_opscale import training_numerics as numeric

    s = scenario
    path = Path(s.run["directory"]) / "training.log"

    class Bounded(io.BytesIO):
        def read(self, count=-1):
            assert count == numeric.TRAINING_LOG_READ_LIMIT
            return super().read(count)

    monkeypatch.setattr(
        Path, "open", lambda self, *args, **kwargs: Bounded(b"x" * 300000)
    )
    report = numeric.training_numeric_tail(path)
    assert report["bytes_read"] == numeric.TRAINING_LOG_READ_LIMIT
    assert report["evidence"] == []
