import json

import pytest
import yaml

from astra_reversal.osmo import prepare_learning_launch as launch
from astra_reversal.osmo.reasoning_policy_learning import load_protocol
from astra_reversal.records import file_sha256


def test_unresolved_budget_prevents_gpu_experiment(tmp_path):
    path = tmp_path / "unresolved.json"
    path.write_text(
        json.dumps({"compute": {"launch_allowed": False, "authorized_gpu_hours": None}})
    )
    with pytest.raises(ValueError, match="authorized GPU-hour budget"):
        load_protocol(path)


def test_prepared_worker_is_single_gpu_hard_capped_and_immutable(tmp_path, monkeypatch):
    protocol = {
        "compute": {"authorized_gpu_hours": 2},
        "pilot": {"development_tasks": [{}], "seeds": [173]},
    }
    monkeypatch.setattr(launch, "load_protocol", lambda: protocol)
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda command, **kwargs: (
            "test-revision\n" if command[1] == "rev-parse" else ""
        ),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("payload.tar.gz", "bootstrap.sh"):
        (bundle / name).write_bytes(b"unit-test synthetic payload")
    identity = {
        "source_revision": "test-revision",
        "payload_sha256": file_sha256(bundle / "payload.tar.gz"),
        "bootstrap_sha256": file_sha256(bundle / "bootstrap.sh"),
    }
    (bundle / "source_identity.json").write_text(json.dumps(identity))
    output = tmp_path / "launch"
    receipt = launch.prepare(bundle, output, phase="preflight", gpu_hours=0.5)
    spec = yaml.safe_load((output / "workflow.yaml").read_text())["workflow"]
    assert len(spec["tasks"]) == spec["resources"]["default"]["gpu"] == 1
    assert spec["timeout"]["exec_timeout"] == "30m"
    assert receipt["status"] == "prepared_not_submitted"
    assert "codex_relay" not in (output / "connect.sh").read_text()
    assert (output / "relay.token").stat().st_mode & 0o777 == 0o600
    (bundle / "payload.tar.gz").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        launch.prepare(bundle, tmp_path / "changed", phase="pilot", gpu_hours=0.5)
    with pytest.raises(ValueError, match="budget"):
        launch.prepare(bundle, tmp_path / "excess", phase="pilot", gpu_hours=3)
