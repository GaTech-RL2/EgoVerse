"""A fresh pilot must not inherit incomplete or changed live commissioning."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from astra_reversal.hardware_interface.protocol import design


def worker_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "experiments/libero_hardware_interface/tools/pilot_worker.py"
    )
    spec = importlib.util.spec_from_file_location("hardware_pilot_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "change", [None, "source", "missing_arm", "replay", "model", "transport"]
)
def test_pilot_requires_all_arms_and_identical_source_and_model(
    tmp_path, monkeypatch, change
):
    worker = worker_module()
    manifest = design("docker.io/library/python@sha256:" + "a" * 64)
    manifest["resolved"] = {
        k: "a" * 64
        for k in (
            "adapter_sha256",
            "observer_sha256",
            "dependency_lock_sha256",
            "asset_manifest_sha256",
            "controller_config_sha256",
            "catalog_sha256",
            "source_allowlist_sha256",
        )
    }
    manifest["readiness"] = {
        k: True
        for k in (
            "reset_replay",
            "interface_equivalence",
            "source_isolation",
            "scratch_isolation",
            "success_per_step",
            "budget_termination",
            "model_smoke_F",
            "model_smoke_B0",
            "model_smoke_B",
        )
    }
    smoke = {
        "result": {"trials": 3, "split": "smoke"},
        "audit": {
            "status": "passed",
            "trials": [
                {"trial": "smoke-" + c, "status": "passed"}
                for c in manifest["conditions"]
            ],
        },
    }
    transport = {
        "passed": True,
        "model": dict(manifest["model"]),
        "arm_tool_schemas": {c: True for c in manifest["conditions"]},
    }
    monkeypatch.setattr(
        worker, "source_hash", lambda: ("b" if change == "source" else "a") * 64
    )
    if change == "missing_arm":
        smoke["audit"]["trials"].pop()
    elif change == "replay":
        smoke["audit"]["trials"][0]["status"] = "failed"
    elif change == "model":
        transport["model"]["reasoning_effort"] = "low"
    elif change == "transport":
        transport["arm_tool_schemas"].pop("B")
    (tmp_path / "ready-preregistration.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "model-smoke.json").write_text(json.dumps(smoke))
    (tmp_path / "transport.json").write_text(json.dumps(transport))
    if change is None:
        assert worker.validate_commissioning(tmp_path) == manifest
    else:
        with pytest.raises(ValueError, match="matching_complete_live_commissioning"):
            worker.validate_commissioning(tmp_path)
