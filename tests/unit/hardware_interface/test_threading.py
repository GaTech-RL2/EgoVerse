"""Policy execution boundaries and native replay semantics for the paired pilot."""

import copy
import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
import yaml
from test_adaptive import ScratchFixture, command, router

from astra_reversal.hardware_interface import threading_worker as worker
from astra_reversal.hardware_interface.common import digest, write_json
from astra_reversal.hardware_interface.policy_modes import PolicyRouter
from astra_reversal.hardware_interface.preflight import source_hash
from astra_reversal.hardware_interface.threading_protocol import (
    ARMS,
    load,
    schedule,
    validate,
)


def policy(tmp_path, mode):
    base = router(tmp_path, "F", name=mode)
    return PolicyRouter(
        base.proxy, None, "F_adaptive", policy_mode=mode, scratch=ScratchFixture()
    )


def test_cap_only_runs_control_through_generated_programs(tmp_path):
    r = policy(tmp_path, "CAP")
    assert (
        r.dispatch_model("act", {"envelope": command(r)})["error"] == "tool_not_allowed"
    )
    assert r.proxy.step_count == 0
    result = r.dispatch_model(
        "scratch_execute", {"code": "result = act(" + repr(command(r)) + ")"}
    )
    assert result["result"]["accepted"] and r.proxy.step_count == 1
    stale = command(r)
    stale["metadata"]["observation_step"] = 0
    result = r.dispatch_model(
        "scratch_execute", {"code": "result = act(" + repr(stale) + ")"}
    )
    assert not result["result"]["accepted"] and r.proxy.step_count == 1


def test_cap_can_discover_and_execute_the_same_command_contract_as_direct(tmp_path):
    cap, direct = policy(tmp_path, "CAP"), policy(tmp_path, "direct")
    tools = {tool["name"]: tool for tool in cap.tools}
    assert "act" not in tools
    description = tools["scratch_execute"]["description"]
    contract = json.loads(description.split("\nact(envelope) schema: ", 1)[1])
    direct_contract = next(t for t in direct.tools if t["name"] == "act")["parameters"][
        "properties"
    ]["envelope"]
    assert contract == direct_contract
    jsonschema.validate(command(cap), contract)
    # A generated program uses only its documented read packet for fresh IDs.
    result = cap.dispatch_model(
        "scratch_execute",
        {
            "code": "\n".join(
                [
                    "results=[]",
                    "for _ in range(2):",
                    " packet=read('robot.eef_position')",
                    " envelope=" + repr(command(cap)),
                    " envelope['metadata']['episode_id']=packet['metadata']['episode_id']",
                    " envelope['metadata']['observation_step']=packet['metadata']['simulator_step']",
                    " results.append(act(envelope))",
                    "result=results",
                ]
            )
        },
    )
    assert all(row["accepted"] for row in result["result"])
    assert cap.proxy.step_count == 2


@pytest.mark.parametrize("mode", ["CAP", "direct"])
def test_adaptive_helpers_cannot_smuggle_a_robot_controller(tmp_path, mode):
    r = policy(tmp_path, mode)
    definition = {
        "name": "hidden_controller",
        "description": "fixture",
        "code": "result = act(params['command'])",
        "read_only": False,
        "reason": "fixture",
    }
    assert "error" in r.dispatch_model("define_interface_tool", definition)
    assert r.proxy.step_count == 0
    definition["read_only"] = True
    assert "error" not in r.dispatch_model("define_interface_tool", definition)
    result = r.dispatch_model(
        "call_interface_tool",
        {
            "name": "hidden_controller",
            "arguments_json": json.dumps({"command": command(r)}),
        },
    )
    assert result["result"]["error"] == "read_only_interface_tool"
    assert r.proxy.step_count == 0
    catalog = r.dispatch_model("interface_catalog", {})
    assert "camera_calibration" in catalog["available_capabilities"]


def test_direct_native_action_and_no_executable_controller(tmp_path):
    r = policy(tmp_path, "direct")
    assert (
        r.dispatch_model("scratch_execute", {"code": "result = act({})"})["error"]
        == "tool_not_allowed"
    )
    assert r.dispatch_model("act", {"envelope": command(r)})["accepted"]
    assert r.proxy.step_count == 1


def test_replay_subprocess_uses_importable_module_from_cli_and_drops_credentials(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(worker, "__name__", "__main__")
    monkeypatch.setattr(
        worker.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setenv("HARDWARE_API_KEY_FILE", "/private/key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fixture")
    worker.isolated_replay(tmp_path, tmp_path / "manifest.yaml")
    args, kwargs = calls[0]
    assert args[1:4] == [
        "-m",
        "astra_reversal.hardware_interface.threading_worker",
        "replay",
    ]
    assert not any(
        k in kwargs["env"] for k in ("HARDWARE_API_KEY_FILE", "R2_SECRET_ACCESS_KEY")
    )


def test_schedule_is_four_cells_on_the_same_ten_starts():
    manifest = load()
    rows = schedule(manifest)
    assert len(rows) == 40 and len({r["trial_id"] for r in rows}) == 40
    for seed in range(10):
        block = [r for r in rows if r["env_seed"] == seed]
        assert {r["condition"] for r in block} == set(ARMS)
        assert {r["interface_condition"] for r in block} == {"F_adaptive"}
        assert len({r["init_state_index"] for r in block}) == 1
    bad = copy.deepcopy(manifest)
    bad["limits"]["steps"] = 1000
    with pytest.raises(ValueError, match="design_changed"):
        validate(bad)


def test_failed_reset_pair_preserves_the_mismatched_fields(tmp_path):
    reference = {
        "reset": {"state_sha256": "same", "model_sha256": "before"},
        "initial_sensor_sha256": "before",
    }
    actual = copy.deepcopy(reference)
    assert worker.check_reset_pair(reference, actual, tmp_path)
    assert not (tmp_path / "reset-mismatch.json").exists()
    actual["reset"]["model_sha256"] = "after"
    actual["initial_sensor_sha256"] = "after"
    assert not worker.check_reset_pair(reference, actual, tmp_path)
    mismatch = json.loads((tmp_path / "reset-mismatch.json").read_text())
    assert mismatch["different_fields"] == ["initial_sensor_sha256", "reset"]
    assert mismatch["different_reset_fields"] == ["model_sha256"]
    assert mismatch["reference"] == reference and mismatch["actual"] == actual


def test_pilot_workflow_has_separate_model_credentials_and_no_libero_launch(
    tmp_path, monkeypatch
):
    from experiments.threading_hardware_interface import prepare as package

    def base(destination):
        destination.mkdir()
        (destination / "bootstrap.sh").write_text("old LIBERO launch")
        (destination / "workflow.yaml").write_text(
            yaml.safe_dump(
                {
                    "workflow": {
                        "name": "old",
                        "groups": [
                            {
                                "name": "old",
                                "tasks": [
                                    {"name": "inference", "environment": {}},
                                    {
                                        "name": "evaluation",
                                        "environment": {
                                            "HARDWARE_MANIFEST": "old-libero.yaml"
                                        },
                                        "credentials": {"r2": {}},
                                    },
                                ],
                            }
                        ],
                    }
                }
            )
        )
        return {"source_commit": "a" * 40}

    monkeypatch.setattr(package, "prepare_group", base)
    destination = tmp_path / "pilot"
    receipt = package.prepare_pilot(destination)
    spec = yaml.safe_load((destination / "workflow.yaml").read_text())
    inference, evaluation = spec["workflow"]["groups"][0]["tasks"]
    assert "credentials" not in inference
    assert (
        evaluation["credentials"]["astra-hardware-inference-20261010"]
        == "/run/hardware-inference"
    )
    assert "HARDWARE_MANIFEST" not in evaluation["environment"]
    assert evaluation["environment"]["HARDWARE_STAGE"] == "pilot"
    assert "RESCHEDULE" not in evaluation["exitActions"]
    assert receipt["planned_trials"] == 40 and receipt["conditions"] == list(ARMS)
    bootstrap = (destination / "bootstrap.sh").read_text()
    assert "python -m experiments.threading_hardware_interface.campaign" in bootstrap
    assert "self_hosted_campaign" not in bootstrap


class ReplayEnvironment:
    def __init__(self, success_at=None, model_changed=False):
        self.steps, self.success_at = 0, success_at
        self.controller = {"same": True}
        self.reset_receipt = {"state_sha256": "same"}
        self.observation = {k: np.zeros([1]) for k in worker.NATIVE_KEYS}
        self.env = SimpleNamespace(
            sim=SimpleNamespace(
                get_state=lambda: SimpleNamespace(
                    flatten=lambda: np.array([self.steps])
                ),
                model=SimpleNamespace(
                    body_pos=np.array([int(model_changed)]), body_quat=np.array([0])
                ),
            )
        )

    def step(self, action):
        self.steps += 1

    def check_success(self):
        return self.steps == self.success_at

    def close(self):
        pass


@pytest.mark.parametrize(
    "fault", [None, "early_success", "terminal_model_changed", "success_changed"]
)
def test_replay_checks_first_success_and_full_terminal_state(
    tmp_path, monkeypatch, fault
):
    manifest = load()
    env = ReplayEnvironment(
        success_at=1 if fault == "early_success" else None,
        model_changed=fault == "terminal_model_changed",
    )
    monkeypatch.setattr(worker, "environment", lambda *_: env)
    write_json(
        tmp_path / "trial.json",
        {
            "row": {"init_state_index": 1},
            "manifest_sha256": digest(manifest),
            "source_sha256": source_hash(),
            "initial": worker.reset_identity(env),
        },
    )
    write_json(
        tmp_path / "outcome.json",
        {"success": fault == "success_changed", "sim_steps": 2},
    )
    (tmp_path / "events.jsonl").write_text(
        "\n".join(
            json.dumps({"event": "action_applied", "applied_action": [0] * 7})
            for _ in range(2)
        )
    )
    np.savez(tmp_path / "terminal.npz", state=[2], body_pos=[0], body_quat=[0])
    if fault:
        with pytest.raises(ValueError, match="threading_replay_"):
            worker.replay(tmp_path, manifest)
        assert not (tmp_path / "independent-replay.json").exists()
    else:
        receipt = worker.replay(tmp_path, manifest)
        assert receipt["status"] == "passed" and receipt["replayed_steps"] == 2
