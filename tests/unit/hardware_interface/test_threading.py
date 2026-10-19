"""Policy execution boundaries and native replay semantics for the paired pilot."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
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
