"""Adaptive capabilities cannot bypass control, reset, or evaluation contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from astra_reversal.hardware_interface.adaptive import CONDITION
from astra_reversal.hardware_interface.common import Events
from astra_reversal.hardware_interface.protocol import (
    adaptive_design,
    design,
    schedule,
    validate,
    without_resource_budgets,
)
from astra_reversal.hardware_interface.proxy import SENSORS, Limits, Proxy
from astra_reversal.hardware_interface.runner import Router, tools_for


class Robot:
    def __init__(self, action_dim):
        self.steps = 0
        self.action_dim = action_dim
        self.obs = {
            v[0]: np.zeros(v[3] or [8, 8, 3], dtype=np.uint8 if v[3] is None else float)
            for v in SENSORS.values()
        }
        self.obs.update(
            robot0_joint_acc=np.ones(7),
            hidden_object_pose=np.ones(3) * 42,
            success=True,
        )

    def read_sensors(self):
        return self.obs

    def step(self, action):
        self.steps += 1
        self.obs["robot0_eef_pos"] += np.asarray(action[:3]) * 0.02
        return self.obs, 99, False, {"hidden_label": True}

    def check_success(self):
        return False

    def hardware_calibration(self):
        return {"front": {"fovy_degrees": 45}}


class ScratchFixture:
    """Test routing only; real process isolation is checked on Linux OSMO."""

    def execute(self, code, dispatch, seconds):
        scope = {
            "read": lambda channel, max_age_ms=1000: dispatch(
                "read", {"channel": channel, "max_age_ms": max_age_ms}
            ),
            "act": lambda envelope: dispatch("act", {"envelope": envelope}),
            "read_capability": lambda name: dispatch("read_capability", {"name": name}),
        }
        exec(code, scope)
        return {"result": scope.get("result")}


def router(tmp_path, condition=CONDITION, action_dim=7, name="episode"):
    env = Robot(action_dim)
    proxy = Proxy(
        env,
        env.obs,
        episode_id=name,
        controller={
            "input_min": [-1] * action_dim,
            "input_max": [1] * action_dim,
            "frequency_hz": 20,
            "version": "fixture",
        },
        events=Events(tmp_path / name, {"trial_id": name}),
        limits=Limits(steps=8),
    )
    return Router(proxy, None, condition, scratch=ScratchFixture())


def define(r, code, read_only=False):
    return r.dispatch(
        "define_interface_tool",
        {
            "name": "move",
            "description": "measured control helper",
            "code": code,
            "read_only": read_only,
            "reason": "test revision",
        },
    )


def command(r, value=None):
    return {
        "channel": "robot.controller_command",
        "value": value or [0.1] + [0.0] * (r.proxy.action_dim - 1),
        "duration_steps": 1,
        "metadata": {
            "mode": "configured_controller",
            "units": "normalized_controller_input",
            "episode_id": r.proxy.episode_id,
            "observation_step": r.proxy.step_count,
        },
    }


def test_original_arms_cannot_gain_adaptive_capabilities(tmp_path):
    for arm in ("F", "B0", "B"):
        r = router(tmp_path, arm, name=arm)
        assert r.evolution is None
        assert "interface_catalog" not in {t["name"] for t in tools_for(arm)}
        assert (
            r.dispatch(
                "enable_capability", {"name": "camera_calibration", "reason": "request"}
            )["error"]
            == "tool_not_allowed"
        )


def test_allowlisted_sensor_activation_and_fresh_episode(tmp_path):
    r = router(tmp_path)
    catalog = r.dispatch("interface_catalog", {})
    assert "camera_calibration" in catalog["available_capabilities"]
    assert (
        r.dispatch("read_capability", {"name": "camera_calibration"})["error"]
        == "hardware_capability_not_enabled"
    )
    assert (
        r.dispatch("enable_capability", {"name": "success", "reason": "no"})["error"]
        == "hardware_capability_unavailable"
    )
    r.dispatch(
        "enable_capability",
        {"name": "additional_proprioception", "reason": "identify drift"},
    )
    result = r.dispatch("read_capability", {"name": "additional_proprioception"})
    assert set(result["values"]) == {"robot0_joint_acc"}
    assert "hidden" not in json.dumps(catalog) + json.dumps(result)
    fresh = router(tmp_path, name="fresh")
    assert fresh.evolution.revision == 0 and fresh.evolution.state["capabilities"] == []


@pytest.mark.parametrize("action_dim", [7, 12])
def test_probe_cost_and_empirical_response(tmp_path, action_dim):
    r = router(tmp_path, action_dim=action_dim)
    result = r.dispatch(
        "probe_dynamics",
        {
            "action": [0.2] + [0.0] * (action_dim - 1),
            "repeat_steps": 3,
            "channels": ["robot.eef_position"],
            "reason": "measure translation response",
        },
    )
    assert (
        result["executed_steps"]
        == r.proxy.step_count
        == r.evolution.calibration_steps
        == 3
    )
    assert result["delta"]["robot.eef_position"] == pytest.approx([0.012, 0, 0])
    assert "hidden_label" not in json.dumps(result)
    # A rejected probe cannot consume steps or bypass native bounds.
    rejected = r.dispatch(
        "probe_dynamics",
        {
            "action": [2.0] * action_dim,
            "repeat_steps": 1,
            "channels": ["robot.eef_position"],
            "reason": "invalid",
        },
    )
    assert rejected["executed_steps"] == 0 and r.proxy.step_count == 3


def test_tool_revision_rollback_and_native_guard(tmp_path):
    r = router(tmp_path)
    first = define(r, "result = act(params['command'])")
    result = r.dispatch(
        "call_interface_tool",
        {"name": "move", "arguments_json": json.dumps({"command": command(r)})},
    )
    assert result["result"]["accepted"] and r.proxy.step_count == 1
    define(r, "result = act(params['command'])", read_only=True)
    assert (
        r.dispatch(
            "call_interface_tool",
            {"name": "move", "arguments_json": json.dumps({"command": command(r)})},
        )["result"]["error"]
        == "read_only_interface_tool"
    )
    assert r.proxy.step_count == 1
    r.dispatch(
        "restore_interface",
        {"revision": first["revision"], "reason": "restore control mode"},
    )
    invalid = r.dispatch(
        "call_interface_tool",
        {
            "name": "move",
            "arguments_json": json.dumps({"command": command(r, [3] * 7)}),
        },
    )
    assert not invalid["result"]["accepted"] and r.proxy.step_count == 1
    assert len(list(r.evolution.directory.glob("revision-*.json"))) == 4
    assert "act(params" in (r.evolution.directory / "revision-0001.json").read_text()


def test_bad_tool_definition_does_not_replace_working_version(tmp_path):
    r = router(tmp_path)
    define(r, "result = params['x'] * 2", read_only=True)
    old = copy.deepcopy(r.evolution.state)
    assert "error" in define(r, "def invalid(")
    assert r.evolution.state == old
    assert r.dispatch(
        "call_interface_tool", {"name": "move", "arguments_json": '{"x": 3}'}
    ) == {"result": 6}


def test_adaptive_schedule_keeps_all_fifty_matched_starts():
    original = without_resource_budgets(
        design("docker.io/library/python@sha256:" + "a" * 64)
    )
    manifest = adaptive_design(original)
    validate(manifest)
    assert original["conditions"] == ["F", "B0", "B"]
    catalog = [
        {"task_id": i, "init_state_hashes": {str(j): str(j) for j in range(15)}}
        for i in range(10)
    ]
    rows = [r for r in schedule(manifest, catalog) if r["split"] == "pilot"]
    assert len(rows) == 100
    assert {r["condition"] for r in rows} == {"F", CONDITION}
    for i in range(0, len(rows), 2):
        assert rows[i]["task_id"] == rows[i + 1]["task_id"]
        assert rows[i]["init_state_hash"] == rows[i + 1]["init_state_hash"]
    manifest["interface_evolution"]["calibration_steps"] = "free"
    with pytest.raises(ValueError, match="interface_design"):
        validate(manifest)


def test_checked_in_adaptive_manifests_and_tool_schemas():
    from jsonschema import Draft202012Validator

    from astra_reversal.hardware_interface.robocasa_protocol import (
        schedule as casa_schedule,
    )
    from astra_reversal.hardware_interface.robocasa_protocol import (
        validate as validate_casa,
    )

    root = Path(__file__).resolve().parents[3]
    for benchmark, validator in (("libero", validate), ("robocasa", validate_casa)):
        path = (
            root
            / "experiments"
            / (benchmark + "_hardware_interface")
            / "preregistration-adaptive.yaml"
        )
        manifest = yaml.safe_load(path.read_text())
        validator(manifest)
        if benchmark == "robocasa":
            assert len(casa_schedule(manifest)) == 100
    for dim in (7, 12):
        for tool in tools_for(CONDITION, dim):
            Draft202012Validator.check_schema(tool["parameters"])


def test_actor_sees_revisions_and_calibration_reaches_native_horizon(tmp_path):
    from astra_reversal.hardware_interface.analysis import audit
    from astra_reversal.hardware_interface.runner import run_trial

    class Source:
        manifest = {"sha256": "a" * 64}

        def verify(self):
            pass

    # Construct the same proxy without pre-creating the per-trial extension.
    base = router(tmp_path, "F")
    manifest = without_resource_budgets(design("fixture"))
    base.proxy.limits = Limits(**{**manifest["limits"], "steps": 8})
    calls = [
        (
            "define_interface_tool",
            {
                "name": "scale",
                "description": "derived control helper",
                "code": "result = params['x'] * 2",
                "read_only": True,
                "reason": "test",
            },
        ),
        (
            "revise_interface_notes",
            {"notes": "Measured response hypothesis: gain 0.02.", "reason": "test"},
        ),
        ("call_interface_tool", {"name": "scale", "arguments_json": '{"x": 3}'}),
        (
            "probe_dynamics",
            {
                "action": [0.1, 0, 0, 0, 0, 0, 0],
                "repeat_steps": 8,
                "channels": ["robot.eef_position"],
                "reason": "finite response",
            },
        ),
    ]
    requests = []

    def post(path, body, timeout):
        assert path == "responses" and "max_output_tokens" not in body
        name, arguments = calls[len(requests)]
        requests.append(body)
        return {
            "model": manifest["model"]["identifier"],
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "output": [
                {
                    "type": "function_call",
                    "name": name,
                    "call_id": str(len(requests)),
                    "arguments": json.dumps(arguments),
                }
            ],
        }

    result = run_trial(
        base.proxy,
        Source(),
        CONDITION,
        "fixture task",
        manifest["model"],
        scratch=ScratchFixture(),
        post=post,
    )
    base.proxy.events.close()
    assert result["terminal_reason"] == "TIMEOUT_STEPS" and result["sim_steps"] == 8
    assert result["known_workflow_tokens"] == 480
    assert result["interface_evolution"]["calibration_steps"] == 8
    assert result["interface_evolution"]["tool_executions"] == 1
    assert '"scale"' in requests[1]["instructions"]
    assert "gain 0.02" in requests[2]["instructions"]
    assert audit(tmp_path, require_replay=False)["status"] == "passed"


def test_camera_calibration_has_explicit_optical_conventions():
    from types import SimpleNamespace

    from astra_reversal.hardware_interface.adaptive import camera_calibration

    sim = SimpleNamespace(
        model=SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=[90]),
        data=SimpleNamespace(
            cam_xpos=np.array([[1, 2, 3]]), cam_xmat=np.eye(3).reshape(1, 9)
        ),
    )
    result = camera_calibration(sim, ["front"], 128)["front"]
    assert result["focal_pixels"] == pytest.approx([64, 64])
    assert result["position_world_m"] == [1, 2, 3]
    assert "-z" in result["camera_axes"] and "ideal" in result["provenance"]
