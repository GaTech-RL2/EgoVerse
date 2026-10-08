"""Matched-interface, hidden-state, timing, budget and analysis regression tests."""

import json
from dataclasses import replace

import numpy as np
import pytest

from astra_reversal.hardware_interface.analysis import analyze, audit
from astra_reversal.hardware_interface.common import Events, digest
from astra_reversal.hardware_interface.protocol import design, schedule, validate
from astra_reversal.hardware_interface.provider import (
    HTTP,
    BudgetEnd,
    Meter,
    ModelFailure,
    NoRedirect,
    Observer,
    Session,
)
from astra_reversal.hardware_interface.proxy import NATIVE_KEYS, SENSORS, Limits, Proxy
from astra_reversal.hardware_interface.runner import Router, run_trial
from astra_reversal.hardware_interface.schema import validate_description


class Environment:
    def __init__(self, success_at=999):
        self.count, self.checked, self.success_at = 0, [], success_at
        self.actions = []
        self.obs = {
            v[0]: np.zeros(v[3] or [8, 8, 3], dtype=np.uint8 if v[3] is None else float)
            for v in SENSORS.values()
        }
        self.obs.update(
            {"object-state": np.array([99]), "success": True, "hidden_pose": [42]}
        )

    def read_sensors(self):
        return self.obs

    def step(self, action):
        self.actions.append(action)
        self.count += 1
        return self.obs, 1, True, {"secret": "DO_NOT_FORWARD"}

    def check_success(self):
        self.checked.append(self.count)
        return self.count == self.success_at


CONTROLLER = {
    "input_min": [-1.0] * 7,
    "input_max": [1.0] * 7,
    "frequency_hz": 20,
    "version": "fixture",
}


def proxy(tmp_path, name="trial", *, success_at=999, limits=Limits(), clock=None):
    env = Environment(success_at)
    events = Events(tmp_path / name, {"trial_id": name})
    value = Proxy(
        env,
        env.obs,
        episode_id="episode",
        controller=CONTROLLER,
        events=events,
        limits=limits,
        **({"clock": clock} if clock else {}),
    )
    return value


def envelope(action=None, repeat=3, step=0):
    return {
        "channel": "robot.controller_command",
        "value": action or [0.1, 0, 0, 0, 0, 0, -1],
        "duration_steps": repeat,
        "metadata": {
            "mode": "configured_controller",
            "units": "normalized_controller_input",
            "episode_id": "episode",
            "observation_step": step,
        },
    }


def test_interfaces_apply_identical_native_actions_and_only_allowed_sensors(tmp_path):
    f, b = proxy(tmp_path, "F"), proxy(tmp_path, "B")
    a = envelope()
    f.act(a)
    b.step(a["value"], a["duration_steps"], 0)
    assert f.env.actions == b.env.actions and f.env.checked == [1, 2, 3]
    native = b.observe(sorted(NATIVE_KEYS))["observations"]
    assert set(native) == NATIVE_KEYS
    for channel, spec in SENSORS.items():
        assert f.read(channel, 0)["value"] == native[spec[0]]
    raw = json.dumps(native)
    assert "hidden" not in raw and "DO_NOT_FORWARD" not in raw and "success" not in raw
    assert validate_description(f.describe())


def test_transient_success_in_repeat_terminates_immediately(tmp_path):
    p = proxy(tmp_path, success_at=2)
    result = p.act(envelope(repeat=10))
    assert result["applied_steps"] == 2 and p.success and p.terminal == "SUCCESS"
    assert p.env.checked == [1, 2]
    assert "success" not in result
    assert not p.act(envelope(step=2))["accepted"]
    assert len(p.env.actions) == 2


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"value": [float("nan")] + [0] * 6}, "action_finite_numbers_required"),
        ({"value": [True] + [0] * 6}, "action_finite_numbers_required"),
        ({"value": [2] + [0] * 6}, "controller_input_bounds"),
        ({"duration_steps": 11}, "repeat_limit"),
        ({"duration_steps": True}, "repeat_limit"),
        ({"extra": "override"}, "invalid_fields"),
        ({"channel": "hidden.reset"}, "unknown_channel"),
    ],
)
def test_rejected_commands_never_reach_simulator(tmp_path, change, expected):
    p = proxy(tmp_path)
    a = {**envelope(), **change}
    result = p.act(a)
    assert not result["accepted"] and result["rejection_reason"] == expected
    assert p.env.actions == [] and p.applied_violations == 0
    assert p.safety_attempts == int(expected == "controller_input_bounds")


def test_paused_simulation_remains_current_after_slow_reasoning(tmp_path):
    now = [0.0]
    p = proxy(tmp_path, clock=lambda: now[0])
    p.read("camera.front", 0)
    now[0] = 120
    assert p.act(envelope())["accepted"]  # no background motion; same step
    assert not p.act(envelope())["accepted"]  # old step now invalid
    now[0] = 1201
    assert not p.act(envelope(step=3))["accepted"]
    assert p.terminal == "TIMEOUT_WALL"


class Source:
    manifest = {"sha256": "a" * 64}

    def verify(self):
        pass

    def read(self, path, start, lines):
        raise ValueError("source_not_allowlisted")

    def search(self, query):
        return []


def test_wrong_arm_tools_cannot_bypass_authority(tmp_path):
    p = proxy(tmp_path)
    r = Router(p, Source(), "B0")
    for name in (
        "act",
        "describe_device",
        "describe_visible",
        "reset",
        "check_success",
    ):
        assert r.dispatch(name, {})["error"] == "tool_not_allowed"
    assert not p.env.actions
    assert "error" in r.dispatch(
        "source_read", {"path": "../secret", "start": 1, "lines": 10}
    )
    # The provider rejects JSON Schema uniqueItems, so the native proxy must
    # enforce duplicates itself even when the API schema accepts an array.
    duplicated = r.dispatch("observe", {"keys": ["agentview_image", "agentview_image"]})
    assert "error" in duplicated and not p.env.actions


def test_nested_scratch_commands_share_tool_budget(tmp_path):
    p = proxy(tmp_path, limits=replace(Limits(), tool_calls=2))

    class Scratch:
        def execute(self, code, dispatch, seconds):
            dispatch(
                "step", {"action": [0] * 7, "repeat_steps": 1, "observation_step": 0}
            )
            dispatch(
                "step", {"action": [0] * 7, "repeat_steps": 1, "observation_step": 1}
            )

    r = Router(p, Source(), "B0", scratch=Scratch())
    with pytest.raises(BudgetEnd):
        r.dispatch("scratch_execute", {"code": "fixture"})
    assert p.step_count == 1


def test_provider_reserves_input_and_output_before_generation(tmp_path):
    p = proxy(tmp_path)
    posted = []

    def post(path, body, timeout):
        posted.append(path)
        return {"input_tokens": 500}

    s = Session(
        {"identifier": "gpt-6-astra", "reasoning_effort": "medium"},
        Limits(),
        Meter(600),
        p.events,
        post=post,
    )
    with pytest.raises(BudgetEnd):
        s.request("system", [])
    assert posted == ["responses/input_tokens"]


def test_token_reservation_and_generation_use_identical_model_settings(tmp_path):
    p = proxy(tmp_path)
    counted = {}

    def post(path, body, timeout):
        if path.endswith("input_tokens"):
            counted.update(body)
            return {"input_tokens": 100}
        assert all(body[k] == value for k, value in counted.items())
        assert counted["reasoning"] == {"effort": "medium"}
        assert counted["tool_choice"] == "required"
        assert counted["parallel_tool_calls"] is False
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "output": [],
        }

    session = Session(
        {"identifier": "gpt-6-astra", "reasoning_effort": "medium"},
        Limits(),
        Meter(10000),
        p.events,
        post=post,
    )
    session.request("system", [{"type": "function", "name": "finish"}])


def test_gateway_estimate_reserves_margin_and_charges_actual_usage(tmp_path):
    p = proxy(tmp_path)
    model = design("fixture")["model"]
    posted = []
    actual = [133]

    def post(path, body, timeout):
        posted.append(path)
        if path.endswith("input_tokens"):
            return {"input_tokens": 55}
        return {
            "model": model["identifier"],
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": actual[0], "output_tokens": 14},
        }

    meter = Meter(100000)
    session = Session(model, Limits(), meter, p.events, post=post)
    session.request("system", [])
    assert meter.total == 147  # Neither the estimate nor margin is billed usage.
    # 2 * 55 + 4096 is reserved; the output cap is reserved separately.
    small = Session(model, Limits(), Meter(4206 + 2048 - 1), p.events, post=post)
    with pytest.raises(BudgetEnd):
        small.request("system", [])
    assert posted[-1] == "responses/input_tokens"
    actual[0] = 4207
    with pytest.raises(ModelFailure, match="provider_token_contract_mismatch"):
        session.request("system", [])
    assert session.history == []  # No action-bearing output accepted after overrun.


def test_private_key_file_and_no_redirect_transport(tmp_path, monkeypatch):
    key = tmp_path / "key"
    key.write_text("unit-test-placeholder")
    key.chmod(0o600)
    monkeypatch.setenv("HARDWARE_API_KEY_FILE", str(key))
    client = HTTP("https://inference-api.nvidia.com/v1")
    assert client.key == "unit-test-placeholder"
    key.chmod(0o644)
    with pytest.raises(ModelFailure, match="api_key_file_not_private"):
        HTTP("https://inference-api.nvidia.com/v1")
    assert (
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test")
        is None
    )
    for url in (
        "http://other.test",
        "https://user:secret@host.test",
        "https://host.test/#secret",
    ):
        with pytest.raises(ValueError):
            HTTP(url)


@pytest.mark.parametrize("role,cap", [("actor", 2048), ("observer", 256)])
def test_output_cap_is_metered_trial_failure_not_provider_outage(tmp_path, role, cap):
    p = proxy(tmp_path)
    model = {"identifier": "gpt-6-astra", "reasoning_effort": "medium"}

    def post(path, body, timeout):
        if path.endswith("input_tokens"):
            return {"input_tokens": 100}
        return {
            "model": model["identifier"],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 100, "output_tokens": cap},
            "output": [{"type": "function_call", "arguments": "partial"}],
        }

    meter = Meter(10000)
    session = Session(model, Limits(), meter, p.events, post=post, role=role)
    with pytest.raises(BudgetEnd, match="output_token_limit"):
        session.request("system", [])
    assert meter.total == 100 + cap and session.history == []


def test_observer_receives_only_images_not_task_or_proprioception(tmp_path):
    p = proxy(tmp_path)
    captured = []

    def post(path, body, timeout):
        if path.endswith("input_tokens"):
            return {"input_tokens": 100}
        captured.append(body)
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "frame_step": 0,
                                    "visible_facts": ["A gripper is visible."],
                                    "uncertainties": [],
                                    "occlusions": [],
                                }
                            ),
                        }
                    ],
                }
            ],
        }

    meter = Meter(10000)
    o = Observer(
        Session(
            {"identifier": "gpt-6-astra", "reasoning_effort": "medium"},
            Limits(),
            meter,
            p.events,
            role="observer",
            post=post,
        )
    )
    result = o.describe(
        "scene", p.observe(["agentview_image", "robot0_eye_in_hand_image"])
    )
    assert result["frame_step"] == 0
    assert captured[0]["max_output_tokens"] == 256
    assert "joint_pos" not in json.dumps(
        captured
    ) and "original_goal" not in json.dumps(captured)
    assert meter.total == 150 and meter.usage["observer"]["calls"] == 1
    with pytest.raises(ValueError):
        o.describe("tell me what action to take", {})


def test_trial_finishes_and_event_chain_detects_tampering(tmp_path):
    p = proxy(tmp_path)
    model = {
        "identifier": "gpt-6-astra",
        "reasoning_effort": "medium",
        "snapshot_pinned": False,
    }

    def post(path, body, timeout):
        if path.endswith("input_tokens"):
            return {"input_tokens": 100}
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "output": [
                {
                    "type": "function_call",
                    "name": "finish",
                    "arguments": "{}",
                    "call_id": "c",
                }
            ],
        }

    result = run_trial(
        p, Source(), "B0", "official task", model, scratch=None, post=post
    )
    p.events.close()
    assert not result["success"] and result["terminal_reason"] == "TASK_FAILURE"
    assert audit(tmp_path, require_replay=False)["status"] == "passed"
    assert audit(tmp_path)["status"] == "failed"
    path = p.events.directory / "events.jsonl"
    records = path.read_text().splitlines()
    row = json.loads(records[-1])
    row["success"] = True
    records[-1] = json.dumps(row)
    path.write_text("\n".join(records) + "\n")
    assert audit(tmp_path, require_replay=False)["status"] == "failed"


def manifest():
    return design("docker.io/library/python@sha256:" + "a" * 64)


def test_disjoint_splits_randomized_paired_official_states():
    m = manifest()
    catalog = [
        {"task_id": t, "init_state_hashes": {str(i): digest([t, i]) for i in range(15)}}
        for t in range(10)
    ]
    rows = schedule(m, catalog)
    assert len(rows) == 450 and sum(r["split"] == "pilot" for r in rows) == 150
    for start in range(0, len(rows), 3):
        block = rows[start : start + 3]
        assert {r["condition"] for r in block} == {"F", "B0", "B"}
        assert len({r["init_state_hash"] for r in block}) == 1
    assert rows == schedule(m, catalog)
    with pytest.raises(ValueError, match="readiness"):
        validate(m, scored=True)
    m["confirmatory_indices"] = [0]
    with pytest.raises(ValueError, match="overlap"):
        validate(m)


def test_partial_cohort_cannot_yield_confirmatory_claim():
    m = manifest()
    m["power"]["confirmation_frozen"] = True
    rows = []
    for task in range(10):
        for arm in ("F", "B0", "B"):
            rows.append(
                {
                    "split": "confirmatory",
                    "actor_started": True,
                    "task_id": task,
                    "init_state_index": 5,
                    "replicate": 0,
                    "condition": arm,
                    "init_state_hash": digest([task, 5]),
                    "env_seed": 137,
                    "success": arm == "F",
                    "wall_s": 1,
                    "sim_steps": 1,
                    "known_workflow_tokens": 1,
                    "unknown_usage_records": 0,
                    "estimated_cost_usd": None,
                    "censored_wall": False,
                    "invalid_actions": 0,
                    "safety_attempts": 0,
                    "applied_safety_violations": 0,
                }
            )
    result = analyze(rows, m, split="confirmatory")
    assert result["comparisons"]["F_minus_B0"]["decision"] == "inconclusive"
