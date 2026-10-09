"""Mobile control authority, transfer scheduling and transport fault accounting."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from astra_reversal.hardware_interface.common import Events, digest
from astra_reversal.hardware_interface.provider import Meter, ModelFailure, Session
from astra_reversal.hardware_interface.proxy import Proxy
from astra_reversal.hardware_interface.robocasa import SENSORS, Environment
from astra_reversal.hardware_interface.robocasa_protocol import (
    limits_for,
    paired_success_times,
    schedule,
)
from astra_reversal.hardware_interface.runner import Router, tools_for

ROOT = Path(__file__).resolve().parents[3]


def manifest():
    return yaml.safe_load(
        (
            ROOT / "experiments/robocasa_hardware_interface/preregistration.yaml"
        ).read_text()
    )


class MobileFixture:
    def __init__(self):
        self.observation = {
            v[0]: np.zeros(v[3] or [8, 8, 3], dtype=float if v[3] else np.uint8)
            for v in SENSORS.values()
        }
        self.observation["hidden_object_position"] = [4, 5, 6]
        self.actions, self.checked = [], []

    def read_sensors(self):
        return self.observation

    def step(self, action):
        self.actions.append(copy.deepcopy(action))
        return self.observation, 999, True, {"privileged_task_state": True}

    def check_success(self):
        self.checked.append(len(self.actions))
        return len(self.actions) == 3


@pytest.mark.parametrize("arm", ["F", "B0", "B"])
def test_mobile_actions_and_images_have_identical_authority(tmp_path, arm):
    env = MobileFixture()
    events = Events(tmp_path / arm, {"trial_id": arm})
    proxy = Proxy(
        env,
        env.observation,
        episode_id="e",
        controller={
            "input_min": [-1] * 12,
            "input_max": [1] * 12,
            "frequency_hz": 20,
            "version": "fixture",
            "device_id": "mobile",
        },
        events=events,
        sensors=SENSORS,
        limits=limits_for(manifest(), 450),
    )
    router = Router(proxy, None, arm)
    schema = next(
        t["parameters"]
        for t in tools_for(arm, 12)
        if t["name"] == ("act" if arm == "F" else "step")
    )
    vector = (
        schema["properties"]["envelope"]["properties"]["value"]
        if arm == "F"
        else schema["properties"]["action"]
    )
    assert vector["minItems"] == vector["maxItems"] == 12
    assert proxy.describe()["actions"][0]["shape"] == [12]
    assert len(proxy.camera_keys) == 3
    assert (
        "hidden_object_position"
        not in proxy.observe(sorted(proxy.native_keys))["observations"]
    )
    assert not proxy.step([0] * 7, 1, 0)["accepted"]
    assert not proxy.step([0] * 11 + [2], 1, 0)["accepted"]
    action = [0.1, 0, 0, 0, 0, 0, -1, 0.2, 0, 0, 0.1, 1]
    if arm == "F":
        result = router.dispatch(
            "act",
            {
                "envelope": {
                    "channel": "robot.controller_command",
                    "value": action,
                    "duration_steps": 5,
                    "metadata": {
                        "mode": "configured_controller",
                        "units": "normalized_controller_input",
                        "episode_id": "e",
                        "observation_step": 0,
                    },
                }
            },
        )
    else:
        result = router.dispatch(
            "step", {"action": action, "repeat_steps": 5, "observation_step": 0}
        )
    assert result["applied_steps"] == 3  # official success checked on every native step
    assert env.actions == [action] * 3 and env.checked == [1, 2, 3]
    assert "privileged_task_state" not in str(result)
    events.close()


def test_upstream_in_place_action_mutation_cannot_change_next_repeat():
    class MutatingNative:
        def __init__(self):
            self.received = []

        def step(self, action):
            self.received.append(action.copy())
            action[7] = 999
            return None

    env = object.__new__(Environment)
    env.env = MutatingNative()
    action = np.arange(12, dtype=float)
    before = action.copy()
    env.step(action)
    env.step(action)
    assert np.array_equal(action, before)
    assert all(np.array_equal(v, before) for v in env.env.received)


def test_transfer_schedule_covers_all_tasks_without_cross_arm_seed_changes():
    frozen = manifest()
    rows = schedule(frozen)
    assert len(rows) == 150 and rows == schedule(frozen)
    assert {r["task_id"] for r in rows} == set(range(50))
    for offset in range(0, len(rows), 3):
        block = rows[offset : offset + 3]
        assert {r["condition"] for r in block} == {"F", "B0", "B"}
        assert len({(r["task_id"], r["env_seed"], r["horizon"]) for r in block}) == 1
    amended = copy.deepcopy(frozen)
    amended["limits"]["workflow_tokens"] = 100000
    with pytest.raises(ValueError, match="resource_caps"):
        schedule(amended)


@pytest.mark.parametrize("code,retries", [(500, 1), (401, 0)])
def test_retry_preserves_request_and_reports_unknown_usage(
    tmp_path, monkeypatch, code, retries
):
    frozen = manifest()
    events = Events(tmp_path / "retry", {"trial_id": "retry"})
    meter = Meter(None)
    sent = []
    monkeypatch.setattr(
        "astra_reversal.hardware_interface.provider.time.sleep", lambda _: None
    )

    def post(path, body, timeout):
        sent.append(copy.deepcopy(body))
        if len(sent) == 1:
            raise ModelFailure("provider_http_" + str(code))
        return {
            "model": frozen["model"]["identifier"],
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 5},
        }

    session = Session(
        frozen["model"], limits_for(frozen, 450), meter, events, post=post
    )
    session.history = [{"role": "user", "content": "example observation"}]
    if retries:
        session.request("instruction", [])
        assert len(sent) == 2 and sent[0] == sent[1]
        assert meter.total == 105
    else:
        with pytest.raises(ModelFailure, match="401"):
            session.request("instruction", [])
        assert len(sent) == 1 and meter.total == 0
    assert meter.unknown == 1
    assert meter.usage["actor"]["failed_transport_requests"] == 1
    events.close()
    records = [
        json.loads(line)
        for line in (events.directory / "events.jsonl").read_text().splitlines()
    ]
    request = next(r for r in records if r["event"] == "model_request")
    assert request["request_sha256"] == digest(sent[0])
    assert request["request"]["input"] == [{"sha256": digest(session.history[0])}]
    assert "example observation" not in json.dumps(request)


def test_time_to_success_excludes_failed_or_unverified_pairs():
    base = {
        "split": "pilot",
        "actor_started": True,
        "replicate": 0,
        "init_state_index": 0,
        "env_seed": 0,
        "init_state_hash": "same",
        "sim_steps": 50,
        "independent_evaluation_passed": True,
    }
    rows = [
        {**base, "task_id": 0, "condition": "F", "success": True, "wall_s": 100},
        {**base, "task_id": 0, "condition": "B0", "success": True, "wall_s": 70},
        {**base, "task_id": 1, "condition": "F", "success": True, "wall_s": 100},
        {**base, "task_id": 1, "condition": "B0", "success": False, "wall_s": 10},
    ]
    result = paired_success_times(rows)["F_minus_B0"]
    assert result["joint_success_pairs"] == 1
    assert result["mean_wall_difference_s"] == 30 and result["F_faster_pairs"] == 0
    rows[1]["independent_evaluation_passed"] = False
    assert paired_success_times(rows)["F_minus_B0"]["joint_success_pairs"] == 0
