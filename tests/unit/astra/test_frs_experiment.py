"""Synthetic end-to-end scheduling and data-boundary checks; no model/network."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.frs_experiment import (
    FRSTaskExperiment,
    first_success,
    load_protocol,
    stream_rng,
    visible_rollout,
)
from astra_reversal.frs_guide import project_policy_pixels
from astra_reversal.records import digest


def observation():
    return {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 21, np.uint8),
        "observation/state": np.zeros(8, np.float32),
    }


class SyntheticPolicy:
    horizon, action_dim = 10, 32
    metadata = {"synthetic": True, "frozen": True}

    @staticmethod
    def tensor(array):
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def noise(rng):
        return rng.standard_normal((1, 10, 32)).astype(np.float32)

    @staticmethod
    def prepare(obs, observation_id, instruction):
        return SimpleNamespace(
            condition_id=digest(obs), state=np.zeros((1, 32), np.float32)
        )

    @staticmethod
    def sample(condition, noise, **kwargs):
        return SimpleNamespace(value=noise * np.float32(0.1), velocity_evaluations=10)

    @staticmethod
    def invert(condition, actions, **kwargs):
        return SimpleNamespace(value=actions / np.float32(0.1), velocity_evaluations=10)

    @staticmethod
    def reference_actions(condition, noise, **kwargs):
        return noise[0, :, :7] * np.float32(0.1)

    @staticmethod
    def input_transform(value):
        return {"actions": np.pad(value["actions"], ((0, 0), (0, 25)))}

    @staticmethod
    def output_transform(value):
        return {"actions": value["actions"][..., :7]}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from astra_reversal import frs_agent, frs_experiment, frs_guide, frs_noise_policy

    requests, training = [], []

    class SyntheticClient:
        fail_critique = False
        fail_edit = False

        def __init__(self, **kwargs):
            self.records = []

        def propose(self, request):
            requests.append(copy.deepcopy(request))
            self.records.append({"role": request["role"]})
            if (request["role"] == "critique" and self.fail_critique) or (
                request["role"] == "action_edit" and self.fail_edit
            ):
                raise frs_agent.ClientError("synthetic rejection")
            response = {
                key: request[key]
                for key in (
                    "schema_version",
                    "role",
                    "episode_id",
                    "attempt_id",
                    "request_index",
                    "observation_step",
                    "request_id",
                )
            }
            response["response_id"] = f"synthetic-{request['request_index']}"
            if request["role"] == "paper_direction":
                fine = request["observation_step"] > 0
                response.update(
                    fine=fine,
                    coords=[0, 0, 0] if fine else [0, 1, 0],
                    motion_amount="less",
                    justification="Visible test target.",
                )
            elif request["role"] == "action_edit":
                response.update(
                    mode="edit",
                    delta_xyz=[0, 0, 0.25],
                    gripper="keep",
                    apply_steps=10,
                    justification="Visible test obstacle.",
                )
            else:
                source = request["inputs"].get(
                    "rollout", request["inputs"].get("candidate")
                )
                response["evidence"] = [
                    {
                        "attempt_id": source["attempt_id"],
                        "step": 0,
                        "camera": "observation/image",
                        "observation": "Synthetic visible scene.",
                    }
                ]
                if request["role"] == "critique":
                    response.update(
                        failure_assessment="Lift clearance is uncertain.",
                        rules=[
                            {
                                "rule_id": "lift",
                                "trigger": "Object is held.",
                                "action": "Lift.",
                            }
                        ],
                    )
                else:
                    response["verdict"] = "better"
            return frs_agent.parse_proposal(response, request)

    class SyntheticActor:
        def __init__(self, **kwargs):
            self.rounds = 0

        def predict(self, obs, rng, base_noise):
            return {
                "noise": base_noise.copy(),
                "receipt": {"trained_rounds": self.rounds},
            }

        def fit_accepted_rollout(self, rollout_id, samples, **kwargs):
            assert "state0" in rollout_id
            assert all("state0" in row["metadata"]["source_id"] for row in samples)
            training.append((rollout_id, samples, kwargs))
            self.rounds += 1
            return {"status": "updated", "losses": np.array([2.0, 1.0], np.float32)}

        def save(self, path):
            path.mkdir()
            return {"rounds": self.rounds, "sha256": digest(self.rounds)}

    def fake_rollout(env, entry, benchmark, callback, **kwargs):
        obs = observation()
        for step in (0, 10):
            chunk = callback(obs, step)
            assert chunk.shape == (10, 7)
        audit = {"episode_id": entry["episode_id"]}
        if kwargs["expected_reset"] is not None:
            assert kwargs["expected_reset"] == audit
        return {
            "episode_id": entry["episode_id"],
            "actions_executed": 20,
            "success": False,
            "reset_audit": audit,
            "snapshots": [
                {"label": label, "step": step, "observation": obs}
                for label, step in (("first", 0), ("last", 20))
            ],
        }

    spec = ActionSpec("synthetic", 10, 32, 0.05, (-1.0,) * 7, (1.0,) * 7, {})
    monkeypatch.setattr(ActionSpec, "from_environment", lambda *args: spec)
    monkeypatch.setattr(frs_agent, "FRSClient", SyntheticClient)
    monkeypatch.setattr(
        frs_agent, "summarize_calls", lambda rows: {"provider_calls": len(rows)}
    )
    monkeypatch.setattr(frs_noise_policy, "AuxiliaryNoisePolicy", SyntheticActor)
    monkeypatch.setattr(frs_experiment, "run_rollout", fake_rollout)
    monkeypatch.setattr(
        frs_guide,
        "gripper_guide",
        lambda obs, env: (
            obs["observation/image"].copy(),
            {"camera_to_controller_signs": [1, -1, 1]},
        ),
    )
    entries = [
        {
            "initial_state_id": i,
            "task_id": 0,
            "seed": 43,
            "episode_id": f"synthetic:state{i}",
            "instruction": "Lift the object.",
        }
        for i in range(11)
    ]
    obj = FRSTaskExperiment(
        SyntheticPolicy(),
        lambda *args: (
            SimpleNamespace(),
            SimpleNamespace(language="Lift the object."),
            None,
        ),
        SimpleNamespace(suite="synthetic"),
        entries,
        load_protocol(),
        tmp_path / "case",
        development=True,
    )
    return obj, requests, training


def test_full_loop_uses_real_request_contracts_and_separates_evaluation(experiment):
    obj, requests, training = experiment
    report = obj.run()
    assert report["status"] == "complete"
    assert len(report["physical_rollouts"]) == 15
    assert len(report["evaluation"]) == 8
    assert len(training) == 3
    assert report["initial_noise_policy_checkpoint"]["rounds"] == 0
    assert (obj.directory / "noise_policy_initial").is_dir()
    assert {
        row["metadata"]["kind"] for _, samples, _ in training for row in samples
    } == {"frs_edit"}
    assert all(
        "state0" in r["attempt_id"]
        for r in requests
        if r["role"] in ("critique", "judge")
    )
    assert all("success" not in json.dumps(r) for r in requests)
    assert len({r["request_index"] for r in requests}) == len(requests)
    saved = json.loads((obj.directory / "summary.json").read_text())
    assert saved["adaptation"]["critique_frs_learning"]["rounds"][0]["update"][
        "losses"
    ]["shape"] == [2]
    events = [
        json.loads(line)
        for line in (obj.directory / "events.jsonl").read_text().splitlines()
    ]
    generations = [r for r in events if r["kind"] == "generation"]
    assert {r["generation_kind"] for r in generations} >= {
        "frs_edit",
        "native_defer",
        "direct_reference",
    }


def test_failed_critique_never_promotes_or_trains(experiment):
    obj, _, training = experiment
    obj.client.fail_critique = True
    report = obj.run()
    assert training == []
    assert not any(
        r["promoted"] for arm in report["adaptation"].values() for r in arm["rounds"]
    )


def test_failed_edit_falls_back_to_current_repeated_noise(experiment):
    obj, _, training = experiment
    obj.client.fail_edit = True
    obj.run()
    assert len(training) == 3
    assert {r["metadata"]["kind"] for _, samples, _ in training for r in samples} == {
        "native_defer"
    }
    assert all(
        np.array_equal(
            r["executed_noise"][0, :, :7],
            np.broadcast_to(r["executed_noise"][0, 0, :7], (10, 7)),
        )
        for _, samples, _ in training
        for r in samples
    )


def test_rollout_boundary_omits_privileged_outcomes():
    result = {
        "attempt_id": "x",
        "snapshots": [],
        "executed_actions": np.zeros((2, 7)),
        "success": True,
        "reward": 1.0,
        "object_poses": {"x": [0, 0, 0]},
    }
    assert set(visible_rollout(result)) == {
        "attempt_id",
        "snapshots",
        "executed_actions",
    }


def test_success_iterations_preserve_censoring():
    assert first_success([{"success": False}] * 4)["first_success_round"] is None
    assert first_success(
        [{"success": False}, {"success": True}, {"success": False}]
    ) == {
        "first_success_round": 1,
        "censored": False,
        "observed_rounds": 2,
        "success_by_round": [False, True, True],
    }


def test_noise_schedule_is_paired_and_changes_with_state_and_step():
    a = {"seed": 43, "episode_id": "a"}
    assert np.array_equal(
        stream_rng(a, 0, 0).normal(size=20), stream_rng(a, 0, 0).normal(size=20)
    )
    assert not np.array_equal(
        stream_rng(a, 0, 0).normal(size=20), stream_rng(a, 10, 0).normal(size=20)
    )


def test_camera_projection_respects_display_reflection():
    xy = project_policy_pixels([[1, 2, 2], [-1, 2, 2]], np.eye(4), 224)
    np.testing.assert_array_equal(xy, [[222.5, 1], [223.5, 1]])
    with pytest.raises(ValueError, match="behind"):
        project_policy_pixels([[0, 0, -1]], np.eye(4))
