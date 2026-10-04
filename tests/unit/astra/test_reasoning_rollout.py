from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.reasoning_learning.rollout import LearningRollout


class SmallPolicy:
    horizon, action_dim = 10, 32

    def __init__(self):
        self.policy = torch.nn.Linear(1, 1).requires_grad_(False)

    def prepare(self, observation, oid, prompt):
        return SimpleNamespace(state=torch.zeros(1, 8))

    def noise(self, rng):
        return torch.zeros(1, 10, 32)

    def sample(self, condition, noise, steps):
        return SimpleNamespace(value=noise.clone())

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32)

    def _preprocess(self, raw):
        return raw

    def input_transform(self, raw):
        return {"actions": np.pad(raw["actions"], ((0, 0), (0, 25)))}

    def output_transform(self, data):
        return {"actions": data["actions"][:, :7]}


class Teacher:
    def __init__(self, preference="win"):
        self.roles, self.requests = [], []
        self.preference = preference

    def propose(self, request):
        from astra_reversal.reasoning_learning.teacher import parse_proposal

        self.roles.append(request["role"])
        self.requests.append(request)
        if request["role"] == "diagnose":
            response = {
                "intervene": True,
                "plan_complete": False,
                "evidence": "More lift needed",
                "rule": "Lift while preserving grasp",
                "completion": "Clear rim",
                "edits": [{"start": 0, "end": 5, "channel": 2, "delta": 0.2}],
            }
        elif request["role"] == "compare":
            response = {
                "selected": "guided0" if self.preference == "win" else "native",
                "judgments": [
                    {
                        "candidate_id": c,
                        "preference": self.preference,
                        "evidence": "Compared z commands",
                    }
                    for c in request["context"]["candidates"]
                ],
            }
        else:
            response = {
                "outcome": "observed_useful",
                "stage": "continuation",
                "plan_complete": False,
                "evidence": "Actual upward motion observed",
            }
        return parse_proposal(response, request)


def observation(step):
    return {
        "observation/state": np.array(
            [0, 0, step * 0.01, 0, 0, 0, 0.02, -0.02], dtype=np.float32
        ),
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8) + step,
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8) + step,
    }


def test_search_never_executes_candidates_and_only_real_prefixes_train(
    tmp_path, monkeypatch
):
    from astra_reversal.reasoning_learning import rollout

    monkeypatch.setattr(rollout, "prepare_velocity", lambda *a, **k: lambda x, t: x * 0)
    teacher = Teacher()
    loop = LearningRollout(
        SmallPolicy(),
        teacher,
        ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        tmp_path / "trial",
        episode_id="e",
        instruction="lift object",
        policy_version=0,
        seed=1,
    )
    for start in (0, 5):
        commands = loop.action(observation(start), start)
        # Generating/judging two candidates has produced NO physical samples.
        assert len(loop.steps) == start
        for j in range(5):
            loop.observed_step(
                observation(start + j),
                commands[j],
                start + j,
                observation(start + j + 1),
                False,
                False,
            )
    windows = loop.finalize()
    assert len(loop.steps) == 10 and len(windows) == 1
    assert loop.events == 1 and loop.assisted_chunks == 2
    assert loop.assisted_seconds == pytest.approx(0.5)
    assert windows[0]["observation_id"] == loop.steps[0]["observation_id"]
    np.testing.assert_array_equal(
        windows[0]["actions"], [r["action"] for r in loop.steps]
    )
    assert teacher.roles == [
        "diagnose",
        "compare",
        "assess",
        "diagnose",
        "compare",
        "assess",
    ]


def test_rejected_proposals_do_not_count_as_intervention_episodes(
    tmp_path, monkeypatch
):
    from astra_reversal.reasoning_learning import rollout

    monkeypatch.setattr(rollout, "prepare_velocity", lambda *a, **k: lambda x, t: x * 0)
    loop = LearningRollout(
        SmallPolicy(),
        Teacher("uncertain"),
        ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        tmp_path / "trial",
        episode_id="e",
        instruction="lift",
        policy_version=0,
        seed=1,
    )
    command = loop.action(observation(0), 0)
    assert not command.any()
    assert loop.events == loop.assisted_chunks == 0


def test_revised_candidates_receive_temporal_collection_evidence(tmp_path, monkeypatch):
    import json

    from astra_reversal.reasoning_learning import rollout

    monkeypatch.setattr(rollout, "prepare_velocity", lambda *a, **k: lambda x, t: x * 0)
    teacher = Teacher()
    prior = [
        {"episode_id": "earlier_reset", "success": False, "evidence": "Missed grasp"}
    ]
    loop = LearningRollout(
        SmallPolicy(),
        teacher,
        ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        tmp_path / "trial",
        episode_id="e",
        instruction="lift object",
        policy_version=0,
        seed=1,
        collection_history=prior,
        temporal_diagnosis=True,
        candidate_configurations=[
            {"strength": 5, "schedule": "rtc_pigdm", "project_gradient": True},
            {"strength": 10, "schedule": "rtc_pigdm", "project_gradient": False},
        ],
    )
    commands = loop.action(observation(0), 0)
    for j in range(5):
        loop.observed_step(
            observation(j), commands[j], j, observation(j + 1), False, False
        )
    loop.action(observation(5), 5)
    diagnosis = [r for r in teacher.requests if r["role"] == "diagnose"][-1]
    assert [r["step"] for r in diagnosis["snapshots"]] == [0, 5]
    assert diagnosis["context"]["previous_collection_attempts"] == prior
    assert (
        diagnosis["context"]["recent_observed_evidence"][0]["evidence"]
        == "Actual upward motion observed"
    )
    decision = json.loads(
        (tmp_path / "trial/decisions.jsonl").read_text().splitlines()[0]
    )
    receipts = [r["receipt"] for r in decision["generation"]]
    assert [r["strength"] for r in receipts] == [5, 10]
    assert all(r["schedule"] == "rtc_pigdm" for r in receipts)
    assert [r["project_gradient"] for r in receipts] == [True, False]
    assert len(loop.steps) == 5  # Replanning/search added no real samples.


def test_autonomous_evaluation_can_omit_redundant_per_step_image_files(tmp_path):
    loop = LearningRollout(
        SmallPolicy(),
        None,
        ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        tmp_path / "evaluation",
        episode_id="e",
        instruction="lift object",
        policy_version=0,
        seed=1,
        retain_step_observations=False,
    )
    commands = loop.action(observation(0), 0)
    for j in range(5):
        loop.observed_step(
            observation(j), commands[j], j, observation(j + 1), False, False
        )
    assert len(loop.steps) == 5 and not loop.observations
    assert len(list((tmp_path / "evaluation").glob("*.npz"))) == 1
