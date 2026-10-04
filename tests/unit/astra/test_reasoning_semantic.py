import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.reasoning_learning import teacher
from astra_reversal.reasoning_learning.rollout import LearningRollout

from .test_reasoning_rollout import SmallPolicy, observation


class SemanticPolicy(SmallPolicy):
    def __init__(self):
        super().__init__()
        self.samples = []

    def prepare(self, observation, oid, prompt):
        return SimpleNamespace(
            state=torch.zeros(1, 8),
            condition_id=prompt,
            displacement=0 if prompt == "original task" else 0.8,
        )

    def prepare_interpolated(
        self, observation, oid, prompt, *, source_prompts, alpha, operator
    ):
        assert prompt == source_prompts[0] == "original task"
        assert source_prompts[1] == "carry bottle over bowl" and operator == "tei"
        return SimpleNamespace(
            state=torch.zeros(1, 8), condition_id=str(alpha), displacement=alpha
        ), {"operator": "tei", "alpha": alpha}

    def sample(self, condition, noise, steps):
        self.samples.append((condition.condition_id, noise.clone()))
        value = noise.clone()
        value[:, :, 0] = condition.displacement
        return SimpleNamespace(value=value)


class SemanticTeacher:
    def __init__(self):
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        if request["role"] == "diagnose":
            response = {
                "intervene": True,
                "plan_complete": False,
                "evidence": "Transport is toward the wrong receptacle",
                "rule": "Carry over the bowl while holding the object",
                "completion": "Bottle above bowl",
                "edits": [],
                "method": "language_subgoal",
                "subgoal_instruction": "carry bottle over bowl",
            }
        elif request["role"] == "compare":
            response = {
                "selected": "subgoal_tei_2",
                "judgments": [
                    {
                        "candidate_id": name,
                        "preference": "win",
                        "evidence": "More useful transport command",
                    }
                    for name in request["context"]["candidates"]
                ],
            }
        else:
            response = {
                "evidence": "Useful actual transport",
                "outcome": "observed_useful",
                "stage": "continuation",
                "plan_complete": False,
            }
        return teacher.parse_proposal(response, request)


def test_semantic_candidates_share_noise_and_only_real_actions_train_original_inputs(
    tmp_path,
):
    policy, client = SemanticPolicy(), SemanticTeacher()
    loop = LearningRollout(
        policy,
        client,
        ActionSpec("test", 10, 32, 0.05, (-1,) * 7, (1,) * 7, {}),
        tmp_path / "trial",
        episode_id="e",
        instruction="original task",
        policy_version=0,
        seed=1,
        semantic_interventions=True,
    )
    for start in (0, 5):
        commands = loop.action(observation(start), start)
        assert (
            len(loop.steps) == start
        )  # Three computational alternatives executed nothing.
        np.testing.assert_allclose(commands[:, 0], 0.67)
        samples = policy.samples[-4:]
        assert len(samples) == 4
        assert all(torch.equal(noise, samples[0][1]) for _, noise in samples)
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
    assert len(windows) == 1 and loop.events == 1 and loop.assisted_chunks == 2
    assert loop.observations[windows[0]["observation_id"]]["prompt"] == "original task"
    np.testing.assert_array_equal(
        windows[0]["actions"], [row["action"] for row in loop.steps]
    )
    decisions = [
        json.loads(line)
        for line in (tmp_path / "trial/decisions.jsonl").read_text().splitlines()
    ]
    for decision in decisions:
        np.testing.assert_array_equal(
            decision["candidates"]["native"], np.zeros((10, 7))
        )
        assert len(decision["generation"]) == 3
        assert len({r["noise_sha256"] for r in decision["generation"]}) == 1
        assert (
            decision["binding"]["rule"]["subgoal_instruction"]
            == "carry bottle over bowl"
        )
    assert all(r["context"]["instruction"] == "original task" for r in client.requests)


def test_semantic_contract_is_version_gated_and_rejects_conflicting_fields():
    from .test_reasoning_teacher import diagnosis, request

    legacy = request()
    assert (
        teacher.build_payload(legacy, "model")["messages"][0]["content"]
        == teacher.SYSTEM_PROMPT
    )
    assert "method" not in teacher.response_schema(legacy)["properties"]
    req = request(semantic_interventions=True)
    value = {
        **diagnosis(),
        "edits": [],
        "method": "language_subgoal",
        "subgoal_instruction": "carry bottle over bowl",
    }
    teacher.parse_proposal(value, req)
    assert (
        teacher.SEMANTIC_EXTENSION
        in teacher.build_payload(req, "model")["messages"][0]["content"]
    )
    with pytest.raises(ValueError):
        teacher.parse_proposal(value, legacy)
    for changes in (
        {"subgoal_instruction": " "},
        {"edits": diagnosis()["edits"]},
        {"intervene": False},
        {"method": "native"},
    ):
        with pytest.raises(ValueError, match="correction fields"):
            teacher.parse_proposal({**value, **changes}, req)
