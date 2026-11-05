"""Synthetic fixtures: these are not Astra responses or benchmark measurements."""

import json
from dataclasses import replace

import numpy as np
import pytest

from astra_reversal import flow
from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.agent import AstraAgent
from astra_reversal.config import RunConfig
from astra_reversal.controller import Controller
from astra_reversal.policy_adapter import Condition
from astra_reversal.records import MemoryRecorder, digest


class SyntheticPolicy:
    horizon = 10
    action_dim = 32
    metadata = {"artifact": "synthetic-test-only", "normalization": "test-transform-v1"}

    def __init__(self):
        self.conditions = []
        self.sample_inputs = []
        self.inverse_inputs = []

    @staticmethod
    def input_transform(data):
        result = {"state": np.pad(data["observation/state"], (0, 24))}
        if "actions" in data:
            result["actions"] = np.pad((data["actions"] - 0.05) / 2, ((0, 0), (0, 25)))
        return result

    @staticmethod
    def output_transform(data):
        return {"actions": data["actions"][:, :7] * 2 + 0.05}

    @staticmethod
    def tensor(array):
        return np.asarray(array, dtype=np.float32)

    def noise(self, rng):
        return rng.standard_normal((1, self.horizon, self.action_dim)).astype(
            np.float32
        )

    @staticmethod
    def assemble_prompt(original, subgoal, constraints, *, observation=None):
        return original + "; " + subgoal + "; " + "; ".join(constraints), []

    def prepare(self, observation, observation_id, prompt):
        raw = {**observation, "prompt": prompt}
        value = (
            0.05 + observation["observation/image"].mean() / 1000 + len(prompt) / 10000
        )

        def velocity(x, t):
            return np.full_like(x, value)

        condition = Condition(
            digest(raw),
            observation_id,
            prompt,
            raw,
            self.input_transform(raw)["state"][None],
            velocity,
            0.0,
        )
        self.conditions.append(condition)
        return condition

    def sample(self, condition, noise, **kwargs):
        self.sample_inputs.append((condition, noise.copy()))
        return flow.generate(condition.velocity, noise, **kwargs)

    def invert(self, condition, actions, **kwargs):
        self.inverse_inputs.append((condition, actions.copy()))
        return flow.invert(condition.velocity, actions, **kwargs)

    def reference_actions(self, condition, noise, *, steps):
        # Independent analytic solution of the constant field, not another call
        # through the integrator under test.
        return self.output_transform(
            {"actions": (noise - condition.velocity(noise, 0))[0]}
        )["actions"]


class SyntheticEnvironment:
    def __init__(self, stop_after=26, successful=True):
        self.actions = []
        self.stop_after = stop_after
        self.successful = successful
        self.success = False
        self.terminated = False

    def observe(self):
        step = len(self.actions)
        return {
            "observation/image": np.full((16, 16, 3), step % 256, np.uint8),
            "observation/wrist_image": np.full((16, 16, 3), step % 256, np.uint8),
            "observation/state": np.asarray(
                [step / 1000, 0, 0, 0, 0, 0, 0.02, -0.02], np.float32
            ),
        }

    def step(self, action):
        self.actions.append(action.copy())
        if len(self.actions) >= self.stop_after:
            self.success = self.successful
            self.terminated = not self.successful


class SyntheticBackend:
    model_version = "synthetic-test-fixture"
    sampling_settings = {"synthetic": True}

    def __init__(self, mutate=None):
        self.requests = []
        self.mutate = mutate

    def generate(self, request):
        self.requests.append(request)
        response = {
            "schema_version": "1.0",
            "episode_id": request["episode_id"],
            "observation_step": request["observation_step"],
            "plan_id": f"{request['episode_id']}:{len(self.requests)}",
            "subgoal_id": "reach",
            "subgoal_instruction": "reach around the object",
            "completion": {
                "type": "eef_position",
                "parameters": {"target": [10, 0, 0], "tolerance": 0.001},
            },
            "timeout_env_steps": 60,
        }
        if request["stage"] == 1:
            response.update(
                action_spec_id=request["action_spec"]["action_spec_id"],
                action_chunk=np.full(
                    (request["action_spec"]["horizon"], 7), 0.2
                ).tolist(),
            )
        else:
            response.update(
                constraints=["avoid the left side"],
                annotations=[
                    {
                        "camera": "observation/image",
                        "kind": "box",
                        "coordinates": [1, 1, 5, 5],
                        "label": "target",
                    }
                ],
            )
        if self.mutate:
            self.mutate(response, request)
        return json.dumps(response)


@pytest.fixture
def spec():
    return ActionSpec(
        "synthetic-controller-v1",
        10,
        32,
        0.05,
        (-1.0,) * 7,
        (1.0,) * 7,
        {"test_only": True},
    )


@pytest.fixture
def make_controller(spec):
    def make(method="reversal", *, mutate=None, config=None):
        config = config or replace(RunConfig(), method=method)
        if method == "direct_astra":
            config = replace(config, agent=replace(config.agent, refresh_env_steps=10))
        policy, backend, recorder = (
            SyntheticPolicy(),
            SyntheticBackend(mutate),
            MemoryRecorder(),
        )
        actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        agent = AstraAgent(backend, config.agent, recorder)
        controller = Controller(config, policy, actions, recorder, agent)
        return controller, policy, backend, recorder

    return make
