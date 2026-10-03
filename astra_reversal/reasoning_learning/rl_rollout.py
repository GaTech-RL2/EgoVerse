"""Real transitions for RL, using the same reset/step/video driver as the teacher."""

import copy
import json
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.intervention_rollout import run_rollout
from astra_reversal.reasoning_learning.evidence import append_record
from astra_reversal.records import digest


def discounted_step_cost(successes, gamma=0.999):
    """Released DSRL -1+success step reward, accumulated over the real prefix."""
    if not successes:
        raise ValueError("A transition requires executed steps")
    return sum(gamma**i * (-1 + float(success)) for i, success in enumerate(successes))


class RLRollout:
    def __init__(
        self,
        policy,
        learner,
        spec,
        directory,
        *,
        instruction,
        evaluation,
        seed,
        progress=None,
    ):
        self.policy, self.learner, self.spec = policy, learner, spec
        self.adapter = ActionAdapter(
            spec, policy.input_transform, policy.output_transform
        )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.instruction, self.evaluation = instruction, evaluation
        self.rng = np.random.default_rng(seed)
        self.progress, self.transitions, self.pending = progress, [], None
        self.steps, self.last_observation = 0, None

    def finish_prefix(self, observation, *, end=False):
        if self.pending is None:
            return
        row = self.pending
        if not row["executed_actions"]:
            raise ValueError("Unexecuted RL proposal is not a transition")
        row.update(
            next_observation=copy.deepcopy(observation),
            next_observation_id=digest({**observation, "prompt": self.instruction}),
            terminal=bool(end or row["successes"][-1] or row["terminated"]),
            sac_reward=discounted_step_cost(row["successes"]),
            ppo_reward=float(any(row["successes"])),
        )
        # Keep raw observations on CPU for replay; store the source observation
        # and actual commanded prefix, not a model tail that never executed.
        append_record(
            self.directory / "transitions.jsonl",
            {
                k: v
                for k, v in row.items()
                if k not in ("observation", "next_observation", "noise", "ppo")
            },
        )
        if not self.evaluation:
            self.transitions.append(row)
        self.pending = None
        if self.progress:
            self.progress()

    def action(self, observation, step):
        self.finish_prefix(observation)
        raw = {**copy.deepcopy(observation), "prompt": self.instruction}
        oid = digest(raw)
        if hasattr(self.learner, "proposal"):
            model_actions, extra = self.learner.proposal(
                observation, self.instruction, evaluation=self.evaluation, rng=self.rng
            )
            noise = extra["noise"]
            state = self.policy._preprocess(raw)["observation.state"]
        else:
            condition = self.policy.prepare(observation, oid, self.instruction)
            noise = self.learner.noise(observation, evaluation=self.evaluation)
            model_actions = self.policy.sample(condition, noise, steps=10).value
            state, extra = condition.state, {}
        commands, clipping = self.adapter.decode(model_actions, state)
        np.savez_compressed(
            self.directory / f"observation_{step}.npz",
            **raw,
            noise=noise.detach().cpu().numpy(),
        )
        self.pending = {
            **extra,
            "step": step,
            "observation": copy.deepcopy(observation),
            "observation_id": oid,
            "noise": noise.detach().cpu(),
            "policy_version": self.learner.version,
            "executed_actions": [],
            "successes": [],
            "terminated": False,
            "clipping": clipping,
        }
        return commands

    def observed_step(self, before, action, step, after, success, terminated):
        self.steps += 1
        self.last_observation = copy.deepcopy(after)
        self.pending["executed_actions"].append(action.tolist())
        self.pending["successes"].append(bool(success))
        self.pending["terminated"] = bool(terminated)
        append_record(
            self.directory / "executed_steps.jsonl",
            {
                "step": step,
                "action": action.tolist(),
                "success": bool(success),
                "terminated": bool(terminated),
                "before_sha256": digest(before),
                "after_sha256": digest(after),
            },
        )


def collect_rl(
    policy,
    learner,
    env,
    entry,
    benchmark,
    directory,
    *,
    evaluation,
    seed,
    progress=None,
):
    loop = RLRollout(
        policy,
        learner,
        ActionSpec.from_environment(env, policy.horizon, policy.action_dim),
        directory,
        instruction=entry["instruction"],
        evaluation=evaluation,
        seed=seed,
        progress=progress,
    )
    result = run_rollout(
        env,
        entry,
        benchmark,
        loop.action,
        execute_steps=5,
        action_budget=300,
        policy_image_size=policy.observation_image_size,
        video_path=Path(directory) / "rollout.mp4",
        step_observer=loop.observed_step,
    )
    result.pop("snapshots")
    if loop.steps:
        # The task's finite 300-control-step horizon is terminal for both RL
        # objectives; no value from an unexecuted continuation is substituted.
        loop.finish_prefix(loop.last_observation, end=True)
    result.update(
        autonomous=True,
        teacher_requests=0,
        policy_version=learner.version,
        initialization_steps=benchmark.stabilization_steps,
        total_control_steps=loop.steps + benchmark.stabilization_steps,
        collection_transitions=len(loop.transitions),
    )
    (Path(directory) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, loop.transitions
