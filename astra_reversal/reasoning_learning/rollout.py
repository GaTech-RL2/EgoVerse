"""A single real trajectory with computational candidate search and honest evidence.

The simulator is paused while reasoning; wall time is recorded. This is not a
real-time controller. Candidate generation never receives the environment.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np

from astra_reversal.action_adapter import ActionAdapter, ActionSpec
from astra_reversal.intervention_rollout import run_rollout
from astra_reversal.lerobot_policy import prepare_velocity
from astra_reversal.records import digest, to_numpy

from . import teacher
from .evidence import CandidateBatch, append_record, training_windows
from .guidance import GuidanceConfig, generate
from .semantic import candidates as semantic_candidates


class LearningRollout:
    def __init__(
        self,
        policy,
        client,
        spec,
        directory,
        *,
        episode_id,
        instruction,
        policy_version,
        seed,
        strengths=(0.25, 1.0),
        candidate_configurations=None,
        collection_history=None,
        temporal_diagnosis=False,
        semantic_interventions=False,
        explicit_execution_prefix=False,
        text_latent_candidates=False,
        comparison_feedback=False,
        bounded_target_candidate=False,
        controller_delta_limits=None,
        retain_step_observations=True,
        max_episodes=3,
        monitor_every=25,
        max_active_actions=50,
        progress=None,
    ):
        self.policy, self.client, self.spec = policy, client, spec
        self.adapter = ActionAdapter(
            spec, policy.input_transform, policy.output_transform
        )
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.episode_id, self.instruction, self.version = (
            episode_id,
            instruction,
            policy_version,
        )
        self.rng = np.random.default_rng(seed)
        self.strengths, self.max_episodes = tuple(strengths), max_episodes
        self.candidate_configurations = (
            [GuidanceConfig(**row) for row in candidate_configurations]
            if candidate_configurations is not None
            else [GuidanceConfig(strength=value) for value in self.strengths]
        )
        self.collection_history = copy.deepcopy(collection_history or [])
        self.temporal_diagnosis = temporal_diagnosis
        self.semantic_interventions = semantic_interventions
        self.explicit_execution_prefix = explicit_execution_prefix
        self.text_latent_candidates = text_latent_candidates
        self.comparison_feedback = comparison_feedback
        self.bounded_target_candidate = bounded_target_candidate
        if bounded_target_candidate and not explicit_execution_prefix:
            raise ValueError(
                "Bounded target candidates require the five-action edit contract"
            )
        self.comparison_history = []
        self.delta_limits = controller_delta_limits
        self.retain_step_observations = retain_step_observations
        if client is not None and not retain_step_observations:
            raise ValueError("Collection must retain all real pre-action observations")
        teacher.controller_delta_limits(controller_delta_limits)
        self.reviews = []
        self.previous_monitor = None
        self.monitor_every, self.max_active_actions = monitor_every, max_active_actions
        self.steps, self.observations, self.decisions = [], {}, []
        self.request_index = 0
        self.active = None
        self.events, self.assisted_chunks, self.assisted_seconds = 0, 0, 0.0
        self.pending = None
        self.last_observation = None
        self.stopped_reason = None
        self.progress = progress

    def _ask(self, role, step, snapshots, context):
        request = teacher.build_request(
            role=role,
            episode_id=self.episode_id,
            request_index=self.request_index,
            step=step,
            snapshots=snapshots,
            context={
                "instruction": self.instruction,
                "action_spec": self.spec.as_dict(),
                "policy_version": self.version,
                **(
                    {"controller_delta_limits": self.delta_limits}
                    if self.delta_limits is not None
                    else {}
                ),
                **(
                    {"semantic_interventions": True}
                    if self.semantic_interventions
                    else {}
                ),
                **(
                    {"execution_prefix_steps": 5}
                    if self.explicit_execution_prefix
                    else {}
                ),
                **(
                    {"text_latent_candidates": True}
                    if self.text_latent_candidates
                    else {}
                ),
                **({"comparison_feedback": True} if self.comparison_feedback else {}),
                **(
                    {"bounded_target_candidate": True}
                    if self.bounded_target_candidate
                    else {}
                ),
                **context,
            },
        )
        self.request_index += 1
        append_record(self.directory / "teacher_requests.jsonl", request)
        if self.progress is not None:
            self.progress()
        response = self.client.propose(request)
        append_record(self.directory / "teacher_responses.jsonl", response)
        return response

    def _finish_prefix(self, observation, step):
        if self.pending is None or self.client is None:
            return
        old = self.pending
        rows = self.steps[old["start_step"] : step]
        if not rows:
            raise ValueError("Outcome review requires executed commands")
        response = self._ask(
            "assess",
            step,
            [
                {
                    "step": old["start_step"],
                    "observation": old["observation"],
                    "label": "Before REAL execution",
                },
                {
                    "step": step,
                    "observation": observation,
                    "label": "After REAL execution",
                },
            ],
            {
                "active_rule": old["rule"],
                "selected": old["selected"],
                "executed_commands": [row["action"] for row in rows],
                "outcome_scope": "executed prefix only; unexecuted tail is unverified",
                "environment_success": rows[-1]["environment_success"],
            },
        )
        for row in rows:
            row["evidence"] = response["outcome"]
            # A guided prefix cannot relabel itself as ungated setup data.
            row["stage"] = (
                "correction" if old["selected"] != "native" else response["stage"]
            )
            row["outcome_review_id"] = response["decision_id"]
        append_record(
            self.directory / "outcomes.jsonl",
            {"start_step": old["start_step"], "end_step": step, "review": response},
        )
        self.reviews.append(
            {
                "start_step": old["start_step"],
                "end_step": step,
                "selected": old["selected"],
                "rule": old["rule"],
                **response,
            }
        )
        if response["plan_complete"] and self.active is not None:
            append_record(
                self.directory / "events.jsonl",
                {
                    **self.active,
                    "status": "completed",
                    "end_step": step,
                    "end_time": time.time(),
                    "evidence": response["evidence"],
                },
            )
            self.active = None
        self.pending = None

    def action(self, observation, step):
        if any(p.requires_grad for p in self.policy.policy.parameters()):
            raise ValueError("Policy weights must remain fixed throughout collection")
        self.last_observation = copy.deepcopy(observation)
        self._finish_prefix(observation, step)
        if (
            self.active is not None
            and step - self.active["start_step"] >= self.max_active_actions
        ):
            append_record(
                self.directory / "events.jsonl",
                {
                    **self.active,
                    "status": "expired",
                    "end_step": step,
                    "end_time": time.time(),
                },
            )
            self.active = None
        raw = {**copy.deepcopy(observation), "prompt": self.instruction}
        oid = digest(raw)
        condition = self.policy.prepare(observation, oid, self.instruction)
        noise = self.policy.noise(self.rng)
        native = self.policy.sample(condition, noise, steps=10).value
        reference, native_clip = self.adapter.decode(native, condition.state)
        rule = {"kind": "native", "episode_id": self.episode_id, "step": step}
        diagnosis = None
        if self.client is not None and (
            self.active is not None or step % self.monitor_every == 0
        ):
            snapshots = [{"step": step, "observation": observation}]
            history_context = {}
            if self.temporal_diagnosis:
                if self.previous_monitor is not None:
                    snapshots.insert(0, self.previous_monitor)
                snapshots[-1]["label"] = "CURRENT pre-action observation"
                history_context = {
                    "recent_observed_evidence": self.reviews[-5:],
                    "previous_collection_attempts": self.collection_history[-3:],
                    "history_scope": "Earlier real collection only; different resets. No autonomous evaluation feedback is available.",
                }
            diagnosis = self._ask(
                "diagnose",
                step,
                snapshots,
                {
                    "native": reference.tolist(),
                    "active_plan": self.active,
                    "recent_outcomes": [
                        {k: row[k] for k in ("step", "evidence", "stage")}
                        for row in self.steps[-10:]
                    ],
                    "correction_episodes_used": self.events,
                    "soft_episode_budget": self.max_episodes,
                    **(
                        {"recent_candidate_comparisons": self.comparison_history[-3:]}
                        if self.comparison_feedback
                        else {}
                    ),
                    **history_context,
                },
            )
            self.previous_monitor = {
                "step": step,
                "observation": copy.deepcopy(observation),
                "label": "EARLIER real observation in this attempt; actions since then actually executed",
            }
            if diagnosis["plan_complete"] and self.active is not None:
                append_record(
                    self.directory / "events.jsonl",
                    {
                        **self.active,
                        "status": "completed_by_monitor",
                        "end_step": step,
                        "end_time": time.time(),
                    },
                )
                self.active = None
            if diagnosis["intervene"]:
                if self.active is None:
                    # Budget exhaustion ends the attempt rather than forcing a
                    # knowingly unproductive unassisted continuation.
                    if self.events >= self.max_episodes:
                        self.stopped_reason = "correction_episode_budget_exhausted"
                        raise CollectionBudgetStop(self.stopped_reason)
                    self.active = {
                        "event_id": f"{self.episode_id}:plan{self.request_index}",
                        "start_step": step,
                        "start_time": time.time(),
                        "intervention_started": False,
                    }
                rule = {
                    "kind": diagnosis.get("method", "target_guidance"),
                    "rule": diagnosis["rule"],
                    "completion": diagnosis["completion"],
                    "request_id": diagnosis["decision_id"],
                    **(
                        {"subgoal_instruction": diagnosis["subgoal_instruction"]}
                        if diagnosis.get("method") == "language_subgoal"
                        else {}
                    ),
                }
                self.active.update(rule=rule)
                append_record(
                    self.directory / "events.jsonl",
                    {
                        **self.active,
                        "status": "rule_version",
                        "observation_id": oid,
                        "policy_version": self.version,
                    },
                )
        batch = CandidateBatch(
            observation_id=oid,
            policy_version=self.version,
            rule=rule,
            reference=reference,
        )
        generation = []
        if rule["kind"] == "language_subgoal":
            try:
                proposals = semantic_candidates(
                    self.policy,
                    self.adapter,
                    observation,
                    oid,
                    self.instruction,
                    rule["subgoal_instruction"],
                    noise,
                    include_tli=self.text_latent_candidates,
                )
                self.active.pop("last_target_error", None)
                for name, commands, receipt in proposals:
                    batch.add(name, commands)
                    generation.append(receipt)
            except ValueError as exc:
                self.active["last_target_error"] = str(exc)
                append_record(
                    self.directory / "rejected_targets.jsonl",
                    {
                        "batch_id": batch.batch_id,
                        "reason": str(exc),
                        "diagnosis": diagnosis,
                    },
                )
        if rule["kind"] == "target_guidance":
            try:
                target, mask = teacher.controller_target(
                    reference,
                    diagnosis["edits"],
                    self.spec,
                    delta_limits=self.delta_limits,
                )
                self.active.pop("last_target_error", None)
                encoded = self.adapter.encode(target, raw)
                if self.bounded_target_candidate:
                    # The existing validator checks cumulative edit limits and
                    # hardware bounds without clipping. This alternative is the
                    # exact controller target, not a sample from the policy.
                    batch.add("bounded_target", target)
                    generation.append(
                        {
                            "candidate_id": "bounded_target",
                            "source": "astra_bounded_controller_target",
                            "target_sha256": digest(target),
                            "mask_sha256": digest(mask),
                            "policy_generated": False,
                            "unexecuted": True,
                            "clipping": {"count": 0, "max_abs": 0.0},
                        }
                    )
                model_mask = np.zeros_like(encoded)
                model_mask[0, :, :7] = mask
                velocity = prepare_velocity(
                    self.policy.policy,
                    self.policy._preprocess(raw),
                    differentiable=True,
                )
                for i, configuration in enumerate(self.candidate_configurations):
                    candidate, receipt = generate(
                        velocity,
                        noise,
                        self.policy.tensor(encoded),
                        self.policy.tensor(model_mask),
                        configuration,
                    )
                    commands, clipping = self.adapter.decode(candidate, condition.state)
                    batch.add(f"guided{i}", commands)
                    generation.append(
                        {
                            "candidate_id": f"guided{i}",
                            "receipt": receipt,
                            "clipping": clipping,
                        }
                    )
                append_record(
                    self.directory / "targets.jsonl",
                    {
                        "batch_id": batch.batch_id,
                        "controller_target": target.tolist(),
                        "controller_mask": mask.tolist(),
                        "encoded_target_sha256": digest(encoded),
                        "noise_sha256": digest(to_numpy(noise)),
                    },
                )
            except ValueError as exc:
                self.active["last_target_error"] = str(exc)
                append_record(
                    self.directory / "rejected_targets.jsonl",
                    {
                        "batch_id": batch.batch_id,
                        "reason": str(exc),
                        "diagnosis": diagnosis,
                    },
                )
        selected = "native"
        candidates = {
            key: value.tolist()
            for key, value in batch.proposals().items()
            if key != "native"
        }
        if candidates:
            comparison = self._ask(
                "compare",
                step,
                [{"step": step, "observation": observation}],
                {
                    "binding": batch.binding,
                    "native": reference.tolist(),
                    "candidates": candidates,
                },
            )
            for row in comparison["judgments"]:
                batch.judge(
                    row["candidate_id"],
                    binding=batch.binding,
                    preference=row["preference"],
                    rationale=row["evidence"],
                )
            selected = comparison["selected"]
            if self.comparison_feedback:
                self.comparison_history.append(
                    {
                        "step": step,
                        "rule": copy.deepcopy(rule),
                        "selected": selected,
                        "judgments": copy.deepcopy(comparison["judgments"]),
                        "scope": "Earlier computational preferences, not physical outcomes of rejected candidates; the next batch uses its own fixed reference and rule.",
                    }
                )
        commands, claim = batch.claim(selected, self.directory / "execution_claims")
        event_id = self.active["event_id"] if self.active else None
        self.pending = {
            "start_step": step,
            "observation": copy.deepcopy(observation),
            "selected": selected,
            "rule": rule,
            "event_id": event_id,
            "batch_id": batch.batch_id,
        }
        self.decisions.append(
            {"step": step, "selected": selected, "batch_id": batch.batch_id}
        )
        append_record(
            self.directory / "decisions.jsonl",
            {
                **claim,
                "step": step,
                "generation": generation,
                "native_clipping": native_clip,
                "candidates": {k: v.tolist() for k, v in batch.proposals().items()},
            },
        )
        if selected != "native":
            if not self.active["intervention_started"]:
                self.events += 1
                self.active["intervention_started"] = True
                append_record(
                    self.directory / "events.jsonl",
                    {**self.active, "status": "first_assisted_chunk", "step": step},
                )
            self.assisted_chunks += 1
        return commands

    def observed_step(self, before, action, step, after, success, terminated):
        raw = {**copy.deepcopy(before), "prompt": self.instruction}
        oid = digest(raw)
        if self.retain_step_observations:
            self.observations[oid] = raw
        if self.retain_step_observations or step == 0:
            np.savez_compressed(self.directory / (oid + ".npz"), **raw)
        pending = self.pending
        row = {
            "episode_id": self.episode_id,
            "step": step,
            "observation_id": oid,
            "action": action.tolist(),
            "executed": True,
            "policy_version": self.version,
            "batch_id": pending["batch_id"],
            "event_id": pending["event_id"],
            "preference": "win" if pending["selected"] != "native" else None,
            "stage": "correction"
            if pending["selected"] != "native"
            else "continuation",
            "evidence": "ambiguous",
            "environment_success": success,
            "terminated": terminated,
            "after_observation_sha256": digest(after),
        }
        self.steps.append(row)
        self.last_observation = copy.deepcopy(after)
        if pending["selected"] != "native":
            self.assisted_seconds += self.spec.timestep_seconds
        append_record(self.directory / "executed_steps.jsonl", row)
        if self.progress is not None and (step + 1) % 5 == 0:
            self.progress()

    def finalize(self):
        if self.steps:
            self._finish_prefix(self.last_observation, len(self.steps))
        windows = training_windows(self.steps, horizon=self.policy.horizon)
        append_record(
            self.directory / "admission.jsonl",
            {"steps": self.steps, "windows": windows},
        )
        return windows


class CollectionBudgetStop(RuntimeError):
    """Expected early stop; executed steps are retained and count toward budget."""


def collect(
    policy, client, env, entry, benchmark, directory, *, version, seed, **options
):
    spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
    loop = LearningRollout(
        policy,
        client,
        spec,
        directory,
        episode_id=entry["episode_id"],
        instruction=entry["instruction"],
        policy_version=version,
        seed=seed,
        **options,
    )
    clock = time.perf_counter()
    try:
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
    except CollectionBudgetStop:
        result = {
            "success": False,
            "actions_executed": len(loop.steps),
            "early_stop": loop.stopped_reason,
            "episode_id": entry["episode_id"],
            "wall_seconds": time.perf_counter() - clock,
        }
    windows = loop.finalize() if client is not None else []
    result.update(
        correction_episodes=loop.events,
        assisted_chunks=loop.assisted_chunks,
        assisted_simulator_seconds=loop.assisted_seconds,
        teacher_requests=loop.request_index,
        admitted_windows=len(windows),
        autonomous=client is None,
        initialization_steps=benchmark.stabilization_steps,
        total_control_steps=len(loop.steps) + benchmark.stabilization_steps,
    )
    if client is not None:
        result["collection_summary"] = {
            "episode_id": loop.episode_id,
            "success": bool(result["success"]),
            "actions_executed": len(loop.steps),
            "correction_episodes": loop.events,
            "assisted_chunks": loop.assisted_chunks,
            "correction_reviews": [
                row for row in loop.reviews if row["selected"] != "native"
            ][-4:],
            "final_observed_reviews": loop.reviews[-3:],
            "source": "Real collection trajectory; no autonomous evaluation results",
        }
    (Path(directory) / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, windows, loop.observations
