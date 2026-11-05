"""Synchronous two-rate execution with bounded reuse and explicit fallbacks."""

import copy
import subprocess
import time
from dataclasses import dataclass

import numpy as np

from .augmentation import augment
from .config import AGENT_FREE_METHODS
from .flow import mix_noise, noise_statistics
from .records import ControlStep, InversionResult, digest, to_numpy


class ProgressMonitor:
    def __init__(self, positive_checks=2, assessor=None, assessor_period=20):
        self.positive_checks, self.assessor = positive_checks, assessor
        self.assessor_period = assessor_period
        self.last_assessment_step = None
        self.streak = 0
        self.key = None

    @property
    def supported(self):
        return ("eef_position", "gripper_width") + (
            ("image_progress",) if self.assessor else ()
        )

    def reset(self):
        self.streak, self.key = 0, None
        self.last_assessment_step = None

    def check(self, observation, proposal, *, step=None):
        if proposal is None:
            self.reset()
            return {"complete": False, "positive": False, "streak": 0}
        key = digest(
            {"subgoal": proposal.subgoal_id, "completion": proposal.completion}
        )
        if key != self.key:
            self.streak, self.key = 0, key
            self.last_assessment_step = None
        kind, params = proposal.completion["type"], proposal.completion["parameters"]
        state = observation["observation/state"]
        extra = {}
        if kind == "eef_position":
            positive = (
                np.linalg.norm(state[:3] - params["target"]) <= params["tolerance"]
            )
        elif kind == "gripper_width":
            width = abs(state[6] - state[7])
            positive = abs(width - params["target"]) <= params["tolerance"]
        elif kind == "image_progress" and self.assessor:
            if (
                step is not None
                and self.last_assessment_step is not None
                and step - self.last_assessment_step < self.assessor_period
            ):
                # Cached positive evidence is not a second independent check.
                return {
                    "positive": None,
                    "streak": self.streak,
                    "complete": self.streak >= self.positive_checks,
                    "assessor_deferred": True,
                }
            self.last_assessment_step = step
            began = time.perf_counter()
            try:
                assessment = self.assessor(observation, params["criterion"])
                if (
                    not isinstance(assessment, dict)
                    or type(assessment.get("positive")) is not bool
                ):
                    raise ValueError(
                        "Progress assessor must return a boolean positive field"
                    )
                positive = assessment["positive"]
                if type(assessment.get("stalled", False)) is not bool:
                    raise ValueError("Progress assessor stalled field must be boolean")
                extra["assessment"] = assessment
                extra["stalled"] = assessment.get("stalled", False)
            except (
                ValueError,
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                positive = False
                extra["assessor_error"] = str(exc)
            extra.update(
                assessor_calls=1, assessor_latency_seconds=time.perf_counter() - began
            )
        else:
            raise ValueError("Unimplemented completion checker")
        self.streak = self.streak + 1 if positive else 0
        return {
            "positive": bool(positive),
            "streak": self.streak,
            "complete": self.streak >= self.positive_checks,
            **extra,
        }


@dataclass
class Subgoal:
    began: int
    deadline: int
    recoveries: int = 0
    complete: bool = False
    exhausted: bool = False


@dataclass
class Plan:
    proposal: object | None
    created: int
    observation: dict
    latent: object | None
    inverse_condition: object | None


class Controller:
    def __init__(self, config, policy, actions, recorder, agent=None, assessor=None):
        config.validate(policy.horizon)
        if config.method not in AGENT_FREE_METHODS and agent is None:
            raise ValueError(f"{config.method} requires an Astra backend")
        self.config, self.policy, self.actions = config, policy, actions
        self.recorder, self.agent = recorder, agent
        self.monitor = ProgressMonitor(
            config.controller.completion_positive_checks,
            assessor,
            config.agent.refresh_env_steps,
        )

    def _noise(self, step, stream=0):
        # Pair noise by episode and observation step, independent of extra solves,
        # agent retries, or stochastic choices made by another method.
        return self.policy.noise(
            np.random.default_rng(np.random.SeedSequence([self.seed, step, stream]))
        )

    def _condition(self, observation, prompt):
        condition = self.policy.prepare(observation, digest(observation), prompt)
        self.recorder.event(
            "condition",
            episode_id=self.episode_id,
            observation_step=self.step,
            condition_id=condition.condition_id,
            observation_id=condition.observation_id,
            prompt=prompt,
            observation=observation,
            preparation_seconds=condition.preparation_seconds,
        )
        return condition

    def _solve(self, condition, endpoint, *, inverse=False, role="generation"):
        flow = self.config.flow
        solve = self.policy.invert if inverse else self.policy.sample
        result = solve(
            condition,
            endpoint,
            steps=flow.inversion_steps if inverse else flow.generation_steps,
            solver=flow.integrator,
            save_trace=self.config.evaluation.save_flow_traces,
            **flow.solver_options,
        )
        self.velocity_evaluations += result.velocity_evaluations
        self.recorder.event(
            "flow",
            episode_id=self.episode_id,
            observation_step=self.step,
            role=role,
            condition_id=condition.condition_id,
            grid=result.grid,
            input=to_numpy(endpoint),
            output=to_numpy(result.value),
            states=result.states,
            velocity_evaluations=result.velocity_evaluations,
            latency_seconds=result.latency_seconds,
        )
        return result

    def _augmented_condition(self, observation, proposal, source):
        method = self.config.method
        visual = method != "language_only"
        augmented, events = augment(observation, proposal, source, visuals=visual)
        prompt, omitted = self.instruction, []
        if method != "observation_only":
            prompt, omitted = self.policy.assemble_prompt(
                self.instruction,
                proposal.subgoal_instruction,
                proposal.constraints,
                observation=augmented,
            )
        self.recorder.event(
            "augmentation",
            episode_id=self.episode_id,
            observation_step=self.step,
            plan_id=proposal.plan_id,
            events=events,
            omitted_prompt_parts=omitted,
            raw_observation_id=digest(observation),
            augmented_observation_id=digest(augmented),
            original_prompt=self.instruction,
            augmented_prompt=prompt,
        )
        return self._condition(augmented, prompt)

    def _validate_new_plan(self, proposal):
        if proposal.plan_id in self.seen_plans:
            raise ValueError("A previously accepted plan_id cannot be reused")
        subgoal = self.subgoals.get(proposal.subgoal_id)
        if subgoal and (subgoal.complete or subgoal.exhausted):
            raise ValueError(
                "Subgoal is already completed or has exhausted recovery replans"
            )

    def _activate(self, observation, proposal):
        method, step = self.config.method, self.step
        if proposal is not None:
            self.seen_plans.add(proposal.plan_id)
            if proposal.subgoal_id not in self.subgoals:
                self.subgoals[proposal.subgoal_id] = Subgoal(
                    step, step + proposal.timeout_env_steps
                )
            self.current_subgoal = proposal
        latent, inverse, reference = None, None, None
        plan_id = proposal.plan_id if proposal else f"{self.episode_id}:noise:{step}"
        if method in ("policy_reused", "random_latent"):
            latent = self._noise(step)
        elif method in ("reversal", "same_condition"):
            inverse = self._condition(observation, self.instruction)
            model_actions = self.actions.encode(
                proposal.action_chunk, {**observation, "prompt": self.instruction}
            )
            result = self._solve(
                inverse,
                self.policy.tensor(model_actions),
                inverse=True,
                role="astra_inversion",
            )
            latent = result.value
            self._record_inversion(
                plan_id,
                proposal.action_chunk,
                model_actions,
                inverse,
                result,
                reference,
            )
        elif self.config.stage == 2 and method != "augmentation_only":
            raw = self._condition(observation, self.instruction)
            known_noise = self._noise(step)
            # The reference retains all model channels, without decoding or renormalizing.
            sampled = self._solve(raw, known_noise, role="reference_generation")
            reference = {
                "source": "frozen_pi05_raw_conditioning",
                "action_space": "normalized_model_tensor",
                "source_observation_id": raw.observation_id,
                "source_prompt": self.instruction,
                "source_condition_id": raw.condition_id,
                "observation_step": step,
                "known_noise": to_numpy(known_noise),
                "model_actions": to_numpy(sampled.value),
                "noise_key": [self.seed, step, 0],
                "checkpoint": self.policy.metadata,
            }
            self.recorder.event(
                "reference", episode_id=self.episode_id, reference=reference
            )
            if method == "known_noise_transfer":
                latent = known_noise
            else:
                inverse = (
                    raw
                    if method in ("inversion_only", "conditioning_transfer")
                    else self._augmented_condition(observation, proposal, observation)
                )
                result = self._solve(
                    inverse, sampled.value, inverse=True, role="reference_inversion"
                )
                latent = result.value
                self._record_inversion(
                    plan_id, None, to_numpy(sampled.value), inverse, result, reference
                )
        if latent is not None and self.config.flow.noise_mix_rho:
            latent = mix_noise(
                latent, self._noise(step, 2), self.config.flow.noise_mix_rho
            )
        self.recorder.event(
            "plan_activated",
            episode_id=self.episode_id,
            observation_step=step,
            plan_id=plan_id,
            latent_statistics=noise_statistics(latent) if latent is not None else None,
        )
        return Plan(proposal, step, copy.deepcopy(observation), latent, inverse)

    def _record_inversion(
        self, plan_id, controller_actions, model_actions, condition, result, reference
    ):
        record = InversionResult(
            plan_id,
            controller_actions,
            model_actions,
            condition.condition_id,
            to_numpy(result.value),
            self.config.flow.integrator,
            result.grid,
            self.policy.metadata,
            result.velocity_evaluations,
            result.latency_seconds,
            reference,
        )
        self.recorder.event("inversion", episode_id=self.episode_id, result=record)

    def _decode(self, condition, noise, role="generation"):
        result = self._solve(condition, noise, role=role)
        actions, clipping = self.actions.decode(
            result.value, condition.state, bounds=self.config.controller.output_bounds
        )
        self.recorder.event(
            "generated_actions",
            episode_id=self.episode_id,
            observation_step=self.step,
            condition_id=condition.condition_id,
            actions=actions,
            clipping=clipping,
        )
        return actions, condition

    def _fresh(self, observation, *, fallback=False):
        condition = self._condition(observation, self.instruction)
        return self._decode(
            condition,
            self._noise(self.step, 1 if fallback else 0),
            "fallback" if fallback else "generation",
        )

    def _generate(self, observation, plan):
        method = self.config.method
        if method == "direct_astra":
            offset = self.step - plan.created
            if not 0 <= offset < self.policy.horizon:
                raise ValueError("Direct Astra proposal window expired")
            return plan.proposal.action_chunk[offset:].copy(), None
        if method == "compute_matched":
            condition = self._condition(observation, self.instruction)
            candidate_count = (
                self.config.controller.compute_matched_candidates
                if self.step == plan.created
                else 1
            )
            candidates = [
                self._decode(
                    condition, self._noise(self.step, i), "compute_matched_candidate"
                )[0]
                for i in range(candidate_count)
            ]
            # Predeclared non-oracle rule: smallest change from the last command.
            previous = np.zeros(7) if not self.history else self.history[-1]["action"]
            scores = [float(np.linalg.norm(x[0] - previous)) for x in candidates]
            selected = int(np.argmin(scores))
            self.recorder.event(
                "candidate_selection",
                episode_id=self.episode_id,
                scores=scores,
                selected=selected,
                candidate_count=candidate_count,
                rule="min_l2_first_action_to_previous_command",
            )
            return candidates[selected], condition
        if method == "policy_fresh":
            return self._fresh(observation)
        if method == "subgoal":
            condition = self._condition(observation, plan.proposal.subgoal_instruction)
            return self._decode(condition, self._noise(self.step))
        if method == "same_condition":
            condition = plan.inverse_condition
        elif self.config.stage == 2 and method != "inversion_only":
            condition = self._augmented_condition(
                observation, plan.proposal, plan.observation
            )
        else:
            condition = self._condition(observation, self.instruction)
        noise = self._noise(self.step) if method == "augmentation_only" else plan.latent
        if noise is None:
            raise ValueError("No valid latent available")
        return self._decode(condition, noise)

    def run_episode(self, env, *, episode_id, instruction, seed, metadata=None):
        """Environment exposes observe(), step(action), success, and terminated.

        Episode counters exclude stabilization, which belongs to the runner.
        All mutable execution state is reset here, including failed subgoals.
        """
        self.episode_id, self.instruction, self.seed = episode_id, instruction, seed
        self.step, self.velocity_evaluations = 0, 0
        self.history, self.subgoals, self.seen_plans = [], {}, set()
        self.current_subgoal = None
        agent_counts = (
            (self.agent.calls, self.agent.invalid_responses, self.agent.retries)
            if self.agent is not None
            else (0, 0, 0)
        )
        self.monitor.reset()
        plan = None
        last_attempt = -self.config.agent.refresh_env_steps
        began = time.perf_counter()
        fallback_count, failure = 0, None
        budget = self.config.benchmark.task_action_budget
        self.recorder.event(
            "episode_start",
            episode_id=episode_id,
            instruction=instruction,
            seed=seed,
            metadata=metadata or {},
        )
        while self.step < budget and not env.success and not env.terminated:
            observation = env.observe()
            progress = self.monitor.check(
                observation, self.current_subgoal, step=self.step
            )
            self.recorder.event(
                "progress",
                episode_id=episode_id,
                observation_step=self.step,
                progress=progress,
            )
            requires_replan = False
            if self.current_subgoal is not None:
                subgoal = self.subgoals[self.current_subgoal.subgoal_id]
                if progress["complete"] and not subgoal.complete:
                    subgoal.complete = True
                    requires_replan = True
                    self.recorder.event(
                        "subgoal_completed",
                        episode_id=episode_id,
                        subgoal_id=self.current_subgoal.subgoal_id,
                        observation_step=self.step,
                    )
                elif (
                    (self.step >= subgoal.deadline or progress.get("stalled", False))
                    and not subgoal.exhausted
                    and not subgoal.complete
                ):
                    requires_replan = True
                    if subgoal.recoveries >= self.config.agent.subgoal_recovery_replans:
                        subgoal.exhausted = True
                    else:
                        subgoal.recoveries += 1
                        subgoal.deadline = (
                            self.step + self.current_subgoal.timeout_env_steps
                        )
                    self.recorder.event(
                        "subgoal_stalled"
                        if progress.get("stalled", False)
                        else "subgoal_timeout",
                        episode_id=episode_id,
                        observation_step=self.step,
                        subgoal_id=self.current_subgoal.subgoal_id,
                        recoveries=subgoal.recoveries,
                        exhausted=subgoal.exhausted,
                    )
            interval = self.config.agent.refresh_env_steps
            elapsed = self.step - last_attempt
            expired = (
                plan is not None
                and self.step - plan.created
                >= self.config.controller.latent_max_age_env_steps
            )
            refresh = elapsed >= interval or expired or requires_replan
            fallback_reason = None
            if refresh:
                # Generation happens only at boundaries; no superseded queued actions survive.
                plan = None
                last_attempt = self.step
                try:
                    proposal = None
                    if (
                        self.agent is not None
                        and self.config.method not in AGENT_FREE_METHODS
                    ):
                        active = (
                            None
                            if self.current_subgoal is None
                            else {
                                "subgoal_id": self.current_subgoal.subgoal_id,
                                "subgoal_instruction": self.current_subgoal.subgoal_instruction,
                                "progress": progress,
                                "recoveries": self.subgoals[
                                    self.current_subgoal.subgoal_id
                                ].recoveries,
                                "exhausted": self.subgoals[
                                    self.current_subgoal.subgoal_id
                                ].exhausted,
                            }
                        )
                        proposal = self.agent.propose(
                            {
                                "episode_id": episode_id,
                                "step": self.step,
                                "observation": observation,
                                "instruction": instruction,
                                "history": self.history[
                                    -self.config.agent.history_limit :
                                ],
                                "active": active,
                                "completed_subgoals": [
                                    key
                                    for key, value in self.subgoals.items()
                                    if value.complete
                                ][-self.config.agent.history_limit :],
                                "spec": self.actions.spec,
                                "stage": self.config.stage,
                                "completion_types": self.monitor.supported,
                            },
                            accept=self._validate_new_plan,
                        )
                    plan = self._activate(observation, proposal)
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    fallback_reason = f"plan_refresh_failed: {exc}"
                    self.recorder.event(
                        "plan_rejected",
                        episode_id=episode_id,
                        observation_step=self.step,
                        reason=fallback_reason,
                    )
            try:
                if plan is None:
                    fallback_reason = fallback_reason or "no_valid_plan"
                    actions, condition = self._fresh(observation, fallback=True)
                else:
                    actions, condition = self._generate(observation, plan)
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                fallback_reason = f"policy_generation_failed: {exc}"
                plan = None
                try:
                    actions, condition = self._fresh(observation, fallback=True)
                except (ValueError, FloatingPointError, RuntimeError) as fallback_error:
                    failure = f"fallback_failed: {fallback_error}"
                    self.recorder.event(
                        "episode_error", episode_id=episode_id, reason=failure
                    )
                    break
            if fallback_reason:
                fallback_count += 1
                self.recorder.event(
                    "fallback",
                    episode_id=episode_id,
                    observation_step=self.step,
                    reason=fallback_reason,
                )
            count = min(
                self.config.policy.execute_steps,
                len(actions),
                budget - self.step,
                interval - (self.step - last_attempt),
            )
            if plan is not None:
                count = min(
                    count,
                    self.config.controller.latent_max_age_env_steps
                    - (self.step - plan.created),
                )
                if plan.proposal is not None:
                    count = min(
                        count,
                        self.subgoals[plan.proposal.subgoal_id].deadline - self.step,
                    )
            if count <= 0:
                failure = "invalid_execution_deadline"
                self.recorder.event(
                    "episode_error", episode_id=episode_id, reason=failure
                )
                break
            for action in actions[:count]:
                observed = env.observe()
                action = np.asarray(action, dtype=np.float32).copy()
                env.step(action)
                plan_id = (
                    plan.proposal.plan_id
                    if plan is not None and plan.proposal
                    else None
                )
                latent_id = (
                    f"{episode_id}:{plan.created}"
                    if plan is not None and plan.latent is not None
                    else None
                )
                record = ControlStep(
                    episode_id,
                    self.step,
                    digest(observed),
                    plan_id,
                    latent_id,
                    condition.condition_id if condition is not None else "direct_astra",
                    action,
                    bool(env.success),
                    fallback_reason,
                    progress,
                    time.time(),
                )
                self.recorder.event("control_step", step=record, observation=observed)
                self.history.append(
                    {
                        "observation_step": self.step,
                        "action": action.tolist(),
                        "subgoal_id": self.current_subgoal.subgoal_id
                        if self.current_subgoal
                        else None,
                        "fallback_reason": fallback_reason,
                    }
                )
                self.history = self.history[-self.config.agent.history_limit :]
                self.step += 1
                if env.success or env.terminated:
                    break
        summary = {
            "episode_id": episode_id,
            "method": self.config.method,
            "success": bool(env.success),
            "actions": self.step,
            "wall_seconds": time.perf_counter() - began,
            "velocity_evaluations": self.velocity_evaluations,
            "fallbacks": fallback_count,
            "agent_calls": self.agent.calls - agent_counts[0] if self.agent else 0,
            "invalid_responses": self.agent.invalid_responses - agent_counts[1]
            if self.agent
            else 0,
            "retry_calls": self.agent.retries - agent_counts[2] if self.agent else 0,
            "completed_subgoals": sum(x.complete for x in self.subgoals.values()),
            "failure": failure,
            **(metadata or {}),
        }
        self.recorder.event("episode_end", **summary)
        return summary
