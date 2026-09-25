"""Observed phase interpolation with fixed resets and explicit decision budgets.

The common baseline uses the recovered original noise. Interpolation changes
conditioning while holding that noise fixed. This tests conditioning transfer;
it does not claim that inverting a policy-generated reference adds information.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .astra_client import ClientError
from .flow import error_metrics
from .intervention_search import InterventionSearch
from .interventions import apply_vision, perturb_noise, random_noise_proposal
from .records import digest

ARMS = (
    "random_noise",
    "oracle_tei",
    "oracle_tli",
    "oracle_tei_tli",
    "astra_tei",
    "astra_tli",
    "astra_tli_vision",
)


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/phase_interpolation_v1.json"
    protocol = json.loads(Path(path).read_text())
    if (
        protocol["schema_version"] != "phase-interpolation-1.0"
        or protocol["arms"] != list(ARMS)
        or protocol["attempt_budget"] != 3
        or protocol["oracle_attempt_budget"] != 2
        or protocol["action_budget"] != 300
        or protocol["execute_steps"] != 5
        or protocol["astra"]["call_interval"] != 25
        or protocol["astra"]["max_calls_per_rollout"] != 12
        or protocol["vision"]["valid_for_actions"] != 5
    ):
        raise ValueError("Unexpected phase protocol; change its version explicitly")
    return protocol


def outcome_feedback(attempt):
    return {
        "attempt_id": attempt["attempt_id"],
        "success": attempt["success"],
        "executed_actions": attempt["actions_executed"],
        "termination": attempt["status"],
        "error": attempt.get("error"),
    }


class OnlineInterpolation:
    """Fresh raw observations in; a held text choice and short-lived marks out."""

    def __init__(
        self,
        client,
        *,
        episode_id,
        attempt_id,
        operator,
        task,
        catalog,
        spec,
        protocol,
        previous_attempt=None,
        feedback=(),
        vision=False,
        record=None,
    ):
        self.client = client
        self.episode_id, self.attempt_id = episode_id, attempt_id
        self.operator, self.task, self.catalog = operator, task, catalog
        self.spec, self.protocol = spec, protocol
        self.previous_attempt, self.feedback = previous_attempt, feedback
        self.vision_enabled = vision
        self.record = record or (lambda *args, **kwargs: None)
        self.observations, self.decisions = [], []
        self.active = None
        self.active_decision_id = None
        self.marks, self.marks_step = [], None
        self.last_step = None

    def update(self, observation, step):
        from .interpolation_agent import build_request

        execute_steps = self.protocol["execute_steps"]
        if (
            type(step) is not int
            or step < 0
            or (self.last_step is None and step != 0)
            or (self.last_step is not None and step != self.last_step + execute_steps)
        ):
            raise ValueError("Online controller requires consecutive policy steps")
        self.last_step = step
        self.observations.append(
            {
                "label": f"step_{step}",
                "step": step,
                "observation": copy.deepcopy(observation),
            }
        )
        self.observations = self.observations[-4:]
        # Marks describe the observed pixels at the call, not a tracked object.
        self.marks = []
        self.marks_step = None
        settings = self.protocol["astra"]
        if step % settings["call_interval"] == 0:
            if len(self.decisions) >= settings["max_calls_per_rollout"]:
                raise ValueError("Scheduled call exceeds the frozen decision budget")
            index = len(self.decisions) + 1
            request = build_request(
                episode_id=self.episode_id,
                attempt_id=self.attempt_id,
                decision_index=index,
                observation_step=step,
                interpolation_mode=self.operator,
                target_task=self.task,
                source_catalog=self.catalog,
                observations=self.observations,
                previous_decisions=self.decisions,
                completed_rollout_feedback=self.feedback,
                previous_attempt=self.previous_attempt,
                active_interpolation=self.active,
                vision_enabled=self.vision_enabled,
                action_spec=self.spec,
                call_interval=settings["call_interval"],
                max_calls=settings["max_calls_per_rollout"],
                action_budget=self.protocol["action_budget"],
            )
            self.record("interpolation_request", request=request)
            proposal, error = None, None
            started = time.perf_counter()
            try:
                proposal = self.client.propose(request)
            except ClientError as exc:
                # The client accounts for the failed physical call. A failure
                # never obtains another call or an invented replacement choice.
                error = str(exc)
            if proposal is not None:
                self.active = {
                    key: proposal[key]
                    for key in ("source_a_id", "source_b_id", "alpha")
                }
                self.active_decision_id = proposal["decision_id"]
                self.marks = copy.deepcopy(proposal["vision"])
                self.marks_step = step
            decision = {
                "decision_index": index,
                "observation_step": step,
                "proposal": proposal,
                "accepted": proposal is not None,
                "error": error,
            }
            self.decisions.append(decision)
            self.record(
                "interpolation_decision",
                attempt_id=self.attempt_id,
                decision=decision,
                wall_seconds=time.perf_counter() - started,
                language_hold_after_failure=proposal is None
                and self.active is not None,
                raw_condition_fallback=proposal is None and self.active is None,
                vision_valid_until_step=step + execute_steps if self.marks else None,
            )
        return copy.deepcopy(self.active), copy.deepcopy(self.marks)


def arm_summary(attempts, budget):
    from .interpolation_agent import summarize_calls

    first = next((row for row in attempts if row["success"]), None)
    through_success = [
        row
        for row in attempts
        if first is None or row["iteration"] <= first["iteration"]
    ]
    calls = [call for row in attempts for call in row.get("provider_records", [])]
    prefix_calls = [
        call for row in through_success for call in row.get("provider_records", [])
    ]
    prefix_usage = summarize_calls(prefix_calls)
    return {
        "success": first is not None,
        "first_success_attempt": first["iteration"] if first else None,
        "full_rollout_revisions_to_success": first["iteration"] - 1 if first else None,
        "within_successful_rollout_decisions": len(first.get("decisions", []))
        if first
        else None,
        "decisions_through_success_or_cap": sum(
            len(row.get("decisions", [])) for row in through_success
        ),
        "physical_decisions": sum(len(row.get("decisions", [])) for row in attempts),
        "rollouts_executed": len(attempts),
        "attempt_budget": budget,
        "censored_without_success": first is None,
        "success_by_attempt": [
            any(row["success"] and row["iteration"] <= index for row in attempts)
            for index in range(1, budget + 1)
        ],
        "actions_through_success_or_cap": sum(
            row["actions_executed"] for row in through_success
        ),
        "velocity_evaluations_through_success_or_cap": sum(
            row["velocity_evaluations"] for row in through_success
        ),
        "rollout_seconds_through_success_or_cap": sum(
            row["wall_seconds"] for row in through_success
        ),
        "provider": summarize_calls(calls),
        "provider_through_success_or_cap": prefix_usage,
        "tokens_to_first_success": prefix_usage if first is not None else None,
        "physical_actions": sum(row["actions_executed"] for row in attempts),
        "physical_velocity_evaluations": sum(
            row["velocity_evaluations"] for row in attempts
        ),
        "development_extra_rollouts_after_success": len(attempts)
        - len(through_success),
    }


class InterpolationSearch(InterventionSearch):
    """One reset, isolated arm histories, a training-only text latent bank."""

    def __init__(self, *args, banks, catalog, oracle, development=False, **kwargs):
        # Reuse the established full-state inversion/native-parity gate. The old
        # development-only pooled embedding probe is unrelated to this study.
        super().__init__(*args, development=False, **kwargs)
        self.phase_development = development
        self.banks, self.catalog, self.oracle = banks, catalog, oracle
        self.sources = {row["source_id"]: row["prompt"] for row in catalog}
        self.report.update(
            schema_version="phase-interpolation-1.0",
            development=development,
            oracle=oracle,
            source_catalog=catalog,
        )
        self.interpolation_checked = False
        self.save()

    def check_interpolation(self, observation):
        from .records import to_numpy

        policy, prompt = self.policy, self.entry["instruction"]
        observation_id = digest(observation)
        source_ids = (self.oracle["source_a_id"], self.oracle["source_b_id"])
        sources = tuple(self.sources[source_id] for source_id in source_ids)
        specifications = (
            ("tei_identical_sources", "tei", 0.37, False),
            ("tli_zero_residual", "tli", 0.5, False),
            ("tei_nonzero_oracle_sources", "tei", 0.0, True),
            ("tli_nonzero_oracle_banks", "tli", 0.0, True),
        )
        gate = {
            "passed": False,
            "status": "running",
            "complete": False,
            "probe_kind": "fixed_weighted_numerical_probe_without_environment_actions",
            "observation_id": observation_id,
            "known_noise_sha256": digest(self.known),
            "solver": copy.deepcopy(self.protocol["execution_solver"]),
            "velocity_evaluations": 0,
            "velocity_evaluations_complete": True,
            "checks": {
                label: {"passed": None, "status": "not_run"}
                for label, _, _, _ in specifications
            },
        }
        self.report["interpolation_gate"] = gate
        active_label = None
        try:
            native = policy.prepare(observation, observation_id, prompt)
            latent = policy.tensor(self.known)
            gate["velocity_evaluations_complete"] = False
            output = policy.sample(native, latent, **self.protocol["execution_solver"])
            gate["velocity_evaluations"] += output.velocity_evaluations
            gate["velocity_evaluations_complete"] = True
            reference = to_numpy(output.value)
            reference_actions = policy.output_transform({"actions": reference[0]})[
                "actions"
            ]
            gate["native_condition_id"] = native.condition_id
            gate["reference_velocity_evaluations"] = output.velocity_evaluations
            self.recorder.event(
                "interpolation_gate_reference",
                observation=observation,
                observation_id=observation_id,
                condition_id=native.condition_id,
                latent=latent,
                generated_actions=output.value,
                decoded_actions=reference_actions,
                solver=gate["solver"],
                velocity_evaluations=output.velocity_evaluations,
            )
            for label, operator, alpha, nonzero in specifications:
                active_label = label
                row = {
                    "passed": False,
                    "status": "running",
                    "operator": operator,
                    "alpha": alpha,
                    "expected_effect": "nonzero" if nonzero else "identity",
                    "source_ids": list(source_ids) if nonzero else None,
                }
                gate["checks"][label] = row
                condition, provenance = policy.prepare_interpolated(
                    observation,
                    observation_id,
                    prompt,
                    source_prompts=sources if nonzero else (prompt, prompt),
                    alpha=alpha,
                    operator=operator,
                    text_latents={
                        "a": self.banks[source_ids[0]],
                        "b": self.banks[source_ids[1]],
                    }
                    if nonzero and operator == "tli"
                    else None,
                )
                row.update(condition_id=condition.condition_id, provenance=provenance)
                gate["velocity_evaluations_complete"] = False
                generated = policy.sample(
                    condition, latent, **self.protocol["execution_solver"]
                )
                gate["velocity_evaluations"] += generated.velocity_evaluations
                gate["velocity_evaluations_complete"] = True
                values = to_numpy(generated.value)
                actions = policy.output_transform({"actions": values[0]})["actions"]
                row.update(
                    errors=error_metrics(reference, values),
                    controlled_channel_errors=error_metrics(
                        reference[:, :, :7], values[:, :, :7]
                    ),
                    decoded_action_errors=error_metrics(reference_actions, actions),
                    velocity_evaluations=generated.velocity_evaluations,
                )
                row["passed"] = (
                    bool(provenance["has_effect"])
                    and row["controlled_channel_errors"]["max_abs"] > 0
                    and row["decoded_action_errors"]["max_abs"] > 0
                    if nonzero
                    else not provenance["has_effect"]
                    and condition.condition_id == native.condition_id
                    and row["errors"]["max_abs"] == 0
                    and row["decoded_action_errors"]["max_abs"] == 0
                )
                row["status"] = "passed" if row["passed"] else "failed"
                self.recorder.event(
                    "interpolation_gate_check",
                    label=label,
                    observation_id=observation_id,
                    latent_sha256=gate["known_noise_sha256"],
                    generated_actions=generated.value,
                    decoded_actions=actions,
                    **row,
                )
            gate["complete"] = True
            gate["passed"] = all(row["passed"] for row in gate["checks"].values())
            gate["status"] = "passed" if gate["passed"] else "failed"
        except Exception as exc:
            gate["status"] = "error"
            gate["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if active_label is not None:
                gate["checks"][active_label].update(
                    passed=False, status="error", error=gate["error"]
                )
            raise
        finally:
            # Counts cover returned solves. A solver exception explicitly marks
            # unknown partial work; failed measured gates still report all five.
            self.recorder.event("interpolation_gate", **gate)
            self.save()
        self.interpolation_checked = gate["passed"]
        if not gate["passed"]:
            raise ValueError(
                "Weighted interpolation identity or nonzero-effect gate failed; measured results were recorded"
            )

    def rollout(
        self, mode, iteration, *, previous=None, history=(), noise_proposal=None
    ):
        from .interpolation_agent import InterpolationClient
        from .interpolation_catalog import paper_alpha
        from .intervention_rollout import run_rollout

        env, _, _ = self.create_env(self.entry["task_id"], self.entry["seed"])
        spec = ActionSpec.from_environment(
            env, self.policy.horizon, self.policy.action_dim
        )
        self.spec = spec
        adapter = ActionAdapter(
            spec, self.policy.input_transform, self.policy.output_transform
        )
        attempt_id = f"{mode}_{iteration}"
        response_log = self.directory / f"{attempt_id}_provider.jsonl"
        operator = (
            mode.removeprefix("astra_").removeprefix("oracle_").removesuffix("_vision")
        )
        online = None
        if mode.startswith("astra_"):
            settings = self.protocol["astra"]
            client = InterpolationClient(
                model=settings["model"],
                response_log=response_log,
                timeout=settings["timeout_seconds"],
                sampling={
                    key: settings[key]
                    for key in ("reasoning_effort", "max_completion_tokens")
                },
            )

            def record_online(kind, **data):
                self.recorder.event(kind, **data)
                if kind == "interpolation_decision":
                    self.report["live"] = {
                        "attempt_id": attempt_id,
                        "mode": mode,
                        "iteration": iteration,
                        "decision_index": data["decision"]["decision_index"],
                        "observation_step": data["decision"]["observation_step"],
                    }
                    self.save()

            online = OnlineInterpolation(
                client,
                episode_id=self.entry["episode_id"],
                attempt_id=attempt_id,
                operator=operator,
                task=self.entry["instruction"],
                catalog=self.catalog,
                spec=spec.as_dict(),
                protocol=self.protocol,
                previous_attempt=previous,
                feedback=[outcome_feedback(row) for row in history],
                vision=mode.endswith("_vision"),
                record=record_online,
            )
        counter = {
            "velocity_evaluations": 0,
            "clipped_values": 0,
            "condition_preparation_seconds": 0.0,
            "vision_active_policy_calls": 0,
            "vision_changed_policy_calls": 0,
            "text_nonzero_policy_calls": 0,
            "accepted_decision_policy_calls": 0,
            "native_condition_fallback_policy_calls": 0,
        }
        executed_conditions = []
        fresh_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.protocol["seed"],
                    int(digest(self.entry["episode_id"])[:8], 16),
                    2,
                ]
            )
        )

        def act(observation, step):
            if self.recovered is None:
                self.initialize_noise(observation, spec)
            if not self.interpolation_checked:
                self.check_interpolation(observation)
            active, marks = None, []
            if online is not None:
                active, marks = online.update(observation, step)
            elif mode.startswith("oracle_"):
                active = {
                    "source_a_id": self.oracle["source_a_id"],
                    "source_b_id": self.oracle["source_b_id"],
                    "alpha": paper_alpha(
                        step // self.protocol["execute_steps"],
                        self.oracle["lambda_calls"],
                    ),
                }
            modified = apply_vision(observation, marks)
            if active is None:
                condition = self.policy.prepare(
                    modified, digest(observation), self.entry["instruction"]
                )
                provenance = None
            else:
                source_a, source_b = active["source_a_id"], active["source_b_id"]
                condition, provenance = self.policy.prepare_interpolated(
                    modified,
                    digest(observation),
                    self.entry["instruction"],
                    source_prompts=(self.sources[source_a], self.sources[source_b]),
                    alpha=active["alpha"],
                    operator=operator,
                    text_latents={"a": self.banks[source_a], "b": self.banks[source_b]}
                    if "tli" in operator
                    else None,
                )
            if mode == "policy_fresh":
                latent = (
                    self.policy.tensor(self.known)
                    if step == 0
                    else self.policy.noise(fresh_rng)
                )
            elif mode == "known_noise":
                latent = self.policy.tensor(self.known)
            else:
                latent = self.policy.tensor(
                    perturb_noise(self.recovered, self.basis, noise_proposal)
                )
            generated = self.policy.sample(
                condition, latent, **self.protocol["execution_solver"]
            )
            actions, clipping = adapter.decode(generated.value, condition.state)
            changed_vision = digest(modified) != digest(observation)
            changed_text = bool(provenance and provenance["has_effect"])
            decision_id = online.active_decision_id if online is not None else None
            held_after_failure = bool(
                online is not None
                and decision_id is not None
                and not online.decisions[-1]["accepted"]
            )
            raw_fallback = online is not None and active is None
            counter["velocity_evaluations"] += generated.velocity_evaluations
            counter["clipped_values"] += clipping["count"]
            counter["condition_preparation_seconds"] += condition.preparation_seconds
            counter["vision_active_policy_calls"] += bool(marks)
            counter["vision_changed_policy_calls"] += changed_vision
            counter["text_nonzero_policy_calls"] += changed_text
            counter["accepted_decision_policy_calls"] += decision_id is not None
            counter["native_condition_fallback_policy_calls"] += raw_fallback
            executed_conditions.append(
                {
                    "step": step,
                    "decision_id": decision_id,
                    "changed_text": changed_text,
                    "changed_vision": changed_vision,
                    "held_after_failure": held_after_failure,
                    "raw_fallback": raw_fallback,
                }
            )
            self.recorder.event(
                "phase_generation",
                mode=mode,
                iteration=iteration,
                attempt_id=attempt_id,
                observation_step=step,
                observation=observation,
                modified_observation_sha256=digest(modified),
                vision=marks,
                active_interpolation=active,
                applied_accepted_decision_id=decision_id,
                text_has_effect=changed_text,
                vision_has_effect=changed_vision,
                held_text_after_failed_call=held_after_failure,
                native_condition_fallback=raw_fallback,
                condition_id=condition.condition_id,
                conditioning=provenance,
                latent=latent,
                generated_actions=generated.value,
                controller_actions=actions,
                clipping=clipping,
                velocity_evaluations=generated.velocity_evaluations,
                flow_seconds=generated.latency_seconds,
            )
            return actions

        result = run_rollout(
            env,
            self.entry,
            self.benchmark,
            act,
            execute_steps=self.protocol["execute_steps"],
            action_budget=self.protocol["action_budget"],
            expected_reset=self.reset_audit,
            policy_image_size=self.policy.observation_image_size,
            video_path=self.directory / f"{attempt_id}.mp4",
        )
        snapshots = result.pop("snapshots")
        if result["initial_success"]:
            raise ValueError(
                "Initially successful reset is not an intervention success"
            )
        if self.reset_audit is None:
            self.reset_audit = result["reset_audit"]
        records = (
            [json.loads(line) for line in response_log.read_text().splitlines()]
            if response_log.exists()
            else []
        )
        applied_ids = []
        applied_actions = {
            "actions_with_accepted_decision": 0,
            "actions_with_nonzero_text": 0,
            "actions_with_changed_vision": 0,
            "actions_with_held_text_after_failed_call": 0,
            "native_condition_fallback_actions": 0,
        }
        for row in executed_conditions:
            count = min(
                self.protocol["execute_steps"], result["actions_executed"] - row["step"]
            )
            if count <= 0:
                raise ValueError(
                    "Generated interpolation condition has no executed actions"
                )
            if row["decision_id"] is not None:
                applied_actions["actions_with_accepted_decision"] += count
                if row["decision_id"] not in applied_ids:
                    applied_ids.append(row["decision_id"])
            for flag, field in (
                ("changed_text", "actions_with_nonzero_text"),
                ("changed_vision", "actions_with_changed_vision"),
                ("held_after_failure", "actions_with_held_text_after_failed_call"),
                ("raw_fallback", "native_condition_fallback_actions"),
            ):
                if row[flag]:
                    applied_actions[field] += count
        attempt = {
            **result,
            **counter,
            "attempt_id": attempt_id,
            "iteration": iteration,
            "mode": mode,
            "decisions": online.decisions if online else [],
            "noise_proposal": noise_proposal,
            "provider_records": records,
            "applied_accepted_decision_ids": applied_ids,
            "accepted_decisions_executed": len(applied_ids),
            **applied_actions,
            "status": "success"
            if result["success"]
            else "terminated"
            if result["terminated"]
            else "budget_exhausted",
        }
        self.recorder.event("phase_attempt", attempt=attempt, snapshots=snapshots)
        return attempt, snapshots

    def summarize_arm(self, attempts, budget):
        summary = arm_summary(attempts, budget)
        summary["standalone_velocity_evaluations_through_success_or_cap"] = (
            summary["velocity_evaluations_through_success_or_cap"]
            + self.report["initialization"]["velocity_evaluations"]
            + self.report["interpolation_gate"]["velocity_evaluations"]
        )
        return summary

    def physical_cost(self):
        from .interpolation_agent import summarize_calls

        attempts = [self.report["baseline"], *self.report["controls"].values()] + [
            row for arm in self.report["arms"].values() for row in arm["attempts"][1:]
        ]
        if len({row["attempt_id"] for row in attempts}) != len(attempts):
            raise ValueError("Physical cost would double count a rollout")
        initialization = self.report["initialization"]["velocity_evaluations"]
        gate = self.report["interpolation_gate"]["velocity_evaluations"]
        rollout_vf = sum(row["velocity_evaluations"] for row in attempts)
        return {
            "rollouts": len(attempts),
            "simulated_actions": sum(row["actions_executed"] for row in attempts),
            "rollout_velocity_evaluations": rollout_vf,
            "initialization_velocity_evaluations": initialization,
            "interpolation_gate_velocity_evaluations": gate,
            "velocity_evaluations": rollout_vf + initialization + gate,
            "rollout_wall_seconds": sum(row["wall_seconds"] for row in attempts),
            "wall_time_note": "Baseline rollout wall time includes initialization, interpolation checks and recording; do not add those times again.",
            "token_usage": summarize_calls(
                [call for row in attempts for call in row.get("provider_records", [])]
            ),
        }

    def validate_development(self):
        checks = {}
        for arm, report in self.report["arms"].items():
            if not arm.startswith("astra_"):
                continue
            attempts = report["attempts"][1:]
            applied_ids = [
                decision_id
                for row in attempts
                for decision_id in row["applied_accepted_decision_ids"]
            ]
            matched = True
            for row in attempts:
                proposals = {
                    decision["proposal"]["decision_id"]: decision["proposal"]
                    for decision in row["decisions"]
                    if decision["accepted"]
                }
                for decision_id in row["applied_accepted_decision_ids"]:
                    proposal = proposals.get(decision_id)
                    candidates = [
                        call
                        for call in row["provider_records"]
                        if proposal is not None
                        and call["provider_call"]
                        and call["accepted"]
                        and call.get("attempt_id") == row["attempt_id"]
                        and call.get("decision_id") == decision_id
                        and call.get("request_fingerprint")
                        == proposal["request_fingerprint"]
                        and call.get("response", {}).get("model")
                        == self.protocol["astra"]["model"]
                    ]
                    if len(candidates) != 1:
                        matched = False
                        continue
                    try:
                        actual = json.loads(
                            candidates[0]["response"]["choices"][0]["message"][
                                "content"
                            ]
                        )
                    except (KeyError, IndexError, TypeError, ValueError):
                        matched = False
                    else:
                        matched = matched and actual == proposal
            checks[arm] = {
                "passed": bool(applied_ids) and matched,
                "accepted_decisions_executed": len(applied_ids),
                "provider_bindings_verified": matched,
                "text_nonzero_policy_calls": sum(
                    row["text_nonzero_policy_calls"] for row in attempts
                ),
                "vision_changed_policy_calls": sum(
                    row["vision_changed_policy_calls"] for row in attempts
                ),
                "note": "An accepted and executed proposal is required. Nonzero conditioning counts are measured separately and are not a success attribution.",
            }
        passed = bool(checks) and all(row["passed"] for row in checks.values())
        self.report["development_validation"] = {"passed": passed, "arms": checks}
        self.recorder.event(
            "phase_development_validation", **self.report["development_validation"]
        )
        self.save()
        if not passed:
            self.report["status"] = "development_failed"
            self.save()
            raise ValueError(
                "Development requires a genuine accepted-and-executed decision in every Astra arm"
            )

    def run(self):
        started = time.perf_counter()
        baseline, initial_snapshots = self.rollout("recovered_noise", 1)
        self.report["baseline"] = baseline
        self.save()
        for mode in ("known_noise", "policy_fresh"):
            self.report["controls"][mode], _ = self.rollout(mode, 1)
            self.save()
        random_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.protocol["seed"],
                    int(digest(self.entry["episode_id"])[:8], 16),
                    3,
                ]
            )
        )
        for arm in self.protocol["arms"]:
            budget = (
                self.protocol["oracle_attempt_budget"]
                if arm.startswith("oracle_")
                else self.budget
            )
            attempts, snapshots = [baseline], initial_snapshots
            for iteration in range(2, budget + 1):
                force = self.phase_development and iteration == 2
                if any(row["success"] for row in attempts) and not force:
                    break
                last = attempts[-1]
                previous = {
                    "feedback": outcome_feedback(last),
                    "decisions": last["decisions"],
                    "snapshots": snapshots,
                }
                proposal = (
                    random_noise_proposal(self.basis, random_rng)
                    if arm == "random_noise"
                    else None
                )
                attempt, snapshots = self.rollout(
                    arm,
                    iteration,
                    previous=previous,
                    history=attempts,
                    noise_proposal=proposal,
                )
                attempts.append(attempt)
                self.report["arms"][arm] = {
                    "attempts": attempts,
                    "summary": self.summarize_arm(attempts, budget),
                }
                self.save()
            self.report["arms"][arm] = {
                "attempts": attempts,
                "summary": self.summarize_arm(attempts, budget),
            }
            self.save()
        self.report["physical_cost"] = self.physical_cost()
        self.report["total_wall_seconds"] = time.perf_counter() - started
        self.save()
        if self.phase_development:
            self.validate_development()
        self.report["status"] = "complete"
        self.save()
        return self.report


def aggregate_reports(reports, protocol):
    if len({row["episode_id"] for row in reports}) != len(reports):
        raise ValueError("Duplicate interpolation episode report")
    return {
        "schema_version": "phase-interpolation-1.0",
        "protocol_sha256": digest(protocol),
        "cases": len(reports),
        "completed_cases": sum(row["status"] == "complete" for row in reports),
        "baseline_successes": sum(
            row.get("baseline", {}).get("success", False) for row in reports
        ),
        "arms": {
            arm: {
                "successes": sum(
                    row.get("arms", {})
                    .get(arm, {})
                    .get("summary", {})
                    .get("success", False)
                    for row in reports
                ),
                "cases_with_arm": sum(arm in row.get("arms", {}) for row in reports),
            }
            for arm in protocol["arms"]
        },
    }
