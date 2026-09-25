"""Closed-loop image-input interventions with fixed language and recovered noise."""

import copy
import json
import time
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .astra_client import ClientError
from .flow import error_metrics
from .image_perturbations import (
    CAMERAS,
    ImagePerturbationLimits,
    apply_image_perturbations,
)
from .interpolation_search import outcome_feedback
from .intervention_search import InterventionSearch
from .interventions import perturb_noise, random_noise_proposal
from .records import digest, to_numpy

ARMS = (
    "random_noise",
    "random_occlusion",
    "random_demo_blend",
    "astra_occlusion",
    "astra_demo_blend",
)


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/image_perturbations_v1.json"
    value = json.loads(Path(path).read_text())
    if (
        value["schema_version"] != "image-perturbations-1.0"
        or value["arms"] != list(ARMS)
        or value["attempt_budget"] != 3
        or value["action_budget"] != 300
        or value["execute_steps"] != 5
        or value["astra"]["call_interval"] != 25
        or value["astra"]["max_calls_per_rollout"] != 12
        or value["images"]["valid_for_actions"] != 25
        or value["images"]["fill_rgb"] != [127, 127, 127]
        or value["images"]["max_occlusion_fraction"] != 0.5
    ):
        raise ValueError("Unexpected image protocol; change its version explicitly")
    return value


def random_image_operations(mode, observation, donor_ids, rng):
    """Frozen matched-space random control, sampled only at decision intervals."""
    if mode not in ("occlusion", "demo_blend"):
        raise ValueError("Unsupported random image mode")
    if rng.random() < 0.1:
        return []
    selected = ((CAMERAS[0],), (CAMERAS[1],), CAMERAS)[int(rng.integers(3))]
    operations = []
    for camera in selected:
        if mode == "demo_blend":
            operations.append(
                {
                    "kind": mode,
                    "camera": camera,
                    "donor_id": donor_ids[int(rng.integers(len(donor_ids)))],
                    "alpha": float(rng.random()),
                }
            )
        else:
            height, width, _ = observation[camera].shape
            w = int(rng.integers(1, width + 1))
            max_height = min(height, (width * height) // (2 * w))
            h = int(rng.integers(1, max_height + 1))
            x = int(rng.integers(0, width - w + 1))
            y = int(rng.integers(0, height - h + 1))
            operations.append(
                {
                    "kind": mode,
                    "camera": camera,
                    "box_xyxy": [x, y, x + w, y + h],
                    "fill_rgb": [127, 127, 127],
                    "strength": float(rng.random()),
                }
            )
    return operations


class OnlineImagePerturbation:
    """A scheduled choice edits fresh frames; failed refreshes clear all edits."""

    def __init__(
        self,
        client,
        *,
        episode_id,
        attempt_id,
        image_mode,
        task,
        library,
        spec,
        protocol,
        previous_attempt=None,
        feedback=(),
        record=None,
    ):
        self.client = client
        self.episode_id, self.attempt_id = episode_id, attempt_id
        self.image_mode, self.task = image_mode, task
        self.library, self.spec, self.protocol = library, spec, protocol
        self.previous_attempt, self.feedback = previous_attempt, feedback
        self.record = record or (lambda *args, **kwargs: None)
        self.observations, self.decisions = [], []
        self.active, self.active_decision_id, self.last_step = [], None, None

    def update(self, observation, step):
        from .image_perturbation_agent import build_request

        if (
            type(step) is not int
            or step < 0
            or (self.last_step is None and step != 0)
            or (
                self.last_step is not None
                and step != self.last_step + self.protocol["execute_steps"]
            )
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
        settings = self.protocol["astra"]
        if step % settings["call_interval"] == 0:
            if len(self.decisions) >= settings["max_calls_per_rollout"]:
                raise ValueError("Scheduled call exceeds the frozen decision budget")
            self.active, self.active_decision_id = [], None
            request = build_request(
                episode_id=self.episode_id,
                attempt_id=self.attempt_id,
                decision_index=len(self.decisions) + 1,
                observation_step=step,
                image_mode=self.image_mode,
                target_task=self.task,
                donor_catalog=self.library.catalog(),
                contact_sheets=self.library.contact_sheets(),
                observations=self.observations,
                previous_decisions=self.decisions,
                completed_rollout_feedback=self.feedback,
                previous_attempt=self.previous_attempt,
                action_spec=self.spec,
                call_interval=settings["call_interval"],
                max_calls=settings["max_calls_per_rollout"],
                action_budget=self.protocol["action_budget"],
            )
            self.record("image_request", request=request)
            proposal, error = None, None
            started = time.perf_counter()
            try:
                proposal = self.client.propose(request)
            except ClientError as exc:
                error = str(exc)
            if proposal is not None:
                self.active = copy.deepcopy(proposal["image_perturbations"])
                self.active_decision_id = proposal["decision_id"]
            decision = {
                "decision_index": len(self.decisions) + 1,
                "observation_step": step,
                "proposal": proposal,
                "accepted": proposal is not None,
                "error": error,
            }
            self.decisions.append(decision)
            self.record(
                "image_decision",
                attempt_id=self.attempt_id,
                decision=decision,
                wall_seconds=time.perf_counter() - started,
                valid_until_step=step + settings["call_interval"],
                raw_condition_fallback=proposal is None,
            )
        return copy.deepcopy(self.active)


def arm_summary(attempts, budget):
    from .image_perturbation_agent import summarize_calls

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


class ImagePerturbationSearch(InterventionSearch):
    def __init__(self, *args, library, development=False, **kwargs):
        super().__init__(*args, development=False, **kwargs)
        self.image_development, self.library = development, library
        self.image_checked = False
        self.report.update(
            schema_version="image-perturbations-1.0",
            development=development,
            image_library=library.metadata(),
        )
        self.save()

    def check_images(self, observation):
        """Real checkpoint: exact no-ops and nonzero decoded-action effects."""
        camera = CAMERAS[0]
        height, width, _ = observation[camera].shape
        donor_id = self.library.catalog()[0]["donor_id"]
        blend = {
            "kind": "demo_blend",
            "camera": camera,
            "donor_id": donor_id,
            "alpha": 0.5,
        }
        mask = {
            "kind": "occlusion",
            "camera": camera,
            "box_xyxy": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            "fill_rgb": [127, 127, 127],
            "strength": 1.0,
        }
        specifications = (
            ("empty_identity", [], False),
            ("blend_zero_identity", [{**blend, "alpha": 0.0}], False),
            ("occlusion_zero_identity", [{**mask, "strength": 0.0}], False),
            ("blend_nonzero", [blend], True),
            ("occlusion_nonzero", [mask], True),
        )
        gate = {
            "passed": False,
            "status": "running",
            "complete": False,
            "probe_kind": "fixed_known_noise_without_environment_actions",
            "observation_id": digest(observation),
            "known_noise_sha256": digest(self.known),
            "solver": copy.deepcopy(self.protocol["execution_solver"]),
            "velocity_evaluations": 0,
            "velocity_evaluations_complete": True,
            "checks": {},
        }
        self.report["image_gate"] = gate
        try:
            native = self.policy.prepare(
                observation, digest(observation), self.entry["instruction"]
            )
            latent = self.policy.tensor(self.known)
            gate["velocity_evaluations_complete"] = False
            output = self.policy.sample(native, latent, **gate["solver"])
            gate["velocity_evaluations"] += output.velocity_evaluations
            gate["velocity_evaluations_complete"] = True
            reference = to_numpy(output.value)
            actions = self.policy.output_transform({"actions": reference[0]})["actions"]
            gate["native_condition_id"] = native.condition_id
            self.recorder.event(
                "image_gate_reference",
                observation=observation,
                latent=latent,
                generated_actions=output.value,
                decoded_actions=actions,
                condition_id=native.condition_id,
                velocity_evaluations=output.velocity_evaluations,
            )
            for label, operations, nonzero in specifications:
                edited, audit = apply_image_perturbations(
                    observation, operations, self.library.resolve
                )
                condition = self.policy.prepare(
                    edited, digest(edited), self.entry["instruction"]
                )
                gate["velocity_evaluations_complete"] = False
                generated = self.policy.sample(condition, latent, **gate["solver"])
                gate["velocity_evaluations"] += generated.velocity_evaluations
                gate["velocity_evaluations_complete"] = True
                values = to_numpy(generated.value)
                decoded = self.policy.output_transform({"actions": values[0]})[
                    "actions"
                ]
                changed = digest(edited) != digest(observation)
                row = {
                    "operations": operations,
                    "expected_effect": "nonzero" if nonzero else "identity",
                    "condition_id": condition.condition_id,
                    "errors": error_metrics(reference, values),
                    "controlled_channel_errors": error_metrics(
                        reference[:, :, :7], values[:, :, :7]
                    ),
                    "decoded_action_errors": error_metrics(actions, decoded),
                    "pixels_changed": changed,
                    "velocity_evaluations": generated.velocity_evaluations,
                }
                row["passed"] = (
                    changed
                    and row["controlled_channel_errors"]["max_abs"] > 0
                    and row["decoded_action_errors"]["max_abs"] > 0
                    if nonzero
                    else not changed
                    and condition.condition_id == native.condition_id
                    and row["errors"]["max_abs"] == 0
                    and row["decoded_action_errors"]["max_abs"] == 0
                )
                gate["checks"][label] = row
                self.recorder.event(
                    "image_gate_check",
                    label=label,
                    observation=observation,
                    modified_observation=edited,
                    image_audit=audit,
                    generated_actions=generated.value,
                    decoded_actions=decoded,
                    **row,
                )
            gate["complete"] = True
            gate["passed"] = all(row["passed"] for row in gate["checks"].values())
            gate["status"] = "passed" if gate["passed"] else "failed"
        except Exception as exc:
            gate.update(
                status="error", error={"type": type(exc).__name__, "message": str(exc)}
            )
            raise
        finally:
            self.recorder.event("image_gate", **gate)
            self.save()
        self.image_checked = gate["passed"]
        if not self.image_checked:
            raise ValueError("Native image identity or nonzero-effect gate failed")

    def rollout(
        self, mode, iteration, *, previous=None, history=(), noise_proposal=None
    ):
        from .image_perturbation_agent import ImagePerturbationClient
        from .intervention_rollout import run_rollout

        env, _, _ = self.create_env(self.entry["task_id"], self.entry["seed"])
        spec = ActionSpec.from_environment(
            env, self.policy.horizon, self.policy.action_dim
        )
        adapter = ActionAdapter(
            spec, self.policy.input_transform, self.policy.output_transform
        )
        attempt_id = f"{mode}_{iteration}"
        response_log = self.directory / f"{attempt_id}_provider.jsonl"
        image_mode = mode.removeprefix("astra_").removeprefix("random_")
        limits = (
            ImagePerturbationLimits(allowed_kinds=(image_mode,))
            if image_mode in ("occlusion", "demo_blend")
            else ImagePerturbationLimits()
        )
        online, random_decisions = None, []
        if mode.startswith("astra_"):
            settings = self.protocol["astra"]
            client = ImagePerturbationClient(
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
                if kind == "image_decision":
                    self.report["live"] = {
                        "attempt_id": attempt_id,
                        "mode": mode,
                        "iteration": iteration,
                        "decision_index": data["decision"]["decision_index"],
                        "observation_step": data["decision"]["observation_step"],
                    }
                    self.save()

            online = OnlineImagePerturbation(
                client,
                episode_id=self.entry["episode_id"],
                attempt_id=attempt_id,
                image_mode=image_mode,
                task=self.entry["instruction"],
                library=self.library,
                spec=spec.as_dict(),
                protocol=self.protocol,
                previous_attempt=previous,
                feedback=[outcome_feedback(row) for row in history],
                record=record_online,
            )
        counter = {
            key: 0
            for key in (
                "velocity_evaluations",
                "clipped_values",
                "image_active_policy_calls",
                "image_changed_policy_calls",
                "accepted_decision_policy_calls",
                "native_condition_fallback_policy_calls",
            )
        }
        counter["condition_preparation_seconds"] = 0.0
        executed = []
        seed_words = [
            self.protocol["seed"],
            int(digest(self.entry["episode_id"])[:8], 16),
        ]
        fresh_rng = np.random.default_rng(np.random.SeedSequence(seed_words + [2]))
        random_rng = np.random.default_rng(
            np.random.SeedSequence(seed_words + [int(digest(mode)[:8], 16), iteration])
        )
        active, random_decision_id = [], None

        def act(observation, step):
            nonlocal active, random_decision_id
            if self.recovered is None:
                self.initialize_noise(observation, spec)
            if not self.image_checked:
                self.check_images(observation)
            if online is not None:
                active = online.update(observation, step)
            elif (
                mode in ("random_occlusion", "random_demo_blend")
                and step % self.protocol["astra"]["call_interval"] == 0
            ):
                active = random_image_operations(
                    image_mode,
                    observation,
                    [row["donor_id"] for row in self.library.catalog()],
                    random_rng,
                )
                random_decision_id = (
                    f"{attempt_id}_decision_{len(random_decisions) + 1}"
                )
                decision = {
                    "decision_index": len(random_decisions) + 1,
                    "observation_step": step,
                    "accepted": True,
                    "error": None,
                    "proposal": {
                        "decision_id": random_decision_id,
                        "image_perturbations": copy.deepcopy(active),
                    },
                }
                random_decisions.append(decision)
                self.recorder.event(
                    "random_image_decision", attempt_id=attempt_id, decision=decision
                )
            edited, audit = apply_image_perturbations(
                observation, active, self.library.resolve, limits=limits
            )
            condition = self.policy.prepare(
                edited, digest(edited), self.entry["instruction"]
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
            changed = digest(edited) != digest(observation)
            decision_id = (
                online.active_decision_id if online is not None else random_decision_id
            )
            fallback = online is not None and decision_id is None
            counter["velocity_evaluations"] += generated.velocity_evaluations
            counter["clipped_values"] += clipping["count"]
            counter["condition_preparation_seconds"] += condition.preparation_seconds
            counter["image_active_policy_calls"] += bool(active)
            counter["image_changed_policy_calls"] += changed
            counter["accepted_decision_policy_calls"] += decision_id is not None
            counter["native_condition_fallback_policy_calls"] += fallback
            executed.append(
                {
                    "step": step,
                    "decision_id": decision_id,
                    "changed": changed,
                    "fallback": fallback,
                }
            )
            self.recorder.event(
                "image_generation",
                mode=mode,
                iteration=iteration,
                attempt_id=attempt_id,
                observation_step=step,
                observation=observation,
                observation_sha256=digest(observation),
                modified_observation=edited,
                modified_observation_sha256=digest(edited),
                image_perturbations=copy.deepcopy(active),
                image_audit=audit,
                applied_accepted_decision_id=decision_id,
                image_has_effect=changed,
                native_condition_fallback=fallback,
                condition_id=condition.condition_id,
                instruction=self.entry["instruction"],
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
        applied = {
            "actions_with_accepted_decision": 0,
            "actions_with_changed_image": 0,
            "native_condition_fallback_actions": 0,
        }
        for row in executed:
            count = min(
                self.protocol["execute_steps"], result["actions_executed"] - row["step"]
            )
            if count <= 0:
                raise ValueError("Generated image condition has no executed actions")
            if row["decision_id"] is not None:
                applied["actions_with_accepted_decision"] += count
                if row["decision_id"] not in applied_ids:
                    applied_ids.append(row["decision_id"])
            if row["changed"]:
                applied["actions_with_changed_image"] += count
            if row["fallback"]:
                applied["native_condition_fallback_actions"] += count
        attempt = {
            **result,
            **counter,
            **applied,
            "attempt_id": attempt_id,
            "iteration": iteration,
            "mode": mode,
            "decisions": online.decisions if online else random_decisions,
            "noise_proposal": noise_proposal,
            "provider_records": records,
            "applied_accepted_decision_ids": applied_ids,
            "accepted_decisions_executed": len(applied_ids),
            "status": "success"
            if result["success"]
            else "terminated"
            if result["terminated"]
            else "budget_exhausted",
        }
        self.recorder.event("image_attempt", attempt=attempt, snapshots=snapshots)
        return attempt, snapshots

    def summarize_arm(self, attempts):
        summary = arm_summary(attempts, self.budget)
        summary["standalone_velocity_evaluations_through_success_or_cap"] = (
            summary["velocity_evaluations_through_success_or_cap"]
            + self.report["initialization"]["velocity_evaluations"]
            + self.report["image_gate"]["velocity_evaluations"]
        )
        return summary

    def physical_cost(self):
        from .image_perturbation_agent import summarize_calls

        attempts = [self.report["baseline"], *self.report["controls"].values()] + [
            row for arm in self.report["arms"].values() for row in arm["attempts"][1:]
        ]
        if len({row["attempt_id"] for row in attempts}) != len(attempts):
            raise ValueError("Physical cost would double count a rollout")
        initialization = self.report["initialization"]["velocity_evaluations"]
        gate = self.report["image_gate"]["velocity_evaluations"]
        rollout_vf = sum(row["velocity_evaluations"] for row in attempts)
        return {
            "rollouts": len(attempts),
            "simulated_actions": sum(row["actions_executed"] for row in attempts),
            "rollout_velocity_evaluations": rollout_vf,
            "initialization_velocity_evaluations": initialization,
            "image_gate_velocity_evaluations": gate,
            "velocity_evaluations": rollout_vf + initialization + gate,
            "rollout_wall_seconds": sum(row["wall_seconds"] for row in attempts),
            "wall_time_note": "Baseline rollout time includes initialization and image gate; do not add those times again.",
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
                value
                for row in attempts
                for value in row["applied_accepted_decision_ids"]
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
                        expected = {
                            key: value
                            for key, value in proposal.items()
                            if key != "request_fingerprint"
                        }
                        matched = matched and actual == expected
            checks[arm] = {
                "passed": bool(applied_ids) and matched,
                "accepted_decisions_executed": len(applied_ids),
                "provider_bindings_verified": matched,
                "image_changed_policy_calls": sum(
                    row["image_changed_policy_calls"] for row in attempts
                ),
            }
        passed = bool(checks) and all(row["passed"] for row in checks.values())
        self.report["development_validation"] = {"passed": passed, "arms": checks}
        self.recorder.event(
            "image_development_validation", **self.report["development_validation"]
        )
        self.save()
        if not passed:
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
            attempts, snapshots = [baseline], initial_snapshots
            for iteration in range(2, self.budget + 1):
                force = self.image_development and iteration == 2
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
                    "summary": self.summarize_arm(attempts),
                }
                self.save()
            self.report["arms"][arm] = {
                "attempts": attempts,
                "summary": self.summarize_arm(attempts),
            }
            self.save()
        self.report["physical_cost"] = self.physical_cost()
        self.report["total_wall_seconds"] = time.perf_counter() - started
        if self.image_development:
            self.validate_development()
        self.report["status"] = "complete"
        self.save()
        return self.report


def aggregate_reports(reports, protocol):
    if len({row["episode_id"] for row in reports}) != len(reports):
        raise ValueError("Duplicate image episode report")
    return {
        "schema_version": "image-perturbations-1.0",
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
