"""Full-rollout feedback search over a frozen policy's inputs and recovered noise."""

import json
import time
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .astra_client import ClientError
from .flow import error_metrics
from .interventions import (
    ARMS,
    apply_vision,
    noise_basis,
    perturb_noise,
    random_noise_proposal,
    success_curve,
)
from .records import Recorder, digest, to_numpy


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/iterative_interventions_v1.json"
    protocol = json.loads(Path(path).read_text())
    if (
        protocol["arms"] != list(ARMS)
        or protocol["attempt_budget"] != 5
        or protocol["noise"]["rank"] != 8
        or protocol["seed"] != 19
    ):
        raise ValueError("Unexpected intervention protocol; version changes explicitly")
    return protocol


def feedback_row(attempt):
    return {
        "candidate_id": attempt["candidate_id"],
        "iteration": attempt["iteration"],
        "proposal": attempt.get("proposal"),
        "outcome": {
            "success": attempt["success"] if attempt["rollout_executed"] else None,
            "executed_steps": attempt["actions_executed"],
            "termination": attempt["status"],
            "error": attempt.get("error"),
        },
    }


def arm_summary(attempts, budget):
    summary = success_curve(attempts, budget)
    first = summary["first_success_attempt"]
    through_success = [
        row for row in attempts if first is None or row["iteration"] <= first
    ]
    summary.update(
        actions_through_success_or_cap=sum(
            row["actions_executed"] for row in through_success
        ),
        velocity_evaluations_through_success_or_cap=sum(
            row["velocity_evaluations"] for row in through_success
        ),
        rollout_seconds_through_success_or_cap=sum(
            row["wall_seconds"] for row in through_success
        ),
    )
    return summary


class InterventionSearch:
    """One case; each arm sees its own history and identical captured reset."""

    def __init__(
        self,
        policy,
        create_env,
        benchmark,
        entry,
        protocol,
        directory,
        *,
        development=False,
        progress=None,
    ):
        from .intervention_agent import InterventionClient

        self.policy, self.create_env, self.benchmark = policy, create_env, benchmark
        self.entry, self.protocol = entry, protocol
        self.directory = Path(directory)
        self.recorder = Recorder(directory)
        self.development, self.progress = development, progress or (lambda: None)
        self.budget = protocol["attempt_budget"]
        self.reset_audit = None
        self.recovered = self.known = None
        seed_words = [protocol["seed"], int(digest(entry["episode_id"])[:8], 16)]
        self.rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        self.basis, self.basis_id = noise_basis(
            (1, policy.horizon, policy.action_dim),
            np.random.SeedSequence(seed_words + [1]),
        )
        self.client_type = InterventionClient
        self.report = {
            "schema_version": "1.0",
            "status": "running",
            "episode_id": entry["episode_id"],
            "suite": benchmark.suite,
            "task_id": entry["task_id"],
            "seed": protocol["seed"],
            "development": development,
            "protocol_sha256": digest(protocol),
            "reset_entry_sha256": digest(entry),
            "checkpoint": policy.metadata,
            "arms": {},
            "controls": {},
        }
        self.recorder.event(
            "case",
            entry=entry,
            protocol=protocol,
            basis=self.basis,
            basis_id=self.basis_id,
        )
        self.save()

    def save(self):
        write_json(self.directory / "summary.json", self.report)
        self.progress()

    def initialize_noise(self, observation, spec):
        """Recover a policy-supported reference; arbitrary action inversion is separate."""
        from .intervention_conditioning import TextEmbeddingIntervention

        started = time.perf_counter()
        policy, solver = self.policy, self.protocol["inversion_solver"]
        condition = policy.prepare(
            observation, digest(observation), self.entry["instruction"]
        )
        known = policy.noise(self.rng)
        reference = policy.sample(condition, known, **solver)
        inverse = policy.invert(condition, reference.value, **solver)
        roundtrip = policy.sample(condition, inverse.value, **solver)
        native_solve = policy.sample(
            condition, known, **self.protocol["execution_solver"]
        )
        native_actions = policy.reference_actions(condition, known, steps=10)
        decoded = policy.output_transform({"actions": to_numpy(native_solve.value)[0]})[
            "actions"
        ]
        zero_condition, zero_provenance = policy.prepare_intervened(
            observation,
            digest(observation),
            self.entry["instruction"],
            text=TextEmbeddingIntervention(self.entry["instruction"], 0.0),
        )
        zero_solve = policy.sample(
            zero_condition, known, **self.protocol["execution_solver"]
        )
        errors = {
            "noise": error_metrics(known, inverse.value),
            "actions": error_metrics(reference.value, roundtrip.value),
            "native_parity": error_metrics(native_actions, decoded),
            "zero_embedding_hook_parity": error_metrics(
                native_solve.value, zero_solve.value
            ),
        }
        gate = self.protocol["numerical_gate"]
        passed = (
            errors["noise"]["max_abs"] <= gate["noise_max_abs"]
            and errors["actions"]["max_abs"] <= gate["action_max_abs"]
            and errors["native_parity"]["max_abs"] <= 1e-5
            and errors["zero_embedding_hook_parity"]["max_abs"] == 0.0
            and zero_condition.condition_id == condition.condition_id
        )
        self.known, self.recovered = (
            to_numpy(known).copy(),
            to_numpy(inverse.value).copy(),
        )
        self.report["initialization"] = {
            "passed": passed,
            "condition_id": condition.condition_id,
            "known_noise_sha256": digest(self.known),
            "recovered_noise_sha256": digest(self.recovered),
            "errors": errors,
            "velocity_evaluations": sum(
                solve.velocity_evaluations
                for solve in (reference, inverse, roundtrip, native_solve, zero_solve)
            )
            + 10,
            "wall_seconds_before_recording": time.perf_counter() - started,
            "note": "Native parity includes ten additional upstream velocity calls. Initialization velocity evaluations are separate; baseline policy and wall times include initialization, logging, and its archive synchronization.",
        }
        self.recorder.event(
            "inversion_initialization",
            observation=observation,
            known_noise=self.known,
            reference=reference.value,
            recovered_noise=self.recovered,
            roundtrip=roundtrip.value,
            native_actions=native_actions,
            adapter_actions=decoded,
            zero_embedding_hook=zero_provenance,
            action_spec=spec.as_dict(),
            **self.report["initialization"],
        )
        self.save()
        if not passed:
            raise ValueError(
                "Policy-reference inversion or native sampler numerical gate failed"
            )

    def rollout(self, mode, iteration, proposal=None):
        from .intervention_conditioning import TextEmbeddingIntervention
        from .intervention_rollout import run_rollout

        env, _, _ = self.create_env(self.entry["task_id"], self.entry["seed"])
        spec = ActionSpec.from_environment(
            env, self.policy.horizon, self.policy.action_dim
        )
        adapter = ActionAdapter(
            spec, self.policy.input_transform, self.policy.output_transform
        )
        self.spec = spec
        candidate_id = proposal["candidate_id"] if proposal else mode
        counter = {
            "velocity_evaluations": 0,
            "clipped_values": 0,
            "condition_preparation_seconds": 0.0,
        }
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
            modified = apply_vision(
                observation, proposal.get("vision", []) if proposal else []
            )
            language = proposal.get("language") if proposal else None
            if language is None:
                condition = self.policy.prepare(
                    modified, digest(observation), self.entry["instruction"]
                )
                conditioning = None
            else:
                condition, conditioning = self.policy.prepare_intervened(
                    modified,
                    digest(observation),
                    self.entry["instruction"],
                    text=TextEmbeddingIntervention(
                        guidance_prompt=self.entry["instruction"]
                        + "\nGuidance: "
                        + language["target_text"],
                        alpha=language["scale"],
                        max_relative_norm=self.protocol["language"][
                            "maximum_relative_norm"
                        ],
                    ),
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
                    perturb_noise(
                        self.recovered,
                        self.basis,
                        proposal.get("noise") if proposal else None,
                    )
                )
            generated = self.policy.sample(
                condition, latent, **self.protocol["execution_solver"]
            )
            actions, clipping = adapter.decode(generated.value, condition.state)
            counter["velocity_evaluations"] += generated.velocity_evaluations
            counter["clipped_values"] += clipping["count"]
            counter["condition_preparation_seconds"] += condition.preparation_seconds
            self.recorder.event(
                "candidate_generation",
                mode=mode,
                iteration=iteration,
                candidate_id=candidate_id,
                observation_step=step,
                observation=observation,
                modified_observation_sha256=digest(modified),
                condition_id=condition.condition_id,
                conditioning=conditioning,
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
            expected_reset=self.reset_audit,
            policy_image_size=self.policy.observation_image_size,
            video_path=self.directory / f"{mode}_{iteration}.mp4",
        )
        snapshots = result.pop("snapshots")
        if result["initial_success"]:
            raise ValueError(
                "Initially successful reset is not an intervention success"
            )
        if self.reset_audit is None:
            self.reset_audit = result["reset_audit"]
        attempt = {
            **result,
            **counter,
            "iteration": iteration,
            "candidate_id": candidate_id,
            "proposal": proposal,
            "rollout_executed": True,
            "status": "success"
            if result["success"]
            else "terminated"
            if result["terminated"]
            else "budget_exhausted",
        }
        self.recorder.event("attempt", attempt=attempt, snapshots=snapshots)
        return attempt, snapshots

    def run(self):
        from .intervention_agent import build_request, summarize_calls

        started = time.perf_counter()
        baseline, initial_snapshots = self.rollout("reversal_identity", 1)
        self.report["baseline"] = baseline
        self.report["controls"]["known_noise"], _ = self.rollout("known_noise", 1)
        self.save()
        self.report["controls"]["policy_fresh"], _ = self.rollout("policy_fresh", 1)
        self.save()
        for arm in self.protocol["arms"]:
            attempts, snapshots = [baseline], initial_snapshots
            incumbent = baseline["candidate_id"]
            response_log = self.directory / f"{arm}_provider.jsonl"
            client = self.client_type(
                model=self.protocol["astra"]["model"],
                response_log=response_log,
                timeout=self.protocol["astra"]["timeout_seconds"],
                sampling={
                    key: self.protocol["astra"][key]
                    for key in ("reasoning_effort", "max_completion_tokens")
                },
            )
            random_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.protocol["seed"],
                        int(digest(self.entry["episode_id"])[:8], 16),
                        3,
                    ]
                )
            )
            for iteration in range(2, self.budget + 1):
                if any(row["success"] for row in attempts) and not (
                    self.development and iteration == 2
                ):
                    break
                began = time.perf_counter()
                previous_incumbent = incumbent
                proposal = None
                try:
                    if arm == "random_noise":
                        proposal = {
                            "candidate_id": f"random_noise_{iteration}",
                            "language": None,
                            "vision": [],
                            "noise": random_noise_proposal(self.basis, random_rng),
                            "rationale": "Matched random direction and prespecified scale, without Astra feedback",
                        }
                    else:
                        request = build_request(
                            episode_id=self.entry["episode_id"],
                            iteration=iteration,
                            arm=arm,
                            task_instruction=self.entry["instruction"],
                            action_spec=self.spec.as_dict(),
                            observations=snapshots,
                            prior_candidates=[feedback_row(row) for row in attempts],
                            basis_id=self.basis_id,
                            incumbent_candidate_id=incumbent,
                            max_iterations=self.budget,
                        )
                        self.recorder.event("intervention_request", request=request)
                        proposal = client.propose(request)
                        incumbent = proposal["best_candidate_id"]
                    self.recorder.event(
                        "intervention_proposal",
                        arm=arm,
                        iteration=iteration,
                        proposal=proposal,
                    )
                    language = proposal.get("language")
                    if language and language["scale"] > 0:
                        guidance = (
                            self.entry["instruction"]
                            + "\nGuidance: "
                            + language["target_text"]
                        )
                        if (
                            self.policy.prompt_length(
                                guidance, snapshots[0]["observation"]
                            )
                            > self.policy.max_token_len
                        ):
                            raise ClientError(
                                "Guidance exceeds the frozen policy token budget"
                            )
                except ClientError as exc:
                    incumbent = previous_incumbent
                    attempt = {
                        "iteration": iteration,
                        "candidate_id": proposal["candidate_id"]
                        if proposal
                        else f"{arm}_error_{iteration}",
                        "success": False,
                        "rollout_executed": False,
                        "status": "proposal_error",
                        "error": str(exc)[:256],
                        "actions_executed": 0,
                        "velocity_evaluations": 0,
                        "wall_seconds": 0.0,
                        "proposal_seconds": time.perf_counter() - began,
                        "proposal": proposal,
                    }
                    self.recorder.event("proposal_error", arm=arm, attempt=attempt)
                else:
                    proposal_seconds = time.perf_counter() - began
                    attempt, snapshots = self.rollout(arm, iteration, proposal)
                    attempt["proposal_seconds"] = proposal_seconds
                attempts.append(attempt)
                attempt["cumulative_token_usage"] = (
                    summarize_calls(response_log)
                    if response_log.exists()
                    else summarize_calls([])
                )
                self.report["arms"][arm] = {
                    "attempts": attempts,
                    "summary": arm_summary(attempts, self.budget),
                    "token_usage": summarize_calls(response_log)
                    if response_log.exists()
                    else summarize_calls([]),
                    "status": "running",
                }
                self.save()
            self.report["arms"][arm] = {
                "attempts": attempts,
                "summary": arm_summary(attempts, self.budget),
                "token_usage": summarize_calls(response_log)
                if response_log.exists()
                else summarize_calls([]),
                "status": "complete",
            }
            arm_report = self.report["arms"][arm]
            first = arm_report["summary"]["first_success_attempt"]
            arm_report["tokens_to_first_success"] = (
                summarize_calls([])
                if first == 1
                else next(
                    row["cumulative_token_usage"]
                    for row in attempts
                    if row["iteration"] == first
                )
                if first is not None
                else None
            )
            arm_report["summary"][
                "standalone_velocity_evaluations_through_success_or_cap"
            ] = (
                arm_report["summary"]["velocity_evaluations_through_success_or_cap"]
                + self.report["initialization"]["velocity_evaluations"]
            )
            self.save()
        self.report["status"] = "complete"
        self.report["physical_wall_seconds"] = time.perf_counter() - started
        physical_attempts = [baseline, *self.report["controls"].values()] + [
            attempt
            for arm in self.report["arms"].values()
            for attempt in arm["attempts"][1:]
        ]
        self.report["physical_cost"] = {
            "rollouts": sum(row["rollout_executed"] for row in physical_attempts),
            "simulated_actions": sum(
                row["actions_executed"] for row in physical_attempts
            ),
            "velocity_evaluations": sum(
                row["velocity_evaluations"] for row in physical_attempts
            )
            + self.report["initialization"]["velocity_evaluations"],
            "token_usage": summarize_calls(
                [
                    json.loads(line)
                    for path in sorted(self.directory.glob("*_provider.jsonl"))
                    for line in path.read_text().splitlines()
                    if line
                ]
            ),
        }
        self.save()
        return self.report


def aggregate_reports(reports, protocol):
    """Descriptive paired pilot results; unsuccessful searches remain censored."""
    complete = [row for row in reports if row["status"] == "complete"]
    result = {
        "schema_version": "1.0",
        "cases": len(reports),
        "complete_cases": len(complete),
        "protocol_sha256": digest(protocol),
        "attempt_budget": protocol["attempt_budget"],
        "baseline_successes": sum(row["baseline"]["success"] for row in complete),
        "arms": {},
        "case_statuses": {row["episode_id"]: row["status"] for row in reports},
    }
    for arm in protocol["arms"]:
        rows = [row["arms"][arm] for row in complete]
        curves = [row["summary"]["success_by_attempt"] for row in rows]
        successful = [
            row["summary"]["first_success_attempt"]
            for row in rows
            if row["summary"]["first_success_attempt"] is not None
        ]
        result["arms"][arm] = {
            "cases": len(rows),
            "successes_by_attempt": [
                sum(curve[i] for curve in curves)
                for i in range(protocol["attempt_budget"])
            ],
            "successful_first_attempts": successful,
            "median_attempt_to_success_among_successes": float(np.median(successful))
            if successful
            else None,
            "censored_cases": len(rows) - len(successful),
            "candidate_rollouts_including_shared_baseline": sum(
                row["summary"]["candidate_rollouts"] for row in rows
            ),
            "proposal_failures": sum(
                row["summary"]["proposal_failures"] for row in rows
            ),
            "actions_through_success_or_cap": sum(
                row["summary"]["actions_through_success_or_cap"] for row in rows
            ),
            "velocity_evaluations_through_success_or_cap": sum(
                row["summary"]["velocity_evaluations_through_success_or_cap"]
                for row in rows
            ),
            "case_token_usage": {
                case["episode_id"]: case["arms"][arm]["token_usage"]
                for case in complete
            },
        }
    result["controls"] = {
        name: {
            "successes": sum(row["controls"][name]["success"] for row in complete),
            "cases": len(complete),
        }
        for name in ("policy_fresh", "known_noise")
    }
    return result
