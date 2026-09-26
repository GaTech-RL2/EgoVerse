"""Astra FRS and judgment-gated noise-policy learning on frozen task resets.

This module owns physical rollouts. Clients never see evaluation predicates;
the report retains them separately. Evaluation episodes never enter replay.
"""

import copy
import json
import time
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .astra_client import ClientError
from .flow import error_metrics, noise_statistics
from .intervention_rollout import run_rollout
from .intervention_search import write_json
from .records import Recorder, digest, to_numpy


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/frs_policy_improvement_v1.json"
    value = json.loads(Path(path).read_text())
    if (
        value["schema_version"] != "frs-policy-improvement-1.0"
        or value["execute_steps"] != 10
        or value["rounds"] != 3
        or value["action_budget"] != 300
        or value["evaluation_states"] != list(range(1, 11))
        or value["astra"]["call_interval"] != 10
        or value["solver"] != {"solver": "euler", "steps": 10, "time_power": 1.0}
    ):
        raise ValueError("Unexpected FRS protocol; version changes explicitly")
    return value


def stream_rng(entry, step, stream):
    """Common per-reset, per-step noise; method and retry never change it."""
    return np.random.default_rng(
        np.random.SeedSequence(
            [entry["seed"], int(digest(entry["episode_id"])[:8], 16), step, stream]
        )
    )


def first_success(attempts):
    index = next((i for i, row in enumerate(attempts) if row["success"]), None)
    return {
        "first_success_round": index,
        "censored": index is None,
        "observed_rounds": len(attempts) - 1,
        "success_by_round": [
            any(row["success"] for row in attempts[: i + 1])
            for i in range(len(attempts))
        ],
    }


def visible_rollout(result):
    """Exact data boundary: no simulator outcome, termination or hidden state."""
    return {
        "attempt_id": result["attempt_id"],
        "snapshots": copy.deepcopy(result["snapshots"]),
        "executed_actions": np.asarray(result["executed_actions"]).copy(),
    }


def compact_arrays(value):
    """Keep review summaries small; full arrays live in the Recorder event."""
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": digest(value),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: compact_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_arrays(item) for item in value]
    return value


class FRSTaskExperiment:
    def __init__(
        self,
        policy,
        create_env,
        benchmark,
        entries,
        protocol,
        directory,
        *,
        development=False,
        progress=None,
    ):
        from .frs_agent import FRSClient

        self.policy, self.create_env, self.benchmark = policy, create_env, benchmark
        self.entries = {row["initial_state_id"]: row for row in entries}
        self.protocol, self.development = protocol, development
        self.directory = Path(directory)
        self.recorder = Recorder(self.directory)
        self.progress = progress or (lambda: None)
        self.reset_audits = {}
        self.parity_checked = False
        self.request_index = 0
        self.physical_runs = []
        self.results_by_id = {}
        settings = protocol["astra"]
        self.client = FRSClient(
            model=settings["model"],
            response_log=self.directory / "provider.jsonl",
            reasoning_effort=settings["reasoning_effort"],
            max_completion_tokens=settings["max_completion_tokens"],
            timeout=settings["timeout_seconds"],
        )
        task = self.entries[0]
        if len({row["task_id"] for row in entries}) != 1:
            raise ValueError("Each adaptation experiment must have one task")
        if set(self.entries) != {0, *protocol["evaluation_states"]}:
            raise ValueError("Every task requires adaptation0 and evaluation1..10")
        self.report = {
            "schema_version": "frs-task-1.0",
            "status": "running",
            "development": development,
            "suite": benchmark.suite,
            "task_id": task["task_id"],
            "instruction": task["instruction"],
            "protocol_sha256": digest(protocol),
            "entries_sha256": digest(entries),
            "checkpoint": policy.metadata,
            "adaptation": {},
            "evaluation": [],
            "physical_rollouts": self.physical_runs,
        }
        self.recorder.event(
            "task", entries=entries, protocol=protocol, checkpoint=policy.metadata
        )
        self.save()

    def save(self):
        write_json(self.directory / "summary.json", self.report)
        self.progress()

    def identity(self, entry, attempt_id, step):
        self.request_index += 1
        return {
            "episode_id": entry["episode_id"],
            "attempt_id": attempt_id,
            "request_index": self.request_index,
            "observation_step": step,
            "target_task": entry["instruction"],
        }

    def request(self, request):
        before = len(self.client.records)
        response, error = None, None
        try:
            response = self.client.propose(request)
        except ClientError as exc:
            error = str(exc)
        if len(self.client.records) != before + 1:
            raise RuntimeError("Each Astra request must have exactly one cost record")
        self.recorder.event(
            "astra_decision",
            request_index=self.request_index,
            role=request["role"],
            request_fingerprint=request["request_fingerprint"],
            response=response,
            error=error,
            provider_record_index=before,
        )
        if self.request_index % 5 == 0:
            self.save()
        return response

    def rollout(self, method, entry, *, round_index=0, rules=(), actor=None):
        from .frs_agent import action_edit_request, direction_request, summarize_calls
        from .frs_guide import gripper_guide
        from .frs_noise_policy import make_training_sample
        from .frs_operators import (
            directional_reference,
            edit_native,
            repeated_gaussian_noise,
            resample_padding_noise,
        )

        allowed = set(self.protocol["evaluation_methods"]) | set(
            self.protocol["adaptation_methods"]
        )
        if method not in allowed:
            raise ValueError("Unknown FRS rollout method")
        attempt_id = f"{method}_state{entry['initial_state_id']}_round{round_index}"
        if attempt_id in self.results_by_id:
            raise ValueError("Physical rollout IDs cannot be repeated")
        policy = self.policy
        env, task, _ = self.create_env(entry["task_id"], entry["seed"])
        if task.language != entry["instruction"]:
            env.close()
            raise ValueError("Task instruction differs from the frozen reset")
        spec = ActionSpec.from_environment(env, policy.horizon, policy.action_dim)
        adapter = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        is_paper = method in ("astra_direction_direct", "astra_frs")
        is_loop = method.startswith("critique_frs_")
        repeated = (
            method == "native_repeated_noise" or is_loop or method == "learned_noise"
        )
        calls_begin = len(self.client.records)
        generated_actions, samples, decisions = [], [], []
        counters = {
            "velocity_evaluations": 0,
            "interventions": 0,
            "deferred": 0,
            "rejected": 0,
            "clipped_actions": 0,
            "auxiliary_inferences": 0,
            "auxiliary_request_seconds": 0.0,
        }
        self.recorder.event(
            "rollout_start",
            attempt_id=attempt_id,
            method=method,
            episode_id=entry["episode_id"],
            round_index=round_index,
            reset_entry_sha256=digest(entry),
            rules=list(rules),
            action_spec=spec,
            evaluation=entry["initial_state_id"] != 0,
        )

        def action_policy(observation, step):
            condition = policy.prepare(
                observation, digest(observation), entry["instruction"]
            )
            base_rng = stream_rng(entry, step, 0)
            base_noise = (
                repeated_gaussian_noise(base_rng)["noise"]
                if repeated
                else to_numpy(policy.noise(base_rng))
            )
            actor_receipt = None
            if actor is not None:
                actor_started = time.perf_counter()
                predicted = actor.predict(
                    observation, stream_rng(entry, step, 1), base_noise
                )
                noise = predicted["noise"]
                actor_receipt = predicted["receipt"]
                counters["auxiliary_request_seconds"] += (
                    time.perf_counter() - actor_started
                )
                counters["auxiliary_inferences"] += int(
                    actor_receipt.get("auxiliary_forward_performed", False)
                )
            else:
                noise = base_noise.copy()
            native = policy.sample(
                condition, policy.tensor(noise), **self.protocol["solver"]
            )
            counters["velocity_evaluations"] += native.velocity_evaluations
            native_actions, native_clipping = adapter.decode(
                native.value, condition.state
            )
            if not self.parity_checked:
                upstream = policy.reference_actions(
                    condition, policy.tensor(noise), steps=10
                )
                decoded_unclipped = policy.output_transform(
                    {"actions": to_numpy(native.value)[0]}
                )["actions"]
                parity = error_metrics(upstream, decoded_unclipped)
                self.recorder.event(
                    "native_parity",
                    attempt_id=attempt_id,
                    step=step,
                    errors=parity,
                    noise=noise,
                    native=upstream,
                    decoded=decoded_unclipped,
                )
                counters["velocity_evaluations"] += 10
                if parity["max_abs"] > 1e-5:
                    raise ValueError("Native pi05 action parity failed")
                from PIL import Image

                guide_probe, guide_probe_receipt = gripper_guide(observation, env)
                Image.fromarray(guide_probe).save(self.directory / "guide_probe.png")
                self.recorder.event(
                    "guide_probe",
                    raw=observation["observation/image"],
                    guide=guide_probe,
                    receipt=guide_probe_receipt,
                )
                self.parity_checked = True
            proposal, guide, guide_receipt = None, None, None
            if is_paper:
                guide, guide_receipt = gripper_guide(observation, env)
                request = direction_request(
                    **self.identity(entry, attempt_id, step),
                    external_image=observation["observation/image"],
                    guide_image=guide,
                )
                proposal = self.request(request)
            elif is_loop:
                request = action_edit_request(
                    **self.identity(entry, attempt_id, step),
                    observation=observation,
                    native_actions=native_actions,
                    rules=list(rules),
                    previous_decisions=decisions[-2:],
                    action_spec=spec.as_dict(),
                )
                proposal = self.request(request)
            if is_paper or is_loop:
                decisions.append(
                    {
                        "request_index": self.request_index,
                        "observation_step": step,
                        "proposal": proposal,
                        "accepted": proposal is not None,
                        "error": None if proposal is not None else "call_rejected",
                    }
                )
                if proposal is None:
                    counters["rejected"] += 1
            reference = inverse = padding = None
            executed_noise, generated, actions, clipping = (
                noise,
                native.value,
                native_actions,
                native_clipping,
            )
            kind = "native_defer"
            steer = proposal is not None and (
                (is_paper and not proposal["fine"])
                or (is_loop and proposal["mode"] == "edit")
            )
            if steer:
                counters["interventions"] += 1
                reference = (
                    directional_reference(
                        adapter,
                        observation,
                        coords=[
                            x * sign
                            for x, sign in zip(
                                proposal["coords"],
                                guide_receipt["camera_to_controller_signs"],
                                strict=True,
                            )
                        ],
                        motion_amount=proposal["motion_amount"],
                    )
                    if is_paper
                    else edit_native(
                        adapter,
                        observation,
                        native_actions,
                        delta_xyz=proposal["delta_xyz"],
                        apply_steps=proposal["apply_steps"],
                        gripper=proposal["gripper"],
                    )
                )
                if method == "astra_direction_direct":
                    generated = policy.tensor(reference["target_model_actions"])
                    actions, clipping = adapter.decode(generated, condition.state)
                    executed_noise = None
                    kind = "direct_reference"
                else:
                    inverse = policy.invert(
                        condition,
                        policy.tensor(reference["target_model_actions"]),
                        **self.protocol["solver"],
                    )
                    padding = resample_padding_noise(
                        to_numpy(inverse.value),
                        stream_rng(entry, step, 2),
                        repeat_mean=is_loop,
                    )
                    executed_noise = padding["noise"]
                    forward = policy.sample(
                        condition,
                        policy.tensor(executed_noise),
                        **self.protocol["solver"],
                    )
                    generated = forward.value
                    counters["velocity_evaluations"] += (
                        inverse.velocity_evaluations + forward.velocity_evaluations
                    )
                    actions, clipping = adapter.decode(generated, condition.state)
                    kind = "frs_edit"
            elif is_paper or is_loop:
                counters["deferred"] += 1
            counters["clipped_actions"] += clipping["count"]
            if repeated:
                samples.append(
                    make_training_sample(
                        observation,
                        executed_noise,
                        kind=kind,
                        observation_step=step,
                        source_id=f"{attempt_id}:step{step}",
                    )
                )
            self.recorder.event(
                "generation",
                attempt_id=attempt_id,
                step=step,
                method=method,
                observation=observation,
                observation_sha256=digest(observation),
                condition_id=condition.condition_id,
                guide=guide,
                guide_receipt=guide_receipt,
                base_noise=base_noise,
                predicted_noise=noise,
                actor_receipt=actor_receipt,
                native_model_actions=native.value,
                native_actions=native_actions,
                native_clipping=native_clipping,
                proposal=proposal,
                reference=reference,
                reversed_noise=to_numpy(inverse.value) if inverse else None,
                noise_transform=padding,
                executed_noise=executed_noise,
                generated_model_actions=generated,
                actions=actions,
                clipping=clipping,
                generation_kind=kind,
                reconstruction=(
                    error_metrics(reference["target_model_actions"], generated)
                    if inverse
                    else None
                ),
                noise_statistics=noise_statistics(executed_noise)
                if executed_noise is not None
                else None,
            )
            generated_actions.extend(actions[: self.protocol["execute_steps"]].copy())
            return actions

        started = time.perf_counter()
        result = run_rollout(
            env,
            entry,
            self.benchmark,
            action_policy,
            execute_steps=self.protocol["execute_steps"],
            action_budget=self.protocol["action_budget"],
            expected_reset=self.reset_audits.get(entry["episode_id"]),
            policy_image_size=224,
            video_path=self.directory / f"{attempt_id}.mp4",
        )
        self.reset_audits.setdefault(entry["episode_id"], result["reset_audit"])
        executed = np.asarray(generated_actions, dtype=np.float32).reshape(-1, 7)[
            : result["actions_executed"]
        ]
        if executed.shape != (result["actions_executed"], 7):
            raise ValueError("Executed actions do not match the physical rollout")
        result.update(
            attempt_id=attempt_id,
            method=method,
            round_index=round_index,
            evaluation=entry["initial_state_id"] != 0,
            counters=counters,
            wall_seconds=time.perf_counter() - started,
            executed_actions=executed,
            provider_usage=summarize_calls(self.client.records[calls_begin:]),
            provider_record_indexes=list(range(calls_begin, len(self.client.records))),
        )
        self.recorder.event("rollout_end", result=result)
        compact = {
            k: v
            for k, v in result.items()
            if k not in ("snapshots", "executed_actions", "video_path")
        }
        compact["physical_run_id"] = f"{entry['episode_id']}:{attempt_id}"
        self.physical_runs.append(compact)
        self.results_by_id[attempt_id] = result
        self.save()
        return result, samples

    def evaluate(self, method, *, round_index=3, rules=(), actor=None):
        states = self.protocol["evaluation_states"]
        if self.development:
            states = states[:1]
        for state in states:
            result, _ = self.rollout(
                method,
                self.entries[state],
                round_index=round_index,
                rules=rules,
                actor=actor,
            )
            self.report["evaluation"].append(
                {
                    "method": method,
                    "round_index": round_index,
                    "physical_run_id": self.physical_runs[-1]["physical_run_id"],
                    "attempt_id": result["attempt_id"],
                    "episode_id": result["episode_id"],
                    "success": result["success"],
                }
            )
        self.save()

    def adapt(self, method, baseline, *, actor=None):
        from .frs_agent import critique_request, judge_request

        entry = self.entries[0]
        latest = incumbent = baseline
        best_rules, rounds, attempts = [], [], [baseline]
        self.report["adaptation"][method] = {
            "rounds": rounds,
            "baseline_attempt_id": baseline["attempt_id"],
        }
        for index in range(1, self.protocol["rounds"] + 1):
            critique = self.request(
                critique_request(
                    **self.identity(
                        entry, latest["attempt_id"], latest["actions_executed"]
                    ),
                    rollout=visible_rollout(latest),
                )
            )
            rules = critique["rules"] if critique else []
            candidate, samples = self.rollout(
                method, entry, round_index=index, rules=rules, actor=actor
            )
            comparison = self.request(
                judge_request(
                    **self.identity(
                        entry, candidate["attempt_id"], candidate["actions_executed"]
                    ),
                    incumbent=visible_rollout(incumbent),
                    candidate=visible_rollout(candidate),
                )
            )
            promoted = (
                critique is not None
                and comparison is not None
                and comparison["verdict"] == "better"
            )
            update = None
            if promoted and actor is not None:
                update = actor.fit_accepted_rollout(
                    candidate["attempt_id"],
                    samples,
                    judge_verdict="better",
                    judge_sha256=digest(comparison),
                )
            checkpoint = None
            if actor is not None:
                checkpoint_dir = self.directory / f"noise_policy_round{index}"
                checkpoint = actor.save(checkpoint_dir)
            record = {
                "round_index": index,
                "critique": critique,
                "rules": rules,
                "candidate_attempt_id": candidate["attempt_id"],
                "incumbent_attempt_id": incumbent["attempt_id"],
                "judge": comparison,
                "promoted": promoted,
                "update": update,
                "checkpoint": checkpoint,
                "incumbent_success_evaluation_only": incumbent["success"],
                "candidate_success_evaluation_only": candidate["success"],
            }
            self.recorder.event("adaptation_round", method=method, **record)
            rounds.append(compact_arrays(record))
            if promoted:
                incumbent, best_rules = candidate, copy.deepcopy(rules)
            latest = candidate
            attempts.append(candidate)
            self.report["adaptation"][method]["first_success"] = first_success(attempts)
            self.save()
            if actor is not None:
                self.evaluate("learned_noise", round_index=index, actor=actor)
        self.report["adaptation"][method]["final_rules"] = best_rules
        return best_rules

    def run(self):
        from .frs_agent import summarize_calls
        from .frs_noise_policy import AuxiliaryNoisePolicy

        # Independent initializations/arms never consume each other's feedback.
        baseline, _ = self.rollout("native_repeated_noise", self.entries[0])
        rules = self.adapt("critique_frs_no_learning", baseline)
        actor = AuxiliaryNoisePolicy(
            task_id=f"{self.benchmark.suite}:{self.entries[0]['task_id']}",
            seed=int(digest(self.entries[0]["episode_id"])[:8], 16),
            device="cuda",
        )
        self.adapt("critique_frs_learning", baseline, actor=actor)
        for method in self.protocol["evaluation_methods"]:
            if method != "learned_noise":
                self.evaluate(
                    method, rules=rules if method == "critique_frs_no_learning" else ()
                )
        self.report["provider_usage"] = summarize_calls(self.client.records)
        updates = [
            row["update"]
            for arm in self.report["adaptation"].values()
            for row in arm["rounds"]
            if row["update"] and row["update"]["status"] == "updated"
        ]
        self.report["physical_cost"] = {
            "rollouts": len(self.physical_runs),
            "actions": sum(row["actions_executed"] for row in self.physical_runs),
            "velocity_evaluations": sum(
                row["counters"]["velocity_evaluations"] for row in self.physical_runs
            ),
            "rollout_wall_seconds": sum(
                row["wall_seconds"] for row in self.physical_runs
            ),
            "provider_calls": self.report["provider_usage"]["provider_calls"],
            "client_attempts": len(self.client.records),
            "accepted_policy_updates": len(updates),
            "optimizer_steps": sum(row.get("optimizer_updates", 0) for row in updates),
            "training_seconds": sum(row.get("wall_seconds", 0.0) for row in updates),
            "auxiliary_inferences": sum(
                row["counters"]["auxiliary_inferences"] for row in self.physical_runs
            ),
            "auxiliary_request_seconds": sum(
                row["counters"]["auxiliary_request_seconds"]
                for row in self.physical_runs
            ),
        }
        self.report["status"] = "complete"
        self.save()
        return self.report
