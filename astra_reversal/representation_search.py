"""Matched-reset, bounded representation interventions on a frozen native policy.

This module measures assisted behavior. It performs no parameter updates and
never uses another arm's feedback to select an intervention.
"""

import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .action_adapter import ActionAdapter, ActionSpec
from .astra_client import ClientError
from .flow import error_metrics
from .interpolation_catalog import donor_catalog
from .intervention_rollout import CAMERAS, run_rollout
from .intervention_search import write_json
from .provider_stop import ProviderUnavailable, require_provider_available
from .records import Recorder, digest, file_sha256, to_numpy

ARMS = (
    "native_retry",
    "random_tli",
    "random_vei",
    "random_vli",
    "astra_tei",
    "astra_tli",
    "astra_vei",
    "astra_vli",
    "astra_tli_vli",
    "astra_pixel_blend",
)
VISION_ARMS = ("native_retry", "random_vei", "random_vli", "astra_vei", "astra_vli")


def load_protocol(path=None):
    path = path or Path(__file__).parent / "configs/representation_steering_v1.json"
    value = json.loads(Path(path).read_text())
    vision_only = value["schema_version"] == "vision-representation-screen-1.0"
    if (
        value["schema_version"]
        not in ("representation-steering-1.0", "vision-representation-screen-1.0")
        or value["arms"] != list(VISION_ARMS if vision_only else ARMS)
        or value["revisions"] != 2
        or value["execute_steps"] != 5
        or value["action_budget"] != 300
        or value["astra"]["call_interval"] != 25
        or value["astra"]["max_calls_per_rollout"] != 12
        or value["learning"]["enabled"]
    ):
        raise ValueError("Change the protocol version before changing the experiment")
    if vision_only and not value["astra"].get("stop_on_provider_unavailable"):
        raise ValueError("The visual screen requires the provider stop rule")
    return value


def keyed_rng(entry, revision, step, stream=0):
    """Method-independent policy noise; separate intervention-choice stream."""
    identity = int(digest(entry["episode_id"])[:16], 16)
    return np.random.default_rng(
        np.random.SeedSequence([entry["seed"], identity, revision, step, stream])
    )


def random_choice(mode, rng, sources, donors):
    language = vision = None
    if mode == "tli":
        pair = rng.choice(sources, size=2, replace=False)
        language = {
            "source_a_id": str(pair[0]),
            "source_b_id": str(pair[1]),
            "alpha": float(rng.uniform(0.0, 1.0)),
        }
    elif mode in ("vei", "vli"):
        vision = {
            "donor_id": str(rng.choice(donors)),
            "alpha": float(rng.uniform(0.0, 1.0)),
        }
    else:
        raise ValueError("Unsupported random representation control")
    return {"mode": "interpolate", "language": language, "vision": vision}


def feedback(attempt):
    return {
        "attempt_id": attempt["attempt_id"],
        "success": attempt["success"],
        "executed_actions": attempt["actions_executed"],
        "termination": attempt["status"],
        "error": None,
    }


def summarize_attempts(attempts, revisions):
    from .representation_agent import summarize_calls

    first = next((row for row in attempts if row["success"]), None)
    prefix = [
        row for row in attempts if first is None or row["revision"] <= first["revision"]
    ]
    calls = [call for row in attempts for call in row.get("provider_records", [])]
    return {
        "success": first is not None,
        "first_success_revision": first["revision"] if first else None,
        "censored": first is None,
        "success_by_revision": [
            any(row["success"] and row["revision"] <= r for row in attempts)
            for r in range(revisions + 1)
        ],
        "physical_attempt_ids": [row["attempt_id"] for row in attempts],
        "actions_through_success_or_cap": sum(
            row["actions_executed"] for row in prefix
        ),
        "decisions_through_success_or_cap": sum(
            len(row["decisions"]) for row in prefix
        ),
        "provider": summarize_calls(calls),
        "provider_through_success_or_cap": summarize_calls(
            [c for row in prefix for c in row.get("provider_records", [])]
        ),
        "development_extra_rollouts": len(attempts) - len(prefix),
    }


class RepresentationSearch:
    def __init__(
        self,
        policy,
        create_env,
        benchmark,
        entry,
        protocol,
        directory,
        *,
        library,
        banks,
        development=False,
        progress=None,
    ):
        self.policy, self.create_env, self.benchmark = policy, create_env, benchmark
        self.entry, self.protocol = entry, protocol
        self.directory = Path(directory)
        self.recorder = Recorder(self.directory)
        self.library, self.banks, self.development = library, banks, development
        self.progress = progress or (lambda: None)
        self.sources = {r["source_id"]: r["prompt"] for r in donor_catalog()}
        self.donors = [r["donor_id"] for r in library.catalog()]
        self.vision_cache = OrderedDict()
        self.reset_audit = None
        self.parity_checked = False
        self.report = {
            "schema_version": protocol["schema_version"],
            "status": "running",
            "episode_id": entry["episode_id"],
            "suite": entry["suite"],
            "task_id": entry["task_id"],
            "instruction": entry["instruction"],
            "seed": entry["seed"],
            "initial_state_id": entry["initial_state_id"],
            "development": development,
            "protocol_sha256": digest(protocol),
            "reset_entry_sha256": digest(entry),
            "image_library_id": library.library_id,
            "physical_rollouts": [],
            "arms": {},
            "checks": {},
        }
        self.save()

    def save(self):
        write_json(self.directory / "summary.json", self.report)

    def _condition(self, observation, choice, mode, counters):
        policy, prompt = self.policy, self.entry["instruction"]
        observation_id = digest(observation)
        if choice is None or choice["mode"] == "native":
            return policy.prepare(observation, observation_id, prompt), None
        language, vision = choice["language"], choice["vision"]
        if mode in ("tei", "tli"):
            a, b = language["source_a_id"], language["source_b_id"]
            return policy.prepare_interpolated(
                observation,
                observation_id,
                prompt,
                source_prompts=(self.sources[a], self.sources[b]),
                alpha=language["alpha"],
                operator=mode,
                text_latents={"a": self.banks[a], "b": self.banks[b]}
                if mode == "tli"
                else None,
            )
        if mode == "pixel_blend":
            alpha = vision["alpha"]
            modified = copy.deepcopy(observation)
            for camera in CAMERAS:
                donor = self.library.resolve(vision["donor_id"], camera)
                modified[camera] = (
                    np.rint(
                        (1 - alpha) * observation[camera].astype(np.float64)
                        + alpha * donor.pixels.astype(np.float64)
                    )
                    .clip(0, 255)
                    .astype(np.uint8)
                )
            return policy.prepare(modified, observation_id, prompt), {
                "operator": mode,
                "vision": vision,
                "has_effect": digest(modified) != observation_id,
                "modified_observation_sha256": digest(modified),
            }
        from .vision_interpolation import (
            capture_vision_bank,
            prepare_vision_interpolated,
        )

        use_layers = mode in ("vli", "tli_vli")
        bank = None
        if vision["alpha"] != 0:
            key = (prompt, vision["donor_id"], use_layers)
            if key not in self.vision_cache:
                started = time.perf_counter()
                pair = {c: self.library.resolve(vision["donor_id"], c) for c in CAMERAS}
                self.vision_cache[key] = capture_vision_bank(
                    policy,
                    pair,
                    prompt=prompt,
                    include_layers=use_layers,
                )
                counters["donor_capture_seconds"] += time.perf_counter() - started
                counters["donor_captures"] += 1
                while len(self.vision_cache) > 3:
                    self.vision_cache.popitem(last=False)
            self.vision_cache.move_to_end(key)
            bank = self.vision_cache[key]
        kwargs = {}
        if mode == "tli_vli":
            a, b = language["source_a_id"], language["source_b_id"]
            kwargs = {
                "source_prompts": (self.sources[a], self.sources[b]),
                "text_latents": {"a": self.banks[a], "b": self.banks[b]},
                "language_alpha": language["alpha"],
            }
        return prepare_vision_interpolated(
            policy,
            observation,
            observation_id,
            prompt,
            bank=bank,
            alpha=vision["alpha"],
            operator="vli" if use_layers else "vei",
            **kwargs,
        )

    def rollout(self, arm, revision, previous=None, history=()):
        from .representation_agent import RepresentationClient, build_request

        env, _, _ = self.create_env(self.entry["task_id"], self.entry["seed"])
        spec = ActionSpec.from_environment(
            env, self.policy.horizon, self.policy.action_dim
        )
        adapter = ActionAdapter(
            spec, self.policy.input_transform, self.policy.output_transform
        )
        attempt_id = f"{arm}_revision{revision}"
        mode = arm.removeprefix("astra_").removeprefix("random_")
        settings = self.protocol["astra"]
        client = None
        if arm.startswith("astra_"):
            client = RepresentationClient(
                model=settings["model"],
                response_log=self.directory / "provider.jsonl",
                reasoning_effort=settings["reasoning_effort"],
                max_completion_tokens=settings["max_completion_tokens"],
                timeout=settings["timeout_seconds"],
            )
        observations, decisions, generated = [], [], []
        active = None
        counters = {
            "velocity_evaluations": 0,
            "parity_velocity_evaluations": 0,
            "probe_velocity_evaluations": 0,
            "donor_captures": 0,
            "donor_capture_seconds": 0.0,
            "condition_seconds": 0.0,
            "clipped_predicted_values": 0,
        }
        self.recorder.event(
            "representation_rollout_start",
            arm=arm,
            revision=revision,
            attempt_id=attempt_id,
            entry=self.entry,
            action_spec=spec,
        )

        def act(observation, step):
            nonlocal active
            observations.append(
                {
                    "label": f"step_{step}",
                    "step": step,
                    "observation": copy.deepcopy(observation),
                }
            )
            del observations[: -settings["max_current_snapshots"]]
            if step % settings["call_interval"] == 0:
                if client is not None:
                    request = build_request(
                        episode_id=self.entry["episode_id"],
                        attempt_id=attempt_id,
                        decision_index=len(decisions) + 1,
                        observation_step=step,
                        representation_mode=mode,
                        target_task=self.entry["instruction"],
                        source_catalog=donor_catalog(),
                        donor_catalog=self.library.catalog(),
                        contact_sheets=self.library.contact_sheets(),
                        observations=observations,
                        previous_decisions=decisions[-2:],
                        previous_attempt=previous,
                        action_spec=spec.as_dict(),
                        completed_rollout_feedback=[feedback(row) for row in history],
                        call_interval=settings["call_interval"],
                        execute_steps=self.protocol["execute_steps"],
                        max_calls=settings["max_calls_per_rollout"],
                        action_budget=self.protocol["action_budget"],
                    )
                    self.recorder.event("representation_request", request=request)
                    proposal, error = None, None
                    try:
                        proposal = client.propose(request)
                    except ClientError as exc:
                        error = str(exc)
                    active = proposal  # Failed calls clear old choices immediately.
                    decisions.append(
                        {
                            "decision_index": len(decisions) + 1,
                            "observation_step": step,
                            "proposal": proposal,
                            "accepted": proposal is not None,
                            "error": error,
                        }
                    )
                    self.recorder.event(
                        "representation_decision",
                        attempt_id=attempt_id,
                        decision=decisions[-1],
                    )
                    self.report["live"] = {
                        "attempt_id": attempt_id,
                        "observation_step": step,
                        "decisions": len(decisions),
                    }
                    self.save()
                    self.progress()
                    if settings.get("stop_on_provider_unavailable"):
                        try:
                            require_provider_available(client.records[-1])
                        except ProviderUnavailable as exc:
                            self.report.update(
                                status="provider_unavailable", provider_stop=exc.receipt
                            )
                            self.save()
                            self.progress()
                            raise
                elif arm.startswith("random_"):
                    active = random_choice(
                        mode,
                        keyed_rng(self.entry, revision, step, 1),
                        list(self.sources),
                        self.donors,
                    )
                    decisions.append(
                        {
                            "decision_index": len(decisions) + 1,
                            "observation_step": step,
                            "proposal": active,
                            "accepted": True,
                            "error": None,
                        }
                    )
                    self.recorder.event(
                        "random_representation_decision",
                        attempt_id=attempt_id,
                        decision=decisions[-1],
                    )
            condition, provenance = self._condition(observation, active, mode, counters)
            noise = self.policy.noise(keyed_rng(self.entry, revision, step))
            sample = self.policy.sample(condition, noise, **self.protocol["solver"])
            actions, clipping = adapter.decode(sample.value, condition.state)
            counters["velocity_evaluations"] += sample.velocity_evaluations
            counters["condition_seconds"] += condition.preparation_seconds
            counters["clipped_predicted_values"] += clipping["count"]
            if not self.parity_checked:
                if active is not None:
                    raise RuntimeError(
                        "Native parity must be checked on the shared baseline"
                    )
                reference = self.policy.reference_actions(condition, noise, steps=10)
                decoded = self.policy.output_transform(
                    {"actions": to_numpy(sample.value)[0]}
                )["actions"]
                errors = error_metrics(reference, decoded)
                counters["parity_velocity_evaluations"] += 10
                self.report["checks"]["native_parity"] = errors
                self.recorder.event(
                    "native_parity", errors=errors, reference=reference, decoded=decoded
                )
                if errors["max_abs"] > 1e-5:
                    raise RuntimeError("Native action sampler parity failed")
                from .vision_interpolation import weighted_vision_probe

                source_ids = list(self.sources)[:2]
                probe = weighted_vision_probe(
                    self.policy,
                    observation,
                    self.entry["instruction"],
                    {c: self.library.resolve(self.donors[2], c) for c in CAMERAS},
                    source_prompts=tuple(self.sources[s] for s in source_ids),
                    text_latents={
                        "a": self.banks[source_ids[0]],
                        "b": self.banks[source_ids[1]],
                    },
                )
                self.report["checks"]["weighted_vision"] = probe
                counters["probe_velocity_evaluations"] += probe["velocity_evaluations"]
                self.recorder.event("weighted_vision_probe", probe=probe)
                if probe["status"] != "passed":
                    raise RuntimeError("Weighted vision intervention checks failed")
                self.parity_checked = True
            generated.append(
                {
                    "step": step,
                    "choice": copy.deepcopy(active),
                    "provider_failure_fallback": bool(
                        client is not None
                        and decisions
                        and not decisions[-1]["accepted"]
                    ),
                    "has_effect": bool(
                        provenance and provenance.get("has_effect", False)
                    ),
                }
            )
            self.recorder.event(
                "representation_generation",
                attempt_id=attempt_id,
                observation_step=step,
                observation=observation,
                active_intervention=active,
                condition_id=condition.condition_id,
                provenance=provenance,
                noise=noise,
                generated_actions=sample.value,
                controller_actions=actions,
                clipping=clipping,
                velocity_evaluations=sample.velocity_evaluations,
                flow_seconds=sample.latency_seconds,
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
            raise ValueError("Initially successful reset is not a rescue")
        if self.reset_audit is None:
            self.reset_audit = result["reset_audit"]
        result.update(counters)
        attribution = {
            "accepted_decision_actions": 0,
            "explicit_native_actions": 0,
            "nonzero_intervention_actions": 0,
            "provider_failure_fallback_actions": 0,
        }
        accepted_ids = set()
        for row in generated:
            count = min(
                self.protocol["execute_steps"], result["actions_executed"] - row["step"]
            )
            if count <= 0:
                raise RuntimeError("A recorded generation had no executed actions")
            choice = row["choice"]
            if client is not None and choice is not None:
                accepted_ids.add(choice["decision_id"])
                attribution["accepted_decision_actions"] += count
                if choice["mode"] == "native":
                    attribution["explicit_native_actions"] += count
            attribution["nonzero_intervention_actions"] += count * row["has_effect"]
            attribution["provider_failure_fallback_actions"] += (
                count * row["provider_failure_fallback"]
            )
        result.update(attribution)
        result["accepted_decisions_executed"] = len(accepted_ids)
        result.update(
            {
                "attempt_id": attempt_id,
                "arm": arm,
                "revision": revision,
                "status": "success"
                if result["success"]
                else "terminated"
                if result["terminated"]
                else "budget_exhausted",
                "decisions": decisions,
                "provider_records": list(client.records) if client else [],
                "generations": generated,
                "video_sha256": file_sha256(result["video_path"]),
                "video_fps": 20,
            }
        )
        self.recorder.event(
            "representation_rollout_complete", attempt=result, snapshots=snapshots
        )
        self.report["physical_rollouts"].append(result)
        self.save()
        self.progress()
        previous = {
            "feedback": feedback(result),
            "decisions": decisions[-2:],
            "snapshots": snapshots,
        }
        return result, previous

    def run(self):
        baseline, baseline_feedback = self.rollout("native", 0)
        self.report["baseline"] = baseline
        for arm in self.protocol["arms"]:
            attempts, previous = [baseline], baseline_feedback
            for revision in range(1, self.protocol["revisions"] + 1):
                if any(row["success"] for row in attempts) and not (
                    self.development and revision == 1
                ):
                    break
                result, previous = self.rollout(arm, revision, previous, attempts)
                attempts.append(result)
            self.report["arms"][arm] = summarize_attempts(
                attempts, self.protocol["revisions"]
            )
            self.save()
            self.progress()
        from .representation_agent import summarize_calls

        rows = self.report["physical_rollouts"]
        integration = {}
        for arm in self.protocol["arms"]:
            if not arm.startswith("astra_"):
                continue
            selected = [row for row in rows if row["arm"] == arm]
            calls = [call for row in selected for call in row["provider_records"]]
            integration[arm] = {
                "physical_calls": sum(
                    bool(call.get("provider_call")) for call in calls
                ),
                "accepted_calls": sum(bool(call.get("accepted")) for call in calls),
                "accepted_decisions_executed": sum(
                    row.get("accepted_decisions_executed", 0) for row in selected
                ),
                "nonzero_intervention_actions": sum(
                    row.get("nonzero_intervention_actions", 0) for row in selected
                ),
                "provider_failure_fallback_actions": sum(
                    row.get("provider_failure_fallback_actions", 0) for row in selected
                ),
                "scope": "Transport/application coverage, not an efficacy threshold; valid neutral choices remain observed behavior.",
            }
        self.report.setdefault("checks", {})["astra_integration"] = integration
        self.report["physical_cost"] = {
            "rollouts": len(rows),
            "actions": sum(r["actions_executed"] for r in rows),
            "velocity_evaluations": sum(
                r["velocity_evaluations"]
                + r["parity_velocity_evaluations"]
                + r["probe_velocity_evaluations"]
                for r in rows
            ),
            "provider": summarize_calls(
                [c for r in rows for c in r["provider_records"]]
            ),
            "donor_captures": sum(r["donor_captures"] for r in rows),
            "donor_capture_seconds": sum(r["donor_capture_seconds"] for r in rows),
            "rollout_wall_seconds": sum(r.get("wall_seconds", 0) for r in rows),
            "policy_seconds": sum(r.get("policy_seconds", 0) for r in rows),
            "environment_seconds": sum(r.get("environment_seconds", 0) for r in rows),
            "condition_seconds": sum(r.get("condition_seconds", 0) for r in rows),
            "timing_note": "Rollout wall time includes provider waits, donor captures, probes, recording and simulator work; do not add nested times again.",
        }
        self.report["status"] = "complete"
        self.report.pop("live", None)
        self.save()
        self.progress()
        return self.report
