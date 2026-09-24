"""Audit recorded Stage-1 Astra inversions and latent reuse without model solves.

Only an exact same-condition, same-latent forward generation is a reconstruction
measurement. Later generations under changed observations are reuse checks, not
action reconstruction tests. Actual Astra actions have no known generating
policy noise, so this audit never reports known-noise recovery for them.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .agent import parse_proposal
from .evaluate import read_events
from .flow import error_metrics
from .records import digest, file_sha256

ACTION_ATOL = 0.02
FULL_SHAPE = (1, 10, 32)


class Arrays:
    def __init__(self, directory):
        self.directory = directory
        self.verified = {}

    def read(self, reference, *, full=False):
        path = (self.directory / reference["array"]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Recorded array path escapes the run directory")
        value = np.load(path, allow_pickle=False)
        if (
            digest(value) != reference["sha256"]
            or list(value.shape) != reference["shape"]
            or str(value.dtype) != reference["dtype"]
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Recorded array integrity mismatch: {reference['array']}")
        if full and (value.shape != FULL_SHAPE or value.dtype != np.float32):
            raise ValueError(
                "Astra flow endpoints must retain full float32 [1,10,32] tensors"
            )
        self.verified[reference["array"]] = dict(reference)
        return value

    def unpack(self, value):
        if isinstance(value, dict):
            if set(value) == {"array", "shape", "dtype", "sha256"}:
                return self.read(value)
            return {key: self.unpack(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.unpack(item) for item in value]
        return value


def _equal(left, right, label):
    if not np.array_equal(left, right):
        raise ValueError(f"Recorded {label} mismatch")


def audit_run(directory):
    directory = Path(directory).resolve()
    events_path, manifest_path = directory / "events.jsonl", directory / "manifest.json"
    input_hashes = {
        "events_sha256": file_sha256(events_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    if config["method"] != "reversal" or config["flow"].get("noise_mix_rho", 0) != 0:
        raise ValueError(
            "This audit requires Stage-1 reversal with unmixed recovered noise"
        )
    if (
        manifest["action_spec"]["horizon"],
        manifest["action_spec"]["model_action_dim"],
    ) != FULL_SHAPE[1:]:
        raise ValueError(
            "This audit requires the recorded 10-by-32 internal action specification"
        )
    arrays = Arrays(directory)
    accepted, conditions, inverse_flows, active, plans = {}, {}, {}, {}, {}
    started, ended, models, requested_models = set(), set(), set(), set()
    fallback_flows = rejected_responses = event_count = 0
    orphan_generations = []

    def condition_for(flow):
        event = conditions[(flow["episode_id"], flow["condition_id"])]
        if (
            event["observation_step"] != flow["observation_step"]
            or event["sequence"] >= flow["sequence"]
        ):
            raise ValueError(
                "Flow has no preceding condition at the same observation step"
            )
        observation = arrays.unpack(event["observation"])
        if (
            digest(observation) != event["observation_id"]
            or digest({**observation, "prompt": event["prompt"]})
            != event["condition_id"]
        ):
            raise ValueError(
                "Condition pixels/state/prompt do not match recorded identities"
            )
        return event

    previous_sequence = -1
    for event in read_events(directory):
        if (
            type(event["sequence"]) is not int
            or event["sequence"] != previous_sequence + 1
        ):
            raise ValueError("Event log must retain its complete contiguous sequence")
        previous_sequence = event["sequence"]
        event_count += 1
        kind = event["kind"]
        if kind == "episode_start":
            if event["episode_id"] in started:
                raise ValueError("Duplicate episode start in one run")
            started.add(event["episode_id"])
            active.pop(event["episode_id"], None)
        elif kind == "episode_end":
            ended.add(event["episode_id"])
            active.pop(event["episode_id"], None)
        elif kind == "agent_response":
            # A new request begins a refresh; the controller discards its old
            # plan before contacting the agent, including on rejected replies.
            active.pop(event["request"]["episode_id"], None)
            if event["accepted"] is not True:
                rejected_responses += 1
                continue
            request, recorded = (
                arrays.unpack(event["request"]),
                arrays.unpack(event["proposal"]),
            )
            if (
                request["stage"] != 1
                or recorded["request_fingerprint"] != request["request_fingerprint"]
            ):
                raise ValueError(
                    "Accepted response is not bound to its Stage-1 request"
                )
            if (
                digest(request["observation"]) != request["observation_id"]
                or digest(
                    {k: v for k, v in request.items() if k != "request_fingerprint"}
                )
                != request["request_fingerprint"]
            ):
                raise ValueError("Accepted request observation/fingerprint mismatch")
            if request["action_spec"] != manifest["action_spec"]:
                raise ValueError(
                    "Accepted request has a different controller action specification"
                )
            proposal = parse_proposal(
                recorded["raw_response"],
                request,
                model_version=recorded["model_version"],
                request_time=recorded["request_time"],
                response_time=recorded["response_time"],
                max_timeout=request["max_timeout_env_steps"],
                completion_types=request["supported_completion_types"],
            )
            if any(
                getattr(proposal, key) != recorded[key]
                for key in (
                    "episode_id",
                    "plan_id",
                    "observation_step",
                    "action_spec_id",
                )
            ):
                raise ValueError(
                    "Accepted proposal identity differs from the raw agent response"
                )
            _equal(
                proposal.action_chunk,
                recorded["action_chunk"],
                "accepted controller action chunk",
            )
            key = (proposal.episode_id, proposal.plan_id)
            if key in accepted:
                raise ValueError("Accepted plan_id was reused within an episode")
            accepted[key] = {
                "sequence": event["sequence"],
                # Pixel arrays have already been verified; do not retain every
                # camera frame while auditing a long rollout collection.
                "request": {
                    "observation_id": request["observation_id"],
                    "task_instruction": request["task_instruction"],
                },
                "proposal": proposal,
            }
            models.add(proposal.model_version)
            requested_models.add(str(request.get("requested_model_version")))
        elif kind == "condition":
            conditions[(event["episode_id"], event["condition_id"])] = event
        elif kind == "flow" and event["role"] == "astra_inversion":
            inverse_flows[event["episode_id"]] = (event, condition_for(event))
        elif kind == "inversion":
            result = event["result"]
            if (
                result.get("reference") is not None
                or result.get("controller_actions") is None
            ):
                raise ValueError(
                    "Stage-1 Astra actions must not be labeled with known generating noise"
                )
            key = (event["episode_id"], result["plan_id"])
            if key in plans or key not in accepted:
                raise ValueError(
                    "Inversion must bind to one unique accepted agent proposal"
                )
            proposal_record = accepted[key]
            proposal, request = proposal_record["proposal"], proposal_record["request"]
            inverse_flow, condition = inverse_flows.pop(event["episode_id"])
            if (
                result["inverse_condition_id"] != inverse_flow["condition_id"]
                or condition["observation_id"] != request["observation_id"]
                or condition["prompt"] != request["task_instruction"]
                or inverse_flow["observation_step"] != proposal.observation_step
                or not proposal_record["sequence"]
                < inverse_flow["sequence"]
                < event["sequence"]
                or result["checkpoint"] != manifest["checkpoint"]
                or result["solver"] != config["flow"]["integrator"]
                or result["grid"] != inverse_flow["grid"]
            ):
                raise ValueError("Astra inversion condition/solver/provenance mismatch")
            endpoint = arrays.read(result["model_actions"], full=True)
            latent = arrays.read(result["noise"], full=True)
            _equal(
                proposal.action_chunk,
                arrays.read(result["controller_actions"]),
                "inverted Astra controller actions",
            )
            _equal(
                endpoint,
                arrays.read(inverse_flow["input"], full=True),
                "encoded full inversion endpoint",
            )
            _equal(
                latent,
                arrays.read(inverse_flow["output"], full=True),
                "recorded recovered latent",
            )
            plans[key] = {
                "endpoint": endpoint,
                "latent": latent,
                "first_generation_input": None,
                "row": {
                    "episode_id": key[0],
                    "plan_id": key[1],
                    "observation_step": proposal.observation_step,
                    "inverse_condition_id": condition["condition_id"],
                    "inverse_observation_id": condition["observation_id"],
                    "agent_response_sequence": proposal_record["sequence"],
                    "inversion_sequence": event["sequence"],
                    "inverse_flow_sequence": inverse_flow["sequence"],
                    "activation_sequence": None,
                    "request_fingerprint": proposal.request_fingerprint,
                    "recorded_model_version": proposal.model_version,
                    "controller_actions_sha256": digest(proposal.action_chunk),
                    "encoded_endpoint_sha256": digest(endpoint),
                    "recovered_latent_sha256": digest(latent),
                    "full_internal_shape": list(endpoint.shape),
                    "same_condition_roundtrip": None,
                    "generations": [],
                },
            }
        elif kind == "plan_activated":
            key = (event["episode_id"], event["plan_id"])
            if key not in plans or plans[key]["row"]["activation_sequence"] is not None:
                raise ValueError(
                    "Activated reversal plan must have exactly one recorded inversion"
                )
            row = plans[key]["row"]
            if row["observation_step"] != event["observation_step"]:
                raise ValueError(
                    "Plan activation observation step differs from its inversion"
                )
            row["activation_sequence"] = event["sequence"]
            active[event["episode_id"]] = key
        elif kind == "plan_rejected":
            active.pop(event["episode_id"], None)
        elif kind == "flow" and event["role"] == "fallback":
            fallback_flows += 1
            active.pop(event["episode_id"], None)
        elif kind == "flow" and event["role"] == "generation":
            condition = condition_for(event)
            latent = arrays.read(event["input"], full=True)
            endpoint = arrays.read(event["output"], full=True)
            key = active.get(event["episode_id"])
            if key is None:
                orphan_generations.append(
                    {
                        "sequence": event["sequence"],
                        "episode_id": event["episode_id"],
                        "observation_step": event["observation_step"],
                    }
                )
                continue
            plan = plans[key]
            row = plan["row"]
            previous = row["generations"][-1] if row["generations"] else None
            if plan["first_generation_input"] is None:
                plan["first_generation_input"] = latent.copy()
            same_condition = event["condition_id"] == row["inverse_condition_id"]
            same_latent = np.array_equal(latent, plan["latent"])
            generation = {
                "sequence": event["sequence"],
                "observation_step": event["observation_step"],
                "condition_sequence": condition["sequence"],
                "condition_id": event["condition_id"],
                "observation_id": condition["observation_id"],
                "input_latent_sha256": digest(latent),
                "latent_matches_recovered": bool(same_latent),
                "latent_matches_first_generation": bool(
                    np.array_equal(latent, plan["first_generation_input"])
                ),
                "same_inverse_condition": same_condition,
                "observation_step_advanced": event["observation_step"]
                > previous["observation_step"]
                if previous
                else None,
                "observation_id_changed": condition["observation_id"]
                != previous["observation_id"]
                if previous
                else None,
                "reconstruction_measured": False,
            }
            if (
                same_condition
                and same_latent
                and row["same_condition_roundtrip"] is None
            ):
                metrics = error_metrics(plan["endpoint"], endpoint)
                row["same_condition_roundtrip"] = {
                    "generation_sequence": event["sequence"],
                    "generation_ordinal": len(row["generations"]) + 1,
                    "observation_step": event["observation_step"],
                    "full_internal_reconstruction": metrics,
                    "action_channel_reconstruction": error_metrics(
                        plan["endpoint"][..., :7], endpoint[..., :7]
                    ),
                    "padding_channel_reconstruction": error_metrics(
                        plan["endpoint"][..., 7:], endpoint[..., 7:]
                    ),
                    "passed_action_tolerance": metrics["max_abs"] <= ACTION_ATOL,
                }
                generation["reconstruction_measured"] = True
            row["generations"].append(generation)

    if input_hashes != {
        "events_sha256": file_sha256(events_path),
        "manifest_sha256": file_sha256(manifest_path),
    }:
        raise ValueError(
            "Recording changed during the audit; rerun after writes finish"
        )
    rows = [plan["row"] for plan in plans.values()]
    for row in rows:
        generations = row["generations"]
        row["latent_reuse_verified"] = bool(generations) and all(
            g["latent_matches_recovered"] and g["latent_matches_first_generation"]
            for g in generations
        )
        row["subsequent_generations"] = len(generations[1:])
        row["generation_steps_advance"] = (
            all(g["observation_step_advanced"] for g in generations[1:])
            if len(generations) > 1
            else None
        )
        row["subsequent_observation_ids_change"] = (
            all(g["observation_id_changed"] for g in generations[1:])
            if len(generations) > 1
            else None
        )
        row["episode_ended"] = row["episode_id"] in ended
        row["missing_roundtrip_reason"] = (
            None
            if row["same_condition_roundtrip"]
            else "No recorded generation in this activation interval used both the inverse condition and recovered latent"
        )
    matched = [
        row["same_condition_roundtrip"]
        for row in rows
        if row["same_condition_roundtrip"]
    ]
    generations = [g for row in rows for g in row["generations"]]
    subsequent = [g for row in rows for g in row["generations"][1:]]
    counts = {
        "episodes_started": len(started),
        "episodes_ended": len(ended),
        "accepted_agent_proposals": len(accepted),
        "accepted_proposals_without_inversion": len(accepted) - len(plans),
        "rejected_agent_responses": rejected_responses,
        "astra_inversions": len(rows),
        "paired_same_condition_roundtrips": len(matched),
        "passing_roundtrips": sum(r["passed_action_tolerance"] for r in matched),
        "failing_roundtrips": sum(not r["passed_action_tolerance"] for r in matched),
        "unpaired_inversions": len(rows) - len(matched),
        "plans_without_activation": sum(r["activation_sequence"] is None for r in rows),
        "paired_on_first_generation": sum(
            r["generation_ordinal"] == 1 for r in matched
        ),
        "generations": len(generations),
        "subsequent_generations": len(subsequent),
        "changed_condition_generations": sum(
            not g["same_inverse_condition"] for g in generations
        ),
        "generations_with_wrong_recovered_latent": sum(
            not g["latent_matches_recovered"] for g in generations
        ),
        "generations_with_latent_changed_within_plan": sum(
            not g["latent_matches_first_generation"] for g in generations
        ),
        "plans_with_latent_mismatch": sum(
            not r["latent_reuse_verified"] for r in rows if r["generations"]
        ),
        "subsequent_steps_advanced": sum(
            g["observation_step_advanced"] for g in subsequent
        ),
        "subsequent_steps_not_advanced": sum(
            not g["observation_step_advanced"] for g in subsequent
        ),
        "subsequent_observation_ids_changed": sum(
            g["observation_id_changed"] for g in subsequent
        ),
        "subsequent_observation_ids_unchanged": sum(
            not g["observation_id_changed"] for g in subsequent
        ),
        "orphan_generation_flows": len(orphan_generations),
        "fallback_flows": fallback_flows,
        "verified_array_files": len(arrays.verified),
    }
    inventory = sorted(arrays.verified.values(), key=lambda value: value["array"])
    return {
        "status": "complete" if started == ended else "incomplete_recording",
        "directory": str(directory),
        "counts": counts,
        "maximum_full_internal_error": max(
            r["full_internal_reconstruction"]["max_abs"] for r in matched
        )
        if matched
        else None,
        "maximum_per_plan_rmse": max(
            r["full_internal_reconstruction"]["rmse"] for r in matched
        )
        if matched
        else None,
        "all_roundtrips_passed": bool(rows)
        and len(matched) == len(rows)
        and all(r["passed_action_tolerance"] for r in matched),
        "all_latents_match_recovered": bool(generations)
        and all(g["latent_matches_recovered"] for g in generations),
        "episodes_without_end": sorted(started - ended),
        "orphan_generation_flows": orphan_generations,
        "recorded_model_versions": sorted(models),
        "requested_model_versions": sorted(requested_models),
        "provenance": {
            **input_hashes,
            "event_count": event_count,
            "method": config["method"],
            "flow_config": config["flow"],
            "checkpoint": manifest["checkpoint"],
            "action_spec_id": manifest["action_spec"]["action_spec_id"],
            "task_manifest_sha256": manifest["task_manifest_sha256"],
            "verified_array_inventory_sha256": digest(inventory),
        },
        "verified_arrays": inventory,
        "plans": rows,
    }


def analyze(directories):
    paths = [Path(path).resolve() for path in directories]
    if len(set(paths)) != len(paths):
        raise ValueError("Do not count the same run directory twice")
    runs = []
    for path in paths:
        try:
            runs.append(audit_run(path))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            runs.append(
                {
                    "directory": str(path),
                    "status": "invalid_recording",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    valid = [run for run in runs if "counts" in run]
    counts = (
        {key: sum(run["counts"][key] for run in valid) for key in valid[0]["counts"]}
        if valid
        else {}
    )
    errors = [
        run["maximum_full_internal_error"]
        for run in valid
        if run["maximum_full_internal_error"] is not None
    ]
    rmses = [
        run["maximum_per_plan_rmse"]
        for run in valid
        if run["maximum_per_plan_rmse"] is not None
    ]
    return {
        "schema_version": "1.0",
        "status": "complete"
        if runs and all(run["status"] == "complete" for run in runs)
        else "incomplete_or_invalid_recording",
        "scope": "Read-only Stage-1 recorded Astra inversion and latent reuse audit; no model or GPU solves",
        "action_atol": ACTION_ATOL,
        "known_noise_recovery": None,
        "known_noise_explanation": "Actual Astra controller actions have no known generating policy noise; recovered latents are checked for reuse only",
        "reconstruction_rule": "First forward generation within the plan activation interval with exactly the inverse condition and recovered full latent",
        "changed_condition_rule": "Changed-condition outputs are never scored as reconstruction errors",
        "observation_progression_interpretation": "Advancing steps and changed observation digests are reported separately; identical digests at later steps do not by themselves prove stale sensing",
        "integrity_scope": "All arrays used for accepted requests/proposals, inverse endpoints/latents, and paired or subsequent generation conditions/input/output; unrelated control-step, fallback and optional trajectory-state arrays are not read",
        "counts": counts,
        "runs_requested": len(runs),
        "runs_with_verified_records": len(valid),
        "maximum_full_internal_error": max(errors) if errors else None,
        "maximum_per_plan_rmse": max(rmses) if rmses else None,
        "all_roundtrips_passed": bool(valid)
        and len(valid) == len(runs)
        and all(run["all_roundtrips_passed"] for run in valid),
        "all_latents_match_recovered": bool(valid)
        and len(valid) == len(runs)
        and all(run["all_latents_match_recovered"] for run in valid),
        "audit_source_sha256": file_sha256(__file__),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    for source in map(lambda path: Path(path).resolve(), args.directory):
        if output in (
            source / "events.jsonl",
            source / "manifest.json",
        ) or output.is_relative_to(source / "arrays"):
            raise ValueError("Audit output must not overwrite recorded input artifacts")
    report = analyze(args.directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "counts",
                    "maximum_full_internal_error",
                    "all_roundtrips_passed",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
