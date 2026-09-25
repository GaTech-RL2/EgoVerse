"""Join recorded interpolation feedback, physical calls and applied decisions.

This is an offline audit of an extracted, completed case. It reads the actual
raw-observation NPY files, verifies their descriptors and file hashes, and joins
them to decoded request PNGs. It makes no provider/model/simulator calls. Native
hidden-state recomputation, vision rasterization and reset physics are separate
audits; an intact failed provider call is valid negative evidence here.
"""

import argparse
import base64
import copy
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .agent import CAMERAS
from .astra_client import _strict_json
from .interpolation_agent import (
    LIMITS,
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    _validate_request,
    parse_proposal,
    summarize_calls,
)
from .interpolation_catalog import donor_catalog, paper_alpha
from .interpolation_search import load_protocol, outcome_feedback
from .records import digest, file_sha256

IDENTITY = (
    "schema_version",
    "episode_id",
    "attempt_id",
    "decision_index",
    "observation_step",
    "interpolation_mode",
    "request_fingerprint",
)
GENERATION_FLAGS = {
    "text_has_effect": "actions_with_nonzero_text",
    "vision_has_effect": "actions_with_changed_vision",
    "held_text_after_failed_call": "actions_with_held_text_after_failed_call",
    "native_condition_fallback": "native_condition_fallback_actions",
}
ALLOWED_EVENTS = {
    "case",
    "inversion_initialization",
    "interpolation_gate_reference",
    "interpolation_gate_check",
    "interpolation_gate",
    "interpolation_request",
    "interpolation_decision",
    "phase_generation",
    "phase_attempt",
    "phase_development_validation",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _operator(mode):
    return mode.removeprefix("astra_").removeprefix("oracle_").removesuffix("_vision")


class _Inputs:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.files, self.arrays, self.absent = {}, {}, set()

    def path(self, relative):
        path = (self.directory / relative).resolve()
        require(
            path.is_relative_to(self.directory),
            "Recorded artifact escapes the case directory",
        )
        return path

    def remember(self, relative):
        path = self.path(relative)
        sha = file_sha256(path)
        require(
            relative not in self.files or self.files[relative] == sha,
            "Input artifact changed during audit",
        )
        self.files[relative] = sha
        return path

    def json(self, relative):
        return _strict_json(self.remember(relative).read_text())

    def lines(self, relative):
        return [
            _strict_json(line)
            for line in self.remember(relative).read_text().splitlines()
            if line.strip()
        ]

    def array(self, reference):
        require(
            isinstance(reference, dict)
            and set(reference) == {"array", "shape", "dtype", "sha256"},
            "Raw observation has an invalid array descriptor",
        )
        name = reference["array"]
        require(
            isinstance(name, str)
            and name.startswith("arrays/")
            and name.endswith(".npy"),
            "Raw array must be a recorded NPY artifact",
        )
        value = np.load(self.remember(name), allow_pickle=False)
        require(
            isinstance(value, np.ndarray)
            and list(value.shape) == reference["shape"]
            and str(value.dtype) == reference["dtype"]
            and digest(value) == reference["sha256"]
            and np.isfinite(value).all(),
            "Recorded raw array integrity mismatch",
        )
        require(
            name not in self.arrays or self.arrays[name] == reference,
            "Array path has conflicting descriptors",
        )
        self.arrays[name] = dict(reference)
        return value

    def observation(self, descriptors):
        require(
            isinstance(descriptors, dict)
            and set(descriptors) == {*CAMERAS, "observation/state"},
            "Recorded raw observation has extra or missing fields",
        )
        value = {
            name: self.array(descriptors[name])
            for name in (*CAMERAS, "observation/state")
        }
        for camera in CAMERAS:
            image = value[camera]
            require(
                image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3,
                "Raw camera must be uint8 HWC RGB",
            )
        require(
            value["observation/state"].shape == (8,),
            "Raw proprioception must have eight entries",
        )
        return value

    def finish(self):
        for relative, sha in self.files.items():
            require(
                file_sha256(self.path(relative)) == sha,
                "Input artifact changed during audit",
            )
        require(
            all(not self.path(name).exists() for name in self.absent),
            "Absent provider sidecar appeared during audit",
        )


def _snapshot_binding(
    inputs, wire, recorded, *, fingerprint, scope, attempt_id, sequence
):
    require(
        (wire["label"], wire["step"]) == (recorded["label"], recorded["step"]),
        "Feedback snapshot label or step differs from its recorded raw frame",
    )
    raw = inputs.observation(recorded["observation"])
    camera_hashes = {}
    for camera in CAMERAS:
        encoded = wire["observation"][camera]
        require(
            set(encoded) == {"encoding", "data"}
            and encoded["encoding"] == "base64_png",
            "Feedback camera is not the declared PNG encoding",
        )
        png = base64.b64decode(encoded["data"], validate=True)
        with Image.open(io.BytesIO(png)) as frame:
            require(
                frame.format == "PNG" and frame.mode == "RGB",
                "Feedback camera is not raw RGB PNG",
            )
            decoded = np.asarray(frame).copy()
        require(
            np.array_equal(decoded, raw[camera])
            and digest(decoded) == recorded["observation"][camera]["sha256"],
            "Feedback PNG pixels differ from the recorded raw camera array",
        )
        camera_hashes[camera] = {
            "decoded_array_sha256": digest(decoded),
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "recorded_array": recorded["observation"][camera]["array"],
        }
    require(
        wire["observation"]["observation/state"] == raw["observation/state"].tolist(),
        "Feedback proprioception differs from recorded raw state",
    )
    return {
        "request_fingerprint": fingerprint,
        "scope": scope,
        "attempt_id": attempt_id,
        "step": wire["step"],
        "recorded_event_sequence": sequence,
        "raw_observation_sha256": digest(raw),
        "cameras": camera_hashes,
        "state_array_sha256": digest(raw["observation/state"]),
        "state_recorded_array": recorded["observation"]["observation/state"]["array"],
    }


def _provider_proposal(record, request, model):
    require(
        record["provider_call"] is True
        and type(record.get("http_status")) is int
        and 200 <= record["http_status"] < 300,
        "Accepted proposal lacks a successful physical provider call",
    )
    response = record.get("response", {})
    require(
        response.get("model") == model,
        "Accepted response substituted the configured model",
    )
    choices = response.get("choices")
    require(
        isinstance(choices, list) and len(choices) == 1,
        "Provider response has an invalid choice count",
    )
    choice = choices[0]
    require(
        isinstance(choice, dict) and choice.get("finish_reason") == "stop",
        "Accepted response did not finish normally",
    )
    message = choice.get("message", {})
    require(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and not message.get("refusal")
        and isinstance(message.get("content"), str),
        "Accepted response is not an assistant JSON proposal",
    )
    return parse_proposal(message["content"], request)


def _physical_attempts(report, protocol):
    require(
        set(report["arms"]) == set(protocol["arms"]),
        "Case arm coverage differs from the declared protocol",
    )
    require(
        set(report["controls"]) == {"known_noise", "policy_fresh"},
        "Case control coverage differs from the declared protocol",
    )
    baseline = report["baseline"]
    require(
        baseline["attempt_id"] == "recovered_noise_1"
        and baseline["mode"] == "recovered_noise"
        and baseline["iteration"] == 1,
        "Shared baseline identity differs",
    )
    physical = [
        baseline,
        report["controls"]["known_noise"],
        report["controls"]["policy_fresh"],
    ]
    priors = {}
    for arm in protocol["arms"]:
        rows = report["arms"][arm]["attempts"]
        require(
            rows and rows[0] == baseline,
            "Arm does not share the identical recorded baseline",
        )
        budget = (
            protocol["oracle_attempt_budget"]
            if arm.startswith("oracle_")
            else protocol["attempt_budget"]
        )
        require(
            1 <= len(rows) <= budget
            and [row["iteration"] for row in rows] == list(range(1, len(rows) + 1)),
            "Arm attempt coverage is not contiguous or exceeds budget",
        )
        for index, row in enumerate(rows[1:], 1):
            require(
                row["mode"] == arm and row["attempt_id"] == f"{arm}_{index + 1}",
                "Attempt identity mixes arms",
            )
            priors[row["attempt_id"]] = rows[:index]
        physical.extend(rows[1:])
    require(
        len({row["attempt_id"] for row in physical}) == len(physical),
        "Duplicate physical rollout identity",
    )
    return physical, priors


def audit_case(directory, *, expected_protocol=None):
    """Return safe provenance receipts; raise on any feedback/ledger integrity gap.

    ``directory`` contains summary.json, events.jsonl, arrays/, and the provider
    sidecars from one extracted case. By default the frozen protocol is required
    (with its development seed for development). An explicit expected_protocol
    supports a separately declared bounded protocol or hermetic test fixture.
    No partial or still-running case is described as complete.
    """
    inputs = _Inputs(directory)
    report = inputs.json("summary.json")
    require(
        report["schema_version"] == "phase-interpolation-1.0"
        and report["status"] in ("complete", "development_failed"),
        "Feedback audit requires a completed phase-interpolation case",
    )
    protocol = (
        copy.deepcopy(expected_protocol)
        if expected_protocol is not None
        else load_protocol()
    )
    if expected_protocol is None and report["development"]:
        protocol["seed"] = protocol["development_seed"]
    require(
        report["protocol_sha256"] == digest(protocol),
        "Case protocol differs from the expected frozen protocol",
    )
    require(
        report["source_catalog"] == donor_catalog(),
        "Case source catalog differs from the restricted donor library",
    )
    physical, priors = _physical_attempts(report, protocol)
    attempts = {row["attempt_id"]: row for row in physical}
    events = inputs.lines("events.jsonl")
    require(events, "Case has no event log")
    counts = Counter()
    request_events, decision_events, generations, endings = (
        {},
        {},
        defaultdict(dict),
        {},
    )
    case_event, spec = None, None
    for index, event in enumerate(events):
        require(
            type(event.get("sequence")) is int and event["sequence"] == index,
            "Case event sequence is incomplete or reordered",
        )
        kind = event.get("kind")
        require(kind in ALLOWED_EVENTS, "Unknown interpolation recording event")
        counts[kind] += 1
        if kind == "case":
            require(
                index == 0 and case_event is None,
                "Case identity event must occur exactly once at the beginning",
            )
            case_event = event
        elif kind == "inversion_initialization":
            require(spec is None, "Duplicate initialization controller specification")
            spec = event["action_spec"]
        elif kind == "interpolation_request":
            request = event["request"]
            _validate_request(request)
            key = request["attempt_id"], request["decision_index"]
            require(
                key not in request_events and key[0] in attempts,
                "Duplicate or unassigned interpolation request",
            )
            request_events[key] = event
        elif kind == "interpolation_decision":
            key = event["attempt_id"], event["decision"]["decision_index"]
            require(
                key in request_events and key not in decision_events,
                "Decision has no unique preceding request",
            )
            decision_events[key] = event
        elif kind == "phase_generation":
            attempt_id, step = event["attempt_id"], event["observation_step"]
            require(
                attempt_id in attempts
                and type(step) is int
                and step >= 0
                and step not in generations[attempt_id],
                "Duplicate, invalid or unassigned policy generation",
            )
            generations[attempt_id][step] = event
        elif kind == "phase_attempt":
            row = event["attempt"]
            require(
                row["attempt_id"] in attempts
                and row["attempt_id"] not in endings
                and row == attempts[row["attempt_id"]],
                "Completed attempt event differs from its summary",
            )
            endings[row["attempt_id"]] = event
    require(
        case_event is not None and case_event["protocol"] == protocol,
        "Recorded case protocol differs",
    )
    entry = case_event["entry"]
    require(
        report["episode_id"] == entry["episode_id"]
        and report["reset_entry_sha256"] == digest(entry),
        "Recorded case/reset identity differs",
    )
    require(
        spec is not None and counts["inversion_initialization"] == 1,
        "Missing recorded controller specification",
    )
    require(
        set(endings) == set(attempts) == set(generations),
        "Physical attempt/generation coverage is incomplete",
    )
    require(
        list(endings) == list(attempts),
        "Physical attempts ran outside the declared arm/control order",
    )

    settings = protocol["astra"]
    sampling = {
        name: settings[name] for name in ("reasoning_effort", "max_completion_tokens")
    }
    interval, chunk = settings["call_interval"], protocol["execute_steps"]
    require(
        chunk
        == protocol["vision"]["valid_for_actions"]
        == LIMITS["vision_chunk_actions"],
        "Vision lifetime differs from the frozen native chunk",
    )
    all_records, records_by_slot, expected_sidecars = [], {}, set()
    for row in physical:
        attempt_id = row["attempt_id"]
        require(
            attempt_id == f"{row['mode']}_{row['iteration']}",
            "Malformed physical attempt identity",
        )
        sidecar = f"{attempt_id}_provider.jsonl"
        if inputs.path(sidecar).exists():
            records = inputs.lines(sidecar)
            expected_sidecars.add(sidecar)
        else:
            records = []
            inputs.absent.add(sidecar)
        require(
            records == row["provider_records"],
            "Provider sidecar differs from recorded attempt records",
        )
        summarize_calls(records)
        require(
            type(row["actions_executed"]) is int
            and 1 <= row["actions_executed"] <= protocol["action_budget"],
            "Attempt has invalid executed-action coverage",
        )
        expected_steps = list(range(0, row["actions_executed"], chunk))
        require(
            sorted(generations[attempt_id]) == expected_steps,
            "Policy replan coverage is incomplete",
        )
        if row["mode"].startswith("astra_"):
            expected_calls = len(range(0, row["actions_executed"], interval))
            require(
                expected_calls <= settings["max_calls_per_rollout"]
                and len(records) == len(row["decisions"]) == expected_calls,
                "Missing, duplicate, hidden retry or unscheduled provider call",
            )
        else:
            require(
                not records and not row["decisions"],
                "A non-Astra rollout contains provider calls or decisions",
            )
        for index, (record, decision) in enumerate(
            zip(records, row["decisions"], strict=True), 1
        ):
            key = attempt_id, index
            require(
                key in request_events and key in decision_events,
                "Provider call is not backed by a recorded request and decision",
            )
            request = request_events[key]["request"]
            require(
                decision_events[key]["decision"] == decision,
                "Decision event differs from recorded attempt decision",
            )
            require(
                record["client_schema_version"] == SCHEMA_VERSION
                and record["prompt_template_version"] == PROMPT_TEMPLATE_VERSION,
                "Provider transport schema/template changed",
            )
            require(
                all(record.get(name) == request[name] for name in IDENTITY),
                "Provider record is bound to a different request identity",
            )
            require(
                record["requested_model"] == settings["model"]
                and record["sampling_settings"] == sampling
                and record["cache"] == {"no-cache": True}
                and record["response_format"] == {"type": "json_object"},
                "Provider model/sampling/cache/response settings changed",
            )
            require(
                decision["decision_index"] == index
                and decision["observation_step"] == (index - 1) * interval,
                "Decision slot is stale or unscheduled",
            )
            require(
                record["accepted"] == decision["accepted"],
                "Provider and controller acceptance disagree",
            )
            if decision["accepted"]:
                proposal = _provider_proposal(record, request, settings["model"])
                require(
                    proposal == decision["proposal"]
                    and proposal["decision_id"] == record["decision_id"]
                    and decision["error"] is None,
                    "Applied proposal differs from the actual accepted provider response",
                )
            else:
                require(
                    decision["proposal"] is None
                    and isinstance(decision["error"], str)
                    and decision["error"]
                    and decision["error"] == record["error"],
                    "Failed call has an invented proposal or mismatched error",
                )
                # A rejected physical response can legitimately have usage or a
                # different model. Ensure an otherwise valid response was not
                # silently discarded while claiming it failed validation.
                try:
                    _provider_proposal(record, request, settings["model"])
                except (ValueError, KeyError, TypeError, IndexError):
                    pass
                else:
                    raise ValueError("A valid provider response was marked rejected")
            records_by_slot[key] = record
        all_records.extend(records)
    actual_sidecars = {path.name for path in inputs.directory.glob("*_provider.jsonl")}
    require(
        actual_sidecars == expected_sidecars,
        "An unassigned provider sidecar reveals uncounted calls",
    )
    require(
        set(request_events) == set(decision_events) == set(records_by_slot),
        "Request/decision/provider coverage is incomplete",
    )

    image_bindings, provider_bindings, application = [], [], []
    for key, event in request_events.items():
        request, row = event["request"], attempts[key[0]]
        require(
            row["mode"].startswith("astra_"),
            "Provider request appeared in a non-Astra arm",
        )
        require(
            request["episode_id"] == entry["episode_id"]
            and request["target_task"] == entry["instruction"]
            and request["source_catalog"] == donor_catalog(),
            "Request leaks a different task, donor catalog or oracle mapping",
        )
        require(
            request["interpolation_mode"] == _operator(row["mode"])
            and request["vision_enabled"] == row["mode"].endswith("_vision"),
            "Request intervention channels differ from the assigned arm",
        )
        require(
            request["action_spec"] == spec,
            "Request controller context differs from recorded specification",
        )
        require(
            request["limits"]
            == {
                **LIMITS,
                "call_interval": interval,
                "max_calls": settings["max_calls_per_rollout"],
                "action_budget": protocol["action_budget"],
            },
            "Request altered the frozen decision budgets",
        )
        prior = priors[key[0]]
        require(
            request["previous_decisions"] == row["decisions"][: key[1] - 1],
            "Current-attempt history omits failures or mixes decisions",
        )
        require(
            request["completed_rollout_feedback"]
            == [outcome_feedback(value) for value in prior],
            "Completed feedback mixes arms or differs from actual outcomes",
        )
        previous = request["previous_attempt"]
        expected_previous = prior[-1]
        old_event = endings[expected_previous["attempt_id"]]
        require(
            old_event["sequence"] < event["sequence"],
            "Prior feedback was not completed before the new request",
        )
        require(
            previous is not None
            and previous["feedback"] == outcome_feedback(expected_previous)
            and previous["decisions"] == expected_previous["decisions"],
            "Previous-attempt feedback or decisions came from another rollout",
        )
        require(
            len(previous["snapshots"]) == len(old_event["snapshots"]),
            "Previous raw snapshot coverage changed",
        )
        for wire, saved in zip(
            previous["snapshots"], old_event["snapshots"], strict=True
        ):
            image_bindings.append(
                _snapshot_binding(
                    inputs,
                    wire,
                    saved,
                    fingerprint=request["request_fingerprint"],
                    scope="previous_attempt",
                    attempt_id=expected_previous["attempt_id"],
                    sequence=old_event["sequence"],
                )
            )
        step = request["observation_step"]
        expected_steps = list(
            range(
                max(0, step - (LIMITS["max_observations"] - 1) * chunk), step + 1, chunk
            )
        )
        require(
            [snapshot["step"] for snapshot in request["observations"]]
            == expected_steps,
            "Current feedback is stale or omits the latest policy observation",
        )
        for wire in request["observations"]:
            generation = generations[key[0]][wire["step"]]
            if wire["step"] == step:
                require(
                    event["sequence"]
                    < decision_events[key]["sequence"]
                    < generation["sequence"],
                    "Current request/decision/generation order is invalid",
                )
            else:
                require(
                    generation["sequence"] < event["sequence"],
                    "Historical current-attempt frame came from the future",
                )
            saved = {
                "step": wire["step"],
                "label": f"step_{wire['step']}",
                "observation": generation["observation"],
            }
            image_bindings.append(
                _snapshot_binding(
                    inputs,
                    wire,
                    saved,
                    fingerprint=request["request_fingerprint"],
                    scope="current_attempt",
                    attempt_id=key[0],
                    sequence=generation["sequence"],
                )
            )
        record = records_by_slot[key]
        choices = record.get("response", {}).get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        content = message.get("content") if isinstance(message, dict) else None
        provider_bindings.append(
            {
                "attempt_id": key[0],
                "decision_index": key[1],
                "observation_step": step,
                "request_fingerprint": request["request_fingerprint"],
                "provider_call": record["provider_call"],
                "accepted": record["accepted"],
                "actual_model": record.get("response", {}).get("model"),
                "http_status": record.get("http_status"),
                "error_kind": record.get("error_kind"),
                "response_text_sha256": hashlib.sha256(content.encode()).hexdigest()
                if isinstance(content, str)
                else None,
                "decision_id": record.get("decision_id"),
                "token_usage": record["token_usage"],
            }
        )

    action_counts, policy_counts = Counter(), Counter()
    for row in physical:
        attempt_id = row["attempt_id"]
        totals, policies, applied = Counter(), Counter(), []
        active, active_id, last_decision = None, None, None
        for step, event in sorted(generations[attempt_id].items()):
            require(
                event["mode"] == row["mode"]
                and event["iteration"] == row["iteration"]
                and event["sequence"] < endings[attempt_id]["sequence"],
                "Generation identity/order differs from its rollout",
            )
            raw = inputs.observation(event["observation"])
            marks, held, fallback = [], False, False
            if row["mode"].startswith("astra_"):
                if step % interval == 0:
                    key = attempt_id, step // interval + 1
                    require(
                        request_events[key]["request"]["active_interpolation"]
                        == active,
                        "Request's held text differs from the last accepted decision",
                    )
                    last_decision = decision_events[key]["decision"]
                    if last_decision["accepted"]:
                        proposal = last_decision["proposal"]
                        active = {
                            name: proposal[name]
                            for name in ("source_a_id", "source_b_id", "alpha")
                        }
                        active_id, marks = proposal["decision_id"], proposal["vision"]
                    expected_until = step + chunk if marks else None
                    logged = decision_events[key]
                    require(
                        logged["vision_valid_until_step"] == expected_until
                        and logged["language_hold_after_failure"]
                        == (not last_decision["accepted"] and active is not None)
                        and logged["raw_condition_fallback"]
                        == (not last_decision["accepted"] and active is None),
                        "Decision failure/vision-expiry metadata differs from the actual held state",
                    )
                held, fallback = (
                    active_id is not None and not last_decision["accepted"],
                    active is None,
                )
            elif row["mode"].startswith("oracle_"):
                oracle = report["oracle"]
                active = {
                    "source_a_id": oracle["source_a_id"],
                    "source_b_id": oracle["source_b_id"],
                    "alpha": paper_alpha(step // chunk, oracle["lambda_calls"]),
                }
            require(
                event["active_interpolation"] == active
                and event["applied_accepted_decision_id"] == active_id
                and event["vision"] == marks,
                "Applied text/vision differs from the accepted decision or expired marks persisted",
            )
            require(
                event["held_text_after_failed_call"] == held
                and event["native_condition_fallback"] == fallback,
                "Failed-call hold/fallback application differs",
            )
            provenance = event["conditioning"]
            if active is None:
                require(
                    provenance is None,
                    "Native condition contains an unassigned text edit",
                )
            else:
                prompts = {
                    item["source_id"]: item["prompt"] for item in donor_catalog()
                }
                require(
                    provenance["operator"] == _operator(row["mode"])
                    and provenance["alpha"] == active["alpha"]
                    and provenance["source_prompts"]
                    == [prompts[active["source_a_id"]], prompts[active["source_b_id"]]],
                    "Conditioning provenance differs from the applied source pair/alpha",
                )
            require(
                event["text_has_effect"]
                == bool(provenance and provenance["has_effect"]),
                "Text-effect flag disagrees with conditioning provenance",
            )
            require(
                event["vision_has_effect"]
                == (event["modified_observation_sha256"] != digest(raw)),
                "Changed-vision flag disagrees with recorded raw/modified identities",
            )
            if not marks:
                require(
                    event["modified_observation_sha256"] == digest(raw),
                    "Raw images changed without an active vision annotation",
                )
            width = min(chunk, row["actions_executed"] - step)
            if active_id is not None:
                totals["actions_with_accepted_decision"] += width
                policies["accepted_decision_policy_calls"] += 1
                if active_id not in applied:
                    applied.append(active_id)
            for flag, field in GENERATION_FLAGS.items():
                require(
                    type(event[flag]) is bool, "Generation effect flag must be boolean"
                )
                totals[field] += width * event[flag]
            policies["text_nonzero_policy_calls"] += event["text_has_effect"]
            policies["vision_changed_policy_calls"] += event["vision_has_effect"]
            policies["vision_active_policy_calls"] += bool(marks)
            policies["native_condition_fallback_policy_calls"] += fallback
        require(
            row["applied_accepted_decision_ids"] == applied
            and row["accepted_decisions_executed"] == len(applied),
            "Applied decision coverage differs from recorded actions",
        )
        for name in ("actions_with_accepted_decision", *GENERATION_FLAGS.values()):
            require(
                row[name] == totals[name], f"Action attribution count differs: {name}"
            )
        for name in (
            "accepted_decision_policy_calls",
            "text_nonzero_policy_calls",
            "vision_changed_policy_calls",
            "vision_active_policy_calls",
            "native_condition_fallback_policy_calls",
        ):
            require(
                row[name] == policies[name],
                f"Policy-call attribution count differs: {name}",
            )
        action_counts.update(totals)
        policy_counts.update(policies)
        application.append(
            {
                "attempt_id": attempt_id,
                "applied_accepted_decision_ids": applied,
                "actions": dict(totals),
                "policy_calls": dict(policies),
            }
        )

    inputs.finish()
    return {
        "schema_version": "interpolation-feedback-audit-1.0",
        "status": "passed",
        "episode_id": entry["episode_id"],
        "case_status": report["status"],
        "protocol_sha256": digest(protocol),
        "catalog_sha256": digest(donor_catalog()),
        "source_sha256": {
            path.name: file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("agent.py"),
                Path(__file__).with_name("astra_client.py"),
                Path(__file__).with_name("intervention_agent.py"),
                Path(__file__).with_name("interpolation_agent.py"),
                Path(__file__).with_name("interpolation_catalog.py"),
                Path(__file__).with_name("interpolation_search.py"),
                Path(__file__).with_name("records.py"),
                Path(__file__).parent / "configs/phase_interpolation_v1.json",
            )
        },
        "counts": {
            "physical_rollouts": len(physical),
            "requests": len(request_events),
            "client_attempts": len(all_records),
            "physical_provider_calls": sum(row["provider_call"] for row in all_records),
            "preflight_failures": sum(not row["provider_call"] for row in all_records),
            "accepted_provider_proposals": sum(row["accepted"] for row in all_records),
            "accepted_decisions_executed": sum(
                len(row["applied_accepted_decision_ids"]) for row in application
            ),
            "raw_snapshot_bindings": len(image_bindings),
            "decoded_camera_bindings": len(image_bindings) * len(CAMERAS),
            "verified_raw_arrays": len(inputs.arrays),
            "policy_generations": sum(len(rows) for rows in generations.values()),
        },
        "provider": summarize_calls(all_records),
        "provider_bindings": provider_bindings,
        "actions": dict(action_counts),
        "policy_calls": dict(policy_counts),
        "applications": application,
        "snapshot_bindings": image_bindings,
        "input_file_sha256": inputs.files,
        "verified_raw_array_descriptors": inputs.arrays,
        "absent_provider_sidecars": sorted(inputs.absent),
        "separate_transport_smoke_included": False,
        "limitations": [
            "Verifies the completed recorded case, not a live prefix or independent simulator replay.",
            "Raw NPY bytes and decoded feedback PNG/state are verified. Numerical hidden-state/action tensors and vision rasterization require the separate weighted array audit.",
            "An accepted or nonzero intervention is not causal evidence of task success; native fallback and failed-call holds remain counted.",
            "Separate transport-only smoke costs are excluded; reconcile their standalone receipt separately.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("--expected-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = (
        _strict_json(args.expected_protocol.read_text())
        if args.expected_protocol
        else None
    )
    receipt = audit_case(args.case_directory, expected_protocol=protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "episode_id": receipt["episode_id"],
                "counts": receipt["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
