"""Read-only representation request/provider/application audit; no model calls.

Run with the frozen audit codec on PYTHONPATH. Live inputs are explicitly
prefixes. A complete receipt additionally requires the passed immutable archive
audit, whose actual-NPY checks supply the byte layer for image descriptor joins.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, features

from astra_reversal.agent import CAMERAS
from astra_reversal.astra_client import DEFAULT_ENDPOINT, ClientError
from astra_reversal.image_donor_bank import load_library
from astra_reversal.interpolation_catalog import donor_catalog
from astra_reversal.records import digest
from astra_reversal.representation_agent import (
    PROMPT_TEMPLATE_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    build_payload,
    parse_proposal,
    summarize_calls,
)
from astra_reversal.representation_search import ARMS

MODEL = "azure/openai/gpt-6-astra"
SAMPLING = {"reasoning_effort": "medium", "max_completion_tokens": 8192}
ASTRA_ARMS = [arm for arm in ARMS if arm.startswith("astra_")]
ROOT = Path(
    os.environ.get("ASTRA_REPOSITORY_ROOT", Path(__file__).resolve().parents[3])
).resolve()
SOURCES = (
    "astra_reversal/representation_agent.py",
    "astra_reversal/representation_search.py",
    "astra_reversal/osmo/representation_steering.py",
    "astra_reversal/image_perturbation_agent.py",
    "astra_reversal/interpolation_agent.py",
    "astra_reversal/intervention_agent.py",
    "astra_reversal/astra_client.py",
    "astra_reversal/records.py",
    "astra_reversal/intervention_rollout.py",
    "astra_reversal/vision_interpolation.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def lines(path, sources):
    """Hash exactly the parsed prefix while retaining no raw rows in receipts."""
    path = Path(path)
    checksum, size, count = hashlib.sha256(), 0, 0
    if path.exists():
        with path.open("rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    require(not line.endswith(b"\n"), "Malformed complete JSONL line")
                    break
                checksum.update(line)
                size += len(line)
                count += 1
                yield row
    sources[path.name] = {
        "sha256": checksum.hexdigest(),
        "bytes": size,
        "records": count,
        "entire_file": not path.exists() or size == path.stat().st_size,
    }


def codec():
    versions = {
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "pillow_zlib": features.version("zlib"),
        "python": platform.python_version(),
    }
    require(
        versions["numpy"] == "1.26.4"
        and versions["pillow"] == "12.3.0"
        and versions["pillow_zlib"] == "1.3",
        "Use the matched frs-audit-codec environment",
    )
    return versions


def observed_descriptor(observation):
    result = {}
    for camera in CAMERAS:
        wire = observation[camera]
        require(
            set(wire) == {"encoding", "data"} and wire["encoding"] == "base64_png",
            "Unexpected request image encoding",
        )
        with Image.open(
            io.BytesIO(base64.b64decode(wire["data"], validate=True))
        ) as image:
            require(
                image.mode == "RGB" and image.format == "PNG",
                "Request image is not RGB PNG",
            )
            array = np.asarray(image).copy()
        result[camera] = {
            "sha256": digest(array),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    result["state"] = observation["observation/state"]
    return result


def match_observation(wire_description, recorded):
    require(
        set(recorded) == {*CAMERAS, "observation/state"},
        "Recorded observation has privileged/unsupported fields",
    )
    for camera in CAMERAS:
        require(
            all(
                recorded[camera][key] == value
                for key, value in wire_description[camera].items()
            ),
            "Request PNG differs from recorded raw camera descriptor",
        )
    state = recorded["observation/state"]
    values = np.asarray(wire_description["state"], dtype=np.dtype(state["dtype"]))
    require(
        list(values.shape) == state["shape"] and digest(values) == state["sha256"],
        "Request robot state differs from recorded state",
    )


def feedback(attempt):
    return {
        "attempt_id": attempt["attempt_id"],
        "success": attempt["success"],
        "executed_actions": attempt["actions_executed"],
        "termination": attempt["status"],
        "error": None,
    }


def error_category(record):
    # Provider messages can contain key aliases/masked fragments. Export only a
    # fixed classification, never the raw message or credentials/headers.
    if record.get("provider_error", {}).get("type") == "budget_exceeded":
        return "budget_exceeded"
    message = record.get("error", "")
    for needle, label in (
        ("echo request_id", "request_id_mismatch"),
        ("different model", "model_mismatch"),
        ("truncated", "truncated_completion"),
        ("NVIDIA_INFERENCE_API_KEY is not set", "credential_preflight"),
    ):
        if needle in message:
            return label
    return record.get("error_kind", "unknown")


def nominal_neutral(proposal, mode):
    if proposal["mode"] == "native":
        return True
    language, vision = proposal["language"], proposal["vision"]
    text_neutral = language is None or (
        mode in ("tli", "tli_vli")
        and (
            language["alpha"] == 0.5
            or language["source_a_id"] == language["source_b_id"]
        )
    )
    return text_neutral and (vision is None or vision["alpha"] == 0)


def match_application(choice, provenance, mode, instruction, library):
    """Bind the accepted parameter choice to the logged conditioning operator."""
    if choice is None or choice["mode"] == "native":
        require(
            provenance is None, "Native generation retained intervention provenance"
        )
        return
    require(
        isinstance(provenance, dict), "Chosen intervention lacks operator provenance"
    )
    language, vision = choice["language"], choice["vision"]
    sources = {row["source_id"]: row["prompt"] for row in donor_catalog()}
    if mode in ("tei", "tli"):
        require(
            provenance["operator"] == mode and provenance["alpha"] == language["alpha"],
            "Applied text operator/alpha differs from proposal",
        )
        require(
            provenance["source_prompts"]
            == [sources[language[key]] for key in ("source_a_id", "source_b_id")],
            "Applied text sources differ from proposal",
        )
        require(
            provenance["target_prompt"] == instruction,
            "Text intervention changed target identity",
        )
        return
    if mode == "pixel_blend":
        require(
            provenance["operator"] == mode and provenance["vision"] == vision,
            "Applied RGB donor/alpha differs from proposal",
        )
        return
    require(
        provenance["operator"] == ("vli" if mode == "tli_vli" else mode)
        and provenance["alpha"] == vision["alpha"],
        "Applied visual operator/alpha differs from proposal",
    )
    require(
        provenance["cameras"] == list(CAMERAS)
        and provenance["target_prompt"] == instruction,
        "Visual operator camera/prompt scope differs",
    )
    if vision["alpha"] == 0:
        require(
            provenance["bank"] is None,
            "Zero visual alpha unexpectedly applied a donor bank",
        )
    else:
        bank = provenance["bank"]
        require(
            bank["bank_id"]
            == digest({key: value for key, value in bank.items() if key != "bank_id"}),
            "Vision bank identity differs from its metadata",
        )
        require(
            bank["provenance"]["target_prompt"] == instruction,
            "Visual bank was not captured under target instruction",
        )
        donor = bank["provenance"]["donor"]
        expected = next(
            row for row in library.catalog() if row["donor_id"] == vision["donor_id"]
        )
        require(
            all(
                donor[key] == expected[key]
                for key in (
                    "donor_id",
                    "library_id",
                    "sample_sha256",
                    "source_id",
                    "prompt",
                    "episode_index",
                    "frame_index",
                    "phase",
                )
            ),
            "Applied visual bank is not the chosen exact donor frame",
        )
    actual_language = provenance["language"]
    if mode == "tli_vli":
        require(
            actual_language["operator"] == "tli"
            and actual_language["alpha"] == language["alpha"],
            "Composite TLI coefficient differs from proposal",
        )
        require(
            actual_language["source_prompts"]
            == [sources[language[key]] for key in ("source_a_id", "source_b_id")],
            "Composite TLI sources differ from proposal",
        )
    else:
        require(
            actual_language["operator"] is None
            and actual_language["source_prompts"] is None,
            "Visual-only arm added text steering",
        )


def summarize_arm(arm, attempts, providers, executed):
    baseline = next((row for row in attempts if row["arm"] == "native"), None)
    selected = [row for row in attempts if row["arm"] == arm]
    applicable = ([baseline] if baseline else []) + selected
    first = next((row for row in applicable if row["success"]), None)
    prefix = [
        row
        for row in applicable
        if first is None or row["revision"] <= first["revision"]
    ]
    ids = {row["attempt_id"] for row in selected}
    if arm.startswith("astra_"):
        ids.update(
            row["attempt_id"]
            for row in providers
            if row.get("attempt_id") in (f"{arm}_revision1", f"{arm}_revision2")
        )
    prefix_ids = {row["attempt_id"] for row in prefix}
    calls = [row for row in providers if row.get("attempt_id") in ids]
    choices = Counter()
    for record in calls:
        if not record["accepted"]:
            choices["rejected"] += 1
            continue
        proposal = json.loads(record["response"]["choices"][0]["message"]["content"])
        choices[proposal["mode"]] += 1
        choices["nominal_neutral"] += nominal_neutral(
            proposal, arm.removeprefix("astra_")
        )
    return {
        "completed_intervention_rollouts": len(selected),
        "intervention_outcomes": [
            {
                "attempt_id": row["attempt_id"],
                "revision": row["revision"],
                "success": row["success"],
                "actions": row["actions_executed"],
                "accepted_decisions_executed": row["accepted_decisions_executed"],
                "nonzero_intervention_actions": row["nonzero_intervention_actions"],
                "provider_failure_fallback_actions": row[
                    "provider_failure_fallback_actions"
                ],
            }
            for row in selected
        ],
        "observed_success": first is not None,
        "first_success_revision": None if first is None else first["revision"],
        "censored_at_cap": first is None and len(selected) == 2,
        "development_extra_rollouts": len(applicable) - len(prefix),
        "choices": dict(choices),
        "provider": summarize_calls(calls),
        "provider_through_success_or_cap": summarize_calls(
            [row for row in providers if row.get("attempt_id") in prefix_ids]
        ),
        "executed": {
            name: sum(executed[attempt_id][name] for attempt_id in ids)
            for name in (
                "actions",
                "accepted_decision_actions",
                "explicit_native_actions",
                "nonzero_intervention_actions",
                "provider_failure_fallback_actions",
                "accepted_decisions_executed",
            )
        },
    }


def audit_case(
    task_dir, worker_dir, *, archive_receipt=None, require_complete=False, library=None
):
    auditor_source_hash = file_sha(__file__)
    task_dir, worker_dir = Path(task_dir), Path(worker_dir)
    environment = codec()
    library = library or load_library(
        ROOT / "astra_reversal/.deps/image-perturbations/donors"
    )
    source_files = {}
    summary_bytes = (task_dir / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    source_files["summary.json"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
    }
    protocol = json_file(worker_dir / "protocol.json")
    runtime = json_file(worker_dir / "runtime.json")
    expected_protocol = json_file(
        ROOT / "astra_reversal/configs/representation_steering_v1.json"
    )
    if runtime["phase"] == "development":
        expected_protocol["seed"] = expected_protocol["development_seed"]
    require(protocol == expected_protocol, "Worker protocol differs from frozen source")
    require(
        summary["protocol_sha256"] == digest(protocol), "Case protocol identity differs"
    )
    require(
        summary["image_library_id"] == library.library_id, "Case donor library differs"
    )
    require(
        runtime["gpu"].find("L40S") >= 0 and runtime["tf32"] is False,
        "Wrong GPU/TF32 runtime",
    )
    require(
        runtime["packages"]["numpy"] == environment["numpy"]
        and runtime["packages"]["pillow"] == environment["pillow"],
        "Worker package versions differ from audit environment",
    )
    for name in (
        "runtime.json",
        "protocol.json",
        "prompts.json",
        "frozen_plan.json",
        "reset_manifest.json",
        "checkpoint.json",
        "bank_inventory.json",
    ):
        path = worker_dir / name
        if path.exists():
            source_files["worker/" + name] = {
                "sha256": file_sha(path),
                "bytes": path.stat().st_size,
            }
    prompts = json_file(worker_dir / "prompts.json")
    require(
        prompts["system_prompt"] == SYSTEM_PROMPT
        and prompts["template_version"] == PROMPT_TEMPLATE_VERSION
        and prompts["sampling"] == protocol["astra"],
        "Saved prompts/settings differ from client source",
    )
    providers = list(lines(task_dir / "provider.jsonl", source_files))
    by_fingerprint = {}
    for record in providers:
        fp = record["request_fingerprint"]
        require(
            fp not in by_fingerprint,
            "Multiple provider rows for one request: hidden retry or duplicated ledger",
        )
        by_fingerprint[fp] = record
        require(
            record["episode_id"] == summary["episode_id"],
            "Provider row belongs to another case",
        )
        require(
            record["client_schema_version"] == SCHEMA_VERSION
            and record["prompt_template_version"] == PROMPT_TEMPLATE_VERSION,
            "Provider client/template differs",
        )
        require(
            record["requested_model"] == MODEL
            and record["endpoint"] == DEFAULT_ENDPOINT,
            "Different provider model/endpoint",
        )
        if record["provider_call"]:
            require(
                record["sampling_settings"] == SAMPLING
                and record["cache"] == {"no-cache": True}
                and record["response_format"] == {"type": "json_object"},
                "Different physical provider settings",
            )
            require(
                record["system_prompt_sha256"]
                == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "Physical system prompt differs",
            )

    stats = Counter()
    starts, generations, decisions, completed, completions = (
        {},
        defaultdict(list),
        defaultdict(list),
        {},
        [],
    )
    requests, expected_proposals, request_raw, all_used_provider = {}, {}, {}, set()
    active, latest_failure = {}, {}
    current = None
    sequence = -1
    for event in lines(task_dir / "events.jsonl", source_files):
        sequence += 1
        require(event["sequence"] == sequence, "Non-contiguous event sequence")
        kind = event["kind"]
        if kind == "native_parity":
            stats["recorded_native_parity_velocity_evaluations"] += 10
        elif kind == "weighted_vision_probe":
            stats["recorded_probe_velocity_evaluations"] += event["probe"][
                "velocity_evaluations"
            ]
        if kind == "representation_rollout_start":
            attempt_id = event["attempt_id"]
            require(
                attempt_id not in starts and current is None,
                "Overlapping/duplicate rollout identity",
            )
            current = attempt_id
            starts[attempt_id] = event
            active[attempt_id], latest_failure[attempt_id] = None, False
            require(
                event["entry"]["episode_id"] == summary["episode_id"]
                and event["entry"]["instruction"] == summary["instruction"],
                "Rollout entry differs from case",
            )
            require(
                digest(event["entry"]) == summary["reset_entry_sha256"],
                "Rollout reset entry differs",
            )
        elif kind == "representation_request":
            request = event["request"]
            attempt_id = request["attempt_id"]
            require(
                attempt_id == current
                and starts[attempt_id]["arm"].startswith("astra_"),
                "Request outside its own Astra rollout",
            )
            arm = starts[attempt_id]["arm"]
            require(
                request["episode_id"] == summary["episode_id"]
                and request["representation_mode"] == arm.removeprefix("astra_")
                and request["target_task"] == summary["instruction"],
                "Wrong request case/arm/task",
            )
            require(
                request["source_catalog"] == donor_catalog()
                and request["donor_catalog"] == library.catalog(),
                "Request donor catalogs differ from frozen non-oracle catalog",
            )
            require(
                request["previous_decisions"] == decisions[attempt_id][-2:],
                "Request leaked or omitted own recent decisions/errors",
            )
            expected_history = [
                row for row in completions if row["arm"] in ("native", arm)
            ]
            require(
                request["completed_rollout_feedback"]
                == [feedback(row) for row in expected_history],
                "Completed feedback is not own-arm plus shared baseline",
            )
            previous = request["previous_attempt"]
            require(
                expected_history and previous is not None,
                "Missing last completed raw rollout feedback",
            )
            latest = completed[expected_history[-1]["attempt_id"]]
            require(
                previous["feedback"] == feedback(latest["attempt"])
                and previous["decisions"] == latest["attempt"]["decisions"][-2:],
                "Previous attempt identity/history differs",
            )
            require(
                len(previous["snapshots"]) == len(latest["snapshots"]),
                "Previous rollout snapshot selection differs",
            )
            for shown, recorded in zip(
                previous["snapshots"], latest["snapshots"], strict=True
            ):
                require(
                    shown["step"] == recorded["step"]
                    and shown["label"] == recorded["label"],
                    "Prior frame timing/label differs",
                )
                match_observation(
                    observed_descriptor(shown["observation"]), recorded["observation"]
                )
                stats["previous_raw_pairs_bound"] += 1
            step = request["observation_step"]
            expected_steps = (
                [row["observation_step"] for row in generations[attempt_id]] + [step]
            )[-4:]
            require(
                [row["step"] for row in request["observations"]] == expected_steps,
                "Current frame history is not the latest raw replans",
            )
            described = [
                observed_descriptor(row["observation"])
                for row in request["observations"]
            ]
            for raw_description, prior_step in zip(
                described[:-1], expected_steps[:-1], strict=True
            ):
                raw_event = next(
                    row
                    for row in generations[attempt_id]
                    if row["observation_step"] == prior_step
                )
                match_observation(raw_description, raw_event["observation"])
                stats["current_raw_pairs_bound"] += 1
            require(
                (attempt_id, step) not in request_raw, "Duplicate scheduled request"
            )
            request_raw[attempt_id, step] = described[-1]
            payload = build_payload(request, MODEL, sampling=SAMPLING)
            fp = request["request_fingerprint"]
            require(fp not in requests, "Duplicate request fingerprint")
            requests[fp] = {
                name: request[name]
                for name in (
                    "request_id",
                    "episode_id",
                    "attempt_id",
                    "decision_index",
                    "observation_step",
                    "representation_mode",
                )
            }
            stats["requests_validated"] += 1
            record = by_fingerprint.get(fp)
            if record is not None:
                require(
                    all(record[name] == value for name, value in requests[fp].items()),
                    "Physical provider/request identity differs",
                )
                require(
                    record["payload_sha256"]
                    == hashlib.sha256(
                        json.dumps(payload, allow_nan=False).encode()
                    ).hexdigest(),
                    "Exact HTTP wire payload differs",
                )
                require(
                    record["source_catalog_sha256"] == digest(request["source_catalog"])
                    and record["donor_catalog_sha256"]
                    == digest(request["donor_catalog"])
                    and record["library_id"] == library.library_id,
                    "Ledger catalog binding differs",
                )
                require(
                    record["contact_sheet_sha256"]
                    == {
                        sheet["camera"]: sheet["sha256"]
                        for sheet in request["contact_sheets"]
                    },
                    "Ledger contact sheet binding differs",
                )
                all_used_provider.add(fp)
                stats["exact_payload_bindings"] += 1
                if record["accepted"]:
                    envelope = record["response"]
                    require(
                        envelope["model"] == MODEL
                        and len(envelope["choices"]) == 1
                        and envelope["choices"][0]["finish_reason"] == "stop",
                        "Accepted provider completion violates model/finish contract",
                    )
                    expected_proposals[fp] = parse_proposal(
                        envelope["choices"][0]["message"]["content"], request
                    )
                    stats["accepted_strict_parses"] += 1
                elif record.get("error_kind") == "proposal_rejected" and record.get(
                    "response", {}
                ).get("choices"):
                    envelope = record["response"]
                    if (
                        envelope.get("model") == MODEL
                        and envelope["choices"][0].get("finish_reason") == "stop"
                    ):
                        try:
                            parse_proposal(
                                envelope["choices"][0]["message"]["content"], request
                            )
                        except ClientError:
                            stats["schema_rejections_reproduced"] += 1
                        else:
                            raise ValueError(
                                "Recorded schema rejection parsed successfully"
                            )
        elif kind in ("representation_decision", "random_representation_decision"):
            attempt_id, decision = event["attempt_id"], event["decision"]
            require(attempt_id == current, "Decision applied outside its rollout")
            require(
                decision["decision_index"] == len(decisions[attempt_id]) + 1
                and decision["observation_step"] == len(decisions[attempt_id]) * 25,
                "Decision call schedule differs",
            )
            if kind == "representation_decision":
                matched = [
                    fp
                    for fp, request in requests.items()
                    if request["attempt_id"] == attempt_id
                    and request["decision_index"] == decision["decision_index"]
                ]
                require(len(matched) == 1, "Decision lacks one request")
                fp = matched[0]
                provider = by_fingerprint.get(fp)
                if provider is not None:
                    require(
                        provider["accepted"] == decision["accepted"],
                        "Provider and decision acceptance differ",
                    )
                    if decision["accepted"]:
                        require(
                            decision["proposal"] == expected_proposals[fp]
                            and decision["error"] is None,
                            "Applied proposal differs from accepted provider text",
                        )
                    else:
                        require(
                            decision["proposal"] is None
                            and decision["error"] == provider["error"],
                            "Failure event differs from physical ledger",
                        )
                    stats["provider_decision_bindings"] += 1
            decisions[attempt_id].append(decision)
            active[attempt_id] = decision["proposal"]
            latest_failure[attempt_id] = not decision["accepted"]
        elif kind == "representation_generation":
            attempt_id, step = event["attempt_id"], event["observation_step"]
            require(
                attempt_id == current and step == len(generations[attempt_id]) * 5,
                "Generation schedule/attempt differs",
            )
            require(
                event["active_intervention"] == active[attempt_id],
                "Generation used stale/different accepted decision",
            )
            match_application(
                event["active_intervention"],
                event["provenance"],
                starts[attempt_id]["arm"]
                .removeprefix("astra_")
                .removeprefix("random_"),
                summary["instruction"],
                library,
            )
            if (attempt_id, step) in request_raw:
                match_observation(
                    request_raw.pop((attempt_id, step)), event["observation"]
                )
                stats["current_raw_pairs_bound"] += 1
            if latest_failure[attempt_id]:
                require(
                    event["active_intervention"] is None
                    and event["provenance"] is None,
                    "Failure did not immediately clear to fresh native",
                )
            if (
                event["active_intervention"] is not None
                and event["active_intervention"]["mode"] == "native"
            ):
                require(
                    event["provenance"] is None,
                    "Explicit native mode retained an intervention",
                )
            generations[attempt_id].append(event)
            stats["generation_application_bindings"] += 1
        elif kind == "representation_rollout_complete":
            attempt = event["attempt"]
            attempt_id = attempt["attempt_id"]
            require(
                attempt_id == current and attempt_id not in completed,
                "Rollout completion identity differs",
            )
            require(
                attempt["decisions"] == decisions[attempt_id],
                "Completed decision history differs from events",
            )
            require(
                0 < attempt["actions_executed"] <= 300
                and not attempt["initial_success"],
                "Invalid zero-action or over-budget attempt",
            )
            require(
                len(generations[attempt_id]) == (attempt["actions_executed"] + 4) // 5,
                "Generation/action coverage differs",
            )
            require(
                attempt["revision"] == starts[attempt_id]["revision"]
                and attempt["arm"] == starts[attempt_id]["arm"],
                "Completed arm/revision differs",
            )
            completed[attempt_id] = event
            completions.append(attempt)
            current = None

    executed = defaultdict(Counter)
    for attempt in completions:
        attempt_id = attempt["attempt_id"]
        rows, accepted_ids = generations[attempt_id], set()
        record_rows = [row for row in providers if row.get("attempt_id") == attempt_id]
        require(
            record_rows == attempt["provider_records"][: len(record_rows)],
            "Completed rollout provider records differ from physical sidecar",
        )
        stats["completed_provider_rows_awaiting_download"] += len(
            attempt["provider_records"]
        ) - len(record_rows)
        require(
            len(attempt["provider_records"]) == len(decisions[attempt_id])
            if attempt["arm"].startswith("astra_")
            else not record_rows and not attempt["provider_records"],
            "Completed call slots differ from provider rows",
        )
        generated = []
        for row in rows:
            step, choice = row["observation_step"], row["active_intervention"]
            count = min(5, attempt["actions_executed"] - step)
            current_decisions = [
                item
                for item in decisions[attempt_id]
                if item["observation_step"] <= step
            ]
            failure = bool(
                attempt["arm"].startswith("astra_")
                and current_decisions
                and not current_decisions[-1]["accepted"]
            )
            effect = bool(
                row["provenance"] and row["provenance"].get("has_effect", False)
            )
            counts = executed[attempt_id]
            counts["actions"] += count
            if attempt["arm"].startswith("astra_") and choice is not None:
                accepted_ids.add(choice["decision_id"])
                counts["accepted_decision_actions"] += count
                counts["explicit_native_actions"] += count * (
                    choice["mode"] == "native"
                )
            counts["nonzero_intervention_actions"] += count * effect
            counts["provider_failure_fallback_actions"] += count * failure
            generated.append(
                {
                    "step": step,
                    "choice": choice,
                    "provider_failure_fallback": failure,
                    "has_effect": effect,
                }
            )
        counts["accepted_decisions_executed"] = len(accepted_ids)
        require(
            generated == attempt["generations"],
            "Completed generation summaries differ from application events",
        )
        for name in (
            "accepted_decisions_executed",
            "accepted_decision_actions",
            "explicit_native_actions",
            "nonzero_intervention_actions",
            "provider_failure_fallback_actions",
        ):
            require(
                attempt[name] == counts[name],
                "Completed action attribution differs: " + name,
            )
        require(
            attempt["velocity_evaluations"]
            == sum(row["velocity_evaluations"] for row in rows),
            "Rollout solver costs differ from recorded generations",
        )

    complete = (
        summary["status"] == "complete"
        and current is None
        and not request_raw
        and len(completions) == len(summary["physical_rollouts"])
        and len(requests) == len(providers) == len(all_used_provider)
        and stats["completed_provider_rows_awaiting_download"] == 0
    )
    for attempt in completions:
        summarized = next(
            (
                row
                for row in summary["physical_rollouts"]
                if row["attempt_id"] == attempt["attempt_id"]
            ),
            None,
        )
        if summarized is not None:
            require(summarized == attempt, "Case summary differs from completed event")
    if require_complete:
        require(
            complete and len(requests) == len(providers) == len(all_used_provider),
            "Incomplete final provider/event coverage",
        )
        require(
            archive_receipt is not None,
            "Complete claim requires passed immutable archive audit",
        )
    archive_binding = None
    if archive_receipt is not None:
        receipt_path = Path(archive_receipt)
        archive = json_file(receipt_path)
        partial_archive = archive.get("status") == "verified_partial_archive"
        require(
            archive.get("status") == "passed"
            or (partial_archive and not require_complete),
            "Archive audit does not support the requested evidence scope",
        )
        require(
            archive["episode_id"] == summary["episode_id"]
            and archive["workflow"] == runtime["workflow"]
            and archive["worker"] == runtime["worker"],
            "Passed archive audit belongs to a different execution",
        )
        require(
            archive["identities"]["protocol_sha256"] == summary["protocol_sha256"]
            and archive["identities"]["image_library_id"] == library.library_id,
            "Passed archive audit has different protocol/library identities",
        )
        checks = archive["checks"]
        if partial_archive:
            require(
                checks["complete_sealed_case"] is False
                and checks["full_archive_stream_verified"] is True
                and checks["all_available_npy_references_verified"] is True
                and checks["published_small_files_match_archive"] is True
                and checks["available_completed_rollout_arrays_verified"] is True,
                "Partial archive receipt lacks the available-byte verification",
            )
            verified = archive["verified_completed_attempts"]
            require(
                len(verified) == len(completions),
                "Partial archive completed-attempt coverage differs",
            )
            for row, actual in zip(verified, completions, strict=True):
                event = completed[actual["attempt_id"]]
                require(
                    row["attempt_id"] == actual["attempt_id"]
                    and row["arm"] == actual["arm"]
                    and row["revision"] == actual["revision"]
                    and row["success"] == actual["success"]
                    and row["actions"] == actual["actions_executed"]
                    and row["attempt_sha256"] == digest(actual)
                    and row["completed_event_digest"] == digest(event)
                    and row["event_sequence"] == event["sequence"],
                    "Partial archive completed attempt differs from exact event",
                )
        else:
            require(
                checks["complete_sealed_case"] is True
                and checks["all_regular_members_verified"] is True
                and checks["all_npy_references_verified"] is True,
                "Archive receipt lacks the actual-NPY byte verification",
            )
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            if (task_dir / name).exists():
                require(
                    archive["input_file_sha256"][name] == source_files[name]["sha256"],
                    "Archive and provider audit read different task bytes: " + name,
                )
        require(
            archive["counts"]["physical_rollouts"] == len(completions)
            and archive["counts"]["actions"]
            == sum(row["actions_executed"] for row in completions),
            "Archive and provider physical coverage differs",
        )
        for name, expected in archive["worker_metadata_sha256"].items():
            require(
                file_sha(worker_dir / name) == expected,
                "Archive and provider worker metadata differs: " + name,
            )
        archive_binding = {
            "status": archive["status"],
            "scope": "available_partial_archive" if partial_archive else "sealed_case",
            "complete_sealed_case": not partial_archive,
            "receipt_sha256": file_sha(receipt_path),
            "receipt_name": receipt_path.name,
            "archive_sha256": archive["archive"]["sha256"],
            "archive_bytes": archive["archive"]["bytes"],
            "array_inventory_sha256": archive["inventory_sha256"]["arrays"],
        }
        # Final archive byte/source join is checked against the sealed per-file
        # inventory below, regardless of additional numerical receipt fields.
    seal = task_dir / "completion_receipt.json"
    if require_complete:
        require(seal.exists(), "Missing immutable completion seal")
        saved_seal = json_file(seal)
        require(
            saved_seal["episode_id"] == summary["episode_id"]
            and saved_seal["workflow"] == runtime["workflow"]
            and saved_seal["worker"] == runtime["worker"],
            "Completion seal identity differs",
        )
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            path = task_dir / name
            if path.exists():
                require(
                    saved_seal["case_files"][name]["sha256"] == file_sha(path),
                    "Sealed source hash differs: " + name,
                )
        for name, expected in saved_seal["worker_files"].items():
            require(
                file_sha(worker_dir / name) == expected,
                "Sealed worker metadata differs: " + name,
            )
        require(
            all(
                row["entire_file"]
                for name, row in source_files.items()
                if name.endswith(".jsonl")
            ),
            "Final JSONL contains an incomplete suffix",
        )

    per_arm = {
        arm: summarize_arm(arm, completions, providers, executed) for arm in ARMS
    }
    if complete:
        for arm, found in per_arm.items():
            reported = summary["arms"][arm]
            for key, value in (
                ("success", found["observed_success"]),
                ("first_success_revision", found["first_success_revision"]),
                ("censored", found["censored_at_cap"]),
                ("provider", found["provider"]),
                (
                    "provider_through_success_or_cap",
                    found["provider_through_success_or_cap"],
                ),
                ("development_extra_rollouts", found["development_extra_rollouts"]),
            ):
                require(
                    reported[key] == value,
                    "Arm outcome/cost differs: " + arm + "/" + key,
                )
        physical = summary["physical_cost"]
        require(
            physical["rollouts"] == len(completions)
            and physical["actions"]
            == sum(row["actions_executed"] for row in completions),
            "Physical baseline-deduplicated coverage differs",
        )
        require(
            physical["provider"] == summarize_calls(providers),
            "Physical token summary differs",
        )
        require(
            physical["velocity_evaluations"]
            == sum(
                row["velocity_evaluations"]
                + row["parity_velocity_evaluations"]
                + row["probe_velocity_evaluations"]
                for row in completions
            ),
            "Physical solver/gate costs differ",
        )
    if seal.exists():
        source_files["completion_receipt.json"] = {
            "sha256": file_sha(seal),
            "bytes": seal.stat().st_size,
        }
    integration = {
        arm: per_arm[arm]["provider"]["accepted_proposals"] > 0
        and per_arm[arm]["executed"]["accepted_decisions_executed"] > 0
        for arm in ASTRA_ARMS
    }
    completed_attempts = [
        {
            "attempt_id": row["attempt_id"],
            "arm": row["arm"],
            "revision": row["revision"],
            "success": row["success"],
            "actions": row["actions_executed"],
            "completion_event_sequence": completed[row["attempt_id"]]["sequence"],
            "completion_event_canonical_sha256": digest(completed[row["attempt_id"]]),
            "flow_velocity_evaluations": row["velocity_evaluations"],
            "executed": dict(executed[row["attempt_id"]]),
            "provider": summarize_calls(
                [
                    record
                    for record in providers
                    if record.get("attempt_id") == row["attempt_id"]
                ]
            ),
        }
        for row in completions
    ]
    incomplete_attempts = []
    for attempt_id in starts.keys() - completed.keys():
        generation_steps = [row["observation_step"] for row in generations[attempt_id]]
        request_steps = [
            row["observation_step"]
            for row in requests.values()
            if row["attempt_id"] == attempt_id
        ]
        minimum = max([0, *generation_steps, *request_steps])
        recorded_chunk_maximum = min(
            300, minimum + (5 if minimum in generation_steps else 0)
        )
        prefix_actions = Counter()
        prefix_accepted = set()
        for row in generations[attempt_id]:
            step, choice = row["observation_step"], row["active_intervention"]
            count = max(0, min(5, minimum - step))
            history = [
                item
                for item in decisions[attempt_id]
                if item["observation_step"] <= step
            ]
            is_astra = starts[attempt_id]["arm"].startswith("astra_")
            failure = bool(is_astra and history and not history[-1]["accepted"])
            prefix_actions["actions"] += count
            if count and is_astra and choice is not None:
                prefix_accepted.add(choice["decision_id"])
                prefix_actions["accepted_decision_actions"] += count
                prefix_actions["explicit_native_actions"] += count * (
                    choice["mode"] == "native"
                )
            prefix_actions["nonzero_intervention_actions"] += count * bool(
                row["provenance"] and row["provenance"].get("has_effect", False)
            )
            prefix_actions["provider_failure_fallback_actions"] += count * failure
        prefix_actions["accepted_decisions_executed"] = len(prefix_accepted)
        incomplete_attempts.append(
            {
                "attempt_id": attempt_id,
                "arm": starts[attempt_id]["arm"],
                "recorded_generations": len(generation_steps),
                "executed_actions_lower_bound": minimum,
                "executed_actions_upper_bound": 300,
                "recorded_chunk_actions_upper_bound": recorded_chunk_maximum,
                "action_bound_basis": "The latest recorded request/generation step proves the lower bound; the protocol cap is the upper bound because unrecorded interrupted work is unknown.",
                "executed_prefix_lower_bound": dict(prefix_actions),
                "flow_velocity_evaluations_recorded": sum(
                    row["velocity_evaluations"] for row in generations[attempt_id]
                ),
                "outcome": "unknown_no_completed_rollout_record",
            }
        )
    require(
        file_sha(__file__) == auditor_source_hash,
        "Auditor source changed during this invocation",
    )
    return {
        "schema_version": "representation-provider-audit-1.0",
        "status": "complete" if require_complete and complete else "preserved_prefix",
        "complete": bool(require_complete and complete),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": summary["episode_id"],
        "workflow": runtime["workflow"],
        "worker": runtime["worker"],
        "protocol_sha256": summary["protocol_sha256"],
        "image_library_id": library.library_id,
        "source_file_sha256": source_files,
        "source_code_sha256": {name: file_sha(ROOT / name) for name in SOURCES},
        "auditor_sha256": auditor_source_hash,
        "codec": environment,
        "archive_audit": archive_binding,
        "bindings": dict(stats),
        "unmatched": {
            "provider_without_request": len(providers) - len(all_used_provider),
            "request_without_provider": len(requests) - len(all_used_provider),
            "request_awaiting_generation": len(request_raw),
        },
        "provider": summarize_calls(providers),
        "rejections": dict(
            Counter(error_category(row) for row in providers if not row["accepted"])
        ),
        "http_429_taxonomy": {
            "all_http_429": sum(row.get("http_status") == 429 for row in providers),
            "explicit_budget_exceeded": sum(
                row.get("http_status") == 429
                and row.get("provider_error", {}).get("type") == "budget_exceeded"
                for row in providers
            ),
            "http_429_without_explicit_budget_type": sum(
                row.get("http_status") == 429
                and row.get("provider_error", {}).get("type") != "budget_exceeded"
                for row in providers
            ),
            "retry_after_metadata_recorded": False,
            "scope": "Only exact structured error types are classified as budget_exceeded; raw provider messages are omitted.",
        },
        "per_arm": per_arm,
        "completed_attempts": completed_attempts,
        "incomplete_attempts": incomplete_attempts,
        "integration_coverage": integration,
        "integration_all_six": all(integration.values()),
        "physical_completed_rollouts": len(completions),
        "physical_completed_actions": sum(
            row["actions_executed"] for row in completions
        ),
        "all_recorded_actions_bounds": {
            bound: sum(row["actions_executed"] for row in completions)
            + sum(row["executed_actions_" + bound] for row in incomplete_attempts)
            for bound in ("lower_bound", "upper_bound")
        },
        "provider_failure_fallback_actions_lower_bound": sum(
            counts["provider_failure_fallback_actions"] for counts in executed.values()
        )
        + sum(
            row["executed_prefix_lower_bound"].get(
                "provider_failure_fallback_actions", 0
            )
            for row in incomplete_attempts
        ),
        "physical_completed_velocity_evaluations": sum(
            row["velocity_evaluations"]
            + row["parity_velocity_evaluations"]
            + row["probe_velocity_evaluations"]
            for row in completions
        ),
        "recorded_flow_velocity_evaluations": sum(
            row["velocity_evaluations"] for rows in generations.values() for row in rows
        ),
        "recorded_total_velocity_evaluations_lower_bound": sum(
            row["velocity_evaluations"] for rows in generations.values() for row in rows
        )
        + stats["recorded_native_parity_velocity_evaluations"]
        + stats["recorded_probe_velocity_evaluations"],
        "limitations": [
            "No model, physics, numerical counterfactual or new provider call was executed by this auditor.",
            "Decoded request PNGs and robot state bind exact recorded array digests; passed archive audit supplies independent actual-NPY byte verification.",
            "Nonzero actions mean actions generated with a logged nonzero conditioning edit, not proven causal action or success improvement.",
            "Native choices are valid observed behavior; integration admission needs accepted/executed decisions, not success or a nonzero threshold.",
            "Reasoning tokens are contained within output tokens. Missing reported usage remains missing; no monetary rate is invented.",
            "A request without a ledger row can be pending or interrupted; its physical-call status and usage are unknown. Token completeness describes recorded calls only.",
            "Preserved prefixes have no complete-case efficacy claim. Pilot outcomes must remain separate from the full20-task study.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--archive-receipt", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_case(
        args.task_dir,
        args.worker_dir,
        archive_receipt=args.archive_receipt,
        require_complete=args.complete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "episode_id": receipt["episode_id"],
                "calls": receipt["provider"]["provider_calls"],
                "accepted": receipt["provider"]["accepted_proposals"],
                "integration_all_six": receipt["integration_all_six"],
                "output_sha256": file_sha(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
