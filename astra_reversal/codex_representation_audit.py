"""Offline Codex VEI/VLI request, history and application audit.

Observation and operator joins are adapted from the existing strict private
representation provider audit. The HTTP auditor remains unchanged. This module
requires a separate passed archive/NPY proof before certifying a complete case.
It does not replay learned hidden states, policy inference or simulator physics.
"""

import base64
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from .agent import CAMERAS
from .astra_client import _strict_json
from .codex_accounting import summarize_codex_calls
from .codex_provider_audit import verify_codex_provider
from .image_donor_bank import load_library
from .interpolation_catalog import donor_catalog
from .records import digest, file_sha256
from .representation_agent import _WIRE_FIELDS, build_request, parse_proposal
from .representation_search import VISION_ARMS, keyed_rng, load_protocol, random_choice


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return _strict_json(Path(path).read_bytes())


def read_prefix(path):
    """Retain exact valid JSONL prefix; malformed terminated lines are errors."""
    path = Path(path)
    raw = path.read_bytes() if path.exists() else b""
    rows, consumed = [], 0
    for line in raw.splitlines(keepends=True):
        try:
            row = _strict_json(line)
            require(isinstance(row, dict), "Event/ledger line must be an object")
        except ValueError:
            require(not line.endswith(b"\n"), "Malformed complete JSONL line")
            break
        rows.append(row)
        consumed += len(line)
    return rows, {
        "sha256": hashlib.sha256(raw[:consumed]).hexdigest(),
        "bytes": consumed,
        "records": len(rows),
        "entire_file": consumed == len(raw),
    }


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


def bind_archive(
    receipt_path, task_dir, worker_dir, summary, sources, source_identity, completed
):
    """Bind the independent archive proof to these exact bytes and source cohort."""
    if receipt_path is None:
        return {
            "verified": False,
            "complete_sealed_case": False,
            "reason": "archive_receipt_missing",
        }
    archive = read_json(receipt_path)
    require(
        archive.get("status") in ("passed", "verified_partial_archive"),
        "Archive audit did not pass",
    )
    require(
        archive.get("schema_version")
        in (
            "representation-case-array-audit-1",
            "representation-partial-array-audit-1",
        ),
        "Unknown representation archive audit",
    )
    partial = archive["schema_version"] == "representation-partial-array-audit-1"
    require(
        archive["status"] == ("verified_partial_archive" if partial else "passed"),
        "Archive status/schema scope differs",
    )
    runtime_identity = read_json(Path(worker_dir) / "runtime.json")
    require(
        archive["workflow"] == runtime_identity["workflow"]
        and archive["worker"] == runtime_identity["worker"],
        "Archive belongs to another worker execution",
    )
    require(
        {
            "runtime.json",
            "protocol.json",
            "checkpoint.json",
            "reset_manifest.json",
            "frozen_plan.json",
            "frozen_weights_before.json",
        }
        <= set(archive["worker_metadata_sha256"]),
        "Archive omits required worker metadata identities",
    )
    require(
        archive["episode_id"] == summary["episode_id"],
        "Archive belongs to another episode",
    )
    for name in ("summary.json", "events.jsonl", "provider.jsonl"):
        if name in archive["input_file_sha256"]:
            require(
                archive["input_file_sha256"][name] == sources[name]["sha256"]
                and sources[name].get("entire_file", True),
                "Archive/task byte identity differs: " + name,
            )
        else:
            require(
                name == "provider.jsonl" and sources[name]["bytes"] == 0,
                "Archive omits required task evidence",
            )
    for name, checksum in archive["worker_metadata_sha256"].items():
        require(
            Path(name).name == name
            and file_sha256(Path(worker_dir) / name) == checksum,
            "Archive worker metadata differs",
        )
    require(
        archive["counts"]["physical_rollouts"] == len(completed)
        and archive["counts"]["actions"]
        == sum(row["actions_executed"] for row in completed),
        "Archive completed physical coverage differs",
    )
    identities = archive["identities"]
    require(
        identities["protocol_sha256"] == summary["protocol_sha256"]
        and identities["image_library_id"] == summary["image_library_id"],
        "Archive protocol/library identity differs",
    )
    if source_identity is None:
        return {
            "verified": False,
            "complete_sealed_case": False,
            "reason": "source_identity_missing",
            "receipt_sha256": file_sha256(receipt_path),
        }
    source = read_json(source_identity)
    runtime = read_json(Path(worker_dir) / "runtime.json")
    require(
        source["payload_sha256"]
        == identities["payload_sha256"]
        == runtime["payload_sha256"]
        and source["source_revision"] == identities["source_revision"],
        "Archive source/payload identity differs",
    )
    checks = archive["checks"]
    if partial:
        require(
            checks["complete_sealed_case"] is False
            and checks["full_archive_stream_verified"] is True
            and checks["all_available_npy_references_verified"] is True
            and checks["published_small_files_match_archive"] is True
            and checks["available_completed_rollout_arrays_verified"] is True,
            "Partial archive byte proof incomplete",
        )
    else:
        require(
            checks["complete_sealed_case"] is True
            and checks["all_regular_members_verified"] is True
            and checks["all_npy_references_verified"] is True,
            "Archive NPY/seal proof incomplete",
        )
        seal = read_json(Path(task_dir) / "completion_receipt.json")
        require(
            seal["episode_id"] == summary["episode_id"]
            and seal["workflow"] == runtime["workflow"]
            and seal["worker"] == runtime["worker"],
            "Completion seal identity differs",
        )
        for name in ("summary.json", "events.jsonl", "provider.jsonl"):
            if (Path(task_dir) / name).exists():
                require(
                    seal["case_files"][name]["sha256"] == sources[name]["sha256"],
                    "Completion seal task bytes differ",
                )
        for name, checksum in seal["worker_files"].items():
            require(
                Path(name).name == name
                and file_sha256(Path(worker_dir) / name) == checksum,
                "Completion seal worker bytes differ",
            )
    return {
        "verified": True,
        "complete_sealed_case": not partial,
        "receipt_sha256": file_sha256(receipt_path),
        "archive_sha256": archive["archive"]["sha256"],
        "source_identity_sha256": file_sha256(source_identity),
        "identities": identities,
    }


def audit_case(
    task_dir,
    worker_dir,
    *,
    jobs_root=None,
    archive_receipt=None,
    source_identity=None,
    library=None,
    require_complete=False,
):
    """Return a complete proof or explicitly scoped preserved-prefix evidence."""
    task_dir, worker_dir = Path(task_dir), Path(worker_dir)
    summary = read_json(task_dir / "summary.json")
    protocol = load_protocol(worker_dir / "protocol.json")
    require(
        protocol["schema_version"] == "vision-codex-representation-screen-1.0"
        and protocol["arms"] == list(VISION_ARMS),
        "Only the Codex five-arm vision cohort is supported",
    )
    runtime = read_json(worker_dir / "runtime.json")
    require(
        runtime["phase"] in ("development", "evaluation")
        and summary["development"] == (runtime["phase"] == "development"),
        "Representation phase differs",
    )
    expected = load_protocol(
        Path(__file__).with_name("configs")
        / "vision_codex_representation_screen_v1.json"
    )
    if summary["development"]:
        expected["seed"] = expected["development_seed"]
    require(
        protocol == expected
        and summary["protocol_sha256"] == digest(protocol)
        and summary["schema_version"] == protocol["schema_version"],
        "Immutable Codex protocol differs",
    )
    require(
        runtime["packages"]["numpy"] == np.__version__
        and runtime["packages"]["pillow"] == PIL.__version__,
        "Use the worker-matched audit codec environment",
    )
    library = library or load_library(
        Path(__file__).parent / ".deps/image-perturbations/donors"
    )
    require(
        summary["image_library_id"] == library.library_id,
        "Frozen donor library differs",
    )
    sources = {
        "summary.json": {
            "sha256": file_sha256(task_dir / "summary.json"),
            "bytes": (task_dir / "summary.json").stat().st_size,
        }
    }
    for name in (
        "protocol.json",
        "runtime.json",
        "prompts.json",
        "checkpoint.json",
        "reset_manifest.json",
        "frozen_plan.json",
        "image_library.json",
    ):
        path = worker_dir / name
        if path.exists():
            sources["worker/" + name] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    if source_identity is not None:
        identity = read_json(source_identity)
        require(
            identity["payload_sha256"] == runtime["payload_sha256"],
            "Runtime/source payload differs",
        )
        sources["source_identity.json"] = {
            "sha256": file_sha256(source_identity),
            "bytes": Path(source_identity).stat().st_size,
        }
    events, sources["events.jsonl"] = read_prefix(task_dir / "events.jsonl")
    providers, sources["provider.jsonl"] = read_prefix(task_dir / "provider.jsonl")
    by_fp = {}
    for record in providers:
        require(
            record.get("backend") == "codex_relay"
            and record["episode_id"] == summary["episode_id"],
            "Cross-backend/case provider row",
        )
        require(
            record["request_fingerprint"] not in by_fp,
            "Duplicate provider invocation for one request",
        )
        by_fp[record["request_fingerprint"]] = record
    starts, completed, generations, decisions = (
        {},
        {},
        defaultdict(list),
        defaultdict(list),
    )
    active, failed, requests, raw_pending, request_slots = {}, {}, {}, {}, {}
    proofs, used, completions = [], set(), []
    current = None
    for index, event in enumerate(events):
        require(
            event["sequence"] == index, "Non-contiguous representation event sequence"
        )
        kind = event["kind"]
        if kind == "representation_rollout_start":
            aid, arm, revision = event["attempt_id"], event["arm"], event["revision"]
            require(current is None and aid not in starts, "Overlapping/reused rollout")
            require(
                arm in ("native", *VISION_ARMS)
                and aid == f"{arm}_revision{revision}"
                and (
                    (arm == "native" and revision == 0)
                    or (arm != "native" and revision in (1, 2))
                ),
                "Invalid representation attempt identity",
            )
            require(
                event["entry"]["episode_id"] == summary["episode_id"]
                and event["entry"]["instruction"] == summary["instruction"]
                and digest(event["entry"]) == summary["reset_entry_sha256"],
                "Rollout reset/task identity differs",
            )
            starts[aid], active[aid], failed[aid], current = event, None, False, aid
        elif kind == "representation_request":
            req = event["request"]
            aid, step, fp = (
                req["attempt_id"],
                req["observation_step"],
                req["request_fingerprint"],
            )
            require(
                aid == current and starts[aid]["arm"].startswith("astra_"),
                "Request outside own Astra rollout",
            )
            arm = starts[aid]["arm"]
            require(
                fp not in requests
                and (aid, req["decision_index"]) not in request_slots,
                "Duplicated request slot",
            )
            require(
                req["episode_id"] == summary["episode_id"]
                and req["target_task"] == summary["instruction"]
                and req["representation_mode"] == arm.removeprefix("astra_"),
                "Request task/arm differs",
            )
            history = [row for row in completions if row["arm"] in ("native", arm)]
            require(
                history
                and req["completed_rollout_feedback"]
                == [feedback(row) for row in history],
                "Request contains missing/cross-arm completed feedback",
            )
            previous = req["previous_attempt"]
            latest = completed[history[-1]["attempt_id"]]
            require(
                previous is not None
                and previous["feedback"] == feedback(latest["attempt"])
                and previous["decisions"] == latest["attempt"]["decisions"][-2:]
                and len(previous["snapshots"]) == len(latest["snapshots"]),
                "Previous own-rollout history differs",
            )
            for shown, recorded in zip(
                previous["snapshots"], latest["snapshots"], strict=True
            ):
                require(
                    shown["step"] == recorded["step"]
                    and shown["label"] == recorded["label"],
                    "Prior snapshot timing differs",
                )
                match_observation(
                    observed_descriptor(shown["observation"]), recorded["observation"]
                )
            expected_steps = (
                [row["observation_step"] for row in generations[aid]] + [step]
            )[-4:]
            require(
                [row["step"] for row in req["observations"]] == expected_steps,
                "Current raw frame history differs",
            )
            descriptions = [
                observed_descriptor(row["observation"]) for row in req["observations"]
            ]
            for description, prior_step in zip(
                descriptions[:-1], expected_steps[:-1], strict=True
            ):
                prior = next(
                    row
                    for row in generations[aid]
                    if row["observation_step"] == prior_step
                )
                match_observation(description, prior["observation"])
            require((aid, step) not in raw_pending, "Duplicated current request frame")
            raw_pending[aid, step] = descriptions[-1]
            rebuilt = build_request(
                episode_id=summary["episode_id"],
                attempt_id=aid,
                decision_index=len(decisions[aid]) + 1,
                observation_step=len(decisions[aid]) * 25,
                representation_mode=arm.removeprefix("astra_"),
                target_task=summary["instruction"],
                source_catalog=donor_catalog(),
                donor_catalog=library.catalog(),
                contact_sheets=library.contact_sheets(),
                observations=req["observations"],
                previous_decisions=decisions[aid][-2:],
                completed_rollout_feedback=[feedback(row) for row in history],
                previous_attempt=previous,
                action_spec=starts[aid]["action_spec"],
                call_interval=25,
                execute_steps=5,
                max_calls=12,
                action_budget=300,
            )
            require(
                rebuilt == req,
                "Reconstructed request/catalog/contact-sheet/history differs",
            )
            requests[fp] = req
            request_slots[aid, req["decision_index"]] = fp
        elif kind in ("representation_decision", "random_representation_decision"):
            aid, decision = event["attempt_id"], event["decision"]
            require(
                aid == current
                and decision["decision_index"] == len(decisions[aid]) + 1
                and decision["observation_step"] == len(decisions[aid]) * 25,
                "Decision schedule/attempt differs",
            )
            if kind == "representation_decision":
                fp = request_slots.get((aid, decision["decision_index"]))
                require(fp in requests, "Decision has no original request")
                if decision["accepted"]:
                    bound = decision["proposal"]
                    require(
                        isinstance(bound, dict), "Accepted decision lacks a proposal"
                    )
                    parsed = parse_proposal(
                        {
                            key: value
                            for key, value in bound.items()
                            if key in _WIRE_FIELDS
                        },
                        requests[fp],
                    )
                    require(
                        parsed == bound,
                        "Logged proposal fails its original request binding",
                    )
                if fp in by_fp:
                    row = by_fp[fp]
                    require(
                        row["accepted"] == decision["accepted"]
                        and (
                            decision["error"] is None
                            if row["accepted"]
                            else decision["error"] == row["error"]
                        ),
                        "Provider/decision acceptance differs",
                    )
                    job = (
                        None
                        if jobs_root is None
                        else Path(jobs_root) / row["invocation_id"]
                    )
                    if job is not None:
                        require(
                            job.resolve().parent == Path(jobs_root).resolve(),
                            "Invocation escapes job root",
                        )
                    proofs.append(
                        verify_codex_provider(
                            requests[fp],
                            row,
                            decision["proposal"],
                            protocol["astra"],
                            job_directory=job,
                        )
                    )
                    used.add(fp)
            else:
                start = starts[aid]
                require(
                    start["arm"].startswith("random_"),
                    "Random decision in an Astra/native arm",
                )
                expected_choice = random_choice(
                    start["arm"].removeprefix("random_"),
                    keyed_rng(
                        start["entry"],
                        start["revision"],
                        decision["observation_step"],
                        1,
                    ),
                    [row["source_id"] for row in donor_catalog()],
                    [row["donor_id"] for row in library.catalog()],
                )
                require(
                    decision["proposal"] == expected_choice
                    and decision["accepted"] is True,
                    "Random control does not follow prescribed matched stream",
                )
            require(
                decision["accepted"] == (decision["proposal"] is not None),
                "Decision acceptance/proposal differs",
            )
            decisions[aid].append(decision)
            active[aid], failed[aid] = decision["proposal"], not decision["accepted"]
        elif kind == "representation_generation":
            aid, step = event["attempt_id"], event["observation_step"]
            require(
                aid == current and step == len(generations[aid]) * 5,
                "Generation step/attempt differs",
            )
            arm = starts[aid]["arm"]
            if arm.startswith(("astra_", "random_")):
                require(
                    decisions[aid]
                    and decisions[aid][-1]["observation_step"] == (step // 25) * 25,
                    "Generation lacks its scheduled fresh decision",
                )
            require(
                event["active_intervention"] == active[aid],
                "Generation used stale/different proposal",
            )
            match_application(
                active[aid],
                event["provenance"],
                arm.removeprefix("astra_").removeprefix("random_"),
                summary["instruction"],
                library,
            )
            if failed[aid]:
                require(
                    active[aid] is None and event["provenance"] is None,
                    "Failed call retained a stale intervention",
                )
            if (aid, step) in raw_pending:
                match_observation(raw_pending.pop((aid, step)), event["observation"])
            generations[aid].append(event)
        elif kind == "representation_rollout_complete":
            attempt, aid = event["attempt"], event["attempt"]["attempt_id"]
            require(
                aid == current
                and aid not in completed
                and attempt["decisions"] == decisions[aid],
                "Completion/decision history differs",
            )
            start = starts[aid]
            require(
                attempt["arm"] == start["arm"]
                and attempt["revision"] == start["revision"],
                "Completed attempt identity differs",
            )
            require(
                type(attempt["success"]) is bool
                and attempt["initial_success"] is False
                and 0 < attempt["actions_executed"] <= 300
                and (
                    attempt["success"]
                    or attempt["terminated"]
                    or attempt["actions_executed"] == 300
                ),
                "Incomplete rollout cannot be counted as a failure",
            )
            require(
                len(generations[aid]) == (attempt["actions_executed"] + 4) // 5,
                "Completed action/generation coverage differs",
            )
            require(
                attempt["status"]
                == (
                    "success"
                    if attempt["success"]
                    else "terminated"
                    if attempt["terminated"]
                    else "budget_exhausted"
                ),
                "Completion status differs from simulator outcome",
            )
            completed[aid] = event
            completions.append(attempt)
            current = None
        else:
            require(
                kind in ("case", "native_parity", "weighted_vision_probe"),
                "Unexpected representation event",
            )
    return _finish(
        task_dir,
        worker_dir,
        summary,
        protocol,
        runtime,
        sources,
        providers,
        starts,
        completed,
        completions,
        generations,
        decisions,
        requests,
        used,
        proofs,
        raw_pending,
        archive_receipt,
        source_identity,
        require_complete,
    )


def _execution(rows, decision_rows, arm, actions):
    counts, accepted = Counter(), set()
    generated = []
    for row in rows:
        step, choice = row["observation_step"], row["active_intervention"]
        count = max(0, min(5, actions - step))
        history = [d for d in decision_rows if d["observation_step"] <= step]
        failure = bool(
            arm.startswith("astra_") and history and not history[-1]["accepted"]
        )
        effect = bool(row["provenance"] and row["provenance"].get("has_effect", False))
        generated.append(
            {
                "step": step,
                "choice": choice,
                "provider_failure_fallback": failure,
                "has_effect": effect,
            }
        )
        counts["actions"] += count
        if count and arm.startswith("astra_") and choice is not None:
            accepted.add(choice["decision_id"])
            counts["accepted_decision_actions"] += count
            counts["explicit_native_actions"] += count * (choice["mode"] == "native")
        counts["nonzero_intervention_actions"] += count * effect
        counts["provider_failure_fallback_actions"] += count * failure
    counts["accepted_decisions_executed"] = len(accepted)
    return counts, generated


def _finish(
    task_dir,
    worker_dir,
    summary,
    protocol,
    runtime,
    sources,
    providers,
    starts,
    completed,
    completions,
    generations,
    decisions,
    requests,
    used,
    proofs,
    raw_pending,
    archive_receipt,
    source_identity,
    require_complete,
):
    from .representation_agent import summarize_calls
    from .representation_search import summarize_attempts

    unresolved_completed_calls = 0
    outcomes = []
    for attempt in completions:
        aid, arm = attempt["attempt_id"], attempt["arm"]
        counts, generated = _execution(
            generations[aid], decisions[aid], arm, attempt["actions_executed"]
        )
        require(
            generated == attempt["generations"], "Completed generation summaries differ"
        )
        for field in (
            "accepted_decisions_executed",
            "accepted_decision_actions",
            "explicit_native_actions",
            "nonzero_intervention_actions",
            "provider_failure_fallback_actions",
        ):
            require(
                attempt[field] == counts[field],
                "Completed action attribution differs: " + field,
            )
        require(
            attempt["velocity_evaluations"]
            == sum(row["velocity_evaluations"] for row in generations[aid]),
            "Completed flow cost differs from generation records",
        )
        recorded = [r for r in providers if r["attempt_id"] == aid]
        require(
            recorded == attempt["provider_records"][: len(recorded)],
            "Completed provider ledger differs",
        )
        require(
            len(attempt["provider_records"]) == len(decisions[aid])
            if arm.startswith("astra_")
            else not attempt["provider_records"],
            "Completed provider call coverage differs",
        )
        unresolved_completed_calls += len(attempt["provider_records"]) - len(recorded)
        outcomes.append(
            {
                "attempt_id": aid,
                "arm": arm,
                "revision": attempt["revision"],
                "simulator_success": attempt["success"],
                "actions": attempt["actions_executed"],
                "event_digest": digest(completed[aid]),
                "executed": dict(counts),
                "rollout_wall_seconds": attempt["wall_seconds"],
            }
        )
    summary_rows = summary["physical_rollouts"]
    overlap = min(len(summary_rows), len(completions))
    require(
        summary_rows[:overlap] == completions[:overlap],
        "Summary and completed event prefixes differ",
    )
    incomplete = []
    for aid in starts.keys() - completed.keys():
        steps = [row["observation_step"] for row in generations[aid]]
        steps += [
            row["observation_step"]
            for row in requests.values()
            if row["attempt_id"] == aid
        ]
        lower = max([0, *steps])
        counts, _ = _execution(
            generations[aid], decisions[aid], starts[aid]["arm"], lower
        )
        incomplete.append(
            {
                "attempt_id": aid,
                "arm": starts[aid]["arm"],
                "outcome": "unknown_no_completed_rollout_record",
                "executed_actions_lower_bound": lower,
                "executed_actions_upper_bound": 300,
                "executed_prefix_lower_bound": dict(counts),
                "recorded_generations": len(generations[aid]),
                "recorded_flow_velocity_evaluations": sum(
                    row["velocity_evaluations"] for row in generations[aid]
                ),
            }
        )
    event_complete = (
        summary["status"] == "complete"
        and not incomplete
        and not raw_pending
        and len(summary_rows) == len(completions)
        and len(used) == len(requests) == len(providers)
        and unresolved_completed_calls == 0
        and all(r.get("entire_file", True) for r in sources.values())
    )
    if event_complete:
        require(
            completions and completions[0]["arm"] == "native", "Shared baseline missing"
        )
        baseline = completions[0]
        require(summary["baseline"] == baseline, "Shared baseline summary differs")
        expected_order = [baseline["attempt_id"]]
        for arm in VISION_ARMS:
            selected = [row for row in completions if row["arm"] == arm]
            expected_revisions = []
            prior = [baseline]
            for revision in (1, 2):
                if any(row["success"] for row in prior) and not (
                    summary["development"] and revision == 1
                ):
                    break
                expected_revisions.append(revision)
                require(
                    len(selected) >= len(expected_revisions),
                    "Prescribed arm revision is missing",
                )
                prior.append(selected[len(expected_revisions) - 1])
            require(
                [row["revision"] for row in selected] == expected_revisions,
                "Arm stopping/revision coverage differs",
            )
            expected_order.extend(row["attempt_id"] for row in selected)
            require(
                summary["arms"][arm] == summarize_attempts([baseline, *selected], 2),
                "Arm outcome/cost summary differs",
            )
        require(
            [row["attempt_id"] for row in completions] == expected_order,
            "Arm execution order changed",
        )
        physical = summary["physical_cost"]
        require(
            physical["provider"] == summarize_calls(providers)
            and physical["rollouts"] == len(completions)
            and physical["actions"]
            == sum(row["actions_executed"] for row in completions)
            and physical["velocity_evaluations"]
            == sum(
                row["velocity_evaluations"]
                + row["parity_velocity_evaluations"]
                + row["probe_velocity_evaluations"]
                for row in completions
            ),
            "Completed physical cost differs",
        )
    archive = bind_archive(
        archive_receipt,
        task_dir,
        worker_dir,
        summary,
        sources,
        source_identity,
        completions,
    )
    local_complete = len(proofs) == len(providers) == len(requests) and all(
        row["status"] == "passed" for row in proofs
    )
    complete = (
        event_complete
        and local_complete
        and archive["verified"]
        and archive["complete_sealed_case"]
    )
    if require_complete:
        require(
            complete,
            "Complete certification needs matching archive, source, event, provider and local job proofs",
        )
    return {
        "schema_version": "codex-representation-provider-audit-1.0",
        "status": "complete" if complete else "preserved_prefix",
        "complete": complete,
        "episode_id": summary["episode_id"],
        "task_id": summary["task_id"],
        "workflow": runtime["workflow"],
        "worker": runtime["worker"],
        "protocol_sha256": summary["protocol_sha256"],
        "image_library_id": summary["image_library_id"],
        "source_file_sha256": sources,
        "archive_audit": archive,
        "local_job_proofs": proofs,
        "local_job_proofs_complete": local_complete,
        "arrays_verified": archive["verified"],
        "event_coverage_complete": event_complete,
        "completed_attempts": outcomes,
        "incomplete_attempts": incomplete,
        "completed_actions": sum(row["actions_executed"] for row in completions),
        "unfinished_actions_lower_bound": sum(
            row["executed_actions_lower_bound"] for row in incomplete
        ),
        "completed_rollout_wall_seconds": sum(
            row["wall_seconds"] for row in completions
        ),
        "provider": summarize_codex_calls(providers),
        "pending_request_frames": len(raw_pending),
        "unjoined_provider_rows": len(providers) - len(used),
        "unpublished_completed_provider_rows": unresolved_completed_calls,
        "summary_only_unverified_rollouts": max(
            0, len(summary_rows) - len(completions)
        ),
        "application_generation_bindings": sum(
            len(rows) for rows in generations.values()
        ),
        "source_code_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "codex_provider_audit.py",
                "codex_accounting.py",
                "representation_agent.py",
                "representation_search.py",
            )
        },
        "limitations": [
            "Observed simulator outcomes, not inferred visual judgments.",
            "Only an exact passed archive receipt establishes recorded NPY bytes/reset/noise/operator proofs.",
            "No policy hidden-state computation or simulator physics is independently rerun.",
            "Incomplete rollouts have unknown outcomes; preserved action/token counts are lower bounds.",
            "Provider waits are included in rollout wall time and must not be added twice.",
            "Unpublished local jobs are outside this worker-ledger receipt and must be accounted separately.",
        ],
    }
