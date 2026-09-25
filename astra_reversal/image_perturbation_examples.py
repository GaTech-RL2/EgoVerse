"""Export actual, audit-bound policy RGB edits without inference or simulation.

Selection uses development/worker/task order and then event order, never outcome.
The images are original 224px arrays pasted at 1:1; labels stay outside their pixels.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_donor_bank import load_library
from .image_perturbation_audit import (
    ArtifactStore,
    _json,
    replay_image_application,
    require,
)
from .image_perturbations import CAMERAS, ImagePerturbationLimits
from .records import digest, file_sha256

SCHEMA = "image-perturbation-examples-1.0"
ARMS = ("astra_occlusion", "astra_demo_blend")
SELECTION = (
    "Within the supplied completed audited cases: development before evaluation, "
    "then ascending worker index, task ID, state ID and episode ID. For each Astra "
    "arm select the first accepted generation with a nonzero specification and "
    "actual changed pixels, in recorded event sequence. Outcomes are not selection inputs."
)


def _read(path):
    return _json(Path(path).read_text())


def _load_case(directory, audit_path, library):
    store = ArtifactStore(directory)
    audit = _read(audit_path)
    require(
        audit.get("schema_version") == "image-perturbation-audit-1.0"
        and audit.get("status") == "passed"
        and audit.get("complete") is True
        and audit.get("library_id") == library.library_id,
        "A complete passing worker audit for this pinned library is required",
    )
    parent = ArtifactStore(store.directory.parent)
    metadata = audit["source_metadata_sha256"]
    for name, checksum in metadata.items():
        require(
            file_sha256(parent.path(name)) == checksum,
            "Audited worker metadata changed",
        )
    runtime = parent.read_json("runtime.json")
    require(
        all(
            runtime[key] == audit[key]
            for key in ("worker", "phase", "workflow", "payload_sha256")
        )
        and runtime["phase"] in ("development", "evaluation")
        and type(runtime["worker"]) is int
        and runtime["worker"] >= 0,
        "Worker audit/runtime identity differs",
    )
    summary, events = store.read_json("summary.json"), store.lines("events.jsonl")
    matches = [
        row for row in audit["cases"] if row["episode_id"] == summary["episode_id"]
    ]
    require(len(matches) == 1, "Case is missing or duplicated in its worker audit")
    receipt = matches[0]
    require(
        receipt["status"] == "passed"
        and receipt["complete"] is True
        and all(
            receipt[key] is True
            for key in (
                "all_arrays_verified",
                "provider_bindings_verified",
                "reset_pairing_verified",
                "decision_lifetimes_verified",
            )
        )
        and summary["status"] == "complete"
        and receipt["library_id"] == library.library_id
        and summary["image_library"] == library.metadata(),
        "Case is not completely audited against this donor library",
    )
    for name, key in (
        ("summary.json", "summary_sha256"),
        ("events.jsonl", "events_sha256"),
    ):
        require(
            store.files[name] == receipt[key] == receipt["input_file_sha256"][name],
            "Audited case summary/events changed",
        )
    require(
        [row["sequence"] for row in events] == list(range(len(events))),
        "Event sequence is incomplete",
    )
    declarations = [row for row in events if row["kind"] == "case"]
    require(
        len(declarations) == 1 and declarations[0]["sequence"] == 0,
        "Missing case declaration",
    )
    case = declarations[0]
    require(
        case["entry"]["episode_id"] == summary["episode_id"]
        and digest(case["entry"]) == receipt["reset_entry_sha256"]
        and digest(case["protocol"])
        == summary["protocol_sha256"]
        == receipt["protocol_sha256"],
        "Case declaration differs from audited protocol/reset",
    )
    store.register(events)
    evidence = {
        "episode_id": summary["episode_id"],
        "worker": runtime["worker"],
        "phase": runtime["phase"],
        "workflow": runtime["workflow"],
        "payload_sha256": runtime["payload_sha256"],
        "task_id": case["entry"]["task_id"],
        "initial_state_id": case["entry"]["initial_state_id"],
        "instruction": case["entry"]["instruction"],
        "protocol_sha256": receipt["protocol_sha256"],
        "reset_entry_sha256": receipt["reset_entry_sha256"],
        "audit_sha256": file_sha256(audit_path),
        "case_audit_digest": digest(receipt),
        "summary_sha256": store.files["summary.json"],
        "events_sha256": store.files["events.jsonl"],
        "runtime_sha256": metadata["runtime.json"],
    }
    return {
        "store": store,
        "events": events,
        "summary": summary,
        "receipt": receipt,
        "evidence": evidence,
        "audit_path": Path(audit_path),
    }


def _binding(case, generation, attempt):
    name, decision_id = (
        generation["attempt_id"],
        generation["applied_accepted_decision_id"],
    )
    decisions = [
        row
        for row in case["events"]
        if row["kind"] == "image_decision"
        and row["attempt_id"] == name
        and row["decision"].get("proposal") is not None
        and row["decision"]["proposal"]["decision_id"] == decision_id
    ]
    require(len(decisions) == 1, "Applied decision has no unique acceptance event")
    event = decisions[0]
    decision, proposal = event["decision"], event["decision"]["proposal"]
    require(
        decision["accepted"] is True
        and decision in attempt["decisions"]
        and proposal["image_perturbations"] == generation["image_perturbations"],
        "Generation is not bound to its accepted operation",
    )
    fingerprint = proposal["request_fingerprint"]
    requests = [
        row
        for row in case["events"]
        if row["kind"] == "image_request"
        and row["request"]["request_fingerprint"] == fingerprint
    ]
    require(len(requests) == 1, "Decision has no unique recorded request")
    request_event, request = requests[0], requests[0]["request"]
    require(
        digest(
            {
                key: value
                for key, value in request.items()
                if key not in ("request_fingerprint", "request_id")
            }
        )
        == fingerprint
        and request["request_id"] == fingerprint[:16]
        and all(
            proposal[key] == request[key]
            for key in (
                "episode_id",
                "attempt_id",
                "decision_index",
                "observation_step",
                "image_mode",
                "request_id",
            )
        )
        and request["attempt_id"] == name
        and request["episode_id"] == case["evidence"]["episode_id"]
        and request_event["sequence"] < event["sequence"] < generation["sequence"]
        and decision["observation_step"]
        <= generation["observation_step"]
        < event["valid_until_step"],
        "Request/decision/generation identity or lifetime differs",
    )
    bindings = [
        row
        for row in case["receipt"]["bindings"]
        if row["kind"] == "provider"
        and row["request_fingerprint"] == fingerprint
        and row["decision_id"] == decision_id
    ]
    require(
        len(bindings) == 1
        and bindings[0]["provider_call"] is True
        and bindings[0]["accepted"] is True,
        "Selected decision has no verified actual-provider binding",
    )
    return {
        "request_sequence": request_event["sequence"],
        "decision_sequence": event["sequence"],
        "generation_sequence": generation["sequence"],
        "request_id": request["request_id"],
        "request_fingerprint": fingerprint,
        "decision_id": decision_id,
        "decision_index": decision["decision_index"],
        "decision_step": decision["observation_step"],
        "valid_until_step": event["valid_until_step"],
        "provider_binding": copy.deepcopy(bindings[0]),
    }


def _inspect_arm(case, arm, library):
    attempts = {
        row["attempt_id"]: row for row in case["summary"]["arms"][arm]["attempts"][1:]
    }
    decisions = [
        decision for attempt in attempts.values() for decision in attempt["decisions"]
    ]
    counts = {
        "intervention_rollouts": len(attempts),
        "accepted_decisions": sum(row["accepted"] is True for row in decisions),
        "rejected_decisions": sum(row["accepted"] is False for row in decisions),
        "accepted_explicit_noop_decisions": sum(
            row["accepted"] is True
            and not any(
                op.get("alpha", op.get("strength", 0)) > 0
                for op in row["proposal"]["image_perturbations"]
            )
            for row in decisions
        ),
        "accepted_generations": 0,
        "accepted_generations_without_pixel_change": 0,
        "accepted_nonzero_spec_generations_without_pixel_change": 0,
        "accepted_pixel_changing_generations": 0,
        "fallback_generations": 0,
    }
    selected = None
    store, limits = (
        case["store"],
        ImagePerturbationLimits(allowed_kinds=(arm.removeprefix("astra_"),)),
    )
    for generation in case["events"]:
        if generation["kind"] != "image_generation" or generation["mode"] != arm:
            continue
        require(
            generation["attempt_id"] in attempts,
            "Generation has no completed arm attempt",
        )
        if generation["applied_accepted_decision_id"] is None:
            counts["fallback_generations"] += 1
            continue
        attempt = attempts[generation["attempt_id"]]
        binding = _binding(case, generation, attempt)
        raw, edited = (
            store.observation(generation[key])
            for key in ("observation", "modified_observation")
        )
        replay = replay_image_application(
            raw,
            edited,
            generation["image_perturbations"],
            library,
            generation["image_audit"],
            limits=limits,
        )
        require(
            digest(raw) == generation["observation_sha256"]
            and digest(edited) == generation["modified_observation_sha256"]
            and generation["image_has_effect"] is replay["has_effect"]
            and generation["native_condition_fallback"] is False,
            "Generation hashes/effect flags disagree with actual pixels",
        )
        require(
            generation["condition_id"]
            == digest({**edited, "prompt": case["evidence"]["instruction"]}),
            "Native condition does not bind the edited RGB and original instruction",
        )
        counts["accepted_generations"] += 1
        if not replay["has_effect"]:
            counts["accepted_generations_without_pixel_change"] += 1
            counts["accepted_nonzero_spec_generations_without_pixel_change"] += any(
                op.get("alpha", op.get("strength", 0)) > 0
                for op in generation["image_perturbations"]
            )
            continue
        require(
            any(
                op.get("alpha", op.get("strength", 0)) > 0
                for op in generation["image_perturbations"]
            ),
            "Changed pixels have no nonzero operation",
        )
        counts["accepted_pixel_changing_generations"] += 1
        if selected is None:
            outcomes = [
                row
                for row in case["events"]
                if row["kind"] == "image_attempt"
                and row["attempt"]["attempt_id"] == attempt["attempt_id"]
            ]
            require(
                len(outcomes) == 1
                and outcomes[0]["attempt"] == attempt
                and outcomes[0]["sequence"] > generation["sequence"],
                "Selected generation lacks its exact completed outcome",
            )
            selected = {
                "generation": generation,
                "raw": raw,
                "edited": edited,
                "audit": replay,
                "binding": binding,
                "attempt": attempt,
                "outcome_sequence": outcomes[0]["sequence"],
            }
    reason = (
        "selected"
        if selected is not None
        else "no_intervention_rollout"
        if not attempts
        else "no_accepted_decision"
        if not counts["accepted_decisions"]
        else "accepted_decisions_not_executed"
        if not counts["accepted_generations"]
        else "accepted_without_pixel_change"
    )
    return {
        "status": reason,
        "counts": counts,
        "first_generation_sequence": selected["generation"]["sequence"]
        if selected
        else None,
    }, selected


def _save_png(output, relative, pixels):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="PNG")
    return {
        "file": relative,
        "file_sha256": file_sha256(path),
        "pixels_sha256": digest(pixels),
        "shape": list(pixels.shape),
        "dtype": str(pixels.dtype),
    }


def _render(output, arm, case, selected, library):
    generation, attempt, binding = (
        selected["generation"],
        selected["attempt"],
        selected["binding"],
    )
    operations = {row["camera"]: row for row in generation["image_perturbations"]}
    roles = (
        ("raw", "donor", "edited") if arm == "astra_demo_blend" else ("raw", "edited")
    )
    width, height = 16 + len(roles) * 240, 704
    canvas = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=13)
    title = f"{arm} | {case['evidence']['phase']} worker {case['evidence']['worker']}"
    lines = [
        title,
        case["evidence"]["episode_id"],
        f"{attempt['attempt_id']} | step {generation['observation_step']} | event {generation['sequence']}",
        f"Recorded rollout: {'success' if attempt['success'] else 'failure'}, {attempt['actions_executed']} actions",
        "First accepted pixel change; no causal success claim.",
    ]
    for index, line in enumerate(lines):
        draw.text((16, 8 + 17 * index), line, font=font, fill="black")
    camera_records, panes = {}, []
    for row, camera in enumerate(CAMERAS):
        name = "external" if row == 0 else "wrist"
        operation = operations.get(camera)
        donor = (
            library.resolve(operation["donor_id"], camera)
            if operation and operation["kind"] == "demo_blend"
            else None
        )
        images = {
            "raw": selected["raw"][camera],
            "edited": selected["edited"][camera],
            "donor": donor.pixels if donor else None,
        }
        saved = {}
        y = 120 + row * 292
        for column, role in enumerate(roles):
            x = 16 + column * 240
            draw.text((x, y - 20), f"{name}: {role}", font=font, fill="black")
            if images[role] is None:
                draw.text(
                    (x, y + 100),
                    "No donor selected for this camera",
                    font=font,
                    fill="black",
                )
                continue
            pixels = images[role]
            canvas.paste(Image.fromarray(pixels), (x, y))
            saved[role] = _save_png(output, f"{arm}/{name}_{role}.png", pixels)
            panes.append(
                {
                    "camera": camera,
                    "role": role,
                    "xywh": [x, y, 224, 224],
                    "pixels_sha256": digest(pixels),
                }
            )
        caption = "unchanged camera"
        if operation:
            caption = (
                f"{operation['donor_id']}; alpha={operation['alpha']:.6g}"
                if donor
                else f"box={operation['box_xyxy']}; fill={operation['fill_rgb']}; strength={operation['strength']:.6g}"
            )
        draw.text((16, y + 232), caption, font=font, fill="black")
        draw.text(
            (16, y + 251),
            f"changed pixels={selected['audit']['cameras'][camera]['changed_pixels']}; RMS={selected['audit']['cameras'][camera]['rms']:.6g}; Linf={selected['audit']['cameras'][camera]['linf']}",
            font=font,
            fill="black",
        )
        camera_records[camera] = {
            "images": saved,
            "operation": copy.deepcopy(operation),
            "pixel_metrics": copy.deepcopy(selected["audit"]["cameras"][camera]),
            "donor_provenance": donor.provenance if donor else None,
        }
    pixels = np.asarray(canvas).copy()
    for pane in panes:
        x, y, w, h = pane["xywh"]
        require(
            digest(pixels[y : y + h, x : x + w]) == pane["pixels_sha256"],
            "Figure labels changed evidence pixels",
        )
    figure = _save_png(output, f"{arm}/comparison.png", pixels)
    figure["panes"] = panes
    array_sources = {}
    for field in ("observation", "modified_observation"):
        for descriptor in generation[field].values():
            name = descriptor["array"]
            array_sources[name] = {
                "file_sha256": case["store"].files[name],
                "descriptor": copy.deepcopy(descriptor),
            }
    return {
        "status": "selected",
        "case": copy.deepcopy(case["evidence"]),
        "attempt_id": attempt["attempt_id"],
        "observation_step": generation["observation_step"],
        **binding,
        "outcome_sequence": selected["outcome_sequence"],
        "operations": copy.deepcopy(generation["image_perturbations"]),
        "raw_observation_sha256": digest(selected["raw"]),
        "edited_observation_sha256": digest(selected["edited"]),
        "condition_id": generation["condition_id"],
        "recorded_outcome": {
            key: attempt[key]
            for key in ("success", "terminated", "actions_executed", "status")
        },
        "cameras": camera_records,
        "array_sources": array_sources,
        "figure": figure,
    }


def export_examples(case_dirs, *, audit_paths, library, output):
    """Export new files from one or more completed cases and full worker receipts."""
    require(
        case_dirs and len(case_dirs) == len(audit_paths),
        "Pair every case with one worker audit receipt",
    )
    output = Path(output)
    if isinstance(library, (str, Path)):
        require(
            not output.resolve().is_relative_to(Path(library).resolve()),
            "Export must not modify the pinned donor library",
        )
        library = load_library(library)
    require(not output.exists(), "Refusing to replace an existing example export")
    cases = [
        _load_case(path, receipt, library)
        for path, receipt in zip(case_dirs, audit_paths, strict=True)
    ]
    require(
        len({row["store"].directory for row in cases}) == len(cases)
        and len(
            {
                (row["evidence"]["episode_id"], row["evidence"]["reset_entry_sha256"])
                for row in cases
            }
        )
        == len(cases),
        "Duplicate case in selection universe",
    )
    require(
        all(
            not output.resolve().is_relative_to(row["store"].directory) for row in cases
        ),
        "Export must not modify source case directories",
    )
    cases.sort(
        key=lambda row: (
            row["evidence"]["phase"] != "development",
            *(
                row["evidence"][key]
                for key in ("worker", "task_id", "initial_state_id", "episode_id")
            ),
        )
    )
    considered, chosen = [], {}
    for case in cases:
        row = {**case["evidence"], "arms": {}}
        for arm in ARMS:
            row["arms"][arm], selected = _inspect_arm(case, arm, library)
            if selected is not None and arm not in chosen:
                chosen[arm] = case, selected
        considered.append(row)
        for name, checksum in case["store"].files.items():
            require(
                checksum == case["receipt"]["input_file_sha256"].get(name)
                and file_sha256(case["store"].path(name)) == checksum,
                "Example source differs from completed audit",
            )
        require(
            file_sha256(case["audit_path"]) == case["evidence"]["audit_sha256"],
            "Audit receipt changed during export",
        )
    output.mkdir(parents=True, exist_ok=False)
    examples = {}
    for arm in ARMS:
        if arm in chosen:
            case, selected = chosen[arm]
            examples[arm] = _render(output, arm, case, selected, library)
        else:
            examples[arm] = {
                "status": "no_qualifying_generation",
                "reason": "See each supplied case's explicit no-op, rejection and execution counts.",
            }
    report = {
        "schema_version": SCHEMA,
        "status": "complete",
        "selection_rule": SELECTION,
        "library_id": library.library_id,
        "considered_cases": considered,
        "examples": examples,
        "scope": "Audited recorded RGB evidence only. No inference/provider/simulator calls; no fabricated edits, task-outcome replay, or causal performance claim. Full audit receipts cover the complete runs; this exporter rechecks accepted Astra RGB generations and bound metadata, not all unselected action/noise arrays.",
        "source_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in (
                "image_perturbation_examples.py",
                "image_perturbation_audit.py",
                "image_perturbations.py",
                "image_donor_bank.py",
                "records.py",
            )
        },
    }
    (output / "examples.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, required=True)
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_examples(
        args.case, audit_paths=args.audit, library=args.library, output=args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": sum(
                    row["status"] == "selected" for row in result["examples"].values()
                ),
                "report_sha256": file_sha256(args.output / "examples.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
