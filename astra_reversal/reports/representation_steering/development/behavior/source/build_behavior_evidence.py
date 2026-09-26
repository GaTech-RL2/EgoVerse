"""Publish allowlisted, archive-bound interrupted-pilot observations; no calls."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from astra_reversal.agent import CAMERAS
from astra_reversal.image_donor_bank import load_library
from astra_reversal.interpolation_catalog import donor_catalog
from astra_reversal.records import digest
from astra_reversal.representation_agent import summarize_calls

CASES = (
    (0, 6, "wine", ("astra_tei_revision1", "astra_tei_revision2")),
    (1, 2, "milk", ("astra_tei_revision2", "astra_tli_revision1")),
    (2, 8, "cabinet", ("astra_vei_revision1", "astra_vli_revision1")),
)
FRAMES = {
    "wine": (
        ("astra_tei_revision1", 100),
        ("astra_tei_revision1", 275),
        ("astra_tei_revision2", 50),
        ("astra_tei_revision2", 75),
    ),
    "milk": (
        ("astra_tei_revision2", 50),
        ("astra_tei_revision2", 275),
        ("astra_tli_revision1", 50),
        ("astra_tli_revision1", 100),
    ),
    "cabinet": (
        ("astra_vei_revision1", 100),
        ("astra_vei_revision1", 175),
        ("astra_vei_revision1", 275),
        ("astra_vli_revision1", 100),
    ),
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def task_path(run_root, worker, task):
    for kind in ("audited", "partial_audited"):
        directory = run_root / kind / f"worker_{worker}" / f"task_{task}"
        if (directory / "audit.json").exists():
            return directory
    raise ValueError(f"No immutable archive audit for worker {worker}")


def decode(value):
    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(value["data"], validate=True)))
    ).copy()


def content_identity(value):
    """Compare verified array content, excluding attempt-specific storage paths."""
    if isinstance(value, dict):
        if all(key in value for key in ("sha256", "shape", "dtype")):
            return {key: value[key] for key in ("sha256", "shape", "dtype")}
        return {key: content_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [content_identity(item) for item in value]
    return value


def bound_case(run_root, receipt_dir, worker, task, label, selected_attempts, output):
    directory = task_path(run_root, worker, task)
    numeric_path = directory / "audit.json"
    numeric = read(numeric_path)
    provider_path = receipt_dir / f"worker_{worker}.json"
    provider = read(provider_path)
    need(
        numeric["status"] in ("passed", "verified_partial_archive")
        and provider["archive_audit"]["receipt_sha256"] == sha(numeric_path),
        "Provider evidence does not bind the available archive audit",
    )
    for name in ("summary.json", "events.jsonl", "provider.jsonl"):
        actual = sha(directory / name)
        need(
            actual
            == provider["source_file_sha256"][name]["sha256"]
            == numeric["input_file_sha256"][name],
            "Source bytes differ from audited files: " + name,
        )
    summary = read(directory / "summary.json")
    ledger = list(rows(directory / "provider.jsonl"))
    physical = {
        row["request_fingerprint"]: (index, row) for index, row in enumerate(ledger)
    }
    completed = {row["attempt_id"]: row for row in provider["completed_attempts"]}
    source_prompts = {row["source_id"]: row["prompt"] for row in donor_catalog()}
    requests, decisions, generations, figures = {}, {}, {}, {}
    all_generations = {}
    for event in rows(directory / "events.jsonl"):
        kind = event["kind"]
        if kind == "representation_request":
            request = event["request"]
            key = request["attempt_id"], request["observation_step"]
            if key[0] not in selected_attempts:
                continue
            if key in FRAMES[label]:
                figures[key] = request["observations"][-1]
            current = request["observations"][-1]
            requests[key] = {
                "request_id": request["request_id"],
                "request_fingerprint": request["request_fingerprint"],
                "request_event_sequence": event["sequence"],
                "request_event_canonical_sha256": digest(event),
                "raw_observation_step": current["step"],
                "raw_camera_sha256": {
                    camera: digest(decode(current["observation"][camera]))
                    for camera in CAMERAS
                },
                "raw_png_sha256": {
                    camera: hashlib.sha256(
                        base64.b64decode(current["observation"][camera]["data"])
                    ).hexdigest()
                    for camera in CAMERAS
                },
                "completed_feedback": request["completed_rollout_feedback"],
                "previous_attempt_id": (
                    request["previous_attempt"]["feedback"]["attempt_id"]
                    if request["previous_attempt"] is not None
                    else None
                ),
                "recent_decision_indices": [
                    row["decision_index"] for row in request["previous_decisions"]
                ],
            }
        elif kind == "representation_decision":
            if event["attempt_id"] in selected_attempts:
                d = event["decision"]
                decisions[event["attempt_id"], d["observation_step"]] = d
        elif kind == "representation_generation":
            key = event["attempt_id"], event["observation_step"]
            all_generations[key] = {
                name: content_identity(event[name])
                for name in (
                    "observation",
                    "condition_id",
                    "noise",
                    "generated_actions",
                    "controller_actions",
                )
            }
            if event["attempt_id"] in selected_attempts:
                generations[key] = event
    timeline = []
    for key, request in requests.items():
        attempt_id, step = key
        need(attempt_id in completed, "Illustrative attempt has no verified completion")
        attempt, decision = completed[attempt_id], decisions[key]
        index, call = physical[request["request_fingerprint"]]
        proposal = decision["proposal"]
        need(
            decision["accepted"] and call["accepted"],
            "Selected illustration was rejected",
        )
        count = min(25, attempt["actions"] - step)
        need(count > 0, "Illustrative decision did not execute")
        current_generations = [
            (s, row)
            for (identity, s), row in generations.items()
            if identity == attempt_id and step <= s < step + count
        ]
        nonzero = sum(
            min(5, attempt["actions"] - s)
            for s, row in current_generations
            if row["provenance"] and row["provenance"].get("has_effect", False)
        )
        own_prefix = [
            row
            for row in ledger[: index + 1]
            if row["attempt_id"].rsplit("_revision", 1)[0] == attempt["arm"]
        ]
        language = proposal["language"]
        timeline.append(
            {
                "attempt_id": attempt_id,
                "arm": attempt["arm"],
                "revision": attempt["revision"],
                "decision_index": decision["decision_index"],
                "step": step,
                **request,
                "accepted": True,
                "actual_model": call["response"]["model"],
                "mode": proposal["mode"],
                "language": (
                    {
                        **language,
                        "source_a_prompt": source_prompts[language["source_a_id"]],
                        "source_b_prompt": source_prompts[language["source_b_id"]],
                    }
                    if language is not None
                    else None
                ),
                "vision": proposal["vision"],
                "observed_phase": proposal["observed_phase"],
                "rationale": proposal["rationale"],
                "executed_actions_for_decision": count,
                "nonzero_conditioned_actions_for_decision": nonzero,
                "call_usage": call["token_usage"],
                "cumulative_arm_provider": summarize_calls(own_prefix),
                "cumulative_case_provider": summarize_calls(ledger[: index + 1]),
            }
        )
    caption = {
        "wine": "Wine: failed TEI revision 1 and successful revision 2",
        "milk": "Milk: failed TEI revision 2 versus successful independent TLI revision 1",
        "cabinet": "Cabinet: forced VEI failure versus forced VLI success; native baseline already succeeded",
    }[label]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.7), dpi=140)
    for column, key in enumerate(FRAMES[label]):
        observation = figures[key]
        for row, camera in enumerate(CAMERAS):
            axes[row, column].imshow(decode(observation["observation"][camera]))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(
                    key[0].removeprefix("astra_").replace("_revision", " revision ")
                    + f"\nraw step {key[1]}",
                    fontsize=10,
                )
    fig.suptitle(caption, fontsize=12)
    fig.text(
        0.5,
        0.015,
        "Exact raw request cameras: external (top), wrist (bottom). "
        "Selected decision-time views; terminal simulator predicates are reported separately.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    figure_name = label + "_raw_feedback.png"
    fig.savefig(output / figure_name)
    plt.close(fig)
    identical_fallbacks = {}
    if label == "cabinet":
        for arm in ("astra_tli_vli", "astra_pixel_blend"):
            keys = [key for key in all_generations if key[0] == arm + "_revision1"]
            identical_fallbacks[arm] = {
                "compared_generations": len(keys),
                "all_raw_condition_noise_and_action_descriptors_equal_native_retry": all(
                    all_generations[key]
                    == all_generations["native_retry_revision1", key[1]]
                    for key in keys
                ),
                "scope": "Compares SHA256/shape/dtype and condition IDs, excluding attempt-specific array storage paths. Independent archive audit verifies referenced NPY bytes; no new policy or physics replay.",
            }
    return {
        "label": label,
        "episode_id": summary["episode_id"],
        "task_instruction": summary["instruction"],
        "worker": worker,
        "task_id": task,
        "baseline": {
            key: summary["baseline"][key]
            for key in ("attempt_id", "success", "actions_executed")
        },
        "archive_audit_status": numeric["status"],
        "provider_audit_status": provider["status"],
        "archive_audit_sha256": sha(numeric_path),
        "provider_audit_sha256": sha(provider_path),
        "archive_sha256": numeric["archive"]["sha256"],
        "source_file_sha256": {
            name: sha(directory / name)
            for name in ("summary.json", "events.jsonl", "provider.jsonl")
        },
        "provider": provider["provider"],
        "http_429_taxonomy": provider["http_429_taxonomy"],
        "integration_coverage": provider["integration_coverage"],
        "completed_attempts": provider["completed_attempts"],
        "incomplete_attempts": provider["incomplete_attempts"],
        "all_recorded_actions_bounds": provider["all_recorded_actions_bounds"],
        "provider_failure_fallback_actions_lower_bound": provider[
            "provider_failure_fallback_actions_lower_bound"
        ],
        "timeline": timeline,
        "figure": {
            "file": figure_name,
            "sha256": sha(output / figure_name),
            "selected_request_ids": [
                requests[key]["request_id"] for key in FRAMES[label]
            ],
        },
        "native_fallback_comparison": identical_fallbacks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--donor-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        bound_case(args.run_root, args.receipt_dir, *case, args.output)
        for case in CASES
    ]
    library = load_library(args.donor_library)
    donor_id = "std18-e398-f74"
    selected_donor = next(
        row for row in library.catalog() if row["donor_id"] == donor_id
    )
    donor_figure = "cabinet_selected_donor.png"
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.6), dpi=140)
    for axis, camera in zip(axes, CAMERAS, strict=True):
        donor = library.resolve(donor_id, camera)
        need(
            digest(donor.pixels) == selected_donor["cameras"][camera]["pixels_sha256"],
            "Selected donor image digest differs",
        )
        axis.imshow(donor.pixels)
        axis.axis("off")
        axis.set_title("External" if camera == CAMERAS[0] else "Wrist", fontsize=10)
    fig.suptitle(donor_id + ": " + selected_donor["prompt"], fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.output / donor_figure)
    plt.close(fig)
    result = {
        "schema_version": "representation-interrupted-pilot-behavior-1.0",
        "status": "verified_individual_observations_in_incomplete_pilot",
        "complete_cohort": False,
        "full20_evaluation_executed": False,
        "builder_sha256": sha(__file__),
        "plotting_environment": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cases": cases,
        "selected_donor": {
            "catalog_row": selected_donor,
            "figure": donor_figure,
            "figure_sha256": sha(args.output / donor_figure),
            "interpretation": "Raw paired donor frame selected in both cabinet VEI and VLI attempts. Original training task text is metadata; the visual bank was captured under the current target instruction.",
        },
        "limitations": [
            "Selected pilot decisions and completed rollouts, not a representative efficacy estimate.",
            "The experiment was interrupted by provider budget errors; unknown usage is not zero cost.",
            "A simulator success record is not independent evidence of stable human-judged placement.",
            "Observed phase/rationale is the model's short output, not an objective reward or hidden chain of thought.",
            "Nonzero conditioned actions identify direct input edits, not causal counterfactual action changes.",
            "No new inference, model solve, or simulator replay was performed for this publication.",
        ],
    }
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "timelines.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case",
                "attempt_id",
                "step",
                "mode",
                "source_a",
                "source_b",
                "language_alpha",
                "donor_id",
                "vision_alpha",
                "executed_actions",
                "nonzero_conditioned_actions",
                "cumulative_arm_calls",
                "cumulative_arm_known_tokens",
                "request_id",
                "request_fingerprint",
                "observed_phase",
                "rationale",
            ),
        )
        writer.writeheader()
        for case in cases:
            for row in case["timeline"]:
                language, vision = row["language"] or {}, row["vision"] or {}
                writer.writerow(
                    {
                        "case": case["label"],
                        "attempt_id": row["attempt_id"],
                        "step": row["step"],
                        "mode": row["mode"],
                        "source_a": language.get("source_a_prompt", ""),
                        "source_b": language.get("source_b_prompt", ""),
                        "language_alpha": language.get("alpha", ""),
                        "donor_id": vision.get("donor_id", ""),
                        "vision_alpha": vision.get("alpha", ""),
                        "executed_actions": row["executed_actions_for_decision"],
                        "nonzero_conditioned_actions": row[
                            "nonzero_conditioned_actions_for_decision"
                        ],
                        "cumulative_arm_calls": row["cumulative_arm_provider"][
                            "provider_calls"
                        ],
                        "cumulative_arm_known_tokens": row["cumulative_arm_provider"][
                            "tokens"
                        ]["total_tokens"]["sum"],
                        "request_id": row["request_id"],
                        "request_fingerprint": row["request_fingerprint"],
                        "observed_phase": row["observed_phase"],
                        "rationale": row["rationale"],
                    }
                )
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(cases),
                "evidence_sha256": sha(args.output / "evidence.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
