"""Build the cabinet development example from an already audited case archive.

Run from the repository root with the project Python environment activated:
python -m astra_reversal.reports.phase_interpolation.development.examples.build_examples \
    --case-dir astra_reversal/.deps/interpolation-development-v1/extracted/worker_2/results/case_8_0

This reads stored records on CPU. It makes no provider, simulator, or model calls.
"""

import argparse
import base64
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

from astra_reversal.interpolation_agent import parse_proposal
from astra_reversal.interpolation_audit import RENDERER, render_linux
from astra_reversal.records import digest, file_sha256

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
CAMERAS = ("observation/image", "observation/wrist_image")
MODEL = "azure/openai/gpt-6-astra"
MODES = ("astra_tei", "astra_tli", "astra_tli_vision")
EXPECTED_ALPHA = {
    "astra_tei": [0.3, 0.3, 0.85, 1.0],
    "astra_tli": [0.2, 0.2, 0.35, 0.85, 0.85],
    "astra_tli_vision": [0.2, 0.2, 0.7, 0.85, 0.85],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def checked_array(case, name, audit):
    require(file_sha256(case / name) == audit["input_file_sha256"][name], name)
    return np.load(case / name, allow_pickle=False)


def select_decision(case, request_event, decision, generation, provider, audit):
    request = request_event["request"]
    fingerprint = request["request_fingerprint"]
    proposal = parse_proposal(decision["proposal"], request)
    require(decision["accepted"] and decision["error"] is None, "Rejected decision")
    require(provider["provider_call"] and provider["accepted"], "Provider acceptance")
    require(provider["http_status"] == 200, "HTTP status")
    require(provider["request_fingerprint"] == fingerprint, "Provider fingerprint")
    require(
        provider["response"]["model"] == provider["requested_model"] == MODEL, "Model"
    )
    require(
        provider["sampling_settings"]
        == {"reasoning_effort": "medium", "max_completion_tokens": 8192},
        "Sampling settings",
    )
    require(provider["cache"] == {"no-cache": True}, "Cache settings")
    content = provider["response"]["choices"][0]["message"]["content"]
    require(parse_proposal(content, request) == proposal, "Exact provider decision")
    require(
        generation["applied_accepted_decision_id"] == proposal["decision_id"],
        "Decision application",
    )
    require(
        request["source_catalog"]
        and all(
            set(row) == {"source_id", "prompt"} for row in request["source_catalog"]
        ),
        "Catalog fields",
    )
    require(len(request["source_catalog"]) == 9, "Donor catalog size")
    require(request["limits"]["vision_chunk_actions"] == 5, "Vision expiry")
    require(request["limits"]["call_interval"] == 25, "Refresh interval")
    latest = request["observations"][-1]
    step = request["observation_step"]
    require(
        latest["step"] == generation["observation_step"] == step, "Current frame step"
    )
    bindings = [
        row
        for row in audit["snapshot_bindings"]
        if row["request_fingerprint"] == fingerprint
        and row["scope"] == "current_attempt"
        and row["step"] == step
    ]
    require(len(bindings) == 1, "Unique current snapshot binding")
    binding = bindings[0]
    require(binding["attempt_id"] == request["attempt_id"], "Snapshot attempt")
    require(
        binding["recorded_event_sequence"] == generation["sequence"], "Snapshot event"
    )
    observation, camera_records = {}, {}
    for camera in CAMERAS:
        encoded = latest["observation"][camera]
        require(encoded["encoding"] == "base64_png", "PNG encoding")
        png = base64.b64decode(encoded["data"], validate=True)
        pixels = np.array(Image.open(io.BytesIO(png)))
        source = binding["cameras"][camera]
        require(hashlib.sha256(png).hexdigest() == source["png_sha256"], "PNG bytes")
        require(digest(pixels) == source["decoded_array_sha256"], "Decoded pixels")
        name = source["recorded_array"]
        stored = checked_array(case, name, audit)
        require(np.array_equal(pixels, stored), "PNG versus recorded NPY")
        require(generation["observation"][camera]["array"] == name, "Generation camera")
        require(
            pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8, "Raw RGB shape"
        )
        observation[camera] = pixels
        camera_records[camera] = {
            **source,
            "npy_file_sha256": audit["input_file_sha256"][name],
            "shape": list(pixels.shape),
        }
    state_name = binding["state_recorded_array"]
    observation["observation/state"] = checked_array(case, state_name, audit)
    require(
        np.array_equal(
            observation["observation/state"],
            np.asarray(latest["observation"]["observation/state"], dtype=np.float32),
        ),
        "Request robot state",
    )
    require(
        digest(observation) == binding["raw_observation_sha256"], "Raw observation hash"
    )
    require(generation["vision"] == proposal["vision"], "Applied annotations")
    modified = render_linux(observation, proposal["vision"])
    require(
        digest(modified) == generation["modified_observation_sha256"],
        "Exact Linux rendered pixels",
    )
    require(
        np.array_equal(modified[CAMERAS[1]], observation[CAMERAS[1]]), "Wrist unchanged"
    )
    provider_binding = [
        row
        for row in audit["provider_bindings"]
        if row["request_fingerprint"] == fingerprint
    ]
    require(
        len(provider_binding) == 1 and provider_binding[0]["accepted"],
        "Audited provider binding",
    )
    require(
        provider_binding[0]["response_text_sha256"]
        == hashlib.sha256(content.encode()).hexdigest(),
        "Provider text hash",
    )
    safe = {
        "attempt_id": request["attempt_id"],
        "decision_id": proposal["decision_id"],
        "step": step,
        "source_a_id": proposal["source_a_id"],
        "source_b_id": proposal["source_b_id"],
        "alpha": proposal["alpha"],
        "annotations": proposal["vision"],
        "request_fingerprint": fingerprint,
        "request_event_sequence": request_event["sequence"],
        "generation_event_sequence": generation["sequence"],
        "response_text_sha256": provider_binding[0]["response_text_sha256"],
        "raw_observation_sha256": binding["raw_observation_sha256"],
        "modified_observation_sha256": generation["modified_observation_sha256"],
        "cameras": camera_records,
        "state": {
            "recorded_array": state_name,
            "array_digest": binding["state_array_sha256"],
            "npy_file_sha256": audit["input_file_sha256"][state_name],
        },
        "token_usage": provider_binding[0]["token_usage"],
    }
    return safe, observation, modified


def figure(samples, *, vision):
    rows = 3 if vision else 2
    fig, axes = plt.subplots(
        rows, len(samples), figsize=(12.4, 8.4) if vision else (10.5, 6.25)
    )
    labels = ["Raw external\nAstra feedback", "Raw wrist\nAstra feedback"]
    if vision:
        labels.append("Marked external\nPolicy input, ≤5 actions")
    for column, (record, raw, modified) in enumerate(samples):
        images = [raw[CAMERAS[0]], raw[CAMERAS[1]]]
        if vision:
            images.append(modified[CAMERAS[0]])
        for row, pixels in enumerate(images):
            ax = axes[row, column]
            ax.imshow(pixels, interpolation="nearest")
            ax.set_axis_off()
            if column == 0:
                ax.text(
                    -0.09,
                    0.5,
                    labels[row],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                )
        axes[0, column].set_title(
            f"Step {record['step']}\nα = {record['alpha']:.2f}", fontsize=12, pad=8
        )
        if vision:
            annotation = record["annotations"][0]
            actions = record["marked_actions_executed"]
            axes[-1, column].text(
                0.5,
                -0.065,
                f"{annotation['kind'].capitalize()} · gain 0.5\n{actions} executed actions",
                transform=axes[-1, column].transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
    title = (
        "Astra TLI + vision: raw feedback and applied marks"
        if vision
        else "Astra TEI: four decisions before recorded success"
    )
    fig.suptitle(title, fontsize=15, y=0.975 if vision else 0.98)
    fig.text(
        0.57,
        0.93 if vision else 0.90,
        "Task: “put the bowl at table center on the cabinet”",
        ha="center",
        fontsize=11,
    )
    note = "A = donor 38 (table-center bowl → plate)     B = donor 18 (bowl → cabinet)"
    fig.text(0.57, 0.055 if vision else 0.06, note, ha="center", fontsize=10)
    footer = (
        "Recorded success at action 102 · 5 decisions · 29,078 tokens · one development case"
        if vision
        else "Recorded success at action 100 · 4 decisions · 21,668 tokens · one development case"
    )
    fig.text(0.57, 0.026 if vision else 0.025, footer, ha="center", fontsize=10)
    fig.subplots_adjust(
        left=0.17,
        right=0.99,
        bottom=0.15 if vision else 0.12,
        top=0.845 if vision else 0.79,
        wspace=0.035,
        hspace=0.035,
    )
    path = HERE / ("cabinet_tli_vision.png" if vision else "cabinet_tei.png")
    fig.savefig(
        path,
        dpi=150,
        facecolor="white",
        pil_kwargs={"optimize": True},
        metadata={"Software": "Matplotlib; verified recorded development frames"},
    )
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    case = args.case_dir
    audits = HERE.parent / "audits"
    receipt_path = audits / "worker_2_receipt.json"
    receipt = read_json(receipt_path)
    require(
        receipt["status"] == "passed" and receipt["phase"] == "development",
        "Verified development receipt",
    )
    array_path = audits / receipt["array_audit"]["file"]
    require(
        file_sha256(array_path) == receipt["array_audit"]["sha256"], "Array audit bytes"
    )
    array_audit = read_json(array_path)
    require(
        array_audit["status"] == "passed" and array_audit["complete"],
        "Complete array audit",
    )
    feedback_meta = receipt["feedback"][0]
    feedback_path = audits / feedback_meta["file"]
    require(
        file_sha256(feedback_path) == feedback_meta["sha256"], "Feedback audit bytes"
    )
    feedback_bytes = gzip.decompress(feedback_path.read_bytes())
    require(
        hashlib.sha256(feedback_bytes).hexdigest()
        == feedback_meta["uncompressed_sha256"],
        "Uncompressed feedback bytes",
    )
    feedback = json.loads(feedback_bytes)
    require(feedback["status"] == "passed", "Passed feedback audit")
    names = [
        "summary.json",
        "events.jsonl",
        *[f"{mode}_2_provider.jsonl" for mode in MODES],
    ]
    for name in names:
        require(
            file_sha256(case / name) == feedback["input_file_sha256"][name],
            f"Audited case file: {name}",
        )
    summary = read_json(case / "summary.json")
    require(
        summary["episode_id"] == "libero_spatial_ood:seed19:task8:state0"
        and summary["development"],
        "Case identity",
    )
    require(
        not summary["baseline"]["success"]
        and not summary["arms"]["oracle_tei"]["summary"]["success"],
        "Baseline and oracle TEI outcomes",
    )
    events = [
        json.loads(line) for line in (case / "events.jsonl").read_text().splitlines()
    ]
    requests = {
        row["request"]["request_fingerprint"]: row
        for row in events
        if row["kind"] == "interpolation_request"
    }
    generations = {
        (row["attempt_id"], row["observation_step"]): row
        for row in events
        if row["kind"] == "phase_generation"
    }
    arms, samples = {}, {}
    for mode in MODES:
        arm = summary["arms"][mode]
        attempt = arm["attempts"][-1]
        require(
            attempt["success"]
            and arm["summary"]["full_rollout_revisions_to_success"] == 1,
            "One successful revision",
        )
        require(attempt["native_condition_fallback_actions"] == 0, "No native fallback")
        providers = {
            row["request_fingerprint"]: row
            for row in map(
                json.loads, (case / f"{mode}_2_provider.jsonl").read_text().splitlines()
            )
        }
        samples[mode] = []
        for decision in attempt["decisions"]:
            proposal = decision["proposal"]
            fingerprint = proposal["request_fingerprint"]
            generation = generations[
                (attempt["attempt_id"], decision["observation_step"])
            ]
            sample = select_decision(
                case,
                requests[fingerprint],
                decision,
                generation,
                providers[fingerprint],
                feedback,
            )
            safe = sample[0]
            require(
                (safe["source_a_id"], safe["source_b_id"]) == ("38", "18"),
                "Donor choice",
            )
            if mode == "astra_tli_vision":
                safe["marked_actions_executed"] = min(
                    5, attempt["actions_executed"] - safe["step"]
                )
                safe["mark_action_interval_half_open"] = [
                    safe["step"],
                    safe["step"] + safe["marked_actions_executed"],
                ]
            samples[mode].append(sample)
        require(
            [row[0]["alpha"] for row in samples[mode]] == EXPECTED_ALPHA[mode],
            "Recorded alpha sequence",
        )
        if mode == "astra_tli_vision":
            require(
                sum(row[0]["marked_actions_executed"] for row in samples[mode])
                == attempt["actions_with_changed_vision"]
                == 22,
                "Mark action count",
            )
        cost = arm["summary"]["tokens_to_first_success"]
        arms[mode] = {
            "attempt_id": attempt["attempt_id"],
            "success": attempt["success"],
            "first_success_attempt": arm["summary"]["first_success_attempt"],
            "full_rollout_revisions_to_success": 1,
            "decisions": [row[0] for row in samples[mode]],
            "intervention_rollout_actions": attempt["actions_executed"],
            "actions_including_common_baseline": arm["summary"][
                "actions_through_success_or_cap"
            ],
            "actions_with_nonzero_text": attempt["actions_with_nonzero_text"],
            "actions_with_changed_vision": attempt["actions_with_changed_vision"],
            "native_condition_fallback_actions": attempt[
                "native_condition_fallback_actions"
            ],
            "provider_calls": cost["provider_calls"],
            "failed_calls": cost["failed_calls"],
            "tokens_to_first_success": {
                name: row["sum"] for name, row in cost["tokens"].items()
            },
            "usage_complete": all(row["complete"] for row in cost["tokens"].values()),
        }
    figures = [
        figure(samples["astra_tei"], vision=False),
        figure(samples["astra_tli_vision"], vision=True),
    ]
    output = {
        "schema_version": "phase-interpolation-development-examples-1",
        "status": "verified",
        "scope": "one development case; not a full evaluation result",
        "episode_id": summary["episode_id"],
        "target_task": next(iter(requests.values()))["request"]["target_task"],
        "workflow": receipt["workflow"],
        "archive_sha256": receipt["archive"]["sha256"],
        "payload_sha256": array_audit["payload_sha256"],
        "protocol_sha256": summary["protocol_sha256"],
        "source_catalog_sha256": feedback["catalog_sha256"],
        "source_catalog_size": 9,
        "oracle_mapping_provided_to_astra": False,
        "chosen_sources": [
            row for row in summary["source_catalog"] if row["source_id"] in {"38", "18"}
        ],
        "operator_formulas": {
            "tei": "(1-alpha)*E_A+alpha*E_B",
            "tli_added_residual": "(1-2*alpha)*(T_A-T_B)",
        },
        "baseline": {
            "success": False,
            "actions": summary["baseline"]["actions_executed"],
        },
        "oracle_tei": {
            "success": False,
            "intervention_rollout_actions": summary["arms"]["oracle_tei"]["attempts"][
                -1
            ]["actions_executed"],
        },
        "astra": arms,
        "provider_settings": {
            "actual_model": MODEL,
            "reasoning_effort": "medium",
            "max_completion_tokens": 8192,
            "cache": {"no-cache": True},
        },
        "audit_files": {
            "../audits/" + p.name: file_sha256(p)
            for p in (receipt_path, array_path, feedback_path)
        },
        "case_file_sha256": {
            name: feedback["input_file_sha256"][name] for name in names
        },
        "renderer": RENDERER,
        "plot_environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "source_file_sha256": {
            str(p.relative_to(PROJECT)): file_sha256(p)
            for p in [
                Path(__file__).resolve(),
                PROJECT / "interpolation_agent.py",
                PROJECT / "interpolation_audit.py",
                PROJECT / "interventions.py",
                PROJECT / "records.py",
            ]
        },
        "artifact_sha256": {
            p.name: file_sha256(p) for p in [*figures, HERE / "README.md"]
        },
        "hash_conventions": {
            "file_and_png_sha256": "SHA256 of exact file/PNG bytes",
            "array_and_observation_digest": "astra_reversal.records.digest; includes dtype, shape and values",
            "request_fingerprint": "records.digest of the entire request excluding request_fingerprint",
        },
        "limitations": [
            "Selected current raw frames are shown; additional recent and prior-rollout frames supplied to Astra are omitted from the figures.",
            "Recorded binary success is not a new simulator replay or a claim about the full 20-case evaluation.",
            "TLI and TLI+vision choose different alpha values and have different trajectories; this does not isolate a causal vision benefit.",
            "Annotations are fresh for at most one five-action policy chunk; the last marked chunk ends after two actions at success.",
            "Reasoning token counts are included within output tokens, not added to totals; hidden reasoning and response rationale are not published.",
            "The separate transport-only smoke (2550 tokens) is excluded from these rollout-specific token totals. No verified monetary rate is available.",
        ],
    }
    path = HERE / "examples.json"
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "requests_checked": sum(len(v) for v in samples.values()),
                "camera_pairs_checked": sum(len(v) for v in samples.values()),
                "examples_sha256": file_sha256(path),
                "figures": {p.name: p.stat().st_size for p in figures},
            }
        )
    )


if __name__ == "__main__":
    main()
