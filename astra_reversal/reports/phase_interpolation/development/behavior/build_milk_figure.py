"""Render actual milk-case feedback only after both full archive audits pass."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from astra_reversal.records import digest, file_sha256
from astra_reversal.reports.phase_interpolation.development.behavior.build_behavior import (
    HERE,
    PROJECT,
    load_case,
    require,
)

CAMERAS = ("observation/image", "observation/wrist_image")
SELECTION = (
    (
        "astra_tei_2",
        50,
        "current_attempt",
        50,
        "TEI revision 1 · decision 3\nStep 50 · α = 0.00\nA14 / B10",
    ),
    (
        "astra_tei_3",
        0,
        "previous_attempt",
        300,
        "Prior TEI revision 1\nStep 300: recorded failure\nFeedback supplied to revision 2",
    ),
    (
        "astra_tei_3",
        0,
        "current_attempt",
        0,
        "TEI revision 2 · decision 1\nStep 0 · α = 0.15\nA13 / B10",
    ),
    (
        "astra_tei_3",
        275,
        "current_attempt",
        275,
        "TEI revision 2 · decision 12\nStep 275 · α = 0.75\nA14 / B13",
    ),
    (
        "astra_tli_2",
        50,
        "current_attempt",
        50,
        "TLI revision 1 · decision 3\nStep 50 · α = 0.20\nA10 / B18",
    ),
    (
        "astra_tli_2",
        75,
        "current_attempt",
        75,
        "TLI revision 1 · decision 4\nStep 75 · α = 0.20\nA10 / B18",
    ),
)


def frame(case, request_event, scope, step, feedback):
    request = request_event["request"]
    fingerprint = request["request_fingerprint"]
    rows = (
        request["observations"]
        if scope == "current_attempt"
        else request["previous_attempt"]["snapshots"]
    )
    snapshots = [row for row in rows if row["step"] == step]
    require(len(snapshots) == 1, "Unique supplied snapshot")
    snapshot = snapshots[0]
    bindings = [
        row
        for row in feedback["snapshot_bindings"]
        if row["request_fingerprint"] == fingerprint
        and row["scope"] == scope
        and row["step"] == step
    ]
    require(len(bindings) == 1, "Unique audited snapshot binding")
    binding = bindings[0]
    expected_attempt = (
        request["attempt_id"]
        if scope == "current_attempt"
        else request["previous_attempt"]["feedback"]["attempt_id"]
    )
    require(
        binding["attempt_id"] == expected_attempt,
        "Current versus prior attempt identity",
    )
    observation, provenance = {}, {}
    for camera in CAMERAS:
        source = binding["cameras"][camera]
        png = base64.b64decode(snapshot["observation"][camera]["data"], validate=True)
        require(
            hashlib.sha256(png).hexdigest() == source["png_sha256"],
            "Exact request PNG bytes",
        )
        pixels = np.array(Image.open(io.BytesIO(png)))
        require(
            digest(pixels) == source["decoded_array_sha256"], "Decoded pixel digest"
        )
        name = source["recorded_array"]
        require(
            file_sha256(case / name) == feedback["input_file_sha256"][name],
            "Recorded NPY bytes",
        )
        stored = np.load(case / name, allow_pickle=False)
        require(
            np.array_equal(stored, pixels) and stored.dtype == pixels.dtype,
            "PNG matches actual NPY",
        )
        observation[camera] = pixels
        provenance[camera] = {
            **source,
            "npy_file_sha256": feedback["input_file_sha256"][name],
        }
    state_name = binding["state_recorded_array"]
    require(
        file_sha256(case / state_name) == feedback["input_file_sha256"][state_name],
        "State NPY bytes",
    )
    observation["observation/state"] = np.load(case / state_name, allow_pickle=False)
    require(
        digest(observation) == binding["raw_observation_sha256"],
        "Complete raw observation binding",
    )
    return observation, {
        "request_fingerprint": fingerprint,
        "request_attempt_id": request["attempt_id"],
        "request_decision_index": request["decision_index"],
        "request_observation_step": request["observation_step"],
        "request_event_sequence": request_event["sequence"],
        "frame_scope": scope,
        "frame_attempt_id": expected_attempt,
        "frame_step": step,
        "recorded_event_sequence": binding["recorded_event_sequence"],
        "raw_observation_sha256": binding["raw_observation_sha256"],
        "cameras": provenance,
        "state": {
            "recorded_array": state_name,
            "array_digest": binding["state_array_sha256"],
            "npy_file_sha256": feedback["input_file_sha256"][state_name],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, feedback, requests, sources = load_case(args.case_dir, 1)
    require(
        summary["episode_id"] == "libero_spatial_ood:seed19:task2:state0",
        "Milk case identity",
    )
    require(
        not summary["arms"]["astra_tei"]["summary"]["success"], "Recorded TEI failure"
    )
    require(summary["arms"]["astra_tli"]["summary"]["success"], "Recorded TLI success")
    by_step = {
        (row["request"]["attempt_id"], row["request"]["observation_step"]): row
        for row in requests.values()
    }
    selected = []
    for attempt, call_step, scope, frame_step, title in SELECTION:
        image, record = frame(
            args.case_dir, by_step[(attempt, call_step)], scope, frame_step, feedback
        )
        selected.append((image, {**record, "figure_label": title}))
    fig, axes = plt.subplots(2, len(selected), figsize=(16, 7))
    for column, (observation, record) in enumerate(selected):
        for row, camera in enumerate(CAMERAS):
            axes[row, column].imshow(observation[camera], interpolation="nearest")
            axes[row, column].set_axis_off()
            if column == 0:
                axes[row, column].text(
                    -0.065,
                    0.5,
                    "Raw external" if row == 0 else "Raw wrist",
                    transform=axes[row, column].transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                )
        axes[0, column].set_title(record["figure_label"], fontsize=10, pad=10)
    fig.suptitle("Milk: failed TEI revisions and a TLI rescue", fontsize=17, y=0.975)
    fig.text(
        0.55,
        0.91,
        f'Task: “{next(iter(requests.values()))["request"]["target_task"]}” · actual current and prior-rollout feedback',
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.55,
        0.075,
        "TEI: two unsuccessful 300-action revisions · TLI: success at action 100 of its first revision",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.55,
        0.032,
        "TLI saw the common failed baseline and its own history; it did not receive the TEI rollouts. One development case.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.095, right=0.995, top=0.77, bottom=0.16, wspace=0.035, hspace=0.04
    )
    figure_path = HERE / "milk_tei_failure_tli_rescue.png"
    fig.savefig(
        figure_path,
        dpi=150,
        facecolor="white",
        pil_kwargs={"optimize": True},
        metadata={"Software": "Matplotlib; verified recorded development frames"},
    )
    plt.close(fig)
    artifact = {
        "schema_version": "phase-interpolation-milk-figure-1",
        "status": "verified",
        "episode_id": summary["episode_id"],
        "scope": "one development case",
        "target_task": next(iter(requests.values()))["request"]["target_task"],
        "frames": [record for _, record in selected],
        "sources": sources,
        "figure_sha256": file_sha256(figure_path),
        "source_file_sha256": {
            str(p.relative_to(PROJECT)): file_sha256(p)
            for p in (
                Path(__file__).resolve(),
                HERE / "build_behavior.py",
                PROJECT / "records.py",
            )
        },
        "interpretation": [
            "All images are raw paired feedback actually present in the identified requests; no annotations were added to the pixels.",
            "The prior-rollout frame was supplied to the next TEI revision and is explicitly labeled. It is not a current reset observation.",
            "TLI does not receive TEI-arm feedback. The figure contrasts separate arms, not learning from TEI into TLI.",
            "The last TLI image is step75; recorded success occurs later at action100. Outcomes are archive records, not a new simulator replay.",
            "Donor contrasts are conditioning hypotheses, not proven semantic cancellation or evidence of full-evaluation generalization.",
        ],
    }
    target = HERE / "milk_example.json"
    target.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "camera_pairs_checked": len(selected),
                "figure_bytes": figure_path.stat().st_size,
                "milk_example_sha256": file_sha256(target),
            }
        )
    )


if __name__ == "__main__":
    main()
