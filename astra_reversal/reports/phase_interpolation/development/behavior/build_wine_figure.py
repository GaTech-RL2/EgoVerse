"""Show the audited wine TLI revision using exact supplied camera PNGs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astra_reversal.records import file_sha256
from astra_reversal.reports.phase_interpolation.development.behavior.build_behavior import (
    HERE,
    PROJECT,
    load_case,
    require,
)
from astra_reversal.reports.phase_interpolation.development.behavior.build_milk_figure import (
    CAMERAS,
    frame,
)

SELECTION = (
    (
        "astra_tli_2",
        125,
        "current_attempt",
        125,
        "TLI revision 1 · decision 6\nStep 125 · α = 1.00\nA14 / B13",
    ),
    (
        "astra_tli_3",
        0,
        "previous_attempt",
        300,
        "Prior TLI revision 1\nStep 300: recorded failure\nFeedback supplied to revision 2",
    ),
    (
        "astra_tli_3",
        0,
        "current_attempt",
        0,
        "TLI revision 2 · decision 1\nStep 0 · α = 0.15\nA14 / B18",
    ),
    (
        "astra_tli_3",
        75,
        "current_attempt",
        75,
        "TLI revision 2 · decision 4\nStep 75 · α = 0.85\nA18 / B13",
    ),
    (
        "astra_tli_3",
        100,
        "current_attempt",
        100,
        "TLI revision 2 · decision 5\nStep 100 · α = 0.85\nA18 / B13",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, feedback, requests, sources = load_case(args.case_dir, 0)
    require(
        summary["episode_id"] == "libero_goal_ood:seed19:task6:state0", "Wine identity"
    )
    attempts = summary["arms"]["astra_tli"]["attempts"]
    require(
        [row["success"] for row in attempts] == [False, False, True],
        "Failed then successful revision",
    )
    require(attempts[-1]["actions_executed"] == 104, "Winning action count")
    by_step = {
        (row["request"]["attempt_id"], row["request"]["observation_step"]): row
        for row in requests.values()
    }
    selected = []
    for attempt, call_step, scope, frame_step, title in SELECTION:
        pixels, record = frame(
            args.case_dir, by_step[(attempt, call_step)], scope, frame_step, feedback
        )
        selected.append((pixels, {**record, "figure_label": title}))
    fig, axes = plt.subplots(2, len(selected), figsize=(13.5, 6.5))
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
    fig.suptitle(
        "Wine: revising the donor contrast after a failed rollout", fontsize=16, y=0.975
    )
    fig.text(
        0.56,
        0.905,
        f'Task: “{next(iter(requests.values()))["request"]["target_task"]}” · actual current and prior-rollout feedback',
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.56,
        0.075,
        "TLI revision 1: failed at 300 actions · TLI revision 2: success at 104 actions",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.56,
        0.031,
        "17 calls across both revisions · 132,062 tokens · one development case · source contrasts are hypotheses",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.11, right=0.995, top=0.745, bottom=0.165, wspace=0.04, hspace=0.04
    )
    figure_path = HERE / "wine_tli_revision.png"
    fig.savefig(
        figure_path,
        dpi=150,
        facecolor="white",
        pil_kwargs={"optimize": True},
        metadata={"Software": "Matplotlib; verified recorded development frames"},
    )
    plt.close(fig)
    artifact = {
        "schema_version": "phase-interpolation-wine-figure-1",
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
                HERE / "build_milk_figure.py",
                PROJECT / "records.py",
            )
        },
        "interpretation": [
            "All images are the raw paired feedback actually supplied in the identified requests; image pixels are unchanged.",
            "The previous revision's final snapshot is explicitly distinguished from the reset state of the next revision.",
            "The model changed its own TLI donor pairs after TLI feedback. No TEI-arm or oracle-arm history was supplied.",
            "The last image is step100; recorded success occurs at action104. The audit verifies recording, not a new simulator replay.",
            "The successful revision supports further testing of this recipe but does not establish semantic cancellation or generalization.",
        ],
    }
    target = HERE / "wine_example.json"
    target.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "camera_pairs_checked": len(selected),
                "figure_bytes": figure_path.stat().st_size,
                "wine_example_sha256": file_sha256(target),
            }
        )
    )


if __name__ == "__main__":
    main()
