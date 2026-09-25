"""Plot success and observed token budgets from a completed intervention audit."""

import argparse
import json
from pathlib import Path

from .interventions import ARMS
from .records import digest, file_sha256

LABELS = {
    "random_noise": "Random noise",
    "noise_only": "Astra: noise",
    "language_only": "Astra: language embedding",
    "vision_only": "Astra: vision",
    "noise_language": "Astra: noise + language",
    "noise_vision": "Astra: noise + vision",
    "language_vision": "Astra: language + vision",
    "joint": "Astra: all three",
}


def render(report_path, output):
    """Write a static scientific figure and its source/output hash receipts."""
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text())
    if (
        report["phase"] != "evaluation"
        or report["status"] != "complete_verified_recording"
        or report["sha256"]
        != digest({key: value for key, value in report.items() if key != "sha256"})
        or set(report["summary"]["arms"]) != set(ARMS)
        or report["summary"]["cases"] != 20
    ):
        raise ValueError("A complete, hash-verified 20-case evaluation is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=False)
    summary = report["summary"]
    cases = summary["cases"]
    baseline = 100 * summary["baseline_successes"] / cases
    random = summary["arms"]["random_noise"]["successes_by_attempt"]
    colors = (
        "#555555",
        "#2166ac",
        "#b35806",
        "#238b45",
        "#7b3294",
        "#008080",
        "#c51b7d",
        "#a63603",
    )
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.6), sharex=True, sharey=True)
    for axis, arm, color in zip(axes.flat, ARMS, colors, strict=True):
        row = summary["arms"][arm]
        counts = row["successes_by_attempt"]
        axis.axhline(baseline, color="#888888", linestyle=":", linewidth=1.2)
        if arm != "random_noise":
            axis.plot(
                range(5),
                [100 * count / cases for count in random],
                color="#555555",
                linestyle="--",
                linewidth=1.3,
                alpha=0.75,
            )
        axis.plot(
            range(5),
            [100 * count / cases for count in counts],
            color=color,
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axis.set_title(f"{LABELS[arm]}: {counts[-1]}/{cases}", loc="left", fontsize=10)
        axis.set_xticks(range(5))
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_xlim(-0.12, 4.12)
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelbottom=True, labelleft=True)
        tokens = row["token_usage_through_success_or_cap"]["tokens"]["total_tokens"]
        cost = (
            "unknown"
            if tokens["missing_calls"] and not tokens["available_calls"]
            else f"{tokens['sum']:,} observed (partial)"
            if tokens["missing_calls"]
            else f"{tokens['sum']:,}"
        )
        rescue = row["conditional_rescue"]
        median = row["median_intervention_iterations_among_rescued_cases"]
        median_text = (
            "no rescue"
            if median is None
            else f"median {median:g} revision{'s' if median != 1 else ''}"
        )
        axis.text(
            0,
            -0.27,
            f"Rescued {rescue['rescued_cases']}/{rescue['baseline_failed_cases']} baseline failures"
            f"; {median_text}\n"
            f"Astra tokens: {cost}; censored: {row['censored_cases']}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            linespacing=1.5,
        )
    fig.suptitle(
        "Frozen pi0.5: iterative interventions on 20 OOD resets", fontsize=15, y=0.98
    )
    fig.legend(
        [
            Line2D([0], [0], color="#555555", linestyle="--"),
            Line2D([0], [0], color="#888888", linestyle=":"),
        ],
        ["Matched random-noise search", "Common recovered-noise baseline"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.supylabel("Cumulative successful cases (%)", x=0.013, fontsize=11)
    fig.supxlabel(
        "Intervention revisions after baseline (maximum 4)", y=0.055, fontsize=11
    )
    fig.text(
        0.5,
        0.012,
        "Seed 19; 10 Goal-OOD + 10 Spatial-OOD cases. Simulator reset access. "
        "Rescue medians exclude failures; censored cases remain unsuccessful at the cap.\n"
        "Tokens include failed searches and observed failed-call usage. "
        "Exploratory follow-up on known tasks; no claim of unseen training compositions.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, top=0.855, bottom=0.19, hspace=0.72, wspace=0.26
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"success_and_tokens.{extension}", dpi=180)
    plt.close(fig)
    manifest = {
        "source_report": str(report_path.resolve()),
        "source_report_file_sha256": file_sha256(report_path),
        "source_report_content_digest": report["sha256"],
        "plot_source_sha256": file_sha256(Path(__file__)),
        "matplotlib_version": matplotlib.__version__,
        "outputs": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
        },
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("output")
    args = parser.parse_args()
    render(args.report, args.output)


if __name__ == "__main__":
    main()
