"""Build a descriptive success/token-cost figure from published experiment records.

Run from any directory. This reads frozen reports and makes no inference calls.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUTPUT = Path(__file__).resolve().parent
REPORTS = OUTPUT.parent
SOURCES = {
    "language": "phase_interpolation/evaluation/results/report.json",
    "pixels": "image_perturbations/evaluation/results/report.json",
    "frs": "frs_policy_improvement/development/report.json",
    "recipe": "learned_correction_recipe/results/report.json",
    "history": "learned_correction_recipe/results/historical_context.json",
}
COLORS = {
    "native": "#697586",
    "random": "#84919e",
    "tei": "#169c97",
    "tli": "#2563b6",
    "annotations": "#9670b6",
    "occlusion": "#c07b12",
    "blend": "#d45839",
    "direct": "#9262a3",
    "frs": "#2563b6",
    "schedule": "#138472",
    "selector": "#2563b6",
    "head": "#b85445",
    "gated": "#b68924",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sources() -> dict:
    return {
        name: json.loads((REPORTS / relative).read_text())
        for name, relative in SOURCES.items()
    }


def provider_totals(providers: list[dict]) -> tuple[int, int, int]:
    """Sum reported input + output; reasoning is already included in output."""
    total = missing = calls = 0
    for provider in providers:
        tokens = provider["tokens"]
        assert (
            tokens["total_tokens"]["sum"]
            == tokens["input_tokens"]["sum"] + tokens["output_tokens"]["sum"]
        )
        total += tokens["total_tokens"]["sum"]
        missing += tokens["total_tokens"]["missing_calls"]
        calls += provider.get("provider_calls", provider.get("calls", 0))
    return total, missing, calls


def make_row(
    cohort: str,
    method: str,
    label: str,
    successes: int,
    cases: int,
    providers: list[dict],
    max_attempts: int,
    median_rescue_tokens: float | None = None,
    median_rescue_revisions: float | None = None,
) -> dict:
    total, missing, calls = provider_totals(providers)
    assert 0 <= successes <= cases
    return {
        "experiment": "B" if cohort == "recipe" else "A",
        "cohort": cohort,
        "method": method,
        "label": label,
        "successes": successes,
        "cases": cases,
        "success_percent": 100 * successes / cases,
        "max_attempts_per_case": max_attempts,
        "known_online_tokens_total": total,
        "mean_known_online_tokens_per_case": total / cases,
        "provider_calls": calls,
        "calls_with_unknown_token_usage": missing,
        "cost_is_lower_bound": missing > 0,
        "median_tokens_among_rescues": median_rescue_tokens,
        "median_revisions_among_rescues": median_rescue_revisions,
        "source_report": SOURCES[cohort],
    }


def collect_rows(data: dict) -> list[dict]:
    rows = []
    language = data["language"]
    assert language["status"] == "complete" and language["seed"] == 29
    group = language["groups"]["pooled"]
    assert group["cases"] == 20
    rows.append(
        make_row(
            "language",
            "native",
            "Native baseline",
            group["baseline_successes"],
            20,
            [],
            1,
        )
    )
    for method, label in {
        "random_noise": "Random noise retries",
        "astra_tei": "Astra TEI",
        "astra_tli": "Astra TLI",
        "astra_tli_vision": "Astra TLI + annotations",
    }.items():
        arm = group["arms"][method]
        assert arm["development_extra_rollouts_after_success"] == 0
        rows.append(
            make_row(
                "language",
                method,
                label,
                arm["successes"],
                arm["cases"],
                [arm["physical_provider"]],
                arm["attempt_cap"],
                arm["median_total_tokens_among_rescues_with_complete_usage"],
                arm["median_rollout_revisions_among_rescues"],
            )
        )
    assert sum(r["known_online_tokens_total"] for r in rows) == 5_757_974

    pixels = data["pixels"]
    assert pixels["complete"] and pixels["efficacy_released"]
    assert len(pixels["cases"]) == 20
    assert {case["seed"] for case in pixels["cases"]} == {37}
    baseline = sum(case["baseline_success"] for case in pixels["cases"])
    rows.append(make_row("pixels", "native", "Native baseline", baseline, 20, [], 1))
    for method, label in {
        "random_noise": "Random noise retries",
        "random_occlusion": "Random occlusion",
        "random_demo_blend": "Random demo blend",
        "astra_occlusion": "Astra occlusion",
        "astra_demo_blend": "Astra demo blend",
    }.items():
        arm = pixels["groups"]["pooled"]["arms"][method]
        cases = [r for r in pixels["case_arm_rows"] if r["arm"] == method]
        assert len(cases) == arm["cases"] == 20
        assert sum(r["success"] for r in cases) == arm["successes"]
        assert all(r["development_extra_rollouts_after_success"] == 0 for r in cases)
        rescue = arm["rescue_only"]
        rows.append(
            make_row(
                "pixels",
                method,
                label,
                arm["successes"],
                arm["cases"],
                [r["provider_through_success_or_cap"] for r in cases],
                3,
                rescue["complete_total_tokens_to_success"]["median"],
                rescue["full_rollout_revisions"]["median"],
            )
        )
    assert (
        sum(r["known_online_tokens_total"] for r in rows if r["cohort"] == "pixels")
        == 11_077_106
    )

    frs = data["frs"]
    assert frs["status"] == "complete" and frs["phase"] == "development"
    cohort = next(c for c in frs["cohorts"] if c["id"] == "evaluation")
    final_round = next(r for r in cohort["rounds"] if r["index"] == 3)
    for method, label in {
        "native_euler10": "Native Euler 10",
        "native_repeated_noise": "Native repeated noise",
        "astra_direction_direct": "Astra direct steering",
        "astra_frs": "Astra FRS",
    }.items():
        episodes = [e for e in final_round["episodes"] if e["method_id"] == method]
        assert len(episodes) == 3
        assert {e["episode_id"] for e in episodes} == set(
            cohort["expected_episode_ids"]
        )
        assert all(not e["initial_success"] and e["actions"] > 0 for e in episodes)
        rows.append(
            make_row(
                "frs",
                method,
                label,
                sum(e["success"] for e in episodes),
                len(episodes),
                [e["provider_usage"] for e in episodes],
                1,
            )
        )

    recipe = data["recipe"]
    assert recipe["status"] == "complete" and recipe["efficacy_released"]
    assert (
        recipe["cost"]["new_provider_tokens"]
        == recipe["cost"]["new_provider_calls"]
        == 0
    )
    for method, label in {
        "native": "Native baseline",
        "recorded_schedule": "Recorded teacher schedule",
        "learned_selector": "Learned TEI/TLI selector",
        "flow_head": "Learned flow head",
        "gated_flow_head": "Selector-gated flow head",
    }.items():
        arm = recipe["groups"]["ood"]["methods"][method]
        assert arm["cases"] == 40
        assert arm["cost"]["provider_calls"] == arm["cost"]["provider_tokens"] == 0
        rows.append(
            make_row("recipe", method, label, arm["successes"], arm["cases"], [], 1)
        )
    return rows


def create_figure(rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#35445a",
            "text.color": "#17283c",
            "axes.edgecolor": "#c9d2dc",
            "xtick.color": "#596779",
            "ytick.color": "#596779",
            "svg.fonttype": "none",
            "svg.hashsalt": "smart-system2-token-results-v1",
            "pdf.fonttype": 42,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(13.6, 10.6))
    fig.subplots_adjust(
        left=0.073, right=0.973, top=0.84, bottom=0.185, hspace=0.48, wspace=0.20
    )
    fig.suptitle(
        "Smart System2: success vs. online Astra token cost",
        x=0.073,
        y=0.967,
        ha="left",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.073,
        0.927,
        "Higher success and fewer tokens are better. Separate cohorts; horizontal scales differ.",
        fontsize=11,
        color="#526277",
    )
    fig.text(
        0.073,
        0.899,
        "Cost = all reported input + output tokens / all evaluated task-reset cases, including failed attempts.",
        fontsize=10,
        color="#526277",
    )
    indexed = {(r["cohort"], r["method"]): r for r in rows}

    def panel(ax, title: str, subtitle: str, xmax: float, ticks: list[int]) -> None:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=27)
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color="#596779")
        ax.set_xlim(-xmax * 0.04, xmax)
        ax.set_ylim(-5, 100)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:g}k")
        )
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel("Success (%)")
        ax.set_xlabel("Mean online Astra tokens per evaluated case", labelpad=8)
        ax.grid(axis="y", color="#e9edf2", linewidth=0.8)
        ax.axvline(0, color="#c9d2dc", linewidth=0.9, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    def point(
        ax,
        cohort: str,
        method: str,
        color: str,
        text_at: tuple,
        label: str | None = None,
        ha: str = "left",
        marker: str = "o",
    ) -> None:
        row = indexed[cohort, method]
        x = row["mean_known_online_tokens_per_case"] / 1000
        y = row["success_percent"]
        ax.scatter(
            [x],
            [y],
            s=110 if color == "annotations" else 70,
            marker=marker,
            facecolor="none" if color == "annotations" else COLORS[color],
            edgecolor=COLORS[color] if color == "annotations" else "white",
            linewidth=1.4 if color == "annotations" else 0.8,
            zorder=4,
        )
        prefix = "≥" if row["cost_is_lower_bound"] else ""
        token_label = "0" if x == 0 else f"{prefix}{x:.1f}k"
        success = f"{y:g}%" if y * 10 == round(y * 10) else f"{y:.1f}%"
        text = f"{label or row['label']}\n{success} ({row['successes']}/{row['cases']}) · {token_label} tokens"
        ax.annotate(
            text,
            (x, y),
            xytext=text_at,
            ha=ha,
            va="center",
            fontsize=9.5,
            color=COLORS[color],
            linespacing=1.35,
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS[color],
                "alpha": 0.6,
                "lw": 0.8,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8},
        )

    ax = axs[0, 0]
    panel(
        ax,
        "Experiment A · Language interventions",
        "20 cases · seed 29 · baseline + up to two rescue attempts",
        145,
        [0, 50, 100],
    )
    point(ax, "language", "native", "native", (7, 24))
    point(ax, "language", "random_noise", "random", (7, 47), label="Random retries")
    point(ax, "language", "astra_tei", "tei", (140, 37), ha="right")
    point(ax, "language", "astra_tli", "tli", (142, 67), ha="right")
    point(
        ax,
        "language",
        "astra_tli_vision",
        "annotations",
        (58, 88),
        label="TLI + annotations",
        ha="center",
        marker="D",
    )

    ax = axs[0, 1]
    panel(
        ax,
        "Experiment A · Image perturbations",
        "20 cases · seed 37 · baseline + up to two rescue attempts",
        350,
        [0, 100, 200, 300],
    )
    point(ax, "pixels", "native", "native", (17, 26))
    point(
        ax,
        "pixels",
        "random_occlusion",
        "random",
        (17, 47),
        label="Random occlusion / noise",
    )
    point(
        ax,
        "pixels",
        "random_demo_blend",
        "random",
        (17, 69),
        label="Random blend",
        marker="s",
    )
    point(ax, "pixels", "astra_occlusion", "occlusion", (340, 88), ha="right")
    point(ax, "pixels", "astra_demo_blend", "blend", (340, 39), ha="right", marker="D")

    ax = axs[1, 0]
    panel(
        ax,
        "Experiment A · Action steering pilot",
        "3 selected cases · seed 19 / reset 1 · one attempt per method",
        45,
        [0, 10, 20, 30, 40],
    )
    point(ax, "frs", "native_euler10", "native", (2, 17), label="Both native controls")
    point(ax, "frs", "astra_frs", "frs", (19, 87), ha="center")
    point(
        ax,
        "frs",
        "astra_direction_direct",
        "direct",
        (44, 48),
        label="Astra direct steering",
        ha="right",
        marker="D",
    )

    ax = axs[1, 1]
    panel(
        ax,
        "Experiment B · Reuse and distillation",
        "40 cases · seed 61 · one attempt · prior teacher cost excluded",
        145,
        [0, 50, 100],
    )
    point(
        ax,
        "recipe",
        "recorded_schedule",
        "schedule",
        (22, 81),
        label="Recorded schedule",
    )
    point(
        ax,
        "recipe",
        "learned_selector",
        "selector",
        (22, 59),
        label="Learned TEI/TLI selector",
    )
    point(
        ax,
        "recipe",
        "gated_flow_head",
        "gated",
        (57, 40),
        label="Gated flow head",
        marker="D",
    )
    point(ax, "recipe", "native", "native", (22, 26))
    point(
        ax,
        "recipe",
        "flow_head",
        "head",
        (22, 10),
        label="Learned flow head",
        marker="s",
    )

    footnotes = [
        "≥ marks incomplete provider usage: one call each for occlusion and direct steering. These token costs are lower bounds.",
        "Experiment B: zero online calls; whole teacher-source studies used ≥6.89M tokens. Exact selected-teacher acquisition cost is unknown.",
        "FRS is a three-case development result. Compare methods within each panel; retries, resets and prompt/vision workloads differ.",
        "This figure excludes development/interruption overhead for the 20-case studies, teacher acquisition, robot-policy compute and coding-agent tokens.",
        "Known benchmark compositions, new resets. Selected completed comparisons; partial VEI/VLI and interrupted full FRS results are not plotted.",
    ]
    for index, line in enumerate(footnotes):
        fig.text(0.073, 0.126 - index * 0.022, line, fontsize=8.7, color="#596779")
    for suffix in ("png", "svg", "pdf"):
        metadata = {"Creator": "Smart System2 report postprocessor"}
        if suffix == "svg":
            metadata["Date"] = None
        if suffix == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        fig.savefig(
            OUTPUT / f"success_vs_tokens.{suffix}",
            dpi=200,
            facecolor="white",
            metadata=metadata,
        )
        if suffix == "svg":
            svg = OUTPUT / "success_vs_tokens.svg"
            svg.write_text(
                "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def main() -> None:
    data = read_sources()
    rows = collect_rows(data)
    source_hashes = {relative: sha(REPORTS / relative) for relative in SOURCES.values()}
    with (OUTPUT / "plotted_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "smart-system2-success-tokens-1.0",
        "x_axis": "Mean known online Astra input + output tokens per task/reset case, including failed attempts and reported failed calls.",
        "y_axis": "Descriptive simulator success percentage within each separate study protocol.",
        "reasoning_tokens_already_in_output": True,
        "missing_usage_is_unknown_not_zero": True,
        "successes_only_cost_is_not_used_for_x_axis": True,
        "excludes": [
            "teacher acquisition",
            "development/interruption overhead for the 20-case studies",
            "policy/simulator compute",
            "coding/review/orchestration tokens",
        ],
        "cohorts_are_not_pooled_or_budget_matched": True,
        "unplotted": [
            "privileged oracle controls",
            "earlier joint intervention search",
            "incomplete VEI/VLI pilot",
            "critique/adaptation-loop arms",
            "interrupted full FRS evaluation",
            "ID retention panel",
        ],
        "history": data["history"],
        "source_sha256": source_hashes,
        "rows": rows,
    }
    (OUTPUT / "plotted_values.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    create_figure(rows)
    # Sources are immutable. Regeneration must never alter original publications.
    assert source_hashes == {
        relative: sha(REPORTS / relative) for relative in SOURCES.values()
    }
    outputs = [
        "README.md",
        "build_results.py",
        "plotted_values.csv",
        "plotted_values.json",
        "success_vs_tokens.png",
        "success_vs_tokens.svg",
        "success_vs_tokens.pdf",
    ]
    manifest = {
        "schema_version": "smart-system2-results-publication-1.0",
        "source_sha256": source_hashes,
        "matplotlib_version": matplotlib.__version__,
        "files": {
            name: {
                "sha256": sha(OUTPUT / name),
                "bytes": (OUTPUT / name).stat().st_size,
            }
            for name in outputs
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"rows": len(rows), "output": str(OUTPUT), "source_reports_unchanged": True}
        )
    )


if __name__ == "__main__":
    main()
