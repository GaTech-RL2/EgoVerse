"""Export scientific FRS figures from complete, audited aggregate recordings.

No inference, outcome collection, or historical-cohort pooling is performed.
The JSON beside the figures contains every plotted value and its source hashes.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .frs_html_report import (
    COLORS,
    _read_json,
    _sum_usage,
    derived_results,
    require,
    validate_report,
)
from .records import file_sha256

PLOT_VERSION = "frs-scientific-figures-1.0"
ROLES = ("paper_direction", "action_edit", "critique", "judge")
ROLE_LABELS = (
    "Online paper\ndirection",
    "Online action\nedit",
    "Between-rollout\ncritique",
    "Trajectory\ncomparison",
)
METHOD_COLORS = {
    "native_euler10": COLORS[0],
    "native_repeated_noise": COLORS[1],
    "astra_direction_direct": COLORS[2],
    "astra_frs": COLORS[3],
    "critique_frs_no_learning": COLORS[4],
    "learned_noise": COLORS[5],
    "critique_frs_learning": COLORS[5],
}


def _bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def plotted_values(report):
    """Return source-bound plotting data; refuse partial or unaudited efficacy."""
    checked = validate_report(report)
    evidence = report.get("producer_evidence", {})
    tasks = evidence.get("tasks", [])
    require(
        report["status"] == evidence.get("status") == "complete"
        and tasks
        and len(tasks) == len(evidence.get("expected_task_keys", []))
        and {row["task_key"] for row in tasks} == set(evidence["expected_task_keys"])
        and all(row["audited"] and row["status"] == "complete" for row in tasks)
        and not any(
            evidence.get(key)
            for key in (
                "missing_task_keys",
                "unaudited_task_keys",
                "unfinished_task_keys",
            )
        ),
        "Scientific result plots require complete coverage and passed task audits",
    )
    require(
        {row["id"] for row in report["cohorts"]} == {"adaptation", "evaluation"}
        and all(not row.get("historical", False) for row in report["cohorts"]),
        "Scientific figures require separate current adaptation/evaluation cohorts",
    )
    derived = derived_results(report)
    labels = {row["id"]: row["label"] for row in report["methods"]}
    cohorts = {}
    for cohort in report["cohorts"]:
        methods = {}
        for method, detail in derived[cohort["id"]].items():
            points = [
                {
                    **point,
                    "unsuccessful_episodes": point["episodes"] - point["successes"],
                    "success_percent": 100 * point["successes"] / point["episodes"],
                }
                for point in detail["points"]
            ]
            # The producer records static controls only at the final round. Do
            # not quietly turn a reused endpoint into a longitudinal measure.
            if cohort["id"] == "evaluation" and method != "learned_noise":
                require(
                    [row["round"] for row in points] == [report["protocol"]["rounds"]],
                    "Static comparators may only be plotted at their measured final round",
                )
            methods[method] = {
                "label": labels[method],
                "points": points,
                "rescue": copy.deepcopy(detail["rescue"]),
            }
        cohorts[cohort["id"]] = {
            "label": cohort["label"],
            "curve_kind": cohort["curve_kind"],
            "expected_episode_ids": cohort["expected_episode_ids"],
            "episodes": len(cohort["expected_episode_ids"]),
            "methods": methods,
        }
    roles = {
        role: _sum_usage(row["provider_by_role"][role] for row in tasks)
        for role in ROLES
    }
    require(
        _sum_usage(roles.values())
        == evidence["physical_provider_usage"]
        == checked["physical_cost"]["provider_usage"],
        "Plot role totals do not conserve the unique physical provider ledger",
    )
    return {
        "schema_version": PLOT_VERSION,
        "phase": report["phase"],
        "seed": report["protocol"]["seed"],
        "tasks": len(tasks),
        "protocol_sha256": evidence["protocol_sha256"],
        "cohorts": cohorts,
        "provider_by_role": roles,
        "physical_cost": checked["physical_cost"],
        "initial_success_physical_runs": evidence["initial_success_physical_runs"],
        "source_tasks": {
            row["task_key"]: {
                "input_file_sha256": row["input_file_sha256"],
                "audit_receipt": row["audit_receipt"],
            }
            for row in tasks
        },
        "conventions": [
            "Adaptation is cumulative success by revision on the same reset; all three revisions still execute.",
            "Checkpoint outcomes are independent recorded evaluations, not accumulated best-of-attempts success.",
            "Static controls appear only at their recorded final round. No interpolated or duplicated earlier result is drawn.",
            "Censor counts refer to adaptation searches with no credited success through the cap. Checkpoint failures are not time-to-success samples.",
            "Provider bars partition all unique physical calls by prompt role, including rejected calls and capped failures. No shared-baseline or historical cost is added twice.",
            "Credited success excludes initially successful and zero-action recordings without removing their prescribed resets from the denominator.",
            "Input and output are shown separately; reasoning is a subset of output. Missing usage makes known sums lower bounds. No dollar cost is inferred.",
            "All compositions are known tasks; separate resets are not held-out tasks. Development is not evaluation evidence.",
        ],
    }


def _success_figure(values):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12.8, 7.3), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.31))
    for column, name in enumerate(("adaptation", "evaluation")):
        cohort = values["cohorts"][name]
        axes = figure.add_subplot(grid[0, column])
        note = figure.add_subplot(grid[1, column])
        note.axis("off")
        lines = []
        for number, (method, detail) in enumerate(cohort["methods"].items()):
            points = detail["points"]
            color = METHOD_COLORS[method]
            axes.plot(
                [row["round"] for row in points],
                [row["success_percent"] for row in points],
                marker="o"
                if len(points) > 1
                else ("s", "^", "D", "v", "P")[number % 5],
                linestyle="-" if len(points) > 1 else "None",
                linewidth=1.8,
                markersize=6,
                color=color,
                label=detail["label"],
            )
            final = points[-1]
            if name == "adaptation":
                rescue = detail["rescue"]
                lines.append(
                    f"{detail['label']}: {final['successes']}/{final['episodes']}; "
                    f"{rescue['rescued']}/{rescue['baseline_failed']} rescued, {rescue['censored']} censored"
                )
            else:
                lines.append(
                    f"{detail['label']}: {final['successes']}/{final['episodes']} at round {final['round']}"
                )
        axes.set(
            title=(
                "Same-reset adaptation"
                if name == "adaptation"
                else "Separate-reset checkpoint evaluation"
            )
            + f" (n={cohort['episodes']})",
            xlabel="Up to this revision (0 = shared baseline)"
            if name == "adaptation"
            else "Completed adaptation round",
            ylabel="Credited success (%)",
            ylim=(-2, 102),
        )
        axes.set_xticks(
            sorted(
                {
                    row["round"]
                    for detail in cohort["methods"].values()
                    for row in detail["points"]
                }
            )
        )
        axes.grid(axis="y", alpha=0.2)
        axes.legend(fontsize=7, loc="best")
        note.text(0, 1, "\n".join(lines), va="top", fontsize=7.6, linespacing=1.7)
    figure.suptitle(
        f"FRS {values['phase']} · seed {values['seed']} · {values['tasks']} known tasks\n"
        "Cumulative retry success and learned checkpoint performance are different estimands",
        fontsize=12,
    )
    figure.supxlabel(
        "Credited success excludes initial-success / zero-action outcomes; "
        f"initial-success physical recordings: {values['initial_success_physical_runs']}. Exploratory known-task study.",
        fontsize=8,
    )
    return figure


def _token_figure(values):
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    figure = Figure(figsize=(11, 6.8), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 0.26))
    axes, note = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    note.axis("off")
    rows = [values["provider_by_role"][role] for role in ROLES]
    lower = [row["tokens"]["input_tokens"]["sum"] for row in rows]
    upper = [row["tokens"]["output_tokens"]["sum"] for row in rows]
    for number, field, amounts, bottom, color in (
        (0, "input_tokens", lower, [0] * len(rows), "#176b87"),
        (1, "output_tokens", upper, lower, "#dc7d23"),
    ):
        bars = axes.bar(
            range(len(rows)),
            amounts,
            bottom=bottom,
            color=color,
            label=("Known input", "Known output")[number],
        )
        for bar, row in zip(bars, rows):
            if row["tokens"][field]["missing_calls"]:
                bar.set_hatch("///")
    for i, row in enumerate(rows):
        missing = (
            row["tokens"]["input_tokens"]["missing_calls"]
            or row["tokens"]["output_tokens"]["missing_calls"]
        )
        axes.annotate(
            f"{'≥' if missing else ''}{lower[i] + upper[i]:,}\n{row['calls']} calls; {row['failed_calls']} failed/rejected",
            (i, lower[i] + upper[i]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes.set_xticks(range(len(rows)), ROLE_LABELS)
    axes.set(
        ylabel="Known input + output tokens",
        ylim=(0, max([1, *[a + b for a, b in zip(lower, upper)]]) * 1.23),
    )
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes.grid(axis="y", alpha=0.2)
    axes.set_axisbelow(True)
    axes.legend(fontsize=9)
    usage = values["physical_cost"]["provider_usage"]
    token_lines = [
        f"{field.replace('_', ' ')}: {usage['tokens'][field]['sum']:,} known; {usage['tokens'][field]['missing_calls']} calls missing usage"
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
        )
    ]
    note.text(
        0,
        1,
        "\n".join(token_lines)
        + f"\n{usage['preflight_failures']} no-network preflights. All recorded revisions and capped failures included; reasoning is part of output."
        + "\nAny missing usage makes that sum a lower bound; input + output known sums need not equal the independently reported total when fields are missing.",
        va="top",
        fontsize=8,
        linespacing=1.5,
    )
    figure.suptitle(
        f"FRS {values['phase']} · seed {values['seed']} · {values['tasks']} known tasks\n"
        f"Unique physical provider cost · adaptation n={values['cohorts']['adaptation']['episodes']}; evaluation n={values['cohorts']['evaluation']['episodes']} per method/checkpoint",
        fontsize=12,
    )
    return figure


def export_plots(report_path, output):
    """Write new PNG/PDF/SVG figures and an exact plotted-value/source manifest."""
    import matplotlib
    from matplotlib import rc_context

    report_path, output = Path(report_path), Path(output)
    require(not output.exists(), "Refusing to overwrite scientific figures")
    raw = report_path.read_bytes()
    report = _read_json(report_path)
    values = plotted_values(report)
    require(report_path.read_bytes() == raw, "Plot source changed during validation")
    values["report_sha256"] = hashlib.sha256(raw).hexdigest()
    values["postprocessor_sources"] = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("frs_plots.py", "frs_report.py", "frs_html_report.py")
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "plotted_values.json").write_bytes(_bytes(values))
    plotted_hash = file_sha256(output / "plotted_values.json")
    description = (
        f"Exact values SHA256 {plotted_hash}; report SHA256 {values['report_sha256']}"
    )
    with rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": PLOT_VERSION,
            "pdf.fonttype": 42,
            "font.size": 9,
        }
    ):
        for name, make in (
            ("success_and_learning", _success_figure),
            ("provider_token_cost", _token_figure),
        ):
            figure = make(values)
            for extension in ("png", "pdf", "svg"):
                metadata = {
                    "Title": f"{name}: {values['phase']}, seed {values['seed']}"
                }
                metadata.update(
                    {"Subject": description, "CreationDate": None, "ModDate": None}
                    if extension == "pdf"
                    else {"Description": description}
                )
                if extension == "svg":
                    metadata["Date"] = None
                figure.savefig(
                    output / f"{name}.{extension}", dpi=300, metadata=metadata
                )
            figure.clear()
    manifest = {
        "schema_version": PLOT_VERSION,
        "report_sha256": values["report_sha256"],
        "plotted_values_sha256": plotted_hash,
        "postprocessor_sources": values["postprocessor_sources"],
        "matplotlib_version": matplotlib.__version__,
        "phase": values["phase"],
        "seed": values["seed"],
        "files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
        },
    }
    (output / "manifest.json").write_bytes(_bytes(manifest))
    require(report_path.read_bytes() == raw, "Plot source changed during export")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_plots(args.report, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("report_sha256", "plotted_values_sha256", "phase", "seed")
            }
        )
    )


if __name__ == "__main__":
    main()
