"""Two audit-gated scientific figures from recorded image-study results only."""

import argparse
import json
import statistics
from pathlib import Path

from .astra_client import _strict_json
from .records import digest, file_sha256

ARMS = (
    "random_noise",
    "random_occlusion",
    "astra_occlusion",
    "random_demo_blend",
    "astra_demo_blend",
)
ASTRA = ("astra_occlusion", "astra_demo_blend")
LABELS = {
    "random_noise": "Random noise",
    "random_occlusion": "Random occlusion",
    "astra_occlusion": "Astra occlusion",
    "random_demo_blend": "Random demo blend",
    "astra_demo_blend": "Astra demo blend",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _count(value):
    _require(type(value) is int and value >= 0, "Expected a nonnegative integer count")
    return value


def _tokens(provider):
    tokens = provider["tokens"]["total_tokens"]
    calls = _count(provider["provider_calls"])
    _require(
        _count(tokens["available_calls"]) + _count(tokens["missing_calls"]) == calls
        and type(tokens["complete"]) is bool
        and tokens["complete"] == (tokens["missing_calls"] == 0),
        "Token availability does not match actual provider calls",
    )
    return _count(tokens["sum"]), calls, tokens["missing_calls"]


def _stats(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def plotted_values(report):
    """Check denominators and independent report views before deriving figure data."""
    cases, groups = report["cases"], report["groups"]["pooled"]
    n = len(cases)
    _require(
        report["schema_version"] == "image-perturbation-report-1.0"
        and report["status"] == "complete"
        and all(
            report[key] is True
            for key in ("complete", "complete_episode_coverage", "efficacy_released")
        )
        and groups["efficacy_released"] is True
        and n > 0
        and all(
            report[key] == n
            for key in ("expected_cases", "completed_cases", "audited_cases")
        )
        and groups["recorded_completed_cases"] == n
        and not any(
            report[key]
            for key in (
                "audit_pending_episode_ids",
                "missing_or_incomplete_episode_ids",
                "problems",
            )
        ),
        "Figures require complete coverage and passed independent audits",
    )
    ids = [case["episode_id"] for case in cases]
    _require(
        len(set(ids)) == n and set(ids) == set(report["expected_episode_ids"]),
        "Episode denominator mismatch",
    )
    _require(
        len({case["seed"] for case in cases})
        == len({case["development"] for case in cases})
        == 1,
        "Do not pool phases or seeds in one figure",
    )
    _require(
        all(
            type(case["development"]) is bool and type(case["baseline_success"]) is bool
            for case in cases
        ),
        "Invalid scope/outcome flags",
    )
    _require(
        digest(report["case_arm_rows"])
        == digest([row for case in cases for row in case["arm_rows"]]),
        "Case-arm rows differ from their case records",
    )
    _require(
        digest(report["physical_attempt_rows"])
        == digest([row for case in cases for row in case["physical_attempt_rows"]]),
        "Physical rows differ from their case records",
    )
    baseline = sum(case["baseline_success"] for case in cases)
    controls = {"recovered_noise": baseline, "known_noise": 0, "policy_fresh": 0}
    rows = {arm: [] for arm in ARMS}
    for case in cases:
        _require(
            case["audit"]["status"] == "passed"
            and case["audit"]["all_arrays_verified"] is True
            and case["audit"]["summary_sha256"] == case["summary_sha256"],
            "Case lacks its bound array audit",
        )
        _require(
            set(case["controls"]) == {"known_noise", "policy_fresh"}
            and {row["arm"] for row in case["arm_rows"]} == set(ARMS)
            and len(case["arm_rows"]) == len(ARMS),
            "Missing/duplicate arm or control",
        )
        for mode in controls:
            matching = [
                row for row in case["physical_attempt_rows"] if row["mode"] == mode
            ]
            expected = (
                case["baseline_success"]
                if mode == "recovered_noise"
                else case["controls"][mode]["success"]
            )
            _require(
                type(expected) is bool
                and len(matching) == 1
                and matching[0]["success"] is expected,
                "Independent native control outcome mismatch",
            )
            if mode != "recovered_noise":
                controls[mode] += expected
        for row in case["arm_rows"]:
            series = row["success_by_attempt"]
            _require(
                len(series) == 3
                and all(type(value) is bool for value in series)
                and series == sorted(series)
                and series[0] is case["baseline_success"]
                and series[-1] is row["success"],
                "Invalid cumulative success sequence/shared baseline",
            )
            first = next(
                (index + 1 for index, value in enumerate(series) if value), None
            )
            _require(
                row["episode_id"] == case["episode_id"]
                and row["baseline_success"] is case["baseline_success"]
                and row["first_success_attempt"] == first
                and row["full_rollout_revisions_to_success"]
                == (first - 1 if first else None)
                and row["rescued"] is (row["success"] and not case["baseline_success"])
                and row["censored_without_success"] is (not row["success"]),
                "Success/rescue/censor fields disagree",
            )
            total, calls, missing = _tokens(row["physical_provider"])
            attempts = [
                value
                for value in case["physical_attempt_rows"]
                if value["mode"] == row["arm"]
            ]
            _require(
                (total, calls, missing)
                == tuple(
                    sum(_tokens(value["provider"])[i] for value in attempts)
                    for i in range(3)
                ),
                "All-call usage differs from unique physical attempts",
            )
            prefix, _, prefix_missing = _tokens(row["provider_through_success_or_cap"])
            _require(
                row["complete_tokens_to_success"]
                == (prefix if first and prefix_missing == 0 else None),
                "Tokens-to-success has incorrect missing-usage semantics",
            )
            rows[row["arm"]].append(row)
    successes, costs = {}, {}
    for arm, values in rows.items():
        group = groups["arms"][arm]
        counts = [
            sum(row["success_by_attempt"][index] for row in values)
            for index in range(3)
        ]
        rescued = [row for row in values if row["rescued"]]
        censored = [row for row in values if row["censored_without_success"]]
        _require(
            group["cases"] == n
            and group["baseline_successes"] == baseline
            and group["successes_by_attempt"] == counts
            and group["successes"] == counts[-1]
            and group["rescues"] == len(rescued)
            and group["censored_without_success"] == len(censored)
            and baseline + len(rescued) + len(censored) == n,
            "Pooled success counts differ from case rows",
        )
        successes[arm] = {
            "successes_by_attempt": counts,
            "rescues": len(rescued),
            "censored": len(censored),
        }
        if arm not in ASTRA:
            continue
        partitions = {"rescued": 0, "censored": 0, "baseline_success": 0}
        calls = missing = 0
        per_case = []
        for row in values:
            tokens, call_count, missing_count = _tokens(row["physical_provider"])
            bucket = (
                "baseline_success"
                if row["baseline_success"]
                else "rescued"
                if row["rescued"]
                else "censored"
            )
            partitions[bucket] += tokens
            calls += call_count
            missing += missing_count
            per_case.append(
                {
                    "episode_id": row["episode_id"],
                    "bucket": bucket,
                    "recorded_tokens": tokens,
                    "calls": call_count,
                    "missing_usage_calls": missing_count,
                    "tokens_to_success": row["complete_tokens_to_success"],
                    "calls_through_success_or_cap": row[
                        "provider_through_success_or_cap"
                    ]["provider_calls"],
                    "revisions_to_success": row["full_rollout_revisions_to_success"],
                }
            )
        rescue_stats = {
            "physical_provider_calls_through_success": _stats(
                [
                    row["provider_through_success_or_cap"]["provider_calls"]
                    for row in rescued
                ]
            ),
            "full_rollout_revisions": _stats(
                [row["full_rollout_revisions_to_success"] for row in rescued]
            ),
            "complete_total_tokens_to_success": _stats(
                [
                    row["complete_tokens_to_success"]
                    for row in rescued
                    if row["complete_tokens_to_success"] is not None
                ]
            ),
        }
        _require(
            all(
                group["rescue_only"][key] == value
                for key, value in rescue_stats.items()
            ),
            "Rescue-only medians differ from individual successes",
        )
        costs[arm] = {
            "all_call_tokens": sum(partitions.values()),
            "token_partitions": partitions,
            "calls": calls,
            "missing_usage_calls": missing,
            "usage_complete": missing == 0,
            "rescues": len(rescued),
            "censored": len(censored),
            "baseline_successes": baseline,
            "rescue_only": rescue_stats,
            "cases": per_case,
        }
    aggregate = _tokens(groups["physical_cost"]["token_usage"])
    _require(
        aggregate
        == tuple(
            sum(cost[key] for cost in costs.values())
            for key in ("all_call_tokens", "calls", "missing_usage_calls")
        ),
        "Astra costs do not reconcile to total physical provider usage",
    )
    return {
        "n": n,
        "seed": cases[0]["seed"],
        "development": cases[0]["development"],
        "episode_ids": ids,
        "suite_counts": {
            suite: sum(case["suite"] == suite for case in cases)
            for suite in sorted({case["suite"] for case in cases})
        },
        "controls": controls,
        "arms": successes,
        "astra_costs": costs,
        "protocol_sha256": report["protocol_sha256"],
        "library_id": report["image_library_id"],
    }


def _heading(values):
    phase = "DEVELOPMENT" if values["development"] else "EXPLORATORY EVALUATION"
    return f"{phase} | N={values['n']} fixed known task cases | seed {values['seed']} | complete independent audits"


def _success_figure(values, plt):
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1, 2.0]}
    )
    n, baseline = values["n"], values["controls"]["recovered_noise"]
    left, right = axes
    numbers = [
        values["controls"][name]
        for name in ("recovered_noise", "known_noise", "policy_fresh")
    ]
    left.barh(range(3), numbers, color="#64748b", height=0.5)
    left.set_yticks(
        range(3),
        ["Recovered noise\n(shared baseline)", "Known fixed noise", "Fresh noise"],
    )
    left.set_ylim(2.65, -0.65)
    left.set_title("Native controls: one rollout", loc="left", fontsize=11)
    for index, count in enumerate(numbers):
        left.text(count + n * 0.025, index, f"{count}/{n}", va="center", fontsize=10)
    y = np.arange(len(ARMS))
    for index, (label, color, offset) in enumerate(
        (("After 1 revision", "#93c5fd", -0.17), ("After 2 revisions", "#1d4ed8", 0.17))
    ):
        counts = [
            values["arms"][arm]["successes_by_attempt"][index + 1] for arm in ARMS
        ]
        right.barh(y + offset, counts, height=0.3, color=color, label=label)
        for row, count in enumerate(counts):
            right.text(
                count + n * 0.022, row + offset, f"{count}/{n}", va="center", fontsize=9
            )
    right.axvline(baseline, color="#475569", linestyle="--", linewidth=1.1)
    right.set_yticks(y, [LABELS[arm] for arm in ARMS])
    right.set_ylim(len(ARMS) - 0.45, -0.55)
    right.set_title(
        f"Cumulative success; common baseline {baseline}/{n} included",
        loc="left",
        fontsize=11,
    )
    figure.legend(
        *right.get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.58, 0.10),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    for axis in axes:
        axis.set_xlim(0, n * 1.15)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axis.set_xlabel("Successful cases")
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Image interventions: success by revision budget",
        x=0.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(0.02, 0.92, _heading(values), fontsize=10, color="#475569")
    figure.text(
        0.02,
        0.035,
        "One revision = one additional full rollout after the shared baseline. Controls are independent single rollouts.\nCounts describe these fixed cases; no population confidence interval. A successful arm need not demonstrate a causal image-edit benefit.",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(left=0.18, right=0.97, top=0.82, bottom=0.26, wspace=0.75)
    return figure


def _cost_figure(values, plt):
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    costs = [values["astra_costs"][arm] for arm in ASTRA]
    colors = {
        "censored": "#94a3b8",
        "rescued": "#0f766e",
        "baseline_success": "#a78bfa",
    }
    labels = {
        "censored": "Capped failed searches",
        "rescued": "Rescued cases",
        "baseline_success": "Baseline already succeeded",
    }
    positions = [0.0, 1.0]
    starts = [0.0, 0.0]
    for bucket in colors:
        widths = [row["token_partitions"][bucket] / 1e6 for row in costs]
        if any(widths):
            left.barh(
                positions,
                widths,
                left=starts,
                color=colors[bucket],
                height=0.38,
                label=labels[bucket],
            )
        starts = [a + b for a, b in zip(starts, widths, strict=True)]
    maximum = max(starts) or 1
    for index, row in enumerate(costs):
        suffix = "+" if not row["usage_complete"] else ""
        left.text(
            starts[index] + maximum * 0.025,
            index - 0.06,
            f"{row['all_call_tokens']:,}{suffix}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        left.text(
            0,
            index + 0.29,
            f"{row['calls']} calls; {row['missing_usage_calls']} missing usage",
            fontsize=9,
            color="#475569",
        )
    left.set_yticks(positions, [LABELS[arm] for arm in ASTRA])
    left.set_xlim(0, maximum * 1.42)
    left.set_ylim(1.65, -0.6)
    left.set_title("All recorded Astra calls", loc="left", fontsize=12)
    left.set_xlabel("Recorded input + output tokens (millions)")
    left.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    left.set_axisbelow(True)
    left.spines[["top", "right"]].set_visible(False)
    figure.legend(
        *left.get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.16, 0.11),
        frameon=False,
        fontsize=8,
        ncol=3,
    )
    right.set_xlim(0, 1)
    right.set_ylim(1.65, -0.6)
    right.axis("off")
    right.set_title("Rescue-only medians to first success", loc="left", fontsize=12)
    columns = (
        (0.14, "Calls", "physical_provider_calls_through_success"),
        (0.44, "Revisions", "full_rollout_revisions"),
        (0.81, "Tokens", "complete_total_tokens_to_success"),
    )
    for x, label, _ in columns:
        right.text(x, -0.39, label, ha="center", fontsize=10, color="#475569")
    for index, row in enumerate(costs):
        for x, _, key in columns:
            metric = row["rescue_only"][key]
            value = (
                "N/A"
                if metric["median"] is None
                else f"{metric['median']:,.0f}"
                if float(metric["median"]).is_integer()
                else f"{metric['median']:,.1f}"
            )
            right.text(
                x,
                index - 0.02,
                value,
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
            )
            if (
                key == "complete_total_tokens_to_success"
                and metric["count"] != row["rescues"]
            ):
                right.text(
                    x,
                    index + 0.15,
                    f"{metric['count']}/{row['rescues']} complete",
                    ha="center",
                    fontsize=8,
                )
        right.text(
            0.01,
            index + 0.30,
            f"{row['rescues']} rescued; {row['censored']} capped failures; {row['baseline_successes']} baseline successes",
            fontsize=9,
            color="#475569",
        )
    figure.suptitle(
        "Astra image interventions: observed token cost",
        x=0.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(0.02, 0.92, _heading(values), fontsize=10, color="#475569")
    figure.text(
        0.02,
        0.035,
        "Left includes rejected calls, failed searches and any development extras. Missing usage makes totals lower bounds (+).\nRight conditions on baseline failure followed by success; capped failures are excluded from medians. No price conversion.",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(left=0.16, right=0.98, top=0.80, bottom=0.24, wspace=0.15)
    return figure


def export_plots(report_path, output, *, svg=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_path, output = Path(report_path), Path(output)
    _require(not output.exists(), "Plot output must be a new directory")
    checksum = file_sha256(report_path)
    report = _strict_json(report_path.read_bytes())
    values = plotted_values(report)
    _require(file_sha256(report_path) == checksum, "Report changed while reading")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": checksum}
    ):
        for name, renderer in (
            ("success_by_revision", _success_figure),
            ("astra_token_cost", _cost_figure),
        ):
            figure = renderer(values, plt)
            for extension in ("png", "svg") if svg else ("png",):
                path = output / f"{name}.{extension}"
                figure.savefig(
                    path,
                    dpi=200,
                    facecolor="white",
                    metadata={"Date": None} if extension == "svg" else None,
                )
                files[path.name] = {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            plt.close(figure)
    source = {
        "schema_version": "image-perturbation-figure-source-1.0",
        "report_sha256": checksum,
        "report_schema_version": report["schema_version"],
        "plotter_source_sha256": file_sha256(__file__),
        "matplotlib_version": matplotlib.__version__,
        "plotted_values": values,
        "figure_files": files,
        "definitions": {
            "revision": "one complete additional rollout after the common baseline; attempt1=baseline, attempt2=one revision, attempt3=two revisions",
            "all_call_tokens": "available input plus output tokens across every physical call, including rejections and censored/development-extra rollouts; reasoning is already included in output",
            "rescue": "baseline failed and this arm subsequently succeeded",
            "censored": "no success before the capped search ended; these cases are excluded from rescue-only medians",
            "rescue_tokens": "median among rescued cases with complete usage; available subset count is explicit",
            "population_confidence_interval": None,
            "monetary_price_assumed": False,
        },
    }
    (output / "figure_source.json").write_text(
        json.dumps(source, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args(argv)
    result = export_plots(args.report, args.output, svg=args.svg)
    print(
        json.dumps(
            {
                "report_sha256": result["report_sha256"],
                "figure_source_sha256": file_sha256(args.output / "figure_source.json"),
                "figures": list(result["figure_files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
