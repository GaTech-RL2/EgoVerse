"""Success-aware speed summaries; failures never become fast completions."""

import math
from collections import Counter
from statistics import median

from .protocol import contrasts

TIMING_FIELDS = (
    "provider_wall_seconds",
    "client_admission_seconds",
    "client_retry_backoff_seconds",
    "simulator_and_step_evaluator_seconds",
)


def nonnegative(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_speed_measurement")
    return value


def summary(values):
    known = sorted(nonnegative(v) for v in values if v is not None)
    return {
        "n": len(known),
        "unknown": len(values) - len(known),
        "median": median(known) if known else None,
        "min": known[0] if known else None,
        "max": known[-1] if known else None,
    }


def success_time(row):
    if not row["success"]:
        return None
    if row.get("independent_evaluation_passed") is not True:
        raise ValueError("speed_requires_verified_success")
    value = row.get("first_success_wall_seconds")
    # Historical receipts have terminal wall time only. Keep the approximation
    # explicit; do not invent a first-success timestamp or rewrite old results.
    return nonnegative(row["wall_s"] if value is None else value)


def timing(row):
    values = {key: row.get(key) for key in TIMING_FIELDS}
    result = {key: None if v is None else nonnegative(v) for key, v in values.items()}
    residual = None
    if all(v is not None for v in values.values()):
        residual = nonnegative(row["wall_s"]) - sum(values.values())
        if residual < -0.01:
            raise ValueError("overlapping_timing_components")
        residual = max(0.0, residual)
    result["other_local_seconds"] = residual
    return result


def analyze_speed(outcomes, manifest, *, split):
    indices = manifest["pilot_indices" if split == "pilot" else "confirmatory_indices"]
    expected = {
        (task, index, replicate)
        for task in manifest["task_ids"]
        for index in indices
        for replicate in range(manifest["replicates"])
    }
    conditions = manifest["conditions"]
    blocks = {}
    for row in outcomes:
        if row["split"] != split:
            continue
        key = (row["task_id"], row["init_state_index"], row["replicate"])
        if key not in expected or row["condition"] not in conditions:
            raise ValueError("speed_unassigned_trial")
        block = blocks.setdefault(key, {})
        if row["condition"] in block:
            raise ValueError("speed_duplicate_trial")
        if any(
            (r["init_state_hash"], r["env_seed"])
            != (row["init_state_hash"], row["env_seed"])
            for r in block.values()
        ):
            raise ValueError("speed_unmatched_initial_state")
        block[row["condition"]] = row
        success_time(row)
    arms = {}
    deadlines = manifest.get("speed_analysis", {}).get(
        "report_times_seconds", [60, 300, 600, 1200, 1800, 3600, 7200, 14400, 28800]
    )
    for condition in conditions:
        rows = [b[condition] for b in blocks.values() if condition in b]
        successful = [r for r in rows if r["success"]]
        times = sorted(success_time(r) for r in successful)
        steps = sorted(nonnegative(r["sim_steps"]) for r in successful)
        n = len(expected)
        pieces = [timing(r) for r in rows]
        usage = {}
        for role in ("actor", "observer"):
            role_rows = [r.get("usage_by_role", {}).get(role, {}) for r in rows]
            usage[role] = {
                key: summary([r.get(key) for r in role_rows])
                for key in (
                    "calls",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "provider_wall_seconds",
                )
            }
            usage[role]["output_tokens_per_round_trip_second"] = summary(
                [
                    r["output_tokens"] / r["provider_wall_seconds"]
                    if r.get("provider_wall_seconds", 0) > 0
                    and r.get("output_tokens") is not None
                    else None
                    for r in role_rows
                ]
            )
        arms[condition] = {
            "assigned": n,
            "received": len(rows),
            "missing": n - len(rows),
            "actor_started": sum(bool(r.get("actor_started")) for r in rows),
            "verified_successes": len(successful),
            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows)),
            "terminal_wall_time_fallbacks": sum(
                r.get("first_success_wall_seconds") is None for r in successful
            ),
            "successful_wall_seconds": times,
            "successful_control_steps": steps,
            "success_by_time": [
                {
                    "seconds": t,
                    "successes": sum(v <= t for v in times),
                    "fraction_of_assigned": sum(v <= t for v in times) / n,
                }
                for t in sorted(set(deadlines + times + [0]))
            ],
            "time_to_fraction_of_assigned": {
                str(p): times[math.ceil(n * p) - 1]
                if len(times) >= math.ceil(n * p)
                else None
                for p in (0.1, 0.25, 0.5)
            },
            "time_components_seconds": {
                key: summary([r[key] for r in pieces])
                for key in (*TIMING_FIELDS, "other_local_seconds")
            },
            "simulated_control_seconds": summary(
                [r.get("simulated_control_seconds") for r in rows]
            ),
            "tool_calls": summary([r.get("tool_calls") for r in rows]),
            "usage_by_role": usage,
        }
    paired = {}
    for reference, other in contrasts(manifest):
        pairs, counts = [], Counter()
        for key in sorted(expected):
            block = blocks.get(key, {})
            if reference not in block or other not in block:
                counts["incomplete"] += 1
                continue
            left, right = block[reference], block[other]
            if left["success"] and right["success"]:
                counts["both_successful"] += 1
                lt, rt = success_time(left), success_time(right)
                pairs.append(
                    {
                        "task_id": key[0],
                        "init_state_index": key[1],
                        "replicate": key[2],
                        "reference_seconds": lt,
                        "other_seconds": rt,
                        "reference_minus_other_seconds": lt - rt,
                        "reference_over_other": lt / rt if rt > 0 else None,
                    }
                )
            elif left["success"]:
                counts["reference_only_successful"] += 1
            elif right["success"]:
                counts["other_only_successful"] += 1
            else:
                counts["neither_successful"] += 1
        ratios = [
            p["reference_over_other"]
            for p in pairs
            if p["reference_over_other"] is not None
        ]
        paired[reference + "_vs_" + other] = {
            "reference": reference,
            "other": other,
            "assigned_pairs": len(expected),
            "counts": dict(counts),
            "joint_success_pairs": pairs,
            "median_reference_minus_other_seconds": median(
                p["reference_minus_other_seconds"] for p in pairs
            )
            if pairs
            else None,
            "median_reference_over_other": median(ratios) if ratios else None,
            "interpretation": "Conditional on both methods succeeding; descriptive, not an unconditional speedup or success-equivalence claim.",
        }
    return {
        "version": 1,
        "split": split,
        "complete": all(a["missing"] == 0 for a in arms.values()),
        "population": "All assigned starts. Failed trials never count as completions; missing trials remain visible in the denominator. Partial curves are observed lower bounds, not final estimates.",
        "clock": "Actor start to first private evaluator success; excludes GPU provisioning, environment reset and independent replay. Historical terminal wall-time fallbacks are counted explicitly.",
        "provider_timing": "Round trips include network, provider queue, prefill and decoding; these are not separately observable. Output-token throughput is not isolated decode throughput.",
        "arms": arms,
        "paired": paired,
    }


def plot_speed(result, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"F": "#247b7b", "B0": "#d87924", "B": "#8260a8", "F_adaptive": "#2365d1"}
    colors.update(
        astra_CAP="#247b7b",
        astra_direct="#57b2ae",
        deepseek_CAP="#8260a8",
        deepseek_direct="#d87924",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    wall_max = max(
        [1]
        + [v for arm in result["arms"].values() for v in arm["successful_wall_seconds"]]
    )
    steps_max = max(
        [1]
        + [
            v
            for arm in result["arms"].values()
            for v in arm["successful_control_steps"]
        ]
    )
    for condition, arm in result["arms"].items():
        for axis, values, scale, xmax in (
            (axes[0], arm["successful_wall_seconds"], 60, wall_max),
            (axes[1], arm["successful_control_steps"], 1, steps_max),
        ):
            axis.step(
                [0] + [v / scale for v in values] + [xmax / scale],
                [0]
                + [100 * (i + 1) / arm["assigned"] for i in range(len(values))]
                + [100 * len(values) / arm["assigned"]],
                where="post",
                color=colors[condition],
                label=f"{condition}: {len(values)}/{arm['assigned']} success, {arm['missing']} missing",
            )
    for axis, label in zip(
        axes, ("Elapsed minutes to success", "Native control steps to success")
    ):
        axis.set(
            xlabel=label,
            ylabel="Verified successes / all assigned starts (%)",
            ylim=(0, 100),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Exploratory success and speed"
        + (" — partial observed lower bounds" if not result["complete"] else "")
    )
    figure.savefig(destination / "success-by-time-and-steps.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(
        1,
        len(result["paired"]),
        squeeze=False,
        figsize=(5 * len(result["paired"]), 4.5),
        constrained_layout=True,
    )
    for axis, pair in zip(axes[0], result["paired"].values()):
        rows = pair["joint_success_pairs"]
        x = [r["other_seconds"] / 60 for r in rows]
        y = [r["reference_seconds"] / 60 for r in rows]
        end = max(x + y + [1])
        axis.plot([0, end], [0, end], color="#888888", linestyle="--", linewidth=1)
        axis.scatter(x, y, s=30, color=colors[pair["reference"]])
        axis.set(
            xlabel=pair["other"] + " minutes to success",
            ylabel=pair["reference"] + " minutes to success",
            title=f"Joint successes: {len(rows)}/{pair['assigned_pairs']} starts\nBelow line: {pair['reference']} faster",
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Matched starts where both succeed — conditional, descriptive")
    figure.savefig(destination / "paired-success-times.png", dpi=180)
    plt.close(figure)
