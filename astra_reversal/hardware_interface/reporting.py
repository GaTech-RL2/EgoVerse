"""Public aggregate artifacts and a transparent pilot-based power worksheet."""

import csv
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .analysis import analyze
from .common import digest, file_hash, write_json


def power_worksheet(outcomes, manifest):
    """Planning approximation, never an automatic confirmation authorization."""
    if manifest["replicates"] != 1:
        raise ValueError("repeated_actor_sessions_require_clustered_power_simulation")
    blocks = {}
    for row in outcomes:
        if row["split"] != "pilot" or not row["actor_started"]:
            continue
        key = (row["task_id"], row["init_state_index"])
        block = blocks.setdefault(key, {})
        if row["condition"] in block:
            raise ValueError("duplicate_pilot_trial")
        block[row["condition"]] = row
    expected = {(t, i) for t in manifest["task_ids"] for i in manifest["pilot_indices"]}
    if set(blocks) != expected or any(
        set(b) != set(manifest["conditions"]) for b in blocks.values()
    ):
        raise ValueError("complete_paired_pilot_required_for_power")
    for block in blocks.values():
        if len({(r["init_state_hash"], r["env_seed"]) for r in block.values()}) != 1:
            raise ValueError("unmatched_pilot_states")
        if any(
            r.get("independent_evaluation_passed") is not True for r in block.values()
        ):
            raise ValueError("audited_pilot_required_for_power")
        if any(
            r.get("terminal_reason")
            in (
                "MODEL_ERROR",
                "TOOL_ERROR",
                "PROTOCOL_DEVIATION",
                "RESET_FAILURE",
                "EVALUATOR_ERROR",
            )
            for r in block.values()
        ):
            raise ValueError("resolve_pilot_infrastructure_failures_before_power")
    n = len(blocks)
    discordant = sum(b["F"]["success"] != b["B0"]["success"] for b in blocks.values())
    q = discordant / n
    z = NormalDist().inv_cdf(0.975)
    center = (q + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    q_high = min(1.0, center + radius)
    target = manifest["power"]
    delta = target["target_risk_difference"]
    if (
        not 0 < delta < 1
        or not 0 < target["alpha"] < 1
        or not 0.5 < target["power"] < 1
    ):
        raise ValueError("invalid_power_design")
    scenarios = {}
    for label, rate in (
        ("pilot_estimate", q),
        ("upper_discordance_sensitivity", q_high),
    ):
        # |delta| <= Pr(discordance) is necessary for a paired binary effect.
        rate = max(rate, delta)
        z_alpha = NormalDist().inv_cdf(1 - target["alpha"] / 2)
        z_power = NormalDist().inv_cdf(target["power"])
        total = math.ceil(
            (z_alpha * math.sqrt(rate) + z_power * math.sqrt(rate - delta * delta)) ** 2
            / (delta * delta)
        )
        per_task = max(10, math.ceil(total / len(manifest["task_ids"])))
        scenarios[label] = {
            "assumed_discordant_rate": rate,
            "initial_states_per_task": per_task,
            "total_paired_blocks": per_task * len(manifest["task_ids"]),
        }
    return {
        "pilot_manifest_sha256": digest(manifest),
        "pilot_outcomes_sha256": digest(
            sorted(
                outcomes,
                key=lambda r: (r["task_id"], r["init_state_index"], r["condition"]),
            )
        ),
        "primary_contrast": "F_minus_B0",
        "pilot_blocks": n,
        "discordant_pairs": discordant,
        "discordance_estimate": q,
        "discordance_wilson_95pct": [max(0, center - radius), q_high],
        "target": target,
        "scenarios": scenarios,
        "recommended_initial_states_per_task": scenarios[
            "upper_discordance_sensitivity"
        ]["initial_states_per_task"],
        "method": "Paired binary normal approximation: N=(z_(1-alpha/2)*sqrt(q)+z_power*sqrt(q-delta^2))^2/delta^2; round to equal task allocation, minimum 10 states/task.",
        "limitations": [
            "Planning approximation, not exact power for the task-blocked bootstrap decision.",
            "The upper Wilson discordance scenario reflects uncertainty in the small pilot.",
            "Before confirmation, verify available disjoint official states, document the design choice, and freeze a new manifest; this function does not authorize a launch.",
        ],
        "confirmation_frozen": False,
    }


def _csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(outcomes, manifest, destination, *, split):
    """No transcripts, credentials, images or private model reasoning are exported."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    result = analyze(outcomes, manifest, split=split)
    write_json(destination / "analysis.json", result)
    rows = []
    for arm, tasks in result["per_task"].items():
        for task, value in tasks.items():
            row = {
                "condition": arm,
                "task_id": task,
                "n": value["n"],
                "successes": value["successes"],
                "success_rate": value["success_rate"],
            }
            for metric in ("steps", "wall_seconds", "tokens", "cost"):
                summary = value[metric]
                row.update(
                    {
                        metric + "_median": summary["median"],
                        metric + "_q25": summary["iqr"][0] if summary["iqr"] else None,
                        metric + "_q75": summary["iqr"][1] if summary["iqr"] else None,
                        metric + "_unknown": summary["n_unknown"],
                    }
                )
            rows.append(row)
    _csv(destination / "per-task.csv", rows)
    _csv(
        destination / "paired-comparisons.csv",
        [
            {
                "contrast": name,
                "task_id": task,
                "risk_difference": value["risk_difference"],
                "ci_lower": value["bootstrap_95pct"][0],
                "ci_upper": value["bootstrap_95pct"][1],
                "clusters": value["initial_state_clusters"],
            }
            for name, comparison in result["comparisons"].items()
            for task, value in comparison.get("per_task", {}).items()
        ],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"F": "#247b7b", "B0": "#d87924", "B": "#8260a8"}
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(manifest["task_ids"]))
    for i, arm in enumerate(manifest["conditions"]):
        tasks = result["per_task"][arm]
        values = [tasks[str(t)]["success_rate"] for t in manifest["task_ids"]]
        bars = axis.bar(
            x + (i - 1) * 0.25,
            [v if v is not None else 0 for v in values],
            width=0.24,
            color=colors[arm],
            label=arm,
        )
        for bar, task in zip(bars, manifest["task_ids"]):
            value = tasks[str(task)]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f'{value["successes"]}/{value["n"]}' if value["n"] else "missing",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if not value["n"] else 0,
            )
    axis.set(
        xticks=x,
        xticklabels=[str(t) for t in manifest["task_ids"]],
        xlabel="Official task ID (order 0)",
        ylabel="Success fraction",
        ylim=(0, 1.16),
        title=f"{split.capitalize()} — per-task results"
        + (" (unscored)" if split == "pilot" else ""),
    )
    axis.legend(frameon=False)
    figure.savefig(destination / "success-by-task.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    selected = [r for r in outcomes if r["split"] == split and r["actor_started"]]
    for axis, metric, label in zip(
        axes,
        ("sim_steps", "wall_s", "known_workflow_tokens"),
        (
            "Control steps (all started)",
            "Wall seconds (observed; timeout censored)",
            "Tokens (known usage only)",
        ),
    ):
        for arm in manifest["conditions"]:
            values = sorted(
                r[metric]
                for r in selected
                if r["condition"] == arm
                and (
                    metric != "known_workflow_tokens" or not r["unknown_usage_records"]
                )
            )
            if values:
                axis.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    color=colors[arm],
                    label=arm,
                )
        axis.set(xlabel=label, ylabel="Empirical cumulative fraction", ylim=(0, 1.04))
        axis.grid(alpha=0.2)
    if selected:
        axes[0].legend(frameon=False)
    figure.suptitle(
        f"{split.capitalize()} resource distributions — no success-only selection"
    )
    figure.savefig(destination / "resource-distributions.png", dpi=180)
    plt.close(figure)
    write_json(
        destination / "artifact-manifest.json",
        {
            "study_manifest_sha256": digest(manifest),
            "analysis_source_sha256": file_hash(
                Path(__file__).with_name("analysis.py")
            ),
            "reporting_source_sha256": file_hash(__file__),
            "files": {
                p.name: file_hash(p)
                for p in sorted(destination.iterdir())
                if p.is_file()
            },
            "raw_transcripts_included": False,
        },
    )
    return result
