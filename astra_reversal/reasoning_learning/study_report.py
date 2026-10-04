"""Build an offline development report from retained experiment evidence.

The input manifest contains local artifact directories, never presigned URLs.
Missing checkpoints remain missing; partial trajectories count as retained
interaction lower bounds, not completed rollouts or evaluation failures.
"""

import argparse
import csv
import io
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .teacher import SEMANTIC_EXTENSION, SYSTEM_PROMPT


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def wilson(successes, count):
    if not count or not 0 <= successes <= count:
        raise ValueError("A completed nonempty evaluation is required")
    z, p = 1.959963984540054, successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    )
    return [max(0, center - radius), min(1, center + radius)]


def usage_by_episode(directory, jobs=None):
    records = []
    if jobs and Path(jobs).exists():
        for path in sorted(Path(jobs).glob("*/completed.json")):
            request = read_json(path.parent / "request.json")
            receipt = read_json(path)["result"]["receipt"]
            records.append((request["episode_id"], receipt))
    else:
        provider = Path(directory) / "provider.jsonl"
        if provider.exists():
            for line in provider.read_text().splitlines():
                row = json.loads(line)
                if row.get("codex_receipt"):
                    records.append((row["episode_id"], row["codex_receipt"]))
    result = {}
    for episode, receipt in records:
        row = result.setdefault(
            episode,
            {
                "completed_calls": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0,
                "incomplete_usage_receipts": 0,
            },
        )
        row["completed_calls"] += 1
        usage = receipt.get("token_usage") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            row[key] += usage.get(key) or 0
        raw = receipt.get("raw_usage") or {}
        row["cached_input_tokens"] += raw.get("cached_input_tokens") or 0
        row["latency_seconds"] += receipt.get("latency_seconds") or 0
        row["incomplete_usage_receipts"] += int(
            receipt.get("token_usage_is_lower_bound", False)
            or any(
                usage.get(k) is None
                for k in ("input_tokens", "output_tokens", "total_tokens")
            )
        )
    return result


def summarize_run(spec):
    directory = Path(spec["directory"])
    runtime = read_json(directory / "runtime.json", {})
    protocol = read_json(directory / "protocol.json", {})
    collection = read_json(directory / "collection.json", [])
    curve = read_json(directory / "learning_curve.json", [])
    usage = usage_by_episode(directory, spec.get("teacher_jobs"))
    updates = read_json(directory / "updates.json", [])
    completion = read_json(directory / "completion.json", {})
    points = []
    for point in curve:
        episodes = point["episodes"]
        count, wins = len(episodes), sum(bool(x["success"]) for x in episodes)
        if (count, wins) != (point["rollouts"], point["successes"]):
            raise ValueError("Evaluation aggregate differs from retained episodes")
        expected_count = len(
            protocol.get("pilot", {}).get("autonomous_evaluation_reset_indices", [])
        )
        if expected_count and count != expected_count:
            raise ValueError(
                "Incomplete scheduled evaluation cannot produce a curve point"
            )
        if len({r["episode_id"] for r in episodes}) != count:
            raise ValueError("Repeated episode IDs in evaluation")
        episode_ids = [
            r["episode_id"] for r in collection[: point["collection_rollouts"]]
        ]
        points.append(
            {
                **{k: v for k, v in point.items() if k != "episodes"},
                "success_rate": wins / count,
                "wilson_95": wilson(wins, count),
                "teacher_total_tokens": sum(
                    usage.get(e, {}).get("total_tokens", 0) for e in episode_ids
                ),
                "episode_results": [
                    {
                        k: r[k]
                        for k in (
                            "episode_id",
                            "success",
                            "actions_executed",
                            "total_control_steps",
                        )
                    }
                    for r in episodes
                ],
            }
        )
    partial = []
    completed_ids = {r["episode_id"] for r in collection}
    for path in sorted((directory / "collection").glob("rollout_*")):
        result = read_json(path / "result.json")
        if result and result["episode_id"] in completed_ids:
            continue
        if result:
            # A saved rollout may precede the top-level aggregate if training
            # was interrupted; its interactions still count.
            partial.append(
                {
                    "directory": path.name,
                    "executed_steps_retained": result["actions_executed"],
                    "initialization_steps": result["initialization_steps"],
                    "rollout_complete_update_unconfirmed": True,
                }
            )
            continue
        steps = path / "executed_steps.jsonl"
        count = len(steps.read_text().splitlines()) if steps.exists() else 0
        initialized = bool(
            count
            or (path / "teacher_requests.jsonl").exists()
            or (path / "observation_0.npz").exists()
        )
        partial.append(
            {
                "directory": path.name,
                "executed_steps_retained": count,
                "initialization_steps": protocol.get("environment", {}).get(
                    "stabilization_steps_counted_separately", 10
                )
                if initialized
                else 0,
                "rollout_complete_update_unconfirmed": False,
            }
        )
    totals = {
        key: sum(row[key] for row in usage.values())
        for key in (
            "completed_calls",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_seconds",
            "incomplete_usage_receipts",
        )
    }
    totals["uncached_input_tokens"] = (
        totals["input_tokens"] - totals["cached_input_tokens"]
    )
    evaluation_steps, partial_evaluation = 0, False
    for path in sorted(directory.glob("evaluation_*/*")):
        result = read_json(path / "result.json")
        if result:
            evaluation_steps += result["total_control_steps"]
        elif (path / "executed_steps.jsonl").exists():
            evaluation_steps += len(
                (path / "executed_steps.jsonl").read_text().splitlines()
            ) + protocol.get("environment", {}).get(
                "stabilization_steps_counted_separately", 10
            )
            partial_evaluation = True
    latest_evaluated = max((p["policy_version"] for p in points), default=None)
    latest_updated = max((r.get("policy_version", 0) for r in updates), default=0)
    return {
        "label": spec["label"],
        "method": spec["method"],
        "workflow": spec["workflow"],
        "status": completion.get("status", spec.get("status", "incomplete")),
        "task": runtime.get("task", spec.get("task")),
        "seed": runtime.get("seed", spec.get("seed")),
        "source_revision": runtime.get("source_revision"),
        "protocol_version": protocol.get("schema_version"),
        "points": points,
        "collection": collection,
        "partial_collection": partial,
        "collection_steps_retained": sum(r["total_control_steps"] for r in collection)
        + sum(
            r["executed_steps_retained"] + r["initialization_steps"] for r in partial
        ),
        "collection_steps_may_be_lower_bound": bool(partial),
        "evaluation_steps_retained": evaluation_steps,
        "evaluation_steps_may_be_lower_bound": partial_evaluation,
        "completed_collection_rollouts": len(collection),
        "collected_successes": sum(bool(r["success"]) for r in collection),
        "policy_updates": sum(bool(r.get("updated", True)) for r in updates),
        "latest_evaluated_policy_version": latest_evaluated,
        "latest_updated_policy_version": latest_updated,
        "latest_update_has_autonomous_evaluation": bool(
            latest_evaluated is not None and latest_evaluated >= latest_updated
        ),
        "teacher_usage": totals,
        "usage_by_episode": usage,
        "threshold_crossing": next(
            (
                {
                    "collection_rollouts": p["collection_rollouts"],
                    "collection_steps": p["collection_steps"],
                }
                for p in points
                if p["success_rate"] >= 0.8
            ),
            None,
        ),
        "threshold_status": "observed"
        if any(p["success_rate"] >= 0.8 for p in points)
        else "not_reached_in_completed_evaluations",
        "videos": [],
    }


def export_figures(data, output):
    """Standalone research figures; each task and random seed stays separate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    colors = {
        "teacher_v1": "#bf571d",
        "teacher_v2": "#d89122",
        "teacher_v3": "#83432a",
        "dsrl": "#057a76",
        "ppo": "#4566ba",
    }
    groups = {}
    for run in data["runs"]:
        if run["task"] and run["points"]:
            task = run["task"]
            groups.setdefault((task["suite"], task["task_id"], run["seed"]), []).append(
                run
            )
    paths = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for (suite, task_id, seed), runs in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, key, title in zip(
            axes,
            ("collection_steps", "teacher_total_tokens"),
            ("Collection control steps", "Cumulative teacher tokens (including cache)"),
            strict=True,
        ):
            for i, run in enumerate(runs):
                points = run["points"]
                label = run["label"]
                if all(p["policy_version"] == 0 for p in points):
                    label += " (initial policy only)"
                x, y = [p[key] for p in points], [p["success_rate"] for p in points]
                errors = [
                    [p["success_rate"] - p["wilson_95"][0] for p in points],
                    [p["wilson_95"][1] - p["success_rate"] for p in points],
                ]
                color = colors.get(run["method"], "#778084")
                ax.errorbar(
                    x, y, yerr=errors, color=color, alpha=0.25, fmt="none", capsize=3
                )
                ax.plot(
                    x,
                    y,
                    marker=("o", "s", "^", "D", "v", "P")[i % 6],
                    color=color,
                    label=label,
                    linewidth=1.8,
                    markersize=6,
                )
            ax.axhline(0.8, color="#a89173", linestyle="--", linewidth=1)
            ax.set_ylim(-0.04, 1.04)
            max_x = max(p[key] for run in runs for p in run["points"])
            ax.set_xlim(-0.04 * max(1, max_x), 1.06 * max(1, max_x))
            if max_x == 0:
                ax.set_xticks([0])
                ax.text(
                    0.52,
                    0.12,
                    "No completed evaluation follows\nteacher spending yet",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="#617378",
                )
            ax.set_xlabel(title, fontsize=10)
            ax.set_ylabel("Autonomous success rate")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}")
            )
            ax.grid(axis="y", alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{suite} / task {task_id} / seed {seed}", fontsize=14, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.045),
            frameon=False,
        )
        fig.text(
            0.5,
            0.02,
            "10 reset states per point · 95% Wilson intervals · Evaluation samples accounted separately · Development evidence",
            ha="center",
            fontsize=8,
            color="#617378",
        )
        fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.29, wspace=0.3)
        stem = f"{suite}_task{task_id}_seed{seed}"
        for extension in ("png", "pdf"):
            path = output / f"{stem}.{extension}"
            fig.savefig(path, dpi=180)
            paths.append(str(Path(output.name) / path.name))
        plt.close(fig)
    return paths


def build(manifest, budget, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specs = read_json(manifest)["runs"]
    runs = [summarize_run(spec) for spec in specs]
    for i, (spec, run) in enumerate(zip(specs, runs, strict=True)):
        directory = Path(spec["directory"])
        for path in sorted(directory.glob("**/rollout.mp4")):
            relative = path.relative_to(directory)
            if relative.parts[0] != "collection" and not relative.parts[0].startswith(
                "evaluation_"
            ):
                continue
            result = read_json(path.parent / "result.json")
            if not result:
                continue
            target = Path("videos") / str(i) / relative
            (output / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output / target)
            run["videos"].append(
                {
                    "path": str(target),
                    "episode_id": result["episode_id"],
                    "success": result["success"],
                    "phase": relative.parts[0],
                    "actions": result["actions_executed"],
                }
            )
    ledger = read_json(budget)
    data = {
        "title": "Reasoning-guided policy learning",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "research_objective_met": False,
        "scope": "Development experiments; no confirmed sample-efficiency advantage over strong RL.",
        "checkpoint": "lerobot/pi05_libero_base@a217bfd3b14673cf2ce597e69997ab21866438dd",
        "budget": ledger,
        "runs": runs,
        "teacher_system_prompt": SYSTEM_PROMPT,
        "teacher_semantic_prompt_extension": SEMANTIC_EXTENSION,
        "notes": [
            "Success is the simulator's binary task predicate. Each scheduled autonomous score uses ten separate reset states; Astra supplies no inference input in these evaluations.",
            "The 80% threshold means at least 8/10 at a scheduled checkpoint. Wilson intervals are descriptive; a single crossing on development data does not prove superiority.",
            "Collection x-values include all executed collection control steps and stabilization. Evaluation interactions are disclosed separately. Partial archived trajectories are retained lower bounds.",
            "DSRL uses the same frozen decoder weights with a learned noise policy whose initial distribution differs from the native Gaussian. PPO uses strict converted weights with verified GELU compatibility.",
            "The RL baselines use pinned RLinf components in a serial OOD harness with documented overrides. This is not a reproduction claim for the stock distributed RLinf benchmarks.",
            "Token counts are CLI-reported usage, including cached input; they are not an API dollar bill. Local completed calls count even if cancellation prevented the worker from receiving them.",
            "A marked shared native baseline reuses the same previously measured episodes after deployment and reset identity checks. It contributes no new evaluation interactions or independent statistical replicate. Updated policies always receive fresh evaluations.",
            "An initial-policy score does not evaluate a later update. Runs stopped between scheduled checkpoints explicitly mark their latest policy update as unevaluated.",
            "Candidate preference is predicted improvement. Only selected commands execute; full observed-useful action windows train. No FRS action steering, physical candidate retries, privileged object poses, or default synthetic training.",
        ],
    }
    data["figures"] = export_figures(data, output / "figures")
    (output / "results.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    stream = io.StringIO()
    fields = [
        "method",
        "workflow",
        "suite",
        "task_id",
        "seed",
        "collection_rollouts",
        "collection_steps",
        "successes",
        "rollouts",
        "success_rate",
        "teacher_total_tokens",
        "evaluation_steps_cumulative",
        "reused_from_workflow",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for run in runs:
        for point in run["points"]:
            writer.writerow(
                {
                    "method": run["label"],
                    "workflow": run["workflow"],
                    **(run["task"] or {}),
                    "seed": run["seed"],
                    **{k: point[k] for k in fields if k in point},
                }
            )
    (output / "learning_curves.csv").write_text(stream.getvalue())
    template = Path(__file__).with_name("study_dashboard.html").read_text()
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__STUDY_DATA__", payload))
    (output / "README.txt").write_text(
        "Open index.html in a browser. All assets are local.\nThe JSON and CSV contain retained results, not projected outcomes.\n"
        + "\n".join(data["notes"])
        + "\n"
    )
    return {
        "runs": len(runs),
        "points": sum(len(r["points"]) for r in runs),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("budget", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.budget, args.output)))


if __name__ == "__main__":
    main()
