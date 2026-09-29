"""Summarize recorded rollout and client timings without new inference calls."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
REPORTS = OUTPUT.parent
SOURCES = {
    "language": "phase_interpolation/evaluation/results/report.json",
    "pixels": "image_perturbations/evaluation/results/report.json",
    "frs": "frs_policy_improvement/development/report.json",
    "recipe": "learned_correction_recipe/results/report.json",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(attempts, *, action_key="actions_executed"):
    assert attempts and all(0 < a[action_key] <= 300 for a in attempts)
    times = [a["wall_seconds"] for a in attempts]
    assert all(math.isfinite(t) and t > 0 for t in times)
    capped = [a["wall_seconds"] for a in attempts if a[action_key] == 300]
    providers = [a["provider"] for a in attempts if "provider" in a]
    calls = sum(p["provider_calls"] for p in providers)
    latency = sum(p["latency_seconds"] for p in providers)
    return {
        "physical_rollouts": len(attempts),
        "actions": sum(a[action_key] for a in attempts),
        "wall_seconds_sum": sum(times),
        "wall_seconds_mean": statistics.mean(times),
        "wall_seconds_median": statistics.median(times),
        "full_300_action_rollouts": len(capped),
        "full_300_action_wall_seconds_mean": statistics.mean(capped)
        if capped
        else None,
        "full_300_action_wall_seconds_median": statistics.median(capped)
        if capped
        else None,
        "provider_calls": calls,
        "client_latency_seconds_sum": latency,
        "client_latency_seconds_mean": latency / calls if calls else None,
    }


def main():
    data = {k: json.loads((REPORTS / p).read_text()) for k, p in SOURCES.items()}
    language, pixels, recipe, frs = (
        data[k] for k in ("language", "pixels", "recipe", "frs")
    )
    assert language["status"] == "complete" and language["seed"] == 29
    assert pixels["complete"] and pixels["efficacy_released"]
    assert recipe["status"] == "complete" and recipe["protocol"]["seed"] == 61
    assert language["protocol"]["astra"]["call_interval"] == 25
    assert language["protocol"]["execute_steps"] == 5
    assert recipe["protocol"]["execute_steps"] == 5
    rows = []
    for cohort, source, arms, action_key in (
        (
            "language",
            language,
            {
                "native": "Native baseline (recovered noise)",
                "astra_tei": "Astra TEI",
                "astra_tli": "Astra TLI",
                "astra_tli_vision": "Astra TLI + annotations",
            },
            "actions_executed",
        ),
        (
            "pixels",
            pixels,
            {
                "native": "Native baseline (recovered noise)",
                "astra_occlusion": "Astra occlusion",
                "astra_demo_blend": "Astra demonstration-image blend",
            },
            "actions",
        ),
    ):
        assert len(source["cases"]) == 20
        baseline = None
        for arm, label in arms.items():
            if cohort == "language":
                attempts = (
                    [c["baseline"] for c in source["cases"]]
                    if arm == "native"
                    else [
                        a
                        for c in source["cases"]
                        for a in c["arms"][arm]["attempts"]
                        if a["mode"] == arm
                    ]
                )
            else:
                mode = "recovered_noise" if arm == "native" else arm
                attempts = [
                    a for a in source["physical_attempt_rows"] if a["mode"] == mode
                ]
            row = {"cohort": cohort, "method": arm, "label": label}
            row.update(summarize(attempts, action_key=action_key))
            if arm == "native":
                assert len(attempts) == 20 and row["provider_calls"] == 0
                baseline = row["wall_seconds_sum"]
            row["baseline_plus_retries_mean_rollout_seconds_per_case"] = (
                row["wall_seconds_sum"] + (baseline if arm != "native" else 0)
            ) / 20
            if arm != "native":
                providers = (
                    [source["groups"]["pooled"]["arms"][arm]["physical_provider"]]
                    if cohort == "language"
                    else [
                        r["provider_through_success_or_cap"]
                        for r in source["case_arm_rows"]
                        if r["arm"] == arm
                    ]
                )
                assert row["provider_calls"] == sum(
                    p["provider_calls"] for p in providers
                )
                assert math.isclose(
                    row["client_latency_seconds_sum"],
                    sum(p["latency_seconds"] for p in providers),
                )
            rows.append(row)

    for arm, label in {
        "native": "Native baseline",
        "recorded_schedule": "Recorded teacher schedule",
        "learned_selector": "Learned TEI/TLI selector",
        "flow_head": "Learned flow head",
        "gated_flow_head": "Selector-gated flow head",
    }.items():
        attempts = [
            a for a in recipe["rows"] if a["cohort"] == "ood" and a["arm"] == arm
        ]
        assert len(attempts) == 40
        cost = recipe["groups"]["ood"]["methods"][arm]["cost"]
        assert cost["provider_calls"] == cost["provider_tokens"] == 0
        row = {"cohort": "recipe", "method": arm, "label": label}
        row.update(summarize(attempts))
        for key in ("policy_seconds", "policy_replans", "environment_seconds"):
            assert math.isclose(sum(a[key] for a in attempts), cost[key])
            row[key] = cost[key]
        assert math.isclose(row["wall_seconds_sum"], cost["wall_seconds"])
        row["policy_callback_seconds_per_replan"] = (
            cost["policy_seconds"] / cost["policy_replans"]
        )
        row["executed_actions_per_wall_second"] = (
            row["actions"] / row["wall_seconds_sum"]
        )
        rows.append(row)

    cohort = next(c for c in frs["cohorts"] if c["id"] == "evaluation")
    final = next(r for r in cohort["rounds"] if r["index"] == 3)
    frs_rows = []
    assert frs["protocol"]["execute_steps"] == 10
    for method in ("native_euler10", "astra_frs"):
        episodes = [e for e in final["episodes"] if e["method_id"] == method]
        assert len(episodes) == 3
        for episode in episodes:
            row = {
                key: episode[key]
                for key in (
                    "episode_id",
                    "method_id",
                    "actions",
                    "wall_seconds",
                    "success",
                    "velocity_evaluations",
                )
            }
            row["provider_calls"] = episode["provider_usage"]["calls"]
            row["frs_edits"] = episode["recorded_counters"]["interventions"]
            if method == "astra_frs":
                assert row["velocity_evaluations"] == (
                    10 * math.ceil(row["actions"] / 10) + 20 * row["frs_edits"]
                )
            frs_rows.append(row)
    payload = {
        "schema_version": "smart-system2-runtime-1.0",
        "source_sha256": {p: sha(REPORTS / p) for p in SOURCES.values()},
        "scope": "Recorded OSMO L40S rollout work; no new executions. Client timings include payload construction, network/provider wait and response validation, not isolated model inference.",
        "full_length_summary": "Median among actual 300-action attempts; counts and trajectories differ across methods. This conditional subset is not used to compute success rates.",
        "rollout_wall_scope": "Includes reset, policy callbacks, simulator, recording and video closure; excludes model loading, worker setup and queue time. Baselines in language/pixel cohorts also include one-time initialization/numerical checks inside the first action callback.",
        "policy_callback_scope": "Combined policy preparation, interventions, generation, decoding and callback logging; not isolated GPU latency or isolated selector/head overhead.",
        "rows": rows,
        "frs_pilot_episodes": frs_rows,
        "files": {"build_runtime.py": sha(Path(__file__).resolve())},
    }
    lines = [
        "# Rollout and intervention speed",
        "",
        "These are recorded wall-clock measurements on the OSMO L40S workers. They describe this synchronous research harness, not an optimized real-robot controller. No new rollouts or inference calls were made for this summary.",
        "",
        "## Experiment A: online Astra",
        "",
        "Astra is queried at actions 0, 25, …, 275 in the language and image studies: at most twelve calls per 300-action attempt. The simulator waits for each response. The selected edit is used between queries; the policy replans every five executed actions. A response can select an ineffective edit or be rejected, so call frequency is not the frequency of effective edits.",
        "",
        "**Astra client latency averages 9–11 seconds per call.** This timer includes payload construction, network/provider wait and response validation. It is not isolated provider inference time. Means include physical calls whose proposals were rejected or whose requests failed.",
        "",
        "| Cohort | Method | Mean client seconds/call (calls) | Median seconds for 300 actions (rollouts) | Mean seconds/physical attempt |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["cohort"] == "recipe":
            continue
        client = (
            f"{row['client_latency_seconds_mean']:.2f} ({row['provider_calls']})"
            if row["provider_calls"]
            else "— (0)"
        )
        lines.append(
            f"| {row['cohort']} | {row['label']} | {client} | "
            f"{row['full_300_action_wall_seconds_median']:.1f} ({row['full_300_action_rollouts']}) | "
            f"{row['wall_seconds_mean']:.1f} |"
        )
    lines += [
        "",
        "The 300-action medians condition on attempts that reached the cap; earlier successes are excluded from that column. The last column includes all actual attempts of that method, including early successes. Online rows contain rescue attempts only, with the shared native baseline counted once in its own row. These are descriptive timings, not paired estimates of the cost of an edit. Language/image native timings include one-time noise initialization and numerical checks; they should not be treated as a steady-state policy benchmark.",
        "",
        "Including each arm's native baseline and all retries through success or the cap, mean summed rollout time per evaluated case was:",
        "",
        "| Cohort | Method | Mean seconds/case | Matched baseline seconds/case |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        if row["cohort"] == "recipe" or row["method"] == "native":
            continue
        baseline = next(
            r for r in rows if r["cohort"] == row["cohort"] and r["method"] == "native"
        )
        lines.append(
            f"| {row['cohort']} | {row['label']} | "
            f"{row['baseline_plus_retries_mean_rollout_seconds_per_case']:.1f} | "
            f"{baseline['wall_seconds_mean']:.1f} |"
        )
    lines += [
        "",
        "These case averages include baseline successes requiring no Astra calls and failed retries. They exclude work outside the rollout timer, worker setup and queue time. Summing parallel workers gives measured work, not experiment elapsed time.",
        "",
        "## Experiment B: deployment without online Astra",
        "",
        "All methods have forty OOD task/reset cases and zero online Astra calls. The policy replans every five actions. The learned selector refreshes every twenty-five actions; the selected operator is applied on intervening replans.",
        "",
        "| Method | Mean policy callback ms/replan | Median seconds for 300 actions (rollouts) | Mean seconds/rollout, all 40 cases | Executed actions/wall second |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["cohort"] != "recipe":
            continue
        lines.append(
            f"| {row['label']} | {1000 * row['policy_callback_seconds_per_replan']:.0f} | "
            f"{row['full_300_action_wall_seconds_median']:.1f} ({row['full_300_action_rollouts']}) | "
            f"{row['wall_seconds_mean']:.1f} | {row['executed_actions_per_wall_second']:.2f} |"
        )
    lines += [
        "",
        "The policy callback includes preparation, editing, sampling, decoding and callback logging. Its weighted mean is total callback seconds divided by all replans, not a GPU kernel microbenchmark. Different trajectories and intervention activity prevent attributing the whole difference to selector/head overhead. The recorded schedule's shorter mean episode mainly reflects fewer executed actions after success; it does not show a faster policy. Teacher acquisition and training costs are separate.",
        "",
        "## FRS: three-case development evaluation",
        "",
        "This pilot queried Astra every ten actions. A non-deferred FRS edit adds ten inverse and ten forward vector-field evaluations, after the native ten-step prediction: thirty evaluations on an edited replan versus ten on native/deferred replans. The published pilot summaries do not isolate per-call client latency, so no FRS inference-only timing is inferred from whole-rollout time.",
        "",
        "| Case | Native seconds / actions | FRS seconds / actions | FRS calls / edits | FRS simulator success |",
        "|---|---:|---:|---:|---|",
    ]
    for row in frs_rows:
        if row["method_id"] != "astra_frs":
            continue
        native = next(
            r
            for r in frs_rows
            if r["episode_id"] == row["episode_id"]
            and r["method_id"] == "native_euler10"
        )
        lines.append(
            f"| {row['episode_id']} | {native['wall_seconds']:.1f} / {native['actions']} | "
            f"{row['wall_seconds']:.1f} / {row['actions']} | "
            f"{row['provider_calls']} / {row['frs_edits']} | {'Yes' if row['success'] else 'No'} |"
        )
    lines += [
        "",
        "All three native attempts failed. FRS succeeded earlier on two cases, so their durations do not compare equal amounts of executed motion. These three cases do not establish timing or efficacy across all twenty tasks; the larger FRS run was interrupted.",
        "",
        "## What the videos and timers mean",
        "",
        "The simulator's configured control frequency is 20 Hz. Three hundred actions correspond to fifteen seconds of simulated control; every twenty-five actions corresponds to 1.25 simulated seconds. A 9–11-second synchronous Astra call is much longer than that interval. The recorded 20-fps videos omit inference pauses and cannot demonstrate wall-clock execution speed. Even the native recipe harness averaged roughly ten executed actions per wall second including its reset, recording and simulation work; these measurements do not establish real-time 20-Hz operation.",
        "",
        "The completed speed result is that offline schedule reuse and the learned selector retain near-native rollout timing while avoiding online Astra waits. Sparse event-triggered or asynchronous Astra intervention could reduce waiting, but that scheduling change has not been evaluated here.",
        "",
        "## Sources and regeneration",
        "",
        *[f"- [{name} source report](../{path})" for name, path in SOURCES.items()],
        "- [Timer boundaries in the shared runner](../../intervention_rollout.py)",
        "- [Exact aggregates and source hashes](runtime.json)",
        "",
        "With the repository environment activated, run `python astra_reversal/reports/smart_system2_results/build_runtime.py`, then `python astra_reversal/reports/smart_system2_results/build_results.py` to refresh the publication manifest.",
        "",
    ]
    (OUTPUT / "runtime.md").write_text("\n".join(lines))
    payload["files"]["runtime.md"] = sha(OUTPUT / "runtime.md")
    (OUTPUT / "runtime.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"runtime_methods": len(rows), "frs_episodes": len(frs_rows)}))


if __name__ == "__main__":
    main()
