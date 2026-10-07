"""Offline paired guidance report, including incomplete or unstarted attempts."""

import argparse
import gzip
import hashlib
import html
import json
import shutil
import statistics
from pathlib import Path

from astra_reversal.complex_manipulation.accounting import summarize_episodes

TASKS = ("LoadPreparedFood", "PackIdenticalLunches")


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def cache_usage(records):
    events = [
        event
        for row in records
        for event in (row.get("raw_usage_events") or [row.get("raw_usage")])
    ]
    complete = all(
        isinstance(event, dict)
        and type(event.get("input_tokens")) is int
        and type(event.get("cached_input_tokens")) is int
        and 0 <= event["cached_input_tokens"] <= event["input_tokens"]
        for event in events
    ) and not any(
        row.get("job_start_unknown") or row.get("token_usage_is_lower_bound")
        for row in records
    )
    return {
        "cached_input_tokens": sum(e["cached_input_tokens"] for e in events)
        if complete
        else None,
        "uncached_input_tokens": sum(
            e["input_tokens"] - e["cached_input_tokens"] for e in events
        )
        if complete
        else None,
    }


def copy_evidence(source, destination):
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl" or path.name.startswith("request_"):
            target.with_name(target.name + ".gz").write_bytes(
                gzip.compress(path.read_bytes(), mtime=0)
            )
        else:
            shutil.copyfile(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "robocasa-language-dev3-results",
            "robocasa-language-resume-dev5-results",
            "robocasa-language-resume2-dev5-results",
        ],
    )
    args = parser.parse_args()
    budget = read(args.artifacts / "budget.json")
    sources = [args.artifacts / name for name in args.runs]
    workers = [read(s / "worker_started.json") for s in sources]
    finishes = [read(s / "worker_finished.json") for s in sources]
    allocations = [
        next(r for r in budget["entries"] if r.get("workflow") == w["workflow"])
        for w in workers
    ]
    if any("actual_gpu_hours" not in a for a in allocations):
        raise ValueError(
            "Close each GPU allocation before publishing its measured cost"
        )
    allocation = {"actual_gpu_hours": sum(a["actual_gpu_hours"] for a in allocations)}
    revisions = sorted({w["source_revision"] for w in workers})
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.native_report, args.output / "native")
    provenance = args.output / "provenance"
    provenance.mkdir()
    for name in (
        "protocol.json",
        "language_protocol.json",
        "language_preflight.json",
        "language_teacher.py",
        "language_protocol_v1.json",
        "language_protocol_v2.json",
        "reset-comparison-v3-cpu-audit.json",
        "language-seed1-xml-diff.txt",
        "language-seed1-reset-difference.json",
    ):
        shutil.copyfile(Path(__file__).parent / name, provenance / name)
    all_attempts, rejected, by_pair = [], [], {}
    for source, worker, finish in zip(sources, workers, finishes):
        evidence_root = args.output / "evidence" / worker["workflow"]
        for name in (
            "archive_receipt.json",
            "guidance_baseline_manifest.json",
            "worker_started.json",
            "worker_finished.json",
        ):
            shutil.copyfile(
                source / name, provenance / (worker["workflow"] + "_" + name)
            )
        for directory in sorted((source / "evaluation").iterdir()):
            if not directory.is_dir() or not (directory / "result.json").exists():
                continue
            result = read(directory / "result.json")
            target = evidence_root / directory.name
            copy_evidence(directory, target)
            attempt = {
                **result,
                "source_revision": worker["source_revision"],
                "evidence": target.relative_to(args.output).as_posix(),
            }
            all_attempts.append(attempt)
            by_pair.setdefault((result["task"], result["seed"]), []).append(
                (directory, attempt)
            )
            if result["executed_actions"] == 0 and not result["episode_complete"]:
                error = (
                    read(directory / "error.json")
                    if (directory / "error.json").exists()
                    else {}
                )
                rejected.append({**attempt, "error": error})
    rows, paired, table, cards, costs = [], [], [], [], []
    for task in TASKS:
        for seed in range(3):
            name = f"{task}_seed{seed}"
            attempts = by_pair.get((task, seed), [])
            executed = [
                (d, r)
                for d, r in attempts
                if r["executed_actions"] or r["episode_complete"]
            ]
            if len(executed) > 1:
                raise ValueError(
                    "This paired screen cannot select a best-of repeated policy attempt"
                )
            native = read(
                args.artifacts
                / "robocasa-native-dev3-results/evaluation"
                / name
                / "result.json"
            )
            if not executed:
                paired.append(
                    {"task": task, "seed": seed, "native": native, "guided": None}
                )
                table.append(
                    f"<tr><th>{task}</th><td>{seed}</td><td>0 / 1</td><td>Not completed / result missing</td><td colspan='6'>No complete guided result; excluded from completed-episode SR.</td></tr>"
                )
                continue
            directory, result = executed[0]
            reset = read(directory / "reset.json")
            predictions = jsonl(directory / "predictions.jsonl")
            commands = jsonl(directory / "executed_steps.jsonl")
            proposals = [
                read(p)
                for p in sorted((directory / "guidance").glob("proposal_*.json"))
            ]
            provider = jsonl(directory / "guidance/provider.jsonl")
            if (
                len(commands) != result["executed_actions"]
                or len(predictions) != result["generated_chunks"]
            ):
                raise ValueError("Trace counts and reported execution differ")
            if len(provider) != result["teacher_calls"]:
                raise ValueError("Teacher call count differs from its provider journal")
            if (
                result["executed_actions"]
                and not read(directory / "guidance/paired_reset.json")["matched"]
            ):
                raise ValueError("Executed guidance has no matched reset receipt")
            relative = result["evidence"]
            target = args.output / relative
            native_relative = f"native/robocasa/evidence/{name}"
            row = {
                **result,
                "proposals": proposals,
                "evidence": relative,
                "median_model_seconds": statistics.median(
                    p["policy_seconds"] for p in predictions
                )
                if predictions
                else None,
                "baseline_episode_id": native["episode_id"],
                "first_policy_call_seconds": predictions[0]["policy_seconds"]
                if predictions
                else None,
                **cache_usage(provider),
            }
            rows.append(row)
            paired.append({"task": task, "seed": seed, "native": native, "guided": row})
            status = (
                "Success"
                if result["success"]
                else "Failed at horizon"
                if result["episode_complete"]
                else "Incomplete"
            )
            tokens = result.get("teacher_tokens")
            token_label = f"{tokens:,}" if tokens is not None else "Unknown / partial"

            def token_count(value):
                return f"{value:,}" if value is not None else "Unknown"

            measured_usage = result["teacher_usage"]["tokens"]
            input_count = (
                measured_usage["input_tokens"]["sum"]
                if measured_usage["input_tokens"]["complete"]
                else None
            )
            output_count = (
                measured_usage["output_tokens"]["sum"]
                if measured_usage["output_tokens"]["complete"]
                else None
            )
            model_ms = (
                f"{row['median_model_seconds'] * 1000:.1f}"
                if row["median_model_seconds"] is not None
                else "—"
            )
            costs.append(
                f"<tr><th>{task}</th><td>{seed}</td><td>{token_count(input_count)}</td><td>{token_count(row['cached_input_tokens'])}</td><td>{token_count(row['uncached_input_tokens'])}</td><td>{token_count(output_count)}</td><td>{native['wall_seconds']:.1f} s</td><td>{result['wall_seconds']:.1f} s</td><td>{result.get('teacher_seconds', 0):.1f} s</td><td>{model_ms} ms</td></tr>"
            )
            table.append(
                f"<tr><th>{task}</th><td>{seed}</td><td>0 / 1</td><td>{status}</td><td>{len(attempts)}</td><td>{result['reset_free_segments']:,}</td><td>{result['assisted_segments']:,}</td><td>{result['teacher_calls']}</td><td>{token_label}</td><td>{result['wall_seconds']:.1f} s</td></tr>"
            )
            decisions = "".join(
                f"<tr><td>{p['observation_step']}</td><td>{html.escape(p['method'])}</td><td>{html.escape(p['subgoal'])}</td><td>{html.escape(p['observed_evidence'])}</td><td>{p['next_review_controls']}</td></tr>"
                for p in proposals
            )
            videos = f"<div class='pair'><figure><figcaption>Native · 0 / 1</figcaption><video controls preload='metadata' poster='{native_relative}/starting_image.png' src='{native_relative}/rollout.mp4'></video></figure>"
            if (target / "rollout.mp4").exists():
                videos += f"<figure><figcaption>Astra phase prompts · {status}</figcaption><video controls preload='metadata' poster='{relative}/starting_image.png' src='{relative}/rollout.mp4'></video></figure>"
            videos += "</div>"
            cards.append(
                f"<article class='card'><div class='eyebrow'>Seed {seed} · {status}</div><h3>{task}</h3><p>{html.escape(reset['instruction'])}</p>{videos}<p class='small'>Guided attempt: {result['teacher_calls']} calls · {token_label} tokens · {result.get('teacher_seconds', 0):.1f} s waiting for Astra · {result['policy_seconds']:.1f} s model inference · {result['executed_actions']:,} controls. The simulator pauses during teacher calls; video playback follows simulation time.</p><details><summary>Every accepted intervention and its visual evidence</summary><div class='table-wrap'><table><thead><tr><th>Control step</th><th>Choice</th><th>Subgoal appended to original instruction</th><th>Reported observation</th><th>Review after controls</th></tr></thead><tbody>{decisions}</tbody></table></div></details><p class='small'><a href='{relative}/result.json'>Result and token accounting</a> · <a href='{relative}/guidance/provider.jsonl.gz'>Provider receipts</a> · <a href='{relative}/predictions.jsonl.gz'>Every model prompt and action chunk</a> · <a href='{relative}/executed_steps.jsonl.gz'>Applied controls</a></p></article>"
            )
    summary = summarize_episodes(all_attempts)
    complete_batch = len(rows) == 6 and summary["completed_episodes"] == 6
    summary["completed_pair_sr"] = (
        summary["completed_episode_sr"] if complete_batch else None
    )
    summary["planned_episodes"] = 6
    summary["missing_episode_results"] = 6 - len(rows)
    tokens_known = all(r.get("teacher_tokens") is not None for r in all_attempts)
    total_tokens = sum(r.get("teacher_tokens") or 0 for r in all_attempts)
    tasks = []
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task]
        completed = [r for r in subset if r["episode_complete"]]
        tasks.append(
            {
                "task": task,
                "native_successes": 0,
                "native_episodes": 3,
                "guided_successes": sum(r["success"] for r in completed),
                "guided_completed": len(completed),
                "guided_planned": 3,
                "full_guided_sr": sum(r["success"] for r in completed) / 3
                if len(completed) == 3
                else None,
                "teacher_tokens_known_sum": sum(
                    r.get("teacher_tokens") or 0 for r in subset
                ),
            }
        )
    measured = {
        "paired_comparison_complete": complete_batch,
        "source_revisions": revisions,
        "allocations": allocations,
        "worker_exit_codes": [f["returncode"] for f in finishes],
        "all_guided_reset_attempts": all_attempts,
        "precontrol_rejected_resets": rejected,
        "summary": summary,
        "tasks": tasks,
        "paired_episodes": paired,
        "batch_gpu_hours": allocation["actual_gpu_hours"],
        "guided_teacher_tokens": total_tokens if tokens_known else None,
        "teacher_preflight_tokens": read(provenance / "language_preflight.json")[
            "total_tokens_including_rejected_probe"
        ],
        "policy_updates": 0,
        "is_ood": False,
        "known_resets": {
            "guided_policy_reset_attempts": summary["reset_episodes_started"],
            "paired_native_policy_reset_attempts": 6,
            "guided_constructor_setup_resets": len(all_attempts),
            "native_constructor_setup_resets": 6,
            "combined_policy_and_constructor_resets": 12
            + summary["reset_episodes_started"]
            + len(all_attempts),
            "scope": "This paired screen only. One documented wrapper setup reset per constructed environment. Earlier incomplete development probes and Bench2Dex resets are separate.",
        },
        "source_parity": "Pinned weights and native inference transforms checked. Independent same-noise output parity against the full upstream training-dependent factory has not been measured.",
        "interpretation": "One guided attempt after one native failure per reset. Development-scene pilot; no learning update. Cost of native preparation and CPU teacher validation is separate, not free.",
    }
    closed_hours = budget["previous_study_gpu_hours"] + sum(
        entry.get("actual_gpu_hours") or 0 for entry in budget["entries"]
    )
    compute_snapshot = {
        "authorized_gpu_hours": budget["authorized_total_gpu_hours"],
        "closed_gpu_hours_including_previous_study": closed_hours,
        "remaining_after_closed_allocations": budget["authorized_total_gpu_hours"]
        - closed_hours,
        "all_allocations_in_this_ledger_closed": all(
            entry.get("actual_gpu_hours") is not None for entry in budget["entries"]
        ),
    }
    measured["study_compute_snapshot"] = compute_snapshot
    capacity_html = ""
    if args.prompt_audit:
        audit = read(args.prompt_audit)
        if audit["kind"] != "posthoc_language_capacity_audit":
            raise ValueError("Unexpected prompt capacity audit")
        audit_runs = {row["run"] for row in audit["logs"]}
        if not {p.name for p in sources}.issubset(audit_runs):
            raise ValueError("Capacity audit does not cover every guided allocation")
        shutil.copyfile(args.prompt_audit, provenance / "prompt_capacity_audit.json")
        shutil.copyfile(
            Path(__file__).with_name("audit_language_prompts.py"),
            provenance / "audit_language_prompts.py",
        )
        guided_warnings = sum(
            row["warning_count"]
            for row in audit["logs"]
            if row["run"] in {p.name for p in sources}
        )
        lost_language = sum(
            row["instruction_prefix_retained"] is False for row in audit["reviews"]
        )
        lost_state = sum(
            row["all_state_values_retained"] is False for row in audit["reviews"]
        )
        unknown_prefix = sum(
            row["instruction_prefix_retained"] is None
            or row["all_state_values_retained"] is None
            for row in audit["reviews"]
        )
        measured["prompt_capacity"] = {
            "guided_policy_calls_with_truncation_warning": guided_warnings,
            "review_steps_audited": audit["review_count"],
            "review_steps_over_capacity": audit["reviews_over_capacity"],
            "review_steps_losing_instruction_prefix": lost_language,
            "review_steps_losing_state_values": lost_state,
            "review_steps_with_unknown_prefix_alignment": unknown_prefix,
            "evidence": "provenance/prompt_capacity_audit.json",
        }
        capacity_html = (
            "<h2>What reached the policy's text input</h2>"
            f"<p>The released policy shares a 200-token budget between task text, the appended phase and discretized robot state. Its logs record {guided_warnings:,} guided model calls with truncation warnings. A post-hoc CPU audit reconstructed all {audit['review_count']} executed teacher-review inputs using the same tokenizer, normalization statistics and recorded robot state: {audit['reviews_over_capacity']} exceeded 200 tokens, {lost_language} lost part of the instruction prefix, and {lost_state} lost state values ({unknown_prefix} prefix comparisons unknown).</p>"
            "<p>These are System1 text tokens, distinct from Astra's reported usage counters above. Robot state was not saved at every intervening model call, so decoded truncation tails are available at review steps only. The audit preserves the original results and does not establish truncation as the cause of failure. <a href='provenance/prompt_capacity_audit.json'>Exact token lengths, decoded inputs, tails and source hashes</a>.</p>"
        )
    (args.output / "results.json").write_text(json.dumps(measured, indent=2) + "\n")
    chart_rows = []
    scale = max(1, max(t["teacher_tokens_known_sum"] for t in tasks))
    for index, task in enumerate(tasks):
        y = 55 + index * 90
        chart_rows.append(
            f"<text x='20' y='{y}'>{task['task']}</text><rect x='250' y='{y - 18}' width='{390 * task['teacher_tokens_known_sum'] / scale:.1f}' height='24' rx='4' fill='#17685d'/><text x='20' y='{y + 30}'>{task['teacher_tokens_known_sum']:,} known tokens · native 0/3 → guided {task['guided_successes']}/{task['guided_completed']} complete ({task['guided_planned']} planned)</text>"
        )
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 820 250' role='img'><rect width='820' height='250' fill='#fff'/><g font-family='system-ui,sans-serif' font-size='13' fill='#182932'>"
        + "".join(chart_rows)
        + "<text x='20' y='233'>Sum across guided attempts; CPU preflight and native preparation reported separately.</text></g></svg>"
    )
    (args.output / "tokens-and-success.svg").write_text(svg)
    mobile_rows = []
    for index, task in enumerate(tasks):
        y = 28 + index * 130
        mobile_rows.append(
            f"<text x='20' y='{y}'>{task['task']}</text><rect x='20' y='{y + 14}' width='{320 * task['teacher_tokens_known_sum'] / scale:.1f}' height='21' rx='4' fill='#17685d'/><text x='20' y='{y + 58}'>{task['teacher_tokens_known_sum']:,} teacher tokens</text><text x='20' y='{y + 81}'>Native 0/3 → guided {task['guided_successes']}/{task['guided_completed']} complete</text>"
        )
    (args.output / "tokens-and-success-mobile.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='360' height='310' viewBox='0 0 360 310' role='img'><rect width='360' height='310' fill='#fff'/><g font-family='system-ui,sans-serif' font-size='14' fill='#182932'>"
        + "".join(mobile_rows)
        + "<text x='20' y='288' font-size='11'>Native + preflight costs shown separately.</text></g></svg>"
    )
    native_html = (args.native_report / "robocasa/index.html").read_text()
    style = native_html.split("<style>", 1)[1].split("</style>", 1)[0]
    style += ".flow .node{position:relative}.flow .node:not(:last-child)::after{content:'→';position:absolute;right:-12px;top:44%;font-size:20px;color:#17685d}@media(max-width:900px){.flow .node::after{display:none}}figure{margin:0}figcaption{font-size:12px;color:var(--muted);margin-bottom:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.flow{grid-template-columns:repeat(5,1fr)}.card{margin-bottom:18px}.flow .node{padding:16px}.flow .node b{font-size:14px}.chart{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}.pair video{aspect-ratio:3/1}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:900px){.flow{grid-template-columns:1fr 1fr}}@media(max-width:650px){.pair{grid-template-columns:1fr}}"
    cards_top = "".join(
        f"<div class='stat'><span>{t['task']}</span><strong>0/3 → {t['guided_successes']}/{t['guided_completed']}</strong><small>Native → guided completed episodes · 3 planned</small></div>"
        for t in tasks
    )
    known_label = f"{total_tokens:,}" if tokens_known else f"≥ {total_tokens:,}"
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Astra phase guidance · Paired RoboCasa pilot</title><style>{style}</style></head><body><main>
    <div class='eyebrow'>Astra × System1 · Matched development pilot</div><h1>Can a better subgoal<br>change the rollout?</h1><p class='lead'>The same released π0.5 checkpoint and starting scenes, with Astra choosing short language instructions from observed execution and one previous failed attempt.</p>
    <div class='notice'><b>{"All six paired guided policy episodes completed." if complete_batch else "The guided batch is incomplete."}</b> These are seen tasks in development scenes, not held-out OOD results. Guidance uses a prior native failure from the same reset. No weights were updated. This experiment measures appended language prompts; it does not measure TEI/TLI, vision interpolation or flow reversal.</div>
    <div class='stats'>{cards_top}<div class='stat'><span>Guided teacher tokens</span><strong>{known_label}</strong><small>Includes rejected / failed jobs when reported</small></div><div class='stat'><span>Guided GPU allocation</span><strong>{allocation["actual_gpu_hours"]:.3f} h</strong><small>L40S · includes restoration and teardown</small></div></div>
    <h2>How this intervention works</h2><div class='flow'><div class='node'><small>01 · PAIRED RESET</small><b>Match the native failure</b><span>Match task, physical state and model settings; retain raw XML hashes and record equivalent format declarations or bounded camera rounding.</span></div><div class='node'><small>02 · OBSERVE</small><b>Current + past images</b><span>Three live policy cameras, 16D state, previous live review and four chronological frames from the native failure.</span></div><div class='node'><small>03 · ASTRA</small><b>Choose the phase</b><span>GPT-6 Astra, medium effort, Codex harness. Select native or a short appended subgoal and review interval.</span></div><div class='node'><small>04 · SYSTEM1</small><b>Generate motor actions</b><span>Original instruction + phase → frozen π0.5 → 50 actions. Execute five, observe and replan.</span></div><div class='node'><small>05 · REVIEW</small><b>Compare observed progress</b><span>Astra reviews after 50/100/250/500 controls. At most 16 calls; later execution reverts to the original prompt.</span></div></div>
    <p>The environment pauses during teacher inference. An unavailable or rejected teacher call stops the episode and is recorded as incomplete. The original environment success predicate determines the outcome; Astra's visual assessment does not decide success. One reset is one trial. Every executed prefix of up to five controls is one reset-free segment.</p>
    <h2>Results and token spending</h2><div class='chart'><picture><source media='(max-width:650px)' srcset='tokens-and-success-mobile.svg'><img src='tokens-and-success.svg' alt='Teacher tokens and observed success per task'></picture></div><div class='table-wrap'><table><thead><tr><th>Task</th><th>Seed</th><th>Native success</th><th>Guided outcome</th><th>New resets</th><th>Segments</th><th>Assisted segments</th><th>Calls</th><th>Tokens</th><th>Rollout wall</th></tr></thead><tbody>{"".join(table)}</tbody></table></div>
    <p class='small'>The six paired native failures required six policy resets and 16,200 controls; they are retained in the native report. Guidance adds {summary["reset_episodes_started"]} reset attempts: {summary["completed_episodes"]} completed policy episodes and {len(rejected)} rejected before controls. Both stopped allocations remain in the evidence; only unfinished pairs were resumed. No completed guided failure was rerolled. Each upstream environment construction also performs a setup reset: {len(all_attempts)} guided and six native. This paired screen therefore accounts for {measured["known_resets"]["combined_policy_and_constructor_resets"]} policy-reset attempts plus constructor setup resets in total; earlier probes and dexterous trials are separate. CPU teacher validation consumed {measured["teacher_preflight_tokens"]:,} tokens across two calls, including the first rejected probe. Input tokens include image and Codex harness overhead and any cached input. Reasoning tokens are included in output. Subscription-backed monetary cost is unverified, not zero. Raw usage receipts are downloadable for every episode.</p>
    <h2>Tokens and rollout speed</h2><div class='table-wrap'><table><thead><tr><th>Task</th><th>Seed</th><th>Input tokens</th><th>Cached input</th><th>Uncached input</th><th>Output tokens</th><th>Native wall</th><th>Guided wall</th><th>Astra wait</th><th>Median guided model call</th></tr></thead><tbody>{"".join(costs)}</tbody></table></div><p class='small'>Cached input is included in the input total; reasoning is included in output. Model calls and teacher calls are different units. Wall time includes compilation and environment stepping after reset recording. The continuation starts a new policy process, so its first model call can include compilation while the corresponding native seed was already warm; first-call times remain in results.json. GPU allocation additionally includes restoration, reset preparation and teardown.</p>
    <h2>Reset validation and continuation</h2><p>The first allocation stopped when seed 1 had one wrist-camera color value differ by 1/255. Scene XML, simulator state, robot proprioception, task and both other camera tensors were identical. Protocol v2 permits at most 16 changed color values per camera, each differing by at most one level, while retaining exact physical-state checks. It records differences and leaves the actual model input unchanged. A subsequent initialization also stopped because an OBJ mesh export omitted the redundant content_type declaration. Protocol v3 compares the equivalent OBJ declarations while preserving raw hashes and every other XML setting; neither the simulator XML nor observations are edited. Both archived resets pass the recorded CPU audit. The completed seed-0 policy episode was retained; continuations cover only the other five pairs. All additional pre-control resets are included above and in <a href='results.json'>the complete attempt ledger</a>. This changes reset validation, not task success or the intervention.</p><h2>Watch each paired attempt</h2>{"".join(cards)}
    {capacity_html}<h2>Exact teacher prompt</h2><details><summary>Read the system prompt used in this pilot</summary><pre>{html.escape(__import__("astra_reversal.complex_manipulation.language_teacher", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT)}</pre></details><p>Each request also includes the original task, active subgoal, two preceding decisions, remaining calls and timestamped image attachments. Full request JSON is gzip-compressed beside each episode's evidence; no private Codex reasoning events are included.</p>
    <h2>What this result can establish</h2><p>This is a small paired guidance screen on two mobile-manipulator tasks. It tests whether language phase selection changes observed execution. It does not establish generalization, a learned policy, latent steerability or the requested four-condition learning comparison at 1/2/4/8/16 collection episodes.</p><p>The RoboCasa runner loads the released checkpoint and native inference transforms without the training dataset stack. Source, weights and transforms were checked; an independent same-noise output comparison against the full upstream factory has not been measured. The <a href='native/index.html'>complete native report</a> also includes both dexterous tasks, their stage outcomes, videos and hardware-interface findings.</p>
    <h2>Compute snapshot</h2><p>Closed allocations across this and earlier studies total {closed_hours:.3f} of {compute_snapshot["authorized_gpu_hours"]} authorized L40S GPU-hours. {compute_snapshot["remaining_after_closed_allocations"]:.3f} hours remain after closed allocations. {"All allocations in this study ledger are closed." if compute_snapshot["all_allocations_in_this_ledger_closed"] else "Other reservations are still active and reduce available headroom."} The included native report is the earlier baseline snapshot; this page contains the current guidance results and accounting.</p>
    <footer><a href='results.json'>Complete measured results</a> · <a href='provenance/language_protocol.json'>Registered protocol</a> · <a href='provenance/protocol.json'>Full study status</a> · <a href='provenance/language_teacher.py'>Teacher implementation</a> · <a href='provenance/language_preflight.json'>CPU probe receipts</a> · <a href='https://github.com/GaTech-RL2/EgoVerse/pull/706'>Code and linear PR chain</a><br>Offline: all plots, videos, images and public evidence are included. Source revisions {html.escape(", ".join(revisions))}. Guided GPU cost {allocation["actual_gpu_hours"]:.6f} hours. Native preparation is separately recorded.</footer></main></body></html>"""
    (args.output / "index.html").write_text(document)
    manifest = [
        {
            "file": p.relative_to(args.output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    ]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete_batch": complete_batch,
                "tasks": tasks,
                "teacher_tokens": known_label,
                "files": len(manifest),
                "portable_bytes": sum(p["bytes"] for p in manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
