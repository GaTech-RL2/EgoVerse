"""Build the first-completed-worker evaluation illustrations from audited files.

No provider, model, GPU, or simulator calls are made. Existing development
artifacts are only read; the request-image verifier is reused without edits.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astra_reversal.interpolation_agent import parse_proposal, summarize_calls
from astra_reversal.interpolation_audit import RENDERER, render_linux
from astra_reversal.records import digest, file_sha256
from astra_reversal.reports.phase_interpolation.development.behavior.build_milk_figure import (
    CAMERAS,
    frame,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
MODEL = "azure/openai/gpt-6-astra"
MODES = ("astra_tei", "astra_tli", "astra_tli_vision")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def read_case(directory, case_name, *, worker=2):
    audits = HERE.parent / "audits"
    receipt_path = audits / f"worker_{worker}_receipt.json"
    receipt = read_json(receipt_path)
    require(
        receipt["status"] == "passed" and receipt["phase"] == "evaluation",
        "Passed evaluation receipt required",
    )
    arrays_path = audits / receipt["array_audit"]["file"]
    require(
        file_sha256(arrays_path) == receipt["array_audit"]["sha256"],
        "Array receipt hash",
    )
    arrays = read_json(arrays_path)
    require(arrays["status"] == "passed" and arrays["complete"], "Complete array audit")
    metadata = next(row for row in receipt["feedback"] if row["case"] == case_name)
    feedback_path = audits / metadata["file"]
    require(file_sha256(feedback_path) == metadata["sha256"], "Feedback receipt hash")
    data = gzip.decompress(feedback_path.read_bytes())
    require(
        hashlib.sha256(data).hexdigest() == metadata["uncompressed_sha256"],
        "Uncompressed feedback hash",
    )
    feedback = json.loads(data)
    require(feedback["status"] == "passed", "Passed feedback audit")
    case = directory / case_name
    source_files = {
        name: sha
        for name, sha in feedback["input_file_sha256"].items()
        if not name.startswith("arrays/")
    }
    for name, sha in source_files.items():
        require(file_sha256(case / name) == sha, f"Audited file differs: {name}")
    summary = read_json(case / "summary.json")
    require(
        summary["status"] == "complete" and not summary["development"],
        "Complete evaluation case",
    )
    require(
        summary["seed"] == 29 and not summary["baseline"]["success"],
        "Evaluation reset and failed baseline",
    )
    require(summary["episode_id"] == feedback["episode_id"], "Audited episode")
    events = [
        json.loads(line) for line in (case / "events.jsonl").read_text().splitlines()
    ]
    requests = {
        row["request"]["request_fingerprint"]: row
        for row in events
        if row["kind"] == "interpolation_request"
    }
    generations = {
        (row["attempt_id"], row["observation_step"]): row
        for row in events
        if row["kind"] == "phase_generation"
    }
    bindings = {
        row["request_fingerprint"]: row for row in feedback["provider_bindings"]
    }
    calls, records, arms = [], [], {}
    for mode in MODES:
        arm = summary["arms"][mode]
        outcomes = []
        for attempt in arm["attempts"][1:]:
            attempt_id = attempt["attempt_id"]
            ledger = [
                json.loads(line)
                for line in (case / f"{attempt_id}_provider.jsonl")
                .read_text()
                .splitlines()
            ]
            require(len(ledger) == len(attempt["decisions"]), "Client slot coverage")
            records.extend(ledger)
            for decision, row in zip(attempt["decisions"], ledger, strict=True):
                fingerprint = row["request_fingerprint"]
                request = requests[fingerprint]["request"]
                binding = bindings[fingerprint]
                require(
                    row["requested_model"] == row["response"]["model"] == MODEL,
                    "Exact provider model",
                )
                require(
                    row["sampling_settings"]
                    == {"max_completion_tokens": 8192, "reasoning_effort": "medium"},
                    "Sampling settings",
                )
                require(row["cache"] == {"no-cache": True}, "No-cache setting")
                require(
                    row["accepted"] == decision["accepted"] == binding["accepted"],
                    "Audited acceptance",
                )
                content = row["response"]["choices"][0]["message"]["content"]
                proposal = json.loads(content)
                if row["accepted"]:
                    require(
                        parse_proposal(content, request) == decision["proposal"],
                        "Accepted response binding",
                    )
                generation = generations[(attempt_id, request["observation_step"])]
                calls.append(
                    {
                        "mode": mode,
                        "attempt_id": attempt_id,
                        "revision": attempt["iteration"] - 1,
                        "decision_index": request["decision_index"],
                        "step": request["observation_step"],
                        "request_fingerprint": fingerprint,
                        "request_event_sequence": requests[fingerprint]["sequence"],
                        "accepted": row["accepted"],
                        "error": decision["error"],
                        "http_status": row["http_status"],
                        "provider_call": row["provider_call"],
                        "response_text_sha256": hashlib.sha256(
                            content.encode()
                        ).hexdigest(),
                        "proposal_fields": {
                            key: proposal.get(key)
                            for key in (
                                "source_a_id",
                                "source_b_id",
                                "alpha",
                                "observed_phase",
                                "rationale",
                                "vision",
                            )
                        },
                        "proposal_fields_applied": row["accepted"],
                        "applied_accepted_decision_id": generation[
                            "applied_accepted_decision_id"
                        ],
                        "native_condition_fallback_at_call": generation[
                            "native_condition_fallback"
                        ],
                        "active_text_before_call": request["active_interpolation"],
                        "completed_rollout_feedback": request[
                            "completed_rollout_feedback"
                        ],
                        "previous_attempt_feedback": request["previous_attempt"][
                            "feedback"
                        ],
                        "previous_attempt_snapshot_steps": [
                            r["step"] for r in request["previous_attempt"]["snapshots"]
                        ],
                        "previous_attempt_decision_count": len(
                            request["previous_attempt"]["decisions"]
                        ),
                        "same_attempt_history_count": len(
                            request["previous_decisions"]
                        ),
                        "token_usage": binding["token_usage"],
                    }
                )
            outcomes.append(
                {
                    key: attempt[key]
                    for key in (
                        "attempt_id",
                        "iteration",
                        "success",
                        "actions_executed",
                        "accepted_decisions_executed",
                        "actions_with_accepted_decision",
                        "actions_with_nonzero_text",
                        "actions_with_changed_vision",
                        "actions_with_held_text_after_failed_call",
                        "native_condition_fallback_actions",
                        "applied_accepted_decision_ids",
                    )
                }
            )
        arms[mode] = {"summary": arm["summary"], "intervention_attempts": outcomes}
    costs = summarize_calls(records)
    require(
        costs == feedback["provider"] == summary["physical_cost"]["token_usage"],
        "Complete token reconciliation",
    )
    require(
        len(calls) == feedback["counts"]["client_attempts"], "Complete call coverage"
    )
    result = {
        "status": "verified",
        "episode_id": summary["episode_id"],
        "reset_entry_sha256": summary["reset_entry_sha256"],
        "target_task": next(iter(requests.values()))["request"]["target_task"],
        "protocol_sha256": summary["protocol_sha256"],
        "source_catalog": summary["source_catalog"],
        "baseline": {
            "success": False,
            "actions": summary["baseline"]["actions_executed"],
        },
        "arm_outcomes": {
            name: {
                "summary": arm["summary"],
                "rollouts": [
                    {
                        "attempt_id": a["attempt_id"],
                        "success": a["success"],
                        "actions": a["actions_executed"],
                    }
                    for a in arm["attempts"]
                ],
            }
            for name, arm in summary["arms"].items()
        },
        "astra_arms": arms,
        "physical_cost": summary["physical_cost"],
        "provider": costs,
        "calls": calls,
        "rejection_reasons": dict(
            Counter(r["error"] for r in calls if not r["accepted"])
        ),
        "sources": {
            "workflow": receipt["workflow"],
            "archive_sha256": receipt["archive"]["sha256"],
            "payload_sha256": arrays["payload_sha256"],
            "case_file_sha256": source_files,
            "audit_file_sha256": {
                "../audits/" + p.name: file_sha256(p)
                for p in (receipt_path, arrays_path, feedback_path)
            },
        },
    }
    return {
        "directory": case,
        "summary": summary,
        "feedback": feedback,
        "requests": requests,
        "generations": generations,
        "safe": result,
    }


def application_segments(case, attempt_id):
    attempt = next(
        a
        for arm in case["summary"]["arms"].values()
        for a in arm["attempts"]
        if a["attempt_id"] == attempt_id
    )
    rows = sorted(
        (g for (name, _), g in case["generations"].items() if name == attempt_id),
        key=lambda g: g["observation_step"],
    )
    segments = []
    for index, row in enumerate(rows):
        end = (
            rows[index + 1]["observation_step"]
            if index + 1 < len(rows)
            else attempt["actions_executed"]
        )
        identity = {
            "applied_decision_id": row["applied_accepted_decision_id"],
            "native_condition_fallback": row["native_condition_fallback"],
            "text_has_effect": row["text_has_effect"],
            "vision_has_effect": row["vision_has_effect"],
        }
        if segments and all(
            segments[-1][key] == value for key, value in identity.items()
        ):
            segments[-1]["end_action_exclusive"] = end
            segments[-1]["generation_event_sequences"].append(row["sequence"])
        else:
            segments.append(
                {
                    "start_action": row["observation_step"],
                    "end_action_exclusive": end,
                    **identity,
                    "generation_event_sequences": [row["sequence"]],
                }
            )
    require(
        sum(
            s["end_action_exclusive"] - s["start_action"]
            for s in segments
            if s["native_condition_fallback"]
        )
        == attempt["native_condition_fallback_actions"],
        "Fallback action sum",
    )
    require(
        sum(
            s["end_action_exclusive"] - s["start_action"]
            for s in segments
            if s["applied_decision_id"] is not None
        )
        == attempt["actions_with_accepted_decision"],
        "Accepted-action sum",
    )
    return segments


def plot_case(case, selection, *, wine):
    by_step = {
        (e["request"]["attempt_id"], e["request"]["observation_step"]): e
        for e in case["requests"].values()
    }
    images, metadata = [], []
    for attempt, call_step, scope, frame_step, title in selection:
        event = by_step[(attempt, call_step)]
        raw, record = frame(
            case["directory"], event, scope, frame_step, case["feedback"]
        )
        modified = None
        application = None
        if scope == "current_attempt":
            generation = case["generations"][(attempt, call_step)]
            modified = render_linux(raw, generation["vision"])
            require(
                digest(modified) == generation["modified_observation_sha256"],
                "Exact applied Linux rendering",
            )
            application = {
                key: generation[key]
                for key in (
                    "vision",
                    "active_interpolation",
                    "applied_accepted_decision_id",
                    "native_condition_fallback",
                    "modified_observation_sha256",
                )
            }
        images.append((raw, modified, title))
        metadata.append(
            {**record, "figure_label": title, "actual_application": application}
        )
    fig, axes = plt.subplots(3, len(images), figsize=(10, 10) if wine else (13, 10))
    for column, (raw, modified, title) in enumerate(images):
        for row, camera in enumerate(CAMERAS):
            axes[row, column].imshow(raw[camera], interpolation="nearest")
        if modified is None:
            axes[2, column].text(
                0.5,
                0.5,
                "Prior rollout\nraw feedback only",
                transform=axes[2, column].transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
            )
        else:
            axes[2, column].imshow(modified[CAMERAS[0]], interpolation="nearest")
        for row in range(3):
            axes[row, column].set_axis_off()
            if column == 0:
                axes[row, column].text(
                    -0.06,
                    0.5,
                    (
                        "Raw external\nAstra feedback",
                        "Raw wrist\nAstra feedback",
                        "Policy external\nActual applied marks",
                    )[row],
                    transform=axes[row, column].transAxes,
                    ha="right",
                    va="center",
                    fontsize=10,
                )
        axes[0, column].set_title(title, fontsize=11, pad=10)
    title = (
        "Wine rescue includes 25 initial native-fallback actions"
        if wine
        else "Goal 2: capped failure despite revised conditioning and marks"
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(
        0.57,
        0.935,
        f'Exact task: “{case["safe"]["target_task"]}”',
        ha="center",
        fontsize=11,
    )
    footer = (
        "Success at 75 actions = 25 native fallback + 50 with accepted Astra text; 10 had changed images."
        if wine
        else "TLI + vision failed both 300-action revisions; no displayed current frame establishes success."
    )
    fig.text(0.57, 0.057, footer, ha="center", fontsize=9)
    fig.text(
        0.57,
        0.025,
        "First completed evaluation worker · seed29 · illustrative selection, not representative performance",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.18 if wine else 0.155,
        right=0.99,
        top=0.84,
        bottom=0.10,
        wspace=0.04,
        hspace=0.04,
    )
    target = HERE / (
        "wine_rescue_with_native_fallback.png" if wine else "goal2_capped_failure.png"
    )
    fig.savefig(
        target,
        dpi=150,
        facecolor="white",
        pil_kwargs={"optimize": True},
        metadata={"Software": "Matplotlib; verified recorded evaluation frames"},
    )
    plt.close(fig)
    return {
        "file": target.name,
        "sha256": file_sha256(target),
        "frames": metadata,
        "renderer": RENDERER,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--orange-results-root", type=Path)
    args = parser.parse_args()
    goal2 = read_case(args.results_root, "case_2_0")
    wine = read_case(args.results_root, "case_6_0")
    require(
        goal2["safe"]["episode_id"] == "libero_goal_ood:seed29:task2:state0",
        "Goal2 identity",
    )
    require(
        wine["safe"]["episode_id"] == "libero_goal_ood:seed29:task6:state0",
        "Wine identity",
    )
    require(
        not any(a["summary"]["success"] for a in goal2["summary"]["arms"].values()),
        "Capped failure",
    )
    require(
        all(a["summary"]["success"] for a in wine["summary"]["arms"].values()),
        "Wine random, oracle and Astra rescues",
    )
    segments = application_segments(wine, "astra_tli_vision_2")
    first = next(
        c
        for c in wine["safe"]["calls"]
        if c["attempt_id"] == "astra_tli_vision_2" and c["decision_index"] == 1
    )
    require(
        not first["accepted"]
        and segments[0]["native_condition_fallback"]
        and segments[0]["start_action"] == 0
        and segments[0]["end_action_exclusive"] == 25,
        "Initial rejection and 25 native fallback actions",
    )
    wine["safe"]["vision_rescue_application_segments"] = segments
    failure_figure = plot_case(
        goal2,
        [
            (
                "astra_tli_vision_3",
                0,
                "previous_attempt",
                300,
                "Previous revision\nStep 300: failed\nRaw feedback supplied to revision 2",
            ),
            (
                "astra_tli_vision_3",
                0,
                "current_attempt",
                0,
                "Revision 2 · decision 1\nStep 0 · A38/B10 · α0.35\nFresh target box",
            ),
            (
                "astra_tli_vision_3",
                50,
                "current_attempt",
                50,
                "Revision 2 · decision 3\nStep 50 · A10/B13 · α0.35\nWrong-object grasp assessment",
            ),
            (
                "astra_tli_vision_3",
                275,
                "current_attempt",
                275,
                "Revision 2 · decision 12\nStep 275 · A14/B13 · α0.25\nTarget acquisition unresolved",
            ),
        ],
        wine=False,
    )
    wine_figure = plot_case(
        wine,
        [
            (
                "astra_tli_vision_2",
                0,
                "current_attempt",
                0,
                "Decision 1 · step 0\nREJECTED: malformed fingerprint\nNext 25 actions: native conditioning",
            ),
            (
                "astra_tli_vision_2",
                25,
                "current_attempt",
                25,
                "Decision 2 · step 25\nACCEPTED · A14/B13 · α0.20\nFresh bowl marker",
            ),
            (
                "astra_tli_vision_2",
                50,
                "current_attempt",
                50,
                "Decision 3 · step 50\nACCEPTED · A14/B13 · α0.75\nFresh bowl marker",
            ),
        ],
        wine=True,
    )
    development_path = (
        PROJECT / "reports/phase_interpolation/development/audits/worker_0_arrays.json"
    )
    development = read_json(development_path)
    previous_case = next(
        c
        for c in development["cases"]
        if c["episode_id"] == "libero_goal_ood:seed19:task6:state0"
    )
    require(
        previous_case["reset_entry_sha256"] != wine["safe"]["reset_entry_sha256"],
        "Distinct development/evaluation reset",
    )
    result = {
        "schema_version": "phase-interpolation-first-evaluation-worker-examples-1",
        "status": "verified",
        "selection": "Both cases from worker2, the first completed evaluation worker; illustrative and not representative selection.",
        "cases": {"goal2": goal2["safe"], "wine": wine["safe"]},
        "figures": [failure_figure, wine_figure],
        "development_wine_reset_comparison": {
            "episode_id": previous_case["episode_id"],
            "reset_entry_sha256": previous_case["reset_entry_sha256"],
            "array_audit_sha256": file_sha256(development_path),
        },
        "source_file_sha256": {
            str(p.relative_to(PROJECT)): file_sha256(p)
            for p in (
                Path(__file__).resolve(),
                PROJECT / "interpolation_agent.py",
                PROJECT / "intervention_agent.py",
                PROJECT / "interpolation_audit.py",
                PROJECT / "records.py",
                PROJECT
                / "reports/phase_interpolation/development/behavior/build_milk_figure.py",
            )
        },
        "limitations": [
            "This first-completed-worker contrast is not an estimate of aggregate evaluation performance.",
            "Wine also succeeds with random noise and every oracle arm; its success is not uniquely attributable to Astra.",
            "Wine's successful TLI+vision rollout includes 25 initial native-condition fallback actions, 50 actions with accepted text, and 10 actions with changed images.",
            "observed_phase and rationale are explicit model response fields, not hidden reasoning or independent measurements of progress.",
            "Each arm receives only the common baseline and its own prior decisions/rollouts, not oracle or other-arm histories.",
            "Figures show current raw feedback and explicitly labeled prior-rollout feedback; marked images were supplied only to the policy for at most 5 actions.",
            "Accepted steering, zero-residual choices, held text after rejection and native fallback are distinct recorded behaviors.",
            "Token totals include rejected calls when usage is available; reasoning tokens are included within output. No monetary rate is assumed.",
            "No new provider, model, GPU or simulator call was made for these examples. Recording audits do not independently rerun task outcomes.",
        ],
    }
    if args.orange_results_root is not None:
        orange = read_case(args.orange_results_root, "case_1_0", worker=1)["safe"]
        require(
            orange["episode_id"] == "libero_goal_ood:seed29:task1:state0",
            "Orange-juice identity",
        )
        require(
            not orange["arm_outcomes"]["random_noise"]["summary"]["success"]
            and orange["arm_outcomes"]["astra_tli"]["summary"]["success"]
            and orange["arm_outcomes"]["astra_tli_vision"]["summary"]["success"],
            "Declared supplementary contrast",
        )
        supplement = {
            "schema_version": "phase-interpolation-orange-evaluation-example-1",
            "status": "verified",
            "selection": "Additional case selected after inspection to illustrate Astra TLI rescues absent from this matched two-revision random-noise search; not representative selection.",
            "case": orange,
            "source_file_sha256": result["source_file_sha256"],
            "limitations": [
                "This case is separate from the first-completed-worker two-case totals and has no additional rendered figure.",
                "Random-noise failure is bounded by two attempts; oracle TEI also succeeds, so success is not unique to Astra.",
                "The explicit response hypotheses about donor cancellation are not proof of semantic cancellation or causal attribution.",
                "Both full array and feedback audits passed; no new provider, model, GPU or simulator calls were made.",
            ],
        }
        supplemental_path = HERE / "orange_juice.json"
        supplemental_path.write_text(
            json.dumps(supplement, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        result["supplementary_case"] = {
            "file": supplemental_path.name,
            "sha256": file_sha256(supplemental_path),
            "excluded_from_primary_two_case_totals": True,
        }
    path = HERE / "examples.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "examples_sha256": file_sha256(path),
                "provider_calls": sum(
                    c["provider"]["provider_calls"] for c in result["cases"].values()
                ),
                "total_tokens": sum(
                    c["provider"]["tokens"]["total_tokens"]["sum"]
                    for c in result["cases"].values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
