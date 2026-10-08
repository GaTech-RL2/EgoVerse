"""Build a portable, receipt-checked comparison of RoboCasa representation arms."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import statistics
from pathlib import Path

from .accounting import summarize_episodes
from .build_language_report import cache_usage, jsonl, read

TASKS = ("LoadPreparedFood", "PackIdenticalLunches")


def link_or_copy(source, target):
    """Reports treat archived evidence as immutable; hard links avoid video copies."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_archive(source):
    receipt = read(source / "archive_receipt.json")
    for relative, expected in receipt["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Archive path escapes its result directory")
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Archived evidence changed: {relative}")
    return receipt


def evidence(source, target):
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        output = target / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl" or path.name.startswith("request_"):
            output.with_name(output.name + ".gz").write_bytes(
                gzip.compress(path.read_bytes(), mtime=0)
            )
        else:
            link_or_copy(path, output)


def totals(rows):
    result = summarize_episodes(rows)
    known = all(r.get("teacher_tokens") is not None for r in rows)
    complete = result["completed_episodes"]
    result.update(
        planned_episodes=6,
        full_batch_sr=result["completed_episode_sr"] if complete == 6 else None,
        teacher_calls=sum(r.get("teacher_calls", 0) for r in rows),
        teacher_tokens=sum(r["teacher_tokens"] for r in rows) if known else None,
        known_teacher_tokens=sum(r.get("teacher_tokens") or 0 for r in rows),
        wall_seconds=sum(r.get("wall_seconds", 0) for r in rows),
        teacher_seconds=sum(r.get("teacher_seconds", 0) for r in rows),
        policy_seconds=sum(r.get("policy_seconds", 0) for r in rows),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--diagnostics", nargs="*", default=[])
    parser.add_argument("--setup-failures", nargs="*", default=[])
    args = parser.parse_args()
    previous = read(args.previous_report / "results.json")
    if not previous["paired_comparison_complete"]:
        raise ValueError("The reference native/phase-prompt comparison is incomplete")
    # Verify the previous portable package before reusing its immutable files.
    previous_manifest = read(args.previous_report / "manifest.json")
    for row in previous_manifest:
        path = args.previous_report / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError("Previous report differs from its portable manifest")
    budget = read(args.artifacts / "budget.json")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        args.previous_report, args.output / "previous", copy_function=link_or_copy
    )
    provenance = args.output / "provenance"
    provenance.mkdir()
    for name in (
        "representation_protocol.json",
        "representation_protocol_v1.json",
        "representation_preflight.json",
        "representation_failures.json",
        "representation_sampler_drift.json",
        "representation_native_equivalence.json",
        "representation_tli_recurrence.json",
        "representation_teacher.py",
        "representation_checkpoint_validation.json",
        "representation_video_setup_failure.json",
        "jax_text_interpolation.py",
        "text_slots.py",
        "release_manifest.json",
    ):
        shutil.copyfile(Path(__file__).parent / name, provenance / name)
    attempts = {"tei": [], "tli": []}
    constructor_counts = {"tei": 0, "tli": 0}
    constructor_unknown = {"tei": 0, "tli": 0}
    allocations, diagnostics, by_pair = [], [], {}
    for name in [*args.runs, *args.diagnostics, *args.setup_failures]:
        source = args.artifacts / name
        verified_archive(source)
        worker = read(source / "worker_started.json")
        finish = read(source / "worker_finished.json")
        allocation = next(
            e for e in budget["entries"] if e.get("workflow") == worker["workflow"]
        )
        if "actual_gpu_hours" not in allocation:
            raise ValueError("Close each allocation before reporting its measured cost")
        allocations.append(allocation)
        destination = args.output / "evidence" / worker["workflow"]
        evidence(source, destination)
        preflight_path = source / "evaluation/interpolation_preflight.json"
        preflight = read(preflight_path) if preflight_path.exists() else None
        runtime = read(source / "evaluation/runtime.json")
        method = runtime["guidance"]["method"]
        constructor_path = source / "evaluation/constructors.json"
        constructors = read(constructor_path) if constructor_path.exists() else []
        if name in args.setup_failures:
            audit = read(provenance / "representation_video_setup_failure.json")
            if audit["workflow"] != worker["workflow"]:
                raise ValueError("Setup failure does not match its archived audit")
            constructors = [{"setup_resets": audit["constructor_setup_resets"]}]
        constructor_counts[method] += sum(c["setup_resets"] or 0 for c in constructors)
        constructor_unknown[method] += sum(
            c["setup_resets"] is None for c in constructors
        )
        diagnostics.append(
            {
                "workflow": worker["workflow"],
                "source_revision": worker["source_revision"],
                "exit_code": finish["returncode"],
                "preflight": preflight,
                "gpu_hours": allocation["actual_gpu_hours"],
                "evidence": destination.relative_to(args.output).as_posix(),
                "constructors": constructors,
            }
        )
        if name in [*args.diagnostics, *args.setup_failures]:
            if list((source / "evaluation").glob("*/started.json")):
                raise ValueError(
                    "An inference-only diagnostic started a policy episode"
                )
            continue
        for directory in sorted((source / "evaluation").glob("*_seed*")):
            result_path = directory / "result.json"
            if not result_path.exists():
                raise ValueError("Started episode lacks its final accounting receipt")
            result = read(result_path)
            arm = {"astra_tei_pi05": "tei", "astra_tli_pi05": "tli"}[result["method"]]
            key = (arm, result["task"], result["seed"])
            if result["task"] not in TASKS or result["seed"] not in (0, 1, 2):
                raise ValueError("Unregistered task/seed pair")
            predictions = jsonl(directory / "predictions.jsonl")
            commands = jsonl(directory / "executed_steps.jsonl")
            provider = jsonl(directory / "guidance/provider.jsonl")
            if (
                len(commands) != result["executed_actions"]
                or len(predictions) != result["generated_chunks"]
            ):
                raise ValueError("Execution journal differs from result counts")
            if len(provider) != result["teacher_calls"]:
                raise ValueError("Provider journal differs from the recorded calls")
            if result["executed_actions"] and (
                not preflight
                or preflight["status"] != "passed"
                or not read(directory / "guidance/paired_reset.json")["matched"]
            ):
                raise ValueError(
                    "Actions executed without validated hooks and matched reset"
                )
            relative = (
                (destination / "evaluation" / directory.name)
                .relative_to(args.output)
                .as_posix()
            )
            row = {
                **result,
                **cache_usage(provider),
                "evidence": relative,
                "source_revision": worker["source_revision"],
                "median_model_seconds": statistics.median(
                    p["policy_seconds"] for p in predictions
                )
                if predictions
                else None,
                "first_policy_seconds": predictions[0]["policy_seconds"]
                if predictions
                else None,
                "proposals": [
                    read(p)
                    for p in sorted((directory / "guidance").glob("proposal_*.json"))
                ],
            }
            attempts[arm].append(row)
            if result["executed_actions"] or result["episode_complete"]:
                if key in by_pair:
                    raise ValueError(
                        "Repeated executed pair cannot be selected best-of"
                    )
                by_pair[key] = row
    pairs = []
    for old in previous["paired_episodes"]:
        task, seed = old["task"], old["seed"]
        native = dict(
            old["native"],
            evidence=f"previous/native/robocasa/evidence/{task}_seed{seed}",
            teacher_tokens=0,
            teacher_calls=0,
        )
        phase = dict(old["guided"], evidence="previous/" + old["guided"]["evidence"])
        reset = read(args.output / native["evidence"] / "reset.json")
        pairs.append(
            {
                "task": task,
                "seed": seed,
                "instruction": reset["instruction"],
                "native": native,
                "phase": phase,
                **{arm: by_pair.get((arm, task, seed)) for arm in attempts},
            }
        )
    summaries = {
        "native": totals([p["native"] for p in pairs]),
        "phase": totals(previous["all_guided_reset_attempts"]),
        **{arm: totals(rows) for arm, rows in attempts.items()},
    }
    data = {
        "schema": "robocasa-representation-report-1",
        "is_ood": False,
        "pairs": pairs,
        "summaries": summaries,
        "all_representation_attempts": attempts,
        "diagnostics": diagnostics,
        "allocations": allocations,
        "representation_gpu_hours_including_stopped_checks": sum(
            a["actual_gpu_hours"] for a in allocations
        ),
        "representation_teacher_preflight": read(
            provenance / "representation_preflight.json"
        ),
        "prior_phase_teacher_preflight_tokens": previous["teacher_preflight_tokens"],
        "policy_updates": 0,
        "teacher_prompt": (provenance / "representation_teacher.py")
        .read_text()
        .split('SYSTEM_PROMPT = """', 1)[1]
        .split('"""', 1)[0],
        "baseline_parity": "Released weights and inference transforms are pinned. Hooks match this study's explicit num_steps=10 baseline at identical noise. The upstream server omits that keyword; independent parity with the full training-dependent factory remains unmeasured.",
        "constructor_resets": {
            "native": 6,
            "phase": len(previous["all_guided_reset_attempts"]),
            **constructor_counts,
        },
        "constructor_reset_unknown_attempts": constructor_unknown,
    }
    table_rows = []
    for pair in pairs:
        for method in ("native", "phase", "tei", "tli"):
            row = pair[method]
            if row is None:
                continue
            table_rows.append(
                {
                    "task": pair["task"],
                    "seed": pair["seed"],
                    "method": method,
                    "episode_complete": row["episode_complete"],
                    "success": row["success"],
                    **{
                        key: row.get(key)
                        for key in (
                            "executed_actions",
                            "reset_free_segments",
                            "assisted_segments",
                            "teacher_calls",
                            "teacher_tokens",
                            "wall_seconds",
                            "teacher_seconds",
                            "median_model_seconds",
                        )
                    },
                    "evidence": row["evidence"],
                }
            )
    with (args.output / "episode_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table_rows[0])
        writer.writeheader()
        writer.writerows(table_rows)
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    template = Path(__file__).with_name("representation_dashboard.html").read_text()
    # Keep arbitrary instruction/proposal text inert in the embedded JSON block.
    embedded = json.dumps(data).replace("<", "\\u003c")
    (args.output / "index.html").write_text(
        template.replace("__REPORT_DATA__", embedded)
    )
    manifest = [
        {
            "file": p.relative_to(args.output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    ]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "summaries": summaries, "files": len(manifest)}
        )
    )


if __name__ == "__main__":
    main()
