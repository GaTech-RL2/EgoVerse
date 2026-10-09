"""Combine completed RoboCasa and Bench2Dex native pilots in one portable report."""

import argparse
import gzip
import hashlib
import html
import json
import shutil
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--robocasa-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifacts / "bench-native-dev1-results"
    if read(source / "worker_finished.json")["returncode"] != 0:
        raise ValueError("Require the completed native dexterous worker")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.robocasa_report, args.output / "robocasa")
    provenance = args.output / "provenance"
    provenance.mkdir()
    for name in ("protocol.json", "release_manifest.json", "README.txt"):
        shutil.copyfile(Path(__file__).parent / name, provenance / name)
    shutil.copyfile(
        source / "archive_receipt.json",
        provenance / "bench_original_archive_receipt.json",
    )
    shutil.copyfile(
        args.artifacts / "derive_bench_videos.py", provenance / "derive_bench_videos.py"
    )
    rows, cards, table = [], [], []
    for task, title in (
        ("73_jigsaw_puzzle_assembly", "Jigsaw Puzzle Assembly"),
        ("34_fridge_wine_interhand_pour", "Fridge Wine Interhand Pour"),
    ):
        directory = source / task
        native_rows = [
            json.loads(x)
            for x in (directory / "native_results/per_episode.jsonl")
            .read_text()
            .splitlines()
        ]
        if len(native_rows) != 1:
            raise ValueError("Expected one native pilot episode per task")
        native = native_rows[0]
        if native.get("error") or native["terminated_reason"] not in (
            "max_steps",
            "stable_success",
        ):
            raise ValueError("Incomplete native result must not enter the pilot table")
        calls = [
            json.loads(x)
            for x in (directory / "policy/inference.jsonl").read_text().splitlines()
        ]
        commands = [
            json.loads(x)
            for x in (directory / "audit/commands.jsonl").read_text().splitlines()
        ]
        mapping = read(directory / "audit/runtime_joint_map.json")
        video = read(directory / "video_receipt.json")
        if (
            len(calls) != native["policy_query_count"]
            or len(commands) != (native["steps"] + 2) // 3
            or video["frames"] != len(commands)
        ):
            raise ValueError(
                "Native physics, command, inference or video counts differ"
            )
        target = args.output / "bench" / task
        target.mkdir(parents=True)
        for name in ("native_results", "policy", "audit", "preflight"):
            shutil.copytree(directory / name, target / name)
        for name in (
            "commands.json",
            "pilot_summary.json",
            "rollout.mp4",
            "starting_image.png",
            "video_receipt.json",
        ):
            shutil.copyfile(directory / name, target / name)
        for path in list(target.rglob("*.jsonl")):
            path.with_suffix(path.suffix + ".gz").write_bytes(
                gzip.compress(path.read_bytes(), mtime=0)
            )
            path.unlink()
        relative = target.relative_to(args.output).as_posix()
        row = {
            "task": task,
            "title": title,
            "seed": native["seed"],
            "success": native["success"],
            "completed_episodes": 1,
            "unique_reset_anchors": 1,
            "active_dof": mapping["active_dof"],
            "full_dof": mapping["full_dof"],
            "mimic_rules": len(mapping["mimic_rules"]),
            "controls": len(commands),
            "physics_steps": native["steps"],
            "reset_free_segments": len(calls),
            "ever_stage_completion": native["latched_stage_completion_rate"],
            "final_stage_completion": native["current_stage_completion_rate"],
            "stage_completion": native["stage_completion"],
            "current_stage_completion": native["current_stage_completion"],
            "median_policy_seconds": statistics.median(x["seconds"] for x in calls),
            "first_policy_seconds_including_jit": calls[0]["seconds"],
            "teacher_calls": 0,
            "teacher_tokens": 0,
            "source_revision": read(source / "worker_started.json")["source_revision"],
            "evidence": relative,
        }
        rows.append(row)
        table.append(
            f"<tr><th>{html.escape(title)}</th><td>{int(native['success'])}/1</td><td>{native['seed']}</td><td>{len(commands):,}</td><td>{len(calls)}</td><td>{100 * row['ever_stage_completion']:.0f}%</td><td>{100 * row['final_stage_completion']:.0f}%</td><td>{row['median_policy_seconds'] * 1000:.1f} ms</td><td>0</td></tr>"
        )
        stages = "".join(
            f"<li>{html.escape(stage)}: {'reached' if reached else 'not reached'}; {'satisfied' if native['current_stage_completion'][stage] else 'not satisfied'} at the end</li>"
            for stage, reached in native["stage_completion"].items()
        )
        cards.append(
            f'<article class="card"><h3>{html.escape(title)}</h3><p>One episode, one recorded scene · seed {native["seed"]} · {mapping["active_dof"]} active / {mapping["full_dof"]} full joints</p><video controls preload="metadata" poster="{relative}/starting_image.png"><source src="{relative}/rollout.mp4" type="video/mp4"></video><details><summary>Native stage outcomes and evidence</summary><ul>{stages}</ul><p><a href="{relative}/native_results/per_episode.jsonl.gz">Complete native result</a> · <a href="{relative}/audit/runtime_joint_map.json">Runtime joint map</a> · <a href="{relative}/audit/commands.jsonl.gz">Command journal</a> · <a href="{relative}/video_receipt.json">Video provenance</a></p></details></article>'
        )
    budget = read(args.artifacts / "budget.json")
    if any("actual_gpu_hours" not in r for r in budget["entries"]):
        raise ValueError(
            "Close all owned GPU allocations before publishing this snapshot"
        )
    used = budget["previous_study_gpu_hours"] + sum(
        r["actual_gpu_hours"] for r in budget["entries"]
    )
    bench_hours = next(
        r["actual_gpu_hours"]
        for r in budget["entries"]
        if r["id"] == "bench-native-dev1"
    )
    robo = read(args.robocasa_report / "results.json")
    record = {
        "scope": "Eight native development episodes across four seen-task or recorded-anchor conditions; not held-out OOD, intervention or learning results.",
        "robocasa": robo,
        "bench2dex": rows,
        "native_policy_episode_resets": 8,
        "native_reset_free_segments": sum(
            r["reset_free_segments"] for r in robo["episodes"] + rows
        ),
        "teacher_calls": 0,
        "teacher_tokens": 0,
        "policy_updates": 0,
        "bench_gpu_hours": bench_hours,
        "total_closed_gpu_hours_including_previous_study": used,
        "authorized_gpu_hours": budget["authorized_total_gpu_hours"],
        "remaining_gpu_hours": budget["authorized_total_gpu_hours"] - used,
        "bench_recording_note": "Raw HDF5 recordings remain in the archived OSMO result set; their hashes and decoded four-camera MP4s accompany this portable report. The pre-physics command journal is not a post-execution acknowledgment on interrupted runs.",
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    native_html = (args.robocasa_report / "index.html").read_text()
    style = native_html.split("<style>", 1)[1].split("</style>", 1)[0]
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond LIBERO · Four native task pilots</title><style>__STYLE__</style></head><body><main><div class="eyebrow">Astra × System1 · Native pilots · 07 October 2026</div><h1>Four harder tasks.<br>The native starting point.</h1><p class="lead">Eight complete episodes with released π0.5 checkpoints across mobile parallel-gripper and bimanual dexterous robots. Inspect success, partial progress, exact control contracts and all rollout videos.</p><div class="notice"><b>No final task succeeded in this small pilot.</b> RoboCasa uses three development seeds per task; Bench2Dex uses one recorded starting scene per task. These are seen-task / recorded-anchor baselines, not held-out OOD measurements. Astra interventions, teacher tokens and policy updates are all zero.</div>
    <div class="stats"><div class="stat"><span>LoadPreparedFood · RoboCasa</span><strong>0/3</strong><small>1,500 controls per episode</small></div><div class="stat"><span>PackIdenticalLunches · RoboCasa</span><strong>0/3</strong><small>3,900 controls per episode</small></div><div class="stat"><span>Jigsaw · Bench2Dex</span><strong>0/1</strong><small>1,213 controls · 50% of stages reached</small></div><div class="stat"><span>Fridge pouring · Bench2Dex</span><strong>0/1</strong><small>1,080 controls · 40% of stages reached</small></div></div>
    <h2>Method: preserve each released policy's interface</h2><div class="flow"><div class="node"><small>01 · PIN</small><b>Checkpoint + native contract</b><span>Source, weights, normalization, camera projection, robot joint names and action semantics.</span></div><div class="node"><small>02 · RESET</small><b>Start a declared trial</b><span>RoboCasa development seed or Bench2Dex recorded scene anchor. Setup resets and homing remain separate.</span></div><div class="node"><small>03 · EXECUTE</small><b>Native action chunks</b><span>RoboCasa: predict 50, execute 5. Bench2Dex: predict and execute 20, with a shorter final prefix if needed.</span></div><div class="node"><small>04 · SCORE</small><b>Native success predicate</b><span>Keep actual commands, camera video, inference latency and benchmark stage outcomes. Every failed full episode counts.</span></div></div>
    <h2>RoboCasa: complete results and six videos</h2><p>Both tasks use the mobile PandaOmron, three cameras and a 16-value state. The released model has 32 internal action channels and returns 12 physical command channels. Median policy inference was 100.34 ms. The six full episodes used 0.544 L40S-hours including initialization.</p><p><a href="robocasa/index.html"><b>Open all RoboCasa instructions, starting images, videos and per-episode results ↗</b></a></p><p class="small">The nested report preserves the earlier RoboCasa snapshot. Its dexterous status was pending at that time; the completed dexterous outcomes below are the later measurements.</p>
    <h2>Bench2Dex: full episodes and partial progress</h2><div class="table-wrap"><table><thead><tr><th>Task</th><th>Success</th><th>Seed</th><th>Controls</th><th>Segments</th><th>Stages ever reached</th><th>Stages at end</th><th>Median model call</th><th>Teacher tokens</th></tr></thead><tbody>__ROWS__</tbody></table></div><p>Jigsaw reached two of four assembly conditions, then lost one. Fridge pouring opened the fridge and retrieved the bottle; it did not reach pouring, returning the bottle or closing the fridge. These are native stage-predicate measurements, not subjective video scores.</p><p class="small">Final success uses the native stable terminal condition. “Ever reached” is a latched progress measure; it does not mean those conditions hold together at the end. Each task currently has one unique anchor, so this is an integration pilot rather than a robust SR estimate.</p>
    <div class="videos">__VIDEOS__</div><p class="small">Video quadrants: top-left stereo-left, top-right stereo-right, bottom-left wrist-left, bottom-right wrist-right. All are recorded live policy camera streams. Fisheye wrist projection is native. Playback is 20 controls per simulated second; it does not represent measured wall speed.</p>
    <h2>What this taught us about agent-friendly robot interfaces</h2><div class="grid"><article class="card"><h3>Declare the control contract</h3><p>The jigsaw robot has 58 full joints but 54 active coordinates. Its native registry identifies four inactive coordinates; they are not mimic joints. In this execution, 56 of 58 joint indices differ between URDF and simulator ordering. Fridge pouring instead requires ten mimic rules to expand 38 active commands to 48 joints.</p><p>Expose joint names, ordering, units, frames, control mode, calibration, normalization and supported policy/checkpoint versions in machine-readable form, with a small reference observation/action fixture.</p></article><article class="card"><h3>Make execution and failure observable</h3><p>A command recorded before physics is not proof that it was applied. Expose accepted and applied commands, timestamps, controller state and measured feedback. Distinguish initialization errors, stale observations, rejected commands, task failure and partial progress.</p><p>For physical hardware, add deadlines, cancellation and a bounded local control loop. These are design recommendations drawn from simulator integration; we have not validated a physical robot in this study.</p></article></div>
    <h2>Comparison still to run</h2><p>The requested four conditions are native frozen policy, Astra guidance only, learning without Astra, and guidance + learning, measured at 1 / 2 / 4 / 8 / 16 collection episodes. The completed work here establishes native execution and initial failure trajectories. There is no measured intervention or learning improvement yet. Assisted performance and an autonomous updated student must be evaluated separately.</p>
    <h2>Compute and complete evidence</h2><p>The user authorized another 24 L40S-hours, raising the ceiling to <b>48</b>. The dexterous pilot used <b>__BENCH_HOURS__</b> hours. Including previous studies and the earlier incomplete probes, closed allocations total <b>__USED__</b> hours, leaving <b>__REMAINING__</b>. All GPU jobs in this snapshot are closed. GPU-zero staging is separate.</p><p class="small">Portable data includes exact native results, joint mappings, predictions, command traces, camera preflights and videos. Large raw HDF5 recordings remain in the archived result set and are identified by hash. The three earlier incomplete RoboCasa episodes remain charged and do not enter these eight completed trials.</p><footer><a href="results.json">All measured results JSON</a> · <a href="provenance/protocol.json">Protocol</a> · <a href="provenance/release_manifest.json">Pinned releases</a> · <a href="https://github.com/GaTech-RL2/EgoVerse/pull/706">Code and linear PR chain</a><br>This report works offline. No published score is substituted for a measured outcome.</footer></main></body></html>"""
    for key, value in {
        "__STYLE__": style,
        "__ROWS__": "".join(table),
        "__VIDEOS__": "".join(cards),
        "__BENCH_HOURS__": f"{bench_hours:.3f}",
        "__USED__": f"{used:.3f}",
        "__REMAINING__": f"{budget['authorized_total_gpu_hours'] - used:.3f}",
    }.items():
        document = document.replace(key, value)
    (args.output / "index.html").write_text(document)
    files = [
        {
            "file": p.relative_to(args.output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    ]
    (args.output / "manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bench_tasks": rows,
                "portable_bytes": sum(x["bytes"] for x in files),
            }
        )
    )


if __name__ == "__main__":
    main()
