"""Publish the six completed RoboCasa development episodes as an offline report."""

import argparse
import gzip
import hashlib
import html
import json
import shutil
import statistics
from pathlib import Path

from astra_reversal.complex_manipulation.accounting import summarize_episodes


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifacts / "robocasa-native-dev3-results"
    worker = read(source / "worker_started.json")
    if read(source / "worker_finished.json")["returncode"] != 0:
        raise ValueError("Publish only the closed successful evaluation worker")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = args.output / "provenance"
    provenance.mkdir()
    for name in ("protocol.json", "release_manifest.json", "README.txt"):
        shutil.copyfile(Path(__file__).parent / name, provenance / name)
    shutil.copyfile(
        source / "archive_receipt.json", provenance / "original_archive_receipt.json"
    )
    shutil.copyfile(
        args.artifacts / "bench-runtime-results/receipt.json",
        provenance / "bench_policy_runtime.json",
    )
    rows, cards, table, all_latencies = [], [], [], []
    for directory in sorted((source / "evaluation").iterdir()):
        if not directory.is_dir():
            continue
        result = read(directory / "result.json")
        reset = read(directory / "reset.json")
        commands = [
            json.loads(x)
            for x in (directory / "executed_steps.jsonl").read_text().splitlines()
        ]
        predictions = [
            json.loads(x)
            for x in (directory / "predictions.jsonl").read_text().splitlines()
        ]
        if (
            len(commands) != result["executed_actions"]
            or len(predictions) != result["generated_chunks"]
        ):
            raise ValueError("Published accounting differs from execution records")
        if not result["eligible_for_completed_episode_sr"]:
            raise ValueError("The full native batch contains an incomplete episode")
        target = args.output / "evidence" / directory.name
        target.mkdir(parents=True)
        for path in directory.iterdir():
            if path.suffix == ".jsonl":
                (target / (path.name + ".gz")).write_bytes(
                    gzip.compress(path.read_bytes(), mtime=0)
                )
            else:
                shutil.copyfile(path, target / path.name)
        relative = target.relative_to(args.output).as_posix()
        latencies = [x["policy_seconds"] for x in predictions]
        all_latencies.extend(latencies)
        row = {
            **result,
            "instruction": reset["instruction"],
            "evidence": relative,
            "source_revision": worker["source_revision"],
            "median_policy_seconds": statistics.median(latencies),
        }
        rows.append(row)
        task = html.escape(result["task"])
        label = "Success" if result["success"] else "Horizon reached"
        table.append(
            f"<tr><th>{task}</th><td>{result['seed']}</td><td>{label}</td><td>1</td><td>{result['reset_free_segments']:,}</td><td>{result['executed_actions']:,}</td><td>{result['wall_seconds']:.1f} s</td><td>{statistics.median(latencies) * 1000:.1f} ms</td><td>0</td></tr>"
        )
        cards.append(
            f'<article class="card"><div class="card-head"><h3>{task}</h3><span>Seed {result["seed"]} · {label}</span></div><p>{html.escape(reset["instruction"])}</p><video controls preload="metadata" poster="{relative}/starting_image.png"><source src="{relative}/rollout.mp4" type="video/mp4"></video><details><summary>Starting view and execution evidence</summary><img src="{relative}/starting_image.png" alt="Native left, right and wrist camera views"><p><a href="{relative}/result.json">Episode result</a> · <a href="{relative}/executed_steps.jsonl.gz">Executed controls</a> · <a href="{relative}/predictions.jsonl.gz">Model predictions</a> · <a href="{relative}/initial_observation.npz">Initial policy input</a></p></details></article>'
        )
    if len(rows) != 6 or {(r["task"], r["seed"]) for r in rows} != {
        (task, seed)
        for task in ("LoadPreparedFood", "PackIdenticalLunches")
        for seed in (0, 1, 2)
    }:
        raise ValueError(
            "Native development batch does not match the declared task/seed matrix"
        )
    by_task = []
    for task in ("LoadPreparedFood", "PackIdenticalLunches"):
        selected = [r for r in rows if r["task"] == task]
        by_task.append(
            {
                "task": task,
                "successes": sum(r["success"] for r in selected),
                "episodes": len(selected),
                "success_rate": sum(r["success"] for r in selected) / len(selected),
            }
        )
    budget = read(args.artifacts / "budget.json")
    allocation = next(r for r in budget["entries"] if r["id"] == "robocasa-native-dev3")
    if "actual_gpu_hours" not in allocation:
        raise ValueError("Close this GPU allocation before publishing its cost")
    charged = budget["previous_study_gpu_hours"] + sum(
        r.get("actual_gpu_hours", 0) for r in budget["entries"]
    )
    summary = summarize_episodes(rows)
    result = {
        "status": "six_native_development_episodes_complete",
        "source_revision": worker["source_revision"],
        "summary": summary,
        "tasks": by_task,
        "episodes": rows,
        "known_constructor_resets": 6,
        "prior_incomplete_policy_episodes": 3,
        "prior_incomplete_controls": 718,
        "prior_incomplete_segments": 144,
        "teacher_calls": 0,
        "teacher_tokens": 0,
        "policy_updates": 0,
        "median_policy_seconds": statistics.median(all_latencies),
        "p95_policy_seconds": sorted(all_latencies)[int(len(all_latencies) * 0.95)],
        "first_policy_seconds_including_jit": all_latencies[0],
        "batch_gpu_hours": allocation["actual_gpu_hours"],
        "closed_allocation_total_gpu_hours": charged,
        "authorized_total_gpu_hours": budget["authorized_total_gpu_hours"],
        "active_gpu_workflows": [
            r.get("workflow") for r in budget["entries"] if "actual_gpu_hours" not in r
        ],
        "scope": "Seen RoboCasa composite tasks, pretrain scenes, development seeds 0/1/2; not held-out OOD.",
        "source_parity": "Pinned weights, dimensions and inference transforms checked. An independent same-noise output comparison against the full upstream factory has not been run.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    task_cards = "".join(
        f'<div class="stat"><span>{r["task"]}</span><strong>{r["successes"]}/{r["episodes"]}</strong><small>{100 * r["success_rate"]:.0f}% completed-episode success</small></div>'
        for r in by_task
    )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond LIBERO · Native baseline results</title><style>
    :root{--paper:#f5f5ef;--ink:#182932;--muted:#59696c;--green:#17685d;--line:#d8dfd9}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:44px 26px 80px}a{color:var(--green);text-underline-offset:4px}h1{font-size:clamp(34px,5.5vw,64px);line-height:1.08;letter-spacing:-.045em;max-width:950px;margin:18px 0}h2{font-size:28px;letter-spacing:-.02em;margin:42px 0 14px}h3{font-size:18px;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:750;color:var(--green)}.lead{font-size:20px;color:var(--muted);max-width:910px}.notice{background:#fff6e4;border-left:4px solid #b78130;padding:18px 22px;margin:26px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat,.card,.node{background:#fff;border:1px solid var(--line);border-radius:12px;padding:22px}.stat span,.stat small{display:block;color:var(--muted);font-size:12px}.stat strong{display:block;font-size:36px;font-weight:650;letter-spacing:-.04em}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:#fff}table{border-collapse:collapse;min-width:920px;width:100%;font-size:13px}th,td{padding:13px 15px;text-align:left;border-bottom:1px solid var(--line)}thead th{background:#edf2ed;font-size:11px;letter-spacing:.04em;text-transform:uppercase}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.node b{display:block;margin-bottom:8px}.node span{font-size:13px;color:var(--muted)}.node small{display:block;color:var(--green);font-size:11px;margin-bottom:10px}.videos{display:grid;gap:18px}.card-head{display:flex;justify-content:space-between;gap:16px;align-items:center}.card-head span{font-size:12px;color:var(--muted)}.card p{font-size:14px;color:var(--muted)}video,img{display:block;width:100%;height:auto;border-radius:7px;background:#17272d}details{margin-top:15px}summary{cursor:pointer;font-size:14px}details img{margin-top:16px}.small,footer{font-size:13px;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:44px;padding-top:22px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .card{height:100%}@media(max-width:800px){main{padding:28px 16px 60px}.stats,.flow{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.card-head{display:block}.card-head span{display:block;margin-top:6px}.card{padding:16px}}@media(max-width:400px){.stats,.flow{grid-template-columns:1fr}}
    </style></head><body><main><div class="eyebrow">Astra × System1 · Complex manipulation · 07 October 2026</div><h1>Six complete episodes.<br>A baseline we can inspect.</h1><p class="lead">Released RoboCasa π0.5, native robot controls, three cameras and full task horizons. Every reset, executed segment and outcome is backed by an archived rollout.</p><div class="notice"><b>These are development baselines on seen tasks.</b> They are not novel-instruction or held-out OOD scores. No Astra interventions or learning updates have run in this batch; teacher tokens are zero. Three episodes per task give only an initial screen.</div><div class="stats">__TASK_CARDS__<div class="stat"><span>Reset-free segments</span><strong>__SEGMENTS__</strong><small>__ACTIONS__ controls · 6 reset episodes</small></div><div class="stat"><span>Median model call</span><strong>__LATENCY__ ms</strong><small>__GPU__ L40S-hours for the full allocation</small></div></div>
    <h2>What the model does on each step</h2><div class="flow"><div class="node"><small>01 · OBSERVE</small><b>Three cameras + state</b><span>Original left, right and wrist RGB; 16 state values; the full task instruction.</span></div><div class="node"><small>02 · PREDICT</small><b>Native π0.5 flow model</b><span>Checkpoint mean/std normalization, 10 flow steps, 50 predicted actions, 32 internal channels.</span></div><div class="node"><small>03 · EXECUTE</small><b>Five control actions</b><span>12 native physical channels for the mobile PandaOmron. No Astra edits or extra candidate search.</span></div><div class="node"><small>04 · REPEAT</small><b>Observe the resulting state</b><span>Continue from the same environment until native success or the task's action horizon.</span></div></div><p class="small">One reset starts one SR trial. A five-action prefix is one reset-free segment; 300 segments in a 1,500-action episode are still one trial.</p>
    <h2>Complete per-episode results</h2><div class="table-wrap"><table><thead><tr><th>Task</th><th>Seed</th><th>Outcome</th><th>Resets</th><th>Segments</th><th>Controls</th><th>Rollout wall</th><th>Median model call</th><th>Teacher tokens</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="small">Wall time starts after reset recording and includes policy calls and environment stepping. Startup, restoration, reset preparation and teardown are additional allocation costs. The first policy call took __JIT__ seconds including compilation. Six additional upstream constructor reset calls are recorded separately. The earlier three incomplete attempts remain charged, with 718 controls and 144 segments; they are not added to this completed-episode SR denominator.</p>
    <h2>Watch the actual rollouts</h2><p>Videos show left, right and wrist views, in that order. Starting images and exact instructions accompany every episode. Playback follows simulator time; the table reports measured wall time.</p><div class="videos">__VIDEOS__</div>
    <h2>The next comparison</h2><div class="grid"><article class="card"><h3>Astra guidance and policy learning</h3><p>The requested conditions are native frozen policy, guidance only, learning only, and guidance + learning at 1 / 2 / 4 / 8 / 16 collection episodes. They have not yet been measured on these new tasks. Assisted success and autonomous student success will be scored separately.</p></article><article class="card"><h3>Dexterous native pilot</h3><p>Bench2Dex jigsaw uses 54 active coordinates and fridge pouring uses 38. Both use four cameras and 20-action chunks. We have staged the released task checkpoints, assets and frozen native policy environment. A separate Isaac job tests one recorded starting scene per task; this report does not claim its result.</p></article></div>
    <h2>What would make robot interfaces easier for agents?</h2><p>These are simulator integration findings, not physical-hardware validation. The largest burden is reconstructing the checkpoint-to-controller contract: joint order, active versus mimic coordinates, action units and reference frames, camera preprocessing, normalization and timing. A versioned machine-readable contract plus a small reference observation/action fixture would resolve many ambiguities before a rollout.</p><p>Typed outcomes and synchronized observations would help next. A failed grasp, stale camera frame, rejected command and simulator startup error need different recovery actions. Logs should include the requested command, the command actually applied, timestamps, control mode and measured feedback. Phase progress should be reported separately from final task success.</p>
    <h2>Compute and provenance</h2><p>The user authorized an additional 24 L40S-hours, raising the total ceiling to <b>48</b>. This batch used <b>__GPU__ hours</b>. Closed allocations across this and the preceding study total <b>__TOTAL__ hours</b> at report generation; active workflows are listed in the JSON and have additional reservations.</p><p class="small">Weights, parameter dimensions and inference transforms were checked against the pinned release. An independent same-noise numerical comparison with the complete upstream policy factory has not been run. This pilot measures our recorded execution; it does not substitute published benchmark scores.</p><footer><a href="results.json">Measured results JSON</a> · <a href="provenance/protocol.json">Protocol</a> · <a href="provenance/release_manifest.json">Pinned releases</a> · <a href="https://github.com/GaTech-RL2/EgoVerse/pull/706">Code / linear PR chain</a><br>Offline report: all videos, images and execution evidence are local. Raw JSONL traces are losslessly gzip-compressed; original archive hashes and a portable-file manifest are included.</footer></main></body></html>"""
    substitutions = {
        "__TASK_CARDS__": task_cards,
        "__SEGMENTS__": f"{sum(r['reset_free_segments'] for r in rows):,}",
        "__ACTIONS__": f"{sum(r['executed_actions'] for r in rows):,}",
        "__LATENCY__": f"{statistics.median(all_latencies) * 1000:.1f}",
        "__GPU__": f"{allocation['actual_gpu_hours']:.3f}",
        "__ROWS__": "".join(table),
        "__JIT__": f"{all_latencies[0]:.2f}",
        "__VIDEOS__": "".join(cards),
        "__TOTAL__": f"{charged:.3f}",
    }
    for key, value in substitutions.items():
        document = document.replace(key, value)
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
                "output": str(args.output),
                "episodes": len(rows),
                "tasks": by_task,
                "portable_bytes": sum(p["bytes"] for p in manifest),
            }
        )
    )


if __name__ == "__main__":
    main()
