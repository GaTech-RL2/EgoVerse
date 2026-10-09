"""Build an offline evidence report from the first three native RoboCasa probes."""

import argparse
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
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = args.output / "provenance"
    provenance.mkdir()
    for name in (
        "README.txt",
        "protocol.json",
        "release_manifest.json",
        "additional_release_audit.json",
    ):
        shutil.copyfile(Path(__file__).parent / name, provenance / name)
    supplement = args.artifacts / "bench-stage-results" / "anchor_supplement"
    if supplement.exists():
        shutil.copytree(supplement, provenance / "bench_anchor_audit")
        for name in (
            "stage_anchor_supplement.py",
            "bench_anchor_asset_supplement.json",
        ):
            shutil.copyfile(args.artifacts / name, provenance / name)
    stage_path = args.artifacts / "bench-stage-results" / "receipt.json"
    bench_stage = None
    if stage_path.exists():
        stage = read(stage_path)
        shutil.copyfile(stage_path, provenance / "bench_stage_receipt.json")
        bench_stage = {
            "workflow": stage["workflow"],
            "status": stage["status"],
            "gpu_count": stage["gpu_count"],
            "policy_rollouts": stage["policy_rollouts"],
            "archived_files": len(stage["uploaded_files"]),
            "archived_bytes": sum(f["bytes"] for f in stage["uploaded_files"]),
        }
    rows, cards, table = [], [], []
    for run in ("robocasa-smoke", "robocasa-loadfood"):
        source = args.artifacts / f"{run}-results"
        target = args.output / "evidence" / run
        shutil.copytree(source / "evaluation", target)
        # Storage catalogs, signed URLs, credentials and cluster node metadata
        # are deliberately outside this portable evidence directory.
        started = read(source / "worker_started.json")
        for directory in sorted(target.iterdir()):
            if not directory.is_dir() or not (directory / "result.json").exists():
                continue
            result, reset = (
                read(directory / "result.json"),
                read(directory / "reset.json"),
            )
            steps = [
                json.loads(x)
                for x in (directory / "executed_steps.jsonl").read_text().splitlines()
            ]
            chunks = [
                json.loads(x)
                for x in (directory / "predictions.jsonl").read_text().splitlines()
            ]
            if (
                len(steps) != result["executed_actions"]
                or len(chunks) != result["generated_chunks"]
            ):
                raise ValueError("Recorded controls/chunks do not match episode counts")
            result["source_revision"] = started["source_revision"]
            result["instruction"] = reset["instruction"]
            result["evidence"] = directory.relative_to(args.output).as_posix()
            result["scope"] = started["scope"]
            rows.append(result)
            relative = result["evidence"]
            label = (
                "5-action smoke probe"
                if run == "robocasa-smoke"
                else "Time-limit interruption"
            )
            title = html.escape(result["task"])
            table.append(
                f'<tr><th>{title}</th><td>{result["seed"]}</td><td>{label}</td><td>1</td><td>{result["reset_free_segments"]}</td><td>{result["executed_actions"]:,} / {result["action_limit"]:,}</td><td><span class="pill amber">Incomplete</span></td><td>Not measured</td></tr>'
            )
            cards.append(
                f'''<article class="video-card"><div class="card-top"><h3>{title}</h3><span class="pill">Seed {result["seed"]}</span></div><p>{label} · {result["executed_actions"]:,} recorded actions</p><video controls preload="metadata" poster="{relative}/starting_image.png"><source src="{relative}/rollout.mp4" type="video/mp4"></video><details><summary>Starting images and native instruction</summary><img src="{relative}/starting_image.png" alt="Left, right and wrist camera images at reset"><blockquote>{html.escape(reset["instruction"])}</blockquote></details><a href="{relative}/result.json">Episode record ↗</a></article>'''
            )
    summary = summarize_episodes(rows)
    full = next(row for row in rows if row["scope"] == "full_episodes")
    chunks = [
        json.loads(x)
        for x in (args.output / full["evidence"] / "predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    warm = [x["policy_seconds"] for x in chunks[1:]]
    budget = read(args.artifacts / "budget.json")
    if any("actual_gpu_hours" not in r for r in budget["entries"]):
        raise ValueError("Close GPU allocation accounting before publishing")
    new_hours = sum(r["actual_gpu_hours"] for r in budget["entries"])
    total_hours = budget["previous_study_gpu_hours"] + new_hours
    record = {
        "status": "native_inference_validated_full_episode_evaluation_incomplete",
        "summary": summary,
        "episodes": rows,
        "success_rate": None,
        "success_rate_reason": "Two deliberate short probes and one interrupted episode; zero completed evaluation episodes.",
        "known_constructor_reset_calls": 3,
        "constructor_reset_note": "One explicit upstream wrapper setup reset per successful environment construction, separate from the three policy episode resets. Lower-level simulator initialization is not an SR trial.",
        "teacher_calls": 0,
        "teacher_tokens": 0,
        "policy_updates": 0,
        "bench_cpu_staging": bench_stage,
        "warm_policy_latency": {
            "sample_count": len(warm),
            "median_seconds": statistics.median(warm),
            "mean_seconds": statistics.mean(warm),
            "first_call_seconds_including_jit": chunks[0]["policy_seconds"],
        },
        "compute": {
            "new_l40s_hours": new_hours,
            "prior_and_new_l40s_hours": total_hours,
            "authorized_l40s_hours": budget["authorized_total_gpu_hours"],
            "remaining_l40s_hours": budget["authorized_total_gpu_hours"] - total_hours,
            "additional_budget": budget["additional_authorization"],
        },
        "scope_note": "Seen RoboCasa composite tasks in pretrain scenes. These are not novel-instruction OOD evaluations.",
        "source_parity": "Released checkpoint shard hashes and parameter shapes validated; loader transforms reviewed against the pinned upstream factory. No independent same-noise numerical comparison against that factory was run.",
        "incomplete_control_note": "708 complete recorded controls in the interrupted episode; a control call in progress at SIGINT may have advanced the simulator partially.",
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond LIBERO · Initial native-policy checks</title><style>
    :root{--ink:#182932;--muted:#56676c;--paper:#f5f5ef;--line:#d8dfd9;--accent:#17685d;--amber:#866024}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1200px;margin:auto;padding:46px 26px 80px}a{color:var(--accent);text-underline-offset:4px}h1{font-size:clamp(34px,5vw,62px);line-height:1.08;letter-spacing:-.045em;max-width:900px;margin:15px 0 20px}h2{font-size:27px;letter-spacing:-.025em;margin:46px 0 14px}h3{margin:0;font-size:19px}p{max-width:890px}.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:750}.lead{font-size:20px;color:var(--muted)}.notice{border-left:4px solid #d2a253;background:#fff7e5;padding:18px 22px;margin:28px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px}.stat strong{display:block;font-size:34px;font-weight:650;letter-spacing:-.04em}.stat span{font-size:13px;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:15px 17px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}thead th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:#eef2ed}tbody tr:last-child>*{border:0}.pill{display:inline-block;white-space:nowrap;border:1px solid var(--line);padding:4px 9px;border-radius:20px;font-size:11px}.amber{background:#fff4dc;color:var(--amber);border-color:#ead4a8}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.node{border:1px solid var(--line);padding:22px;background:white;border-radius:12px;position:relative}.node b{display:block;margin-bottom:8px}.node p{font-size:14px;margin:0;color:var(--muted)}.node.active{border:2px solid var(--accent)}.node.pending{border-style:dashed;background:transparent}.node small{display:block;color:var(--accent);font-size:11px;margin-bottom:10px}.methods{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.methods article{padding:22px;border:1px solid var(--line);border-radius:12px}.methods p{font-size:14px;color:var(--muted);margin-bottom:0}.video-grid{display:grid;gap:20px}.video-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px}.card-top{display:flex;justify-content:space-between;gap:12px;align-items:center}.video-card p{font-size:14px;color:var(--muted);margin:8px 0 18px}video,img{width:100%;height:auto;display:block;border-radius:7px;background:#182932}details{margin:14px 0}summary{cursor:pointer;font-size:14px;padding:5px 0}details img{margin-top:12px}blockquote{margin:18px 0;padding-left:18px;border-left:2px solid var(--line);font-size:14px;color:var(--muted)}.small{font-size:13px;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:50px;padding-top:22px;font-size:13px;color:var(--muted)}@media(max-width:750px){main{padding:28px 16px 60px}.stats,.flow{grid-template-columns:1fr 1fr}.methods{grid-template-columns:1fr}.card-top{align-items:start;flex-direction:column}th,td{padding:12px}.video-card{padding:15px}}@media(max-width:400px){.stats,.flow{grid-template-columns:1fr}}
    </style></head><body><main><div class="eyebrow">Astra × System1 · Next benchmark study · 06 October 2026</div><h1>Beyond LIBERO.<br>Start with a verified native policy.</h1><p class="lead">Initial checks for the parallel-gripper and dexterous-task proposals. Real RoboCasa observations, released π0.5 weights, explicit episode accounting, and the evidence collected so far.</p><div class="notice"><b>No success rate has been established.</b> Both five-action probes passed the runtime checks. The longer LoadPreparedFood attempt stopped at its worker time limit after 708 of 1,500 allowed actions. There are zero completed evaluation episodes and no Astra interventions or policy updates.</div>
    <div class="stats"><div class="stat"><strong>3</strong><span>reset episodes started · all incomplete</span></div><div class="stat"><strong>144</strong><span>executed segments · 718 recorded actions</span></div><div class="stat"><strong>100 ms</strong><span>median warm policy call · 141 samples</span></div><div class="stat"><strong>__GPU__</strong><span>L40S-hours used for these probes</span></div></div>
    <h2>What has actually run</h2><p>RoboCasa365, mobile PandaOmron, the released <code>pi05_pretrain_human300</code> checkpoint. Three cameras and 16 state values feed a model with 32 internal action dimensions. It generates 50-step chunks; we execute the native five-action prefix before observing again. Normalization comes from this checkpoint. These are <b>seen composite tasks in pretrain scenes</b>.</p><div class="table-wrap"><table><thead><tr><th>Task</th><th>Seed</th><th>Run</th><th>Resets</th><th>Segments</th><th>Actions / limit</th><th>Completion</th><th>SR</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="small">A reset-free segment is one executed prefix, not another rollout for SR. The interrupted episode's last segment contains three recorded actions. Three additional constructor reset calls prepared the environments; they did not start policy evaluation trials. An interrupted control call may have advanced the simulator partially beyond the last complete recorded action.</p>
    <h2>The experimental sequence</h2><div class="flow"><div class="node"><small>01 · VERIFIED FOR INITIAL PAIRS</small><b>Pin the release</b><p>Source revision, checkpoint shards, normalization, cameras, action dimensions and horizon.</p></div><div class="node active"><small>02 · CURRENT PHASE</small><b>Validate native execution</b><p>Two RoboCasa smoke probes passed. Full-episode baseline evaluation remains incomplete.</p></div><div class="node pending"><small>03 · NOT RUN</small><b>Compare intervention and learning</b><p>Four conditions with a frozen protocol and explicit physical interaction counts.</p></div><div class="node pending"><small>04 · NOT RUN</small><b>Evaluate generalization</b><p>Separate held-out tasks/resets and autonomous student evaluation from assisted performance.</p></div></div>
    <h2>The requested comparison</h2><p>Planned collection checkpoints: <b>1, 2, 4, 8 and 16 reset episodes</b>. None of these guidance or learning comparisons has run on the new benchmarks. The intervention implementation and learning rule must be frozen after native policy validation.</p><div class="methods"><article><h3>Neither</h3><p>Frozen released policy. Native full episodes establish the comparison baseline.</p></article><article><h3>Guidance only</h3><p>Astra observes images, robot state and collection history, then steers the frozen policy. Log interventions, tokens, latency and every reset.</p></article><article><h3>Learning only</h3><p>Improve from the same declared interaction budget without Astra. Evaluate the updated student independently.</p></article><article><h3>Guidance + learning</h3><p>Use Astra-guided collection and update between episodes. Report assisted execution and the autonomous student as distinct outcomes.</p></article></div>
    <h2>Task inventory</h2><div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>Requested tasks</th><th>What is verified</th><th>What remains</th></tr></thead><tbody><tr><th>RoboCasa365</th><td>LoadPreparedFood<br>PackIdenticalLunches</td><td>Released weights restored; real camera inputs, native predictions and environment steps.</td><td>Complete native episodes; guidance and learning experiments.</td></tr><tr><th>Bench2Dex</th><td>Jigsaw Puzzle Assembly<br>Fridge Wine Interhand Pour</td><td>Pinned task policies and replay anchors; 54 / 38 active joints; four cameras; 20-action chunks.</td><td>Simulator and runtime joint-order validation. __BENCH_STAGE__ Native limits: 1,213 / 1,080 control actions. No rollout has run.</td></tr><tr><th>BEHAVIOR-1K</th><td>Make Pizza<br>Sorting Household Items<br>Clean Up Your Desk</td><td>Specialized checkpoints mapped to tasks 49 / 27 / 29.</td><td>Modified PiBehavior interface and simulation. Task embeddings differ from stock text conditioning.</td></tr><tr><th>EmbodiedSWE</th><td>PC assembly<br>IKEA table assembly</td><td>Requested environment configurations exist.</td><td>A capable matching trained System1 checkpoint is not verified.</td></tr><tr><th>DexVerse</th><td>OvenBakeSalmon<br>CleanTable</td><td>Public asset/demo release inspected.</td><td>Matching policies or demonstrations for these exact long-horizon tasks remain unverified.</td></tr></tbody></table></div>
    <h2>Rollout evidence</h2><p>Each video shows the actual left, right and wrist views. The two short clips contain only five actions. The longer clip contains the recorded part of one interrupted episode; it is not a completed failure.</p><div class="video-grid">__CARDS__</div>
    <h2>Timing, cost and interpretation</h2><p>The longer attempt generated 142 chunks and executed 708 recorded controls in 122.1 seconds after reset recording. Its first inference call took 43.15 seconds including JIT compilation. The remaining 141 calls averaged 100.06 ms. Asset restoration and environment setup are additional costs, included in the GPU allocation ledger.</p><p><b>Teacher calls: 0. Teacher tokens: 0. Policy updates: 0.</b> No conclusion about Astra's benefit follows from these probes. Camera and action mappings were checked against the pinned upstream code; an independent same-noise numerical comparison against its full factory was not run.</p><p>Prior study plus these probes: <b>__TOTAL__ / 24 L40S-hours</b>. Additional compute authorization is pending. The new GPU jobs are closed. GPU-zero preparation is accounted separately. Large setup costs made the bounded full-episode attempt incomplete; it must be rerun with enough time before scoring.</p><footer><a href="results.json">Download the measured records</a> · <a href="provenance/protocol.json">Experiment protocol</a> · <a href="provenance/release_manifest.json">Pinned releases</a> · <a href="https://github.com/GaTech-RL2/EgoVerse/pull/706">Code and linear PR chain</a><br>Report uses local assets and needs no server. Checkpoint/source revisions and per-episode records accompany the videos. Published benchmark scores are not substituted for measurements here.</footer></main></body></html>"""
    document = (
        document.replace("__GPU__", f"{new_hours:.3f}")
        .replace(
            "__BENCH_STAGE__",
            "Selected weights, assets and two replay anchors archived and hash-checked, with a separate supplement for two anchor distractors."
            if bench_stage
            else "Selected weights, assets and two replay anchors downloaded and hash-checked; archival in progress.",
        )
        .replace("__TOTAL__", f"{total_hours:.3f}")
        .replace("__ROWS__", "".join(table))
        .replace("__CARDS__", "".join(cards))
    )
    (args.output / "index.html").write_text(document)
    manifest = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(args.output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": len(manifest),
                "completed_episodes": summary["completed_episodes"],
                "recorded_actions": summary["executed_actions"],
            }
        )
    )


if __name__ == "__main__":
    main()
