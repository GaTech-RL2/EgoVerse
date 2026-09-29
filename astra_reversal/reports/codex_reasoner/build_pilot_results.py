"""Build a public pilot page from audited records, without exporting CLI events."""

import argparse
import html
import json
import shutil
from pathlib import Path

from astra_reversal.records import file_sha256

LABELS = {
    "native_euler10": "Native π0.5",
    "native_repeated_noise": "Native, repeated noise",
    "astra_frs": "Codex Astra + FRS",
}


def read(path):
    return json.loads(path.read_text())


def build(root, output):
    root, output = Path(root), Path(output)
    task = root / "extracted/worker_0/results/task_6"
    audit = read(root / "frs_codex_audit.json")
    archive = read(root / "frs_codex_archive_audit.json")
    summary = read(task / "summary.json")
    assert audit["status"] == archive["status"] == "passed"
    assert audit["summary_sha256"] == file_sha256(task / "summary.json")
    assert summary["development"] is True and summary["task_id"] == 6
    assert audit["cohort_scope"] == "development_first_reset_only"
    assert len(summary["physical_rollouts"]) == 3
    assert {r["method"] for r in summary["physical_rollouts"]} == set(LABELS)
    output.mkdir(parents=True, exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    (output / "decisions").mkdir(exist_ok=True)
    rows = []
    for rollout in summary["physical_rollouts"]:
        name = rollout["attempt_id"] + ".mp4"
        target = output / "videos" / name
        shutil.copyfile(task / name, target)
        usage = rollout["provider_usage"]
        rows.append(
            {
                "method": rollout["method"],
                "label": LABELS[rollout["method"]],
                "success": rollout["success"],
                "actions": rollout["actions_executed"],
                "wall_seconds": rollout["wall_seconds"],
                "policy_seconds_including_reasoner": rollout["policy_seconds"],
                "environment_seconds": rollout["environment_seconds"],
                "codex_jobs": usage["provider_calls"],
                "total_tokens": usage["tokens"]["total_tokens"]["sum"],
                "input_tokens": usage["tokens"]["input_tokens"]["sum"],
                "output_tokens": usage["tokens"]["output_tokens"]["sum"],
                "reasoner_wait_seconds": usage["latency_seconds"],
                "interventions": rollout["counters"]["interventions"],
                "native_deferrals": rollout["counters"]["deferred"],
                "video": "videos/" + name,
                "video_sha256": file_sha256(target),
            }
        )
    providers = [
        json.loads(line) for line in (task / "provider.jsonl").read_text().splitlines()
    ]
    events = [
        json.loads(line) for line in (task / "events.jsonl").read_text().splitlines()
    ]
    decisions = []
    for event in events:
        if event["kind"] != "astra_decision":
            continue
        response = event["response"]
        provider = providers[event["provider_record_index"]]
        job = root / "jobs" / provider["invocation_id"]
        image = job / "image_0.png"
        assert (
            file_sha256(image) == provider["codex_receipt"]["image_hashes"][0]["sha256"]
        )
        destination = output / "decisions" / f"step_{response['observation_step']}.png"
        shutil.copyfile(image, destination)
        attached = provider["codex_receipt"]["image_hashes"]
        assert len(attached) == 2
        guide = job / attached[1]["file"]
        assert file_sha256(guide) == attached[1]["sha256"]
        guide_destination = (
            output / "decisions" / f"step_{response['observation_step']}_guide.png"
        )
        shutil.copyfile(guide, guide_destination)
        decisions.append(
            {
                "step": response["observation_step"],
                "native_deferral": response["fine"],
                "coords": response["coords"],
                "motion_amount": response["motion_amount"],
                "visible_justification": response["justification"],
                "image": "decisions/" + destination.name,
                "image_sha256": file_sha256(destination),
                "guide_image": "decisions/" + guide_destination.name,
                "guide_image_sha256": file_sha256(guide_destination),
                "total_tokens": provider["token_usage"]["total_tokens"],
                "relay_seconds": provider["latency_seconds"],
            }
        )
    result = {
        "schema_version": "codex-pilot-results-1.0",
        "scope": "One prespecified development task/reset; not the full evaluation cohort.",
        "workflow": read(root / "submission.json")["name"],
        "source": read(root / "source_identity.json"),
        "model": "gpt-6-astra",
        "served_model": None,
        "effort": "medium",
        "backend": "local Codex CLI via authenticated OSMO relay",
        "suite": summary["suite"],
        "task_id": summary["task_id"],
        "instruction": summary["instruction"],
        "seed": 19,
        "reset": 1,
        "rollouts": rows,
        "decisions": decisions,
        "audits": {
            "flow_and_codex": {
                "status": audit["status"],
                "sha256": file_sha256(root / "frs_codex_audit.json"),
            },
            "sealed_archive": archive,
        },
        "limitations": [
            "Simulator success is env.check_success(), not Astra's judgment.",
            "One rollout per method on one matched development reset; no population success rate or improvement estimate.",
            "Fresh Codex job per decision. No policy update or cross-rollout critique in this FRS arm.",
            "Token counts include the Codex harness. Jobs are not raw HTTP request counts; no subscription dollar estimate.",
            "Videos run at 20 fps and omit model waits and the terminal post-action image.",
            "Audits verify saved artifacts, transforms and receipts; they do not rerun model inference or physics.",
        ],
    }
    (output / "pilot_results.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(root / "frs_codex_audit.json", output / "pilot_frs_audit.json")
    shutil.copyfile(
        root / "frs_codex_archive_audit.json", output / "pilot_archive_audit.json"
    )
    table = "\n".join(
        f"<tr><th>{html.escape(r['label'])}</th><td>{'Success' if r['success'] else 'Failed'}</td>"
        f"<td>{r['actions']}</td><td>{r['wall_seconds']:.1f}s</td>"
        f"<td>{r['codex_jobs']}</td><td>{r['total_tokens']:,}</td></tr>"
        for r in rows
    )
    videos = "\n".join(
        f'<article><h3>{html.escape(r["label"])}</h3><video controls preload="metadata" '
        f'src="{r["video"]}"></video><p>{r["actions"]} actions · '
        f'{r["wall_seconds"]:.1f}s wall time · {"success" if r["success"] else "failed"}</p></article>'
        for r in rows
    )
    buttons = "\n".join(
        f'<button class="{"native" if r["native_deferral"] else "edit"}" '
        f'data-index="{i}">{r["step"]}<small>{"Native" if r["native_deferral"] else "FRS"}</small></button>'
        for i, r in enumerate(decisions)
    )
    data = json.dumps(decisions).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Codex Astra · LIBERO pilots</title>
<style>
:root{font:16px/1.5 system-ui,sans-serif;color:#e5edf7;background:#101827;color-scheme:dark}body{max-width:1120px;margin:0 auto;padding:38px 24px}h1{font-size:2.4rem;line-height:1.15;margin:.35em 0}h2{margin-top:2em}p{max-width:85ch}a{color:#91c4ff}.muted,small{color:#a5b5c9}.tag{color:#7de0bd;letter-spacing:.08em;font-size:.8rem;text-transform:uppercase}.notice{border-left:3px solid #e6b964;padding:12px 18px;background:#20293b}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.cards article{padding:14px;background:#192438;border-radius:12px}.cards h3{font-size:1rem;margin:0 0 12px}video{width:100%;border-radius:8px}.cards p{font-size:.86rem}table{border-collapse:collapse;width:100%;margin:22px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #364257}thead{color:#a5b5c9;font-size:.9rem}.scroll{overflow-x:auto}.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:24px 0}.box{background:#192f46;border:1px solid #386584;padding:12px;border-radius:8px;text-align:center;flex:1;min-width:135px}.arrow{color:#7de0bd;font-size:1.6em}.timeline{display:flex;gap:7px;margin:15px 0;flex-wrap:wrap}button{font:inherit;cursor:pointer;border:1px solid #4b647e;border-radius:8px;padding:8px 18px;color:#ecf6ff;background:#2e4059}button.edit{background:#14564d;border-color:#5dceaa}button.active{outline:2px solid #f0c678;outline-offset:2px}button small{display:block;color:inherit;font-size:.7rem}.decision{display:grid;grid-template-columns:240px 240px 1fr;gap:25px;align-items:start}.decision img{width:100%;border-radius:10px}.decision p{margin-top:0}footer{margin-top:40px;padding-top:20px;border-top:1px solid #364257;font-size:.85rem}@media(max-width:720px){.cards{grid-template-columns:1fr}.decision{grid-template-columns:1fr}.decision img{max-width:300px}h1{font-size:1.9rem}th,td{padding:8px}}
</style></head><body>
<span class="tag">Smart System2 · audited development pilot · 28 September 2026</span>
<h1>Codex Astra steers π0.5 through flow reversal</h1>
<p>“Put the wine bottle in the bowl” · LIBERO Goal OOD task 6 · seed 19, reset 1 · GPT-6 Astra, medium effort · OSMO L40S.</p>
<p class="notice"><strong>One matched development reset.</strong> FRS succeeded; both native controls failed. This is an implementation pilot, not a result for the 20-task evaluation. Frozen weights and all nine Codex request/response bindings passed artifact audits.</p>
<div class="scroll"><table><thead><tr><th>Method</th><th>Outcome</th><th>Actions</th><th>Wall time</th><th>Astra jobs</th><th>Total tokens</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p class="muted">Success comes from the simulator predicate after each action, with a 300-action cap. FRS used three interventions and six native deferrals. Its 112.92 seconds of reasoner waits are included in the 121.86-second rollout.</p>
<h2>How the intervention works</h2>
<div class="flow" role="img" aria-label="Current camera to Astra direction to reverse flow to guided noise to native forward flow to robot actions">
<div class="box">Current camera<br><small>VLM gripper guide</small></div><span class="arrow">→</span>
<div class="box">Astra direction<br><small>or defer to native</small></div><span class="arrow">→</span>
<div class="box">Reference action<br>→ reverse flow</div><span class="arrow">→</span>
<div class="box">Guided noise<br>→ forward flow</div><span class="arrow">→</span>
<div class="box">π0.5 actions<br><small>execute up to 10</small></div></div>
<p>Astra chooses a coarse translation direction every 10 actions. An intervention builds a normalized reference, reverses π0.5’s flow to infer noise, refreshes padded noise dimensions, and runs the frozen policy’s forward flow. A native deferral uses ordinary policy generation. There is no policy update in this arm.</p>
<h2>Watch the matched rollouts</h2><div class="cards">__VIDEOS__</div>
<p class="muted">20 fps playback excludes inference pauses and the terminal post-action frame. Read the measured wall times above for execution speed.</p>
<h2>What Astra saw and chose</h2><p>Choose an action step to inspect the actual attached VLM image and returned short explanation. The gripper guide is shown only to Astra; π0.5 receives raw camera images.</p>
<div class="timeline">__BUTTONS__</div><div class="decision"><div><small>Raw camera attachment</small><img id="decision-image" alt="Actual raw image supplied to Astra at the selected action step"></div><div><small>Guide attachment, Astra only</small><img id="guide-image" alt="Actual separate gripper guide supplied to Astra"></div><div><h3 id="decision-title"></h3><p id="decision-text"></p><p class="muted" id="decision-cost"></p></div></div>
<h2>Evaluation cohorts</h2><p>The full FRS protocol covers all 20published Goal/Spatial OOD tasks, ten resets per task, and three methods:600 physical rollouts. The VEI/VLI screen covers the same20tasks at one reset each, with matched random controls and up to two rescue attempts per arm. Development pilots are excluded from both cohorts.</p>
<footer><a href="pilot_results.json">Machine-readable results</a> · <a href="pilot_frs_audit.json">FRS / Codex audit</a> · <a href="pilot_archive_audit.json">Archive seal audit</a> · <a href="README.md">Harness and protocol details</a> · <a href="../smart_system2_results/README.md">Earlier results</a><p>Tokens include the Codex harness. Internal provider requests and subscription dollar cost are not inferred. Auditing checks saved evidence; it does not independently rerun physics or hidden model computation.</p></footer>
<script>const decisions=__DATA__;const buttons=[...document.querySelectorAll('button[data-index]')];function show(i){const d=decisions[i];document.getElementById('decision-image').src=d.image;document.getElementById('guide-image').src=d.guide_image;document.getElementById('decision-title').textContent=`Step ${d.step}: ${d.native_deferral?'native deferral':'FRS intervention'}`;document.getElementById('decision-text').textContent=d.visible_justification;document.getElementById('decision-cost').textContent=`${d.total_tokens.toLocaleString()} total tokens · ${d.relay_seconds.toFixed(2)}s reasoner wait`;buttons.forEach((b,j)=>b.classList.toggle('active',i===j));}buttons.forEach((b,i)=>b.addEventListener('click',()=>show(i)));show(0);</script></body></html>
"""
    for key, value in {
        "TABLE": table,
        "VIDEOS": videos,
        "BUTTONS": buttons,
        "DATA": data,
    }.items():
        page = page.replace("__" + key + "__", value)
    (output / "index.html").write_text(page)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frs_pilot", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    result = build(args.frs_pilot, args.output)
    print(json.dumps({"scope": result["scope"], "rollouts": len(result["rollouts"])}))


if __name__ == "__main__":
    main()
