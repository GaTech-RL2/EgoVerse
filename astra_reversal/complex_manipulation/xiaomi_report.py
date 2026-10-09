"""Build a portable native qualification report from verified rollout receipts."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from astra_reversal.complex_manipulation.worker import safe_relative, sha256


def build(results, output):
    source = Path(__file__).parent
    receipt = json.loads((results / "archive_receipt.json").read_text())
    for relative, item in receipt["files"].items():
        path = results / str(safe_relative(relative))
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Result archive is not intact")
    summary = json.loads((results / "evaluation/summary.json").read_text())
    protocol = json.loads((results / "protocol.json").read_text())
    selection = json.loads((source / "xiaomi_selection.json").read_text())
    equivalence = json.loads((source / "xiaomi_reset_path_equivalence.json").read_text())
    expected = {(r["task"], r["seed"]): r for r in equivalence["rows"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "media").mkdir()
    (output / "evidence").mkdir()
    rows = []
    for row in summary["episodes"]:
        group = next(g for g in protocol["groups"] if g["cohort"] == row["cohort"]
                     and g["task"] == row["task"] and row["seed"] in g["expected_seeds"])
        folder = results / "evaluation" / group["id"]
        reset_path = folder / f"seed{row['seed']}" / "reset.json"
        reset = json.loads(reset_path.read_text())
        identity = f"{group['id']}-seed{row['seed']}"
        video = folder / row["task"] / (
            f"episode_{row['episode']:03d}_seed_{row['seed']}_"
            f"{'success' if row['success'] else 'failure'}.mp4")
        image = reset_path.parent / "starting_image.png"
        shutil.copyfile(video, output / "media" / (identity + ".mp4"))
        shutil.copyfile(image, output / "media" / (identity + ".png"))
        shutil.copyfile(reset_path, output / "evidence" / (identity + "-reset.json"))
        value = {**row, "instruction": reset["instruction"],
                 "video": "media/" + identity + ".mp4", "image": "media/" + identity + ".png",
                 "paired_initial_state": None, "paired_model_xml": None, "paired_instruction": None}
        if row["cohort"] == "previous_pi05_seeds":
            old = expected[row["task"], row["seed"]]
            value.update(paired_initial_state=reset["state_sha256"] == old["state_sha256"],
                         paired_model_xml=reset["model_sha256"] == old["model_sha256_under_xiaomi_root"],
                         paired_instruction=reset["instruction"] == old["instruction"])
        rows.append(value)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cohort"], row["task"]].append(row)
    table = []
    for (cohort, task), episodes in grouped.items():
        n = len(episodes)
        table.append({"cohort": cohort, "task": task, "successes": sum(r["success"] for r in episodes),
                      "episodes": n, "minutes": sum(r["wall_seconds"] for r in episodes) / 60,
                      "controls": sum(r["steps"] for r in episodes),
                      "queries": sum(r["policy_queries"] for r in episodes),
                      "query_seconds": sum(r["policy_seconds"] for r in episodes),
                      "paired_resets": sum(bool(r["paired_initial_state"] and r["paired_model_xml"]
                                                and r["paired_instruction"]) for r in episodes)})
    constructors = json.loads((results / "evaluation/constructors.json").read_text())
    report = {"schema": "astra-xiaomi-native-report-1", "selection": selection,
              "summary": {k: v for k, v in summary.items() if k != "episodes"},
              "groups": table, "episodes": rows, "constructors": constructors,
              "native_source_revision": json.loads((results / "worker_started.json").read_text())["source_revision"],
              "verified_archive_files": len(receipt["files"]),
              "archive_receipt_sha256": sha256(results / "archive_receipt.json")}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "episode_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for path in (results / "protocol.json", source / "xiaomi_manifest.json",
                 source / "xiaomi_reset_path_equivalence.json", results / "archive_receipt.json",
                 results / "evaluation/environment.json", results / "worker_finished.json"):
        shutil.copyfile(path, output / "evidence" / path.name)
    embedded = json.dumps(report).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__DATA__", embedded))
    manifest = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RoboCasa · Native baseline qualification</title>
<style>
:root{color-scheme:light;--ink:#16322e;--muted:#627771;--green:#17664e;--line:#d6e0d9;--paper:#f5f5ef}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:48px 28px 80px}.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:var(--green)}
h1{font-size:clamp(32px,5vw,62px);line-height:1.08;max-width:850px;margin:18px 0}h2{font-size:25px;margin:0 0 16px}p{max-width:850px}.muted,small{color:var(--muted)}
.lead{font-size:19px;max-width:780px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:32px 0}.card{padding:24px;border:1px solid var(--line);border-radius:14px;background:#fff}
.value{font-size:32px;font-weight:750;line-height:1.3}.card small{display:block}.section{border-top:1px solid var(--line);margin-top:36px;padding-top:30px}.scroll{overflow:auto}
table{border-collapse:collapse;width:100%;white-space:nowrap;background:#fff;border-radius:12px}th,td{padding:14px 16px;text-align:left;border-bottom:1px solid var(--line)}th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.flow div{background:#e5ede6;border-radius:12px;padding:20px}.flow b{display:block;color:var(--green);margin-bottom:8px}
.viewer{display:grid;grid-template-columns:1.6fr 1fr;gap:24px;margin-top:18px}video,img{width:100%;border-radius:10px;background:#13251f}select{padding:12px;border:1px solid var(--line);border-radius:8px;font:inherit;width:100%;background:#fff;color:var(--ink)}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;background:#e5ede6;font-size:13px;font-weight:700}.failed{background:#f3e5d7;color:#845020}a{color:var(--green)}details{border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:14px}summary{cursor:pointer;font-weight:650}.links{display:flex;gap:22px;flex-wrap:wrap}
@media(max-width:720px){main{padding:28px 16px}.cards,.viewer{grid-template-columns:1fr}.flow{grid-template-columns:1fr 1fr}.value{font-size:28px}th,td{padding:10px}}
</style><main>
<div class="eyebrow">Smart System2 research · Native policy qualification</div>
<h1>A stronger starting point for RoboCasa.</h1>
<p class="lead">Xiaomi-Robotics-1 runs through its released inference pipeline, before any Astra intervention. This small screen tests native competence and checks the earlier π0.5 task/seed cases.</p>
<div class="cards"><div class="card"><small>Policy under test</small><div class="value">Xiaomi-Robotics-1</div><small>RoboCasa365 release · frozen weights</small></div>
<div class="card"><small>Completed physical episodes</small><div class="value" id="completed"></div><small>One attempt per reset · full task horizons</small></div>
<div class="card"><small>Astra calls / tokens</small><div class="value">0 / 0</div><small>Motor actions come from the native policy</small></div></div>
<section class="section"><h2>Why this baseline</h2><p>The <a href="https://robocasa.ai/leaderboard.html">official leaderboard</a> lists Xiaomi as the highest-scoring policy with released code and weights. These are published benchmark scores, separate from our measurements below.</p>
<div class="scroll"><table><thead><tr><th>Published policy</th><th>Overall</th><th>Atomic seen</th><th>Composite seen</th><th>Composite unseen</th></tr></thead><tbody>
<tr><td>Xiaomi-Robotics-1</td><td>57.4%</td><td>80.2%</td><td>57.1%</td><td>32.1%</td></tr>
<tr><td>π0.5</td><td>16.9%</td><td>39.6%</td><td>7.1%</td><td>1.2%</td></tr></tbody></table></div>
<p class="muted">Checked October 8, 2026. A separate <a href="https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/main/eval_robocasa365/summary.json">released evaluation trace</a> reports 1,432/2,500 = 57.28%, slightly different from the leaderboard entry. Our screen does not reproduce that complete benchmark.</p></section>
<section class="section"><h2>The native control loop</h2><div class="flow"><div><b>01 · Observe</b>Three cameras, robot state and the original task instruction.</div><div><b>02 · Encode</b>Released processor; four observation frames sampled every two control steps; 95% center crop.</div><div><b>03 · Predict</b>The released VLM and flow action head generate a chunk using five flow steps.</div><div><b>04 · Execute & repeat</b>Decode 12-dimensional actions; execute 16, then observe again. Stop at success or the full horizon.</div></div>
<p class="muted">The upstream server, processor, action decoder and rollout loop are retained. Instrumentation records evidence and checks finite actions. The model uses a persistent RNG stream initialized at 7; the smaller task schedule differs from the original full benchmark.</p></section>
<section class="section"><h2>Measured results</h2><p>“Reference first five” uses the first five episode indices of the released task schedule, selected before outcomes. The earlier-case cohort uses seeds 0, 1 and 2. Report success as successes / completed physical episodes; action chunks are not additional trials.</p>
<div class="scroll"><table><thead><tr><th>Cohort / task</th><th>Xiaomi success</th><th>Earlier π0.5</th><th>Episode time</th><th>Actions / chunks</th><th>Mean query</th></tr></thead><tbody id="results"></tbody></table></div>
<p id="pairing" class="muted"></p><p class="muted">CloseFridge is an atomic seen task. PackIdenticalLunches is a seen composite; ArrangeBreadBasket is a held-out composite. LoadPreparedFood is in Human300 but outside target50. All use pretrain kitchens. Small per-task samples are qualification evidence, not precise leaderboard estimates.</p></section>
<section class="section"><h2>Every rollout, including failures</h2><select id="selector" aria-label="Choose a rollout"></select><div class="viewer"><div><video id="video" controls preload="metadata"></video><p class="muted">Original left, right and wrist cameras. Playback follows the environment's control time; policy inference pauses are omitted.</p></div><div><span id="status" class="badge"></span><p id="instruction"></p><p id="episodeStats" class="muted"></p><img id="start" alt="Initial views from all three robot cameras"></div></div></section>
<section class="section"><h2>Audit and reproducibility</h2><div class="links"><a href="episode_results.csv" download>Per-episode CSV</a><a href="results.json">Complete results</a><a href="evidence/protocol.json">Registered protocol</a><a href="evidence/environment.json">Runtime & source</a><a href="manifest.json">File hashes</a></div>
<details><summary>Reset and rollout accounting</summary><p id="accounting"></p><p>One physical rollout begins at an explicit episode reset. The native gym constructor also performs a setup reset. Reset-free action chunks continue the same physical episode; there are no best-of retries. Success comes from the native environment checker. A worker interruption is incomplete, not a completed failure.</p></details>
<details><summary>Comparison limits</summary><p>The six earlier cases are called paired only when instruction, simulator state and model XML agree. The sole allowed XML change is the declared installation-directory prefix; no physical parameters are normalized. Xiaomi executes 16 actions per prediction versus five in the earlier π0.5 pilot. Episode wall time includes evidence/video work, and query timing includes native processing and socket inference.</p><p>The earlier π0.5 pilot passed its recorded numerical checks; independent output parity with its full upstream policy factory remains unmeasured. Xiaomi uses its released model factory and inference pipeline directly.</p><p>These observations do not measure Astra TEI/TLI, FRS, vision edits or policy learning on Xiaomi. Earlier intervention results used π0.5 and remain a separate experiment.</p></details>
</section></main><script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
$('completed').textContent=d.summary.completed_episodes+' / '+d.summary.intended_episodes;
for(const g of d.groups){const tr=document.createElement('tr');const label=(g.cohort==='reference_first5'?'Reference first five · ':'Earlier cases · ')+g.task;const vals=[label,g.successes+'/'+g.episodes+' ('+(100*g.successes/g.episodes).toFixed(0)+'%)',g.cohort==='previous_pi05_seeds'?'0/3':'Not run in our earlier pilot',g.minutes.toFixed(1)+' min',g.controls.toLocaleString()+' / '+g.queries,(g.query_seconds/g.queries).toFixed(2)+' s'];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}$('results').appendChild(tr)}
const paired=d.episodes.filter(r=>r.cohort==='previous_pi05_seeds'),matches=paired.filter(r=>r.paired_initial_state&&r.paired_model_xml&&r.paired_instruction).length;
$('pairing').textContent='Reset audit: '+matches+'/'+paired.length+' earlier task/seed cases pass all three equivalence checks. '+(matches===paired.length?'Their initial conditions are paired after the declared asset-path substitution.':'Cases that fail these checks must not be treated as paired resets.');
for(let i=0;i<d.episodes.length;i++){const r=d.episodes[i],o=document.createElement('option');o.value=i;o.textContent=r.task+' · seed '+r.seed+' · '+(r.success?'success':'failure')+' · '+r.cohort;$('selector').appendChild(o)}
function show(){const r=d.episodes[Number($('selector').value)];$('video').src=r.video;$('start').src=r.image;$('status').textContent=r.success?'Success':'Failure';$('status').className=r.success?'badge':'badge failed';$('instruction').textContent=r.instruction;$('episodeStats').textContent=r.steps+' / '+r.horizon+' actions · '+r.policy_queries+' chunks · '+(r.wall_seconds/60).toFixed(1)+' min wall time';}$('selector').addEventListener('change',show);show();
$('accounting').textContent=d.summary.completed_episodes+' completed episodes; '+d.constructors.filter(r=>r.status==='ready').length+' successful environment constructions with one documented setup reset each; '+d.summary.policy_control_steps+' policy actions; '+d.summary.reset_free_action_chunks+' reset-free action chunks. '+d.verified_archive_files+' source archive files verified by size and SHA-256.';
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.results, args.output)
    print(json.dumps({"output": str(args.output), "groups": report["groups"]}))


if __name__ == "__main__":
    main()
