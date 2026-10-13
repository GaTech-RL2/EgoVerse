"""Render a before/after harness example from authenticated trial archives.

Recorded model-written programs are displayed as text, never executed.
"""

import argparse
import base64
import difflib
import html
import json
from pathlib import Path

from experiments.libero_hardware_interface.tools.build_process_dashboard import (
    canonical,
    sha,
    validate_archive,
    validate_chain,
)

from .framework_diagrams import Figure


def read_example(source, recordings):
    cases, logs, provenance = {}, {}, {}
    for condition in ("F", "F_adaptive"):
        directory = source / condition
        raw = (directory / "events.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        chain = validate_chain(rows)
        archive = validate_archive(directory)
        outcome = json.loads((directory / "outcome.json").read_text())
        replay = json.loads((directory / "independent-replay.json").read_text())
        assert replay["status"] == "passed"
        assert replay["success"] == outcome["success"]
        assert replay["replayed_steps"] == outcome["sim_steps"]
        assert replay["terminal_state_matched"] is True
        logs[condition] = rows
        case = next(
            json.loads(p.read_text())
            for p in (recordings / "recordings").glob("*.json")
            if json.loads(p.read_text())["condition"] == condition
        )
        assert case["trial_id"] == outcome["trial_id"]
        assert case["provenance"]["event_log_sha256"] == sha(raw)
        cases[condition] = case
        provenance[condition] = {
            "trial_id": outcome["trial_id"],
            "archive": archive,
            "archive_receipt_sha256": sha(
                (directory / "archive-receipt.json").read_bytes()
            ),
            "event_log_sha256": sha(raw),
            "last_event_sha256": chain,
            "event_count": len(rows),
            "independent_replay": replay,
            "source_sha256": case["trial"]["source_sha256"],
            "source_commit": case["provenance"]["source_commit"],
        }
    identity = (
        "task_id",
        "init_state_index",
        "init_state_hash",
        "env_seed",
        "replicate",
    )
    assert all(
        cases["F"]["outcome"][k] == cases["F_adaptive"]["outcome"][k] for k in identity
    )
    assert provenance["F"]["source_sha256"] == provenance["F_adaptive"]["source_sha256"]
    initial = json.loads(
        (source / "F_adaptive/interface-evolution/revision-0000.json").read_text()
    )
    assert initial == {"capabilities": [], "tools": {}, "notes": ""}
    revisions = [
        {
            "revision": 0,
            "simulator_step": 0,
            "sequence": None,
            "monotonic_offset": 0,
            "state": initial,
            "reason": "Recorded empty interface at episode initialization.",
        }
    ]
    for event in logs["F_adaptive"]:
        if event["event"] == "interface_revised":
            stored = json.loads(
                (
                    source
                    / "F_adaptive/interface-evolution"
                    / ("revision-%04d.json" % event["revision"])
                ).read_text()
            )
            assert event["state"] == stored["state"]
            for definition in event["state"]["tools"].values():
                assert sha(canonical(definition["code"])) == definition["code_sha256"]
            revisions.append(
                {
                    k: event[k]
                    for k in (
                        "revision",
                        "simulator_step",
                        "sequence",
                        "monotonic_offset",
                        "state",
                        "reason",
                        "operation",
                    )
                }
            )

    def request(condition, name, helper=None):
        return next(
            {
                k: event[k]
                for k in (
                    "sequence",
                    "simulator_step",
                    "monotonic_offset",
                    "tool",
                    "arguments",
                )
            }
            for event in logs[condition]
            if event["event"] == "tool_request"
            and event["tool"] == name
            and (helper is None or event["arguments"].get("name") == helper)
        )

    tri_calls = []
    for index, event in enumerate(logs["F_adaptive"]):
        if (
            event["event"] == "tool_request"
            and event["tool"] == "call_interface_tool"
            and event["arguments"]["name"] == "tri"
        ):
            result = next(
                row
                for row in logs["F_adaptive"][index + 1 :]
                if row["event"] == "tool_response"
                and row["tool"] == "call_interface_tool"
            )
            tri_calls.append(
                {
                    "sequence": event["sequence"],
                    "simulator_step": event["simulator_step"],
                    "arguments": event["arguments"],
                    "response": result["response"],
                }
            )
    assert revisions[4]["simulator_step"] == revisions[5]["simulator_step"] == 211
    before, after = (revisions[i]["state"]["tools"]["tri"]["code"] for i in (4, 5))
    frames = []
    for label, sequence in (
        ("Episode start", 28),
        ("Immediately before revision 4 · native step 211", revisions[4]["sequence"]),
    ):
        latest = {}
        for event in cases["F_adaptive"]["events"]:
            if event["sequence"] > sequence:
                break
            for image in event["images"]:
                latest[image["camera"]] = image
        frames.append(
            {
                "label": label,
                "cameras": [
                    {
                        **image,
                        "data_url": "data:image/png;base64,"
                        + base64.b64encode(
                            (recordings / image["path"]).read_bytes()
                        ).decode(),
                    }
                    for image in latest.values()
                ],
            }
        )
    return {
        "selection_note": "One matched pair selected to illustrate an explicit mid-rollout code revision. It is not a representative sample or a treatment-effect estimate.",
        "instruction": cases["F_adaptive"]["instruction"],
        "identity": {k: cases["F_adaptive"]["outcome"][k] for k in identity},
        "provenance": provenance,
        "base_tools": [
            tool["name"] for tool in next(iter(cases["F"]["schemas"].values()))
        ],
        "adaptive_tools": [
            tool["name"] for tool in next(iter(cases["F_adaptive"]["schemas"].values()))
        ],
        "revisions": revisions,
        "fixed_act": request("F", "act"),
        "fixed_scratch": request("F", "scratch_execute"),
        "adaptive_move": request("F_adaptive", "call_interface_tool", "move"),
        "tri_calls": tri_calls,
        "tri_diff": "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="tri: revision 4",
                tofile="tri: revision 5",
            )
        ),
        "frames": frames,
        "outcomes": {
            key: {
                k: case["outcome"][k]
                for k in ("success", "sim_steps", "wall_s", "terminal_reason")
            }
            for key, case in cases.items()
        },
        "videos": {
            key: {
                "data_url": "data:video/mp4;base64,"
                + base64.b64encode(
                    (recordings / case["video"]["path"]).read_bytes()
                ).decode(),
                "frames": case["video"]["frames"],
                "sha256": case["video"]["sha256"],
            }
            for key, case in cases.items()
        },
    }


def diagram(data):
    f = Figure(
        "Recorded harness evolution: before and after",
        "A matched fixed-F interface and an adaptive episode that adds three reusable tools and camera calibration, with a recorded triangulation-code revision at native step 211.",
        1050,
    )
    f.text(
        56, 52, "RECORDED EXAMPLE · LIBERO TASK 8 / INITIAL STATE 3", 15, "#235eb0", 700
    )
    f.text(56, 103, "From F’s base API to Astra’s evolved interface", 36, weight=700)
    f.text(
        56,
        145,
        "Task: put both moka pots on the stove. The evolution below occurs within the F-adaptive episode.",
        21,
        "#4b657a",
    )
    f.box(
        56,
        207,
        452,
        225,
        "F · fixed interface",
        [
            "8 tools: typed read / act, source, scratch",
            "No named interface-tool registry",
            "F can still write Python feedback loops",
            "Recorded scratch program at step 198",
        ],
        "#087f70",
        "#eaf7f3",
    )
    f.box(
        574,
        207,
        452,
        225,
        "F-adaptive · step 0",
        [
            "Same base API + 8 evolution tools",
            "Initially empty registry and capabilities",
            "Enables camera_calibration",
            "Defines project (read) and move (control)",
        ],
        "#aac4e8",
        "#edf4ff",
    )
    f.box(
        1092,
        207,
        452,
        225,
        "F-adaptive · step 211",
        [
            "Adds tri: two-camera triangulation",
            "Revision 4: one vertical sign for both",
            "Revision 5: separate sign per camera",
            "Registry now contains 3 reusable tools",
        ],
        "#235eb0",
        "#edf4ff",
    )
    f.arrow("M1027 318H1082")
    f.text(
        58,
        473,
        "F and F-adaptive are separate matched trials; the fixed-F trial does not change treatment mid-rollout.",
        20,
        "#4b657a",
    )
    f.box(
        56,
        522,
        720,
        225,
        "Recorded fixed-F invocation · step 0",
        [
            "act(envelope)",
            "value: [0.8, 0.35, -0.6, 0, 0, 0, -1]",
            "duration_steps: 5",
            "episode + observation-step metadata supplied",
        ],
        "#087f70",
        "#eaf7f3",
    )
    f.box(
        824,
        522,
        720,
        225,
        "Recorded adaptive invocation · step 0",
        [
            "call_interface_tool(name='move', arguments_json=…)",
            "target: [-0.045, 0.077, 1.04]",
            "Tool reads position, clips feedback and calls act",
            "Every native step still passes the shared checks",
        ],
        "#235eb0",
        "#edf4ff",
    )
    f.text(
        56,
        785,
        "These calls illustrate interface shapes; they request different motions, not an equivalent-action comparison.",
        20,
        "#4b657a",
    )
    f.box(
        56,
        833,
        1488,
        156,
        "What the recording establishes",
        [
            "5 interface revisions · 3 reusable tools · 30 tool executions · no probe_dynamics calls in this episode",
            "Adaptive: success at 572 native steps. Matched fixed F: task failure at 658 steps. Both pass independent replay.",
            "The code was changed and used. This selected pair does not establish that the revision caused success.",
        ],
        "#d2dde6",
    )
    f.text(
        56,
        1027,
        "The companion HTML includes exact code, revision selection, original camera observations, videos and archive provenance.",
        18,
        "#4b657a",
    )
    return f.finish()


def page(data, svg, recordings_href=None):
    def code(value):
        return (
            "<pre>"
            + html.escape(
                value if isinstance(value, str) else json.dumps(value, indent=2)
            )
            + "</pre>"
        )

    frames = "".join(
        "<article><h3>"
        + html.escape(group["label"])
        + '</h3><div class="images">'
        + "".join(
            '<figure><img src="'
            + image["data_url"]
            + '"><figcaption>'
            + html.escape(image["camera"])
            + " · native step "
            + str(image["step"])
            + "</figcaption></figure>"
            for image in group["cameras"]
        )
        + "</div></article>"
        for group in data["frames"]
    )
    videos = "".join(
        "<article><h3>"
        + condition
        + '</h3><video controls preload="metadata" src="'
        + video["data_url"]
        + '"></video><p>'
        + str(video["frames"])
        + " recorded observation snapshots, played at 3 snapshots/second.</p></article>"
        for condition, video in data["videos"].items()
    )
    before, after = (
        data["revisions"][i]["state"]["tools"]["tri"]["code"] for i in (4, 5)
    )
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recorded F / F-adaptive harness example</title><style>
*{box-sizing:border-box}body{font:16px/1.6 system-ui,sans-serif;color:#203649;background:#f6f8fb;margin:0}main{max-width:1480px;margin:auto;padding:32px}h1{font-size:38px;line-height:1.2;margin:8px 0 18px}h2{font-size:26px;margin-top:38px}h3{margin:4px 0 14px}a{color:#235eb0}p{max-width:1160px}.eyebrow{font-size:13px;color:#235eb0;font-weight:700;letter-spacing:1px}.note{padding:18px 24px;background:#edf4ff;border-left:4px solid #235eb0;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.card,article{min-width:0;background:white;border:1px solid #d2dde6;border-radius:12px;padding:22px}pre{white-space:pre;overflow:auto;background:#142d46;color:#eef5fc;padding:18px;border-radius:8px;font:13px/1.6 ui-monospace,Menlo,monospace}code{font-family:ui-monospace,Menlo,monospace}select,button{font:inherit;padding:10px;margin:8px 0;border:1px solid #a7bbce;border-radius:6px;background:white}svg{width:100%;height:auto}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.images{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:128px}figure img{width:100%;max-width:256px;height:auto;border-radius:6px}figcaption{font-size:13px;color:#4b657a}video{width:100%;max-height:360px;background:#102130}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;vertical-align:top;padding:14px;border-bottom:1px solid #d2dde6}.links{display:flex;gap:22px;flex-wrap:wrap}@media(max-width:760px){main{padding:18px}.grid{grid-template-columns:1fr}h1{font-size:29px}}@media print{details{break-inside:avoid}.grid{display:block}video{display:none}}
</style></head><body><main><div class="eyebrow">ACTUAL ROLLOUT · VERIFIED ARCHIVE AND EVENT CHAIN</div><h1>What did Astra change in its harness?</h1><p><strong>Task:</strong> @@INSTRUCTION@@ · LIBERO task 8, initial state 3.</p><p class="note">F and F-adaptive are separate matched trials. F-adaptive begins with F’s base interface plus evolution tools and an empty registry. The “after” below is a recorded interface state reached during that adaptive episode. This pair was selected to illustrate the revision; it is not a representative sample or a causal estimate.</p><div class="links"><a href="framework.html">Framework guide</a>@@RECORDINGS_LINK@@<a href="harness-example.json" download>Download example evidence JSON</a><button id="svg-download">Download example SVG</button></div><div>@@SVG@@</div>
<h2>Before versus after: inspect the actual registry</h2><label for="revision">Adaptive revision</label> <select id="revision"></select><div class="grid"><article><h3>Fixed F / shared base interface</h3><p id="base-tools"></p><p>F has typed sensing and action tools, source inspection and Python scratch. It can write feedback loops: this F trial did so at native step 198. F-adaptive adds explicit named-tool versioning, capability access, probes and notes.</p><details><summary>Recorded fixed-F scratch code at step 198</summary>@@SCRATCH@@</details></article><article><h3 id="revision-title"></h3><p id="revision-reason"></p><p id="capabilities"></p><div id="tools"></div><details><summary>Exact saved interface state</summary><pre id="state"></pre></details></article></div>
<h2>How an action request looks</h2><p>These are exact recorded requests from the two trials, illustrating their interfaces. They request different motions.</p><div class="grid"><article><h3>F: native action envelope</h3>@@ACT@@</article><article><h3>F-adaptive: invoke its stored move program</h3>@@MOVE@@<p>The program reads current end-effector position, clips a proportional feedback command, and submits one native step at a time through <code>act</code>.</p></article></div>
<h2>An actual code revision at native step 211</h2><p>Astra first defined <code>tri</code> to test one shared vertical-image sign for both cameras. It then replaced that program with a version testing each camera’s sign independently: two combinations became four. Both revisions happened at the same native step while physics was paused.</p><div class="grid"><article><h3>Revision 4: shared sign</h3><pre>for sign in [1,-1]:
    # both camera rays use this sign</pre><details><summary>Exact recorded program</summary>@@TRI_BEFORE@@</details></article><article><h3>Revision 5: independent signs</h3><pre>for s1,s2 in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    # front ray uses s1; wrist ray uses s2</pre><details><summary>Exact recorded replacement</summary>@@TRI_AFTER@@</details></article></div><details><summary>Exact code diff</summary>@@DIFF@@</details><p>The two small snippets above are explanatory excerpts; the expandable programs and diff preserve the recorded code. On the immediate repeated query, the smallest ray-to-ray residual stayed about 0.00418 m in both versions. This is a geometric consistency residual, not ground-truth localization error. The trace demonstrates a change and its use; it does not establish an accuracy improvement.</p><details><summary>Exact triangulation requests and returned values</summary>@@TRI_CALLS@@</details>
<h2>What the actor saw</h2><p>Original saved PNGs. The two images at step 211 precede the triangulation revision; no new camera frame is invented between versions.</p><div class="grid">@@FRAMES@@</div><h2>Watch the recorded observation sequences</h2><p>These videos condense saved observations. They are not real-time playback or continuous simulator capture; native steps and wall times are marked in the frames.</p><div class="grid">@@VIDEOS@@</div>
<h2>Outcome and interpretation</h2><table><thead><tr><th>Trial</th><th>Official outcome</th><th>Native steps</th><th>Wall time</th></tr></thead><tbody><tr><td>F</td><td>Task failure</td><td>658</td><td>879.2 s</td></tr><tr><td>F-adaptive</td><td>Success</td><td>572</td><td>626.1 s</td></tr></tbody></table><p>Both traces passed independent native-action replay. F-adaptive recorded five revisions and 30 calls to its named tools; it did not call <code>probe_dynamics</code> in this episode. Camera calibration was additional supplied information, not a calibration estimate learned through those probes. Aggregate results belong to the full paired cohort.</p><details><summary>Archive and replay provenance</summary>@@PROVENANCE@@</details></main><script id="example-data" type="application/json">@@DATA@@</script><script>
const d=JSON.parse(document.getElementById('example-data').textContent),sel=document.getElementById('revision');d.revisions.forEach(r=>{const o=document.createElement('option');o.value=r.revision;o.textContent='Revision '+r.revision+' · native step '+r.simulator_step+' · '+r.monotonic_offset.toFixed(1)+' s';sel.append(o)});document.getElementById('base-tools').textContent=d.base_tools.join(' · ');function render(){const r=d.revisions[Number(sel.value)];document.getElementById('revision-title').textContent='F-adaptive · revision '+r.revision;document.getElementById('revision-reason').textContent='Recorded reason: '+r.reason;document.getElementById('capabilities').textContent='Enabled capabilities: '+(r.state.capabilities.join(', ')||'none');const container=document.getElementById('tools');container.replaceChildren();for(const [name,tool] of Object.entries(r.state.tools)){const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=name+' · '+(tool.read_only?'read only':'control')+' · '+tool.description;pre.textContent=tool.code;details.append(summary,pre);container.append(details)}if(!Object.keys(r.state.tools).length)container.textContent='No named tools defined yet.';document.getElementById('state').textContent=JSON.stringify(r.state,null,2)}sel.value='5';sel.addEventListener('change',render);render();document.getElementById('svg-download').onclick=()=>{const url=URL.createObjectURL(new Blob([document.querySelector('svg').outerHTML],{type:'image/svg+xml'}));const a=document.createElement('a');a.href=url;a.download='harness-before-after.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></body></html>"""
    # Video bytes are embedded in their media elements, not duplicated in JSON.
    embedded = {k: v for k, v in data.items() if k not in ("videos", "frames")}
    values = {
        "INSTRUCTION": html.escape(data["instruction"]),
        "RECORDINGS_LINK": (
            '<a href="'
            + html.escape(recordings_href, quote=True)
            + '">Open exact process recordings</a>'
            if recordings_href
            else ""
        ),
        "SVG": svg,
        "SCRATCH": code(data["fixed_scratch"]["arguments"]["code"]),
        "ACT": code(data["fixed_act"]),
        "MOVE": code(data["adaptive_move"]),
        "TRI_BEFORE": code(before),
        "TRI_AFTER": code(after),
        "DIFF": code(data["tri_diff"]),
        "TRI_CALLS": code(data["tri_calls"]),
        "FRAMES": frames,
        "VIDEOS": videos,
        "PROVENANCE": code(data["provenance"]),
        "DATA": json.dumps(embedded, ensure_ascii=True).replace("<", "\\u003c"),
    }
    for key, value in values.items():
        document = document.replace("@@" + key + "@@", value)
    assert not any("@@" + key + "@@" in document for key in values)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--recordings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--recordings-href",
        help="Optional relative link to a process-recording dashboard packaged alongside this page.",
    )
    args = parser.parse_args()
    data = read_example(args.source, args.recordings)
    # This figure intentionally documents one particular authenticated example.
    assert (
        data["identity"]["task_id"] == 8 and data["identity"]["init_state_index"] == 3
    )
    args.output.mkdir(parents=True, exist_ok=False)
    svg = diagram(data)
    (args.output / "harness-before-after.svg").write_text(svg)
    (args.output / "harness-example.html").write_text(
        page(data, svg, args.recordings_href)
    )
    evidence = {k: v for k, v in data.items() if k not in ("videos", "frames")}
    (args.output / "harness-example.json").write_text(json.dumps(evidence, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verified_trials": 2,
                "revisions": len(data["revisions"]) - 1,
                "model_code_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
