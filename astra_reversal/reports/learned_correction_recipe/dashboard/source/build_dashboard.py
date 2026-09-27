"""Build a portable dashboard from the sealed learned-recipe results.

Run with the project environment active. No network, policy or simulator calls.
All numerical displays are derived from the complete recorded report.
"""

import hashlib
import html
import json
import zipfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
OUTPUT = SOURCE.parent
RECIPE = OUTPUT.parent
ARMS = [
    "native",
    "recorded_schedule",
    "learned_selector",
    "flow_head",
    "gated_flow_head",
]
METHODS = {
    "native": {
        "name": "Frozen native π0.5",
        "short": "Native",
        "color": "#536781",
        "kind": "REFERENCE POLICY",
        "description": "The original policy turns the current images, robot state and task instruction into fresh actions at every replan.",
        "trigger": "Every 5 executed actions",
        "changes": "No intervention",
        "parameters": "0 learned",
        "astra": "No teacher needed",
        "training": "No additional learning. The pinned base checkpoint and its original normalization are used throughout.",
        "reading": "The paired reference for every method. A successful reset here can still become a failure under an intervention; those losses are counted explicitly.",
    },
    "recorded_schedule": {
        "name": "Recorded teacher schedule",
        "short": "Teacher schedule",
        "color": "#a8701e",
        "kind": "REPLAY A STORED LANGUAGE PROGRAM",
        "description": "Look up a successful Astra teacher’s TEI/TLI choice by task instruction and executed-action count. π0.5 still generates new actions from live observations.",
        "trigger": "Task + clock, every 25 actions",
        "changes": "Text conditioning",
        "parameters": "Stored choices; no fit",
        "astra": "Earlier successful rollout",
        "training": "One teacher is chosen by a frozen priority for each of eight covered tasks. Replay operator, source pair and α; hold the last choice after the teacher ends. Uncovered tasks use native.",
        "reading": "Measures reuse of a task-specific teacher program on new resets. Its language schedule does not adapt to visual progress, even though the underlying policy remains closed-loop.",
    },
    "learned_selector": {
        "name": "Learned TEI/TLI selector",
        "short": "Text selector",
        "color": "#3166c5",
        "kind": "LEARN WHEN AND HOW TO EDIT TEXT",
        "description": "A small selector reads frozen native visual and instruction features plus robot state. It chooses native or a TEI/TLI operator, source pair and interpolation strength.",
        "trigger": "Live features every 25 actions",
        "changes": "Text conditioning",
        "parameters": "535,321 selector weights",
        "astra": "Labels from prior teachers",
        "training": "1,000 offline updates on successful teacher choices and native anchors. No explicit clock, task ID or future outcome. Choices persist between refreshes; the original instruction can identify a familiar task.",
        "reading": "The gate imitates teacher intervention support. A score above 0.5 enables an edit; it is not a calibrated failure probability or a prediction that editing will help.",
    },
    "flow_head": {
        "name": "Learned flow-velocity head",
        "short": "Flow head",
        "color": "#8055b6",
        "kind": "LEARN A SMALL ACTION CORRECTION",
        "description": "A bounded residual changes the policy’s action vector field during each Euler step. Original camera inputs and language conditioning are preserved.",
        "trigger": "Every flow evaluation",
        "changes": "First 5 physical action rows",
        "parameters": "7,175 head weights",
        "astra": "Executed teacher actions",
        "training": "1,000 offline updates from actually executed teacher prefixes, with zero-residual native anchors. The base stays frozen. Unexecuted suffixes are imputed context, never demonstrated labels.",
        "reading": "The head is always enabled. Training flow-matching loss decreased, but the paired rollout counts show that lower training loss did not establish better control.",
    },
    "gated_flow_head": {
        "name": "Selector-gated flow head",
        "short": "Gated head",
        "color": "#147b72",
        "kind": "USE THE GATE TO LIMIT ACTION CORRECTIONS",
        "description": "The same learned selector enables the same learned flow head. Only the gate is used: this arm applies no TEI/TLI edit and keeps original conditioning.",
        "trigger": "Gate refreshed every 25 actions",
        "changes": "Flow residual only if gate > 0.5",
        "parameters": "Same selector + same head",
        "astra": "Both modules learned offline",
        "training": "No separate refit. Load the identical selector and head used in the other two arms. With the gate off, the vector field is exactly native under the matched noise.",
        "reading": "Tests whether limiting correction exposure reduces harm. Its gate is inherited from text-intervention labels; it was not trained on paired evidence of whether this action head helps.",
    },
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def text(x, y, value, size=15, color="#536781", weight=400):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(value)}</text>'


def box(x, y, w, title, lines, color="#536781", h=100, fill="#f6f8fc"):
    result = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="1.5"/>'
    result += text(x + 16, y + 30, title, 17, "#172a43", 650)
    for i, line in enumerate(lines):
        result += text(x + 16, y + 55 + 21 * i, line, 13)
    return result


def arrow(path, dashed=False, color="#8091a7"):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"{dash}/>'


def wrap(title, body, height, description, width=1220):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(description)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#8091a7"/></marker></defs>
<rect width="{width}" height="{height}" fill="white"/><g font-family="Inter,Arial,sans-serif">{body}</g></svg>'''


def flow(arm):
    m = METHODS[arm]
    c = m["color"]
    b = text(28, 32, m["kind"], 12, c, 750)
    b += text(
        28,
        62,
        "Astra is offline. The robot receives new observations after every five executed actions.",
        14,
    )
    if arm == "native":
        b += box(28, 140, 230, "Live observations", ["Two cameras + robot state"])
        b += box(
            335,
            140,
            230,
            "Native conditioning",
            ["Original task instruction", "Original image/text features"],
        )
        b += box(
            642,
            140,
            230,
            "Frozen π0.5",
            ["Native Gaussian noise", "10 Euler flow steps"],
        )
        b += box(950, 140, 230, "Execute 5 actions", ["Then observe again"])
        b += arrow("M258 190H335") + arrow("M565 190H642") + arrow("M872 190H950")
        b += arrow("M1065 240V315H143V240", True)
        b += text(470, 305, "CLOSED LOOP · FRESH OBSERVATIONS", 12)
        return wrap(m["name"], b, 355, m["description"])
    if arm == "recorded_schedule":
        b += box(
            28,
            106,
            255,
            "Earlier successful rollout",
            ["Astra chooses TEI/TLI", "Save source pair + α + step"],
            c,
            fill="#fff8ed",
        )
        b += box(
            355,
            106,
            255,
            "Stored teacher schedule",
            ["A fixed program for the task", "No new Astra decisions"],
            c,
            fill="#fff8ed",
        )
        b += box(
            725,
            106,
            350,
            "Task instruction + action clock",
            ["Look up every 25 executed actions", "Unknown task → native"],
            c,
            fill="#fff8ed",
        )
        b += arrow("M283 156H355") + arrow("M610 156H725")
        b += box(28, 300, 230, "Live observations", ["Two cameras + robot state"])
        b += box(
            355,
            300,
            255,
            "Apply stored text edit",
            ["TEI or TLI, pair and α", "Keep raw camera observations"],
            c,
            fill="#fff8ed",
        )
        b += box(
            665,
            300,
            230,
            "Frozen π0.5",
            ["Fresh actions for this state", "10 Euler flow steps"],
        )
        b += box(950, 300, 230, "Execute 5 actions", ["Advance the action clock"])
        b += arrow("M258 350H355") + arrow("M610 350H665") + arrow("M895 350H950")
        b += arrow("M900 206V255H482V300", True, c)
        b += arrow("M1065 400V465H143V400", True)
        b += text(
            365,
            455,
            "LANGUAGE CHOICES ARE REPLAYED · MOTOR ACTIONS ARE GENERATED LIVE",
            12,
        )
        return wrap(m["name"], b, 505, m["description"])
    if arm == "learned_selector":
        b += text(
            330, 104, "LEARNED OFFLINE · 1,000 updates · 535,321 parameters", 12, c, 650
        )
        b += box(
            28,
            237,
            240,
            "Live native inputs",
            ["Images + original instruction", "+ robot state"],
            h=112,
        )
        b += box(
            330,
            237,
            240,
            "Learned selector",
            [
                "Pool frozen native features",
                "Gate + operator + pair + α",
                "Refresh every 25 actions",
            ],
            c,
            h=133,
            fill="#eff5ff",
        )
        b += box(650, 148, 230, "Native conditioning", ["Use original inputs"])
        b += box(
            650,
            350,
            230,
            "TEI / TLI conditioning",
            ["Apply predicted pair and α", "Hold until next refresh"],
            c,
            fill="#eff5ff",
        )
        b += box(
            950, 247, 245, "Frozen π0.5", ["Fresh action chunk", "10 Euler flow steps"]
        )
        b += box(950, 485, 245, "Execute 5 actions", ["Then observe again"])
        b += arrow("M268 293H330")
        b += arrow("M570 271H610V198H650") + arrow("M570 335H610V400H650")
        b += text(590, 170, "gate ≤ 0.5", 12) + text(588, 436, "gate > 0.5", 12, c)
        b += arrow("M880 198H916V277H950") + arrow("M880 400H916V317H950")
        b += arrow("M1072 347V485") + arrow("M1072 585V630H148V349", True)
        b += text(500, 620, "NO CLOCK INPUT · CURRENT OBSERVATIONS DRIVE SELECTION", 12)
        return wrap(m["name"], b, 667, m["description"])
    gated = arm == "gated_flow_head"
    y = 240 if gated else 180
    if gated:
        b += box(
            315,
            100,
            245,
            "Learned selector: gate",
            ["Native image/text features", "Refresh every 25 actions"],
            c,
            fill="#eef9f6",
        )
        b += arrow(f"M143 {y}V150H315", True, c)
        b += text(28, 127, "native features", 12, c)
    b += box(
        28,
        y,
        230,
        "Original native inputs",
        ["Live images + robot state", "Original instruction"],
    )
    b += box(
        315,
        y,
        245,
        "Frozen action expert",
        ["At the current flow state", "Native velocity + hidden h"],
    )
    b += box(650, y - 65, 225, "Native velocity", ["v native"])
    b += box(
        650,
        y + 120,
        225,
        "Bounded residual head",
        [
            "7,175 learned parameters",
            "First 5 rows × 7 channels",
            "δv = 0.5 tanh((Wh+b)/0.5)",
        ],
        "#8055b6",
        h=132,
        fill="#f7f1fc",
    )
    b += box(
        960,
        y,
        235,
        "Flow integration",
        [
            "v = v native + g · δv" if gated else "v = v native + δv",
            "Repeat for 10 Euler steps",
        ],
        c,
        fill="#eef9f6" if gated else "#f7f1fc",
    )
    b += box(960, y + 285, 235, "Execute 5 actions", ["Then observe again"])
    b += arrow(f"M258 {y + 50}H315") + arrow(f"M560 {y + 35}H606V{y - 15}H650")
    b += arrow(f"M437 {y + 100}V{y + 183}H650")
    b += text(448, y + 174, "hidden h", 12, "#8055b6")
    b += arrow(f"M875 {y - 15}H918V{y + 25}H960")
    b += arrow(f"M875 {y + 185}H918V{y + 75}H960")
    b += arrow(f"M1078 {y + 100}V{y + 285}")
    if gated:
        b += arrow(f"M560 150H585V{y + 145}H650", True, c)
        b += text(651, y + 281, "g = 1 if gate > 0.5; else δv = 0", 12, c, 650)
    else:
        b += text(651, y + 279, "Always enabled", 12, c, 650)
    b += arrow(f"M1078 {y + 385}V{y + 440}H143V{y + 100}", True)
    b += text(
        395, y + 430, "ORIGINAL CONDITIONING · NO TEI / TLI IN EITHER HEAD ARM", 12
    )
    return wrap(m["name"], b, y + 478, m["description"])


def overview():
    b = text(28, 42, "Five methods · where the intervention enters", 27, "#172a43", 700)
    b += text(
        28,
        73,
        "All use the same frozen π0.5 base. Live observations close the loop after each five-action execution.",
        15,
    )
    steps = {
        "native": [
            ("Live observations", "Original task instruction"),
            ("Native conditioning", "No additional learning"),
            ("Frozen π0.5", "10 Euler flow steps"),
            ("Execute 5 actions", "Observe again"),
        ],
        "recorded_schedule": [
            ("Task + action count", "Stored Astra teacher program"),
            ("Recorded TEI / TLI", "Apply pair + α to live input"),
            ("Frozen π0.5", "Fresh motor actions"),
            ("Execute 5 actions", "Advance clock; observe again"),
        ],
        "learned_selector": [
            ("Native image/text features", "+ robot state"),
            ("Learned gate + choice", "Native or predicted TEI / TLI"),
            ("Frozen π0.5", "Fresh motor actions"),
            ("Execute 5 actions", "Refresh selector every 25"),
        ],
        "flow_head": [
            ("Original native inputs", "Live images, state, instruction"),
            ("Frozen action expert", "Native velocity + hidden h"),
            ("Add learned δv", "Bounded; always enabled"),
            ("Integrate + execute", "10 Euler steps; 5 actions"),
        ],
        "gated_flow_head": [
            ("Original native inputs", "Selector predicts gate only"),
            ("Frozen action expert", "Native velocity + hidden h"),
            ("Gate the same δv", "Off: native; on: correction"),
            ("Integrate + execute", "No extra TEI / TLI edit"),
        ],
    }
    for i, arm in enumerate(ARMS):
        y = 118 + i * 148
        m = METHODS[arm]
        b += text(28, y + 31, f"0{i + 1}", 18, m["color"], 750)
        b += text(28, y + 57, m["short"], 15, "#172a43", 650)
        for j, (title, subtitle) in enumerate(steps[arm]):
            x = 202 + j * 284
            b += box(
                x,
                y,
                248,
                title,
                [subtitle],
                m["color"] if j == 1 or j == 2 else "#91a1b5",
                89,
            )
            if j < 3:
                b += arrow(f"M{x + 248} {y + 44}H{x + 284}")
        b += text(
            202,
            y + 115,
            "Astra disabled at evaluation · matched reset and noise · one attempt per case",
            12,
        )
    return wrap(
        "All five policy methods",
        b,
        885,
        "Five rows compare native conditioning, recorded text edits, learned text edits, an always-active flow residual and a gated flow residual.",
        1370,
    )


def main():
    report_path = RECIPE / "results/report.json"
    report = read(report_path)
    assert report["status"] == "complete" and report["efficacy_released"]
    assert len(report["rows"]) == 240
    index = {(r["episode_id"], r["arm"]): r for r in report["rows"]}
    assert len(index) == 240 and len({key[0] for key in index}) == 48
    # Verify source bytes against the existing publication, before deriving UI data.
    original = read(RECIPE / "results/manifest.json")
    for name in ["report.json", "gallery.json", "osmo_task_cost.json"]:
        assert sha(RECIPE / "results" / name) == original["files"][name]["sha256"]
    for cohort in [
        "ood",
        "libero_goal_ood",
        "libero_spatial_ood",
        "ood_teacher_tasks",
        "ood_other_tasks",
        "id_panel",
    ]:
        for arm in ARMS:
            rows = [
                r
                for r in report["rows"]
                if r["arm"] == arm
                and (
                    (cohort == "ood" and r["cohort"] == "ood")
                    or (cohort == r["suite"])
                    or (cohort == "id_panel" and r["cohort"] == "id_panel")
                    or (
                        cohort == "ood_teacher_tasks"
                        and r["cohort"] == "ood"
                        and r["correction_teacher_task"]
                    )
                    or (
                        cohort == "ood_other_tasks"
                        and r["cohort"] == "ood"
                        and not r["correction_teacher_task"]
                    )
                )
            ]
            expected = report["groups"][cohort]["methods"][arm]
            assert len(rows) == expected["cases"]
            assert sum(r["credited_success"] for r in rows) == expected["successes"]
    fields = [
        "episode_id",
        "arm",
        "suite",
        "task_id",
        "initial_state_id",
        "instruction",
        "correction_teacher_task",
        "credited_success",
        "actions_executed",
        "action_budget",
        "gate_active_actions",
        "head_active_actions",
        "summary_sha256",
    ]
    compact_rows = []
    for row in report["rows"]:
        value = {k: row[k] for k in fields}
        value["decisions"] = [
            {k: d.get(k) for k in ["step", "choice", "gate_active", "gate_probability"]}
            for d in row["decisions"]
        ]
        compact_rows.append(value)
    gallery = read(RECIPE / "results/gallery.json")
    for clip in gallery["videos"]:
        assert clip["available"]
        assert sha(RECIPE / "results" / clip["path"]) == clip["sha256"]
        row = index[clip["episode_id"], clip["arm"]]
        assert (
            row["video_sha256"] == clip["sha256"]
            and row["actions_executed"] == clip["actions"]
        )
    flows = {arm: flow(arm) for arm in ARMS}
    (OUTPUT / "flows").mkdir(parents=True, exist_ok=True)
    for arm, svg in {**flows, "all-methods": overview()}.items():
        (OUTPUT / "flows" / f"{arm}.svg").write_text(svg + "\n")
    with zipfile.ZipFile(
        OUTPUT / "flows.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted((OUTPUT / "flows").glob("*.svg")):
            info = zipfile.ZipInfo(path.name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    data = {
        "schema_version": "learned-recipe-dashboard-1.0",
        "report_sha256": sha(report_path),
        "arms": ARMS,
        "methods": METHODS,
        "flows": flows,
        "rows": compact_rows,
        "groups": report["groups"],
        "gallery": gallery,
        "cost": read(RECIPE / "results/osmo_task_cost.json"),
        "extensions": None,
    }
    extension = RECIPE.parent / "ood_extensions/v1/manifest.json"
    if extension.exists():
        data["extensions"] = read(extension)
    write_json(OUTPUT / "data.json", data)
    serialized = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    page = (
        (SOURCE / "template.html")
        .read_text()
        .replace("__STYLE__", (SOURCE / "dashboard.css").read_text())
    )
    page = page.replace("__DATA__", serialized).replace(
        "__SCRIPT__", (SOURCE / "dashboard.js").read_text()
    )
    assert (
        "__DATA__" not in page and "__STYLE__" not in page and "__SCRIPT__" not in page
    )
    (OUTPUT / "index.html").write_text(page)
    files = [
        p
        for p in sorted(OUTPUT.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    manifest = {
        "schema_version": "recipe-dashboard-publication-1.0",
        "source_report_sha256": sha(report_path),
        "physical_rollouts": 240,
        "method_count": 5,
        "video_count": len(gallery["videos"]),
        "source_results_unchanged": True,
        "source": "source/build_dashboard.py",
        "files": {
            str(p.relative_to(OUTPUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in files
        },
    }
    if extension.exists():
        manifest["extension_manifest_sha256"] = sha(extension)
    write_json(OUTPUT / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "dashboard": str(OUTPUT / "index.html"),
                "files": len(files),
                "rollouts": 240,
                "report_sha256": sha(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
