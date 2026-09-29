"""Publish the 20 benchmark instructions and four recorded starting images.

No simulator, model, or provider call is made. Video frames are decoded by ffmpeg;
the cabinet PNG is copied byte-for-byte from an audited action-zero request.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
PROJECT = REPORTS.parent.parent
COVERAGE = REPORTS / "frs_policy_improvement/coverage.json"
RECIPE = REPORTS / "learned_correction_recipe/results"
CABINET = REPORTS / "phase_interpolation/development/examples/examples.json"
CASE = (
    PROJECT
    / "astra_reversal/.deps/interpolation-development-v1/extracted/worker_2/results/case_8_0"
)
CAMERA = "observation/image"
SOURCE_FILES = [
    COVERAGE,
    RECIPE / "gallery.json",
    RECIPE / "manifest.json",
    RECIPE / "report.json",
    CABINET,
    PROJECT / "astra_reversal/libero_runner.py",
]
SELECTIONS = [
    ("libero_goal_ood", 0, "G0"),
    ("libero_goal_ood", 5, "G5"),
    ("libero_spatial_ood", 2, "S2"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_png(path: Path) -> None:
    with Image.open(path) as picture:
        assert picture.format == "PNG" and picture.size == (224, 224), path
        assert picture.mode == "RGB", path
        assert any(low != high for low, high in picture.getextrema()), path


def extract_video_start(
    suite: str, task_id: int, short_id: str, instruction: str
) -> dict:
    episode_id = f"{suite}:seed61:task{task_id}:state1"
    gallery = read(RECIPE / "gallery.json")
    clip = next(
        v
        for v in gallery["videos"]
        if v["episode_id"] == episode_id and v["arm"] == "native"
    )
    assert clip["available"]
    source = RECIPE / clip["path"]
    published = read(RECIPE / "manifest.json")["files"][clip["path"]]
    assert sha(source) == clip["sha256"] == published["sha256"]
    row = next(
        r
        for r in read(RECIPE / "report.json")["rows"]
        if r["episode_id"] == episode_id and r["arm"] == "native"
    )
    assert row["instruction"] == instruction
    target = HERE / "starting_images" / f"{short_id.lower()}_start.png"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-y",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    validate_png(target)
    return {
        "task_id": short_id,
        "suite": suite,
        "instruction": instruction,
        "episode_id": episode_id,
        "seed": 61,
        "reset_index": 1,
        "observation_step": 0,
        "source_frame_index": 0,
        "source_arm": "native",
        "camera": "external",
        "image": target.relative_to(HERE).as_posix(),
        "image_sha256": sha(target),
        "image_size": [224, 224],
        "provenance": "First decoded frame of the original video, before the first policy action and after reset stabilization. Video compression is retained; no crop, annotation or image edit.",
        "source_video": source.relative_to(PROJECT).as_posix(),
        "source_video_sha256": clip["sha256"],
        "source_gallery_selection": clip["selection"],
        "source_summary_sha256": clip["summary_sha256"],
        "workflow": clip["workflow"],
    }


def extract_cabinet_start(instruction: str) -> dict:
    source = read(CABINET)
    decision = source["astra"]["astra_tei"]["decisions"][0]
    assert decision["step"] == 0 and decision["annotations"] == []
    evidence = decision["cameras"][CAMERA]
    target = HERE / "starting_images/s8_start.png"
    if (CASE / "events.jsonl").exists():
        assert (
            read(CASE / "summary.json")["episode_id"]
            == "libero_spatial_ood:seed19:task8:state0"
        )
        request = None
        with (CASE / "events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("sequence") == decision["request_event_sequence"]:
                    request = event["request"]
                    break
        assert request is not None
        assert request["request_fingerprint"] == decision["request_fingerprint"]
        assert request["observation_step"] == 0
        snapshot = request["observations"][-1]
        assert snapshot["step"] == 0
        encoded = snapshot["observation"][CAMERA]
        assert encoded["encoding"] == "base64_png"
        content = base64.b64decode(encoded["data"], validate=True)
        assert hashlib.sha256(content).hexdigest() == evidence["png_sha256"]
        target.write_bytes(content)
    # A portable rebuild can reuse the exact published PNG when the private
    # archive extraction is absent. Its bytes must match the original audit.
    assert target.exists() and sha(target) == evidence["png_sha256"]
    validate_png(target)
    return {
        "task_id": "S8",
        "suite": "libero_spatial_ood",
        "instruction": instruction,
        "episode_id": "libero_spatial_ood:seed19:task8:state0",
        "seed": 19,
        "reset_index": 0,
        "observation_step": 0,
        "source_arm": "astra_tei",
        "camera": "external",
        "image": target.relative_to(HERE).as_posix(),
        "image_sha256": sha(target),
        "image_size": [224, 224],
        "provenance": "Exact raw camera PNG from the audited phase-development request at action zero, before intervention or action execution; no image edit.",
        "source_example": CABINET.relative_to(PROJECT).as_posix(),
        "source_example_sha256": sha(CABINET),
        "source_archive_sha256": source["archive_sha256"],
        "source_request_fingerprint": decision["request_fingerprint"],
        "source_request_sequence": decision["request_event_sequence"],
        "source_camera_evidence": evidence,
    }


def make_contact_sheet(examples: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.8))
    fig.subplots_adjust(
        left=0.055, right=0.945, top=0.86, bottom=0.175, hspace=0.42, wspace=0.15
    )
    fig.suptitle(
        "LIBERO OOD: instructions and starting observations",
        fontsize=17,
        fontweight="bold",
        x=0.055,
        y=0.97,
        ha="left",
        color="#17283c",
    )
    fig.text(
        0.055,
        0.925,
        "Recorded external-camera observations at action 0",
        fontsize=12,
        color="#526277",
    )
    for ax, example in zip(axes.flat, examples, strict=True):
        ax.imshow(Image.open(HERE / example["image"]), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        suite = "Goal" if example["suite"] == "libero_goal_ood" else "Spatial"
        ax.set_title(
            f"{example['task_id']} · {suite} OOD",
            fontsize=11,
            fontweight="bold",
            color="#17283c",
            pad=9,
        )
        instruction = example["instruction"]
        if example["task_id"] == "S8":
            instruction = "put the bowl at table center\non the cabinet"
        ax.text(
            0.5,
            -0.055,
            instruction,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=11.5,
            color="#17283c",
        )
        offset = -0.205 if example["task_id"] == "S8" else -0.145
        ax.text(
            0.5,
            offset,
            f"seed {example['seed']} · reset {example['reset_index']} · action 0",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            color="#596779",
        )
    fig.text(
        0.055,
        0.035,
        "Illustrative recorded resets from two studies; these images add no evaluation trials.\nInstruction novelty follows the released benchmark; checkpoint training overlap is unknown.",
        fontsize=9,
        color="#596779",
        linespacing=1.6,
    )
    fig.savefig(HERE / "task_examples.png", dpi=180, facecolor="white")
    fig.savefig(
        HERE / "task_examples.pdf",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def publish_pages(tasks: list[dict], examples: list[dict]) -> None:
    intro = "The evaluated paper benchmark contains 20 released OOD instructions: ten Goal and ten Spatial tasks. IDs are zero-based release order. Wording is preserved exactly, including ‘bbq source’. These are benchmark compositions; their absence from all checkpoint training data has not been established."
    lines = [
        "# Task inventory and recorded starting images",
        "",
        intro,
        "",
        "[Standalone HTML](tasks.html) · [CSV](tasks.csv) · [Example sheet PNG](task_examples.png) · [Example sheet PDF](task_examples.pdf)",
        "",
        "| ID | Suite | Exact instruction |",
        "|---|---|---|",
    ]
    for task in tasks:
        lines.append(
            f"| {task['id']} | {task['suite_label']} | {task['instruction']} |"
        )
    lines += [
        "",
        "## Example starting observations",
        "",
        "The three seed-61 images are first decoded frames of native evaluation videos, after reset stabilization and before the first policy action. S8 is the exact raw action-zero PNG sent to Astra in the seed-19 phase-development study. Image content is unedited. The original camera observations are 224 × 224; the contact sheet enlarges them for display.",
        "",
        "Examples come from available recorded artifacts, including the existing outcome-selected video gallery; they are illustrations, not a representative sample or an additional evaluation cohort.",
        "",
        "![Four recorded starting observations](task_examples.png)",
        "",
    ]
    for example in examples:
        lines += [
            f"**{example['task_id']}: {example['instruction']}**",
            "",
            f"Seed {example['seed']}, reset {example['reset_index']}, external camera, action 0.",
            "",
            f"![{example['task_id']} starting observation]({example['image']})",
            "",
        ]
    lines += [
        "## Sources",
        "",
        "[Released task inventory](../frs_policy_improvement/coverage.json) · [Video provenance](../learned_correction_recipe/results/gallery.json) · [Cabinet image audit](../phase_interpolation/development/examples/examples.json) · [Image provenance](task_examples.json) · [File hashes](tasks_manifest.json)",
        "",
        "The sixteen additional generated compositions remain a [separate task catalog](../ood_extensions/v1/index.html), with policy evaluation pending. They are not included in this twenty-task table.",
        "",
        "Regenerate with `python astra_reversal/reports/smart_system2_results/build_task_examples.py` in the activated project environment. This uses existing video files and the audited cabinet PNG; no simulator, model or provider calls are made.",
        "",
    ]
    (HERE / "tasks.md").write_text("\n".join(lines))
    table = "".join(
        f"<tr><td>{t['id']}</td><td>{t['suite_label']}</td><td>{html.escape(t['instruction'])}</td></tr>"
        for t in tasks
    )
    cards = []
    for example in examples:
        encoded = base64.b64encode((HERE / example["image"]).read_bytes()).decode()
        cards.append(
            f"<figure><img width='224' height='224' src='data:image/png;base64,{encoded}' alt='{html.escape(example['instruction'], quote=True)}: starting observation'><figcaption><strong>{example['task_id']}: {html.escape(example['instruction'])}</strong><br>Seed {example['seed']} · reset {example['reset_index']} · action 0</figcaption></figure>"
        )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIBERO OOD task instructions and starting images</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#17283c;background:#fafbfc;max-width:1050px;margin:36px auto;padding:0 22px}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dce3eb}}th{{background:#edf2f6}}.examples{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}}figure{{margin:0;padding:18px;background:white;border:1px solid #dce3eb;border-radius:10px}}img{{width:100%;height:auto;image-rendering:pixelated}}figcaption{{padding-top:12px}}.note{{color:#526277}}@media(max-width:620px){{.examples{{grid-template-columns:1fr}}th,td{{padding:8px 6px;font-size:13px}}}}</style><h1>LIBERO OOD: tasks and starting images</h1><p>{html.escape(intro)}</p><h2>Recorded starting observations</h2><p class="note">External camera, action 0. G0, G5 and S2: first frames of seed-61 native evaluation videos. S8: raw PNG from a seed-19 development request. No image edits or new rollouts.</p><div class="examples">{''.join(cards)}</div><h2>All twenty evaluated task instructions</h2><table><thead><tr><th>ID</th><th>Suite</th><th>Exact instruction</th></tr></thead><tbody>{table}</tbody></table><p class="note">Examples are illustrations from available recordings, including the existing outcome-selected gallery. They are not a representative sample or an additional evaluation cohort. The sixteen newly generated task compositions are separate and have no policy evaluation results here.</p></html>\n"""
    (HERE / "tasks.html").write_text(page)


def main() -> None:
    (HERE / "starting_images").mkdir(exist_ok=True)
    coverage = read(COVERAGE)
    assert coverage["task_count"] == len(coverage["tasks"]) == 20
    tasks = []
    for task in sorted(coverage["tasks"], key=lambda t: (t["suite"], t["task_id"])):
        goal = task["suite"] == "libero_goal_ood"
        assert task["suite"] in {"libero_goal_ood", "libero_spatial_ood"}
        tasks.append(
            {
                "id": ("G" if goal else "S") + str(task["task_id"]),
                "suite": task["suite"],
                "suite_label": "Goal" if goal else "Spatial",
                "task_id": task["task_id"],
                "instruction": task["instruction"],
                "bddl_sha256": task["bddl_sha256"],
                "bddl_source": task["bddl_source"],
                "released_goal_expression": task["released_goal_expression"],
            }
        )
    assert {t["id"] for t in tasks} == {
        f"{prefix}{i}" for prefix in "GS" for i in range(10)
    }
    by_id = {t["id"]: t for t in tasks}
    examples = [
        extract_video_start(suite, task_id, short_id, by_id[short_id]["instruction"])
        for suite, task_id, short_id in SELECTIONS
    ]
    examples.append(extract_cabinet_start(by_id["S8"]["instruction"]))
    write_json(
        HERE / "tasks.json",
        {
            "schema_version": "smart-system2-tasks-1.0",
            "release": coverage["release"],
            "tasks": tasks,
        },
    )
    with (HERE / "tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    write_json(
        HERE / "task_examples.json",
        {
            "schema_version": "smart-system2-starting-images-1.0",
            "new_rollouts": 0,
            "images_edited": False,
            "examples": examples,
        },
    )
    make_contact_sheet(examples)
    publish_pages(tasks, examples)
    files = [
        "build_task_examples.py",
        "tasks.md",
        "tasks.html",
        "tasks.csv",
        "tasks.json",
        "task_examples.json",
        "task_examples.png",
        "task_examples.pdf",
    ] + [e["image"] for e in examples]
    manifest = {
        "schema_version": "smart-system2-task-publication-1.0",
        "task_count": len(tasks),
        "example_count": len(examples),
        "sources": {p.relative_to(PROJECT).as_posix(): sha(p) for p in SOURCE_FILES},
        "files": {
            name: {"sha256": sha(HERE / name), "bytes": (HERE / name).stat().st_size}
            for name in files
        },
    }
    write_json(HERE / "tasks_manifest.json", manifest)
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "recorded_starting_images": len(examples),
                "new_rollouts": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
