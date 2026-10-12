"""Export an offline dashboard from archived Astra hardware-interface trials.

The input directories stay private and unchanged. This exports recorded tools,
messages and pixels, validates event chains and archive hashes, and never calls
a model or simulator. Opaque encrypted reasoning is replaced by its digest.
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ASSETS = Path(__file__).with_name("process_dashboard")
COHORTS = {
    "libero-hardware-interface-20261008-uncapped-3": {
        "title": "LIBERO · fresh expansion",
        "benchmark": "LIBERO",
        "stage": "pilot",
        "planned": 150,
        "commit": "9c0a45faa77a4df86dee9e1a384cf3576b26d7c7",
        "note": "Completed, archived trials from the fresh expansion. A snapshot of an ongoing pilot; historical cohorts are not pooled.",
    },
    "robocasa-hardware-interface-20261009-pilot-5": {
        "title": "RoboCasa · commissioning smokes",
        "benchmark": "RoboCasa",
        "stage": "smoke",
        "planned": 3,
        "commit": "13d1c95bfb004d1d48c07b2672c5f4e3a46a863f",
        "note": "Short live integration checks. These are not full-horizon benchmark results and are excluded from LIBERO comparisons.",
    },
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, separators=(",", ":")))


def validate_chain(rows):
    previous = "0" * 64
    for index, row in enumerate(rows):
        if row["sequence"] != index or row["previous_sha256"] != previous:
            raise ValueError(f"Broken event chain at {index}")
        unsigned = {k: v for k, v in row.items() if k != "sha256"}
        if sha(canonical(unsigned)) != row["sha256"]:
            raise ValueError(f"Event digest mismatch at {index}")
        previous = row["sha256"]
    if not rows or rows[-1]["event"] != "trial_end":
        raise ValueError("Only complete, terminated trial recordings can be exported")
    return previous


def validate_archive(directory):
    receipt = json.loads((directory / "archive-receipt.json").read_text())
    files = receipt.get("files", [])
    verified = []
    for item in files:
        relative = item.get("path", item.get("relative"))
        path = directory / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe archive path")
        if not path.exists():
            continue
        raw = path.read_bytes()
        if sha(raw) != item["sha256"]:
            raise ValueError(f"Archive digest mismatch: {path}")
        verified.append(relative)
    if "events.jsonl" not in verified:
        raise ValueError("Archive receipt does not authenticate the event log")
    return {"prefix": receipt["prefix"], "verified_files": verified}


def camera_name(path):
    value = path.lower()
    if "wrist" in value or "eye_in_hand" in value:
        return "Wrist"
    if "left" in value:
        return "Front left"
    if "right" in value:
        return "Front right"
    return "Front"


def export_value(value, output, images, *, path="result", step=0):
    """Preserve JSON values, externalize exact PNG bytes and opaque ciphertext."""
    if isinstance(value, dict):
        label = value.get("channel", path)
        sample_step = value.get("metadata", {}).get(
            "simulator_step", value.get("simulator_step", step)
        )
        if value.get("encoding") == "image/png" and "base64" in value:
            raw = base64.b64decode(value["base64"], validate=True)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Expected PNG camera bytes")
            digest = sha(raw)
            relative = "media/" + digest + ".png"
            target = output / relative
            if not target.exists():
                target.write_bytes(raw)
            images.append(
                {
                    "camera": camera_name(path),
                    "path": relative,
                    "step": sample_step,
                    "sha256": digest,
                }
            )
            exported = {k: v for k, v in value.items() if k != "base64"}
            exported.update({"file": relative, "png_sha256": digest})
            return exported
        result = {}
        for key, item in value.items():
            if key == "encrypted_content" and isinstance(item, str):
                result[key] = {
                    "opaque_payload_omitted": True,
                    "sha256": sha(item.encode()),
                    "bytes": len(item.encode()),
                }
            else:
                result[key] = export_value(
                    item, output, images, path=str(label) + "/" + key, step=sample_step
                )
        return result
    if isinstance(value, list):
        return [
            export_value(v, output, images, path=path + "/" + str(i), step=step)
            for i, v in enumerate(value)
        ]
    return value


def summarize_event(row):
    kind = row["event"]
    if kind == "tool_request":
        return (
            row["tool"]
            + "("
            + json.dumps(row["arguments"], separators=(",", ":"))[:180]
            + ")"
        )
    if kind == "tool_response":
        response = row["response"]
        if isinstance(response, dict) and (
            response.get("accepted") is False or "error" in response
        ):
            return (
                row["tool"]
                + " · "
                + str(response.get("rejection_reason", response.get("error")))
            )
        return row["tool"] + " · recorded result"
    if kind == "model_response":
        parts = []
        for item in row["response"].get("output", []):
            if item.get("type") == "function_call":
                parts.append("calls " + item["name"])
            elif item.get("type") == "message":
                parts.extend(
                    c.get("text", "")[:180]
                    for c in item.get("content", [])
                    if c.get("type") == "output_text"
                )
        return (row.get("role", "actor") + " · " + "; ".join(parts))[:240]
    if kind == "action_rejected":
        return "Command rejected · " + row["rejection_reason"]
    if kind == "error":
        return str(row.get("reason", row.get("error_class", "error"))) + (
            " · API retry scheduled" if row.get("will_retry") else ""
        )
    if kind == "action_applied":
        return (
            "Native step "
            + str(row["simulator_step_after"])
            + " · "
            + json.dumps(row["applied_action"])
        )
    if kind == "trial_end":
        return row["terminal_reason"] + " · " + str(row["sim_steps"]) + " native steps"
    return kind.replace("_", " ")


def make_video(case, output):
    from PIL import Image, ImageDraw, ImageFont

    frames = [
        event
        for event in case["events"]
        if event["images"] and event["kind"] == "tool_response"
    ]
    if not frames:
        return None
    cameras = sorted({im["camera"] for event in frames for im in event["images"]})
    width, height = max(800, len(cameras) * 310), 432
    destination = output / "videos" / (case["id"] + ".mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "3",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = (
        ImageFont.truetype(font_path, 17)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    small = (
        ImageFont.truetype(font_path, 13)
        if Path(font_path).exists()
        else ImageFont.load_default()
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    latest = {}
    positions = []
    for index, event in enumerate(frames):
        for im in event["images"]:
            latest[im["camera"]] = im
        canvas = Image.new("RGB", (width, height), "#10212d")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 16),
            f"{case['condition']}  /  {case['trial_id']}  /  native step {event['step']}",
            font=font,
            fill="white",
        )
        draw.text(
            (20, 44),
            f"Recorded observation {index + 1}/{len(frames)} · wall +{event['time']:.1f}s · event #{event['sequence']}",
            font=small,
            fill="#bbcdd8",
        )
        cell = width // len(cameras)
        for position, camera in enumerate(cameras):
            x = position * cell + (cell - 288) // 2
            im = latest.get(camera)
            label = camera + (
                f" · sampled at step {im['step']}" if im else " · not yet observed"
            )
            draw.text((x, 78), label, font=small, fill="#bbcdd8")
            if im:
                with Image.open(output / im["path"]) as source:
                    pixels = source.convert("RGB").resize(
                        (288, 288), Image.Resampling.NEAREST
                    )
                canvas.paste(pixels, (x, 105))
        draw.text(
            (20, 409),
            "Saved observations · 3 snapshots/s · inference pauses condensed · no scene interpolation",
            font=small,
            fill="#bbcdd8",
        )
        process.stdin.write(canvas.tobytes())
        positions.append(
            {"seconds": index / 3, "sequence": event["sequence"], "step": event["step"]}
        )
    process.stdin.close()
    errors = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError("Video encoding failed: " + errors)
    return {
        "path": str(destination.relative_to(output)),
        "frames": len(frames),
        "fps": 3,
        "index": positions,
        "sha256": sha(destination.read_bytes()),
    }


def build_case(directory, output):
    workflow = directory.parent.parent.name
    raw = (directory / "events.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    last_hash = validate_chain(rows)
    archive = validate_archive(directory)
    start = next(row for row in rows if row["event"] == "trial_start")
    outcome = json.loads((directory / "outcome.json").read_text())
    replay = json.loads((directory / "independent-replay.json").read_text())
    trial = json.loads((directory / "trial.json").read_text())
    case_id = (
        "libero-" if workflow.startswith("libero-") else "robocasa-"
    ) + directory.name
    case = {
        "id": case_id,
        "trial_id": directory.name,
        "workflow": workflow,
        "cohort": workflow,
        "condition": outcome["condition"],
        "instruction": start["instruction"],
        "outcome": outcome,
        "trial": trial,
        "replay": replay,
        "started": start["utc"],
        "ended": rows[-1]["utc"],
        "task": str(outcome["task_id"]),
        "state": outcome.get("init_state_index", outcome.get("env_seed")),
        "events": [],
        "requests": {},
        "schemas": {},
        "markers": [],
        "prompts": {
            "actor": start["system_prompt"],
            "actor_sha256": start["system_prompt_sha256"],
            "observer": None,
            "task": start["instruction"],
        },
        "provenance": {
            "event_log_sha256": sha(raw),
            "last_event_sha256": last_hash,
            "chain_verified": True,
            "event_count": len(rows),
            "archive": archive,
            "source_view_sha256": start.get("source_sha256"),
            "source_commit": COHORTS[workflow]["commit"],
            "image_export": "Original PNG bytes; matched by SHA-256. JSON base64 fields externalized as file references.",
            "opaque_reasoning": "Encrypted reasoning payloads are replaced by byte count and SHA-256; readable private reasoning is not available.",
        },
    }
    step, last_gripper, previous_rejected = 0, None, False
    counts = Counter()
    examples = {}
    for row in rows:
        kind = row["event"]
        counts[kind] += 1
        step = row.get("simulator_step_after", row.get("simulator_step", step))
        images = []
        fields = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "sequence",
                "event",
                "utc",
                "monotonic_offset",
                "trial_id",
                "condition",
                "task_id",
                "init_state_index",
                "init_state_hash",
                "env_seed",
                "split",
                "replicate",
            }
        }
        detail = export_value(fields, output, images, step=step)
        if kind == "model_request":
            request = detail["request"]
            for field, target in (("instructions", "requests"), ("tools", "schemas")):
                content = request[field]
                key = sha(canonical(content))
                case[target][key] = content
                request[field] = {
                    "dashboard_reference": target + "/" + key,
                    "content_sha256": key,
                }
            if row.get("role") == "observer":
                case["prompts"]["observer"] = row["request"]["instructions"]
                case["prompts"]["observer_format"] = row["request"].get("text")
            elif "tools" not in case["prompts"]:
                case["prompts"]["tools"] = row["request"]["tools"]
        event = {
            "sequence": row["sequence"],
            "kind": kind,
            "role": row.get("role"),
            "tool": row.get("tool"),
            "step": step,
            "time": row["monotonic_offset"],
            "utc": row["utc"],
            "summary": summarize_event(row),
            "detail": detail,
            "images": images,
        }
        case["events"].append(event)
        marker = None
        if kind == "tool_request":
            tool = row["tool"]
            if tool not in examples:
                examples[tool] = {
                    "sequence": row["sequence"],
                    "arguments": detail["arguments"],
                }
            if tool == "describe_visible":
                marker = ("observer", "Request visual evidence")
            elif tool in {"step", "act"}:
                args = row["arguments"]
                values = args.get("action", args.get("envelope", {}).get("value", []))
                if values and COHORTS[workflow]["benchmark"] == "LIBERO":
                    gripper = 1 if values[-1] > 0 else -1 if values[-1] < 0 else 0
                    if last_gripper is not None and last_gripper != gripper:
                        marker = (
                            "gripper",
                            f"Gripper command changes {last_gripper:+d} → {gripper:+d}",
                        )
                    last_gripper = gripper
        elif kind == "tool_response":
            response = row["response"]
            rejected = isinstance(response, dict) and (
                response.get("accepted") is False or "error" in response
            )
            if rejected:
                marker = (
                    "rejection",
                    "Tool rejected: "
                    + str(response.get("rejection_reason", response.get("error"))),
                )
            elif previous_rejected:
                marker = ("recovery", "Next tool call accepted after rejection")
            previous_rejected = rejected
            if row["tool"] in examples and "response" not in examples[row["tool"]]:
                examples[row["tool"]]["response"] = detail["response"]
        elif kind == "error":
            marker = (
                "api_retry" if row.get("will_retry") else "error",
                str(row.get("reason", "Recorded error")),
            )
        elif kind == "trial_end":
            marker = ("terminal", outcome["terminal_reason"])
        if marker:
            case["markers"].append(
                {
                    "sequence": row["sequence"],
                    "step": step,
                    "time": row["monotonic_offset"],
                    "kind": marker[0],
                    "label": marker[1],
                }
            )
    case["counts"], case["examples"] = dict(counts), examples
    if "\nDevice description: " in case["prompts"]["actor"]:
        case["device"] = json.loads(
            case["prompts"]["actor"].split("\nDevice description: ", 1)[1]
        )
    case["observer_input_note"] = (
        "Observer requests in this cohort store input hashes. Descriptions are recorded exactly; observer-only camera payloads are not reconstructed. Actor camera reads remain available."
    )
    case["evaluator_images"] = []
    for path in directory.glob("*.png"):
        destination = output / "media" / (sha(path.read_bytes()) + ".png")
        if not destination.exists():
            shutil.copyfile(path, destination)
        case["evaluator_images"].append(
            {"name": path.name, "path": str(destination.relative_to(output))}
        )
    case["video"] = make_video(case, output)
    destination = output / "recordings" / (case_id + ".json")
    write_json(destination, case)
    (output / "recordings" / (case_id + ".js")).write_text(
        "window.ASTRA_RECORDINGS = window.ASTRA_RECORDINGS || {}; window.ASTRA_RECORDINGS["
        + json.dumps(case_id)
        + "] = "
        + destination.read_text()
        + ";\n"
    )
    summary = {
        k: case[k]
        for k in (
            "id",
            "trial_id",
            "workflow",
            "cohort",
            "condition",
            "instruction",
            "task",
            "state",
            "started",
            "ended",
            "outcome",
            "replay",
            "counts",
            "markers",
            "provenance",
        )
    }
    summary["video"] = (
        {k: v for k, v in case["video"].items() if k != "index"}
        if case["video"]
        else None
    )
    summary["recording"] = "recordings/" + case_id + ".json"
    summary["thumbnail"] = next(
        (e["images"][0]["path"] for e in case["events"] if e["images"]), None
    )
    return summary


def build(source, output, catalog_time):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("media", "videos", "recordings"):
        (output / name).mkdir()
    for path in ASSETS.iterdir():
        if path.is_file():
            shutil.copyfile(path, output / path.name)
    cases = []
    for directory in sorted(source.glob("*/trials/*")):
        if (
            directory.parent.parent.name not in COHORTS
            or not (directory / "independent-replay.json").exists()
        ):
            continue
        if directory.parent.parent.name.startswith(
            "robocasa-"
        ) and not directory.name.startswith("smoke-"):
            continue
        case = build_case(directory, output)
        cases.append(case)
        print(
            json.dumps(
                {
                    "exported": case["id"],
                    "events": case["provenance"]["event_count"],
                    "video_frames": case["video"]["frames"] if case["video"] else 0,
                }
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_utc": datetime.fromtimestamp(catalog_time, timezone.utc).isoformat(),
        "cohorts": COHORTS,
        "cases": cases,
        "recording_note": "Exact recorded prompts, emitted tool calls, tool results and original camera PNGs. Videos condense saved observations at 3 snapshots/s; they are not continuous simulator capture. Encrypted reasoning is opaque and omitted with a digest. Input-hashed requests retain their recorded hashes; no missing content is invented.",
        "retry_note": "A gripper command change is an observable marker, not proof of a new grasp attempt. Tool correction markers mean the next tool call was accepted, not that the task recovered. Provider retries, when present, are separately marked. No episode-reset tool is available to the actor.",
    }
    write_json(output / "manifest.json", manifest)
    (output / "manifest.js").write_text(
        "window.ASTRA_DASHBOARD = " + (output / "manifest.json").read_text() + ";\n"
    )
    (output / "START-HERE.txt").write_text(
        "ASTRA PROCESS RECORDINGS\n\nUnzip the entire folder, then open index.html. Internet and a local server are not required.\n\n"
        + manifest["recording_note"]
        + "\n\n"
        + manifest["retry_note"]
        + "\n\nSnapshot: "
        + manifest["catalog_utc"]
        + "\nLIBERO and RoboCasa smoke trials are kept separate. This is a snapshot, not a live monitor.\n"
    )
    receipt = {
        str(p.relative_to(output)): sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(output / "SHA256SUMS.json", receipt)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, output.name + "/" + str(path.relative_to(output)))
    print(
        json.dumps(
            {
                "dashboard": str(output / "index.html"),
                "zip": str(archive),
                "MB": round(archive.stat().st_size / 1e6, 1),
                "trials": len(cases),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.download_receipt.read_text())
    build(args.source, args.output, receipt["catalog_time"])
