"""Render accepted stored actions into an HTML video gallery and contact sheet."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

from ..pushshapes.agents import CONTROL_GAPS
from ..pushshapes.env import PushShapesEnv
from .articulation_quality import interaction
from .articulation_status import EMBODIMENTS


def render_one(task):
    path, out = map(Path, task)
    store = zarr.open_group(path, mode="r")
    attrs = dict(store.attrs)
    quality = attrs["quality"]
    emb, gap = quality["embodiment"], attrs["control_gap"]
    actions = store["actions"][:]
    held = store["engaged"][:].ravel()
    first = int(np.flatnonzero(held)[0]) + 1
    indices = [
        0,
        first,
        first + (len(actions) - first) // 3,
        first + 2 * (len(actions) - first) // 3,
        len(actions),
    ]
    env = PushShapesEnv(pusher_shape=emb, image_size=512, render_mode="rgb_array")
    env.agent.control_gap = CONTROL_GAPS[gap]
    env._skip_obs_render = True
    env.reset(seed=attrs["reset_seed"])
    font = ImageFont.load_default(size=18)
    output = out / (emb + ".mp4")
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
        "512x560",
        "-framerate",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    panels = {}
    with (out / (emb + ".render.log")).open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index in range(len(actions) + 1):
                if index:
                    env.step(actions[index - 1].astype(np.float64))
                if index % 4 and index not in indices:
                    continue
                rgb = env.render()
                frame = Image.new("RGB", (512, 560), "#102025")
                frame.paste(Image.fromarray(rgb), (0, 48))
                draw = ImageDraw.Draw(frame)
                draw.text(
                    (12, 6),
                    f'{emb} | {gap} | seed {attrs["reset_seed"]}',
                    fill="white",
                    font=font,
                )
                draw.text(
                    (12, 27),
                    f"Frame {index}/{len(actions)} | coverage {env._coverage():.3f} | interaction {int(interaction(env))}",
                    fill="#a8efcc",
                    font=font,
                )
                if index in indices:
                    panels[index] = frame.copy().resize((256, 280))
                if index % 4 == 0 or index == len(actions):
                    process.stdin.write(frame.tobytes())
            for _ in range(45):
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            code = process.wait()
            env.close()
        if code:
            raise RuntimeError(f"ffmpeg failed; see {emb}.render.log")
    row = Image.new("RGB", (1280, 280))
    for column, index in enumerate(indices):
        row.paste(panels[index], (column * 256, 0))
    row.save(out / (emb + ".png"))
    result = dict(
        embodiment=emb,
        gap=gap,
        episode=str(path),
        seed=attrs["reset_seed"],
        frames=len(actions),
        coverage=quality["final_coverage"],
        jerk_speed=quality["jerk_speed"],
        interaction_steps=quality["engaged_steps"],
        carry_distance=quality["carry_distance"],
    )
    (out / (emb + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for emb in EMBODIMENTS:
        candidates = []
        for collection in (
            "audit_sample",
            "pilot_all_modes",
            "pilot_suction",
            "pilot_contact",
        ):
            candidates = sorted(
                (a.run_root / collection / "ideal" / emb).glob("shard*/episode_*.zarr")
            )
            if candidates:
                break
        if not candidates:
            raise RuntimeError("No accepted ideal episode for " + emb)
        tasks.append((str(candidates[0]), str(a.out)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(render_one, tasks))
    sheet = Image.new("RGB", (1280, 280 * len(rows)))
    cards = []
    for index, row in enumerate(rows):
        emb = row["embodiment"]
        sheet.paste(Image.open(a.out / (emb + ".png")), (0, 280 * index))
        cards.append(
            f'<article><h2>{html.escape(emb)}</h2><video controls muted playsinline preload="metadata" src="{emb}.mp4"></video>'
            f'<p>Coverage {row["coverage"]:.3f} · jerk/speed {row["jerk_speed"]:.3f}<br>'
            f'{row["interaction_steps"]} interaction frames · seed {row["seed"]}</p></article>'
        )
    sheet.save(a.out / "contact_sheet.png")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Articulated demonstrations</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{background:#102025;color:#eaf3ef;font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px}h1{font-size:32px}p{line-height:1.5;color:#b7cec2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}article{background:#1c3237;border-radius:12px;padding:16px}h2{font-size:20px;margin:0 0 12px}video{width:100%;border-radius:6px}a{color:#a8efcc}</style>
<h1>Articulated demonstrations</h1><p>One accepted ideal-mode episode per embodiment, replayed from its stored actions. Videos play at 4× simulation speed. The overlay reports measured interaction and goal coverage. Grasping tools use attachment constraints; the four contact tools use their physical contact mechanisms.</p>
<p><a href="contact_sheet.png">Open the full contact sheet</a></p><div class="grid">"""
        + "".join(cards)
        + "</div></html>"
    )
    (a.out / "index.html").write_text(page)
    (a.out / "previews.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(a.out / "index.html")


if __name__ == "__main__":
    main()
