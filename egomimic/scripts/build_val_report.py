"""
Build a self-contained HTML val-comparison report.

Layout:
    <REPORT_DIR>/
        index.html
        videos/{aria,mecka}/{HUMAN_BIMANUAL,EVA_BIMANUAL}/{tf,ar}/{predicted,validation}_{0..4}.mp4

The mp4s are SYMLINKED (not copied) from the latest offline-eval dirs of each
cotrain, so the report reflects whatever's on disk today. Download the whole
directory (``scp -r``) if you want a portable copy — mp4s follow the symlinks
on scp -L.

The HTML shows two videos side-by-side (aria on the left, mecka on the right)
with a single Play/Pause + seek bar + dropdowns for embodiment / mode / video
type / episode. Playback is time-synced: the smaller-currentTime video pulls
the other back if drift exceeds a threshold.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ARIA_ROOT = Path(
    "/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/aria_eva_cotrain_wam"
)
MECKA_ROOT = Path(
    "/storage/home/hcoda1/5/acheluva3/r-dxu345-0/EgoVerse/logs/mecka_eva_cotrain_wam"
)

# Latest offline longeva evals for each cotrain.
ARIA_TF = ARIA_ROOT / "aria_eva_cotrain_eval_tf_longeva_2026-08-04_15-57-11"
ARIA_AR = ARIA_ROOT / "aria_eva_cotrain_eval_ar_longeva_2026-08-04_16-13-20"
MECKA_TF = MECKA_ROOT / "mecka_eva_cotrain_eval_tf_longeva_2026-08-07_23-39-48"
MECKA_AR = MECKA_ROOT / "mecka_eva_cotrain_eval_ar_longeva_2026-08-07_23-39-48"

EMBODIMENTS = ["HUMAN_BIMANUAL", "EVA_BIMANUAL"]
MODES = ["tf", "ar"]
KINDS = ["predicted", "validation"]


def _link(src: Path, dst: Path) -> bool:
    """Create ``dst`` as a symlink to ``src``. Returns True iff src exists."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())
    return True


def _collect_videos(
    src_root: Path, model_slug: str, mode_slug: str, report_dir: Path
) -> dict[str, dict[str, list[str]]]:
    """Symlink one eval dir's mp4s into the report layout; return a nested
    ``{embodiment: {kind: [rel_path, ...]}}`` index (rel to report_dir)."""
    result: dict[str, dict[str, list[str]]] = {}
    videos_dir = src_root / "videos" / "epoch_0"
    for emb in EMBODIMENTS:
        result[emb] = {kind: [] for kind in KINDS}
        emb_dir = videos_dir / emb
        if not emb_dir.is_dir():
            continue
        for kind in KINDS:
            # Files are ``<kind>_video_<N>.mp4``.
            paths = sorted(
                emb_dir.glob(f"{kind}_video_*.mp4"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            for p in paths:
                idx = int(p.stem.rsplit("_", 1)[-1])
                rel_dst = (
                    Path("videos") / model_slug / emb / mode_slug / f"{kind}_{idx}.mp4"
                )
                dst = report_dir / rel_dst
                if _link(p, dst):
                    result[emb][kind].append(str(rel_dst))
    return result


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>WAM Cotrain Val Comparison — Aria vs Mecka</title>
<style>
:root {{
    --bg: #0f1218;
    --panel: #171b23;
    --border: #262b36;
    --text: #dfe5ef;
    --muted: #97a1b3;
    --accent: #4d9dff;
    --danger: #ff6b6b;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text);
}}
h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
.sub {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
.controls {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px;
    display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center;
    margin-bottom: 18px;
}}
.control {{ display: flex; align-items: center; gap: 8px; }}
.control label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
select, button {{
    background: #1f2530; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer;
}}
select:focus, button:focus {{ outline: 2px solid var(--accent); outline-offset: 0px; }}
button.primary {{
    background: var(--accent); border-color: var(--accent); color: #0b1017;
    font-weight: 600; padding: 8px 16px; min-width: 88px;
}}
button.primary:hover {{ filter: brightness(1.1); }}
.grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
}}
.card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden; display: flex; flex-direction: column;
}}
.card header {{
    padding: 10px 14px; background: #10141b;
    border-bottom: 1px solid var(--border); font-weight: 600; font-size: 14px;
}}
.card .badge {{
    display: inline-block; margin-left: 8px; padding: 2px 8px;
    background: #232935; color: var(--muted);
    border-radius: 4px; font-size: 11px; font-weight: 400;
}}
video {{
    width: 100%; aspect-ratio: 16 / 9; background: #000; display: block;
}}
.meta {{
    padding: 10px 14px; color: var(--muted); font-size: 12px;
    border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; gap: 8px;
}}
.timebar {{
    display: flex; align-items: center; gap: 10px; margin-top: 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px;
}}
input[type="range"] {{ flex: 1; accent-color: var(--accent); }}
.tstr {{ color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; min-width: 110px; text-align: right; }}
.error {{
    background: #2a1414; color: var(--danger); border: 1px solid #4c1c1c;
    border-radius: 6px; padding: 10px 14px; margin-top: 10px;
    font-size: 13px; display: none;
}}
</style>
</head>
<body>
<h1>WAM Cotrain Val Comparison</h1>
<div class="sub">
    Side-by-side offline eval videos from the two cotrain checkpoints
    (aria+eva vs mecka+eva). Latest longeva evals as of {timestamp}.
    Playback is time-synced across both panes.
</div>

<div class="controls">
    <div class="control">
        <label for="embodiment">Embodiment</label>
        <select id="embodiment">
            <option value="HUMAN_BIMANUAL">HUMAN_BIMANUAL</option>
            <option value="EVA_BIMANUAL">EVA_BIMANUAL</option>
        </select>
    </div>
    <div class="control">
        <label for="mode">Mode</label>
        <select id="mode">
            <option value="tf">TF (teacher-forced)</option>
            <option value="ar">AR (autoregressive)</option>
        </select>
    </div>
    <div class="control">
        <label for="kind">Video</label>
        <select id="kind">
            <option value="predicted">predicted</option>
            <option value="validation">validation</option>
        </select>
    </div>
    <div class="control">
        <label for="episode">Episode</label>
        <select id="episode"></select>
    </div>
    <button id="playpause" class="primary">▶ Play</button>
    <button id="reset">⟲ Reset</button>
</div>

<div id="error" class="error"></div>

<div class="grid">
    <div class="card">
        <header>Model: aria + eva cotrain <span class="badge" id="badge-left"></span></header>
        <video id="left" preload="metadata" playsinline muted></video>
        <div class="meta">
            <span id="path-left">–</span>
            <span id="time-left">0.00 / 0.00</span>
        </div>
    </div>
    <div class="card">
        <header>Model: mecka + eva cotrain <span class="badge" id="badge-right"></span></header>
        <video id="right" preload="metadata" playsinline muted></video>
        <div class="meta">
            <span id="path-right">–</span>
            <span id="time-right">0.00 / 0.00</span>
        </div>
    </div>
</div>

<div class="timebar">
    <span id="global-time" class="tstr">0.00s / 0.00s</span>
    <input type="range" id="seek" min="0" max="1" step="0.01" value="0">
    <span id="rate" class="tstr">1.00×</span>
</div>

<script>
const INDEX = {index_json};
const $ = (sel) => document.querySelector(sel);
const left = $("#left"), right = $("#right");
const emb = $("#embodiment"), mode = $("#mode"), kind = $("#kind"), ep = $("#episode");
const playbtn = $("#playpause"), reset = $("#reset");
const seek = $("#seek"), globalTime = $("#global-time");
const pathL = $("#path-left"), pathR = $("#path-right");
const timeL = $("#time-left"), timeR = $("#time-right");
const badgeL = $("#badge-left"), badgeR = $("#badge-right");
const errorBox = $("#error");
const SYNC_TOL = 0.15;  // seconds — realign when drift exceeds this

function pathFor(model) {{
    const paths = INDEX[model]?.[emb.value]?.[mode.value]?.[kind.value] ?? [];
    const idx = parseInt(ep.value, 10);
    return paths[idx] ?? null;
}}

function refreshEpisodeSelect() {{
    // Recompute episode dropdown to the min number of episodes available
    // across BOTH models for the current (emb, mode, kind).
    const nL = INDEX.aria?.[emb.value]?.[mode.value]?.[kind.value]?.length ?? 0;
    const nR = INDEX.mecka?.[emb.value]?.[mode.value]?.[kind.value]?.length ?? 0;
    const n = Math.min(nL, nR);
    const prev = parseInt(ep.value, 10) || 0;
    ep.innerHTML = "";
    for (let i = 0; i < n; i++) {{
        const opt = document.createElement("option");
        opt.value = i; opt.textContent = String(i);
        ep.appendChild(opt);
    }}
    if (n === 0) {{
        errorBox.style.display = "block";
        errorBox.textContent = `No videos for ${{emb.value}} / ${{mode.value}} / ${{kind.value}}`;
    }} else {{
        errorBox.style.display = "none";
    }}
    ep.value = Math.min(prev, Math.max(0, n - 1));
}}

function loadCurrent() {{
    const l = pathFor("aria"), r = pathFor("mecka");
    pathL.textContent = l ?? "(missing)";
    pathR.textContent = r ?? "(missing)";
    badgeL.textContent = `${{emb.value}} · ${{mode.value.toUpperCase()}} · ${{kind.value}} · ep ${{ep.value}}`;
    badgeR.textContent = `${{emb.value}} · ${{mode.value.toUpperCase()}} · ${{kind.value}} · ep ${{ep.value}}`;
    if (l) left.src = l; else left.removeAttribute("src");
    if (r) right.src = r; else right.removeAttribute("src");
    left.currentTime = 0; right.currentTime = 0;
    playbtn.textContent = "▶ Play";
    seek.value = 0;
}}

let isPlaying = false;
async function togglePlay() {{
    if (isPlaying) {{
        left.pause(); right.pause();
        playbtn.textContent = "▶ Play";
    }} else {{
        try {{ await Promise.all([left.play(), right.play()]); }}
        catch (e) {{ console.error(e); }}
        playbtn.textContent = "⏸ Pause";
    }}
    isPlaying = !isPlaying;
}}

function fmt(t) {{ return (isFinite(t) ? t : 0).toFixed(2) + "s"; }}

function updateReadouts() {{
    const dL = isFinite(left.duration) ? left.duration : 0;
    const dR = isFinite(right.duration) ? right.duration : 0;
    const cL = isFinite(left.currentTime) ? left.currentTime : 0;
    const cR = isFinite(right.currentTime) ? right.currentTime : 0;
    timeL.textContent = `${{fmt(cL)}} / ${{fmt(dL)}}`;
    timeR.textContent = `${{fmt(cR)}} / ${{fmt(dR)}}`;
    const dur = Math.max(dL, dR);
    const cur = Math.max(cL, cR);
    globalTime.textContent = `${{fmt(cur)}} / ${{fmt(dur)}}`;
    if (dur > 0) {{
        seek.max = dur.toFixed(2);
        seek.value = cur.toFixed(2);
    }}
}}

// Drift correction: whichever video is ahead pulls the other forward.
function syncDrift() {{
    if (!isPlaying) return;
    const cL = left.currentTime, cR = right.currentTime;
    if (Math.abs(cL - cR) > SYNC_TOL) {{
        const target = Math.max(cL, cR);
        // Nudge the lagging one forward; skip if the video isn't seekable yet.
        if (cL < target && left.readyState >= 2) left.currentTime = target;
        if (cR < target && right.readyState >= 2) right.currentTime = target;
    }}
}}

left.addEventListener("timeupdate", () => {{ syncDrift(); updateReadouts(); }});
right.addEventListener("timeupdate", () => {{ syncDrift(); updateReadouts(); }});
left.addEventListener("loadedmetadata", updateReadouts);
right.addEventListener("loadedmetadata", updateReadouts);
left.addEventListener("ended", () => {{ isPlaying = false; playbtn.textContent = "▶ Play"; }});

seek.addEventListener("input", () => {{
    const t = parseFloat(seek.value);
    left.currentTime = t; right.currentTime = t;
    updateReadouts();
}});

playbtn.addEventListener("click", togglePlay);
reset.addEventListener("click", () => {{
    left.pause(); right.pause();
    left.currentTime = 0; right.currentTime = 0;
    isPlaying = false; playbtn.textContent = "▶ Play";
    updateReadouts();
}});
for (const el of [emb, mode, kind]) {{
    el.addEventListener("change", () => {{ refreshEpisodeSelect(); loadCurrent(); }});
}}
ep.addEventListener("change", loadCurrent);

refreshEpisodeSelect();
loadCurrent();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/storage/project/r-dxu345-0/acheluva3/EgoVerse/logs/val_comparison_report",
        help="Output report directory (default: logs/val_comparison_report)",
    )
    args = ap.parse_args()
    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {
        "aria": {
            emb: {
                "tf": {"predicted": [], "validation": []},
                "ar": {"predicted": [], "validation": []},
            }
            for emb in EMBODIMENTS
        },
        "mecka": {
            emb: {
                "tf": {"predicted": [], "validation": []},
                "ar": {"predicted": [], "validation": []},
            }
            for emb in EMBODIMENTS
        },
    }

    for model_slug, mode_slug, src_root in [
        ("aria", "tf", ARIA_TF),
        ("aria", "ar", ARIA_AR),
        ("mecka", "tf", MECKA_TF),
        ("mecka", "ar", MECKA_AR),
    ]:
        collected = _collect_videos(src_root, model_slug, mode_slug, report_dir)
        for emb, by_kind in collected.items():
            for kind, rel_paths in by_kind.items():
                index[model_slug][emb][mode_slug][kind] = rel_paths

    # Summary print + write HTML
    for model, by_emb in index.items():
        for emb, by_mode in by_emb.items():
            for mode, by_kind in by_mode.items():
                counts = {k: len(v) for k, v in by_kind.items()}
                print(f"  {model}.{emb}.{mode}: {counts}")

    from datetime import datetime

    html_out = HTML_TEMPLATE.format(
        index_json=json.dumps(index, indent=None),
        timestamp=html.escape(datetime.now().strftime("%Y-%m-%d %H:%M")),
    )
    (report_dir / "index.html").write_text(html_out, encoding="utf-8")
    print(f"\nReport written to: {report_dir / 'index.html'}")
    print(f"Symlinked mp4 tree under: {report_dir / 'videos'}")


if __name__ == "__main__":
    main()
