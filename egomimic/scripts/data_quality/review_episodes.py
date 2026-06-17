"""Interactive data-quality review tool for RLDB episodes.

Eyeball each episode's raw front-camera video in a small Flask web app and
accept / reject / skip it. Every decision is appended to a JSONL file the
instant it is made, so the review is fully resumable and the accepted set can
later be wired into a filtered data config.

This is a read-only consumer of the episode registry: it never writes to the
SQL table (``is_deleted`` is untouched) and never trains anything. It only
downloads the matched episodes' zarr stores and renders their ``images.front_1``
frames to MP4 for viewing.

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------
Activate the venv first: ``source emimic/bin/activate``

1) Sanity-check the SQL filter (no downloads, no server):

       python -m egomimic.scripts.data_quality.review_episodes --list-only

2) Review over an SSH port-forward (`ssh -L 5007:127.0.0.1:5007 <host>`):

       python -m egomimic.scripts.data_quality.review_episodes \
           --lab mecka --task folding_clothes \
           --cache-dir /path/to/qc_cache

   then open http://127.0.0.1:5007 in your local browser.
   Keys: [a]ccept  [r]eject  [s]kip  [<-]/[->] prev/next.

3) Export the accepted episode hashes as a ready-to-paste Python set:

       python -m egomimic.scripts.data_quality.review_episodes --print-accepted

   Paste the printed set into a copy of the data config, e.g.
   ``egomimic/hydra_configs/data/mecka_pi_fold_clothes_freeform.yaml``:

       filter_lambdas:
         - "lambda row: row['lab'] == 'mecka' and row['task'] == 'folding_clothes'"
         - "lambda row: row['episode_hash'] in {'<hash1>', '<hash2>', ...}"

   (The ``episode_hash in {...}`` pattern is already used in
   ``mecka_pi_eval.yaml``.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg  # noqa: F401  (ensures a bundled ffmpeg is importable)
import numpy as np
import simplejpeg
from flask import Flask, Response, abort, jsonify, request, send_file

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver, ZarrEpisode
from egomimic.utils.aws.aws_data_utils import load_env

# Primary egocentric RGB key for all human/mecka embodiments (see CONTRIBUTING_DATA.md).
IMAGE_KEY = "images.front_1"
BUCKET_NAME = "rldb"
VALID_DECISIONS = ("accept", "reject", "uncertain", "skip")

# Shared on-cluster store where episodes are already downloaded (= paths.dataset_dir
# in egomimic/hydra_configs/paths/default.yaml). Treated as read-only.
DEFAULT_DATASET_DIR = Path("/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets")


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------
def discover_episodes(
    lab: str, task: str, extra_filters: list[str], debug: int | None
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Query the SQL registry and return matched episodes.

    Returns:
        (ordered_hashes, hash_to_s3path, hash_to_task)
    """
    lambdas = [f"lambda row: row['lab'] == {lab!r} and row['task'] == {task!r}"]
    lambdas.extend(extra_filters or [])
    filters = DatasetFilter(filter_lambdas=lambdas)

    paths, hash_to_task = S3EpisodeResolver._get_filtered_paths(
        filters, debug=debug if debug else None
    )
    hash_to_s3 = {h: p for (p, h) in paths}
    ordered = sorted(hash_to_s3.keys())
    return ordered, hash_to_s3, hash_to_task


# ---------------------------------------------------------------------------
# Download + render
# ---------------------------------------------------------------------------
def resolve_episode_dir(
    dataset_dir: Path, cache_dir: Path, episode_hash: str, s3_path: str
) -> Path:
    """Return the local zarr store dir for an episode.

    Prefers the existing read-only shared store (``dataset_dir``); falls back to a
    writable ``cache_dir`` for any episode not already present, downloading it
    there only if needed.
    """
    shared = dataset_dir / episode_hash
    if shared.is_dir():
        return shared
    cached = cache_dir / episode_hash
    if not cached.is_dir():
        S3EpisodeResolver._sync_s3_to_local(
            bucket_name=BUCKET_NAME,
            s3_paths=[(s3_path, episode_hash)],
            local_dir=cache_dir,
        )
    if not cached.is_dir():
        raise FileNotFoundError(
            f"Episode {episode_hash} not in {dataset_dir} and download failed"
        )
    return cached


def effective_stride(total: int, stride: int, max_frames: int) -> int:
    """Frame stride that keeps the clip within ``max_frames`` while spanning the
    WHOLE episode. Uses ``stride`` for short episodes, but increases it for long
    ones so the video always covers start-to-end (no truncated tail)."""
    s = max(1, stride)
    if max_frames and total > 0:
        s = max(s, math.ceil(total / max_frames))
    return s


def _decode_json_entry(value):
    """Decode one zarr JSON-encoded entry (mirrors ZarrDataset._decode_json_entry)."""
    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def _episode_labels(ep: ZarrEpisode) -> list[dict]:
    """Decoded language annotations for an open episode, sorted by start frame.

    Each entry is ``{"text": str, "start_idx": int, "end_idx": int}``. Returns an
    empty list if the episode has no ``annotations`` array.
    """
    if "annotations" not in ep._collect_keys():
        return []
    raw = ep._store["annotations"][:]
    out: list[dict] = []
    for x in raw:
        try:
            d = _decode_json_entry(x)
        except Exception:
            continue
        if isinstance(d, dict):
            out.append(
                {
                    "text": str(d.get("text", "")),
                    "start_idx": int(d.get("start_idx", -1)),
                    "end_idx": int(d.get("end_idx", -1)),
                }
            )
    out.sort(key=lambda a: a["start_idx"])
    return out


def read_annotations(zarr_dir: Path) -> list[dict]:
    """Convenience wrapper: open an episode and return its language annotations."""
    return _episode_labels(ZarrEpisode(zarr_dir))


def _placeholder_mp4(out_path: Path, text: str, fps: int) -> None:
    """Write a tiny 1-second clip with an error message (unreadable episodes)."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        text[:60],
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    with imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        format="FFMPEG",
    ) as w:
        for _ in range(fps):
            w.append_data(frame)


def render_video(
    zarr_dir: Path,
    out_path: Path,
    fps: int | None,
    stride: int,
    max_frames: int,
    max_width: int,
    crf: int = 30,
    preset: str = "veryfast",
) -> None:
    """Decode ``images.front_1`` JPEGs from one episode into a browser-playable MP4.

    Tuned for fast review over an SSH tunnel: libjpeg decodes large frames at a
    reduced scale (``min_width``) so decode is cheap, x264 uses a fast ``preset``
    and ``crf`` for small files, and ``+faststart`` moves the moov atom to the
    front so the browser can begin playback before the file finishes downloading.
    """
    try:
        ep = ZarrEpisode(zarr_dir)
        total = len(ep)
        meta_fps = int(ep.metadata.get("fps", 30) or 30)
        out_fps = fps if fps else max(1, min(meta_fps, 30))

        if IMAGE_KEY not in ep._collect_keys():
            _placeholder_mp4(out_path, f"NO KEY {IMAGE_KEY}", out_fps)
            return

        blobs = ep.read({IMAGE_KEY: (0, total)})[IMAGE_KEY]
        # Stride chosen so the clip spans the whole episode within the frame budget.
        eff_stride = effective_stride(total, stride, max_frames)
        idxs = list(range(0, total, eff_stride))
        if not idxs:
            _placeholder_mp4(out_path, "EMPTY EPISODE", out_fps)
            return

        # libx264 output params: small files + fast encode + progressive playback.
        out_params = [
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
        ]
        tmp_path = out_path.with_suffix(".tmp.mp4")
        wrote_any = False
        with imageio.get_writer(
            str(tmp_path),
            fps=out_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=16,
            format="FFMPEG",
            output_params=out_params,
        ) as writer:
            for i in idxs:
                try:
                    # min_width lets libjpeg downscale in the DCT domain (cheap)
                    # for large source frames; fast flags trade a little quality
                    # for speed (fine for QC playback).
                    frame = simplejpeg.decode_jpeg(
                        bytes(blobs[i]),
                        colorspace="RGB",
                        fastdct=True,
                        fastupsample=True,
                        min_width=max_width,
                    )
                except Exception:
                    continue
                h, w = frame.shape[:2]
                if w > max_width:
                    new_w = max_width
                    new_h = int(round(h * (max_width / w)))
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                writer.append_data(np.ascontiguousarray(frame))
                wrote_any = True

        if not wrote_any:
            tmp_path.unlink(missing_ok=True)
            _placeholder_mp4(out_path, "ALL FRAMES UNREADABLE", out_fps)
            return
        os.replace(tmp_path, out_path)
    except Exception as exc:  # noqa: BLE001 - surface any failure as a viewable clip
        try:
            _placeholder_mp4(out_path, f"RENDER ERROR: {exc}", 10)
        except Exception:
            raise


# ---------------------------------------------------------------------------
# Decisions persistence (JSONL, last-line-wins per hash)
# ---------------------------------------------------------------------------
class DecisionLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.decisions: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h, d = rec.get("episode_hash"), rec.get("decision")
                if h and d in VALID_DECISIONS:
                    self.decisions[h] = d

    def record(
        self, episode_hash: str, decision: str, task: str, lab: str, index: int
    ) -> None:
        rec = {
            "episode_hash": episode_hash,
            "decision": decision,
            "task": task,
            "lab": lab,
            "index": index,
            "ts": time.time(),
        }
        line = json.dumps(rec)
        with self._lock:
            with self.path.open("a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.decisions[episode_hash] = decision

    def counts(self) -> dict[str, int]:
        c = {k: 0 for k in VALID_DECISIONS}
        for d in self.decisions.values():
            c[d] = c.get(d, 0) + 1
        return c


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Episode QC</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:10px 16px;background:#1b1b1b;border-bottom:1px solid #333;
        display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 .pill{padding:3px 9px;border-radius:12px;font-size:13px}
 .acc{background:#1d4d2b}.rej{background:#5a1d1d}.unc{background:#7a5a16}.skp{background:#444}
 #wrap{max-width:920px;margin:18px auto;padding:0 16px}
 video{width:100%;background:#000;border-radius:8px}
 .row{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
 button{font-size:15px;padding:10px 18px;border:0;border-radius:8px;cursor:pointer;color:#fff}
 .b-acc{background:#2e8b4f}.b-rej{background:#b03434}.b-unc{background:#c08a1e}
 .b-skip{background:#666}.b-nav{background:#345}.b-rev{background:#5a4ea3}
 .meta{font-size:13px;color:#aaa;margin-top:8px;word-break:break-all}
 .cur{font-weight:600}
 kbd{background:#333;border-radius:4px;padding:1px 5px;font-size:12px}
 #labels{margin-top:16px;background:#181818;border:1px solid #333;border-radius:8px;
         padding:10px 14px;font-size:14px;max-height:230px;overflow:auto}
 #labels ul{margin:6px 0 0;padding-left:18px}
 #labels li{margin:3px 0;line-height:1.35;padding:2px 6px;border-radius:5px;
            transition:background .1s}
 #labels li.active{background:#2e8b4f;color:#fff;font-weight:600}
 #labels li.active .span{color:#cde9d6}
 .span{color:#7fa8e0;font-family:monospace;font-size:12px;margin-right:6px}
</style></head>
<body>
<header>
  <span class="cur" id="progress"></span>
  <span class="pill acc" id="c-accept">accept 0</span>
  <span class="pill rej" id="c-reject">reject 0</span>
  <span class="pill unc" id="c-uncertain">uncertain 0</span>
  <span class="pill skp" id="c-skip">skip 0</span>
  <span style="font-size:12px;color:#888">
    <kbd>a</kbd> accept &nbsp; <kbd>r</kbd> reject &nbsp; <kbd>u</kbd> uncertain
    &nbsp; <kbd>s</kbd> skip &nbsp; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> nav</span>
</header>
<div id="wrap">
  <video id="vid" controls autoplay loop muted></video>
  <div class="meta">hash: <span id="hash"></span> &nbsp;|&nbsp; task: <span id="task"></span>
       &nbsp;|&nbsp; current decision: <span id="curdec">-</span></div>
  <div class="row">
    <button class="b-acc" onclick="decide('accept')">Accept (a)</button>
    <button class="b-rej" onclick="decide('reject')">Reject (r)</button>
    <button class="b-unc" onclick="decide('uncertain')">Uncertain (u)</button>
    <button class="b-skip" onclick="decide('skip')">Skip (s)</button>
    <button class="b-nav" onclick="go(idx-1)">&larr; Prev</button>
    <button class="b-nav" onclick="go(idx+1)">Next &rarr;</button>
    <button class="b-rev" onclick="nextUncertain()">Next uncertain &#8635;</button>
  </div>
  <div id="labels"></div>
</div>
<script>
let idx=0, total=0, episodes=[];
let curLabels=[], curOrigPerSec=1;
function setCounts(c){ for(const k of ['accept','reject','uncertain','skip'])
  document.getElementById('c-'+k).textContent = k+' '+(c[k]||0); }
async function loadState(){
  const s = await (await fetch('/state')).json();
  total = s.total; episodes = s.episodes;
  setCounts(s.counts);
  if (idx>=total) idx=total-1; if (idx<0) idx=0;
  render();
}
function render(){
  const e = episodes[idx];
  document.getElementById('progress').textContent =
     `Episode ${idx+1} / ${total}  (decided ${decidedCount()}, ${undecidedCount()} left)`;
  document.getElementById('hash').textContent = e.hash;
  document.getElementById('task').textContent = e.task;
  document.getElementById('curdec').textContent = e.decision || '-';
  const v = document.getElementById('vid');
  v.src = '/video/'+e.hash;  v.load();
  loadLabels(e.hash);
  prefetch(idx+1);
}
function escapeHtml(s){ return (s||'').replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function loadLabels(h){
  const el=document.getElementById('labels');
  el.innerHTML='<span style="color:#666">loading language labels…</span>';
  curLabels=[]; curOrigPerSec=1;
  try{
    const r=await (await fetch('/annotations/'+h)).json();
    if(h!==episodes[idx].hash) return;  // user advanced; ignore stale response
    const a=r.labels||[];
    curLabels=a; curOrigPerSec=r.orig_per_sec||1;
    if(!a.length){ el.innerHTML='<span style="color:#666">(no language labels)</span>'; return; }
    el.innerHTML='<b>Language labels ('+a.length+')</b><ul>'+a.map((x,i)=>
      `<li id="lab-${i}"><span class="span">[${x.start_idx}–${x.end_idx}]</span>${escapeHtml(x.text)}</li>`
    ).join('')+'</ul>';
    highlightLabel();
  }catch(e){ el.innerHTML='<span style="color:#a55">labels unavailable</span>'; }
}
function highlightLabel(){
  if(!curLabels.length) return;
  const v=document.getElementById('vid');
  const f=v.currentTime*curOrigPerSec;  // current ORIGINAL frame index
  let active=-1;
  for(let i=0;i<curLabels.length;i++){
    if(f>=curLabels[i].start_idx && f<curLabels[i].end_idx){ active=i; break; }
  }
  for(let i=0;i<curLabels.length;i++){
    const li=document.getElementById('lab-'+i); if(!li) continue;
    const on=(i===active); li.classList.toggle('active',on);
    if(on) li.scrollIntoView({block:'nearest'});
  }
}
function decidedCount(){ return episodes.filter(e=>e.decision).length; }
function undecidedCount(){ return episodes.filter(e=>!e.decision).length; }
async function decide(d){
  const e = episodes[idx];
  const r = await (await fetch('/decide',{method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify({hash:e.hash, decision:d, index:idx})})).json();
  e.decision = d;
  setCounts(r.counts);
  // After re-deciding while reviewing uncertains, hop to the next uncertain;
  // otherwise just advance to the next episode.
  if(d!=='uncertain' && episodes.some(x=>x.decision==='uncertain')
     && !episodes.slice(idx+1).some(x=>!x.decision)) nextUncertain();
  else go(idx+1);
}
function go(i){ if(i<0||i>=total) {render(); return;} idx=i; render(); }
function nextUncertain(){
  for(let k=1;k<=total;k++){ const i=(idx+k)%total;
    if(episodes[i].decision==='uncertain'){ idx=i; render(); return; } }
  alert('No uncertain episodes remaining.');
}
const PREFETCH_AHEAD=4;
function prefetch(start){ for(let k=0;k<PREFETCH_AHEAD;k++){const i=start+k;
  if(i>=0&&i<total) fetch('/prefetch/'+episodes[i].hash);} }
document.addEventListener('keydown',ev=>{
  if(ev.key==='a') decide('accept');
  else if(ev.key==='r') decide('reject');
  else if(ev.key==='u') decide('uncertain');
  else if(ev.key==='s') decide('skip');
  else if(ev.key==='ArrowLeft') go(idx-1);
  else if(ev.key==='ArrowRight') go(idx+1);
});
document.getElementById('vid').addEventListener('timeupdate', highlightLabel);
loadState().then(()=>{ // jump to first undecided
  const first = episodes.findIndex(e=>!e.decision);
  if(first>=0){ idx=first; render(); }
});
</script>
</body></html>
"""


def build_app(cfg, ordered, hash_to_s3, hash_to_task, log) -> Flask:
    app = Flask(__name__)
    render_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def _lock_for(h: str) -> threading.Lock:
        with locks_guard:
            return render_locks.setdefault(h, threading.Lock())

    def _ensure_video(h: str) -> Path:
        out = cfg.video_dir / f"{h}.mp4"
        if out.exists():
            return out
        with _lock_for(h):
            if out.exists():
                return out
            zarr_dir = resolve_episode_dir(
                cfg.dataset_dir, cfg.cache_dir, h, hash_to_s3[h]
            )
            render_video(
                zarr_dir,
                out,
                cfg.fps,
                cfg.stride,
                cfg.max_frames,
                cfg.max_width,
                crf=cfg.crf,
                preset=cfg.preset,
            )
        return out

    def _start_prewarm(n_workers: int) -> None:
        """Render undecided episodes ahead of time in the background so they're
        already cached (only the transfer cost remains) by the time the user
        reaches them. Bounded concurrency; skips already-rendered clips."""
        cursor = {"i": 0}
        cur_lock = threading.Lock()

        def worker():
            while True:
                with cur_lock:
                    i = cursor["i"]
                    cursor["i"] += 1
                if i >= len(ordered):
                    return
                h = ordered[i]
                if (cfg.video_dir / f"{h}.mp4").exists() or h in log.decisions:
                    continue
                try:
                    _ensure_video(h)
                except Exception:  # noqa: BLE001 - keep warming the rest
                    continue

        for _ in range(n_workers):
            threading.Thread(target=worker, daemon=True).start()

    if getattr(cfg, "prewarm", 0):
        _start_prewarm(cfg.prewarm)

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/state")
    def state():
        eps = [
            {
                "hash": h,
                "task": hash_to_task.get(h, cfg.task),
                "decision": log.decisions.get(h),
            }
            for h in ordered
        ]
        return jsonify(total=len(ordered), counts=log.counts(), episodes=eps)

    @app.get("/video/<h>")
    def video(h: str):
        if h not in hash_to_s3:
            abort(404)
        out = _ensure_video(h)
        return send_file(str(out), mimetype="video/mp4", conditional=True)

    ann_cache: dict[str, dict] = {}

    @app.get("/annotations/<h>")
    def annotations(h: str):
        if h not in hash_to_s3:
            abort(404)
        if h not in ann_cache:
            try:
                zarr_dir = resolve_episode_dir(
                    cfg.dataset_dir, cfg.cache_dir, h, hash_to_s3[h]
                )
                ep = ZarrEpisode(zarr_dir)
                meta_fps = int(ep.metadata.get("fps", 30) or 30)
                out_fps = cfg.fps if cfg.fps else max(1, min(meta_fps, 30))
                total = len(ep)
                eff_stride = effective_stride(total, cfg.stride, cfg.max_frames)
                # The video samples every `eff_stride`-th original frame and plays
                # at `out_fps`, so one second of video advances this many ORIGINAL
                # frames — used client-side to map currentTime -> annotation span.
                ann_cache[h] = {
                    "labels": _episode_labels(ep),
                    "orig_per_sec": out_fps * eff_stride,
                    "total_frames": total,
                }
            except Exception:  # noqa: BLE001 - missing/unreadable annotations -> empty
                ann_cache[h] = {"labels": [], "orig_per_sec": 1, "total_frames": 0}
        return jsonify(ann_cache[h])

    @app.get("/prefetch/<h>")
    def prefetch(h: str):
        if h in hash_to_s3:
            threading.Thread(target=_ensure_video, args=(h,), daemon=True).start()
        return ("", 204)

    @app.post("/decide")
    def decide():
        data = request.get_json(force=True)
        h, d = data.get("hash"), data.get("decision")
        if h not in hash_to_s3 or d not in VALID_DECISIONS:
            abort(400)
        log.record(
            h, d, hash_to_task.get(h, cfg.task), cfg.lab, int(data.get("index", -1))
        )
        return jsonify(ok=True, counts=log.counts())

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lab", default="mecka")
    p.add_argument("--task", default="folding_clothes")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Existing (read-only) shared store of downloaded episodes "
        "(= paths.dataset_dir). Episodes found here are never re-downloaded.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./qc_cache"),
        help="Writable fallback cache for episodes NOT in --dataset-dir.",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=Path("./qc_videos"),
        help="Local cache for rendered MP4s.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Decisions JSONL (default ./data_quality_review/<lab>_<task>_decisions.jsonl).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5007)
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Playback fps (default: episode fps, capped 30).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Use every Nth frame (higher = smaller/faster).",
    )
    p.add_argument("--max-frames", type=int, default=450, help="Cap frames per clip.")
    p.add_argument(
        "--max-width",
        type=int,
        default=480,
        help="Resize cap in px (lower = smaller/faster).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=30,
        help="x264 quality (higher = smaller file, 28-32 good for QC).",
    )
    p.add_argument(
        "--preset", default="veryfast", help="x264 speed preset (ultrafast..medium)."
    )
    p.add_argument(
        "--prewarm",
        type=int,
        default=2,
        help="Background workers that pre-render upcoming episodes so they "
        "load instantly. 0 disables. Use more CPU headroom (e.g. on a "
        "compute node) to raise this.",
    )
    p.add_argument(
        "--debug", type=int, default=None, help="Limit to first N matched episodes."
    )
    p.add_argument(
        "--extra-filter",
        action="append",
        default=[],
        help="Extra filter lambda string (repeatable).",
    )
    p.add_argument(
        "--list-only",
        action="store_true",
        help="Print matched episodes and exit (no downloads, no server).",
    )
    p.add_argument(
        "--print-accepted",
        action="store_true",
        help="Print accepted hashes from the decisions file as a Python set and exit.",
    )
    p.add_argument(
        "--print-uncertain",
        action="store_true",
        help="Print uncertain (flagged-to-revisit) hashes from the decisions file and exit.",
    )
    args = p.parse_args()
    if args.out is None:
        args.out = (
            Path("./data_quality_review") / f"{args.lab}_{args.task}_decisions.jsonl"
        )
    # Resolve to absolute paths: Flask's send_file interprets relative paths
    # against the app root_path (the package dir), not the process cwd.
    args.dataset_dir = args.dataset_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.video_dir = args.video_dir.resolve()
    args.out = args.out.resolve()
    return args


def main():
    cfg = parse_args()
    load_env()

    if cfg.print_accepted or cfg.print_uncertain:
        log = DecisionLog(cfg.out)
        want = "accept" if cfg.print_accepted else "uncertain"
        hashes = sorted(h for h, d in log.decisions.items() if d == want)
        print(f"# {len(hashes)} {want} episodes from {cfg.out}")
        inner = ", ".join(repr(h) for h in hashes)
        print("lambda row: row['episode_hash'] in {" + inner + "}")
        return

    ordered, hash_to_s3, hash_to_task = discover_episodes(
        cfg.lab, cfg.task, cfg.extra_filter, cfg.debug
    )
    print(f"Matched {len(ordered)} episodes for lab={cfg.lab!r} task={cfg.task!r}")

    if cfg.list_only:
        for h in ordered:
            print(f"{h}\t{hash_to_s3[h]}")
        return

    if not ordered:
        raise SystemExit("No episodes matched; nothing to review.")

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_dir.mkdir(parents=True, exist_ok=True)
    log = DecisionLog(cfg.out)

    present = sum(1 for h in ordered if (cfg.dataset_dir / h).is_dir())
    missing = len(ordered) - present
    print(
        f"Episodes already in {cfg.dataset_dir}: {present}/{len(ordered)}"
        + (
            f"  ({missing} will be fetched to {cfg.cache_dir} on demand)"
            if missing
            else ""
        )
    )
    done = sum(1 for h in ordered if h in log.decisions)
    print(f"Resuming: {done}/{len(ordered)} already decided. Decisions -> {cfg.out}")
    print(
        f"Open http://{cfg.host}:{cfg.port}  (SSH: ssh -L {cfg.port}:127.0.0.1:{cfg.port} <host>)"
    )

    app = build_app(cfg, ordered, hash_to_s3, hash_to_task, log)
    app.run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()
