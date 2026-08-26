"""Dataset Browser — a third inspector view.

Scrub any episode in a dataset *folder* (searchable by filename or by
annotation text), play it back as a timeline, overlay the recorded actions
(cartesian trajectory / orientation axes / keypoints), and toggle the
language annotations on/off.

Reuses the inspector's dark theme (`views`), zarr image IO (`images`), and
annotation parser (`language`). Frames are rendered server-side — the
`/dataset_frame/<episode>/<frame>` Flask route decodes the JPEG, draws the
requested overlay with `egomimic.utils.viz_utils`, optionally burns in the
active annotation, and returns a JPEG. The browser just points an <img> at
that URL, so scrubbing the slider / flipping the overlay only swaps a src
string (the heavy work is cached server-side).

Overlay rendering is wrapped defensively: if an episode's action/keypoint
layout or camera calibration doesn't match, the route falls back to the
clean frame plus a small "overlay unavailable" badge rather than erroring —
so browsing + annotations always work even on datasets the projection
can't handle yet.
"""

from __future__ import annotations

import io as _io
import logging
import os
import threading
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from .images import (
    _bytes_from_zarr_element,
    _candidate_image_keys,
    _resolve_zarr_path,
    open_zarr_for_hash,
)
from .language import annotation_intervals, interval_for_frame
from .views import ACCENT, BORDER, CANVAS, CARD_STYLE, LABEL_STYLE, MUTED, PANEL, TEXT

logger = logging.getLogger(__name__)

# ---- render cache + prefetch (so scrubbing/playback don't re-render) -------
# Rendered JPEG bytes keyed by (root, ep, frame, overlay, annotate, intr, extr).
# Bounded LRU; a couple of fully-warmed episodes fit comfortably.
_RENDER_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_RENDER_CACHE_MAX = 4000
_RENDER_LOCK = threading.Lock()
# Episodes (per overlay/annot variant) currently being warmed in a bg thread.
_WARMING: set = set()
_WARM_LOCK = threading.Lock()
# Cap the served frame's long side — full-res decode for overlay accuracy, then
# downsize for fast encode + small transfer over the SSH tunnel.
MAX_RENDER_SIDE = 720

# Overlay modes offered in the UI. Applicability is detected per-episode;
# unsupported ones still render (clean frame + badge) rather than vanish.
OVERLAY_OPTIONS = [
    {"label": " None", "value": "none"},
    {"label": " Cartesian (xyz path)", "value": "cartesian"},
    {"label": " Orientation (rot axes)", "value": "orientation"},
    {"label": " Keypoints", "value": "keypoint"},
]

# Default camera calibration keys — match egomimic/scripts/data_visualization.py
# (intrinsics_key="base", extrinsics_key="x5Nov18_3"). Overridable per-app.
DEFAULT_INTRINSICS_KEY = "base"
DEFAULT_EXTRINSICS_KEY = "x5Dec13_2"


# ====================================================================== #
# Episode discovery + lightweight metadata
# ====================================================================== #
def _is_zarr_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        path.endswith(".zarr")
        or os.path.exists(os.path.join(path, "zarr.json"))  # zarr v3
        or os.path.exists(os.path.join(path, ".zgroup"))  # zarr v2
    )


@lru_cache(maxsize=16)
def list_episodes(dataset_root: str) -> tuple[str, ...]:
    """Sorted episode directory names under `dataset_root` (cached)."""
    try:
        entries = sorted(os.listdir(dataset_root))
    except OSError as e:
        logger.warning("Cannot list dataset_root %s: %s", dataset_root, e)
        return tuple()
    return tuple(
        e for e in entries if _is_zarr_dir(os.path.join(dataset_root, e))
    )


def _frame_count(grp, image_key: str) -> int:
    for key in _candidate_image_keys(grp, image_key):
        arr = _resolve_zarr_path(grp, key)
        if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 1:
            return int(arr.shape[0])
    return 0


def _resolve_image_key(grp, image_key: str) -> str | None:
    for key in _candidate_image_keys(grp, image_key):
        arr = _resolve_zarr_path(grp, key)
        if (
            arr is not None
            and hasattr(arr, "shape")
            and len(arr.shape) >= 1
            and arr.shape[0] > 0
        ):
            return key
    return None


@lru_cache(maxsize=2048)
def episode_meta(dataset_root: str, episode: str, image_key: str) -> dict:
    """(cached) Per-episode summary: frame count, which overlays its arrays
    support, and how many annotation intervals it has."""
    grp = open_zarr_for_hash(dataset_root, episode)
    if grp is None:
        return {"frames": 0, "overlays": ["none"], "n_annot": 0, "img_key": None}
    try:
        keys = set(_all_array_keys(grp))
    except Exception:
        keys = set()
    overlays = ["none"]
    has_pose = any(k.endswith("obs_ee_pose") for k in keys) or "actions" in keys
    has_kp = any("keypoint" in k for k in keys)
    if has_pose:
        overlays += ["cartesian", "orientation"]
    if has_kp:
        overlays += ["keypoint"]
    n_annot = len(annotation_intervals(dataset_root, episode))
    return {
        "frames": _frame_count(grp, image_key),
        "overlays": overlays,
        "n_annot": n_annot,
        "img_key": _resolve_image_key(grp, image_key),
    }


def _all_array_keys(grp, prefix: str = "", depth: int = 0):
    """Yield leaf array key paths (dot-or-slash names as stored)."""
    if depth > 3:
        return
    try:
        items = list(grp.keys())
    except Exception:
        return
    for k in items:
        try:
            sub = grp[k]
        except Exception:
            continue
        path = f"{prefix}{k}"
        if hasattr(sub, "shape"):
            yield path
        else:
            yield from _all_array_keys(sub, prefix=f"{path}.", depth=depth + 1)


# ====================================================================== #
# Annotation-text search index (episode -> concatenated lowercase text)
# ====================================================================== #
@lru_cache(maxsize=16)
def _annotation_text_index(dataset_root: str) -> tuple[tuple[str, str], ...]:
    """(episode, lowercased-joined-annotation-text) for every episode.
    Cached; built lazily on first annotation-search."""
    out: list[tuple[str, str]] = []
    for ep in list_episodes(dataset_root):
        intervals = annotation_intervals(dataset_root, ep)
        joined = " ".join(i[2] for i in intervals) if intervals else ""
        out.append((ep, joined))
    return tuple(out)


def search_episodes(dataset_root: str, query: str) -> list[str]:
    """Episodes matching `query` by filename substring OR annotation text.
    Empty query returns all."""
    eps = list_episodes(dataset_root)
    q = (query or "").strip().lower()
    if not q:
        return list(eps)
    name_hits = [e for e in eps if q in e.lower()]
    text_hits = [e for e, txt in _annotation_text_index(dataset_root) if q in txt]
    # preserve discovery order, dedup
    seen, out = set(), []
    for e in eps:
        if (e in name_hits or e in text_hits) and e not in seen:
            seen.add(e)
            out.append(e)
    return out


# ====================================================================== #
# Frame decode + overlay rendering (server-side)
# ====================================================================== #
def _decode_frame_rgb(grp, img_key: str, frame: int):
    arr = _resolve_zarr_path(grp, img_key)
    if arr is None or not hasattr(arr, "shape") or arr.shape[0] == 0:
        return None
    frame = max(0, min(int(frame), int(arr.shape[0]) - 1))
    try:
        raw = _bytes_from_zarr_element(arr[frame])
        from PIL import Image

        return np.asarray(Image.open(_io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        logger.debug("frame decode failed (%s,%d): %s", img_key, frame, e)
        return None


def _read_arm_array(grp, suffix: str, frame: int | None = None):
    """Return {'left': arr, 'right': arr} for keys like 'left.<suffix>'.
    If `frame` is given, returns that row; else the full (T, D) array."""
    out = {}
    for arm in ("left", "right"):
        a = _resolve_zarr_path(grp, f"{arm}.{suffix}")
        if a is None:
            continue
        try:
            out[arm] = a[frame] if frame is not None else a[:]
        except Exception:
            continue
    return out


def _badge(img_rgb, text: str):
    """Stamp a small bottom-left badge onto an RGB frame (cv2)."""
    try:
        import cv2

        h = img_rgb.shape[0]
        fs = max(0.4, h / 1000)
        th = max(1, int(h / 500))
        y = img_rgb.shape[0] - max(8, int(h * 0.02))
        cv2.putText(img_rgb, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img_rgb, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 210, 90), th, cv2.LINE_AA)
    except Exception:
        pass
    return img_rgb


def _draw_overlay(img_rgb, grp, frame: int, overlay: str,
                  intrinsics_key: str, extrinsics_key: str, horizon: int = 16):
    """Best-effort overlay. Returns (rgb, ok, note). On any failure returns
    the clean frame with ok=False and a short note, never raises."""
    if overlay in (None, "none"):
        return img_rgb, True, ""
    try:
        from egomimic.utils.egomimicUtils import (
            CameraTransforms,
            cam_frame_to_cam_pixels,
            ee_pose_to_cam_frame,
        )
    except Exception as e:  # egomimic projection deps unavailable
        return _badge(img_rgb.copy(), f"overlay {overlay}: deps n/a"), False, str(e)

    try:
        ct = CameraTransforms(intrinsics_key=intrinsics_key,
                              extrinsics_key=extrinsics_key)
        intr = ct.intrinsics
    except Exception as e:
        return _badge(img_rgb.copy(), f"overlay {overlay}: calib?"), False, str(e)

    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    def _to_px(pts_xyz, arm):
        """base/robot-frame xyz (N,3) -> pixel (N,2). Mirrors rollout.py's
        draw path: ee_pose_to_cam_frame(xyz, extrinsics[arm]) then project via
        the camera intrinsics."""
        pts = np.asarray(pts_xyz, dtype=float).reshape(-1, 3)
        try:
            cam = ee_pose_to_cam_frame(pts, ct.extrinsics[arm])
        except Exception:
            cam = pts
        px = cam_frame_to_cam_pixels(cam, intr)
        return np.asarray(px)[:, :2]

    try:
        import cv2

        if overlay == "cartesian":
            ee = _read_arm_array(grp, "obs_ee_pose")
            drew = False
            for arm, color in (("left", (80, 200, 255)), ("right", (255, 160, 80))):
                seq = ee.get(arm)
                if seq is None:
                    continue
                seq = np.asarray(seq)
                lo, hi = max(0, frame), min(seq.shape[0], frame + horizon)
                xyz = seq[lo:hi, :3]
                if xyz.shape[0] < 2:
                    continue
                px = _to_px(xyz, arm)
                pts = np.round(px).astype(np.int32)
                for i in range(1, len(pts)):
                    if (0 <= pts[i - 1, 0] < w and 0 <= pts[i - 1, 1] < h
                            and 0 <= pts[i, 0] < w and 0 <= pts[i, 1] < h):
                        cv2.line(vis, tuple(pts[i - 1]), tuple(pts[i]), color, 2,
                                 cv2.LINE_AA)
                        drew = True
                if 0 <= pts[0, 0] < w and 0 <= pts[0, 1] < h:
                    cv2.circle(vis, tuple(pts[0]), 4, color, -1)
            if not drew:
                return _badge(img_rgb.copy(), "cartesian: no in-frame pts"), False, "off"
            return vis, True, ""

        if overlay == "orientation":
            ee = _read_arm_array(grp, "obs_ee_pose", frame=frame)
            drew = False
            for arm, anchor in (("left", (255, 180, 80)), ("right", (80, 180, 255))):
                pose = ee.get(arm)
                if pose is None:
                    continue
                pose = np.asarray(pose, dtype=float).reshape(-1)
                xyz = pose[:3]
                rot = _rot_from_pose(pose[3:])
                if rot is None:
                    continue
                L = 0.05
                axpts = np.vstack([xyz, xyz + rot[:, 0] * L,
                                   xyz + rot[:, 1] * L, xyz + rot[:, 2] * L])
                px = np.round(_to_px(axpts, arm)).astype(np.int32)
                x0, y0 = px[0]
                if not (0 <= x0 < w and 0 <= y0 < h):
                    continue
                for i, col in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)], 1):
                    x1, y1 = px[i]
                    if 0 <= x1 < w and 0 <= y1 < h:
                        cv2.line(vis, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
                cv2.circle(vis, (x0, y0), 4, anchor, -1)
                drew = True
            if not drew:
                return _badge(img_rgb.copy(), "orientation: no in-frame pts"), False, "off"
            return vis, True, ""

        if overlay == "keypoint":
            drew = False
            for arm, col in (("left", (0, 120, 255)), ("right", (255, 80, 0))):
                a = (_resolve_zarr_path(grp, f"{arm}.obs_keypoints")
                     or _resolve_zarr_path(grp, f"{arm}.obs_aria_keypoints"))
                if a is None:
                    continue
                try:
                    kp = np.asarray(a[frame], dtype=float).reshape(-1, 3)
                except Exception:
                    continue
                px = np.round(_to_px(kp, arm)).astype(np.int32)
                for (x, y) in px:
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(vis, (int(x), int(y)), 3, col, -1)
                        cv2.circle(vis, (int(x), int(y)), 3, (255, 255, 255), 1)
                        drew = True
            if not drew:
                return _badge(img_rgb.copy(), "keypoint: no in-frame pts"), False, "off"
            return vis, True, ""

    except Exception as e:
        logger.debug("overlay %s failed: %s", overlay, e)
        return _badge(img_rgb.copy(), f"overlay {overlay}: error"), False, str(e)

    return img_rgb, True, ""


def _rot_from_pose(rot_part):
    """(3,) euler ZYX or (4,) quat -> 3x3 rotation matrix, or None."""
    try:
        from scipy.spatial.transform import Rotation as R

        rot_part = np.asarray(rot_part, dtype=float).reshape(-1)
        if rot_part.shape[0] >= 4:
            return R.from_quat(rot_part[:4]).as_matrix()
        if rot_part.shape[0] == 3:
            return R.from_euler("ZYX", rot_part, degrees=False).as_matrix()
    except Exception:
        return None
    return None


def _annotate_frame(img_rgb, dataset_root: str, episode: str, frame: int):
    """Burn the active annotation interval's text onto the frame."""
    intervals = annotation_intervals(dataset_root, episode)
    match = interval_for_frame(intervals, int(frame))
    if not match:
        return img_rgb
    try:
        from egomimic.utils.viz_utils import _viz_annotations

        return _viz_annotations(img_rgb, [match[2]])
    except Exception:
        return img_rgb


def render_frame_jpeg(dataset_root: str, episode: str, frame: int, *,
                      overlay: str, annotate: bool, image_key: str,
                      intrinsics_key: str, extrinsics_key: str) -> bytes | None:
    key = (dataset_root, episode, int(frame), overlay, bool(annotate),
           intrinsics_key, extrinsics_key)
    with _RENDER_LOCK:
        hit = _RENDER_CACHE.get(key)
        if hit is not None:
            _RENDER_CACHE.move_to_end(key)
            return hit
    grp = open_zarr_for_hash(dataset_root, episode)
    if grp is None:
        return None
    img_key = _resolve_image_key(grp, image_key)
    if img_key is None:
        return None
    rgb = _decode_frame_rgb(grp, img_key, frame)
    if rgb is None:
        return None
    rgb, _ok, _note = _draw_overlay(rgb, grp, int(frame), overlay,
                                    intrinsics_key, extrinsics_key)
    if annotate:
        rgb = _annotate_frame(rgb, dataset_root, episode, int(frame))
    try:
        from PIL import Image

        img = Image.fromarray(rgb.astype(np.uint8))
        if max(img.size) > MAX_RENDER_SIDE:
            img.thumbnail((MAX_RENDER_SIDE, MAX_RENDER_SIDE), Image.BILINEAR)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = buf.getvalue()
    except Exception:
        return None
    with _RENDER_LOCK:
        _RENDER_CACHE[key] = data
        _RENDER_CACHE.move_to_end(key)
        while len(_RENDER_CACHE) > _RENDER_CACHE_MAX:
            _RENDER_CACHE.popitem(last=False)
    return data


def warm_episode(dataset_root: str, episode: str, n_frames: int, *,
                 overlay: str, annotate: bool, image_key: str,
                 intrinsics_key: str, extrinsics_key: str) -> None:
    """Pre-render every frame of `episode` (this overlay/annot variant) into the
    cache on a background daemon thread, so subsequent scrub/playback are cache
    hits. One thread per (episode, overlay, annotate) variant."""
    if not episode or n_frames <= 0:
        return
    wkey = (dataset_root, episode, overlay, bool(annotate),
            intrinsics_key, extrinsics_key)
    with _WARM_LOCK:
        if wkey in _WARMING:
            return
        _WARMING.add(wkey)

    def _run():
        try:
            from concurrent.futures import ThreadPoolExecutor

            def _one(f):
                render_frame_jpeg(
                    dataset_root, episode, f, overlay=overlay,
                    annotate=annotate, image_key=image_key,
                    intrinsics_key=intrinsics_key, extrinsics_key=extrinsics_key)

            # parallel decode (JPEG decode releases the GIL) — warms a ~700-frame
            # episode in ~1 min instead of ~3.5 min single-threaded.
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_one, range(int(n_frames))))
        except Exception as e:
            logger.debug("warm_episode %s failed: %s", episode, e)
        finally:
            with _WARM_LOCK:
                _WARMING.discard(wkey)

    threading.Thread(target=_run, daemon=True).start()


# ====================================================================== #
# The Dash view
# ====================================================================== #
class DatasetView:
    """Episode browser + player + action overlays + annotation toggle."""

    def __init__(self, app, dataset_root: str, image_key: str,
                 lang_key: str | None = None,
                 intrinsics_key: str = DEFAULT_INTRINSICS_KEY,
                 extrinsics_key: str = DEFAULT_EXTRINSICS_KEY):
        self.app = app
        self.dataset_root = dataset_root
        self.image_key = image_key
        self.lang_key = lang_key
        self.intrinsics_key = intrinsics_key
        self.extrinsics_key = extrinsics_key

    # ---- sidebar controls (shown only in dataset mode) ----------------
    def sidebar_card(self, visible: bool = True):
        from dash import dcc, html

        eps = list_episodes(self.dataset_root)
        opts = [{"label": e, "value": e} for e in eps]
        return html.Div(
            id="dataset_controls",
            style={**CARD_STYLE, "display": ("block" if visible else "none")},
            children=[
                html.Div("Dataset folder", style=LABEL_STYLE),
                html.Div(self.dataset_root, title=self.dataset_root,
                         style={"fontSize": "10px", "color": MUTED,
                                "fontFamily": "ui-monospace, monospace",
                                "wordBreak": "break-all", "marginBottom": "8px"}),
                html.Div("Search (filename or annotation)", style=LABEL_STYLE),
                dcc.Input(id="ds_search", type="text", value="", debounce=True,
                          placeholder="e.g. pick, 2026-04, fold …",
                          style={"width": "100%", "padding": "6px 8px",
                                 "fontSize": "13px", "border": f"1px solid {BORDER}",
                                 "borderRadius": "4px", "boxSizing": "border-box",
                                 "marginBottom": "10px"}),
                html.Div("Episode", style=LABEL_STYLE),
                dcc.Dropdown(id="ds_episode", options=opts,
                             value=(eps[0] if eps else None), clearable=False,
                             style={"fontSize": "12px"}),
                html.Div(f"{len(eps)} episodes", id="ds_episode_count",
                         style={"fontSize": "10px", "color": MUTED,
                                "marginTop": "4px", "marginBottom": "10px"}),
                html.Div("Action overlay", style=LABEL_STYLE),
                dcc.RadioItems(id="ds_overlay", options=OVERLAY_OPTIONS,
                               value="cartesian",
                               labelStyle={"display": "block", "fontSize": "13px",
                                           "marginBottom": "3px", "cursor": "pointer"}),
                html.Div(style={"marginTop": "10px"}, children=[
                    dcc.Checklist(id="ds_annot",
                                  options=[{"label": " Show annotations", "value": "on"}],
                                  value=["on"],
                                  labelStyle={"fontSize": "13px", "cursor": "pointer"}),
                ]),
                html.Div("Playback fps", style={**LABEL_STYLE, "marginTop": "10px"}),
                dcc.Slider(id="ds_fps", min=2, max=30, step=2, value=10,
                           marks={2: "2", 10: "10", 30: "30"}),
            ],
        )

    # ---- main panel (the player) --------------------------------------
    def panel(self, visible: bool = True):
        from dash import dcc, html

        return html.Div(
            id="dataset_view",
            style={"display": ("flex" if visible else "none"),
                   "flexDirection": "row", "flex": "1", "minWidth": "0"},
            children=[
                # left: video + timeline
                html.Div(style={"flex": "1", "padding": "16px", "minWidth": "0",
                                "display": "flex", "flexDirection": "column"},
                         children=[
                    html.Div(id="ds_title", style={"fontSize": "13px",
                             "fontWeight": 600, "color": TEXT, "marginBottom": "8px"}),
                    dcc.Loading(type="circle", color=ACCENT, children=html.Img(
                        id="ds_frame_img",
                        style={"maxWidth": "100%", "maxHeight": "calc(100vh - 230px)",
                               "borderRadius": "6px", "border": f"1px solid {BORDER}",
                               "background": "#000", "objectFit": "contain"})),
                    html.Div(style={"display": "flex", "alignItems": "center",
                                    "gap": "12px", "marginTop": "12px"}, children=[
                        html.Button("▶", id="ds_play", n_clicks=0,
                                    style={"padding": "6px 14px", "fontSize": "14px",
                                           "fontWeight": 600, "background": ACCENT,
                                           "color": "white", "border": "none",
                                           "borderRadius": "6px", "cursor": "pointer"}),
                        html.Div(dcc.Slider(id="ds_frame", min=0, max=0, step=1, value=0,
                                            marks=None, updatemode="drag",
                                            tooltip={"placement": "bottom",
                                                     "always_visible": False}),
                                 style={"flex": "1"}),
                        html.Div(id="ds_frame_label", style={"fontSize": "12px",
                                 "color": MUTED, "minWidth": "92px",
                                 "textAlign": "right",
                                 "fontFamily": "ui-monospace, monospace"}),
                    ]),
                    dcc.Interval(id="ds_interval", interval=100, disabled=True),
                    dcc.Store(id="ds_playing", data=False),
                    dcc.Store(id="ds_nframes", data=0),
                    dcc.Store(id="ds_ann_store", data=[]),
                    dcc.Store(id="ds_warm_sink", data=0),
                    dcc.Store(id="ds_preload_sink", data=0),
                ]),
                # right: annotation + metadata
                html.Div(style={"width": "340px", "padding": "16px",
                                "borderLeft": f"1px solid {BORDER}", "background": PANEL,
                                "overflowY": "auto"}, children=[
                    html.Div("Annotation", style=LABEL_STYLE),
                    html.Pre(id="ds_annot_text", style={"whiteSpace": "pre-wrap",
                             "background": "#f1f5f9", "padding": "10px",
                             "fontSize": "12px", "borderRadius": "4px",
                             "border": f"1px solid {BORDER}", "marginBottom": "16px",
                             "minHeight": "44px"}),
                    html.Div("Episode info", style=LABEL_STYLE),
                    html.Pre(id="ds_meta", style={"background": "#f1f5f9",
                             "padding": "10px", "fontSize": "12px",
                             "borderRadius": "4px", "border": f"1px solid {BORDER}",
                             "fontFamily": "ui-monospace, monospace",
                             "whiteSpace": "pre-wrap"}),
                ]),
            ],
        )

    # ---- Flask frame route --------------------------------------------
    def register_route(self):
        @self.app.server.route("/dataset_frame/<episode>/<frame>")
        def _dataset_frame_route(episode: str, frame: str):
            from flask import Response, abort, request

            try:
                frame_i = int(frame)
            except ValueError:
                return abort(404)
            overlay = request.args.get("overlay", "none")
            annotate = request.args.get("annot", "0") == "1"
            data = render_frame_jpeg(
                self.dataset_root, episode, frame_i, overlay=overlay,
                annotate=annotate, image_key=self.image_key,
                intrinsics_key=self.intrinsics_key,
                extrinsics_key=self.extrinsics_key)
            if data is None:
                return abort(404)
            return Response(data, mimetype="image/jpeg",
                            headers={"Cache-Control": "public, max-age=3600"})

    def _src(self, episode, frame, overlay, annotate):
        if not episode:
            return ""
        a = "1" if annotate else "0"
        return f"/dataset_frame/{episode}/{int(frame)}?overlay={overlay}&annot={a}"

    # ---- callbacks -----------------------------------------------------
    def register(self):
        import dash
        from dash import Input, Output, State

        app = self.app
        self.register_route()

        # search -> filtered episode options
        @app.callback(
            Output("ds_episode", "options"),
            Output("ds_episode", "value"),
            Output("ds_episode_count", "children"),
            Input("ds_search", "value"),
            State("ds_episode", "value"),
        )
        def _filter_eps(query, cur):
            eps = search_episodes(self.dataset_root, query or "")
            opts = [{"label": e, "value": e} for e in eps]
            value = cur if cur in eps else (eps[0] if eps else None)
            return opts, value, f"{len(eps)} episodes"

        # episode selected -> slider range + meta + annotation store + bg warm
        @app.callback(
            Output("ds_frame", "max"),
            Output("ds_frame", "value"),
            Output("ds_nframes", "data"),
            Output("ds_meta", "children"),
            Output("ds_title", "children"),
            Output("ds_ann_store", "data"),
            Input("ds_episode", "value"),
            State("ds_overlay", "value"),
            State("ds_annot", "value"),
        )
        def _load_episode(episode, overlay, annot_val):
            if not episode:
                return 0, 0, 0, "", "", []
            m = episode_meta(self.dataset_root, episode, self.image_key)
            n = m["frames"]
            meta = (f"frames     {n}\n"
                    f"overlays   {', '.join(o for o in m['overlays'] if o != 'none') or '—'}\n"
                    f"annots     {m['n_annot']} interval(s)\n"
                    f"image_key  {m['img_key']}")
            intervals = [[iv[0], iv[1], iv[3]]
                         for iv in annotation_intervals(self.dataset_root, episode)]
            # NB: no server-side bulk warm here — the clientside preloader below
            # warms BOTH the server render-cache and the browser cache as it
            # fetches, and starts from the current frame. A separate server-side
            # 4-worker warm just floods the server and starves the one fetch the
            # user is actually waiting on (esp. on overlay change).
            return max(0, n - 1), 0, n, meta, episode, intervals

        # frame/overlay/annot -> image src + label + annotation text, CLIENTSIDE
        # (no server round-trip per frame: the browser builds the route URL and
        #  looks up the annotation from the store, so scrub/playback are smooth).
        app.clientside_callback(
            """
            function(frame, overlay, annotVal, episode, intervals, nframes) {
                if (!episode) { return ["", "", ""]; }
                frame = frame || 0;
                var annotate = annotVal && annotVal.indexOf("on") >= 0;
                var src = "/dataset_frame/" + episode + "/" + frame +
                          "?overlay=" + (overlay || "none") +
                          "&annot=" + (annotate ? "1" : "0");
                var label = frame + " / " + Math.max(0, (nframes || 1) - 1);
                var txt = "(no annotation at this frame)";
                if (intervals) {
                    for (var i = 0; i < intervals.length; i++) {
                        var iv = intervals[i];
                        if (iv[0] <= frame && frame < iv[1]) {
                            txt = iv[2] + "\\n[" + iv[0] + "\\u2013" + iv[1] + "]";
                            break;
                        }
                    }
                }
                return [src, label, txt];
            }
            """,
            Output("ds_frame_img", "src"),
            Output("ds_frame_label", "children"),
            Output("ds_annot_text", "children"),
            Input("ds_frame", "value"),
            Input("ds_overlay", "value"),
            Input("ds_annot", "value"),
            Input("ds_episode", "value"),
            Input("ds_ann_store", "data"),
            Input("ds_nframes", "data"),
        )

        # CLIENTSIDE preloader: when the episode/overlay/annot changes, fetch
        # every frame URL into the browser's HTTP cache (batched). Without this,
        # rapidly swapping the <img> src during playback/scrub makes the browser
        # cancel each in-flight load before it paints, so the frame appears to
        # freeze until motion stops. Once cached, each src-swap is an instant
        # cache hit and the image updates on every frame change. A token cancels
        # a superseded preload when the user switches episode/overlay.
        app.clientside_callback(
            """
            function(episode, overlay, annotVal, nframes, curFrame) {
                if (!episode || !nframes) { return 0; }
                var annotate = annotVal && annotVal.indexOf("on") >= 0;
                var ov = overlay || "none";
                var token = episode + "|" + ov + "|" + (annotate ? 1 : 0);
                window._dsTok = token;
                var n = nframes, start = curFrame || 0;
                // visit order: current frame FIRST, then forward, wrapping —
                // so the frame the user is looking at caches first and an
                // overlay switch updates immediately.
                var order = [];
                for (var k = 0; k < n; k++) { order.push((start + k) % n); }
                var idx = 0;
                function step() {
                    if (window._dsTok !== token) { return; }  // superseded -> stop
                    var b = 0;
                    // gentle batches (4 / 80ms) so the preloader never saturates
                    // the browser's connection pool and starve the visible frame.
                    while (idx < order.length && b < 4) {
                        var im = new Image();
                        im.src = "/dataset_frame/" + episode + "/" + order[idx] +
                                 "?overlay=" + ov + "&annot=" + (annotate ? "1" : "0");
                        idx++; b++;
                    }
                    if (idx < order.length) { setTimeout(step, 80); }
                }
                step();
                return 0;
            }
            """,
            Output("ds_preload_sink", "data"),
            Input("ds_episode", "value"),
            Input("ds_overlay", "value"),
            Input("ds_annot", "value"),
            Input("ds_nframes", "data"),
            State("ds_frame", "value"),
        )

        # play/pause toggle (server: just flips the interval on/off).
        # prevent_initial_call so it does NOT auto-fire (and auto-play) on load.
        @app.callback(
            Output("ds_interval", "disabled"),
            Output("ds_interval", "interval"),
            Output("ds_play", "children"),
            Output("ds_playing", "data"),
            Input("ds_play", "n_clicks"),
            Input("ds_fps", "value"),
            State("ds_playing", "data"),
            prevent_initial_call=True,
        )
        def _toggle_play(n_clicks, fps, playing):
            ctx = dash.callback_context
            trig = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
            interval_ms = int(1000 / max(1, int(fps or 10)))
            # fps change while playing: keep current play state, just retune rate
            if trig.startswith("ds_fps"):
                return (not playing), interval_ms, ("❚❚" if playing else "▶"), playing
            new_playing = not bool(playing)
            return (not new_playing), interval_ms, ("❚❚" if new_playing else "▶"), new_playing

        # interval tick -> advance frame (loops), CLIENTSIDE (no server RTT).
        # All-flat deps (Output, Input, State, State) so the JS receives
        # (n_intervals, frame, nframes) in order — a mixed flat/list form
        # mis-groups the args and breaks the increment.
        app.clientside_callback(
            """
            function(n_intervals, frame, nframes) {
                var n = nframes || 1;
                return ((frame || 0) + 1) % Math.max(1, n);
            }
            """,
            Output("ds_frame", "value", allow_duplicate=True),
            Input("ds_interval", "n_intervals"),
            State("ds_frame", "value"),
            State("ds_nframes", "data"),
            prevent_initial_call=True,
        )
