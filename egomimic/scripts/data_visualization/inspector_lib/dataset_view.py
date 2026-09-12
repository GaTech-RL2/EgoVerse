"""Dataset Browser — a third inspector view.

Scrub any episode in a dataset *folder* (searchable by filename or by
annotation text), play it back as a timeline, overlay the recorded actions
(cartesian trajectory / orientation axes / keypoints), and toggle the
language annotations on/off.

Reuses the inspector's dark theme (`views`), zarr image IO (`images`), and
annotation parser (`language`). Frames are rendered server-side — the
`/dataset_frame/<episode>/<frame>` Flask route decodes the JPEG, draws the
requested overlay via the episode's embodiment class `viz` method, optionally burns in the
active annotation, and returns a JPEG. The browser just points an <img> at
that URL, so scrubbing the slider / flipping the overlay only swaps a src
string (the heavy work is cached server-side).

Overlay rendering is wrapped defensively: if an episode's action/keypoint
layout or camera calibration doesn't match, the route falls back to the
clean frame plus a small "overlay unavailable" badge rather than erroring —
so browsing + annotations always work even on datasets the projection
can't handle yet.

Projection convention (human egocentric `human_bimanual`): the head IS the
camera. Per-episode intrinsics are read from `zarr.json` through
`read_calibration` (the `calibration` block, or the legacy `intrinsics`
attribute of an older episode); poses in `*.obs_ee_pose`
(xyz + quat wxyz, SLAM-world) are transformed world → head frame using
`obs_head_pose` (same xyz+wxyz layout) and projected with that K. There is
no per-arm extrinsic. (This matches the validated `Human.viz` path; the old
keyed `CameraTransforms` calibration API was removed from this branch.)
"""

from __future__ import annotations

import io as _io
import logging
import os
import threading
from collections import OrderedDict
from functools import lru_cache

import numpy as np

# Convert source poses to the camera frame in this module. The selected
# `Embodiment.viz` method projects the poses and draws the overlay.
# `projectaria_tools` is optional in the standalone visualization environment.
# If an embodiment import fails, the browser shows an unavailable badge.
from egomimic.rldb.zarr.calibration import camera_name

from .images import (
    _bytes_from_zarr_element,
    _candidate_image_keys,
    _resolve_zarr_path,
    open_zarr_for_hash,
)
from .language import annotation_intervals, interval_for_frame
from .views import ACCENT, BORDER, CARD_STYLE, LABEL_STYLE, MUTED, PANEL, TEXT

logger = logging.getLogger(__name__)

try:
    from egomimic.rldb.embodiment import (
        Embodiment,
        Eva,
        Human,
        get_embodiment_class,
    )
except Exception as _e:  # pragma: no cover - optional heavy dep
    Embodiment = None
    Eva = None
    Human = None
    get_embodiment_class = None
    logger.warning("Embodiment classes unavailable, overlays disabled: %s", _e)


def _embodiment_class(grp):
    """Return the class identified by ``grp.attrs["embodiment"]``.

    The function returns ``Human`` if the attribute is absent or unknown. It
    returns ``None`` if the embodiment package did not import.
    """
    if Human is None:
        return None
    try:
        emb = str(dict(grp.attrs).get("embodiment", ""))
    except Exception:
        emb = ""
    return get_embodiment_class(emb, default=Human)

# ---- render cache + prefetch (so scrubbing/playback don't re-render) -------
# Rendered JPEG bytes keyed by (root, ep, frame, overlay, annotate).
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
            return min(int(arr.shape[0]), int(grp.attrs.get("total_frames", arr.shape[0])))
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
    try:
        from egomimic.rldb.zarr.validate import validate_episode

        validation = validate_episode(os.path.join(dataset_root, episode)).text()
    except Exception as exc:
        validation = f"Validation unavailable: {exc}"
    return {
        "validation": validation,
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


def _badge(img_rgb, text: str):
    """Keep capability/status text readable above the bottom annotation."""
    try:
        import cv2

        h, w = img_rgb.shape[:2]
        fs = max(0.4, h / 1000)
        th = max(1, int(h / 500))
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines, line = [], ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if line and cv2.getTextSize(candidate, font, fs, th)[0][0] > w - 16:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        if len(lines) > 4:
            lines = lines[:4]
            lines[-1] += "..."
        dy = cv2.getTextSize("Ag", font, fs, th)[0][1] + 10
        cv2.rectangle(img_rgb, (0, 0), (w, len(lines) * dy + 8), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            cv2.putText(img_rgb, line, (8, (index + 1) * dy), font, fs,
                        (255, 210, 90), th, cv2.LINE_AA)
    except Exception:
        pass
    return img_rgb


# ====================================================================== #
# Interactive 3D world-frame figure (plotly) — shown below the video.
# ====================================================================== #
# Both surfaces receive RGB arrays; cv2 writes the supplied channel tuples
# directly, so Plotly uses the same tuples without a BGR conversion.
_ARM_COLORS = {"left": "rgb(255,120,0)", "right": "rgb(0,80,255)"}
# Plotly bg to match the dark theme (CANVAS/PANEL are CSS colors from `views`).
_FIG_BG = "#0e1117"

# Visible style for the 3D graph wrapper (border/bg/spacing live here, not on the
# inner Graph, so the callback can swap the whole block to {"display": "none"}
# for overlay == "none" without losing the framing when it comes back).
_DS_3D_WRAP_STYLE = {
    "marginTop": "12px",
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px",
    "background": _FIG_BG,
}


def _rgb_tuple_to_css(rgb) -> str:
    """`(r,g,b)` 0-255 int tuple -> `'rgb(r,g,b)'` for plotly."""
    r, g, b = (int(c) for c in rgb)
    return f"rgb({r},{g},{b})"


def build_3d_figure(grp, frame: int, overlay: str, traj_window: int = 60):
    """Build a plotly `go.Figure` of the stored-frame scene at `frame`, showing
    ONLY the traces that match the selected `overlay` (mirrors the `ds_overlay`
    dropdown so the 3D view and the 2D image overlay stay in sync):

      - "cartesian"   -> only the EE trajectory (xyz path) over
                         [frame, min(total_frames, frame+traj_window)), plus small EE
                         position markers for context.
      - "orientation" -> only the EE orientation axes (x=red,y=green,z=blue),
                         0.05 m segments along the rotation-matrix columns
                         (quat wxyz -> xyzw for scipy), plus EE markers.
      - "keypoint"    -> only the MANO keypoint markers + skeleton edges (from
                         the embodiment's FINGER_EDGES / FINGER_EDGE_RANGES /
                         FINGER_COLORS).
      - "none"/unknown -> an empty figure (no data traces).

    Reuses whatever raw zarr arrays are present, per arm (left/right), and is
    defensive — a missing/short array just drops its trace, never raises.
    `uirevision` is keyed to the embodiment+nframes (a per-episode-stable id) so
    the user's rotate/zoom is preserved across frame changes and only resets on
    episode change. Coordinates are RAW stored xyz — no projection."""
    import plotly.graph_objects as go

    AXIS_LEN = 0.05  # metres
    want_traj = overlay == "cartesian"
    want_axes = overlay == "orientation"
    want_kp = overlay == "keypoint"
    want_ee_marker = want_traj or want_axes  # context marker for ee overlays
    emb_cls = _embodiment_class(grp)
    finger_colors = getattr(emb_cls, "FINGER_COLORS", Human.FINGER_COLORS if Human else {})

    frame = int(frame)
    traces = []

    sources = {}
    try:
        from egomimic.rldb.zarr.overlay import keypoint_source

        resolved = Embodiment.from_attrs(grp.attrs)
    except Exception:
        resolved = None

    def _arm_kp(arm):
        if resolved is None or arm not in resolved.end_effectors:
            return None
        spec = resolved.end_effectors[arm]
        a, key, arm_edges, ranges, root = keypoint_source(grp, arm, spec)
        sources[arm] = (arm_edges, ranges)
        total = min(int(grp.attrs.get("total_frames", 0)), a.shape[0]) if a is not None else 0
        if not 0 <= frame < total:
            return None
        try:
            pts = np.asarray(a[frame], dtype=float).reshape(spec.keypoints.n_slots, 3)
        except (ValueError, IndexError):
            return None
        valid = np.isin(np.arange(len(pts)), spec.keypoints.valid)
        valid &= np.isfinite(pts).all(axis=-1) & (np.abs(pts) < 1e8).all(axis=-1)
        pts[~valid] = np.nan
        # Preserve slot indices: deleting a missing point rewires the skeleton.
        return pts

    def _arm_ee(arm):
        a = _resolve_zarr_path(grp, f"{arm}.obs_ee_pose")
        if a is None or not hasattr(a, "shape") or a.shape[0] == 0:
            return None
        return a

    for arm in ("left", "right"):
        arm_css = _ARM_COLORS[arm]
        # --- keypoint markers + skeleton edges ---
        pts = _arm_kp(arm) if want_kp else None
        if pts is not None and pts.shape[0] >= 1:
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker={"size": 3, "color": arm_css},
                name=f"{arm} keypoints",
                hovertext=[f"kp{i}" for i in range(pts.shape[0])],
            ))
            # skeleton edges as one line trace with None breaks between segments.
            # Per-vertex RGB colors match the shared 2D skeleton.
            ex, ey, ez, ec = [], [], [], []
            arm_edges, ranges = sources[arm]
            arm_edge_colors = [arm_css] * len(arm_edges)
            for finger, start, end in ranges:
                for edge_i in range(start, min(end, len(arm_edges))):
                    arm_edge_colors[edge_i] = _rgb_tuple_to_css(finger_colors[finger])
            for ei, (a_idx, b_idx) in enumerate(arm_edges):
                if a_idx >= pts.shape[0] or b_idx >= pts.shape[0] or not np.isfinite(pts[[a_idx, b_idx]]).all():
                    continue
                ex += [pts[a_idx, 0], pts[b_idx, 0], None]
                ey += [pts[a_idx, 1], pts[b_idx, 1], None]
                ez += [pts[a_idx, 2], pts[b_idx, 2], None]
                col = arm_edge_colors[ei]
                ec += [col, col, col]
            if ex:
                traces.append(go.Scatter3d(
                    x=ex, y=ey, z=ez, mode="lines",
                    line={"color": ec, "width": 4},
                    name=f"{arm} skeleton", showlegend=True,
                ))

        # --- EE marker + orientation axes + trajectory ---
        ee = _arm_ee(arm) if (want_ee_marker or want_traj or want_axes) else None
        if ee is not None:
            f = max(0, min(frame, int(ee.shape[0]) - 1))
            try:
                pose = np.asarray(ee[f], dtype=float).reshape(-1)
            except Exception:
                pose = None
            if pose is not None and pose.shape[0] >= 7 and np.all(np.isfinite(pose[:7])):
                pos = pose[:3]
                if want_ee_marker:
                    traces.append(go.Scatter3d(
                        x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers",
                        marker={"size": 6, "color": arm_css, "symbol": "diamond"},
                        name=f"{arm} EE",
                    ))
                # orientation axes (quat wxyz -> xyzw)
                if want_axes:
                    try:
                        from scipy.spatial.transform import Rotation as R

                        quat = pose[3:7]
                        Rm = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                        for ai, axc in enumerate(("red", "green", "blue")):
                            end = pos + AXIS_LEN * Rm[:, ai]
                            traces.append(go.Scatter3d(
                                x=[pos[0], end[0]], y=[pos[1], end[1]],
                                z=[pos[2], end[2]], mode="lines",
                                line={"color": axc, "width": 5},
                                name=f"{arm} {'xyz'[ai]}-axis", showlegend=False,
                            ))
                    except Exception:
                        pass
            # trajectory over [frame, min(total_frames, frame+traj_window))
            if want_traj:
                try:
                    lo = max(0, frame)
                    hi = min(int(ee.shape[0]), int(grp.attrs["total_frames"]), frame + int(traj_window))
                    if hi - lo >= 2:
                        seg = np.asarray(ee[lo:hi], dtype=float)[:, :3]
                        seg = seg[np.all(np.isfinite(seg), axis=1)]
                        if seg.shape[0] >= 2:
                            traces.append(go.Scatter3d(
                                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                                line={"color": arm_css, "width": 2, "dash": "dot"},
                                name=f"{arm} EE traj", opacity=0.6,
                            ))
                except Exception:
                    pass

    # uirevision: stable per-episode id (embodiment + frame count) so the
    # user's rotate/zoom survives frame changes and only resets on episode swap.
    try:
        ee_l = _resolve_zarr_path(grp, "left.obs_ee_pose")
        nfr = int(ee_l.shape[0]) if ee_l is not None and hasattr(ee_l, "shape") else 0
    except Exception:
        nfr = 0
    rev = f"{dict(grp.attrs).get('embodiment','')}|{nfr}"

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=_FIG_BG,
        plot_bgcolor=_FIG_BG,
        font={"color": "#cbd5e1", "size": 11},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        legend={"x": 0, "y": 1, "bgcolor": "rgba(14,17,23,0.6)",
                "font": {"size": 10}},
        uirevision=rev,
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x", "backgroundcolor": _FIG_BG,
                      "gridcolor": "#2a2f3a", "color": "#94a3b8"},
            "yaxis": {"title": "y", "backgroundcolor": _FIG_BG,
                      "gridcolor": "#2a2f3a", "color": "#94a3b8"},
            "zaxis": {"title": "z", "backgroundcolor": _FIG_BG,
                      "gridcolor": "#2a2f3a", "color": "#94a3b8"},
        },
    )
    return fig


def _draw_overlay(img_rgb, grp, frame: int, overlay: str, horizon: int = 16,
                  camera: str = "front_1"):
    """Best-effort overlay. Returns (rgb, ok, note). On any failure returns
    the clean frame with ok=False and a short note, never raises.

    Builds CAMERA-frame inputs (per-episode zarr.json intrinsics + the human
    head-frame convention: head IS the camera, each world pose transformed
    world -> head via `obs_head_pose`) and hands them to the resolved EMBODIMENT
    CLASS's `viz` method — `<EmbClass>.viz(image, viz_data, mode=..., intrinsics=K)`
    — so the actual drawing lives in one place, the overlay matches the validated
    GT/pred path, and the browser is embodiment-generic. `viz` projects the
    cam-frame viz_data to pixels internally, so this module never projects."""
    if overlay in (None, "none"):
        return img_rgb, True, ""

    try:
        if Embodiment is None:
            raise ImportError("embodiment dependencies unavailable; RGB browsing remains available")
        from egomimic.rldb.zarr.overlay import render_overlay

        image, diagnostic = render_overlay(
            grp, frame, mode=overlay, image=img_rgb, horizon=horizon, camera=camera
        )
        notes = list(diagnostic.get("warnings", []))
        coverage = diagnostic.get("coverage", {})
        notes.extend(coverage.get("limitations", []))
        if coverage.get("estimated"):
            image = _badge(image, "ESTIMATED DATA - analysis only")
        return image, True, "; ".join(notes)
    except Exception as exc:
        logger.debug("overlay %s unavailable: %s", overlay, exc)
        return _badge(img_rgb.copy(), f"{overlay} overlay unavailable: {exc}"), False, str(exc)


def _annotate_frame(img_rgb, grp, dataset_root: str, episode: str, frame: int):
    """Burn the active annotation interval's text onto the frame via the
    embodiment class's `viz(mode="annotations")` (the base `Embodiment.viz`
    dispatches annotations to `_viz_annotations`)."""
    intervals = annotation_intervals(dataset_root, episode)
    match = interval_for_frame(intervals, int(frame))
    if not match:
        return img_rgb
    emb_cls = _embodiment_class(grp)
    if emb_cls is None:
        return img_rgb
    try:
        return emb_cls.viz(img_rgb, [match[2]], mode="annotations")
    except Exception:
        return img_rgb


def render_frame_jpeg(dataset_root: str, episode: str, frame: int, *,
                      overlay: str, annotate: bool, image_key: str, horizon: int = 16) -> bytes | None:
    key = (dataset_root, episode, int(frame), overlay, bool(annotate), image_key, int(horizon))
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
    rgb, _ok, _note = _draw_overlay(
        rgb, grp, int(frame), overlay, horizon=horizon, camera=camera_name(img_key) or img_key
    )
    if annotate:
        rgb = _annotate_frame(rgb, grp, dataset_root, episode, int(frame))
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
                 overlay: str, annotate: bool, image_key: str) -> None:
    """Pre-render every frame of `episode` (this overlay/annot variant) into the
    cache on a background daemon thread, so subsequent scrub/playback are cache
    hits. One thread per (episode, overlay, annotate) variant."""
    if not episode or n_frames <= 0:
        return
    wkey = (dataset_root, episode, overlay, bool(annotate))
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
                    annotate=annotate, image_key=image_key)

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
                 lang_key: str | None = None):
        self.app = app
        self.dataset_root = dataset_root
        self.image_key = image_key
        self.lang_key = lang_key

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
                html.Div("Horizon (frames)", style=LABEL_STYLE),
                dcc.Input(id="ds_horizon", type="number", value=16, min=1, max=1000, step=1, debounce=True),
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
                html.Div(style={"marginTop": "10px"}, children=[
                    dcc.Checklist(id="ds_show3d",
                                  options=[{"label": " 3D", "value": "on"}],
                                  value=["on"], inline=True,
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
                # left: video + timeline. Scrollable so the 3D panel below the
                # video is never clipped: the video + 420px graph + controls can
                # exceed the viewport, so the column scrolls (minHeight:0 lets the
                # flex children shrink; overflowY:auto + maxHeight:100vh give scroll).
                html.Div(style={"flex": "1", "padding": "16px", "minWidth": "0",
                                "display": "flex", "flexDirection": "column",
                                "minHeight": "0", "overflowY": "auto",
                                "maxHeight": "100vh", "maxWidth": "900px"},
                         children=[
                    html.Div(id="ds_title", style={"fontSize": "13px",
                             "fontWeight": 600, "color": TEXT, "marginBottom": "8px"}),
                    dcc.Loading(type="circle", color=ACCENT, children=html.Img(
                        id="ds_frame_img",
                        style={"width": "100%", "height": "auto",
                               "borderRadius": "6px", "border": f"1px solid {BORDER}",
                               "background": "#000"})),
                    # interactive world-frame 3D scene, synced to the slider frame.
                    # Wrapped so the callback can hide it (display:none) when the
                    # overlay is "none" — no empty 420px gap. _DS_3D_WRAP_STYLE is
                    # the visible style the callback restores for non-none overlays.
                    html.Div(id="ds_3d_wrap", style=_DS_3D_WRAP_STYLE,
                             children=[
                        dcc.Graph(id="ds_3d",
                                  style={"height": "420px", "width": "100%",
                                         "background": _FIG_BG},
                                  config={"displayModeBar": True}),
                    ]),
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
            try:
                horizon = max(1, min(1000, int(request.args.get("horizon", 16))))
            except ValueError:
                return abort(400)
            annotate = request.args.get("annot", "0") == "1"
            data = render_frame_jpeg(
                self.dataset_root, episode, frame_i, overlay=overlay,
                annotate=annotate, image_key=self.image_key, horizon=horizon)
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
                    f"image_key  {m['img_key']}\n\n{m.get('validation', '')}")
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
            function(frame, overlay, annotVal, episode, intervals, nframes, horizon) {
                if (!episode) { return ["", "", ""]; }
                frame = frame || 0;
                var annotate = annotVal && annotVal.indexOf("on") >= 0;
                var src = "/dataset_frame/" + episode + "/" + frame +
                          "?overlay=" + (overlay || "none") +
                          "&annot=" + (annotate ? "1" : "0") + "&horizon=" + (horizon || 16);
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
            Input("ds_horizon", "value"),
        )

        # CLIENTSIDE preloader: when the episode/overlay/annot changes, fetch
        # a bounded window of frame URLs into the browser's HTTP cache. Without this,
        # rapidly swapping the <img> src during playback/scrub makes the browser
        # cancel each in-flight load before it paints, so the frame appears to
        # freeze until motion stops. Once cached, each src-swap is an instant
        # cache hit and the image updates on every frame change. A token cancels
        # a superseded preload when the user switches episode/overlay.
        app.clientside_callback(
            """
            function(episode, overlay, annotVal, nframes, horizon, curFrame) {
                if (!episode || !nframes) { return 0; }
                var annotate = annotVal && annotVal.indexOf("on") >= 0;
                var ov = overlay || "none";
                var token = episode + "|" + ov + "|" + (annotate ? 1 : 0) + "|" + horizon;
                window._dsTok = token;
                // Keep only two requests in flight. The old timer enqueued
                // thousands of images, delaying the user's next overlay/frame.
                (window._dsImages || []).forEach(function(im) {
                    im.onload = im.onerror = null;
                    im.src = "";
                });
                var active = [];
                window._dsImages = active;
                var start = curFrame || 0, idx = 0;
                var count = Math.min(nframes, 120);
                function next() {
                    if (window._dsTok !== token || idx >= count) { return; }
                    var frame = (start + idx++) % nframes;
                    var im = new Image();
                    active.push(im);
                    im.onload = im.onerror = function() {
                        var i = active.indexOf(im);
                        if (i >= 0) { active.splice(i, 1); }
                        setTimeout(next, 80);
                    };
                    im.src = "/dataset_frame/" + episode + "/" + frame +
                             "?overlay=" + ov + "&annot=" + (annotate ? "1" : "0") +
                             "&horizon=" + (horizon || 16);
                }
                next(); next();
                return 0;
            }
            """,
            Output("ds_preload_sink", "data"),
            Input("ds_episode", "value"),
            Input("ds_overlay", "value"),
            Input("ds_annot", "value"),
            Input("ds_nframes", "data"),
            Input("ds_horizon", "value"),
            State("ds_frame", "value"),
        )

        # play/pause toggle (server: just flips the interval on/off).
        # prevent_initial_call so it does NOT auto-fire (and auto-play) on load.
        # CLIENTSIDE: flip the interval on/off in the browser (no server RTT), so
        # play/pause is instant even while the render server is busy serving
        # frames during playback. Mirrors the old server logic exactly:
        #   - play click -> toggle playing + retune interval to fps
        #   - fps change  -> keep play state, just retune interval
        app.clientside_callback(
            """
            function(n_clicks, fps, playing) {
                var ms = Math.floor(1000 / Math.max(1, (parseInt(fps, 10) || 10)));
                var ctx = (window.dash_clientside && dash_clientside.callback_context) || {};
                var trig = (ctx.triggered && ctx.triggered.length) ? ctx.triggered[0].prop_id : "";
                if (trig.indexOf("ds_fps") === 0) {
                    // fps change while playing: keep current play state, just retune rate
                    return [!playing, ms, (playing ? "❚❚" : "▶"), playing];
                }
                var new_playing = !playing;
                return [!new_playing, ms, (new_playing ? "❚❚" : "▶"), new_playing];
            }
            """,
            Output("ds_interval", "disabled"),
            Output("ds_interval", "interval"),
            Output("ds_play", "children"),
            Output("ds_playing", "data"),
            Input("ds_play", "n_clicks"),
            Input("ds_fps", "value"),
            State("ds_playing", "data"),
            prevent_initial_call=True,
        )

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

        # frame/episode/overlay/show3d -> rebuild the interactive 3D world-frame
        # figure AND toggle the wrapper's visibility. The 3D mirrors the overlay
        # dropdown (cartesian -> traj, orientation -> axes, keypoint -> skeleton).
        # Visibility rule: show the 3D iff the "3D" toggle is on AND an overlay is
        # selected; otherwise hide the wrapper (display:none) so it takes no space.
        # The 3D is cheap to rebuild, so it updates on EVERY frame change (scrub
        # AND playback) — no playback gate.
        @app.callback(
            Output("ds_3d", "figure"),
            Output("ds_3d_wrap", "style"),
            Input("ds_frame", "value"),
            Input("ds_episode", "value"),
            Input("ds_overlay", "value"),
            Input("ds_show3d", "value"),
            Input("ds_horizon", "value"),
            prevent_initial_call=False,
        )
        def _update_3d(frame, episode, overlay, show3d, horizon):
            hidden_style = {**_DS_3D_WRAP_STYLE, "display": "none"}
            show = show3d and "on" in show3d
            # hide (and skip building) when the 3D toggle is off or no overlay.
            if not show or overlay in (None, "none"):
                return dash.no_update, hidden_style
            if not episode:
                return dash.no_update, _DS_3D_WRAP_STYLE
            grp = open_zarr_for_hash(self.dataset_root, episode)
            if grp is None:
                return dash.no_update, _DS_3D_WRAP_STYLE
            try:
                return build_3d_figure(grp, int(frame or 0), overlay, traj_window=int(horizon or 16)), _DS_3D_WRAP_STYLE
            except Exception as e:
                logger.debug("build_3d_figure failed (%s,%s,%s): %s",
                             episode, frame, overlay, e)
                return dash.no_update, _DS_3D_WRAP_STYLE
