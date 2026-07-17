"""Scatter and KNN-browser views + their Dash callbacks.

Originally a single ~2500-line block of inner functions inside
`build_app`. Split into ScatterView (3D/2D point cloud + click-to-inspect)
and BrowserView (frame grid + raw-key KNN), both sharing a LayerStore.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np

from .caches import LayerStore
from .images import load_image_b64
from .io import LazyStringArray
from .language import (
    all_lang_concat_lower,
    annotation_intervals,
    interval_for_frame,
    load_language_prompt,
)

logger = logging.getLogger(__name__)


# ----- Style tokens (shared by scatter + browser layouts) ------------------
BG = "#0f172a"  # deep slate (header / sidebar background)
PANEL = "#ffffff"
CANVAS = "#f8fafc"  # main canvas
BORDER = "#e2e8f0"
TEXT = "#1e293b"
MUTED = "#64748b"
ACCENT = "#0d9488"  # teal-600

LABEL_STYLE = {
    "fontSize": "11px",
    "fontWeight": 600,
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "color": MUTED,
    "marginBottom": "4px",
    "display": "block",
}
CARD_STYLE = {
    "background": PANEL,
    "padding": "16px",
    "borderRadius": "8px",
    "border": f"1px solid {BORDER}",
    "marginBottom": "12px",
}

# Map a UI Reduction value → (csv-column key, human label).
_REDUCTION_TO_COORDS = {
    "umap": ("umap_xyz", "UMAP 3D"),
    "pca_umap": ("pca_umap_xyz", "PCA→UMAP 3D"),
    "tsne2d": ("tsne2d_xy", "t-SNE 2D"),
}


def _get_langs_per_row(data: dict, zarr_root: str) -> np.ndarray:
    """Return a numpy object-array, one entry per CSV row, holding the
    lowercase concatenation of every annotation interval that covers
    that frame. Cached on the data dict so the per-frame zarr scan
    only happens once per (run, layer). We use `all_lang_concat_lower`
    rather than a single annotation so substring filters (e.g. 'home')
    catch ANY paraphrase, not just one of N."""
    cached = data.get("langs_lower")
    if cached is not None:
        return cached
    hashes = data["hashes"]
    frames = data["frame_idx"]
    # Resolve unique (hash, frame) once, then expand. Image-token slices
    # have ~64 tokens/frame, so this is ~64x cheaper than per-row lookup.
    frame_lang: dict[tuple[str, int], str] = {}
    for h, f in zip(hashes.tolist(), frames.tolist()):
        key = (h, int(f))
        if key in frame_lang:
            continue
        intervals = annotation_intervals(zarr_root, h)
        frame_lang[key] = all_lang_concat_lower(intervals, int(f)) if intervals else ""
    out = np.array(
        [frame_lang[(h, int(f))] for h, f in zip(hashes.tolist(), frames.tolist())],
        dtype=object,
    )
    data["langs_lower"] = out
    return out


def _filter_data_by_lang(data: dict, excludes: list[str], zarr_root: str):
    """Return (filtered_data, n_dropped, keep_mask). `filtered_data` is a
    shallow-copy of `data` with all per-row arrays masked down to rows
    whose language doesn't contain any of `excludes`; `keep_mask` is the
    boolean row mask (None when nothing was filtered) so callers can map
    filtered row indices back to original CSV rows. If `excludes` is
    empty, returns `data` unchanged."""
    if not excludes:
        return data, 0, None
    langs = _get_langs_per_row(data, zarr_root)
    keep = np.ones(len(langs), dtype=bool)
    for sub in excludes:
        keep &= np.array([sub not in (text or "") for text in langs])
    if keep.all():
        return data, 0, None
    out = {}
    for k, v in data.items():
        if isinstance(v, (np.ndarray, LazyStringArray)) and len(v) == len(keep):
            out[k] = v[keep]
        else:
            out[k] = v
    return out, int((~keep).sum()), keep


def _action_key(zarr_root: str, video_hash: str, frame_idx: int):
    """Identity of the action instance a row belongs to: the recording plus
    the first annotation interval covering the frame. Rows with no covering
    interval (or unreadable annotations) collapse into one whole-recording
    action — same fallback as the same-embodiment action exclusion."""
    for s, e, *_ in annotation_intervals(zarr_root, str(video_hash)):
        if s <= frame_idx < e:
            return (str(video_hash), s, e)
    return (str(video_hash), -1, -1)


def _frames_look_like_source_indices(hashes, frames: np.ndarray, uniq) -> bool:
    """False when the frame_idx column is the legacy per-run sample counter
    (or missing entirely), i.e. rows CANNOT be mapped to annotation spans.

    Legacy banks (pre source-frame_idx eval_latent) wrote a global 0..K-1
    sample counter: the value set is contiguous from 0 and every value
    belongs to exactly one recording. True source indices repeat across
    recordings (every episode has a frame 0 / stride multiples) and are not
    a contiguous 0..K-1 run."""
    if frames.size == 0 or int(frames.min()) < 0:
        return False
    vals = np.unique(frames)
    if int(vals[0]) != 0 or int(vals[-1]) != int(vals.size) - 1:
        return True
    per_hash_unique = 0
    for h in uniq:
        hmask = np.asarray(hashes == h)
        per_hash_unique += int(np.unique(frames[hmask]).size)
    return per_hash_unique != int(vals.size)


# (cache_key, n_rows) -> (N,) int64 action ids. Small: a few layers' worth of
# int arrays. Keyed per (run, layer) so LayerStore cache turnover can't serve
# stale rows.
_ACTION_IDS_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_ACTION_IDS_CACHE_MAX = 8


def _row_action_ids(zarr_root: str, data: dict, cache_key: tuple | None = None):
    """(N,) int array assigning every row its action-instance id (recording +
    first covering annotation interval). Rows NOT covered by any annotation
    span get id ``-1`` — `_dedupe_neighbors_by_action` and pair_rank treat
    those as filtered-out, so unannotated frames never surface as KNN
    neighbors or pair-rank candidates.

    Exception: legacy banks whose frame_idx column is the per-run sample
    counter (detected via `_frames_look_like_source_indices`) cannot be
    mapped to spans at all; they keep the old whole-recording fallback id
    (no filtering) and log a warning — re-extract to get true frame indices.

    Built vectorized per unique recording with the intervals prefetched in
    parallel — per-row python/zarr lookups made click latency scale with both
    row count and episode count on the network FS."""
    n = len(np.asarray(data["frame_idx"]))
    key = (cache_key, n) if cache_key is not None else None
    if key is not None and key in _ACTION_IDS_CACHE:
        _ACTION_IDS_CACHE.move_to_end(key)
        return _ACTION_IDS_CACHE[key]

    hashes = data["hashes"]
    frames = np.asarray(data["frame_idx"])
    uniq = sorted({str(x) for x in hashes})

    # Warm the annotation_intervals lru in parallel (NFS-bound, like
    # _populate_browser_list does).
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(16, max(1, len(uniq)))) as ex:
        list(ex.map(lambda h: annotation_intervals(zarr_root, h), uniq))

    src_frames = _frames_look_like_source_indices(hashes, frames, uniq)
    if not src_frames:
        logger.warning(
            "_row_action_ids: frame_idx looks like the legacy per-run sample "
            "counter (or is missing) for %s — rows cannot be mapped to "
            "annotation spans, so unannotated frames are NOT filtered from "
            "the KNN. Re-extract the bank with the source-frame_idx "
            "eval_latent to enable annotation filtering.",
            cache_key,
        )

    ids = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for h in uniq:
        hmask = np.asarray(hashes == h)
        if not hmask.any():
            continue
        assigned = np.zeros(n, dtype=bool)
        spans = []
        for s, e, *_ in annotation_intervals(zarr_root, h):
            if (s, e) not in spans:
                spans.append((s, e))
        # Process in list order so overlapping spans resolve to the first
        # covering interval — same semantics as _action_key.
        for s, e in spans:
            sel = hmask & ~assigned & (frames >= s) & (frames < e)
            if sel.any():
                ids[sel] = next_id
                assigned |= sel
            next_id += 1
        rest = hmask & ~assigned
        if rest.any() and not src_frames:
            ids[rest] = next_id  # legacy whole-recording fallback action
        # With true source frames, `rest` rows stay -1: unannotated frames
        # are excluded from KNN results and pair ranking.
        next_id += 1

    if src_frames:
        n_filtered = int((ids < 0).sum())
        if n_filtered:
            logger.info(
                "_row_action_ids: %d/%d rows outside every annotation span "
                "(excluded from KNN) for %s",
                n_filtered,
                n,
                cache_key,
            )

    if key is not None:
        _ACTION_IDS_CACHE[key] = ids
        while len(_ACTION_IDS_CACHE) > _ACTION_IDS_CACHE_MAX:
            _ACTION_IDS_CACHE.popitem(last=False)
    return ids


def _dedupe_neighbors_by_action(
    zarr_root: str, data: dict, candidates, k: int, cache_key: tuple | None = None
):
    """Keep only the nearest row per action instance, up to k rows.
    `candidates` is an iterable of (distance, row_idx) pairs already sorted
    by ascending distance; consumed lazily, so callers can pass a generator
    over a full argsort without materializing it. Rows with action id -1
    (frame not covered by any annotation span) are dropped outright —
    unannotated frames must not surface as KNN neighbors."""
    ids = _row_action_ids(zarr_root, data, cache_key=cache_key)
    seen: set = set()
    out: list[tuple[float, int]] = []
    for dist, ridx in candidates:
        a = int(ids[ridx])
        if a < 0:
            continue
        if a in seen:
            continue
        seen.add(a)
        out.append((dist, ridx))
        if len(out) >= k:
            break
    return out


def _find_pair_action(
    zarr_root: str, data: dict, video_hash: str, frame_idx: int, src_emb: str
):
    """Locate the matching action in the paired opposite-embodiment episode.

    Perfect-pair episodes carry (near-)identical annotation sets, so the twin
    is the opposite-embodiment episode sharing the most (start, end, text)
    annotation entries with the clicked episode. The matching action is the
    twin's interval equal to the clicked frame's covering interval, with a
    text-only fallback (two aria pair episodes share canonical texts but not
    every rephrasing). Returns (twin_hash, start, end) or None when the frame
    is unannotated or no opposite-embodiment episode shares any entry."""
    intervals = annotation_intervals(zarr_root, str(video_hash))
    cover = [(s, e) for s, e, _, _ in intervals if s <= frame_idx < e]
    if not cover:
        return None
    s0, e0 = cover[0]
    own = {(s, e, low) for s, e, low, _ in intervals}
    embs = np.asarray(data["embs"])
    opp_hashes = sorted({str(x) for x in data["hashes"][embs != src_emb]})
    best_hash, best_iv, best_score = None, None, 0
    for h2 in opp_hashes:
        iv2 = annotation_intervals(zarr_root, h2)
        score = sum((s, e, low) in own for s, e, low, _ in iv2)
        if score > best_score:
            best_hash, best_iv, best_score = h2, iv2, score
    if best_hash is None:
        return None
    # The matching action is the twin's interval with the same boundaries
    # (pair members share segmentation); fall back to max frame-overlap.
    spans = sorted({(s, e) for s, e, _, _ in best_iv})
    if (s0, e0) in spans:
        s2, e2 = s0, e0
    else:
        s2, e2 = max(spans, key=lambda sp: max(0, min(sp[1], e0) - max(sp[0], s0)))
        if min(e2, e0) - max(s2, s0) <= 0:
            return None
    # Select the twin's prompt with the SAME rule the clicked-frame display
    # uses (interval_for_frame → last paraphrase covering the frame). With
    # byte-identical pair annotations this makes the paired prompt equal the
    # clicked prompt; when they differ it stays an apples-to-apples compare.
    m = interval_for_frame(best_iv, s2)
    prompt = m[2] if m is not None else ""
    return best_hash, s2, e2, prompt


def _pair_action_rows(data: dict, twin_hash: str, start: int, end: int):
    """Row indices of the twin episode's frames inside the action interval."""
    frames = np.asarray(data["frame_idx"])
    return np.where((data["hashes"] == twin_hash) & (frames >= start) & (frames < end))[
        0
    ]


def _pair_distance_text(
    twin_hash: str,
    start: int,
    end: int,
    prompt: str,
    dists_rows,
    rows,
    frames,
    space_label: str,
) -> str:
    """Render the paired-action distance block shown under the clicked frame."""
    j = int(np.argmin(dists_rows))
    return (
        f"\n\n-- paired action ({space_label}) --\n"
        f"{twin_hash}  seg [{start},{end})\n"
        f"prompt: {prompt}\n"
        f"min dist {float(dists_rows[j]):.4f} @ frame {int(frames[int(rows[j])])}"
    )


class ScatterView:
    """3D/2D scatter of latent reduction coords + click-to-inspect pane."""

    def __init__(
        self,
        app,
        store: LayerStore,
        zarr_root: str,
        lang_key: str | None,
        image_key: str,
        default_sample: int,
    ):
        self.app = app
        self.store = store
        self.zarr_root = zarr_root
        self.lang_key = lang_key
        self.image_key = image_key
        self.default_sample = default_sample
        self._knn_trees = {}  # (run_path, layer, reduction, target_emb) -> (cKDTree, target_indices)
        # (run, layer, drop_k, rows-md5) -> (rows, umap_xyz) for the live
        # PCA-removal recompute. Small: only sampled points are embedded.
        self._live_umap_cache = {}

    def _live_umap_coords(self, run_path: str, layer: str, drop_k: int, rows):
        """UMAP-3D embedding of the Fisher-filtered PCA features for the
        given original-CSV `rows` (the sampled points). Cached by content,
        so re-clicking Apply with unchanged controls is instant. Returns
        (coords (len(rows), 3), None) or (None, error_message)."""
        import hashlib
        import time

        cache_key = (
            run_path,
            layer,
            int(drop_k),
            hashlib.md5(np.ascontiguousarray(rows).tobytes()).hexdigest(),
        )
        cached = self._live_umap_cache.get(cache_key)
        if cached is not None:
            return cached, None

        feats = self.store.pca_features_dropped(
            run_path, layer, n_components=50, drop_k=drop_k
        )
        if feats is None:
            return None, (
                f"PCA removal needs raw keys + both embodiments — "
                f"{layer}_keys.pt missing or Fisher ranking unavailable "
                f"(re-run eval with +force_reeval=true to write raw keys)"
            )
        if feats.shape[1] == 0:
            return None, "all PCA dims removed — lower the slider"
        if rows.max() >= feats.shape[0]:
            return None, (
                f"PCA features rows ({feats.shape[0]}) mismatch CSV rows — "
                f"stale {layer}_keys.pt?"
            )
        try:
            import umap
        except ImportError:
            return None, "umap-learn not installed — pip install umap-learn"
        t0 = time.perf_counter()
        X = feats[rows]
        coords = umap.UMAP(n_components=3, random_state=42).fit_transform(X)
        coords = np.asarray(coords, dtype=np.float32)
        logger.info(
            "live UMAP for %s | %s drop_k=%d on %d pts in %.1fs",
            os.path.basename(run_path),
            layer,
            drop_k,
            len(rows),
            time.perf_counter() - t0,
        )
        self._live_umap_cache[cache_key] = coords
        while len(self._live_umap_cache) > 8:
            self._live_umap_cache.pop(next(iter(self._live_umap_cache)))
        return coords, None

    def build_figure(
        self,
        run_path: str,
        layer: str,
        sample: int,
        color_by: str,
        reduction: str,
        remove_outliers: bool = False,
        outlier_thresh: float = 3.0,
        excludes: list[str] | None = None,
        pca_drop_k: int = 0,
    ):
        import time

        import plotly.graph_objects as go

        from .io import (
            stratified_sample,
        )

        _t_overall = time.perf_counter()
        _t_phases: dict[str, float] = {}

        def _phase(name: str, t0: float):
            _t_phases[name] = time.perf_counter() - t0

        if not run_path or not layer:
            fig = go.Figure()
            fig.update_layout(
                title="Pick a Run and Layer, then click Apply.",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            return fig
        t0 = time.perf_counter()
        data = self.store.load(run_path, layer)
        _phase("store.load", t0)

        pca_drop_k = int(pca_drop_k or 0)

        # Trigger the selected reduction's lazy materialization NOW (before
        # filter/sample). Lazy fields are None placeholders in the dict;
        # without this, the filtered/sampled copy propagates None and the
        # scatter goes blank. Skipped when PCA removal is active — coords
        # then come from a live UMAP recompute, not the CSV.
        if pca_drop_k <= 0:
            t0 = time.perf_counter()
            reduction_key = _REDUCTION_TO_COORDS.get(reduction, ("umap_xyz", "?"))[0]
            _ = data[reduction_key]
            _phase(f"materialize[{reduction_key}]", t0)

        # Apply lang filter (e.g. drop "home" frames) BEFORE sampling so the
        # filter affects what shows up on the plot, not just whether the
        # `sample` points include filtered ones.
        t0 = time.perf_counter()
        n_lang_dropped = 0
        lang_keep = None
        if excludes:
            data, n_lang_dropped, lang_keep = _filter_data_by_lang(
                data, list(excludes), self.zarr_root
            )
        _phase("lang_filter", t0)

        n = len(data["hashes"])
        if n == 0:
            return go.Figure()

        t0 = time.perf_counter()
        groups = data["embs"] if color_by == "embodiment" else data["hashes"]
        keep = stratified_sample(groups, sample)
        sub = {k: (v[keep] if v is not None else None) for k, v in data.items()}
        _phase("sample", t0)
        sub_groups = groups[keep]
        n_outliers_removed = 0
        # Original-CSV row indices of the sampled points — needed to slice
        # the full-N PCA features when PCA removal is active.
        orig_rows = (
            np.flatnonzero(lang_keep)[keep]
            if lang_keep is not None
            else np.asarray(keep)
        )

        # All reductions are read directly from precomputed coords produced
        # by eval_latent — no client-side recompute. If the user picks a
        # reduction whose data isn't in this layer's .pt/.csv, return an
        # empty figure with a clear "no X exists" message instead of
        # silently falling back to a different reduction.
        coords = None
        axis_prefix = reduction
        d = 0
        missing_msg = None
        # (reduction-name → (data-key, display-label, dims, enrich-config-flag))
        spec = {
            "umap": ("umap_xyz", "UMAP", 3, "evaluator.compute_umap=true"),
            "pca_umap": (
                "pca_umap_xyz",
                "PCA→UMAP",
                3,
                "evaluator.compute_pca_umap=true",
            ),
            "tsne2d": ("tsne2d_xy", "t-SNE 2D", 2, "evaluator.compute_tsne_2d=true"),
            "tsne3d": ("tsne3d_xyz", "t-SNE 3D", 3, "evaluator.compute_tsne_3d=true"),
            "pca": ("pca_xyz", "PCA", 3, "evaluator.compute_pca=true"),
        }
        if pca_drop_k > 0:
            # PCA removal active: ignore the precomputed reductions and embed
            # the sampled points' Fisher-filtered PCA features with a live
            # UMAP fit (cached per (run, layer, k, sample)).
            t0 = time.perf_counter()
            coords, live_err = self._live_umap_coords(
                run_path, layer, pca_drop_k, orig_rows
            )
            _phase("live_umap", t0)
            if coords is None:
                missing_msg = live_err
            else:
                axis_prefix, d = "umap", 3
        elif reduction not in spec:
            missing_msg = f"unknown reduction: {reduction!r}"
        else:
            key, label, dims, flag = spec[reduction]
            arr = sub.get(key)
            if arr is not None:
                axis_prefix = {
                    "umap": "umap",
                    "pca_umap": "pca_umap",
                    "tsne2d": "tsne",
                    "tsne3d": "tsne",
                    "pca": "pca",
                }[reduction]
                coords, d = arr, dims
            else:
                missing_msg = (
                    f"no {label} exists for this layer — re-run enrichment "
                    f"with `{flag}` to add it"
                )

        if coords is None:
            fig = go.Figure()
            fig.update_layout(
                title=f"{layer} — {missing_msg}",
                annotations=[
                    dict(
                        text=missing_msg,
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14, color="#b91c1c"),
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            return fig

        # Optional outlier filter: drop rows whose coords are >`outlier_thresh`
        # std-devs from the centroid in any dimension. Per-dim z-score, max
        # across dims as the outlier metric. Applied AFTER sampling so users
        # can see how many of the displayed points get hidden.
        t_plot = time.perf_counter()
        if remove_outliers and coords.shape[0] > 0 and outlier_thresh > 0:
            mu = coords.mean(axis=0)
            sigma = coords.std(axis=0)
            sigma = np.where(sigma == 0, 1.0, sigma)  # avoid div-by-zero
            z = np.abs((coords - mu) / sigma)
            keep_mask = z.max(axis=1) <= outlier_thresh
            n_outliers_removed = int((~keep_mask).sum())
            if n_outliers_removed > 0:
                coords = coords[keep_mask]
                sub_groups = sub_groups[keep_mask]
                # Filter the customdata fields the same way.
                for fld in ("hashes", "embs", "frame_idx", "token_idx"):
                    if sub.get(fld) is not None:
                        sub[fld] = sub[fld][keep_mask]

        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        unique = sorted(set(sub_groups.tolist()))
        color_for = {g: palette[i % len(palette)] for i, g in enumerate(unique)}

        fig = go.Figure()
        for g in unique:
            mask = sub_groups == g
            pts = coords[mask]
            customdata = np.stack(
                [
                    sub["hashes"][mask],
                    sub["frame_idx"][mask].astype(str),
                    sub["token_idx"][mask].astype(str),
                    sub["embs"][mask],
                ],
                axis=-1,
            )
            hovertemplate = (
                "hash=%{customdata[0]}<br>"
                "frame=%{customdata[1]}<br>"
                "token=%{customdata[2]}<br>"
                "emb=%{customdata[3]}<extra></extra>"
            )
            common = dict(
                marker=dict(size=4 if d == 3 else 5, color=color_for[g], opacity=0.7),
                name=f"{g} (n={int(mask.sum())})",
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
            if d == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="markers",
                        **common,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        **common,
                    )
                )
        fig.update_layout(
            title=(
                f"{layer} ({axis_prefix} {d}D, n={coords.shape[0]}"
                + (
                    f", live UMAP on PCA-50 −top{pca_drop_k} Fisher dims"
                    if pca_drop_k > 0
                    else ""
                )
                + (
                    f", {n_outliers_removed} outliers hidden"
                    if n_outliers_removed > 0
                    else ""
                )
                + (f", {n_lang_dropped} lang-filtered" if n_lang_dropped > 0 else "")
                + ")"
            ),
            scene=dict(
                xaxis_title=f"{axis_prefix}_x",
                yaxis_title=f"{axis_prefix}_y",
                zaxis_title=f"{axis_prefix}_z",
                aspectmode="data",
            )
            if d == 3
            else None,
            xaxis_title=f"{axis_prefix}_x" if d == 2 else None,
            yaxis_title=f"{axis_prefix}_y" if d == 2 else None,
            uirevision="constant",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(itemsizing="constant"),
        )
        _phase("plotly_build", t_plot)
        # One-line breakdown so the slow phase is obvious.
        phase_str = " ".join(
            f"{name}={dt*1000:.0f}ms" for name, dt in _t_phases.items()
        )
        logger.info(
            "build_figure %s/%s reduction=%s in %.2fs | %s",
            os.path.basename(run_path),
            layer,
            reduction,
            time.perf_counter() - _t_overall,
            phase_str,
        )
        return fig

    def preset_button(self, label, value):
        from dash import html

        return html.Button(
            label,
            id={"role": "preset", "value": value},
            n_clicks=0,
            style={
                "padding": "4px 10px",
                "fontSize": "12px",
                "background": "#f1f5f9",
                "color": TEXT,
                "border": f"1px solid {BORDER}",
                "borderRadius": "4px",
                "marginRight": "4px",
                "marginBottom": "4px",
                "cursor": "pointer",
            },
        )

    def _get_knn_tree(self, data, run_path, layer, reduction, target_emb, coords=None):
        """Get or build a cKDTree for the target embodiment's coords."""
        from scipy.spatial import cKDTree

        cache_key = (run_path, layer, reduction, target_emb)
        if cache_key in self._knn_trees:
            return self._knn_trees[cache_key]

        if coords is None:
            coord_key = _REDUCTION_TO_COORDS.get(reduction, ("umap_xyz", "?"))[0]
            coords = data.get(coord_key)
        if coords is None:
            return None

        embs = data["embs"]
        target_mask = embs == target_emb
        if not target_mask.any():
            self._knn_trees[cache_key] = None
            return None

        target_idx = np.where(target_mask)[0]
        tree = cKDTree(coords[target_idx])
        entry = (tree, target_idx)
        self._knn_trees[cache_key] = entry
        if len(self._knn_trees) > 16:
            oldest = next(iter(self._knn_trees))
            del self._knn_trees[oldest]
        return entry

    def knn_other_embodiment(
        self,
        data,
        src_idx,
        src_emb,
        reduction,
        k=10,
        coords_override=None,
        run_path=None,
        layer=None,
    ):
        """Return list of dicts for the K closest opposite-embodiment rows.
        Uses a cKDTree for O(log N) queries instead of brute-force."""
        if coords_override is not None:
            coords = coords_override
        else:
            coord_key = _REDUCTION_TO_COORDS.get(reduction, ("umap_xyz", "?"))[0]
            coords = data.get(coord_key)
        if coords is None:
            return None

        embs = data["embs"]
        unique_embs = set(np.unique(embs))
        unique_embs.discard(src_emb)
        if not unique_embs:
            return []

        all_neighbors = []
        for target_emb in unique_embs:
            if run_path and layer:
                entry = self._get_knn_tree(
                    data, run_path, layer, reduction, target_emb, coords
                )
            else:
                entry = None

            if entry is not None:
                tree, target_idx = entry
                query_pt = coords[src_idx].reshape(1, -1)
                n_target = tree.data.shape[0]
                # At most one row per action instance: the raw top-k is
                # dominated by temporally-adjacent rows of a single
                # opposite-embodiment motion, so over-query and dedupe,
                # expanding until k distinct actions are found or the
                # embodiment is exhausted.
                query_k = min(max(64, k * 16), n_target)
                while True:
                    dists_arr, idx_arr = tree.query(query_pt, k=query_k)
                    cand = (
                        (float(d), int(target_idx[int(i)]))
                        for d, i in zip(
                            np.atleast_1d(np.squeeze(dists_arr)),
                            np.atleast_1d(np.squeeze(idx_arr)),
                        )
                    )
                    kept = _dedupe_neighbors_by_action(
                        self.zarr_root,
                        data,
                        cand,
                        k,
                        cache_key=(run_path, layer),
                    )
                    if len(kept) >= k or query_k >= n_target:
                        break
                    query_k = min(query_k * 8, n_target)
                all_neighbors.extend(kept)
            else:
                target_mask = embs == target_emb
                target_idx = np.where(target_mask)[0]
                diff = coords[target_idx] - coords[src_idx]
                dists = np.linalg.norm(diff, axis=1)
                order = np.argsort(dists)
                cand = ((float(dists[int(o)]), int(target_idx[int(o)])) for o in order)
                all_neighbors.extend(
                    _dedupe_neighbors_by_action(
                        self.zarr_root, data, cand, k, cache_key=(run_path, layer)
                    )
                )

        all_neighbors.sort(key=lambda x: x[0])
        # Re-dedupe across embodiments (no-op when there is only one
        # opposite embodiment) and truncate to k.
        all_neighbors = _dedupe_neighbors_by_action(
            self.zarr_root, data, all_neighbors, k, cache_key=(run_path, layer)
        )
        out = []
        for rank, (dist, ridx) in enumerate(all_neighbors, start=1):
            out.append(
                {
                    "rank": rank,
                    "hash": str(data["hashes"][ridx]),
                    "frame_idx": int(data["frame_idx"][ridx]),
                    "token_idx": int(data["token_idx"][ridx]),
                    "embodiment": str(data["embs"][ridx]),
                    "distance": dist,
                }
            )
        return out

    def knn_buttons(self, neighbors):
        """Render neighbor list as a column of html.Buttons. Each row shows a
        thumbnail + hash + frame/distance/embodiment + language prompt, and
        is clickable to navigate the right pane to that neighbor."""
        from dash import html

        if neighbors is None:
            return [
                html.Div(
                    "(no coords for this reduction in this layer)",
                    style={"color": "#94a3b8"},
                )
            ]
        if not neighbors:
            return [
                html.Div(
                    "(no rows from a different embodiment)", style={"color": "#94a3b8"}
                )
            ]
        rows = []
        for n in neighbors:
            short_hash = n["hash"][:20] + ("…" if len(n["hash"]) > 20 else "")
            lang_text, _ = load_language_prompt(
                self.zarr_root,
                n["hash"],
                n["frame_idx"],
                lang_key=self.lang_key,
            )
            lang_display = lang_text or "(no language prompt)"
            rows.append(
                html.Button(
                    children=html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "gap": "10px",
                            "width": "100%",
                        },
                        children=[
                            html.Span(
                                f"{n['rank']:02d}",
                                style={
                                    "fontFamily": (
                                        "ui-monospace, SFMono-Regular, "
                                        "Menlo, Consolas, monospace"
                                    ),
                                    "fontSize": "11px",
                                    "color": MUTED,
                                    "fontWeight": 500,
                                    "minWidth": "20px",
                                    "textAlign": "right",
                                    "flexShrink": "0",
                                    "paddingTop": "2px",
                                },
                            ),
                            html.Img(
                                src=f"/thumbnail/{n['hash']}/{n['frame_idx']}",
                                style={
                                    "width": "72px",
                                    "height": "44px",
                                    "objectFit": "cover",
                                    "borderRadius": "4px",
                                    "background": CANVAS,
                                    "flexShrink": "0",
                                    "border": f"1px solid {BORDER}",
                                },
                            ),
                            html.Div(
                                style={
                                    "flex": "1",
                                    "textAlign": "left",
                                    "overflow": "hidden",
                                    "minWidth": "0",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "2px",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "6px",
                                            "fontSize": "11px",
                                            "color": TEXT,
                                            "fontWeight": 500,
                                        },
                                        children=[
                                            html.Span(
                                                style={
                                                    "width": "5px",
                                                    "height": "5px",
                                                    "borderRadius": "999px",
                                                    "background": ACCENT,
                                                    "flexShrink": "0",
                                                },
                                            ),
                                            html.Span(n["embodiment"]),
                                            html.Span(
                                                f"d={n['distance']:.3f}",
                                                style={
                                                    "marginLeft": "auto",
                                                    "color": MUTED,
                                                    "fontFamily": (
                                                        "ui-monospace, "
                                                        "SFMono-Regular, Menlo, "
                                                        "Consolas, monospace"
                                                    ),
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        short_hash,
                                        style={
                                            "fontSize": "11px",
                                            "color": MUTED,
                                            "fontFamily": (
                                                "ui-monospace, SFMono-Regular, "
                                                "Menlo, Consolas, monospace"
                                            ),
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "whiteSpace": "nowrap",
                                        },
                                    ),
                                    html.Div(
                                        lang_display,
                                        style={
                                            "fontSize": "11px",
                                            "color": TEXT,
                                            "marginTop": "2px",
                                            "lineHeight": "1.35",
                                            "whiteSpace": "normal",
                                            "overflowWrap": "anywhere",
                                        },
                                        title=lang_display,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    id={
                        "role": "knn_row",
                        "hash": n["hash"],
                        "frame": n["frame_idx"],
                        "token": n["token_idx"],
                    },
                    n_clicks=0,
                    style={
                        "padding": "8px 10px",
                        "background": PANEL,
                        "border": f"1px solid {BORDER}",
                        "cursor": "pointer",
                        "borderRadius": "6px",
                        "textAlign": "left",
                        "transition": "border-color 0.12s ease",
                    },
                )
            )
        return rows

    def build_inspect_payload(
        self,
        run_path,
        layer,
        video_hash,
        frame_idx,
        token_idx,
        emb,
        reduction,
        pca_drop_k: int = 0,
    ):
        """Compute everything needed to populate the right pane for one
        clicked frame: meta text, image URI, lang text, KNN buttons, label.
        When `pca_drop_k` > 0 the KNN is computed in the Fisher-filtered
        PCA-50 space (matching the live-UMAP scatter) instead of the
        precomputed reduction coords."""
        from dash import html

        meta = (
            f"video_hash : {video_hash}\n"
            f"frame_idx  : {frame_idx}\n"
            f"token_idx  : {token_idx}\n"
            f"embodiment : {emb}"
        )
        img_src, img_tried = load_image_b64(
            self.zarr_root, video_hash, frame_idx, self.image_key
        )
        lang_text, lang_tried = load_language_prompt(
            self.zarr_root,
            video_hash,
            frame_idx,
            lang_key=self.lang_key,
        )
        if img_src is None:
            meta = (
                meta
                + "\n\nImage load failed. Paths tried:\n  "
                + "\n  ".join(img_tried[:15])
            )
        if lang_text:
            lang_display = lang_text
        else:
            lang_display = (
                "(no language prompt found)\n\n"
                "Paths attempted:\n  " + "\n  ".join(lang_tried[:25])
            )

        pca_drop_k = int(pca_drop_k or 0)
        if pca_drop_k > 0:
            red_label = f"PCA-50 −top{pca_drop_k} Fisher dims"
        else:
            red_label = _REDUCTION_TO_COORDS.get(reduction, ("?", reduction))[1]
        knn_label = f"10 closest opposite-embodiment frames ({red_label}, 1 per action)"
        knn_buttons = [html.Div("(no run/layer selected)", style={"color": "#94a3b8"})]
        if run_path and layer:
            try:
                data = self.store.load(run_path, layer)
                m = (
                    (data["hashes"] == str(video_hash))
                    & (data["frame_idx"] == frame_idx)
                    & (data["token_idx"] == token_idx)
                )
                where = np.where(m)[0]
                if len(where) == 0:
                    knn_buttons = [
                        html.Div(
                            "(could not find clicked row in cached data)",
                            style={"color": "#b91c1c"},
                        )
                    ]
                else:
                    src_idx = int(where[0])
                    src_emb = str(data["embs"][src_idx])
                    coords_override = None
                    knn_reduction = reduction
                    if pca_drop_k > 0:
                        # Same Fisher-filtered space the live-UMAP scatter is
                        # built from. The tag keeps the cKDTree cache keyed
                        # per drop level.
                        coords_override = self.store.pca_features_dropped(
                            run_path, layer, n_components=50, drop_k=pca_drop_k
                        )
                        knn_reduction = f"pca50_drop{pca_drop_k}"
                        if coords_override is None or coords_override.shape[0] != len(
                            data["hashes"]
                        ):
                            raise RuntimeError(
                                "Fisher-filtered PCA features unavailable or "
                                "misaligned — re-run eval with "
                                "+force_reeval=true to write raw keys"
                            )
                    neighbors = self.knn_other_embodiment(
                        data,
                        src_idx,
                        src_emb,
                        knn_reduction,
                        k=10,
                        coords_override=coords_override,
                        run_path=run_path,
                        layer=layer,
                    )
                    knn_buttons = self.knn_buttons(neighbors)
                    # Paired-action distance, in the same space as the KNN
                    # list, appended below the clicked frame's language.
                    if coords_override is not None:
                        pair_coords = coords_override
                    else:
                        coord_key = _REDUCTION_TO_COORDS.get(
                            knn_reduction, ("umap_xyz", "?")
                        )[0]
                        pair_coords = data.get(coord_key)
                    pair = _find_pair_action(
                        self.zarr_root, data, str(video_hash), int(frame_idx), src_emb
                    )
                    if pair is not None and pair_coords is not None:
                        twin, s2, e2, pair_prompt = pair
                        rows = _pair_action_rows(data, twin, s2, e2)
                        if rows.size:
                            d = np.linalg.norm(
                                pair_coords[rows] - pair_coords[src_idx], axis=1
                            )
                            lang_display = lang_display + _pair_distance_text(
                                twin,
                                s2,
                                e2,
                                pair_prompt,
                                d,
                                rows,
                                np.asarray(data["frame_idx"]),
                                red_label,
                            )
            except Exception as e:
                knn_buttons = [
                    html.Div(
                        f"(KNN error: {type(e).__name__}: {e})",
                        style={"color": "#b91c1c"},
                    )
                ]
        return (meta, img_src, lang_display, knn_label, knn_buttons)

    def resolve_emb(self, run_path, layer, video_hash, frame_idx, token_idx):
        """Look up the embodiment of a (hash, frame, token) row in cached data."""
        if not run_path or not layer:
            return "?"
        try:
            data = self.store.load(run_path, layer)
            m = (
                (data["hashes"] == str(video_hash))
                & (data["frame_idx"] == frame_idx)
                & (data["token_idx"] == token_idx)
            )
            where = np.where(m)[0]
            if len(where) > 0:
                return str(data["embs"][int(where[0])])
        except Exception:
            pass
        return "?"

    def empty_payload(self):
        return ("Click a point to inspect.", None, "", "", [], [], True)

    def register(self):
        """Wire all scatter-mode Dash callbacks."""
        import dash
        from dash import Input, Output, State

        app = self.app
        default_sample = self.default_sample

        # Sync sample-preset buttons → the Sample input.
        @app.callback(
            Output("sample", "value"),
            Input({"role": "preset", "value": dash.ALL}, "n_clicks"),
            State("sample", "value"),
            prevent_initial_call=True,
        )
        def _on_preset(_clicks, current):
            ctx = dash.callback_context
            if not ctx.triggered or not any(c for c in (_clicks or [])):
                return current
            triggered = ctx.triggered[0]["prop_id"]
            # prop_id looks like '{"role":"preset","value":1000}.n_clicks'
            try:
                import json

                comp_id = json.loads(triggered.split(".")[0])
                return int(comp_id["value"])
            except Exception:
                return current

        # When the user picks a different Run, refresh the Layer dropdown to
        # reflect the layers in that run. Default the value to the first layer.
        @app.callback(
            Output("layer", "options"),
            Output("layer", "value"),
            Input("run", "value"),
        )
        def _on_run_change(run_path):
            layers = self.store.layers_for(run_path) if run_path else []
            opts = [{"label": ln, "value": ln} for ln in layers]
            return opts, (layers[0] if layers else None)

        @app.callback(
            Output("scatter", "figure"),
            Input("apply", "n_clicks"),
            State("run", "value"),
            State("layer", "value"),
            State("sample", "value"),
            State("reduction", "value"),
            State("remove_outliers", "value"),
            State("outlier_thresh", "value"),
            State("browser_hide_home", "value"),
            State("browser_lang_exclude", "value"),
            State("pca_drop_k", "value"),
        )
        def update_figure(
            _n_clicks,
            run_path,
            layer,
            sample,
            reduction,
            remove_outliers_val,
            outlier_thresh_val,
            hide_home_val,
            lang_exclude_val,
            pca_drop_k,
        ):
            # Only re-render when the user clicks Apply. Pre-1st-click also fires
            # once (n_clicks=0) so the initial figure renders on page load.
            sample = int(sample) if sample else default_sample
            remove_out = bool(remove_outliers_val and "on" in remove_outliers_val)
            thresh = float(outlier_thresh_val) if outlier_thresh_val else 3.0
            excludes: list[str] = []
            if hide_home_val and "on" in hide_home_val:
                excludes.append("home")
            custom = (lang_exclude_val or "").strip().lower()
            if custom:
                excludes.append(custom)
            return self.build_figure(
                run_path,
                layer,
                sample,
                "embodiment",
                reduction,
                remove_outliers=remove_out,
                outlier_thresh=thresh,
                excludes=excludes,
                pca_drop_k=int(pca_drop_k or 0),
            )

        # Unified click handler — fires on:
        #   - scatter point click (push to nav_stack)
        #   - any KNN-row button click (push to nav_stack)
        #   - back_button click (pop from nav_stack)
        @app.callback(
            Output("meta", "children"),
            Output("frame_img", "src"),
            Output("lang", "children"),
            Output("knn_label", "children"),
            Output("knn_list", "children"),
            Output("nav_stack", "data"),
            Output("back_button", "disabled"),
            Input("scatter", "clickData"),
            Input(
                {
                    "role": "knn_row",
                    "hash": dash.ALL,
                    "frame": dash.ALL,
                    "token": dash.ALL,
                },
                "n_clicks",
            ),
            Input("back_button", "n_clicks"),
            State("nav_stack", "data"),
            State("run", "value"),
            State("layer", "value"),
            State("reduction", "value"),
            State("pca_drop_k", "value"),
        )
        def on_inspect(
            clickData,
            _knn_clicks,
            _back_clicks,
            nav_stack,
            run_path,
            layer,
            reduction,
            pca_drop_k,
        ):
            nav_stack = list(nav_stack or [])
            ctx = dash.callback_context
            if not ctx.triggered:
                return self.empty_payload()

            triggered_prop = ctx.triggered[0]["prop_id"]

            # ---- Branch 1: Back button ----
            if triggered_prop.startswith("back_button"):
                if len(nav_stack) <= 1:
                    # Nothing to go back to — clear and return empty pane.
                    return ("Click a point to inspect.", None, "", "", [], [], True)
                nav_stack.pop()  # discard current top
                prev = nav_stack[-1]  # peek new top (the previous frame)
                payload = self.build_inspect_payload(
                    run_path,
                    layer,
                    prev["hash"],
                    int(prev["frame"]),
                    int(prev["token"]),
                    prev["emb"],
                    reduction,
                    pca_drop_k=int(pca_drop_k or 0),
                )
                back_disabled = len(nav_stack) <= 1
                return (*payload, nav_stack, back_disabled)

            # ---- Branch 2: KNN row click ----
            new_pt = None
            if triggered_prop.startswith("{"):
                # Only treat as a real click if at least one knn_row n_clicks > 0;
                # otherwise this is the initial "all buttons render with n_clicks=0"
                # phantom trigger Dash sometimes emits.
                if not any(_knn_clicks or []):
                    if not clickData:
                        return self.empty_payload()
                else:
                    try:
                        import json

                        comp_id = json.loads(triggered_prop.split(".n_clicks")[0])
                        if comp_id.get("role") == "knn_row":
                            new_pt = {
                                "hash": comp_id["hash"],
                                "frame": int(comp_id["frame"]),
                                "token": int(comp_id["token"]),
                                "emb": self.resolve_emb(
                                    run_path,
                                    layer,
                                    comp_id["hash"],
                                    int(comp_id["frame"]),
                                    int(comp_id["token"]),
                                ),
                            }
                    except Exception as e:
                        return (
                            f"could not parse knn click: {e}",
                            None,
                            "",
                            "",
                            [],
                            nav_stack,
                            len(nav_stack) <= 1,
                        )

            # ---- Branch 3: scatter click ----
            if new_pt is None and triggered_prop.startswith("scatter"):
                if not clickData:
                    return self.empty_payload()
                try:
                    cd = clickData["points"][0]["customdata"]
                    video_hash, frame_idx, token_idx, emb = cd
                    new_pt = {
                        "hash": str(video_hash),
                        "frame": int(frame_idx),
                        "token": int(token_idx),
                        "emb": str(emb),
                    }
                except Exception as e:
                    return (
                        f"could not parse clickData: {e}",
                        None,
                        "",
                        "",
                        [],
                        nav_stack,
                        len(nav_stack) <= 1,
                    )

            if new_pt is None:
                # Trigger we don't recognize — keep state.
                return (
                    self.empty_payload()
                    if not nav_stack
                    else (
                        *self.build_inspect_payload(
                            run_path,
                            layer,
                            nav_stack[-1]["hash"],
                            int(nav_stack[-1]["frame"]),
                            int(nav_stack[-1]["token"]),
                            nav_stack[-1]["emb"],
                            reduction,
                            pca_drop_k=int(pca_drop_k or 0),
                        ),
                        nav_stack,
                        len(nav_stack) <= 1,
                    )
                )

            # Don't push a duplicate of the current top onto the stack.
            if not nav_stack or nav_stack[-1] != new_pt:
                nav_stack.append(new_pt)

            payload = self.build_inspect_payload(
                run_path,
                layer,
                new_pt["hash"],
                new_pt["frame"],
                new_pt["token"],
                new_pt["emb"],
                reduction,
                pca_drop_k=int(pca_drop_k or 0),
            )
            back_disabled = len(nav_stack) <= 1
            return (*payload, nav_stack, back_disabled)


class BrowserView:
    """Frame-grid + raw-key KNN browser."""

    def __init__(
        self,
        app,
        store: LayerStore,
        zarr_root: str,
        lang_key: str | None,
        image_key: str,
    ):
        self.app = app
        self.store = store
        self.zarr_root = zarr_root
        self.lang_key = lang_key
        self.image_key = image_key

    def browser_knn_buttons(self, neighbors):
        from dash import html

        rows = []
        for n in neighbors:
            short_hash = n["hash"][:22] + ("…" if len(n["hash"]) > 22 else "")
            lang_text, _ = load_language_prompt(
                self.zarr_root,
                n["hash"],
                n["frame_idx"],
                lang_key=self.lang_key,
            )
            lang_display = lang_text or "(no language prompt)"
            rows.append(
                html.Button(
                    children=html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "gap": "10px",
                            "width": "100%",
                        },
                        children=[
                            # Rank: plain number, muted, monospace — no chip.
                            html.Span(
                                f"{n['rank']:02d}",
                                style={
                                    "fontFamily": (
                                        "ui-monospace, SFMono-Regular, "
                                        "Menlo, Consolas, monospace"
                                    ),
                                    "fontSize": "11px",
                                    "color": MUTED,
                                    "fontWeight": 500,
                                    "minWidth": "20px",
                                    "textAlign": "right",
                                    "flexShrink": "0",
                                    "paddingTop": "2px",
                                },
                            ),
                            html.Img(
                                src=f"/thumbnail/{n['hash']}/{n['frame_idx']}",
                                style={
                                    "width": "72px",
                                    "height": "44px",
                                    "objectFit": "cover",
                                    "borderRadius": "4px",
                                    "background": CANVAS,
                                    "flexShrink": "0",
                                    "border": f"1px solid {BORDER}",
                                },
                            ),
                            html.Div(
                                style={
                                    "flex": "1",
                                    "textAlign": "left",
                                    "overflow": "hidden",
                                    "minWidth": "0",
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "2px",
                                },
                                children=[
                                    # Top line: dot + embodiment in plain text.
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "6px",
                                            "fontSize": "11px",
                                            "color": TEXT,
                                            "fontWeight": 500,
                                        },
                                        children=[
                                            html.Span(
                                                style={
                                                    "width": "5px",
                                                    "height": "5px",
                                                    "borderRadius": "999px",
                                                    "background": ACCENT,
                                                    "flexShrink": "0",
                                                },
                                            ),
                                            html.Span(n["embodiment"]),
                                            html.Span(
                                                f"d={n['distance']:.3f}",
                                                style={
                                                    "marginLeft": "auto",
                                                    "color": MUTED,
                                                    "fontFamily": (
                                                        "ui-monospace, "
                                                        "SFMono-Regular, Menlo, "
                                                        "Consolas, monospace"
                                                    ),
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        short_hash,
                                        style={
                                            "fontSize": "11px",
                                            "color": MUTED,
                                            "fontFamily": (
                                                "ui-monospace, SFMono-Regular, "
                                                "Menlo, Consolas, monospace"
                                            ),
                                            "overflow": "hidden",
                                            "textOverflow": "ellipsis",
                                            "whiteSpace": "nowrap",
                                        },
                                    ),
                                    html.Div(
                                        lang_display,
                                        style={
                                            "fontSize": "11px",
                                            "color": TEXT,
                                            "marginTop": "2px",
                                            "lineHeight": "1.35",
                                            "whiteSpace": "normal",
                                            "overflowWrap": "anywhere",
                                        },
                                        title=lang_display,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    id={
                        "role": "browser_knn",
                        "hash": n["hash"],
                        "frame": n["frame_idx"],
                        "token": n["token_idx"],
                    },
                    n_clicks=0,
                    style={
                        "padding": "8px 10px",
                        "background": PANEL,
                        "border": f"1px solid {BORDER}",
                        "cursor": "pointer",
                        "borderRadius": "6px",
                        "textAlign": "left",
                        "transition": "border-color 0.12s ease",
                    },
                )
            )
        return rows

    def render_browser_detail(
        self,
        run_path,
        layer,
        h,
        f,
        tok,
        knn_space="raw",
        pca_drop_k=0,
        pair_mode="cross",
    ):
        """Build (meta, img_src, lang_display, knn_label, knn_buttons) for one
        clicked/popped (hash, frame, token). `knn_space` is 'raw' (full-D
        keys) or 'pca' (50-d PCA features fitted on the layer's keys).
        `pca_drop_k` > 0 removes the top-k PCA dims ranked by aria-vs-eva
        Fisher score from the KNN space: dropped columns in 'pca' mode,
        projected-out component directions in 'raw' mode. `pair_mode` is
        'cross' (neighbors from OTHER embodiments — default) or 'same'
        (neighbors from the clicked frame's own embodiment, self excluded)."""
        import time

        from dash import html

        _t_overall = time.perf_counter()
        _t_phases: dict[str, float] = {}

        def _phase(name: str, t0: float):
            _t_phases[name] = time.perf_counter() - t0

        pca_drop_k = int(pca_drop_k or 0)
        pair_mode = pair_mode or "cross"
        logger.info(
            "render_browser_detail ENTER layer=%s h=%s f=%s tok=%s knn_space=%s drop_k=%d pairs=%s",
            layer,
            h,
            f,
            tok,
            knn_space,
            pca_drop_k,
            pair_mode,
        )

        t0 = time.perf_counter()
        img_src, _ = load_image_b64(self.zarr_root, h, f, self.image_key)
        _phase("load_image", t0)
        t0 = time.perf_counter()
        lang_text, _ = load_language_prompt(
            self.zarr_root, h, f, lang_key=self.lang_key
        )
        _phase("load_lang", t0)
        lang_display = lang_text if lang_text else "(no language prompt found)"

        if not run_path or not layer:
            return (
                f"video_hash : {h}\nframe_idx  : {f}\ntoken_idx  : {tok}",
                img_src,
                lang_display,
                "",
                [html.Div("(no run/layer selected)", style={"color": "#94a3b8"})],
            )

        try:
            data = self.store.load(run_path, layer)
        except Exception as e:
            return (
                f"video_hash : {h}\nframe_idx  : {f}\ntoken_idx  : {tok}",
                img_src,
                lang_display,
                "",
                [html.Div(f"(load error: {e})", style={"color": "#b91c1c"})],
            )

        t0 = time.perf_counter()
        m = (
            (data["hashes"] == h)
            & (data["frame_idx"] == f)
            & (data["token_idx"] == tok)
        )
        where = np.where(m)[0]
        src_idx = int(where[0]) if len(where) else None
        src_emb = str(data["embs"][src_idx]) if src_idx is not None else "?"
        _phase("locate_row", t0)
        meta = (
            f"video_hash : {h}\n"
            f"frame_idx  : {f}\n"
            f"token_idx  : {tok}\n"
            f"embodiment : {src_emb}"
        )

        # Fast path: eval_latent precomputed cross-embodiment KNN for the
        # raw-key space and dumped it as `<layer>_knn.pt`. When present
        # AND aligned with the current CSV, we skip the full-D distance
        # scan entirely. PCA mode still computes on demand because the
        # n_components knob is inspector-side. PCA removal and
        # same-embodiment pairs also bypass this path — the precomputed
        # neighbors were built cross-embodiment in the full raw space.
        if (
            knn_space != "pca"
            and pca_drop_k <= 0
            and pair_mode != "same"
            and src_idx is not None
        ):
            t_knn = time.perf_counter()
            knn_pre = self.store.load_knn(run_path, layer)
            _phase("load_knn", t_knn)
            if knn_pre is not None and knn_pre["indices"].shape[0] == len(
                data["hashes"]
            ):
                # Slice ONE row out of the mmap-backed tensors, then
                # materialize just that K-element row to numpy. This is
                # the per-click-O(K) path — the (N, K) tensors stay on
                # disk except for whatever the OS page cache fetches.
                t_knn = time.perf_counter()
                pre_idx = np.asarray(knn_pre["indices"][src_idx], dtype=np.int32)
                pre_dist = np.asarray(knn_pre["distances"][src_idx], dtype=np.float32)
                # Drop padded slots (eval_latent uses dist=+inf, idx=-1
                # when fewer cross-emb candidates than K exist).
                valid = (pre_idx >= 0) & np.isfinite(pre_dist)
                pre_idx = pre_idx[valid]
                pre_dist = pre_dist[valid]
                _phase("knn_slice", t_knn)
                space_label = f"raw (precomputed K={knn_pre['k']})"
                if pre_idx.size == 0:
                    knn_label = f"10 closest in {space_label}"
                    knn_buttons = [
                        html.Div(
                            f"(no rows from a different embodiment than "
                            f"'{src_emb}' in this layer)",
                            style={"color": "#94a3b8"},
                        )
                    ]
                    phase_str = " ".join(
                        f"{n}={dt*1000:.0f}ms" for n, dt in _t_phases.items()
                    )
                    logger.info(
                        "render_browser_detail (precomputed-KNN path) %s in %.2fs | %s",
                        layer,
                        time.perf_counter() - _t_overall,
                        phase_str,
                    )
                    return meta, img_src, lang_display, knn_label, knn_buttons
                # One row per action instance. The sidecar stores only the
                # K (=8) globally-nearest rows, which usually collapse to a
                # couple of distinct actions — only take this fast path when
                # it can still fill the list; otherwise fall through to the
                # full computed scan below.
                cand = (
                    (float(d), int(o))
                    for d, o in zip(pre_dist.tolist(), pre_idx.tolist())
                )
                deduped = _dedupe_neighbors_by_action(
                    self.zarr_root, data, cand, 10, cache_key=(run_path, layer)
                )
                if len(deduped) >= 10:
                    # Paired-action distance (raw space — same space the
                    # precomputed neighbors live in). Row-only gather from
                    # the mmap-backed keys keeps this path per-click cheap.
                    pair = _find_pair_action(self.zarr_root, data, h, f, src_emb)
                    if pair is not None:
                        twin, s2, e2, pair_prompt = pair
                        pair_rows = _pair_action_rows(data, twin, s2, e2)
                        keys_arr = (
                            self.store.load_keys(run_path, layer)
                            if pair_rows.size
                            else None
                        )
                        if keys_arr is not None and keys_arr.shape[0] == len(
                            data["hashes"]
                        ):
                            d = np.linalg.norm(
                                np.asarray(keys_arr[pair_rows], dtype=np.float32)
                                - np.asarray(keys_arr[src_idx], dtype=np.float32),
                                axis=1,
                            )
                            lang_display = lang_display + _pair_distance_text(
                                twin,
                                s2,
                                e2,
                                pair_prompt,
                                d,
                                pair_rows,
                                np.asarray(data["frame_idx"]),
                                "raw",
                            )
                    neighbors = []
                    for rank, (dist, o) in enumerate(deduped, start=1):
                        neighbors.append(
                            {
                                "rank": rank,
                                "hash": str(data["hashes"][o]),
                                "frame_idx": int(data["frame_idx"][o]),
                                "token_idx": int(data["token_idx"][o]),
                                "embodiment": str(data["embs"][o]),
                                "distance": dist,
                            }
                        )
                    knn_label = (
                        f"{len(neighbors)} closest CROSS-embodiment in "
                        f"{space_label} (source='{src_emb}', 1 per action)"
                    )
                    knn_buttons = self.browser_knn_buttons(neighbors)
                    phase_str = " ".join(
                        f"{n}={dt*1000:.0f}ms" for n, dt in _t_phases.items()
                    )
                    logger.info(
                        "render_browser_detail (precomputed-KNN path) %s in %.2fs | %s",
                        layer,
                        time.perf_counter() - _t_overall,
                        phase_str,
                    )
                    return meta, img_src, lang_display, knn_label, knn_buttons

        # `drop_dirs` is only set in raw mode with removal active: the
        # (drop_k, D) component directions whose contribution gets
        # subtracted from the raw distances below.
        drop_dirs = None
        if knn_space == "pca":
            feats = self.store.pca_features_dropped(
                run_path, layer, n_components=50, drop_k=pca_drop_k
            )
            space_label = (
                f"PCA-50 −top{pca_drop_k} Fisher dims" if pca_drop_k > 0 else "PCA-50"
            )
            missing_msg = (
                f"({layer}_keys.pt missing — PCA features can't be "
                f"fitted; re-run eval with +force_reeval=true to "
                f"write raw keys)"
            )
            if pca_drop_k > 0 and feats is not None and feats.shape[1] == 0:
                feats = None
                missing_msg = "(all PCA dims removed — lower the slider)"
        else:
            feats = self.store.load_keys(run_path, layer)
            space_label = f"raw D={feats.shape[1]}" if feats is not None else "raw keys"
            missing_msg = (
                f"({layer}_keys.pt missing — re-run eval with "
                f"+force_reeval=true to write raw keys "
                f"(KNN is bundled inside the same file))"
            )
            if pca_drop_k > 0 and feats is not None:
                drop_dirs = self.store.fisher_dropped_directions(
                    run_path, layer, n_components=50, drop_k=pca_drop_k
                )
                if drop_dirs is None:
                    feats = None
                    missing_msg = (
                        "(Fisher ranking unavailable — needs PCA on raw keys "
                        "and both aria/eva embodiments in this layer)"
                    )
                else:
                    space_label += f" −top{pca_drop_k} Fisher PCA dims"

        if feats is None:
            knn_label = f"10 closest in {space_label}"
            knn_buttons = [html.Div(missing_msg, style={"color": "#b91c1c"})]
        elif src_idx is None:
            knn_label = f"10 closest in {space_label}"
            knn_buttons = [
                html.Div(
                    "(could not find clicked row in this layer's data)",
                    style={"color": "#b91c1c"},
                )
            ]
        elif feats.shape[0] != len(data["hashes"]):
            knn_label = f"10 closest in {space_label}"
            knn_buttons = [
                html.Div(
                    f"(features shape {feats.shape} mismatches row count "
                    f"{len(data['hashes'])})",
                    style={"color": "#b91c1c"},
                )
            ]
        else:
            diff = feats - feats[src_idx]
            if drop_dirs is not None:
                # Raw-space removal: subtract the dropped components'
                # contribution from the squared distances instead of
                # materializing a filtered copy of the (N, D) keys.
                d2 = np.einsum("ij,ij->i", diff, diff)
                proj = diff @ drop_dirs.T
                d2 -= np.einsum("ij,ij->i", proj, proj)
                dists = np.sqrt(np.clip(d2, 0.0, None))
            else:
                dists = np.linalg.norm(diff, axis=1)
            # Paired-action distance, appended below the clicked frame's
            # language. Computed before the pair-mode masking so the twin's
            # (opposite-embodiment) rows still hold finite distances.
            pair = _find_pair_action(self.zarr_root, data, h, f, src_emb)
            if pair is not None:
                twin, s2, e2, pair_prompt = pair
                pair_rows = _pair_action_rows(data, twin, s2, e2)
                if pair_rows.size:
                    lang_display = lang_display + _pair_distance_text(
                        twin,
                        s2,
                        e2,
                        pair_prompt,
                        dists[pair_rows],
                        pair_rows,
                        np.asarray(data["frame_idx"]),
                        space_label,
                    )
            # Candidate masking by pair mode. Cross (default): mask self AND
            # every row whose embodiment matches the clicked frame's —
            # without this the neighbors are dominated by frames from the
            # same recording type, which defeats the point of the
            # visualization. Same: keep only the clicked frame's embodiment,
            # masking the clicked row itself.
            if pair_mode == "same":
                mask_out = np.asarray(data["embs"] != src_emb)
                mask_out[src_idx] = True
                # Same-mode results are not action-deduped, so apply the
                # annotation filter directly: rows whose frame is outside
                # every annotation span (action id -1) never surface.
                mask_out |= (
                    _row_action_ids(self.zarr_root, data, cache_key=(run_path, layer))
                    < 0
                )
                # Exclude the clicked frame's own ACTION: rows from the same
                # recording whose frame falls inside any annotation interval
                # covering the clicked frame. Otherwise the list is just the
                # temporally adjacent frames of the same motion. When no
                # interval covers the frame (or annotations are unreadable),
                # fall back to excluding the whole recording.
                same_hash = np.asarray(data["hashes"] == h)
                intervals = annotation_intervals(self.zarr_root, h)
                spans = [(s, e) for s, e, *_ in intervals if s <= f < e]
                if spans:
                    frames = np.asarray(data["frame_idx"])
                    in_action = np.zeros(len(frames), dtype=bool)
                    for s, e in spans:
                        in_action |= (frames >= s) & (frames < e)
                    mask_out |= same_hash & in_action
                else:
                    mask_out |= same_hash
                pair_label = "SAME"
                pair_suffix = ", same action excluded"
                empty_msg = (
                    f"(no rows from embodiment '{src_emb}' outside the "
                    f"clicked action in this layer)"
                )
            else:
                mask_out = np.asarray(data["embs"] == src_emb)
                pair_label = "CROSS"
                pair_suffix = ", 1 per action"
                empty_msg = (
                    f"(no rows from a different embodiment than '{src_emb}' "
                    f"in this layer)"
                )
            dists[mask_out] = np.inf
            n_candidates = int((~mask_out).sum())
            if n_candidates == 0:
                knn_label = f"10 closest in {space_label}"
                knn_buttons = [
                    html.Div(
                        empty_msg,
                        style={"color": "#94a3b8"},
                    )
                ]
            else:
                order = np.argsort(dists)
                if pair_mode == "same":
                    k = min(10, n_candidates)
                    picked = [(float(dists[int(o)]), int(o)) for o in order[:k]]
                else:
                    # Cross mode: at most one row per action instance —
                    # otherwise the list is 10 temporally-adjacent frames
                    # of a single opposite-embodiment motion. argsort puts
                    # the masked (inf) rows last, so capping the walk at
                    # n_candidates only visits valid rows.
                    cand = (
                        (float(dists[int(o)]), int(o)) for o in order[:n_candidates]
                    )
                    picked = _dedupe_neighbors_by_action(
                        self.zarr_root, data, cand, 10, cache_key=(run_path, layer)
                    )
                neighbors = []
                for rank, (dist, o) in enumerate(picked, start=1):
                    neighbors.append(
                        {
                            "rank": rank,
                            "hash": str(data["hashes"][o]),
                            "frame_idx": int(data["frame_idx"][o]),
                            "token_idx": int(data["token_idx"][o]),
                            "embodiment": str(data["embs"][o]),
                            "distance": dist,
                        }
                    )
                knn_label = (
                    f"10 closest {pair_label}-embodiment in {space_label} "
                    f"(source='{src_emb}'{pair_suffix})"
                )
                knn_buttons = self.browser_knn_buttons(neighbors)

        phase_str = " ".join(f"{n}={dt*1000:.0f}ms" for n, dt in _t_phases.items())
        logger.info(
            "render_browser_detail (compute-KNN path) %s in %.2fs | %s",
            layer,
            time.perf_counter() - _t_overall,
            phase_str,
        )
        return meta, img_src, lang_display, knn_label, knn_buttons

    def register(self):
        """Wire all browser-mode Dash callbacks."""
        import dash
        from dash import Input, Output, State, html

        app = self.app

        @app.callback(
            Output("scatter_view", "style"),
            Output("knn_browser_view", "style"),
            Output("scatter_controls", "style"),
            Output("apply", "style"),
            Output("browser_knn_space_card", "style"),
            Input("view_mode", "value"),
            State("scatter_view", "style"),
            State("knn_browser_view", "style"),
            State("scatter_controls", "style"),
            State("apply", "style"),
            State("browser_knn_space_card", "style"),
        )
        def _toggle_view(
            mode,
            scatter_style,
            browser_style,
            ctrls_style,
            apply_style,
            knn_space_card_style,
        ):
            s = dict(scatter_style or {})
            b = dict(browser_style or {})
            c = dict(ctrls_style or {})
            a = dict(apply_style or {})
            ks = dict(knn_space_card_style or {})
            if mode == "browser":
                s["display"] = "none"
                b["display"] = "flex"
                c["display"] = "none"
                a["display"] = "none"
                ks["display"] = "block"
            else:
                s["display"] = "flex"
                b["display"] = "none"
                c["display"] = "block"
                a["display"] = "block"
                ks["display"] = "none"
            return s, b, c, a, ks

        # Populate the frame list when entering browser mode (or when run/layer
        # change while already in browser mode). Skipped while in scatter mode
        # so we don't pay the dedup cost for nothing.
        # When run/layer/view changes, we always reset visible_count to 60 so
        # the user starts at the top with a small number of thumbnails. The
        # Load-more callback below bumps it up by 60 each click.
        @app.callback(
            Output("browser_visible_count", "data", allow_duplicate=True),
            Input("run", "value"),
            Input("layer", "value"),
            Input("view_mode", "value"),
            Input("browser_hide_home", "value"),
            Input("browser_lang_exclude", "value"),
            Input("browser_shuffle", "value"),
            prevent_initial_call=True,
        )
        def _reset_visible_count(*_):
            return 60

        @app.callback(
            Output("browser_frame_list", "children"),
            Output("browser_frame_count", "children"),
            Input("run", "value"),
            Input("layer", "value"),
            Input("view_mode", "value"),
            Input("browser_visible_count", "data"),
            Input("browser_hide_home", "value"),
            Input("browser_lang_exclude", "value"),
            Input("browser_shuffle", "value"),
        )
        def _populate_browser_list(
            run_path,
            layer,
            mode,
            visible_count,
            hide_home_val,
            lang_exclude_val,
            shuffle_val,
        ):
            import time as _time

            _t_pop_start = _time.perf_counter()
            if mode != "browser" or not run_path or not layer:
                return dash.no_update, dash.no_update
            try:
                visible_count = int(visible_count) if visible_count else 60
            except Exception:
                visible_count = 60
            hide_home = bool(hide_home_val and "on" in hide_home_val)
            excludes = []
            if hide_home:
                excludes.append("home")
            custom = (lang_exclude_val or "").strip().lower()
            if custom:
                excludes.append(custom)
            logger.info(
                "_populate_browser_list ENTER layer=%s visible=%d excludes=%s",
                layer,
                visible_count,
                excludes,
            )
            try:
                data = self.store.load(run_path, layer)
            except Exception as e:
                return [html.Div(f"(load error: {e})", style={"color": "#b91c1c"})], ""
            logger.info(
                "_populate_browser_list after store.load: N=%d (%.2fs elapsed)",
                len(data["hashes"]),
                _time.perf_counter() - _t_pop_start,
            )

            # Dedupe by (hash, frame_idx) — pick the first token in CSV order.
            # Vectorized: at 10M rows the per-row Python loop took 20-60s.
            # Strategy: encode each (hash, frame) pair as a single int64 key
            # (hash -> small int code via np.unique, frame fits in int32), then
            # let np.unique return the indices of first occurrences.
            _t = _time.perf_counter()
            hashes_arr = np.asarray(data["hashes"])
            frames_arr = np.asarray(data["frame_idx"], dtype=np.int64)
            tokens_arr = np.asarray(data["token_idx"], dtype=np.int64)
            embs_arr = np.asarray(data["embs"])
            logger.info(
                "_populate_browser_list np.asarray done in %.2fs",
                _time.perf_counter() - _t,
            )
            if len(hashes_arr) == 0:
                seen: dict[tuple[str, int], tuple[int, str]] = {}
            else:
                unique_h, hash_codes = np.unique(hashes_arr, return_inverse=True)
                # Pack (hash_code, frame) into one int64. Frame ids are < 1e9
                # in practice; multiplier guards against collisions if not.
                FRAME_MULT = np.int64(1_000_000_000)
                combined = hash_codes.astype(np.int64) * FRAME_MULT + frames_arr
                # return_index gives the first index for each unique key. Then
                # re-sort to original CSV order so the round-robin below is
                # deterministic w.r.t. the file.
                _, first_pos = np.unique(combined, return_index=True)
                first_pos.sort()
                _t = _time.perf_counter()
                seen = {
                    (str(hashes_arr[i]), int(frames_arr[i])): (
                        int(tokens_arr[i]),
                        str(embs_arr[i]),
                    )
                    for i in first_pos.tolist()
                }
                logger.info(
                    "_populate_browser_list dedupe done: unique_pairs=%d in %.2fs",
                    len(seen),
                    _time.perf_counter() - _t,
                )

            # Round-robin across recordings so the first screen-full of cards
            # comes from many different videos instead of frame=32, 33, 34… of
            # the same one. Within a recording, frames stay in order; recording
            # order is hashed (deterministic across reloads).
            from itertools import zip_longest as _zl

            groups: dict[str, list] = {}
            for (h, f), (tok, emb) in seen.items():
                groups.setdefault(h, []).append((h, f, tok, emb))
            for h in groups:
                groups[h].sort(key=lambda x: x[1])  # frames in order within a recording
            ordered_hashes = sorted(groups.keys(), key=lambda h: (groups[h][0][3], h))
            group_lists = [groups[h] for h in ordered_hashes]
            items = []
            for batch in _zl(*group_lists):
                for entry in batch:
                    if entry is not None:
                        h, f, tok, emb = entry
                        items.append(((h, f), (tok, emb)))
            logger.info(
                "_populate_browser_list grouping/ordering done: items=%d (%.2fs elapsed)",
                len(items),
                _time.perf_counter() - _t_pop_start,
            )
            # Optional shuffle of the browse order. Fixed seed: the full list
            # is rebuilt on every callback (incl. Load-more re-fires), so a
            # stable permutation keeps already-rendered cards in place while
            # paging. Applied before the lang-filter slicing below so the
            # filtered window matches what gets rendered.
            if shuffle_val and "on" in shuffle_val and items:
                rng = np.random.default_rng(12345)
                items = [items[int(i)] for i in rng.permutation(len(items))]
            # Apply lang filter, if any. KEY OPTIMIZATION: only filter the
            # `visible_count + FILTER_BUFFER` items we're about to render —
            # NOT the whole 249-card list. Each filter hit costs a zarr open
            # (~1.4s on this filesystem); doing 249 of them on first paint
            # blocks the UI for minutes when we'll only ever show 60 cards.
            # Tail items stay unfiltered until the user clicks "Load more",
            # at which point `visible_count` bumps, this callback re-fires,
            # and the new slice gets filtered (with warm caches from the
            # previous call).
            FILTER_BUFFER = 40
            n_filtered_out = 0
            if excludes:
                _t = _time.perf_counter()
                filter_target = visible_count + FILTER_BUFFER
                items_to_filter = items[:filter_target]
                items_remaining = items[filter_target:]
                unique_hashes = list({h for (h, _), _ in items_to_filter})
                # zarr-open is I/O-bound (each call hits a separate zarr group
                # dir on the network FS), and annotation_intervals is cached
                # per-hash. Fan out the cold calls across threads so 249 hashes
                # finish in ~one I/O batch instead of 249 serial roundtrips.
                from concurrent.futures import ThreadPoolExecutor, as_completed

                intervals_by_hash: dict = {}
                workers = min(32, max(4, len(unique_hashes)))
                logger.info(
                    "_populate_browser_list lang-filter pre-resolve START: "
                    "n_hashes=%d workers=%d",
                    len(unique_hashes),
                    workers,
                )
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_to_hash = {
                        ex.submit(annotation_intervals, self.zarr_root, h): h
                        for h in unique_hashes
                    }
                    done_n = 0
                    last_log = _time.perf_counter()
                    for fut in as_completed(fut_to_hash):
                        h = fut_to_hash[fut]
                        try:
                            intervals_by_hash[h] = fut.result()
                        except Exception as e:
                            logger.warning("pre-resolve failed for hash=%s: %s", h, e)
                            intervals_by_hash[h] = tuple()
                        done_n += 1
                        # Heartbeat every ~2s OR every 25 hashes — whichever
                        # comes first — so a stalled batch is visible in logs.
                        now = _time.perf_counter()
                        if done_n % 25 == 0 or (now - last_log) >= 2.0:
                            logger.info(
                                "_populate_browser_list pre-resolve progress: "
                                "%d/%d hashes (%.2fs elapsed)",
                                done_n,
                                len(unique_hashes),
                                now - _t,
                            )
                            last_log = now
                logger.info(
                    "_populate_browser_list resolved %d unique-hash intervals in %.2fs (workers=%d)  "
                    "[filtered slice: %d/%d items]",
                    len(intervals_by_hash),
                    _time.perf_counter() - _t,
                    workers,
                    len(items_to_filter),
                    len(items),
                )
                _t = _time.perf_counter()
                kept = []
                for (h, f), v in items_to_filter:
                    intervals = intervals_by_hash.get(h, tuple())
                    text = all_lang_concat_lower(intervals, f) if intervals else ""
                    if any(sub in text for sub in excludes):
                        n_filtered_out += 1
                        continue
                    kept.append(((h, f), v))
                # Filtered head + unfiltered tail. The tail is only ever shown
                # if the user scrolls / clicks "Load more", which re-fires this
                # callback with a larger `visible_count` and re-runs the filter
                # on the new (larger) slice.
                items = kept + items_remaining
                logger.info(
                    "_populate_browser_list lang filter done: kept=%d dropped=%d (tail unfiltered=%d) in %.2fs",
                    len(kept),
                    n_filtered_out,
                    len(items_remaining),
                    _time.perf_counter() - _t,
                )

            MAX_ROWS = 5000
            total_items = len(items)
            truncated = total_items > MAX_ROWS
            items = items[:MAX_ROWS]

            # Render the first `visible_count` cards. Anything beyond stays
            # uninstantiated until the Load-more sentinel scrolls into view.
            visible_items = items[:visible_count]

            rows = []
            for (h, f), (tok, emb) in visible_items:
                short_hash = h[:18] + ("…" if len(h) > 18 else "")
                rows.append(
                    html.Button(
                        children=html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "4px",
                                "width": "100%",
                            },
                            children=[
                                # Thumbnail (served on-demand by the /thumbnail route).
                                # `loading="lazy"` was dropped — Dash 4.1's
                                # html.Img doesn't accept the attribute. The
                                # browser still HTTP-caches each thumbnail via
                                # the Flask route, so re-renders are cheap.
                                html.Img(
                                    src=f"/thumbnail/{h}/{f}",
                                    style={
                                        "width": "100%",
                                        "aspectRatio": "16/10",
                                        "objectFit": "cover",
                                        "borderRadius": "4px",
                                        "background": CANVAS,
                                        "display": "block",
                                        "marginBottom": "6px",
                                    },
                                ),
                                # Minimal / refined: dot + plain text instead
                                # of a colored pill. Monospace for identifiers
                                # so hashes & numeric metadata feel anchored.
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "8px",
                                        "fontSize": "12px",
                                        "color": TEXT,
                                        "fontWeight": 500,
                                    },
                                    children=[
                                        html.Span(
                                            style={
                                                "width": "6px",
                                                "height": "6px",
                                                "borderRadius": "999px",
                                                "background": ACCENT,
                                                "flexShrink": "0",
                                            },
                                        ),
                                        html.Span(emb),
                                    ],
                                ),
                                html.Div(
                                    short_hash,
                                    style={
                                        "fontSize": "11px",
                                        "color": MUTED,
                                        "fontFamily": (
                                            "ui-monospace, SFMono-Regular, "
                                            "Menlo, Consolas, monospace"
                                        ),
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                        "whiteSpace": "nowrap",
                                    },
                                ),
                                html.Div(
                                    f"f={f}  ·  tok={tok}",
                                    style={
                                        "fontSize": "11px",
                                        "color": MUTED,
                                        "fontFamily": (
                                            "ui-monospace, SFMono-Regular, "
                                            "Menlo, Consolas, monospace"
                                        ),
                                    },
                                ),
                            ],
                        ),
                        id={"role": "browser_row", "hash": h, "frame": f, "token": tok},
                        n_clicks=0,
                        style={
                            "padding": "10px",
                            "background": PANEL,
                            "border": f"1px solid {BORDER}",
                            "cursor": "pointer",
                            "borderRadius": "6px",
                            "textAlign": "left",
                            "transition": "border-color 0.12s ease",
                        },
                    )
                )
            # Sentinel: a "Load more" button at the end of the rendered cards.
            # The clientside callback below uses IntersectionObserver to click
            # this when it scrolls into view (≈ infinite scroll). The button is
            # also clickable manually as a fallback.
            rendered = len(visible_items)
            if rendered < len(items):
                remaining = len(items) - rendered
                rows.append(
                    html.Button(
                        f"Load more ({remaining} remaining)",
                        id="browser_load_more",
                        n_clicks=0,
                        style={
                            "gridColumn": "1 / -1",  # span the full grid row
                            "padding": "12px",
                            "background": "white",
                            "color": MUTED,
                            "border": f"1px dashed {BORDER}",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "fontSize": "12px",
                        },
                    )
                )

            count_msg = f"{len(seen)} unique frames"
            if n_filtered_out:
                count_msg += f"  ·  {n_filtered_out} hidden by lang filter"
            if truncated:
                count_msg += f"  ·  capped at {MAX_ROWS}"
            count_msg += f"  ·  showing {rendered}/{len(items)}"
            logger.info(
                "_populate_browser_list DONE: rendered=%d total_items=%d in %.2fs",
                rendered,
                len(items),
                _time.perf_counter() - _t_pop_start,
            )
            return rows, count_msg

        # Click handler: a row in the frame list, a row in the raw-KNN list,
        # or the Back button. All three update the detail pane + nav stack.
        # The stack lets the user drill into a neighbor and step back out.
        @app.callback(
            Output("browser_meta", "children"),
            Output("browser_img", "src"),
            Output("browser_lang", "children"),
            Output("browser_knn_label", "children"),
            Output("browser_knn_list", "children"),
            Output("browser_nav_stack", "data"),
            Output("browser_back", "disabled"),
            Input(
                {
                    "role": "browser_row",
                    "hash": dash.ALL,
                    "frame": dash.ALL,
                    "token": dash.ALL,
                },
                "n_clicks",
            ),
            Input(
                {
                    "role": "browser_knn",
                    "hash": dash.ALL,
                    "frame": dash.ALL,
                    "token": dash.ALL,
                },
                "n_clicks",
            ),
            Input("browser_back", "n_clicks"),
            Input("browser_knn_space", "value"),
            Input("pca_drop_k", "value"),
            Input("browser_knn_pairs", "value"),
            State("browser_nav_stack", "data"),
            State("run", "value"),
            State("layer", "value"),
            prevent_initial_call=True,
        )
        def _on_browser_click(
            _row_clicks,
            _knn_clicks,
            _back_clicks,
            knn_space,
            pca_drop_k,
            knn_pairs,
            nav_stack,
            run_path,
            layer,
        ):
            nav_stack = list(nav_stack or [])
            ctx = dash.callback_context
            if not ctx.triggered:
                logger.info("_on_browser_click fired with no trigger — bailing")
                return (dash.no_update,) * 7
            triggered = ctx.triggered[0]["prop_id"]
            logger.info(
                "_on_browser_click trigger=%s knn_space=%s drop_k=%s pairs=%s run=%s layer=%s nav_depth=%d",
                triggered,
                knn_space,
                pca_drop_k,
                knn_pairs,
                os.path.basename(run_path) if run_path else None,
                layer,
                len(nav_stack),
            )

            # ---- Branch 0: KNN-space radio, PCA-removal slider, or pair-mode
            # toggle changed ----
            # Re-render the current selection (if any) using the new space.
            # No nav_stack mutation.
            if triggered.startswith(
                ("browser_knn_space", "pca_drop_k", "browser_knn_pairs")
            ):
                if not nav_stack:
                    return (dash.no_update,) * 7
                cur = nav_stack[-1]
                payload = self.render_browser_detail(
                    run_path,
                    layer,
                    str(cur["hash"]),
                    int(cur["frame"]),
                    int(cur["token"]),
                    knn_space=knn_space,
                    pca_drop_k=pca_drop_k,
                    pair_mode=knn_pairs,
                )
                return (*payload, nav_stack, len(nav_stack) <= 1)

            # ---- Branch 1: Back button ----
            if triggered.startswith("browser_back"):
                if len(nav_stack) <= 1:
                    # Nothing to go back to.
                    return (
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        dash.no_update,
                        nav_stack,
                        True,
                    )
                nav_stack.pop()
                prev = nav_stack[-1]
                payload = self.render_browser_detail(
                    run_path,
                    layer,
                    str(prev["hash"]),
                    int(prev["frame"]),
                    int(prev["token"]),
                    knn_space=knn_space,
                    pca_drop_k=pca_drop_k,
                    pair_mode=knn_pairs,
                )
                return (*payload, nav_stack, len(nav_stack) <= 1)

            # ---- Branch 2/3: row click or KNN click ----
            if not triggered.startswith("{"):
                return (dash.no_update,) * 7

            # Dash fires a phantom 0-click for every freshly-rendered button on
            # the first frame; bail out unless someone actually clicked.
            all_clicks = list(_row_clicks or []) + list(_knn_clicks or [])
            if not any(c for c in all_clicks if c and c > 0):
                return (dash.no_update,) * 7

            try:
                import json

                comp_id = json.loads(triggered.split(".n_clicks")[0])
                if comp_id.get("role") not in ("browser_row", "browser_knn"):
                    return (dash.no_update,) * 7
                h = str(comp_id["hash"])
                f = int(comp_id["frame"])
                tok = int(comp_id["token"])
            except Exception as e:
                return (
                    f"parse error: {e}",
                    None,
                    "",
                    "",
                    [],
                    nav_stack,
                    len(nav_stack) <= 1,
                )

            new_pt = {"hash": h, "frame": f, "token": tok}
            if not nav_stack or nav_stack[-1] != new_pt:
                nav_stack.append(new_pt)

            payload = self.render_browser_detail(
                run_path,
                layer,
                h,
                f,
                tok,
                knn_space=knn_space,
                pca_drop_k=pca_drop_k,
                pair_mode=knn_pairs,
            )
            return (*payload, nav_stack, len(nav_stack) <= 1)

        # Click "Load more" → bump visible_count by 60. The populate callback
        # re-renders the list with more cards and a fresh sentinel.
        @app.callback(
            Output("browser_visible_count", "data", allow_duplicate=True),
            Input("browser_load_more", "n_clicks"),
            State("browser_visible_count", "data"),
            prevent_initial_call=True,
        )
        def _on_load_more(n_clicks, current):
            if not n_clicks:
                return dash.no_update
            return (int(current) if current else 60) + 60

        # Clientside: after the frame list updates, attach an
        # IntersectionObserver to the Load-more sentinel so it auto-clicks
        # itself when scrolled into view. Result is infinite scroll: cards
        # load 60 at a time as the user scrolls, with no manual click needed.
        # Falls back gracefully if the button is absent (last page) or if
        # IntersectionObserver isn't supported (very old browsers).
        app.clientside_callback(
            """
            function(_children) {
                setTimeout(function() {
                    const btn = document.getElementById('browser_load_more');
                    if (!btn) return;
                    if (btn.dataset.observed === '1') return;
                    btn.dataset.observed = '1';
                    if (typeof IntersectionObserver === 'undefined') return;
                    const obs = new IntersectionObserver(function(entries) {
                        entries.forEach(function(entry) {
                            if (entry.isIntersecting) {
                                obs.disconnect();
                                entry.target.click();
                            }
                        });
                    }, {rootMargin: '400px'});
                    obs.observe(btn);
                }, 50);
                return window.dash_clientside.no_update;
            }
            """,
            Output("browser_load_more", "title"),
            Input("browser_frame_list", "children"),
            prevent_initial_call=True,
        )
