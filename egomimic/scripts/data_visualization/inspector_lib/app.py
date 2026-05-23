"""Top-level `build_app` factory.

Wires up the LayerStore caches, the /thumbnail Flask route, the scatter
and KNN-browser views, and the Dash layout. Returns a configured
`dash.Dash` instance ready for `app.run(...)`.
"""

from __future__ import annotations

import logging

from .caches import LayerStore
from .thumbnails import (
    ThumbnailService,
)
from .views import (
    ACCENT,
    BORDER,
    CANVAS,
    CARD_STYLE,
    LABEL_STYLE,
    MUTED,
    PANEL,
    TEXT,
    BrowserView,
    ScatterView,
)

logger = logging.getLogger(__name__)


def build_app(
    runs: list[tuple[str, str]],
    zarr_root: str,
    image_key: str,
    default_sample: int,
    lang_key: str | None = None,
):
    """`runs` is a list of (display_name, abs_path_to_epoch_dir) — the user
    picks one from a dropdown, and the layers in that dir become available
    in the Layer dropdown."""
    import dash
    from dash import dcc, html

    if not runs:
        raise SystemExit("No runs found.")

    store = LayerStore()
    thumb_service = ThumbnailService(zarr_root=zarr_root, image_key=image_key)
    app = dash.Dash(__name__, title="Latent Inspector")

    scatter = ScatterView(
        app=app,
        store=store,
        zarr_root=zarr_root,
        lang_key=lang_key,
        image_key=image_key,
        default_sample=default_sample,
    )
    browser = BrowserView(
        app=app,
        store=store,
        zarr_root=zarr_root,
        lang_key=lang_key,
        image_key=image_key,
    )

    # ----- Thumbnail route -------------------------------------------------
    # Serves resized JPEGs at /thumbnail/<hash>/<frame>. The browser-mode list
    # uses these via <img loading="lazy">, so only thumbnails actually scrolled
    # into view are decoded. The result is cached in-process by zarr group
    # (4 groups via open_zarr_for_hash's lru_cache) and a tiny LRU here.
    thumb_service.register(app)

    app.layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "background": CANVAS,
            "color": TEXT,
        },
        children=[
            # Deep-linkable URL state. `refresh=False` so callbacks rewriting
            # `search` don't reload the page. The sync callbacks below encode
            # current selection into the query string on every change, and
            # decode it back on initial page load (so bookmarks/share work).
            dcc.Location(id="url", refresh=False),
            # Header: full-width slate bar with a teal accent stripe at the
            # bottom, generous horizontal padding to breathe, and a flex row
            # that lets the zarr-root info push to the right without clipping.
            html.Div(
                style={
                    "background": ("linear-gradient(180deg, #0f172a 0%, #111c33 100%)"),
                    "color": "white",
                    "padding": "14px 28px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "gap": "24px",
                    "borderBottom": f"2px solid {ACCENT}",
                    "width": "100%",
                    "boxSizing": "border-box",
                    "flexShrink": "0",
                },
                children=[
                    # Left: brand mark (teal dot) + title + sub-label.
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "12px",
                            "minWidth": "0",
                        },
                        children=[
                            html.Span(
                                style={
                                    "width": "10px",
                                    "height": "10px",
                                    "borderRadius": "999px",
                                    "background": ACCENT,
                                    "boxShadow": ("0 0 0 4px rgba(13, 148, 136, 0.15)"),
                                    "flexShrink": "0",
                                },
                            ),
                            html.Div(
                                style={"display": "flex", "flexDirection": "column"},
                                children=[
                                    html.Div(
                                        "Latent Inspector",
                                        style={
                                            "fontSize": "15px",
                                            "fontWeight": 600,
                                            "letterSpacing": "-0.01em",
                                            "lineHeight": "1.2",
                                        },
                                    ),
                                    html.Div(
                                        f"{len(runs)} run{'s' if len(runs) != 1 else ''} discovered",
                                        style={
                                            "fontSize": "11px",
                                            "color": "#94a3b8",
                                            "marginTop": "2px",
                                            "letterSpacing": "0.02em",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Right: zarr-root path in a subtle "key: value" pill. The
                    # value gets `direction: rtl` + `textAlign: left` so long
                    # paths truncate from the LEFT (preserving the meaningful
                    # tail) rather than the right.
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "8px",
                            "minWidth": "0",
                            "maxWidth": "60%",
                        },
                        children=[
                            html.Span(
                                "zarr_root",
                                style={
                                    "fontSize": "10px",
                                    "fontWeight": 600,
                                    "letterSpacing": "0.06em",
                                    "textTransform": "uppercase",
                                    "color": "#94a3b8",
                                    "flexShrink": "0",
                                },
                            ),
                            html.Span(
                                zarr_root,
                                title=zarr_root,
                                style={
                                    "fontSize": "12px",
                                    "color": "#e2e8f0",
                                    "fontFamily": (
                                        "ui-monospace, SFMono-Regular, "
                                        "Menlo, Consolas, monospace"
                                    ),
                                    "background": "rgba(255, 255, 255, 0.06)",
                                    "padding": "4px 10px",
                                    "borderRadius": "6px",
                                    "border": "1px solid rgba(255, 255, 255, 0.08)",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "whiteSpace": "nowrap",
                                    "direction": "rtl",
                                    "textAlign": "left",
                                    "minWidth": "0",
                                },
                            ),
                        ],
                    ),
                ],
            ),
            # Body: sidebar + main + right
            html.Div(
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "flex": "1",
                    "minHeight": "0",
                },
                children=[
                    # ---------- Sidebar (controls) -------------------------
                    html.Div(
                        style={
                            "width": "300px",
                            "padding": "16px",
                            "background": "#f1f5f9",
                            "borderRight": f"1px solid {BORDER}",
                            "overflowY": "auto",
                        },
                        children=[
                            html.Div(
                                style=CARD_STYLE,
                                children=[
                                    html.Div("Run", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="run",
                                        options=[
                                            {"label": disp, "value": path}
                                            for disp, path in runs
                                        ],
                                        value=runs[0][1],
                                        clearable=False,
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style=CARD_STYLE,
                                children=[
                                    html.Div("View", style=LABEL_STYLE),
                                    dcc.RadioItems(
                                        id="view_mode",
                                        options=[
                                            {"label": " Scatter", "value": "scatter"},
                                            {
                                                "label": " KNN Browser (raw keys)",
                                                "value": "browser",
                                            },
                                        ],
                                        value="scatter",
                                        labelStyle={
                                            "display": "block",
                                            "fontSize": "13px",
                                            "marginBottom": "4px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                style=CARD_STYLE,
                                children=[
                                    html.Div("Layer", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="layer",
                                        options=[
                                            {"label": ln, "value": ln}
                                            for ln in store.layers_for(runs[0][1])
                                        ],
                                        value=(store.layers_for(runs[0][1]) or [None])[
                                            0
                                        ],
                                        clearable=False,
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                            ),
                            # Browser-only: which feature space to use for KNN.
                            # Hidden when view_mode=='scatter'.
                            html.Div(
                                id="browser_knn_space_card",
                                style={**CARD_STYLE},
                                children=[
                                    html.Div("KNN space", style=LABEL_STYLE),
                                    dcc.RadioItems(
                                        id="browser_knn_space",
                                        options=[
                                            {"label": " Raw keys", "value": "raw"},
                                            {"label": " PCA-50", "value": "pca"},
                                        ],
                                        value="raw",
                                        labelStyle={
                                            "display": "block",
                                            "fontSize": "13px",
                                            "marginBottom": "4px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    html.Div(
                                        "PCA fitted on this layer's raw keys (cached).",
                                        style={
                                            "fontSize": "10px",
                                            "color": MUTED,
                                            "marginTop": "6px",
                                        },
                                    ),
                                ],
                            ),
                            # Scatter-only controls: hidden when view_mode='browser'.
                            html.Div(
                                id="scatter_controls",
                                children=[
                                    html.Div(
                                        style=CARD_STYLE,
                                        children=[
                                            html.Div("Sample size", style=LABEL_STYLE),
                                            dcc.Input(
                                                id="sample",
                                                type="number",
                                                value=default_sample,
                                                min=2,
                                                max=500000,
                                                step=1,
                                                style={
                                                    "width": "100%",
                                                    "padding": "6px 8px",
                                                    "fontSize": "13px",
                                                    "border": f"1px solid {BORDER}",
                                                    "borderRadius": "4px",
                                                    "boxSizing": "border-box",
                                                },
                                            ),
                                            html.Div(
                                                style={"marginTop": "8px"},
                                                children=[
                                                    scatter.preset_button("100", 100),
                                                    scatter.preset_button("1k", 1000),
                                                    scatter.preset_button("5k", 5000),
                                                    scatter.preset_button("10k", 10000),
                                                    scatter.preset_button("50k", 50000),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style=CARD_STYLE,
                                        children=[
                                            html.Div("Reduction", style=LABEL_STYLE),
                                            dcc.RadioItems(
                                                id="reduction",
                                                options=[
                                                    {
                                                        "label": " UMAP 3D",
                                                        "value": "umap",
                                                    },
                                                    {
                                                        "label": " PCA→UMAP 3D",
                                                        "value": "pca_umap",
                                                    },
                                                    {
                                                        "label": " t-SNE 2D",
                                                        "value": "tsne2d",
                                                    },
                                                ],
                                                value="umap",
                                                labelStyle={
                                                    "display": "block",
                                                    "fontSize": "13px",
                                                    "marginBottom": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "All read from CSV — no recompute.",
                                                style={
                                                    "fontSize": "10px",
                                                    "color": MUTED,
                                                    "marginTop": "6px",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style=CARD_STYLE,
                                        children=[
                                            html.Div("Outliers", style=LABEL_STYLE),
                                            dcc.Checklist(
                                                id="remove_outliers",
                                                options=[
                                                    {
                                                        "label": " Hide outliers",
                                                        "value": "on",
                                                    }
                                                ],
                                                value=[],  # off by default
                                                labelStyle={
                                                    "fontSize": "13px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "Threshold (std devs from centroid):",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": MUTED,
                                                    "marginTop": "8px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="outlier_thresh",
                                                type="number",
                                                value=3.0,
                                                min=0.5,
                                                max=10.0,
                                                step=0.5,
                                                style={
                                                    "width": "100%",
                                                    "padding": "6px 8px",
                                                    "fontSize": "13px",
                                                    "border": f"1px solid {BORDER}",
                                                    "borderRadius": "4px",
                                                    "boxSizing": "border-box",
                                                    "marginTop": "4px",
                                                },
                                            ),
                                            html.Div(
                                                "Lower = stricter (hides more). 3 = ~99% kept for normal data.",
                                                style={
                                                    "fontSize": "10px",
                                                    "color": MUTED,
                                                    "marginTop": "4px",
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            # Browser-mode lang filter: drop frames whose
                            # annotation contains a substring (case-insensitive).
                            # Default OFF: enabling the filter triggers a
                            # zarr-open per unique hash, which is many seconds
                            # each on this network filesystem and would block
                            # the first paint for minutes. The user can opt in
                            # by ticking the box once the grid is up.
                            html.Div(
                                style=CARD_STYLE,
                                children=[
                                    html.Div(
                                        "Lang filter (scatter + browser)",
                                        style=LABEL_STYLE,
                                    ),
                                    dcc.Checklist(
                                        id="browser_hide_home",
                                        options=[
                                            {
                                                "label": " Hide 'home' frames",
                                                "value": "on",
                                            }
                                        ],
                                        value=[],
                                        labelStyle={
                                            "fontSize": "13px",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    html.Div(
                                        "Exclude annotation substring (case-insensitive):",
                                        style={
                                            "fontSize": "11px",
                                            "color": MUTED,
                                            "marginTop": "8px",
                                        },
                                    ),
                                    dcc.Input(
                                        id="browser_lang_exclude",
                                        type="text",
                                        value="",
                                        placeholder="e.g. fold, pick, …",
                                        debounce=True,
                                        style={
                                            "width": "100%",
                                            "padding": "6px 8px",
                                            "fontSize": "13px",
                                            "border": f"1px solid {BORDER}",
                                            "borderRadius": "4px",
                                            "boxSizing": "border-box",
                                            "marginTop": "4px",
                                        },
                                    ),
                                    html.Div(
                                        "Combined: drops any frame whose lang "
                                        "contains 'home' (if checked) OR your "
                                        "substring.",
                                        style={
                                            "fontSize": "10px",
                                            "color": MUTED,
                                            "marginTop": "6px",
                                        },
                                    ),
                                ],
                            ),
                            html.Button(
                                "Apply",
                                id="apply",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "10px",
                                    "fontSize": "14px",
                                    "fontWeight": 600,
                                    "background": ACCENT,
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "marginTop": "4px",
                                    "boxShadow": "0 1px 2px rgba(0,0,0,0.1)",
                                },
                            ),
                            html.Div(
                                "Tip: change controls then click Apply.",
                                style={
                                    "fontSize": "11px",
                                    "color": MUTED,
                                    "marginTop": "12px",
                                    "lineHeight": 1.4,
                                },
                            ),
                        ],
                    ),
                    # ---------- Scatter view (default): plot + right pane ---
                    html.Div(
                        id="scatter_view",
                        style={
                            "display": "flex",
                            "flexDirection": "row",
                            "flex": "1",
                            "minWidth": "0",
                        },
                        children=[
                            html.Div(
                                style={"flex": "1", "padding": "16px", "minWidth": "0"},
                                children=[
                                    dcc.Loading(
                                        id="scatter_loading",
                                        type="circle",
                                        color=ACCENT,
                                        children=dcc.Graph(
                                            id="scatter",
                                            style={"height": "calc(100vh - 110px)"},
                                            config={
                                                "displaylogo": False,
                                                "modeBarButtonsToRemove": ["toImage"],
                                            },
                                        ),
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "width": "340px",
                                    "padding": "16px",
                                    "borderLeft": f"1px solid {BORDER}",
                                    "background": PANEL,
                                    "overflowY": "auto",
                                },
                                children=[
                                    dcc.Store(id="nav_stack", data=[]),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "space-between",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            html.Button(
                                                "← Back",
                                                id="back_button",
                                                n_clicks=0,
                                                disabled=True,
                                                style={
                                                    "padding": "6px 12px",
                                                    "fontSize": "12px",
                                                    "fontWeight": 500,
                                                    "background": "#e2e8f0",
                                                    "color": TEXT,
                                                    "border": f"1px solid {BORDER}",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "Clicked frame",
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": 600,
                                                    "letterSpacing": "0.04em",
                                                    "textTransform": "uppercase",
                                                    "color": MUTED,
                                                },
                                            ),
                                        ],
                                    ),
                                    dcc.Loading(
                                        id="click_loading",
                                        type="default",
                                        color=ACCENT,
                                        children=html.Div(
                                            [
                                                html.Pre(
                                                    id="meta",
                                                    style={
                                                        "background": "#f1f5f9",
                                                        "padding": "10px",
                                                        "fontSize": "12px",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "fontFamily": "ui-monospace, monospace",
                                                        "marginBottom": "16px",
                                                        "whiteSpace": "pre-wrap",
                                                    },
                                                ),
                                                html.Div("Image", style=LABEL_STYLE),
                                                html.Img(
                                                    id="frame_img",
                                                    style={
                                                        "maxWidth": "100%",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "display": "block",
                                                        "marginBottom": "16px",
                                                    },
                                                ),
                                                html.Div(
                                                    "Language prompt", style=LABEL_STYLE
                                                ),
                                                html.Pre(
                                                    id="lang",
                                                    style={
                                                        "whiteSpace": "pre-wrap",
                                                        "background": "#f1f5f9",
                                                        "padding": "10px",
                                                        "fontSize": "12px",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "marginBottom": "16px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="knn_label", style=LABEL_STYLE
                                                ),
                                                html.Div(
                                                    id="knn_list",
                                                    style={
                                                        "background": "#f1f5f9",
                                                        "padding": "10px",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "fontFamily": "ui-monospace, monospace",
                                                        "fontSize": "11px",
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "gap": "2px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # ---------- KNN Browser view (raw keys) -----------------
                    # Hidden by default; shown when view_mode == 'browser'.
                    # Two-column layout:
                    #   - LEFT (flex):  full-width frame grid; each row has an
                    #     inline lazy-loaded thumbnail + metadata. Clicking a
                    #     row populates the right pane.
                    #   - RIGHT (380px): clicked-frame image + lang + raw-key
                    #     KNN list combined.
                    html.Div(
                        id="knn_browser_view",
                        style={
                            "display": "none",
                            "flexDirection": "row",
                            "flex": "1",
                            "minWidth": "0",
                        },
                        children=[
                            # Frame grid (full available width).
                            html.Div(
                                style={
                                    "flex": "1",
                                    "padding": "16px",
                                    "background": CANVAS,
                                    "overflowY": "auto",
                                    "minWidth": "0",
                                },
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "baseline",
                                            "gap": "12px",
                                            "marginBottom": "12px",
                                        },
                                        children=[
                                            html.Div("Frames", style=LABEL_STYLE),
                                            html.Div(
                                                id="browser_frame_count",
                                                style={
                                                    "fontSize": "12px",
                                                    "color": MUTED,
                                                },
                                            ),
                                            html.Div(
                                                "(click a card to inspect)",
                                                style={
                                                    "fontSize": "11px",
                                                    "color": MUTED,
                                                    "marginLeft": "auto",
                                                },
                                            ),
                                        ],
                                    ),
                                    dcc.Loading(
                                        id="browser_list_loading",
                                        type="default",
                                        color=ACCENT,
                                        children=html.Div(
                                            id="browser_frame_list",
                                            style={
                                                "display": "grid",
                                                "gridTemplateColumns": "repeat(auto-fill, minmax(260px, 1fr))",
                                                "gap": "8px",
                                            },
                                        ),
                                    ),
                                    # How many cards to render right now.
                                    # `_populate_browser_list` slices the
                                    # round-robin items list to this length
                                    # and appends a sentinel "Load more"
                                    # button when more remain. A clientside
                                    # IntersectionObserver auto-clicks the
                                    # button when it scrolls into view, so
                                    # the user gets infinite-scroll feel
                                    # without all 345 thumbnails firing at
                                    # once.
                                    dcc.Store(id="browser_visible_count", data=60),
                                ],
                            ),
                            # Detail + KNN (right side).
                            html.Div(
                                style={
                                    "width": "380px",
                                    "padding": "16px",
                                    "borderLeft": f"1px solid {BORDER}",
                                    "background": PANEL,
                                    "overflowY": "auto",
                                },
                                children=[
                                    # Nav stack: each entry is {"hash","frame","token"}.
                                    # Pushed when the user clicks a frame card or a
                                    # KNN neighbor; popped by the back button.
                                    dcc.Store(id="browser_nav_stack", data=[]),
                                    html.Div(
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "space-between",
                                            "marginBottom": "8px",
                                        },
                                        children=[
                                            html.Button(
                                                "← Back",
                                                id="browser_back",
                                                n_clicks=0,
                                                disabled=True,
                                                style={
                                                    "padding": "6px 12px",
                                                    "fontSize": "12px",
                                                    "fontWeight": 500,
                                                    "background": "#e2e8f0",
                                                    "color": TEXT,
                                                    "border": f"1px solid {BORDER}",
                                                    "borderRadius": "4px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                "Clicked frame",
                                                style={
                                                    "fontSize": "11px",
                                                    "fontWeight": 600,
                                                    "letterSpacing": "0.04em",
                                                    "textTransform": "uppercase",
                                                    "color": MUTED,
                                                },
                                            ),
                                        ],
                                    ),
                                    # Wrap all five click-driven outputs in
                                    # a single dcc.Loading so a spinner
                                    # appears whenever the click callback is
                                    # in flight (image fetch + KNN compute
                                    # can take a second on first click of a
                                    # layer). Without this the right pane
                                    # looks frozen — confusing UX.
                                    dcc.Loading(
                                        id="browser_detail_loading",
                                        type="circle",
                                        color=ACCENT,
                                        # `parent_style` keeps the spinner
                                        # centered over the children area
                                        # rather than shifting layout.
                                        parent_style={
                                            "position": "relative",
                                            "minHeight": "200px",
                                        },
                                        children=html.Div(
                                            children=[
                                                html.Img(
                                                    id="browser_img",
                                                    style={
                                                        "maxWidth": "100%",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "display": "block",
                                                        "marginBottom": "12px",
                                                    },
                                                ),
                                                html.Pre(
                                                    id="browser_meta",
                                                    style={
                                                        "background": "#f1f5f9",
                                                        "padding": "10px",
                                                        "fontSize": "12px",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "fontFamily": "ui-monospace, monospace",
                                                        "marginBottom": "12px",
                                                        "whiteSpace": "pre-wrap",
                                                    },
                                                ),
                                                html.Div(
                                                    "Language prompt", style=LABEL_STYLE
                                                ),
                                                html.Pre(
                                                    id="browser_lang",
                                                    style={
                                                        "whiteSpace": "pre-wrap",
                                                        "background": "#f1f5f9",
                                                        "padding": "10px",
                                                        "fontSize": "12px",
                                                        "borderRadius": "4px",
                                                        "border": f"1px solid {BORDER}",
                                                        "marginBottom": "16px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="browser_knn_label",
                                                    style={
                                                        **LABEL_STYLE,
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                html.Div(
                                                    id="browser_knn_list",
                                                    style={
                                                        "display": "flex",
                                                        "flexDirection": "column",
                                                        "gap": "4px",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    scatter.register()
    browser.register()

    # ----- state → URL sync ------------------------------------------------
    # Whenever the run/layer/view/KNN-space changes or a new frame is clicked,
    # encode the current selection into the URL query string. `refresh=False`
    # on dcc.Location keeps the page from reloading, so this is purely cosmetic
    # (deep-linkable / copy-paste-able). The reverse direction (restoring state
    # from URL on page load) is deliberately not implemented here — adding it
    # would require careful loop-breaking with `allow_duplicate=True` and an
    # init-only guard, which we can layer on later if you want bookmark restore.
    from urllib.parse import urlencode

    @app.callback(
        dash.Output("url", "search"),
        dash.Input("run", "value"),
        dash.Input("layer", "value"),
        dash.Input("view_mode", "value"),
        dash.Input("browser_knn_space", "value"),
        dash.Input("browser_nav_stack", "data"),
        dash.Input("nav_stack", "data"),
    )
    def _state_to_url(run, layer, view_mode, knn_space, browser_nav, scatter_nav):
        params: dict[str, str] = {}
        if run:
            params["run"] = run
        if layer:
            params["layer"] = layer
        if view_mode:
            params["view"] = view_mode
        if knn_space:
            params["knn"] = knn_space
        # The most recent clicked frame is the tail of whichever nav stack
        # belongs to the active view.
        active_nav = (browser_nav if view_mode == "browser" else scatter_nav) or []
        if active_nav:
            cur = active_nav[-1]
            if isinstance(cur, dict):
                if cur.get("hash") is not None:
                    params["hash"] = str(cur["hash"])
                if cur.get("frame") is not None:
                    params["frame"] = str(cur["frame"])
                if cur.get("token") is not None:
                    params["token"] = str(cur["token"])
        return ("?" + urlencode(params)) if params else ""

    return app
