"""
Gigantic 2D t-SNE scatter plot colored by a chosen metadata column.

Reads:
- manifest.json (for zarr + metadata paths)
- metadata.parquet (label column is configurable; defaults to lab-like columns)
- embeddings zarr group (expects dataset 'tsne_2d' by default)

Writes:
- a large PNG scatter plot to the data directory

Plot config notes:
- `plot_background_color`: figure/axes background (e.g. "#ecdbc7"). Empty/None disables.
- `plot_background_alpha`: optional float in [0,1] (defaults to 1.0).
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import zarr


mpl.rcParams["font.family"] = "monospace"
mpl.rcParams["font.monospace"] = [
    "SF Mono",
    "Menlo",
    "Monaco",
    "Source Code Pro",
    "IBM Plex Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
]


def _pick_label_column(df: pd.DataFrame, label_col: str) -> str:
    """
    Resolve which metadata column to use as labels/colors.
    - If label_col is provided, require it to exist.
    - Else, fall back to common "lab" column names.
    """
    if label_col:
        if label_col not in df.columns:
            raise KeyError(
                "Requested --label-col '{}' not found in metadata. Available columns (truncated): {}".format(
                    label_col, list(df.columns)[:50]
                )
            )
        return label_col

    for c in ("lab", "db.lab", "metadata.lab"):
        if c in df.columns:
            return c
    raise KeyError(
        "Could not infer a default label column. Tried: lab, db.lab, metadata.lab. "
        "Pass --label-col to choose a column explicitly. Available columns (truncated): {}".format(
            list(df.columns)[:50]
        )
    )


def _load_omit_configs(*, omit_configs_json: str, omit_configs_file: str) -> list[dict]:
    """
    Loads omit configs as a list of dicts.

    Semantics:
    - Each dict is a conjunction (AND) of column==value matches.
    - The list is a disjunction (OR) across dicts.
    - Any row matching ANY omit dict is removed from the plot.
    """

    omit_configs: list[dict] = []

    if omit_configs_json:
        parsed = json.loads(omit_configs_json)
        if not isinstance(parsed, list) or not all(isinstance(x, dict) for x in parsed):
            raise TypeError("--omit-configs-json must be a JSON list of dicts")
        omit_configs.extend(parsed)

    if omit_configs_file:
        p = Path(omit_configs_file)
        parsed = json.loads(p.read_text())
        if not isinstance(parsed, list) or not all(isinstance(x, dict) for x in parsed):
            raise TypeError("--omit-configs-file must point to a JSON file containing a list of dicts")
        omit_configs.extend(parsed)

    # normalize any weird entries (e.g. empty dicts)
    omit_configs = [d for d in omit_configs if len(d) > 0]
    return omit_configs


def _apply_omit_configs(
    meta_df: pd.DataFrame, y: np.ndarray, *, omit_configs: list[dict]
) -> tuple[pd.DataFrame, np.ndarray]:
    if not omit_configs:
        return meta_df, y

    for i, cfg in enumerate(omit_configs):
        missing = [k for k in cfg.keys() if k not in meta_df.columns]
        if missing:
            raise KeyError(
                "omit_configs[{}] refers to missing columns: {}. Available columns (truncated): {}".format(
                    i, missing, list(meta_df.columns)[:50]
                )
            )

    omit_mask = np.zeros(len(meta_df), dtype=bool)
    for cfg in omit_configs:
        m = pd.Series(True, index=meta_df.index)
        for k, v in cfg.items():
            col = meta_df[k]
            if v is None:
                m = m & col.isna()
            elif isinstance(v, str):
                m = m & (col.astype(str) == v)
            else:
                m = m & (col == v)
        omit_mask |= m.to_numpy(dtype=bool)

    keep_mask = ~omit_mask
    kept = int(keep_mask.sum())
    removed = int(omit_mask.sum())
    print(
        "[INFO] omit_configs removed {} / {} rows (kept {})".format(
            removed, len(meta_df), kept
        )
    )
    meta_df = meta_df.loc[keep_mask].reset_index(drop=True)
    y = y[keep_mask]
    return meta_df, y


def _apply_sample_every_k(
    meta_df: pd.DataFrame, y: np.ndarray, *, sample_every_k: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if sample_every_k <= 1:
        return meta_df, y
    before = len(meta_df)
    meta_df = meta_df.iloc[::sample_every_k].reset_index(drop=True)
    y = y[::sample_every_k]
    print(
        "[INFO] sample_every_k={} kept {} / {} rows".format(
            sample_every_k, len(meta_df), before
        )
    )
    return meta_df, y


def _load_plot_config(*, plot_config_json: str, plot_config_file: str) -> dict:
    """
    Load a plotting config dict.

    Supported schema (both forms accepted):
    - {"label_col": "robot_name",
       "label_col_name": [{"eva_bimanual": {"color": "#...", "legend_name": "Robot"}}, ...]}
    - {"label_col": "robot_name",
       "label_col_name": {"eva_bimanual": {"color": "#...", "legend_name": "Robot"}, ...}}
    """
    cfg: dict = {}
    if plot_config_json:
        cfg = json.loads(plot_config_json)
        if not isinstance(cfg, dict):
            raise TypeError("--plot-config-json must be a JSON object (dict)")
        return cfg
    if plot_config_file:
        p = Path(plot_config_file)
        cfg = json.loads(p.read_text())
        if not isinstance(cfg, dict):
            raise TypeError("--plot-config-file must point to a JSON file containing an object (dict)")
        return cfg
    return {}


def _normalize_label_styles(plot_cfg: dict) -> tuple[list[str], dict[str, dict]]:
    """
    Returns (ordered_label_values, label_value->style_dict).
    """
    label_col_name = plot_cfg.get("label_col_name", None)
    if not label_col_name:
        return [], {}

    if isinstance(label_col_name, dict):
        ordered = list(label_col_name.keys())
        styles = label_col_name
    elif isinstance(label_col_name, list):
        ordered = []
        styles = {}
        for entry in label_col_name:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise TypeError(
                    "plot_config['label_col_name'] entries must be dicts with a single key, got: {}".format(
                        entry
                    )
                )
            (k, v), = entry.items()
            ordered.append(str(k))
            styles[str(k)] = v if isinstance(v, dict) else {}
    else:
        raise TypeError(
            "plot_config['label_col_name'] must be a dict or list, got: {}".format(type(label_col_name))
        )
    return ordered, {str(k): (v if isinstance(v, dict) else {}) for k, v in styles.items()}


def _build_colors_and_legend(
    labels: np.ndarray,
    *,
    ordered_styles: list[str],
    style_map: dict[str, dict],
) -> tuple[np.ndarray, list[plt.Line2D], list[str]]:
    """
    Returns (per_point_rgba Nx4, legend_handles, legend_names).

    - Labels listed in ordered_styles get their provided colors (if any) and legend names (if any).
    - Remaining labels get colors from tab20.
    - Legend order: ordered_styles first (if present in data), then remaining in first-seen order.
    """
    labels = labels.astype(str)
    present = set(labels.tolist())
    ordered_present = [v for v in ordered_styles if v in present]

    # Stable "first seen" order for labels not in ordered_styles
    remainder = []
    seen = set(ordered_present)
    for v in labels.tolist():
        if v in present and v not in seen:
            seen.add(v)
            remainder.append(v)

    # Assign colors
    label_to_rgba: dict[str, tuple[float, float, float, float]] = {}
    for v in ordered_present:
        style = style_map.get(v, {})
        if "color" in style and style["color"]:
            label_to_rgba[v] = to_rgba(style["color"])
        else:
            # fallback color if not provided
            label_to_rgba[v] = to_rgba("#4a4e69")

    if remainder:
        cmap = plt.get_cmap("tab20", max(1, len(remainder)))
        for i, v in enumerate(remainder):
            label_to_rgba[v] = cmap(i)

    point_colors = np.asarray([label_to_rgba[v] for v in labels], dtype=float)

    # Legend labels (names)
    legend_order = ordered_present + remainder
    legend_names = []
    handles = []
    for v in legend_order:
        style = style_map.get(v, {})
        legend_names.append(str(style.get("legend_name", v)))
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", color=label_to_rgba[v], markersize=12))

    return point_colors, handles, legend_names


def _safe_filename(s: str, *, max_len: int = 120) -> str:
    s = s.strip()
    # Replace whitespace with underscores
    s = re.sub(r"\s+", "_", s)
    # Keep only common safe characters
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    # Collapse repeats and trim
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        s = "plot"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s


def _apply_plot_background(*, fig: plt.Figure, ax: plt.Axes, plot_cfg: dict) -> None:
    """
    Apply a plot background (figure + axes facecolor) from plot_cfg.
    """
    bg = plot_cfg.get("plot_background_color", None)
    if bg is None:
        return
    bg = str(bg).strip()
    if not bg:
        return

    alpha = plot_cfg.get("plot_background_alpha", 1.0)
    try:
        alpha = float(alpha)
    except Exception:
        alpha = 1.0
    alpha = float(np.clip(alpha, 0.0, 1.0))

    rgba = to_rgba(bg, alpha=alpha)
    fig.patch.set_facecolor(rgba)
    ax.set_facecolor(rgba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="egomimic/scripts/visualization_process/fold_clothes_aria_eva_all_labs/manifest.json",
    )
    ap.add_argument("--image-key", type=str, default="", help="Defaults to first manifest image key.")
    ap.add_argument(
        "--reduce-method",
        type=str,
        default="tsne",
        choices=("tsne", "umap", "pca"),
        help="Which 2D reduction result to visualize (selects <method>_2d by default).",
    )
    ap.add_argument(
        "--reduce-name",
        dest="reduce_name",
        type=str,
        default=None,
        help="Dataset name inside the zarr group to visualize (overrides --reduce-method).",
    )
    # Backwards-compatible alias (tsne-name historically meant "which 2D coords dataset to plot")
    ap.add_argument(
        "--tsne-name",
        dest="reduce_name",
        type=str,
        default=None,
        help="(Deprecated) Same as --reduce-name.",
    )
    ap.add_argument(
        "--label-col",
        type=str,
        default="robot_name",
        help=(
            "Metadata column to color points by (e.g. 'lab', 'db.operator', 'task', 'episode_hash'). "
            "If omitted, tries lab-like columns: lab, db.lab, metadata.lab."
        ),
    )
    ap.add_argument("--out", type=str, default="", help="Output png path (defaults next to manifest).")
    ap.add_argument("--figsize", type=float, nargs=2, default=(12, 12), help="Figure size in inches (W H).")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--point-size", type=float, default=40.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="If provided, overrides the default plot title.",
    )
    ap.add_argument(
        "--omit-configs-json",
        type=str,
        default="",
        help=(
            "JSON list of dicts specifying metadata rows to OMIT. "
            "Example: '[{\"robot_name\":\"eva_bimanual\"}, {\"lab\":\"song\",\"operator\":\"rl2\"}]'. "
            "Each dict is an AND across keys; the list is OR across dicts."
        ),
    )
    ap.add_argument(
        "--omit-configs-file",
        type=str,
        default="",
        help="Path to a JSON file containing a list of dicts (same format as --omit-configs-json).",
    )
    ap.add_argument(
        "--plot-config-json",
        type=str,
        default="",
        help=(
            "JSON object configuring label styles (colors/legend names). "
            "If provided, overrides the in-script default mapping."
        ),
    )
    ap.add_argument(
        "--plot-config-file",
        type=str,
        default="",
        help="Path to a JSON file containing a plotting config object (same as --plot-config-json).",
    )
    ap.add_argument(
        "--sample-every-k",
        type=int,
        default=1,
        help="Keep every k-th datapoint (applied after omit filters). Use 1 to disable.",
    )
    args = ap.parse_args()

    default_plot_config = {
        "label_col": args.label_col,
        "plot_background_color": "#FFFFFF",
        "label_col_name": [
            {"eva_bimanual": {
                "color": "#009e73",
                "legend_name": "Robot"
            }},
            {"aria_bimanual": {
                "color": "#2462a3",
                "legend_name": "EgoVerse-A"
            }},
            {"mecka_bimanual": {
                "color": "#e5a423",
                "legend_name": "EgoVerse-I"
            }}
        ]
    }

    plot_cfg = default_plot_config | _load_plot_config(
        plot_config_json=args.plot_config_json,
        plot_config_file=args.plot_config_file,
    )
    label_col = plot_cfg.get("label_col", args.label_col)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    if args.image_key:
        image_key = args.image_key
    else:
        image_key = manifest["image_keys"][0]

    zarr_path = Path(manifest["embeddings"][image_key])
    meta_path = Path(manifest["metadata_parquet"])

    meta_df = pd.read_parquet(meta_path)
    label_col = _pick_label_column(meta_df, label_col)

    root = zarr.open_group(str(zarr_path), mode="r")
    reduce_name = args.reduce_name if args.reduce_name else f"{args.reduce_method}_2d"
    if reduce_name not in root:
        raise KeyError(
            "Could not find '{}' in zarr group. Available arrays: {}".format(
                reduce_name, list(root.array_keys())
            )
        )
    y = np.asarray(root[reduce_name][:])  # (N,2)
    if y.ndim != 2 or y.shape[1] != 2:
        raise RuntimeError("Unexpected 2D reduction shape for '{}': {}".format(reduce_name, y.shape))

    if len(meta_df) != y.shape[0]:
        raise RuntimeError(
            "Row mismatch: metadata has {} rows but '{}' has {} rows".format(
                len(meta_df), reduce_name, y.shape[0]
            )
        )

    omit_configs = _load_omit_configs(
        omit_configs_json=args.omit_configs_json,
        omit_configs_file=args.omit_configs_file,
    )
    meta_df, y = _apply_omit_configs(meta_df, y, omit_configs=omit_configs)
    meta_df, y = _apply_sample_every_k(meta_df, y, sample_every_k=args.sample_every_k)

    labels = meta_df[label_col].astype(str).fillna("unknown").to_numpy()
    ordered_styles, style_map = _normalize_label_styles(plot_cfg)
    point_colors, legend_handles, legend_names = _build_colors_and_legend(
        labels, ordered_styles=ordered_styles, style_map=style_map
    )

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    _apply_plot_background(fig=fig, ax=ax, plot_cfg=plot_cfg)
    ax.scatter(
        y[:, 0],
        y[:, 1],
        c=point_colors,
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
        rasterized=True,
    )

    if args.title:
        title = args.title
    else:
        title = "t-SNE of embeddings (colored by {}: {})".format("label", label_col)
    # Title at the very top (above legend + axes)
    fig.suptitle(title, y=0.99, fontsize=24)
    ax.grid(False)

    # Legend (label key): place at top, horizontal layout (figure-level for tighter spacing)
    ncol = min(max(1, len(legend_names)), 10)
    fig.legend(
        legend_handles,
        legend_names,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        frameon=False,
        fontsize=24,
        ncol=ncol,
        borderaxespad=0.0,
        columnspacing=1.0,
    )

    # Reserve minimal top space for suptitle + legend
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if args.out:
        out_path = Path(args.out)
    else:
        if args.title:
            out_path = manifest_path.parent / f"{_safe_filename(args.title)}.png"
        else:
            safe_label = label_col.replace("/", "_").replace(".", "_")
            out_path = manifest_path.parent / f"tsne_by_{safe_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
