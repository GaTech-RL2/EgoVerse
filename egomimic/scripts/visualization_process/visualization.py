"""
Gigantic 2D t-SNE scatter plot colored by a chosen metadata column.

Reads:
- manifest.json (for zarr + metadata paths)
- metadata.parquet (label column is configurable; defaults to lab-like columns)
- embeddings zarr group (expects dataset 'tsne_2d' by default)

Writes:
- a large PNG scatter plot to the data directory
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    meta_df = meta_df.iloc[::sample_every_k].reset_index(drop=True)
    y = y[::sample_every_k]
    print(
        "[INFO] sample_every_k={} kept {} / {} rows".format(
            sample_every_k, len(meta_df), len(y) * sample_every_k
        )
    )
    return meta_df, y


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
        "--sample-every-k",
        type=int,
        default=1,
        help="Keep every k-th datapoint (applied after omit filters). Use 1 to disable.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    if args.image_key:
        image_key = args.image_key
    else:
        image_key = manifest["image_keys"][0]

    zarr_path = Path(manifest["embeddings"][image_key])
    meta_path = Path(manifest["metadata_parquet"])

    meta_df = pd.read_parquet(meta_path)
    label_col = _pick_label_column(meta_df, args.label_col)

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
    uniq_labels, label_codes = np.unique(labels, return_inverse=True)

    # Build a categorical colormap with enough distinct colors
    cmap = plt.get_cmap("tab20", max(1, len(uniq_labels)))

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    ax.scatter(
        y[:, 0],
        y[:, 1],
        c=label_codes,
        cmap=cmap,
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
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=cmap(i), markersize=6)
        for i in range(len(uniq_labels))
    ]
    ncol = min(max(1, len(uniq_labels)), 10)
    fig.legend(
        handles,
        uniq_labels.tolist(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        frameon=False,
        fontsize=16,
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
    fig.savefig(out_path, bbox_inches="tight")
    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
