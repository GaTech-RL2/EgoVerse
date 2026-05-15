"""
Plot train loss, val MSE, boundary rate, and avg chunk len across all
H-Net variant runs. Reads ``logs/hnet_variants/*/csv_logs/lightning_logs/
version_0/metrics.csv`` (one per variant) and emits a single multi-panel
PNG.

Variants discovered automatically by directory name prefix; pass
``--variants foo,bar`` to filter.

Run:
    python scripts/plot_variants.py --out variant_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import OrderedDict

import matplotlib.pyplot as plt

LOG_ROOT = "logs"

# Friendly names + display order. The right side of each entry maps the
# log dir prefix to a clean label and a colour.
VARIANT_TABLE = OrderedDict(
    [
        # (prefix matched against log dir name, label, colour)
        ("hnet_smoke/recipe_off", ("baseline AdaLN (4 ep smoke)", "#bdbdbd")),
        ("hnet_smoke/convergence", ("baseline AdaLN (30 ep)", "#888888")),
        ("hnet_pushshapes/full_run", ("baseline AdaLN (150 ep)", "#1f77b4")),
        ("hnet_variants/delta_100ep", ("delta + AdaLN (100 ep)", "#ff7f0e")),
        ("hnet_variants/big_100ep", ("bigger AdaLN (100 ep)", "#9467bd")),
        ("hnet_variants/crossattn_100ep", ("cross-attention (100 ep)", "#e377c2")),
        ("hnet_variants/fused_100ep", ("fused tokens (100 ep)", "#2ca02c")),
        (
            "hnet_variants/adaln_recipe_lowlr_100ep",
            ("AdaLN + recipe + lr/2 (100 ep)", "#d62728"),
        ),
        ("hnet_variants/fused_lowlr_100ep", ("fused + lr/2 (100 ep)", "#17becf")),
    ]
)

# Column names in the lightning CSV.
COL_EPOCH = "epoch"
COL_STEP = "step"
COL_TRAIN_LOSS = "Train/Loss"
COL_TRAIN_EMB_LOSS = "Train/emb15_action_loss"
COL_TRAIN_BOUNDARY = "Train/emb15_boundary_rate"
COL_TRAIN_AVG_CHUNK_LEN = "Train/emb15_avg_chunk_len"
COL_TRAIN_RATIO_LOSS = "Train/emb15_ratio_loss"
COL_VAL_PAIRED_MSE = "Valid/emb15_actions_paired_mse"


def load_csv(path: str) -> dict[str, list]:
    """Return a dict of col -> list of values, with empty cells as None."""
    cols: dict[str, list] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v if v != "" else None)
    return cols


def get_series(cols: dict[str, list], key: str) -> tuple[list, list]:
    """Return (epochs, values) for ``key``, dropping rows where value is None."""
    if key not in cols or COL_EPOCH not in cols:
        return [], []
    ep, val = [], []
    for e, v in zip(cols[COL_EPOCH], cols[key]):
        if e is None or v is None:
            continue
        ep.append(int(float(e)))
        val.append(float(v))
    return ep, val


def find_metrics_csv(prefix: str) -> str | None:
    """Find the most recent metrics.csv for a log-dir prefix."""
    pattern = os.path.join(
        LOG_ROOT, f"{prefix}*/csv_logs/lightning_logs/version_0/metrics.csv"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="variant_comparison.png")
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated subset of variant keys to include (default: all found).",
    )
    parser.add_argument(
        "--ymax-loss",
        type=float,
        default=0.05,
        help="Y-axis cap for the loss panels (otherwise the noisy "
        "epoch-0 spike crushes the curves).",
    )
    parser.add_argument(
        "--ymax-val", type=float, default=None, help="Y-axis cap for the val MSE panel."
    )
    args = parser.parse_args()

    if args.variants:
        wanted = set(args.variants.split(","))
        items = [(k, v) for k, v in VARIANT_TABLE.items() if k.split("/")[-1] in wanted]
    else:
        items = list(VARIANT_TABLE.items())

    loaded = []
    for prefix, (label, colour) in items:
        path = find_metrics_csv(prefix)
        if path is None:
            print(f"  skip (no csv): {label}  [prefix={prefix}]")
            continue
        cols = load_csv(path)
        loaded.append((label, colour, cols, path))
        print(f"  loaded: {label}  [{path}]")

    if not loaded:
        raise SystemExit("No metrics CSVs found.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("H-Net variant comparison (pushshapes packed training)", fontsize=14)

    panels = [
        (
            axes[0, 0],
            "Train/Loss (action MSE + ratio loss)",
            COL_TRAIN_LOSS,
            True,
            args.ymax_loss,
        ),
        (
            axes[0, 1],
            "Train action loss (per-emb, no ratio loss)",
            COL_TRAIN_EMB_LOSS,
            True,
            args.ymax_loss,
        ),
        (
            axes[0, 2],
            "Val paired MSE (teacher-forced, unnorm)",
            COL_VAL_PAIRED_MSE,
            True,
            args.ymax_val,
        ),
        (
            axes[1, 0],
            "Chunker boundary rate (target=0.125)",
            COL_TRAIN_BOUNDARY,
            False,
            None,
        ),
        (
            axes[1, 1],
            "Chunker avg chunk len (target=8)",
            COL_TRAIN_AVG_CHUNK_LEN,
            True,
            None,
        ),
        (axes[1, 2], "Chunker ratio loss", COL_TRAIN_RATIO_LOSS, False, None),
    ]

    for ax, title, col, log_y, ymax in panels:
        for label, colour, cols, _ in loaded:
            ep, val = get_series(cols, col)
            if not ep:
                continue
            ax.plot(ep, val, label=label, color=colour, linewidth=1.4, alpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        if ymax is not None:
            cur_min, _ = ax.get_ylim()
            ax.set_ylim(top=ymax)
        if title.startswith("Chunker boundary"):
            ax.axhline(0.125, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
        if title.startswith("Chunker avg chunk"):
            ax.axhline(8.0, color="k", linestyle=":", linewidth=0.8, alpha=0.6)

    # One shared legend below all panels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
