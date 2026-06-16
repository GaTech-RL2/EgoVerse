#!/usr/bin/env python3
"""
Re-runnable in-training REPLAY sim-coverage plotter for the 7-run 3000-epoch
RL training campaign on the PACE cluster.

What it does (every run is idempotent / safe to re-run):
  1. For each of the 7 runs, resolve the NEWEST timestamped run dir on PACE
     (over ssh, head-node `ls` only -- NO python on the head node) and scp the
     latest metrics.csv into a local temp dir.
  2. Parse coverage-vs-epoch from each CSV (drop NaN/blank coverage rows).
  3. Produce ONE matplotlib figure:
        x = epoch (0..2000), y = coverage (0..1)
        one line+markers per run; cotrain runs show emb15 (solid) + emb17 (dashed)
        horizontal dashed red line at y=0.5 = the (offline) val-SEED GOAL
        runs with no val data show in the legend as "(run) -- no val yet"
  4. Save PNG + print a per-run latest-coverage summary.

IMPORTANT semantics:
  * 'Valid/emb15_sim_coverage' / 'Valid/emb17_sim_coverage' are REPLAY-protocol
    (in-distribution) coverage logged DURING training. This is NOT the
    seed-protocol >0.5 metric the campaign goal targets (that is measured
    OFFLINE). The plot is labelled REPLAY everywhere to avoid confusion.
  * emb15 = pushshapes_sim (big circle); emb17 = pushshapes_sim_small_circle.
  * Validation runs every 500 epochs -> coverage cells are blank except at
    ep500/1000/1500/2000. Early in the campaign there may be ZERO val rows;
    that is EXPECTED and the script must not crash.
  * 3 runs (hpt_circle3000, hpt_cot3000, hnet_big_circle3000) may still be
    PENDING with no run dir / metrics.csv -> handled gracefully (skipped).

Usage:
    python plot_campaign_coverage.py            # pull + plot
    python plot_campaign_coverage.py --no-pull  # use already-pulled local CSVs
"""

import argparse
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SSH_HOST = "pacerh9"
# Use an explicit ssh/scp path: in some sandboxed shells the bare `ssh` name is
# blocked but /usr/bin/ssh works. Fall back to PATH lookup if it's absent.
SSH_BIN = "/usr/bin/ssh" if os.path.exists("/usr/bin/ssh") else "ssh"
SCP_BIN = "/usr/bin/scp" if os.path.exists("/usr/bin/scp") else "scp"

REMOTE_LOGS_ROOT = (
    "/storage/project/r-dxu345-0/paphiwetsa3/projects/EgoVerse-gmm/logs"
)

LOCAL_DIR = os.path.expanduser("~/Documents/rl2")
CSV_DIR = os.path.join(LOCAL_DIR, "campaign_csv")
PNG_PATH = os.path.join(LOCAL_DIR, "campaign_coverage_replay.png")

MAX_EPOCHS = 2000
VAL_EVERY = 500  # vals at 500/1000/1500/2000

# All 7 runs. is_cotrain -> also has emb17 coverage.
RUNS = [
    {"name": "hpt_circle3000", "cotrain": False},
    {"name": "hpt_cot3000", "cotrain": True},
    {"name": "txar_circle3000", "cotrain": False},
    {"name": "txar_cot3000", "cotrain": True},
    {"name": "hnet_big_circle3000", "cotrain": False},
    {"name": "hnet_big_cot3000", "cotrain": True},
    {"name": "hnet_big_rsched3000", "cotrain": False},
]

# Coverage column names in the lightning metrics.csv.
COV15 = "Valid/emb15_sim_coverage"
COV17 = "Valid/emb17_sim_coverage"

# A stable colour per run so emb15/emb17 of the same run share a hue.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]
RUN_COLOR = {r["name"]: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(RUNS)}


# ----------------------------------------------------------------------------
# Remote pull
# ----------------------------------------------------------------------------
def resolve_remote_csv(run_name):
    """Return the newest remote metrics.csv path for a run, or None if missing."""
    pattern = (
        f"{REMOTE_LOGS_ROOT}/{run_name}/{run_name}_*/"
        f"csv_logs/lightning_logs/version_0/metrics.csv"
    )
    # ls -dt sorts newest-first; head -1 takes the most recent.
    remote_cmd = f"ls -dt {pattern} 2>/dev/null | head -1"
    try:
        out = subprocess.run(
            [SSH_BIN, "-o", "ConnectTimeout=25", SSH_HOST, remote_cmd],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{run_name}] ssh timed out resolving path", file=sys.stderr)
        return None
    path = out.stdout.strip()
    return path or None


def scp_csv(run_name, remote_path, dest_dir):
    """scp a remote metrics.csv to dest_dir/<run>.csv. Return local path or None."""
    dest = os.path.join(dest_dir, f"{run_name}.csv")
    try:
        r = subprocess.run(
            [SCP_BIN, "-q", "-o", "ConnectTimeout=25",
             f"{SSH_HOST}:{remote_path}", dest],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{run_name}] scp timed out", file=sys.stderr)
        return None
    if r.returncode != 0:
        print(f"  [{run_name}] scp failed: {r.stderr.strip()}", file=sys.stderr)
        return None
    return dest


def pull_all(dest_dir):
    """Resolve + pull every run's CSV. Return {run_name: local_csv_path_or_None}."""
    os.makedirs(dest_dir, exist_ok=True)
    local = {}
    for r in RUNS:
        name = r["name"]
        remote = resolve_remote_csv(name)
        if remote is None:
            print(f"  [{name}] no run dir / metrics.csv on PACE yet (pending)")
            local[name] = None
            continue
        lp = scp_csv(name, remote, dest_dir)
        if lp:
            print(f"  [{name}] pulled {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(remote)))))}")
        local[name] = lp
    return local


# ----------------------------------------------------------------------------
# Parse
# ----------------------------------------------------------------------------
def parse_coverage(csv_path, col):
    """
    Return a DataFrame [epoch, coverage] for the given coverage column, with
    NaN/blank coverage rows dropped and one (max) value kept per epoch.
    Returns an empty DataFrame if the file/column is absent or has no val rows.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return pd.DataFrame(columns=["epoch", "coverage"])
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  warn: could not read {csv_path}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["epoch", "coverage"])
    if col not in df.columns or "epoch" not in df.columns:
        # Column not written yet (no validation has run) -> no data.
        return pd.DataFrame(columns=["epoch", "coverage"])
    sub = df[["epoch", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col, "epoch"])
    if sub.empty:
        return pd.DataFrame(columns=["epoch", "coverage"])
    sub["epoch"] = pd.to_numeric(sub["epoch"], errors="coerce")
    sub = sub.dropna(subset=["epoch"])
    sub["epoch"] = sub["epoch"].astype(int)
    # If a (rare) duplicate epoch appears, keep the max coverage at that epoch.
    sub = sub.groupby("epoch", as_index=False)[col].max()
    sub = sub.rename(columns={col: "coverage"}).sort_values("epoch")
    return sub.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def build_plot(local_csvs):
    """Build the figure; return (per_run_summary, any_data_bool)."""
    fig, ax = plt.subplots(figsize=(11, 7))

    per_run = {}      # name -> human-readable latest-coverage string
    any_data = False

    for r in RUNS:
        name = r["name"]
        color = RUN_COLOR[name]
        csv = local_csvs.get(name)

        if csv is None:
            per_run[name] = "PENDING (no run dir on PACE)"
            # Still represent it in the legend so all 7 are visible.
            ax.plot([], [], color=color, marker="o", linestyle="-",
                    label=f"{name} -- pending (no run dir)")
            continue

        d15 = parse_coverage(csv, COV15)
        series_drawn = False
        latest_parts = []

        if not d15.empty:
            any_data = True
            series_drawn = True
            lbl = f"{name} emb15" if r["cotrain"] else name
            ax.plot(d15["epoch"], d15["coverage"], color=color, marker="o",
                    linestyle="-", linewidth=2, markersize=7, label=lbl)
            last = d15.iloc[-1]
            latest_parts.append(
                f"emb15={last['coverage']:.3f}@ep{int(last['epoch'])}"
            )

        if r["cotrain"]:
            d17 = parse_coverage(csv, COV17)
            if not d17.empty:
                any_data = True
                series_drawn = True
                ax.plot(d17["epoch"], d17["coverage"], color=color, marker="s",
                        linestyle="--", linewidth=2, markersize=7,
                        label=f"{name} emb17")
                last = d17.iloc[-1]
                latest_parts.append(
                    f"emb17={last['coverage']:.3f}@ep{int(last['epoch'])}"
                )

        if series_drawn:
            per_run[name] = ", ".join(latest_parts)
        else:
            per_run[name] = "no val yet"
            # Placeholder legend entry so the run is visible as "no val yet".
            ax.plot([], [], color=color, marker="o", linestyle="-",
                    label=f"{name} -- no val yet")

    # Goal line (offline val-SEED target -- NOT this replay metric).
    ax.axhline(
        y=0.5, color="red", linestyle="--", linewidth=1.5,
        label="GOAL (val-SEED, measured offline -- NOT this replay metric)",
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("sim coverage (REPLAY protocol, in-distribution)")
    ax.set_xlim(0, MAX_EPOCHS)
    ax.set_ylim(0, 1.0)
    # Mark the validation epochs on the x axis so the cadence is obvious.
    ax.set_xticks(list(range(0, MAX_EPOCHS + 1, VAL_EVERY)))
    ax.grid(True, alpha=0.3)
    ax.set_title("In-training REPLAY sim coverage -- 7-run 3000-ep campaign")

    if not any_data:
        ax.text(
            0.5, 0.55,
            "No validation rows yet\n(first validation at epoch 500)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="gray",
            bbox=dict(boxstyle="round", fc="#f2f2f2", ec="gray"),
        )

    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
    fig.savefig(PNG_PATH, dpi=130)
    plt.close(fig)
    return per_run, any_data


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-pull", action="store_true",
        help="skip the scp pull; use CSVs already in the local temp dir",
    )
    args = ap.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)

    if args.no_pull:
        print(f"Using already-pulled CSVs in {CSV_DIR} (--no-pull)")
        local_csvs = {}
        for r in RUNS:
            p = os.path.join(CSV_DIR, f"{r['name']}.csv")
            local_csvs[r["name"]] = p if os.path.exists(p) else None
    else:
        print(f"Pulling latest metrics.csv for {len(RUNS)} runs from "
              f"{SSH_HOST} ...")
        local_csvs = pull_all(CSV_DIR)

    per_run, any_data = build_plot(local_csvs)

    print("\n" + "=" * 70)
    print("Per-run latest REPLAY sim coverage:")
    print("=" * 70)
    for r in RUNS:
        print(f"  {r['name']:<22}: {per_run.get(r['name'], 'n/a')}")
    print("=" * 70)
    if any_data:
        print(f"\nSaved figure -> {PNG_PATH}")
    else:
        print("\nNo validation rows yet across all runs "
              f"(first val at ep{VAL_EVERY}). This is EXPECTED for an early "
              "campaign -- the figure was still written as a placeholder.")
        print(f"Saved figure -> {PNG_PATH}")


if __name__ == "__main__":
    main()
