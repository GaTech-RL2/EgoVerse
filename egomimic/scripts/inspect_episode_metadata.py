#!/usr/bin/env python3
"""
Read-only inspection of the app.episodes table to decide how human (Aria) and
robot (Eva) pick_place episodes can be paired for side-by-side eval.

It answers three questions per embodiment:
  1. Are the `scene` / `objects` columns populated? (-> metadata-join pairing)
  2. What distinct `task_description` values exist? (-> the base/object/alignment
     sub-protocols, and another field to match on)
  3. How many episodes are actually processed (non-empty zarr_processed_path)?

No writes are performed; this only SELECTs via episode_table_to_df.

Usage:
    # needs the same DB creds as the rest of the pipeline (SECRETS_ARN or
    # ~/.egoverse_env via setup_secret.sh)
    python -m egomimic.scripts.inspect_episode_metadata
    python -m egomimic.scripts.inspect_episode_metadata --task pick_place \
        --embodiments eva_bimanual aria_bimanual
    python -m egomimic.scripts.inspect_episode_metadata --csv-out /tmp/episodes.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

# Columns we care about for pairing. Missing ones are reported, not fatal,
# because app.episodes is autoloaded from the live DB and may drift.
PAIRING_COLS = ["scene", "objects", "task_description"]


def _nonempty(series: pd.Series) -> pd.Series:
    """Boolean mask of cells that are neither NaN nor empty/whitespace string."""
    return series.notna() & (series.astype(str).str.strip() != "")


def _value_counts(df: pd.DataFrame, col: str, limit: int) -> None:
    if col not in df.columns:
        print(f"    [column '{col}' not present in app.episodes]")
        return
    filled = _nonempty(df[col])
    n_filled, n_total = int(filled.sum()), len(df)
    pct = (100.0 * n_filled / n_total) if n_total else 0.0
    print(f"    {col}: {n_filled}/{n_total} non-empty ({pct:.0f}%)")
    if n_filled == 0:
        return
    vc = df.loc[filled, col].astype(str).str.strip().value_counts()
    shown = vc.head(limit)
    for val, cnt in shown.items():
        disp = val if len(val) <= 80 else val[:77] + "..."
        print(f"        {cnt:>5}  {disp!r}")
    if len(vc) > limit:
        print(f"        ... (+{len(vc) - limit} more distinct values)")


def _processed_count(df: pd.DataFrame) -> str:
    if "zarr_processed_path" not in df.columns:
        return "?"
    return f"{int(_nonempty(df['zarr_processed_path']).sum())}/{len(df)} processed"


def inspect(task: str | None, embodiments: list[str], limit: int, csv_out: str | None):
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    if df is None or df.empty:
        print("No rows in app.episodes.")
        return

    print(f"\nLoaded {len(df)} total episodes. Columns: {sorted(df.columns)}\n")

    if "is_deleted" in df.columns:
        before = len(df)
        df = df[~df["is_deleted"].fillna(False).astype(bool)]
        print(f"Dropped {before - len(df)} is_deleted rows -> {len(df)} live rows.")

    if task and "task" in df.columns:
        df = df[df["task"].astype(str).str.strip() == task]
        print(f"Filtered to task == {task!r}: {len(df)} episodes.")

    if csv_out:
        df.to_csv(csv_out, index=False)
        print(f"Wrote full filtered table to {csv_out}")

    if "embodiment" not in df.columns:
        print("\n[!] No 'embodiment' column; cannot split by embodiment.")
        return

    present = df["embodiment"].astype(str).str.strip().unique().tolist()
    targets = embodiments or sorted(present)
    print(f"\nEmbodiments present for this task: {sorted(present)}")

    for emb in targets:
        sub = df[df["embodiment"].astype(str).str.strip() == emb]
        print("\n" + "=" * 70)
        print(f"EMBODIMENT: {emb}   ({len(sub)} episodes, {_processed_count(sub)})")
        print("=" * 70)
        if sub.empty:
            print("    (no episodes)")
            continue
        for col in PAIRING_COLS:
            _value_counts(sub, col, limit)

    # --- verdict: which pairing route is viable -------------------------------
    print("\n" + "#" * 70)
    print("# PAIRING-ROUTE VERDICT")
    print("#" * 70)
    scene_ok = "scene" in df.columns and bool(_nonempty(df["scene"]).any())
    objects_ok = "objects" in df.columns and bool(_nonempty(df["objects"]).any())
    if scene_ok and objects_ok:
        print("  scene + objects are populated -> ROUTE A (metadata join) is viable:")
        print("    pair Aria<->Eva rows sharing the same (scene, objects).")
    elif scene_ok or objects_ok:
        col = "scene" if scene_ok else "objects"
        print(f"  only '{col}' is populated -> partial metadata join possible on it.")
    else:
        print("  scene/objects are EMPTY -> ROUTE A not viable.")
        print("    Fall back to ROUTE B (object/action signature from the language")
        print("    annotations) or ROUTE C (latent nearest-neighbor) for pairing.")
    print("  task_description always available as a secondary match key (see above).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="pick_place", help="task filter (or 'all')")
    p.add_argument(
        "--embodiments",
        nargs="*",
        default=["eva_bimanual", "aria_bimanual"],
        help="embodiments to report (default: eva_bimanual aria_bimanual; "
        "pass nothing after the flag to report all present)",
    )
    p.add_argument("--limit", type=int, default=25, help="max distinct values to list")
    p.add_argument(
        "--csv-out", default=None, help="optional: dump filtered table to CSV"
    )
    args = p.parse_args()
    task = None if args.task == "all" else args.task
    inspect(task, args.embodiments, args.limit, args.csv_out)


if __name__ == "__main__":
    main()
