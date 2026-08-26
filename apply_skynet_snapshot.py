#!/usr/bin/env python3
"""
Compare / apply the Skynet code snapshot pulled by ``pull_dp3_adapt3r.sh``.

The DP3 / Adapt3R model + serving code lives only in the Skynet working tree
(``/coc/flash7/czhang883/Documents/EgoVerse``), not on the GitHub branch, so
serving those checkpoints locally needs the changed/new ``egomimic/`` files.
This tool keeps that transparent instead of blind-rsyncing over local edits:

  default        dry-run: categorize snapshot vs local (new / changed /
                 identical), write full unified diffs to
                 skynet_snapshot/DIFF_REPORT.txt
  --apply        copy NEW files in, and replace CHANGED files after saving
                 ``<file>.local.bak`` backups
  --only-new     copy only NEW files (never touch an existing local file)

Files where the LOCAL side has uncommitted work you may care about are still
replaced under --apply (with a .local.bak); the report calls them out — known
example: egomimic/serving/egoverse_policy.py carries a local num_samples /
ghost-fan feature used by --viz-num-samples in the SEW rollouts. If the
snapshot version lacks it, re-merge from the .local.bak (feature is additive).

Usage:
  cd ~/RB_Y1_workspace/EgoVerse
  python apply_skynet_snapshot.py
  python apply_skynet_snapshot.py --apply
"""

from __future__ import annotations

import argparse
import difflib
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SNAP = REPO / "skynet_snapshot" / "egomimic"
LOCAL = REPO / "egomimic"

# Local files known to carry deliberate robot-host-only edits; flagged loudly.
WATCHLIST = {
    "serving/egoverse_policy.py": "local num_samples ghost-fan feature (SEW --viz-num-samples)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--apply", action="store_true",
                    help="Copy new files and replace changed files (with .local.bak).")
    ap.add_argument("--only-new", action="store_true",
                    help="Copy only files that do not exist locally.")
    args = ap.parse_args()

    if not SNAP.is_dir():
        print(f"ERROR: snapshot not found at {SNAP}.\n"
              "Run `bash pull_dp3_adapt3r.sh` first.")
        return 1

    new_files: list[Path] = []
    changed: list[Path] = []
    identical = 0
    for src in sorted(SNAP.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(SNAP)
        dst = LOCAL / rel
        if not dst.exists():
            new_files.append(rel)
        elif filecmp.cmp(src, dst, shallow=False):
            identical += 1
        else:
            changed.append(rel)

    print(f"Snapshot vs local egomimic/: {identical} identical, "
          f"{len(new_files)} new, {len(changed)} changed.\n")

    if new_files:
        print("NEW (snapshot-only — required for DP3/Adapt3R unpickle/serving):")
        for rel in new_files:
            print(f"  + {rel}")
        print()
    if changed:
        print("CHANGED (snapshot differs from local):")
        for rel in changed:
            note = WATCHLIST.get(str(rel))
            flag = f"   << WATCH: {note}" if note else ""
            print(f"  ~ {rel}{flag}")
        print()

    # Full diff report for the changed set.
    report = SNAP.parent / "DIFF_REPORT.txt"
    with open(report, "w") as f:
        for rel in changed:
            local_lines = (LOCAL / rel).read_text(errors="replace").splitlines(keepends=True)
            snap_lines = (SNAP / rel).read_text(errors="replace").splitlines(keepends=True)
            f.writelines(difflib.unified_diff(
                local_lines, snap_lines,
                fromfile=f"local/{rel}", tofile=f"skynet/{rel}",
            ))
            f.write("\n")
    print(f"Unified diffs for changed files: {report}")

    if not (args.apply or args.only_new):
        print("\nDry run only. Use --apply (or --only-new) to copy files.")
        return 0

    n_copied = 0
    for rel in new_files:
        dst = LOCAL / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SNAP / rel, dst)
        n_copied += 1
    print(f"\nCopied {n_copied} new files.")

    if args.apply and not args.only_new:
        n_replaced = 0
        for rel in changed:
            dst = LOCAL / rel
            bak = dst.with_suffix(dst.suffix + ".local.bak")
            if not bak.exists():
                shutil.copy2(dst, bak)
            shutil.copy2(SNAP / rel, dst)
            n_replaced += 1
            if str(rel) in WATCHLIST:
                print(f"  !! replaced watched file {rel} "
                      f"(local kept at {bak.name}) — {WATCHLIST[str(rel)]}")
        print(f"Replaced {n_replaced} changed files (backups: <file>.local.bak).")

    print("\nNext: serve a pcd checkpoint and run the replay plumbing test:")
    print("  source emimic/bin/activate")
    print("  python egomimic/scripts/serve_policy.py \\")
    print("    --checkpoint checkpoints/RBY1_dp3_pcd1024/dp3_pcd1024_glass_2k/"
          "checkpoints/epoch_epoch=1299.ckpt --port 8000")
    print("  python egomimic/scripts/test_serve_pcd_replay.py "
          "--dataset datasets/rby1_teleop_pcd1024_glass --episode 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
