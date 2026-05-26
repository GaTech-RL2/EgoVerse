#!/usr/bin/env python3
"""Scan a logs/ tree for .ckpt files and classify each checkpoints/ dir into
a cleanup bucket. Prints a plan; does NOT delete anything.

Buckets (based on newest mtime of any .ckpt in the dir):
  - skip:       newest ckpt < 14 days old
  - downsample: 14 <= age < 30 days; keep middle-epoch + highest-epoch ckpts
  - delete_all: age >= 30 days; delete every .ckpt
"""

import os
import re
import sys
import time
from pathlib import Path

EPOCH_RE = re.compile(r"epoch_epoch=(\d+)\.ckpt$")
DAY = 86400.0


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def pick_keepers(ckpts):
    """Given list of (path, mtime, size), return the set of paths to keep.

    Prefer epoch-numbered ckpts: keep middle + highest by epoch number.
    If none have epoch numbers, keep the single newest by mtime.
    """
    epoch_ckpts = []
    non_epoch = []
    for p, mt, sz in ckpts:
        m = EPOCH_RE.search(p)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
        else:
            non_epoch.append((mt, p))

    keep = set()
    if epoch_ckpts:
        epoch_ckpts.sort()
        last = epoch_ckpts[-1][1]
        middle = epoch_ckpts[len(epoch_ckpts) // 2][1]
        keep.add(last)
        keep.add(middle)
    else:
        # No epoch-numbered ckpts — keep newest by mtime
        non_epoch.sort(reverse=True)
        if non_epoch:
            keep.add(non_epoch[0][1])
    return keep


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "logs"
    root = Path(root).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    now = time.time()
    # Group ckpts by their containing directory
    by_dir: dict[str, list] = {}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        ckpts = [f for f in filenames if f.endswith(".ckpt")]
        if not ckpts:
            continue
        entries = []
        for f in ckpts:
            full = os.path.join(dirpath, f)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            entries.append((full, st.st_mtime, st.st_size))
        if entries:
            by_dir[dirpath] = entries

    delete_all = []   # list of (dir, age_days, [(path, size), ...])
    downsample = []   # list of (dir, age_days, keepers, [(path, size), ...])
    skipped = 0

    for d, entries in sorted(by_dir.items()):
        newest_mtime = max(e[1] for e in entries)
        age_days = (now - newest_mtime) / DAY
        if age_days < 14:
            skipped += 1
            continue
        if age_days >= 30:
            delete_all.append(
                (d, age_days, [(p, sz) for p, _mt, sz in entries])
            )
        else:
            keep = pick_keepers(entries)
            to_delete = [(p, sz) for p, _mt, sz in entries if p not in keep]
            if to_delete:
                downsample.append((d, age_days, sorted(keep), to_delete))

    total_files = 0
    total_bytes = 0
    rm_commands = []

    print("=" * 70)
    print(f"CHECKPOINT CLEANUP PLAN — root: {root}")
    print("=" * 70)
    print()
    print("Rules:")
    print("  - age > 30 days: delete ALL .ckpt in directory")
    print("  - age 14-30 days: keep only middle-epoch + highest-epoch ckpts")
    print("  - age < 14 days: skip")
    print()

    if delete_all:
        print(f"### DELETE-ALL ({len(delete_all)} directories, > 30 days old)")
        for d, age, files in delete_all:
            print(f"\n  {d}  ({age:.1f} days old)")
            for p, sz in sorted(files):
                print(f"    DEL  {human_bytes(sz):>10}  {os.path.basename(p)}")
                rm_commands.append(p)
                total_files += 1
                total_bytes += sz
        print()

    if downsample:
        print(f"### DOWNSAMPLE ({len(downsample)} directories, 14-30 days old)")
        for d, age, keep, to_delete in downsample:
            print(f"\n  {d}  ({age:.1f} days old)")
            for p in keep:
                print(f"    KEEP                  {os.path.basename(p)}")
            for p, sz in sorted(to_delete):
                print(f"    DEL  {human_bytes(sz):>10}  {os.path.basename(p)}")
                rm_commands.append(p)
                total_files += 1
                total_bytes += sz
        print()

    print("=" * 70)
    print(f"Skipped (< 14 days):  {skipped} directories")
    print(f"To free:              {total_files} files, {human_bytes(total_bytes)}")
    print("=" * 70)

    if rm_commands:
        print()
        print("### rm commands that will run (one per line):")
        for p in rm_commands:
            # Quote path safely
            print(f"rm {shell_quote(p)}")


def shell_quote(s: str) -> str:
    if not s or any(c in s for c in " \t\n\"'\\$`!*?[](){}<>|&;#"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


if __name__ == "__main__":
    main()
