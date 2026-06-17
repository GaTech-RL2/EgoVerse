"""Download + arrange a DROID-style multi-task RoboTwin split for PI RICL.

Downloads aloha-agilex slices for the given train/eval tasks and symlinks N demos
per task into ``train_root/`` and ``eval_root/`` with the ``<task>/{data,instructions}``
layout ``RoboTwinCorpus`` expects (one group per task). Train and eval task sets are
DISJOINT, so eval-task demos are never seen during training — at eval they're used
only as test-time in-context retrieval examples (within-group leave-one-out).

Example:
    python egomimic/ricl/scripts/setup_multitask_robotwin.py \\
      --train-tasks beat_block_hammer,lift_pot,... --eval-tasks click_alarmclock,... \\
      --n-demos 20 --setting clean \\
      --raw-dir egomimic/ricl/outputs/robotwin_raw_mt \\
      --train-root egomimic/ricl/outputs/robotwin_mt/train \\
      --eval-root  egomimic/ricl/outputs/robotwin_mt/eval
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from egomimic.ricl.scripts.download_robotwin import download_robotwin_slice


def _ep_num(path: str) -> int:
    m = re.search(r"episode(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 1 << 30


def arrange_task(extract_dir: str, task: str, dst_root: str, n: int) -> int:
    """Symlink the first ``n`` (numeric) episodes of ``task`` into dst_root/<task>."""
    data_dirs = sorted(
        glob.glob(os.path.join(extract_dir, "**", "data"), recursive=True)
    )
    data_dirs = [d for d in data_dirs if glob.glob(os.path.join(d, "episode*.hdf5"))]
    if not data_dirs:
        raise FileNotFoundError(f"no episode*.hdf5 under {extract_dir}")
    data_dir = data_dirs[0]
    instr_dir = os.path.join(os.path.dirname(data_dir), "instructions")
    eps = sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5")), key=_ep_num)[:n]

    dst_data = os.path.join(dst_root, task, "data")
    dst_instr = os.path.join(dst_root, task, "instructions")
    os.makedirs(dst_data, exist_ok=True)
    os.makedirs(dst_instr, exist_ok=True)
    for ep in eps:
        name = os.path.basename(ep)
        link = os.path.join(dst_data, name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(ep), link)
        ij = os.path.join(instr_dir, name.replace(".hdf5", ".json"))
        if os.path.exists(ij):
            ilink = os.path.join(dst_instr, os.path.basename(ij))
            if not os.path.lexists(ilink):
                os.symlink(os.path.abspath(ij), ilink)
    return len(eps)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-tasks", required=True, help="comma-separated task names")
    p.add_argument("--eval-tasks", required=True, help="comma-separated task names")
    p.add_argument("--n-demos", type=int, default=20)
    p.add_argument("--embodiment", default="aloha-agilex")
    p.add_argument("--setting", default="clean", choices=["clean", "randomized"])
    p.add_argument("--raw-dir", required=True, help="download/extract scratch dir")
    p.add_argument("--train-root", required=True)
    p.add_argument("--eval-root", required=True)
    args = p.parse_args()

    train = [t for t in args.train_tasks.split(",") if t]
    evald = [t for t in args.eval_tasks.split(",") if t]
    overlap = set(train) & set(evald)
    if overlap:
        raise SystemExit(
            f"train/eval tasks must be DISJOINT; overlap: {sorted(overlap)}"
        )

    for split, tasks, root in (
        ("train", train, args.train_root),
        ("eval", evald, args.eval_root),
    ):
        for t in tasks:
            extract_dir = download_robotwin_slice(
                args.raw_dir, task=t, embodiment=args.embodiment, setting=args.setting
            )
            n = arrange_task(extract_dir, t, root, args.n_demos)
            print(f"[{split}] {t}: linked {n} demos -> {root}/{t}", flush=True)

    print(
        f"DONE: train {len(train)} tasks -> {args.train_root} | "
        f"eval {len(evald)} tasks -> {args.eval_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
