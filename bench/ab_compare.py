"""Compare ABMetrics runs: is the arm gap bigger than the seed gap?

A single pair of runs cannot answer whether an augmentation change moved
accuracy, because two seeds of the SAME arm already differ. This reports the
between-arm difference next to the between-seed difference; the change is only
suspicious if the first is the larger of the two.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def tail_mean(series, k):
    return statistics.fmean(v for _, v in series[-k:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--arms", nargs=2, default=["loop", "batched"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--tail", type=int, default=5, help="eval points to average")
    args = ap.parse_args()

    d = Path(args.results_dir)
    runs = {}
    for arm in args.arms:
        for seed in args.seeds:
            p = d / f"{arm}_s{seed}.json"
            if p.exists():
                runs[(arm, seed)] = load(p)
            else:
                print(f"missing {p}")

    print(
        f"{'arm':>9} {'seed':>5} {'steps':>6} {'train(last100)':>15} "
        f"{'heldout(final)':>15} {f'heldout(last{args.tail})':>17} {'wall s':>8}"
    )
    for (arm, seed), r in sorted(runs.items()):
        tr = r["train_losses"]
        ho = r["heldout"]
        print(
            f"{arm:>9} {seed:>5} {len(tr):>6} "
            f"{statistics.fmean(v for _, v in tr[-100:]):>15.3f} "
            f"{ho[-1][1]:>15.3f} {tail_mean(ho, args.tail):>17.3f} "
            f"{r.get('wall_s') or 0:>8.0f}"
        )

    a0, a1 = args.arms
    print(f"\nheld-out loss, mean of last {args.tail} eval points")
    arm_gaps = []
    for seed in args.seeds:
        if (a0, seed) in runs and (a1, seed) in runs:
            x = tail_mean(runs[(a0, seed)]["heldout"], args.tail)
            y = tail_mean(runs[(a1, seed)]["heldout"], args.tail)
            arm_gaps.append(y - x)
            print(
                f"  seed {seed}: {a0} {x:.3f} -> {a1} {y:.3f}   "
                f"delta {y - x:+.3f} ({100 * (y - x) / x:+.2f}%)"
            )

    seed_gaps = []
    for arm in args.arms:
        vals = [
            tail_mean(runs[(arm, s)]["heldout"], args.tail)
            for s in args.seeds
            if (arm, s) in runs
        ]
        if len(vals) > 1:
            seed_gaps.append(max(vals) - min(vals))
            print(
                f"  {arm}: seed spread {max(vals) - min(vals):.3f} "
                f"over seeds {args.seeds}"
            )

    if arm_gaps and seed_gaps:
        mean_arm = statistics.fmean(arm_gaps)
        max_seed = max(seed_gaps)
        print(f"\nmean arm effect {mean_arm:+.3f}, largest seed spread {max_seed:.3f}")
        if abs(mean_arm) <= max_seed:
            print("ARM EFFECT WITHIN SEED NOISE")
        else:
            print("ARM EFFECT EXCEEDS SEED NOISE -- investigate")

    print(
        "\nheld-out curve (step: "
        + ", ".join(f"{a}/{s}" for a in args.arms for s in args.seeds)
        + ")"
    )
    keys = [(a, s) for a in args.arms for s in args.seeds if (a, s) in runs]
    n = min(len(runs[k]["heldout"]) for k in keys)
    for i in range(n):
        step = runs[keys[0]]["heldout"][i][0]
        cells = "  ".join(f"{runs[k]['heldout'][i][1]:9.3f}" for k in keys)
        print(f"  {step:>6}  {cells}")


if __name__ == "__main__":
    main()
