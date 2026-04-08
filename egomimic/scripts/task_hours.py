"""Print the list of tasks ordered by hours of recorded data.

Reads the `app.episodes` SQL table, filters to lab == "mecka", totals the
`num_frames` for each task, and converts to hours assuming 30 fps episodes.

By default only episodes that are actually trainable (non-empty
`zarr_processed_path`, matching the training data configs) are counted, so the
hours here reflect what a training run would really see.
"""

import argparse

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

FPS = 30
LAB = "mecka"
# Episode excluded by the mecka_pi* training data configs.
BAD_HASH = "692ea23dc621d7f4aac3aaa1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-hours",
        type=float,
        default=1.0,
        help="Only show tasks with at least this many hours of data (default: 1.0).",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Count every mecka row instead of only trainable (zarr-processed) ones.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names; show only these and their combined total "
        "(useful for verifying a subset). Overrides --min-hours.",
    )
    args = parser.parse_args()

    engine = create_default_engine()
    df = episode_table_to_df(engine)

    df = df[df["lab"] == LAB]
    if not args.all_rows:
        # Match the mecka_pi* training data filter.
        df = df[(df["zarr_processed_path"] != "") & (df["episode_hash"] != BAD_HASH)]

    # Total frames per task, then convert to hours at 30 fps.
    frames_per_task = df.groupby("task")["num_frames"].sum()
    hours_per_task = (frames_per_task / FPS / 3600).sort_values(ascending=False)
    counts = df.groupby("task").size()

    scope = "all mecka rows" if args.all_rows else "trainable (zarr-processed) rows"

    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in wanted if t not in hours_per_task.index]
        if missing:
            print(f"WARNING: tasks not found in data: {missing}\n")
        shown = hours_per_task[hours_per_task.index.isin(wanted)]
        header = f"Selected subset of {len(shown)} tasks"
    else:
        shown = hours_per_task[hours_per_task >= args.min_hours]
        header = f"Tasks filtered to >= {args.min_hours} hr"

    print(f"\n{header} for lab '{LAB}' ({FPS} fps, {scope}):\n")
    print(f"{'task':<35}{'episodes':>10}{'hours':>12}")
    print("-" * 57)
    for task, hours in shown.items():
        print(f"{task:<35}{counts[task]:>10}{hours:>12.2f}")
    print("-" * 57)
    print(
        f"{'SHOWN':<35}{counts[shown.index].sum():>10}{shown.sum():>12.2f}  "
        f"({len(shown)} tasks)"
    )
    print(
        f"{'TOTAL (all tasks)':<35}{counts.sum():>10}{hours_per_task.sum():>12.2f}  "
        f"({len(hours_per_task)} tasks)"
    )


if __name__ == "__main__":
    main()
