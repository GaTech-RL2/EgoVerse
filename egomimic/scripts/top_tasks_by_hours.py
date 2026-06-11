"""Print the top 10 tasks by total hours of episode data, optionally filtered.

Example:
    python -m egomimic.scripts.top_tasks_by_hours --filter lab=mecka
    python -m egomimic.scripts.top_tasks_by_hours --filter lab=mecka embodiment=human
"""

import argparse

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

FPS = 30


def parse_filters(filter_args):
    filters = {}
    for arg in filter_args:
        if "=" not in arg:
            raise ValueError(f"Filter must be in 'column=value' form, got: {arg}")
        col, val = arg.split("=", 1)
        filters[col.strip()] = val.strip()
    return filters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        nargs="*",
        default=[],
        help="Equality filters like lab=mecka. Multiple filters are ANDed.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of tasks to show.")
    args = parser.parse_args()

    filters = parse_filters(args.filter)

    engine = create_default_engine()
    df = episode_table_to_df(engine)

    df = df[~df["is_deleted"]]
    df = df[df["num_frames"] > 0]

    for col, val in filters.items():
        if col not in df.columns:
            raise ValueError(f"Unknown column '{col}'. Available: {list(df.columns)}")
        df = df[df[col].astype(str) == val]

    if df.empty:
        print(f"No episodes match filters {filters}.")
        return

    hours = df.groupby("task")["num_frames"].sum() / FPS / 3600
    top = hours.sort_values(ascending=False).head(args.top)

    filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items()) or "(no filter)"
    print(f"\nTop {args.top} tasks by hours [{filter_desc}] @ {FPS} fps:\n")
    print(f"{'task':<40} {'hours':>10}  {'episodes':>10}")
    print("-" * 64)
    counts = df.groupby("task").size()
    for task, hrs in top.items():
        print(f"{task:<40} {hrs:>10.2f}  {counts[task]:>10}")


if __name__ == "__main__":
    main()
