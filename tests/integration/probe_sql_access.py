from __future__ import annotations

import argparse

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe SQL read access and no-op write permission via create_default_engine()."
    )
    parser.add_argument(
        "--table",
        default="app.episodes",
        help="Fully-qualified table to probe. Defaults to app.episodes.",
    )
    args = parser.parse_args()

    engine = create_default_engine()

    select_sql = text(f"SELECT count(*) FROM {args.table}")
    update_sql = text(f"UPDATE {args.table} SET num_frames = num_frames WHERE false")

    with engine.connect() as conn:
        row_count = conn.execute(select_sql).scalar_one()
    print({"select_ok": True, "table": args.table, "row_count": row_count})

    try:
        with engine.begin() as conn:
            result = conn.execute(update_sql)
        print(
            {
                "noop_update_ok": True,
                "table": args.table,
                "rows_affected": result.rowcount,
            }
        )
    except Exception as exc:
        print(
            {
                "noop_update_ok": False,
                "table": args.table,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
