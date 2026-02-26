#!/usr/bin/env python3
"""One-time migration: add zarr status columns to app.episodes."""

from __future__ import annotations

import sys

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

DDL_STATEMENTS = (
    (
        "zarr_processed_path",
        """
        ALTER TABLE app.episodes
        ADD COLUMN IF NOT EXISTS zarr_processed_path TEXT NOT NULL DEFAULT ''
        """,
    ),
    (
        "zarr_mp4_path",
        """
        ALTER TABLE app.episodes
        ADD COLUMN IF NOT EXISTS zarr_mp4_path TEXT NOT NULL DEFAULT ''
        """,
    ),
    (
        "zarr_processing_error",
        """
        ALTER TABLE app.episodes
        ADD COLUMN IF NOT EXISTS zarr_processing_error TEXT NOT NULL DEFAULT ''
        """,
    ),
)


def main() -> int:
    engine = create_default_engine()
    try:
        with engine.begin() as conn:
            for column_name, ddl in DDL_STATEMENTS:
                conn.execute(text(ddl))
                print(f"Applied/verified column: {column_name}")

        verify_sql = text(
            """
            SELECT column_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema='app'
              AND table_name='episodes'
              AND column_name IN (
                  'zarr_processed_path',
                  'zarr_mp4_path',
                  'zarr_processing_error'
              )
            ORDER BY column_name
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(verify_sql).all()

        if len(rows) != 3:
            print(
                "ERROR: expected 3 zarr columns in app.episodes after migration.",
                file=sys.stderr,
            )
            for row in rows:
                print(f"Found column: {row.column_name}", file=sys.stderr)
            return 1

        for row in rows:
            print(
                "Verified "
                f"{row.column_name}: is_nullable={row.is_nullable}, default={row.column_default}"
            )

        print("Migration completed successfully.")
        return 0
    except Exception as exc:
        print(f"ERROR: migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
