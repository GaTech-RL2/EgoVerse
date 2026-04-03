#!/usr/bin/env python3
"""
Migration script: copy robot_name into embodiment for ALL rows, then drop the
robot_name column from app.episodes entirely.

Steps:
  1. UPDATE app.episodes SET embodiment = robot_name  (all rows)
  2. ALTER TABLE app.episodes DROP COLUMN robot_name


Usage:
    python3 fix_sql_embodiment.py
    python3 fix_sql_embodiment.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine


def main():
    parser = argparse.ArgumentParser(
        description="Copy robot_name→embodiment for all rows, then drop robot_name column"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = create_default_engine()

    # Preview: count rows and show a sample
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM app.episodes")).scalar()
        sample = (
            conn.execute(
                text(
                    "SELECT episode_hash, embodiment, robot_name FROM app.episodes LIMIT 5"
                )
            )
            .mappings()
            .all()
        )

    print(f"Total rows: {count}")
    print("Sample (first 5):")
    for row in sample:
        print(
            f"  {row['episode_hash']}  embodiment={row['embodiment']!r}  robot_name={row['robot_name']!r}"
        )
    print("Will set embodiment = robot_name for all rows, then drop robot_name column.")

    if args.dry_run:
        print("[DRY RUN] No changes written.")
        return 0

    with engine.begin() as conn:
        conn.execute(text("UPDATE app.episodes SET embodiment = robot_name"))
        conn.execute(text("ALTER TABLE app.episodes DROP COLUMN robot_name"))

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
