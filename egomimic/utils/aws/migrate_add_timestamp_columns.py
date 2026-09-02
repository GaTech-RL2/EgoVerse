#!/usr/bin/env python3
"""One-time migration: add created_at / updated_at timestamps to app.episodes.

- created_at TIMESTAMPTZ NOT NULL DEFAULT now()  (Time Created; set once on INSERT)
- updated_at TIMESTAMPTZ NOT NULL DEFAULT now()  (Last Modified)

`updated_at` is bumped automatically on every UPDATE by a BEFORE UPDATE trigger
(app.set_updated_at), so no application code needs to set it. For the existing
rows, `created_at` is backfilled from the episode_hash when it is a recording
timestamp (``YYYY-MM-DD-HH-MM-SS-ffffff``, interpreted as UTC); UUID/ObjectId
hashed vendor episodes keep the migration-time default. Idempotent: safe to
re-run (ADD COLUMN IF NOT EXISTS / CREATE OR REPLACE / DROP TRIGGER IF EXISTS),
but the created_at backfill re-parses every timestamp-hashed row each run
(harmless — it writes the same value).
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

# recording-timestamp episode_hash, e.g. 2026-07-17-02-06-28-000000
_TS_HASH_REGEX = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{6}$"


def main() -> int:
    engine = create_default_engine()
    try:
        # 1) columns + created_at backfill (one transaction; timezone pinned to
        #    UTC so to_timestamp reads the hash's wall-clock as UTC). The trigger
        #    does not exist yet, so this UPDATE does not touch updated_at.
        with engine.begin() as conn:
            conn.execute(text("SET LOCAL timezone TO 'UTC'"))
            conn.execute(text(
                "ALTER TABLE app.episodes ADD COLUMN IF NOT EXISTS "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ))
            conn.execute(text(
                "ALTER TABLE app.episodes ADD COLUMN IF NOT EXISTS "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ))
            res = conn.execute(text(
                "UPDATE app.episodes "
                "SET created_at = to_timestamp(episode_hash, 'YYYY-MM-DD-HH24-MI-SS-US') "
                f"WHERE episode_hash ~ '{_TS_HASH_REGEX}'"
            ))
            print(f"Added created_at/updated_at; backfilled created_at from "
                  f"episode_hash for {res.rowcount} timestamp-hashed rows.")

        # 2) updated_at auto-update trigger (separate transaction so a
        #    permissions issue here doesn't roll back the columns above).
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "CREATE OR REPLACE FUNCTION app.set_updated_at() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN NEW.updated_at = now(); "
                    "RETURN NEW; END; $$"
                ))
                conn.execute(text(
                    "DROP TRIGGER IF EXISTS episodes_set_updated_at ON app.episodes"
                ))
                conn.execute(text(
                    "CREATE TRIGGER episodes_set_updated_at BEFORE UPDATE ON "
                    "app.episodes FOR EACH ROW EXECUTE FUNCTION app.set_updated_at()"
                ))
            print("Installed BEFORE UPDATE trigger episodes_set_updated_at.")
        except Exception as exc:
            print(f"WARNING: columns added but trigger install failed: {exc}",
                  file=sys.stderr)
            print("  updated_at will only reflect inserts until the trigger exists.",
                  file=sys.stderr)

        # 3) verify
        with engine.connect() as conn:
            cols = conn.execute(text(
                "SELECT column_name, is_nullable, data_type, column_default "
                "FROM information_schema.columns WHERE table_schema='app' "
                "AND table_name='episodes' AND column_name IN "
                "('created_at','updated_at') ORDER BY column_name"
            )).all()
            trig = conn.execute(text(
                "SELECT tgname FROM pg_trigger WHERE tgname='episodes_set_updated_at' "
                "AND NOT tgisinternal"
            )).all()
            sample = conn.execute(text(
                "SELECT episode_hash, created_at, updated_at FROM app.episodes "
                f"WHERE episode_hash ~ '{_TS_HASH_REGEX}' ORDER BY episode_hash DESC LIMIT 3"
            )).all()

        if len(cols) != 2:
            print(f"ERROR: expected 2 timestamp columns, found {len(cols)}.",
                  file=sys.stderr)
            return 1
        for c in cols:
            print(f"Verified {c.column_name}: {c.data_type}, "
                  f"nullable={c.is_nullable}, default={c.column_default}")
        print(f"Trigger present: {bool(trig)}")
        print("Sample backfilled rows (hash -> created_at should match):")
        for s in sample:
            print(f"  {s.episode_hash}  created_at={s.created_at}  updated_at={s.updated_at}")
        print("Migration completed successfully.")
        return 0
    except Exception as exc:
        print(f"ERROR: migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
