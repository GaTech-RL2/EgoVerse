#!/usr/bin/env python3
"""
Quick cross-account RDS parity test.

This test intentionally uses hardcoded DB credentials (no SECRETS_ARN).
It defines a local create_default_engine(host, ...) helper so we can compare
old vs new hosts directly.

Example:
  python3 egomimic/utils/aws/test_compare_old_new_rds.py \
    --old-host lowuse-pg-east2.claua8sacyu5.us-east-2.rds.amazonaws.com \
    --new-host lowuse-pg-east2.cdc8824mase4.us-east-2.rds.amazonaws.com \
    --dbname appdb \
    --user appuser \
    --password 'APPUSER_STRONG_PW' \
    --port 5432
"""

import argparse
import sys
import types

import pandas as pd
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine, inspect

# aws_sql imports boto3 at module import time, but this test does not use it.
try:
    from egomimic.utils.aws.aws_sql import episode_table_to_df
except ModuleNotFoundError as exc:
    if exc.name == "boto3":
        sys.modules["boto3"] = types.SimpleNamespace()
        from egomimic.utils.aws.aws_sql import episode_table_to_df
    else:
        raise


def create_default_engine(host: str, dbname: str, user: str, password: str, port: int):
    engine = create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}?sslmode=require",
        pool_pre_ping=True,
    )
    insp = inspect(engine)
    print(f"{host} tables in schema 'app':", insp.get_table_names(schema="app"))
    return engine


def fetch_df(
    host: str, dbname: str, user: str, password: str, port: int
) -> pd.DataFrame:
    engine = create_default_engine(host, dbname, user, password, port)
    try:
        df = episode_table_to_df(engine)
    finally:
        engine.dispose()

    if df is None:
        return pd.DataFrame()
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "episode_hash" in out.columns:
        out = out.sort_values(by=["episode_hash"], kind="mergesort")

    # Keep deterministic column order for comparison.
    out = out.reindex(sorted(out.columns), axis=1)
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old/new RDS episode tables.")
    parser.add_argument(
        "--old-host",
        default="lowuse-pg-east2.claua8sacyu5.us-east-2.rds.amazonaws.com",
    )
    parser.add_argument(
        "--new-host",
        default="lowuse-pg-east2.cdc8824mase4.us-east-2.rds.amazonaws.com",
    )
    parser.add_argument("--dbname", default="appdb")
    parser.add_argument("--user", default="appuser")
    parser.add_argument("--password", default="APPUSER_STRONG_PW")
    parser.add_argument("--port", type=int, default=5432)
    args = parser.parse_args()

    old_df = fetch_df(args.old_host, args.dbname, args.user, args.password, args.port)
    new_df = fetch_df(args.new_host, args.dbname, args.user, args.password, args.port)

    old_df_n = normalize(old_df)
    new_df_n = normalize(new_df)

    print(f"old rows={len(old_df_n)} cols={len(old_df_n.columns)}")
    print(f"new rows={len(new_df_n)} cols={len(new_df_n.columns)}")

    assert_frame_equal(old_df_n, new_df_n, check_dtype=False, check_like=False)
    print("PASS: old/new DataFrames are identical.")


if __name__ == "__main__":
    main()
