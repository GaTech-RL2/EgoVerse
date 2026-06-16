from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import MetaData, Table, update

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)


def main():
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    df = df[df["embodiment"] == "eva"]

    s3 = boto3.client("s3")
    now = datetime.now(timezone.utc)

    def _processed_path_to_stats_s3(processed_path: str | None):
        if not processed_path:
            return None
        if processed_path.startswith("s3://"):
            path = processed_path[len("s3://") :]
        elif ":/" in processed_path:
            bucket, key = processed_path.split(":/", 1)
            path = f"{bucket}/{key.lstrip('/')}"
        else:
            return None
        bucket, key = path.split("/", 1)
        stats_key = f"{key.rstrip('/')}/meta/stats.json"
        return bucket, stats_key

    def _stats_last_modified(processed_path: str | None):
        s3_loc = _processed_path_to_stats_s3(processed_path)
        if not s3_loc:
            return None
        bucket, key = s3_loc
        try:
            return s3.head_object(Bucket=bucket, Key=key)["LastModified"]
        except ClientError:
            return None

    df["stats_json_last_modified"] = df["processed_path"].apply(_stats_last_modified)
    df["stats_json_age_days"] = df["stats_json_last_modified"].apply(
        lambda dt: (now - dt).total_seconds() / 86400 if dt else None
    )

    # cutoff = datetime(2026, 1, 14, tzinfo=timezone.utc)
    stale_df = df[(df["stats_json_age_days"] > 1) | (df["stats_json_age_days"].isna())]
    episodes_tbl = Table("episodes", MetaData(), autoload_with=engine, schema="app")
    clear_stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash.in_(stale_df["episode_hash"].tolist()))
        .values(processed_path="")
    )
    # NOTE: Do not execute yet. To run:
    with engine.begin() as conn:
        conn.execute(clear_stmt)


if __name__ == "__main__":
    main()
