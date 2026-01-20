import json
import os
from dataclasses import asdict, dataclass

import boto3
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    delete,
    insert,
    inspect,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError


@dataclass
class TableRow:
    episode_hash: str
    operator: str
    lab: str
    task: str
    embodiment: str
    robot_name: str
    num_frames: int = -1  # Updateable
    task_description: str = ""
    scene: str = ""
    objects: str = ""
    processed_path: str = ""  # Updateable
    processing_error: str = ""  # Updateable
    mp4_path: str = ""  # Updateable
    is_deleted: bool = False
    is_eval: bool = False
    eval_score: float = -1
    eval_success: bool = True


def create_default_engine():
    # Try to get credentials from Secrets Manager if SECRETS_ARN is set
    SECRETS_ARN = os.environ.get("SECRETS_ARN")
    if SECRETS_ARN:
        secrets = boto3.client("secretsmanager")
        sec = secrets.get_secret_value(SecretId=SECRETS_ARN)["SecretString"]
        cfg = json.loads(sec)
        HOST = cfg.get("host", cfg.get("HOST"))
        DBNAME = cfg.get("dbname", cfg.get("DBNAME", "appdb"))
        USER = cfg.get("username", cfg.get("user", cfg.get("USER")))
        PASSWORD = cfg.get("password", cfg.get("PASSWORD"))
        PORT = cfg.get("port", 5432)
    else:
        # Fallback to hardcoded values for local testing
        HOST = "lowuse-pg-east2.claua8sacyu5.us-east-2.rds.amazonaws.com"
        DBNAME = "appdb"
        USER = "appuser"
        PASSWORD = "APPUSER_STRONG_PW"
        PORT = 5432

    # --- 1) connect via SQLAlchemy ---
    engine = create_engine(
        f"postgresql+psycopg://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require",
        pool_pre_ping=True,
    )

    # --- 2) list tables in the schema 'app' ---
    insp = inspect(engine)
    print("Tables in schema 'app':", insp.get_table_names(schema="app"))

    return engine


def _episodes_table(engine):
    md = MetaData()
    return Table("episodes", md, autoload_with=engine, schema="app")


def add_episode(engine, episode) -> bool:
    """
    Insert one row into app.episodes.
    Raises sqlalchemy.exc.IntegrityError if the row violates a unique/PK constraint.
    """
    episodes_tbl = _episodes_table(engine)
    row = asdict(episode)

    try:
        with engine.begin() as conn:
            conn.execute(insert(episodes_tbl).values(**row))
        return True
    except IntegrityError as e:
        # Duplicate (or other constraint) → surface a clear error
        raise RuntimeError(f"Insert failed (likely duplicate episode_hash): {e}") from e


def update_episode(engine, episode: TableRow):
    """
    Update a row in a PostgreSQL table using SQLAlchemy Core (SQLAlchemy 2 compatible).

    Args:
        engine: SQLAlchemy Engine instance.
        episode (TableRow): TableRow object.
    """
    episodes_tbl = _episodes_table(engine)

    # Create a dict out of episode fields
    row = asdict(episode)
    episode_hash = row.pop("episode_hash")  # Remove episode_hash from the update values

    stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash == episode_hash)
        .values(**row)
    )

    with (
        engine.begin() as conn
    ):  # use engine.begin() for transactional context (SQLAlchemy 2 style)
        conn.execute(stmt)
    return True


def episode_hash_to_table_row(engine, episode_hash):
    t = _episodes_table(engine)
    fields = set(TableRow.__dataclass_fields__.keys())
    if {c.name for c in t.columns} != fields:
        raise ValueError("Schema mismatch between TableRow and app.episodes")

    stmt = select(t).where(t.c.episode_hash == episode_hash).limit(1)
    with engine.connect() as conn:
        rec = conn.execute(stmt).mappings().first()

    return None if rec is None else TableRow(**rec)


def delete_episodes(engine, episode_hashes: list[int]):
    episodes_tbl = _episodes_table(engine)
    with engine.begin() as conn:
        conn.execute(
            delete(episodes_tbl).where(episodes_tbl.c.episode_hash.in_(episode_hashes))
        )
    return True


def delete_all_episodes(engine):
    episodes_tbl = _episodes_table(engine)
    with engine.begin() as conn:
        conn.execute(delete(episodes_tbl))
    return True


def episode_table_to_df(engine):
    """
    Prints all rows in the 'episodes' table in a nicely formatted table.
    """
    metadata = MetaData()
    episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")

    import pandas as pd

    with engine.connect() as conn:
        result = conn.execute(select(episodes_tbl))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if not df.empty:
            return df
        else:
            print("No rows found in table 'episodes'.")
            return df
