"""Recording metadata values as SQL rows, manifests and zarr attrs give them, shared by the
Lance converters, the migration planner and the committer."""

import math
from pathlib import Path

import pandas as pd

from egomimic.rldb.zarr.required_metadata import is_placeholder


def text_or_none(v) -> str | None:
    """None for a missing value (None, or NaN as pandas reads a SQL null), else str(v)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def placeholder(v) -> bool:
    """True for a missing, blank or placeholder value (required_metadata.is_placeholder)."""
    t = text_or_none(v)
    return t is None or is_placeholder(t)


def objects_list(v) -> list[str]:
    """Object names from a list or from a SQL array or JSON list literal ('{a,b}', '["a"]')."""
    if text_or_none(v) is None:
        return []
    items = v.strip("{}[]").split(",") if isinstance(v, str) else list(v)
    names = (str(o).strip().strip("\"'") for o in items if o is not None)
    return [s for s in names if s and s != "NULL"]


def read_manifest(path: Path) -> dict[str, dict]:
    """{episode_hash: row} of a parquet or CSV with an episode_hash column (a
    plan_zarr_migration manifest or an app.episodes export); blank CSV cells are NaN."""
    path = Path(path)
    df = (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, dtype={"episode_hash": str})
    )
    return {r["episode_hash"]: r for r in df.to_dict("records")}
