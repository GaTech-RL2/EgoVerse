"""Resolve a model-data-config-style filter into a list of rating spans.

This mirrors how episodes are selected to feed a model. The model data configs
(egomimic/hydra_configs/data/*.yaml) select episodes with a ``DatasetFilter``:

    filters:
      _target_: egomimic.rldb.filters.DatasetFilter
      filter_lambdas:
        - "lambda row: row['task'] in ['folding_clothes', ...]"
        - "lambda row: row['data_type'] == 'flagship'"

The ego-rating ``dataset`` config block accepts the SAME ``filter_lambdas`` (or a
path to one of those data configs, or a structured ``where`` / raw ``sql_where``),
resolves them against ``app.episodes``, and turns every matching episode into one
rating span:

    span_id   = episode_hash
    video_uri = episode_hash      (-> presigned R2 URL via s3_resolver/MongoDB)
    start/end = 0 .. num_frames/fps
    scene/operator/label from the episode row.

Performance: ``app.episodes`` has 200k+ rows, far too many to load and filter in
Python. So the common ``filter_lambdas`` shapes — ``row['col'] in [...]`` and
``row['col'] <op> const`` — are translated to a SQL ``WHERE`` and pushed to
Postgres, with ``LIMIT`` / ``ORDER BY random()`` pushed down too. Predicates we
can't translate (e.g. DataFrame-level lambdas) are applied in Python on the
already-bounded result set, so each filter group must contain at least one
SQL-pushable predicate (or a ``where`` / ``sql_where``).

We replicate the tiny ``DatasetFilter`` rather than importing
egomimic.rldb.filters (which pulls in scaleapi). ``filter_lambdas`` / ``sql_where``
are eval'd / interpolated — the config is trusted operator input, same trust
model as the training configs (column names are validated against the table).
"""

from __future__ import annotations

import ast
import os
import random
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import yaml

from backend import s3_resolver  # reuse its quote-stripping load_env

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_FPS = 30.0

_engine = None
_columns: Optional[set[str]] = None
_lock = threading.Lock()

_CMP_OPS = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


# ---------------------------------------------------------------------------
# DatasetFilter (replica of egomimic.rldb.filters.DatasetFilter, base only) —
# used only for the Python refinement pass on already-bounded result sets.
# ---------------------------------------------------------------------------
class DatasetFilter:
    def __init__(self, filter_lambdas: Optional[Sequence[str]] = None) -> None:
        self.filter_lambdas = list(filter_lambdas or [])
        self.filters = []
        for expr in self.filter_lambdas:
            try:
                predicate = eval(expr)  # noqa: S307 — trusted config input
            except Exception as exc:
                raise ValueError(f"Invalid filter: {expr}") from exc
            if not callable(predicate):
                raise ValueError(f"Invalid filter (not callable): {expr}")
            self.filters.append(predicate)

    def filter_df(self, df: Any) -> Any:
        for predicate in self.filters:
            try:
                result = predicate(df)
                if hasattr(result, "iterrows"):
                    df = result
            except Exception:
                pass
        return df

    def matches(self, row: Mapping[str, Any]) -> bool:
        row = dict(row)
        if row.get("is_deleted", False):
            return False
        for expr, predicate in zip(self.filter_lambdas, self.filters):
            try:
                result = predicate(row)
            except (TypeError, KeyError, AttributeError):
                continue
            if not isinstance(result, bool):
                raise TypeError(f"Filter must return bool: {expr}")
            if not result:
                return False
        return True


# ---------------------------------------------------------------------------
# DB engine + columns (mirrors aws_sql.create_default_engine)
# ---------------------------------------------------------------------------
def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        s3_resolver.load_env()  # strips quotes (DATABASE_URL may be quoted)
        from sqlalchemy import URL, create_engine

        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            ).replace("postgres://", "postgresql+psycopg://", 1)
            _engine = create_engine(database_url, pool_pre_ping=True)
        elif os.environ.get("PG_HOST"):
            _engine = create_engine(
                URL.create(
                    "postgresql+psycopg",
                    username=os.environ["PG_USER"],
                    password=os.environ["PG_PASSWORD"],
                    host=os.environ["PG_HOST"],
                    port=int(os.environ.get("PG_PORT", "5432")),
                    database=os.environ.get("PG_DATABASE", "defaultdb"),
                    query={"sslmode": "require"},
                ),
                pool_pre_ping=True,
            )
        else:
            raise RuntimeError(
                "No DB credentials: set DATABASE_URL (or PG_HOST/PG_USER/"
                "PG_PASSWORD) in the env or ~/.egoverse_env."
            )
    return _engine


def _episode_columns() -> set[str]:
    global _columns
    if _columns is not None:
        return _columns
    from sqlalchemy import text

    with _get_engine().connect() as conn:
        res = conn.execute(text("SELECT * FROM app.episodes LIMIT 0"))
        _columns = set(res.keys())
    return _columns


def clear_cache() -> None:
    global _columns
    with _lock:
        _columns = None


def _check_col(col: str) -> str:
    if col not in _episode_columns():
        raise ValueError(
            f"unknown episode column {col!r}; valid columns: "
            f"{sorted(_episode_columns())}"
        )
    return col


# ---------------------------------------------------------------------------
# Lambda -> SQL translation (for the common, SQL-pushable shapes)
# ---------------------------------------------------------------------------
def _consts(node: ast.AST) -> list:
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        out = []
        for el in node.elts:
            if not isinstance(el, ast.Constant):
                raise ValueError("non-constant in container")
            out.append(el.value)
        return out
    raise ValueError("not a container literal")


def _col_of(node: ast.AST) -> Optional[str]:
    # row['col']
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _translate_lambda(expr: str, params: dict) -> Optional[str]:
    """Return a SQL WHERE fragment for a simple ``row[...] <op> const/container``
    lambda (mutating ``params`` with bind values), or None if not translatable."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Lambda):
        return None
    body = tree.body.body
    if not isinstance(body, ast.Compare) or len(body.ops) != 1:
        return None
    col = _col_of(body.left)
    if col is None:
        return None
    _check_col(col)
    op, comp = body.ops[0], body.comparators[0]

    def _bind(value) -> str:
        key = f"p{len(params)}"
        params[key] = value
        return f":{key}"

    if isinstance(op, (ast.In, ast.NotIn)):
        try:
            values = _consts(comp)
        except ValueError:
            return None
        kw = "NOT IN" if isinstance(op, ast.NotIn) else "IN"
        if not values:
            return "1=1" if isinstance(op, ast.NotIn) else "1=0"
        placeholders = ", ".join(_bind(v) for v in values)
        return f'"{col}" {kw} ({placeholders})'
    if type(op) in _CMP_OPS and isinstance(comp, ast.Constant):
        return f'"{col}" {_CMP_OPS[type(op)]} {_bind(comp.value)}'
    return None


def _structured_where(where: Mapping[str, Any], params: dict) -> list[str]:
    """Compile a structured ``where`` dict {col: value | [values]} to SQL."""
    conds = []
    for col, val in where.items():
        _check_col(col)

        def _bind(value) -> str:
            key = f"p{len(params)}"
            params[key] = value
            return f":{key}"

        if isinstance(val, (list, tuple, set)):
            vals = list(val)
            if not vals:
                conds.append("1=0")
            else:
                conds.append(f'"{col}" IN ({", ".join(_bind(v) for v in vals)})')
        else:
            conds.append(f'"{col}" = {_bind(val)}')
    return conds


# ---------------------------------------------------------------------------
# Config -> filter groups
# ---------------------------------------------------------------------------
def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    for base in (Path.cwd(), REPO_ROOT, BASE_DIR):
        cand = base / path_str
        if cand.exists():
            return cand
    raise FileNotFoundError(f"data_config not found: {path_str}")


def _extract_lambda_groups_from_data_config(path: Path) -> list[list[str]]:
    """Pull every ``filters.filter_lambdas`` out of a model data config's
    ``*_datasets`` sections (each list is an AND-group; OR across datasets)."""
    cfg = yaml.safe_load(path.read_text()) or {}
    groups: list[list[str]] = []
    for key, val in cfg.items():
        if key.endswith("_datasets") and isinstance(val, dict):
            for ds in val.values():
                lambdas = ((ds or {}).get("filters") or {}).get("filter_lambdas")
                if lambdas:
                    groups.append(list(lambdas))
    return groups


def _lambda_groups(dataset_cfg: dict) -> list[list[str]]:
    groups: list[list[str]] = []
    if dataset_cfg.get("data_config"):
        groups.extend(
            _extract_lambda_groups_from_data_config(
                _resolve_path(dataset_cfg["data_config"])
            )
        )
    inline = (dataset_cfg.get("filters") or {}).get("filter_lambdas")
    if inline:
        groups.append(list(inline))
    return groups


# ---------------------------------------------------------------------------
# Span building
# ---------------------------------------------------------------------------
def _row_to_span(row: Mapping[str, Any], fps: float) -> dict:
    try:
        n_frames = int(row.get("num_frames") or 0)
    except (TypeError, ValueError):
        n_frames = 0
    end = (n_frames / fps) if (n_frames > 0 and fps > 0) else 0.0
    return {
        "id": str(row["episode_hash"]),
        "video": str(row["episode_hash"]),  # -> presigned R2 via s3_resolver
        "start": 0.0,
        "end": float(end),
        "scene": str(row.get("scene") or "unknown"),
        "operator": str(row.get("operator") or "unknown"),
        "label": str(row.get("task_description") or row.get("task") or ""),
    }


def _query_rows(sql: str, params: dict) -> list[dict]:
    from sqlalchemy import text

    with _get_engine().connect() as conn:
        return [dict(m) for m in conn.execute(text(sql), params).mappings().all()]


# ---------------------------------------------------------------------------
# Public: resolve config -> spans
# ---------------------------------------------------------------------------
def resolve_spans(dataset_cfg: dict) -> list[dict]:
    fps = float(dataset_cfg.get("fps") or DEFAULT_FPS)
    limit = dataset_cfg.get("limit")
    do_shuffle = bool(dataset_cfg.get("shuffle", False))

    # Global constraints AND-ed into every group.
    base_conds: list[str] = []
    base_params: dict = {}
    where = dataset_cfg.get("where") or (dataset_cfg.get("filters") or {}).get("where")
    if where:
        base_conds.extend(_structured_where(where, base_params))
    if dataset_cfg.get("sql_where"):
        base_conds.append(f"({dataset_cfg['sql_where']})")

    # Explicit episode-hash allowlist (e.g. the split-viz fold_clothes_all.json —
    # `allowed_episode_ids` in those training configs). Bounds the query directly.
    allow = list(dataset_cfg.get("episode_ids") or [])
    if dataset_cfg.get("episode_ids_file"):
        import json

        allow.extend(
            json.loads(_resolve_path(dataset_cfg["episode_ids_file"]).read_text())
        )
    if allow:
        base_conds.extend(
            _structured_where({"episode_hash": list(dict.fromkeys(allow))}, base_params)
        )

    groups = _lambda_groups(dataset_cfg)
    if not groups:
        if base_conds:
            groups = [[]]  # structured/sql_where only
        elif limit:
            groups = [[]]  # bare limit -> arbitrary N episodes
        else:
            raise ValueError(
                "dataset has no filters and no limit — refusing to load every "
                "episode (200k+). Add filters.filter_lambdas / where / data_config, "
                "or a limit."
            )

    matched: dict[str, dict] = {}
    for group in groups:
        params = dict(base_params)
        sql_conds = list(base_conds)
        py_lambdas: list[str] = []
        for expr in group:
            frag = _translate_lambda(expr, params)
            (sql_conds if frag else py_lambdas).append(frag or expr)
        if not sql_conds and not limit:
            raise ValueError(
                f"filter group {group!r} can't be pushed to SQL and there's no "
                "limit — add an SQL-pushable predicate (row['col'] in [...] / "
                "== / comparison), a `where`, `sql_where`, or a `limit`."
            )

        sql = "SELECT * FROM app.episodes WHERE NOT is_deleted"
        if sql_conds:
            sql += " AND " + " AND ".join(f"({c})" for c in sql_conds)
        bind = dict(params)
        if not py_lambdas and limit:
            # Pure-SQL group: push the sample/limit down to Postgres.
            if do_shuffle:
                sql += " ORDER BY random()"
            sql += " LIMIT :_lim"
            bind["_lim"] = int(limit)

        rows = _query_rows(sql, bind)
        if py_lambdas:
            # Refine the bounded subset in Python (full group, incl. df-level).
            filt = DatasetFilter(group)
            import pandas as pd

            df = filt.filter_df(pd.DataFrame(rows)) if rows else pd.DataFrame(rows)
            rows = [r for r in df.to_dict("records") if filt.matches(r)]

        for row in rows:
            eh = row.get("episode_hash")
            if eh and str(eh) not in matched:
                matched[str(eh)] = _row_to_span(row, fps)

    spans = list(matched.values())
    if do_shuffle:
        random.shuffle(spans)
    if limit:
        spans = spans[: int(limit)]
    return spans
