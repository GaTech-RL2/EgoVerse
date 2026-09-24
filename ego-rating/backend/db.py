"""SQLite init + config-driven span upsert.

Startup behavior (see ``init_app``):
  1. Create tables if they don't exist (migrating a pre-axes DB forward).
  2. Read ``config.yaml``.
  3. Upsert every span from config into the ``spans`` table (insert or replace
     on ``span_id``).
  4. Expose ``config["annotation"]`` / ``config["axes"]`` to all routes (via
     :func:`get_annotation` / :func:`get_axes`).

A comparison is one submitted judgement of a pair: either a skip, or one
``axis_ratings`` row per configured axis (outcome a / b / equal). The schema is
intentionally plain so it ports cleanly to Postgres later.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # the ego-rating/ root
CONFIG_PATH = BASE_DIR / "config.yaml"
# EGO_RATING_DATA_DIR relocates the SQLite DB (e.g. onto a Modal Volume so a
# cloud deployment persists across container restarts).
DATA_DIR = Path(os.environ.get("EGO_RATING_DATA_DIR") or (BASE_DIR / "data"))
DB_PATH = DATA_DIR / "ego_rating.db"

# ---------------------------------------------------------------------------
# Schema (kept verbatim from the spec; types chosen to map cleanly to Postgres)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  span_id   TEXT PRIMARY KEY,
  video_uri TEXT NOT NULL,
  start     REAL NOT NULL,
  end       REAL NOT NULL,
  scene     TEXT NOT NULL,
  operator  TEXT NOT NULL,
  label     TEXT DEFAULT '',        -- per-episode task/description, shown as rating context
  segments  TEXT DEFAULT '[]'       -- subtask annotations (JSON list of {label,start_seconds,end_seconds}) from the legacy DB
);

CREATE TABLE IF NOT EXISTS raters (
  rater_id INTEGER PRIMARY KEY,
  name     TEXT
);

-- Pairwise comparisons: one row per submitted judgement of a pair (a full
-- rating over all axes, or a skip). Glicko ratings are DERIVED from this log +
-- axis_ratings (replayed in ts order), never stored. skipped=1 records that
-- the rater saw the pair but declined to judge (ignored by Glicko, not
-- re-served).
CREATE TABLE IF NOT EXISTS comparisons (
  comparison_id INTEGER PRIMARY KEY,
  span_a        TEXT    REFERENCES spans(span_id),
  span_b        TEXT    REFERENCES spans(span_id),
  rater_id      INTEGER REFERENCES raters(rater_id),
  skipped       INTEGER NOT NULL DEFAULT 0,
  ts            DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Per-axis outcome of a comparison. 'a'/'b' are decisive; 'equal' is a
-- Glicko-2 draw (score 0.5 for both).
CREATE TABLE IF NOT EXISTS axis_ratings (
  axis_rating_id INTEGER PRIMARY KEY,
  comparison_id  INTEGER NOT NULL REFERENCES comparisons(comparison_id) ON DELETE CASCADE,
  axis           TEXT    NOT NULL,
  outcome        TEXT    NOT NULL CHECK(outcome IN ('a', 'b', 'equal'))
);

-- Per-axis weight for the TOTAL score (weighted mean of the 1-10 axis scores).
-- Adjusted live from the leaderboard UI; config.yaml only seeds a new axis's
-- weight, a reload never clobbers a user-adjusted value.
CREATE TABLE IF NOT EXISTS axis_weights (
  axis   TEXT PRIMARY KEY,
  weight REAL NOT NULL DEFAULT 1.0
);

-- Small key/value store (currently: `selection_key`, the fingerprint of the
-- config's dataset/spans block that the persisted span pool was resolved from).
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE INDEX IF NOT EXISTS idx_comparisons_rater ON comparisons(rater_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_pair  ON comparisons(span_a, span_b);
CREATE INDEX IF NOT EXISTS idx_axis_ratings_cmp  ON axis_ratings(comparison_id);
CREATE INDEX IF NOT EXISTS idx_axis_ratings_axis ON axis_ratings(axis);
"""

# Module-level annotation + axes, refreshed by ``load_and_upsert_spans``.
_ANNOTATION: str = ""
_AXES: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------
def connect() -> sqlite3.Connection:
    """Open a connection with row access by name and FK enforcement on.

    ``check_same_thread=False`` because FastAPI runs sync endpoints in a
    threadpool: a request's connection is opened, used, and closed across
    possibly-different worker threads. Safe here — each request gets its own
    fresh connection and never shares it concurrently.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Multiple raters hit the API concurrently: WAL lets readers proceed during
    # a write, busy_timeout makes a second writer wait instead of erroring.
    # WAL needs shared-memory support — on filesystems without it (e.g. WSL's
    # /mnt/c drvfs) SQLite just keeps the default journal mode, so best effort.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def get_db() -> Iterable[sqlite3.Connection]:
    """FastAPI dependency: yields a connection and always closes it."""
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def read_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}
    if "annotation" not in config:
        raise ValueError("config.yaml must define a top-level 'annotation' string")
    config["axes"] = _normalize_axes(config.get("axes"))
    if not config.get("dataset") and not config.get("spans"):
        raise ValueError(
            "config.yaml must define either 'dataset' (filter-based episode "
            "selection) or 'spans' (an explicit list)"
        )
    # Validate explicit spans up front (the dataset path validates on resolve).
    seen: set[str] = set()
    for span in config.get("spans") or []:
        sid = span.get("id")
        if not sid:
            raise ValueError(f"every span needs an 'id': {span!r}")
        if sid in seen:
            raise ValueError(f"duplicate span id in config: {sid!r}")
        seen.add(sid)
    return config


def _normalize_axes(axes: Any) -> list[dict[str, str]]:
    """Validate the fixed axis set: a non-empty list of names (or {name,
    description} dicts), unique, no reserved '_'-prefixed names."""
    if not axes or not isinstance(axes, list):
        raise ValueError(
            "config.yaml must define a top-level 'axes' list — the fixed set of "
            "rating axes, e.g. axes: [{name: task_completion, description: ...}]"
        )
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ax in axes:
        if isinstance(ax, str):
            ax = {"name": ax}
        name = str((ax or {}).get("name") or "").strip()
        if not name:
            raise ValueError(f"every axis needs a 'name': {ax!r}")
        if name.startswith("_"):
            raise ValueError(f"axis names starting with '_' are reserved: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate axis name in config: {name!r}")
        seen.add(name)
        try:
            weight = float(ax.get("weight", 1.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"axis {name!r} has a non-numeric weight: {ax.get('weight')!r}"
            )
        if weight < 0:
            raise ValueError(f"axis {name!r} has a negative weight: {weight}")
        out.append(
            {
                "name": name,
                "description": str(ax.get("description") or ""),
                "weight": weight,
            }
        )
    return out


def upsert_spans(conn: sqlite3.Connection, spans: list[dict[str, Any]]) -> int:
    """Insert-or-replace each config span keyed on span_id. Returns count."""
    import json

    rows = [
        (
            s["id"],
            s["video"],
            float(s["start"]),
            float(s["end"]),
            s["scene"],
            s["operator"],
            s.get("label", ""),
            json.dumps(s.get("segments") or []),
        )
        for s in spans
    ]
    conn.executemany(
        """
        INSERT INTO spans (span_id, video_uri, start, end, scene, operator, label, segments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(span_id) DO UPDATE SET
          video_uri = excluded.video_uri,
          start     = excluded.start,
          end       = excluded.end,
          scene     = excluded.scene,
          operator  = excluded.operator,
          label     = excluded.label,
          segments  = excluded.segments
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _enrich_segments(spans: list[dict[str, Any]]) -> None:
    """Attach subtask annotations (`segments`) to each span in place, keyed by
    episode_hash (the span id). Precedence: segments already on the span
    (offline config) > the live legacy DB > the bundled cache file
    (segments_cache.json, for deploys without legacy creds). Best-effort:
    spans keep an empty list if no source has them."""
    from backend import legacy_db

    need = [s["id"] for s in spans if not s.get("segments")]
    if not need:
        return
    by_hash = legacy_db.fetch_segments(need)  # live legacy DB (may be {})
    cache = legacy_db.load_cache()  # bundled fallback
    n = 0
    for s in spans:
        if s.get("segments"):
            continue
        segs = by_hash.get(s["id"]) or cache.get(s["id"])
        if segs:
            s["segments"] = segs
            n += 1
    src = "legacy DB" if by_hash else ("cache" if cache else "no source")
    print(
        f"[ego-rating] attached subtask annotations to {n}/{len(spans)} spans "
        f"({src})."
    )


def _backfill_segments(conn: sqlite3.Connection, sel_key: str) -> None:
    """Populate empty `segments` on the already-pinned pool (no re-resolve, no
    data loss). Runs at most once per pinned pool: a `segments_key` meta flag
    tracks completion, but is only set when a source (live legacy DB or bundled
    cache) was actually reachable — so a transient legacy outage doesn't
    permanently disable the backfill."""
    import json

    if _get_meta(conn, "segments_key") == sel_key:
        return
    empty = [
        r["span_id"]
        for r in conn.execute(
            "SELECT span_id FROM spans WHERE segments IS NULL OR segments IN ('', '[]')"
        ).fetchall()
    ]
    if not empty:
        _set_meta(conn, "segments_key", sel_key)
        conn.commit()
        return

    from backend import legacy_db

    by_hash = legacy_db.fetch_segments(empty)
    cache = legacy_db.load_cache()
    updates = []
    for sid in empty:
        segs = by_hash.get(sid) or cache.get(sid)
        if segs:
            updates.append((json.dumps(segs), sid))
    if updates:
        conn.executemany("UPDATE spans SET segments = ? WHERE span_id = ?", updates)
    if by_hash or cache:  # a source worked -> don't retry every startup
        _set_meta(conn, "segments_key", sel_key)
    conn.commit()
    src = "legacy DB" if by_hash else ("cache" if cache else "no source")
    print(
        f"[ego-rating] backfilled subtask annotations for {len(updates)}/"
        f"{len(empty)} pinned spans ({src})."
    )


def _selection_key(config: dict[str, Any]) -> str:
    """Stable fingerprint of WHAT the config selects (the dataset/spans block
    only — annotation/axes/param edits don't invalidate the resolved pool)."""
    import hashlib
    import json

    sel = {"dataset": config.get("dataset"), "spans": config.get("spans")}
    return hashlib.sha256(
        json.dumps(sel, sort_keys=True, default=str).encode()
    ).hexdigest()


def _get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def load_and_upsert_spans(conn: sqlite3.Connection, force: bool = False) -> int:
    """Read config.yaml, refresh the module annotation/axes, sync spans.

    Spans come from either:
      * ``dataset`` — a model-data-config-style filter, resolved against
        app.episodes (one span per matching episode); or
      * ``spans`` — an explicit list (legacy / offline).

    **The resolved pool is pinned.** With a sampled selection (``shuffle`` /
    ``limit``), re-resolving on every startup would draw a *different* random
    sample — and the authoritative sync below would then delete the previous
    pool's comparisons. So the pool is only re-resolved when the config's
    selection block actually changed (fingerprint mismatch), the DB has no
    spans yet, or ``force=True`` (the explicit re-sample endpoint).
    """
    global _ANNOTATION, _AXES
    config = read_config()
    _ANNOTATION = str(config["annotation"])
    _AXES = config["axes"]
    sync_axis_weights(conn)

    sel_key = _selection_key(config)
    n_existing = conn.execute("SELECT count(*) FROM spans").fetchone()[0]
    if not force and n_existing and _get_meta(conn, "selection_key") == sel_key:
        # Unchanged selection -> keep the persisted (pinned) pool, but still
        # backfill subtask annotations onto it. This matters when the segments
        # feature (or a fresh legacy source) postdates the pool's last resolve:
        # pinning would otherwise skip enrichment forever, leaving segments '[]'.
        _backfill_segments(conn, sel_key)
        return n_existing

    if config.get("spans"):
        spans = config["spans"]
    else:
        from backend import episode_source  # lazy: only the dataset path needs DB deps

        spans = episode_source.resolve_spans(config["dataset"])

    _enrich_segments(spans)

    # Config is the source of truth for spans: drop any span (and its comparisons)
    # no longer selected, so a narrowed/changed filter doesn't leave orphans in the
    # leaderboard, dropdowns, or queue. Guard against an empty resolve (likely a
    # bad filter) wiping everything.
    new_ids = {s["id"] for s in spans}
    if new_ids:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _keep (id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM _keep")
        conn.executemany("INSERT INTO _keep (id) VALUES (?)", [(i,) for i in new_ids])
        # Explicit child-first delete (belt and braces alongside the FK cascade).
        conn.execute(
            "DELETE FROM axis_ratings WHERE comparison_id IN ("
            "  SELECT comparison_id FROM comparisons"
            "  WHERE span_a NOT IN (SELECT id FROM _keep)"
            "     OR span_b NOT IN (SELECT id FROM _keep))"
        )
        conn.execute(
            "DELETE FROM comparisons WHERE span_a NOT IN (SELECT id FROM _keep) "
            "OR span_b NOT IN (SELECT id FROM _keep)"
        )
        conn.execute("DELETE FROM spans WHERE span_id NOT IN (SELECT id FROM _keep)")
        conn.commit()

    n = upsert_spans(conn, spans)
    _set_meta(conn, "selection_key", sel_key)
    # A full resolve already wrote segments; mark the pool so the pinned-pool
    # backfill is a no-op next startup (unless the selection changes again).
    _set_meta(conn, "segments_key", sel_key)
    conn.commit()
    return n


def get_annotation() -> str:
    """The single shared instruction string for the session."""
    return _ANNOTATION


def get_axes() -> list[dict[str, str]]:
    """The fixed axis set, as [{"name", "description"}] in config order."""
    return list(_AXES)


def axis_names() -> list[str]:
    return [a["name"] for a in _AXES]


# ---------------------------------------------------------------------------
# Axis weights (for the weighted total score)
# ---------------------------------------------------------------------------
def sync_axis_weights(conn: sqlite3.Connection) -> None:
    """Seed a weight row for every configured axis (from its config `weight`,
    default 1.0) and drop rows for axes no longer configured. Existing rows are
    left alone so a config reload never clobbers a UI-adjusted weight."""
    for ax in _AXES:
        conn.execute(
            "INSERT INTO axis_weights (axis, weight) VALUES (?, ?) "
            "ON CONFLICT(axis) DO NOTHING",
            (ax["name"], ax["weight"]),
        )
    names = axis_names()
    placeholders = ", ".join("?" for _ in names) or "''"
    conn.execute(f"DELETE FROM axis_weights WHERE axis NOT IN ({placeholders})", names)
    conn.commit()


def get_weights(conn: sqlite3.Connection) -> dict[str, float]:
    """Current weight per configured axis (stored value, else config default)."""
    stored = {
        r["axis"]: r["weight"]
        for r in conn.execute("SELECT axis, weight FROM axis_weights").fetchall()
    }
    return {a["name"]: float(stored.get(a["name"], a["weight"])) for a in _AXES}


def set_weights(
    conn: sqlite3.Connection, updates: dict[str, float]
) -> dict[str, float]:
    """Upsert the given axis weights (validated by the caller); returns all."""
    conn.executemany(
        "INSERT INTO axis_weights (axis, weight) VALUES (?, ?) "
        "ON CONFLICT(axis) DO UPDATE SET weight = excluded.weight",
        [(ax, float(w)) for ax, w in updates.items()],
    )
    conn.commit()
    return get_weights(conn)


# ---------------------------------------------------------------------------
# Shared query helpers
# ---------------------------------------------------------------------------
def span_filter(scene: str | None, operator: str | None) -> tuple[str, list]:
    """Build a WHERE-clause fragment + params for spans, on scene/operator."""
    clauses, params = [], []
    if scene:
        clauses.append("scene = ?")
        params.append(scene)
    if operator:
        clauses.append("operator = ?")
        params.append(operator)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def ensure_rater(
    conn: sqlite3.Connection, rater_id: int, name: str | None = None
) -> None:
    """Create the rater row if absent (keeps the FK satisfied)."""
    conn.execute(
        "INSERT INTO raters (rater_id, name) VALUES (?, ?) "
        "ON CONFLICT(rater_id) DO NOTHING",
        (rater_id, name),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def init_db(conn: sqlite3.Connection) -> None:
    # Migration: the pre-axes schema stored a single winner ('a'/'b'/'skip') per
    # comparison. Carry those rows forward as sessions — skips become skipped=1,
    # decisive winners are preserved under the reserved axis '_legacy' (never
    # replayed into any configured axis's rating, but the pair stays "seen").
    cmp_cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(comparisons)").fetchall()
    }
    if "winner" in cmp_cols:
        conn.execute("ALTER TABLE comparisons RENAME TO comparisons_legacy")
        conn.executescript(SCHEMA)
        rows = conn.execute(
            "SELECT span_a, span_b, winner, rater_id, ts FROM comparisons_legacy "
            "ORDER BY ts, comparison_id"
        ).fetchall()
        for r in rows:
            cur = conn.execute(
                "INSERT INTO comparisons (span_a, span_b, rater_id, skipped, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    r["span_a"],
                    r["span_b"],
                    r["rater_id"],
                    1 if r["winner"] == "skip" else 0,
                    r["ts"],
                ),
            )
            if r["winner"] in ("a", "b"):
                conn.execute(
                    "INSERT INTO axis_ratings (comparison_id, axis, outcome) "
                    "VALUES (?, ?, ?)",
                    (cur.lastrowid, "_legacy", r["winner"]),
                )
        conn.execute("DROP TABLE comparisons_legacy")
        print(f"[ego-rating] migrated {len(rows)} pre-axes comparisons.")
    else:
        conn.executescript(SCHEMA)
    # Additive migrations.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(spans)").fetchall()}
    if "label" not in cols:
        conn.execute("ALTER TABLE spans ADD COLUMN label TEXT DEFAULT ''")
    if "segments" not in cols:
        conn.execute("ALTER TABLE spans ADD COLUMN segments TEXT DEFAULT '[]'")
    conn.commit()


def init_app() -> None:
    """Create tables and load config — call once at process startup."""
    conn = connect()
    try:
        init_db(conn)
        n = load_and_upsert_spans(conn)
        print(f"[ego-rating] initialized DB at {DB_PATH}; upserted {n} spans.")
    finally:
        conn.close()
