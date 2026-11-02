"""Read-only access to the LEGACY episode DB (AWS RDS), used only to enrich
spans with subtask annotations (`segments`) the new DB doesn't carry.

The new DB (``DATABASE_URL``, DigitalOcean) drives episode selection. The legacy
DB (an older AWS RDS reached via an AWS Secrets Manager ARN) still has a
per-episode ``segments`` column — a list of ``{label, start_seconds,
end_seconds}`` subtask annotations. We look those up by ``episode_hash`` at
span-resolve time (batched, best-effort) and cache the resulting SQLite column,
so serving never needs a live legacy connection.

Credentials come from a SEPARATE env file (default ``~/.egoverse_env_old``):
``SECRETS_ARN`` (+ ``AWS_DEFAULT_REGION`` and, on hosts without a shared AWS
creds file, ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
``AWS_SESSION_TOKEN``). On Modal, provide these via a secret (see modal_app.py).
Everything here is best-effort: any failure logs once and yields no segments,
so the app still runs when the legacy DB is unreachable.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

LEGACY_ENV_PATH = os.environ.get("EGO_RATING_LEGACY_ENV", "~/.egoverse_env_old")
DEFAULT_REGION = "us-east-2"

_engine = None
_engine_tried = False
_lock = threading.Lock()


def _parse_env_file(path: str) -> dict[str, str]:
    """Quote-stripping .env parser that returns a dict WITHOUT touching
    os.environ (so it can't collide with the new DB's credentials)."""
    out: dict[str, str] = {}
    p = Path(path).expanduser()
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        out[k] = v
    return out


def _build_engine():
    """Build the legacy RDS engine from the isolated env (Secrets Manager ARN
    or direct PG_* / LEGACY_DATABASE_URL). Returns None if unconfigured."""
    import json

    import boto3
    from sqlalchemy import URL, create_engine

    env = _parse_env_file(LEGACY_ENV_PATH)
    # Env vars already in the process win (e.g. a Modal secret), then the file.
    get = lambda k, d=None: os.environ.get(k) or env.get(k, d)  # noqa: E731

    direct = get("LEGACY_DATABASE_URL")
    if direct:
        direct = direct.replace("postgresql://", "postgresql+psycopg://", 1).replace(
            "postgres://", "postgresql+psycopg://", 1
        )
        return create_engine(direct, pool_pre_ping=True)

    secrets_arn = get("SECRETS_ARN")
    # Region: prefer the ARN's own region (arn:aws:secretsmanager:<region>:...),
    # then the legacy env FILE, then default. NOT os.environ's
    # AWS_DEFAULT_REGION — that's often R2's "auto" sentinel, which is invalid
    # for AWS Secrets Manager.
    region = None
    if (
        secrets_arn
        and secrets_arn.startswith("arn:aws:")
        and len(secrets_arn.split(":")) > 3
    ):
        region = secrets_arn.split(":")[3] or None
    region = region or env.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    if secrets_arn:
        session = boto3.session.Session(
            aws_access_key_id=get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=get("AWS_SESSION_TOKEN"),
            region_name=region,
        )
        sec = json.loads(
            session.client("secretsmanager").get_secret_value(SecretId=secrets_arn)[
                "SecretString"
            ]
        )
        host = sec.get("host", sec.get("HOST"))
        user = sec.get("username", sec.get("user", sec.get("USER")))
        password = sec.get("password", sec.get("PASSWORD"))
        dbname = sec.get("dbname", sec.get("DBNAME", "appdb"))
        port = int(sec.get("port", 5432))
    elif get("PG_HOST_LEGACY"):
        host, user = get("PG_HOST_LEGACY"), get("PG_USER_LEGACY")
        password = get("PG_PASSWORD_LEGACY")
        dbname = get("PG_DATABASE_LEGACY", "appdb")
        port = int(get("PG_PORT_LEGACY", "5432"))
    else:
        return None

    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
            query={"sslmode": "require"},
        ),
        pool_pre_ping=True,
    )


def _get_engine():
    """Cached legacy engine, or None if unconfigured/unreachable (logged once)."""
    global _engine, _engine_tried
    if _engine is not None or _engine_tried:
        return _engine
    with _lock:
        if _engine is not None or _engine_tried:
            return _engine
        _engine_tried = True
        try:
            _engine = _build_engine()
            if _engine is None:
                print(
                    "[ego-rating] legacy DB not configured; subtask "
                    "annotations disabled."
                )
        except Exception as e:
            print(
                f"[ego-rating] legacy DB unavailable; subtask annotations "
                f"disabled: {e}"
            )
            _engine = None
    return _engine


def _normalize_segments(raw: Any) -> list[dict]:
    """Coerce a raw `segments` value into a clean list of
    {label, start_seconds, end_seconds}. Tolerates str-encoded JSON and
    missing/renamed keys; drops anything unparseable."""
    import json

    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        label = seg.get("label") or seg.get("name") or seg.get("text") or ""
        start = seg.get("start_seconds", seg.get("start", seg.get("start_time")))
        end = seg.get("end_seconds", seg.get("end", seg.get("end_time")))

        def _num(x):
            try:
                return round(float(x), 2)
            except (TypeError, ValueError):
                return None

        out.append(
            {
                "label": str(label),
                "start_seconds": _num(start),
                "end_seconds": _num(end),
            }
        )
    return out


def fetch_segments(episode_hashes: list[str]) -> dict[str, list[dict]]:
    """Batch-fetch subtask segments for episode_hashes from the legacy DB.
    Returns {episode_hash: [segment, ...]}; empty dict on any failure (the app
    degrades to no subtask annotations rather than erroring)."""
    hashes = [h for h in dict.fromkeys(episode_hashes) if h]
    if not hashes:
        return {}
    eng = _get_engine()
    if eng is None:
        return {}
    try:
        from sqlalchemy import text

        out: dict[str, list[dict]] = {}
        # Chunk to keep the ANY(:h) bind array reasonable.
        with eng.connect() as c:
            for i in range(0, len(hashes), 1000):
                chunk = hashes[i : i + 1000]
                rows = c.execute(
                    text(
                        "SELECT episode_hash, segments FROM app.episodes "
                        "WHERE episode_hash = ANY(:h)"
                    ),
                    {"h": chunk},
                ).fetchall()
                for r in rows:
                    segs = _normalize_segments(r[1])
                    if segs:
                        out[str(r[0])] = segs
        return out
    except Exception as e:
        print(f"[ego-rating] legacy segments fetch failed: {e}")
        return {}


def fetch_one(episode_hash: str) -> list[dict]:
    return fetch_segments([episode_hash]).get(episode_hash, [])


# ---------------------------------------------------------------------------
# Bundled cache (backend/segments_cache.json) — generated by
# scripts/dump_segments.py so deploys without legacy-DB creds (e.g. Modal)
# still attach subtask annotations.
# ---------------------------------------------------------------------------
CACHE_PATH = Path(__file__).resolve().parent / "segments_cache.json"
_cache: Optional[dict[str, list[dict]]] = None


def load_cache() -> dict[str, list[dict]]:
    """{episode_hash: segments} from the bundled cache file (empty if absent)."""
    global _cache
    if _cache is not None:
        return _cache
    import json

    try:
        raw = json.loads(CACHE_PATH.read_text())
        _cache = {str(k): _normalize_segments(v) for k, v in raw.items()}
    except Exception:
        _cache = {}
    return _cache
