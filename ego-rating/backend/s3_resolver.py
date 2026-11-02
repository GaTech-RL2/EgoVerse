"""Resolve a span's ``video`` reference to a streamable URL — without ever
downloading the whole video to this server.

A span's ``video`` field may be any of (checked in this order):

  * ``http://`` / ``https://`` URL        -> used as-is (passthrough)
  * ``r2://bucket/key`` or ``s3://...``     -> short-lived presigned R2 GET URL
  * a 24-hex ``episode_hash``               -> MongoDB ``mecka-ai.episodes._id`` ->
                                               its ``video_1`` storage key -> presigned URL
  * a bare R2 key (``a/b/clip.mp4``)        -> presigned against the default bucket
  * a local file under ``videos/``          -> ``None`` (served by the static /videos mount)

The browser fetches bytes directly from R2 via the presigned URL. R2 honors HTTP
Range, so seeking the ``#t=start,end`` media fragment works and only the watched
bytes are transferred — no full download, server- or client-side.

Credentials & connection mirror the main repo so the same ``~/.egoverse_env`` /
env vars work here:
  * R2:    egomimic/utils/aws/aws_data_utils.py::get_boto3_s3_client and
           modal_mecka_to_zarr.py::_get_r2_client  (R2_ENDPOINT_URL, R2_ACCESS_KEY_ID,
           R2_SECRET_ACCESS_KEY, or AWS_* / R2_ACCOUNT_ID fallbacks)
  * Mongo: modal_mecka_to_zarr.py::_get_mongo_db   (MONGODB_URI, db "mecka-ai")

boto3/pymongo are imported lazily so the app still runs (for local clips) when
they're absent or no credentials are configured.
"""

from __future__ import annotations

import mimetypes
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

_EPISODE_HASH_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# Defaults mirror the repo (legacy bucket "rldb"); override via R2_BUCKET / BUCKET.
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "mecka-ai")
MONGO_EPISODES_COLLECTION = os.environ.get("MONGO_EPISODES_COLLECTION", "episodes")
MONGO_VIDEO_FIELD = os.environ.get("MONGO_VIDEO_FIELD", "video_1")
PRESIGN_EXPIRY = int(os.environ.get("PRESIGN_EXPIRY", "3600"))  # seconds

_env_loaded = False
_s3_client = None
_mongo_db = None
_init_lock = threading.Lock()
# ref -> (url, expires_at_epoch)
_url_cache: dict[str, tuple[str, float]] = {}


# ---------------------------------------------------------------------------
# Lazy clients
# ---------------------------------------------------------------------------
def load_env(path: str = "~/.egoverse_env") -> None:
    """Populate os.environ from ~/.egoverse_env (matches the repo's load_env)."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    p = Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        # Strip one pair of surrounding matching quotes (.env convention) —
        # e.g. MONGODB_URI='mongodb+srv://...'.
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        os.environ.setdefault(k, v)


def _get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    with _init_lock:
        if _s3_client is not None:
            return _s3_client
        load_env()
        import boto3  # lazy

        endpoint_url = (
            os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("R2_ENDPOINT_URL")
            or os.environ.get("R2_ENDPOINT")
        )
        if not endpoint_url:
            account_id = os.environ.get("R2_ACCOUNT_ID")
            if account_id:
                endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        access_key = (
            os.environ.get("R2_ACCESS_KEY_ID")
            or os.environ.get("R2_ACCESS_KEY")
            or os.environ.get("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            os.environ.get("R2_SECRET_ACCESS_KEY")
            or os.environ.get("R2_SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        if not (endpoint_url and access_key and secret_key):
            raise RuntimeError(
                "R2/S3 credentials not configured. Set R2_ENDPOINT_URL, "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (or put them in "
                "~/.egoverse_env). Only needed for remote video references."
            )
        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=boto3.session.Config(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
        )
    return _s3_client


def _get_mongo_db():
    global _mongo_db
    if _mongo_db is not None:
        return _mongo_db
    with _init_lock:
        if _mongo_db is not None:
            return _mongo_db
        load_env()
        from pymongo import MongoClient  # lazy

        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MONGODB_URI not set; cannot resolve episode_hash video "
                "references. Set it (or use r2://key / https:// refs instead)."
            )
        _mongo_db = MongoClient(uri)[MONGO_DB_NAME]
    return _mongo_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_bucket() -> str:
    load_env()
    return os.environ.get("R2_BUCKET") or os.environ.get("BUCKET") or "rldb"


def _parse_storage_key(storage_key: str) -> tuple[str, str]:
    """``r2://bucket/key`` / ``s3://bucket/key`` -> (bucket, key); bare -> default
    bucket. Mirrors modal_mecka_to_zarr.py::_parse_storage_key."""
    for scheme in ("r2://", "s3://"):
        if storage_key.startswith(scheme):
            bucket, _, key = storage_key[len(scheme) :].partition("/")
            return bucket, key
    return _default_bucket(), storage_key.lstrip("/")


def _nested_get(doc: dict, dotted: str):
    """Read a dotted path (e.g. ``a.b.c``) out of a Mongo document."""
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _presign(bucket: str, key: str) -> str:
    # Force a correct Content-Type on the response — R2 objects often have none
    # stored, and Safari is strict about it for <video> playback.
    ctype, _ = mimetypes.guess_type(key)
    return _get_s3_client().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentType": ctype or "video/mp4",
        },
        ExpiresIn=PRESIGN_EXPIRY,
    )


def _local_path(video_ref: str) -> Optional[Path]:
    """The local file for a ref, or None if it's a URL / episode_hash / missing."""
    if _SCHEME_RE.match(video_ref) or _EPISODE_HASH_RE.match(video_ref):
        return None
    cand = Path(video_ref) if os.path.isabs(video_ref) else (BASE_DIR / video_ref)
    return cand if cand.is_file() else None


def is_local(video_ref: str) -> bool:
    return _local_path(video_ref) is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_to_url(video_ref: str) -> Optional[str]:
    """Return a streamable URL for a remote ref, or ``None`` if it's a local file
    (the caller serves those via the static /videos mount). Presigned URLs are
    cached until ~5 min before expiry, then re-signed on demand.
    """
    if not video_ref:
        return None
    if _local_path(video_ref) is not None:
        return None  # local — caller uses the static mount

    now = time.time()
    cached = _url_cache.get(video_ref)
    if cached and cached[1] - now > 300:
        return cached[0]

    url = _resolve_remote(video_ref)
    _url_cache[video_ref] = (url, now + PRESIGN_EXPIRY)
    return url


def _resolve_remote(video_ref: str) -> str:
    if video_ref.startswith(("http://", "https://")):
        return video_ref
    if video_ref.startswith(("r2://", "s3://")):
        return _presign(*_parse_storage_key(video_ref))
    if _EPISODE_HASH_RE.match(video_ref):
        from bson import ObjectId  # lazy (ships with pymongo)

        doc = _get_mongo_db()[MONGO_EPISODES_COLLECTION].find_one(
            {"_id": ObjectId(video_ref)}
        )
        if not doc:
            raise KeyError(f"episode not found in MongoDB: {video_ref}")
        storage_key = _nested_get(doc, MONGO_VIDEO_FIELD)
        if not storage_key:
            raise KeyError(
                f"episode {video_ref} has no '{MONGO_VIDEO_FIELD}' video field"
            )
        return _presign(*_parse_storage_key(storage_key))
    # Treat anything else as a bare R2 key under the default bucket.
    return _presign(*_parse_storage_key(video_ref))
