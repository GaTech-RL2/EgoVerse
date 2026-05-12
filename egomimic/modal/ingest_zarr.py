"""Download zarr episodes from Cloudflare R2 into the Modal volume.

Looks up each episode hash in the legacy app.episodes SQL table to find its
zarr_processed_path, then downloads that zarr store from R2 (bucket: rldb)
into the Modal volume directly at /mnt/zarr-data/<episode_hash>/.

Prerequisites
-------------
AWS credentials must be configured for the EgoVerse AWS account:

    aws configure
    # AccessKeyId:     AKIAYDKH4BNCAYHE5NG2
    # SecretAccessKey: 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
    # Default region:  us-east-2

R2 credentials must be in ~/.egoverse_env.  If not set, run:

    ./egomimic/utils/aws/setup_secret.sh

Usage
-----
    modal run --env robotics egomimic/modal/ingest_zarr.py -- \\
        692eb32bb05428c72fb37657 69b242e50e15ec767bedb8b8

Notes
-----
- SQL lookup happens locally (submitting machine); only the download runs remotely.
- Each episode downloads in its own parallel Modal container.
- Downloads are idempotent: s5cmd skips files already present on the volume.
- This is the ONLY script that should query the legacy AWS RDS database.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Standalone Modal app — self-contained, no egomimic imports at module level
# ---------------------------------------------------------------------------

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

_image = (
    modal.Image.debian_slim()
    .pip_install("s5cmd", "sqlalchemy", "psycopg[binary]", "boto3")
)

_volume = modal.Volume.from_name("egoverse-zarr-data")
_app = modal.App("egomimic-ingest-zarr", image=_image)

_VOLUME_MOUNT = "/mnt/zarr-data"

# ---------------------------------------------------------------------------
# Local helpers (run on the submitting machine)
# ---------------------------------------------------------------------------

_DB_SECRET_NAMES = ["rds/appdb/appuser", "rds/appdb/appuser-readonly"]
_DB_REGION = "us-east-2"


def _load_r2_creds() -> dict[str, str]:
    """Read old rldb R2 credentials from ~/.egoverse_env_old."""
    env_file = Path("~/.egoverse_env_old").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))

    missing, creds = [], {}
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        val = os.environ.get(key, "")
        if not val:
            missing.append(key)
        else:
            creds[key] = val
    if missing:
        raise RuntimeError(
            f"Missing env vars: {missing}. "
            "Run ./egomimic/utils/aws/setup_secret.sh first."
        )
    # Session token is optional (present for temporary/STS credentials)
    session_token = os.environ.get("R2_SESSION_TOKEN", "")
    if session_token:
        creds["R2_SESSION_TOKEN"] = session_token
    return creds


def _old_sql_engine():
    """Connect to the legacy AWS RDS via Secrets Manager (never touches new DB)."""
    import json
    import boto3
    from sqlalchemy import create_engine, URL

    client = boto3.client("secretsmanager", region_name=_DB_REGION)
    sec = None
    for name in _DB_SECRET_NAMES:
        try:
            sec = json.loads(client.get_secret_value(SecretId=name)["SecretString"])
            break
        except Exception:
            continue

    if sec is None:
        raise RuntimeError(
            f"Could not retrieve DB secret from AWS Secrets Manager "
            f"(tried: {_DB_SECRET_NAMES}). "
            "Run `aws configure` with the EgoVerse AWS keys first."
        )

    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=sec.get("username", sec.get("user")),
            password=sec.get("password"),
            host=sec.get("host"),
            port=sec.get("port", 5432),
            database=sec.get("dbname", "appdb"),
            query={"sslmode": "require"},
        ),
        pool_pre_ping=True,
    )


def _lookup_zarr_paths(episode_hashes: list[str]) -> dict[str, str]:
    """Return {episode_hash: zarr_processed_path} from the legacy SQL table."""
    from sqlalchemy import MetaData, Table, select

    engine = _old_sql_engine()
    md = MetaData()
    tbl = Table("episodes", md, autoload_with=engine, schema="app")

    with engine.connect() as conn:
        rows = conn.execute(
            select(tbl.c.episode_hash, tbl.c.zarr_processed_path).where(
                tbl.c.episode_hash.in_(episode_hashes)
            )
        ).fetchall()

    found = {r.episode_hash: r.zarr_processed_path for r in rows}
    result, skipped = {}, []

    for ep_hash in episode_hashes:
        path = found.get(ep_hash, "")
        if not path:
            label = "not found" if ep_hash not in found else "zarr_processed_path is empty"
            print(f"  [WARN] {ep_hash}: {label} — skipping")
            skipped.append(ep_hash)
        else:
            result[ep_hash] = path

    if skipped:
        print(f"\n{len(skipped)} episode(s) skipped.")
    return result


# ---------------------------------------------------------------------------
# Container function
# ---------------------------------------------------------------------------


@_app.function(
    volumes={_VOLUME_MOUNT: _volume},
    timeout=3600,
    cpu=4.0,
    memory=8192,
)
def _download_episode(
    episode_hash: str,
    s3_path: str,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    session_token: str = "",
) -> str:
    import os
    import subprocess

    # Place the zarr store directly under the volume root as <episode_hash>/
    dest = f"{_VOLUME_MOUNT}/{episode_hash}"
    os.makedirs(dest, exist_ok=True)

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
    # R2 does not support STS session tokens — explicitly unset to avoid 400s
    env.pop("AWS_SESSION_TOKEN", None)
    env.pop("AWS_SECURITY_TOKEN", None)

    # s5cmd cp with a wildcard downloads all objects under the prefix,
    # including nested chunks and metadata (S3 key space is flat).
    src = s3_path.rstrip("/") + "/*"
    dst = dest.rstrip("/") + "/"

    print(f"[{episode_hash}] {src} → {dst}")
    result = subprocess.run(
        ["s5cmd", "--endpoint-url", endpoint_url, "cp", src, dst],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"[{episode_hash}] s5cmd exited {result.returncode}")

    _volume.commit()
    print(f"[{episode_hash}] committed to volume")
    return dest


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@_app.local_entrypoint()
def ingest(*episode_hashes: str) -> None:
    """Look up episodes in the legacy SQL table and download their zarr stores.

    Example:
        modal run --env robotics egomimic/modal/ingest_zarr.py -- \\
            692eb32bb05428c72fb37657 69b242e50e15ec767bedb8b8
    """
    if not episode_hashes:
        raise SystemExit(
            "Provide at least one episode hash.\n"
            "  modal run --env robotics egomimic/modal/ingest_zarr.py -- <hash1> [hash2 ...]"
        )

    print(f"Looking up {len(episode_hashes)} episode(s) in app.episodes...")
    zarr_paths = _lookup_zarr_paths(list(episode_hashes))

    if not zarr_paths:
        raise SystemExit("No valid zarr paths found — nothing to download.")

    creds = _load_r2_creds()
    hashes = list(zarr_paths.keys())
    paths = list(zarr_paths.values())
    n = len(hashes)
    session_token = creds.get("R2_SESSION_TOKEN", "")

    print(f"\nSubmitting {n} download(s) in parallel...")
    for ep_hash, dest in zip(
        hashes,
        _download_episode.map(
            hashes,
            paths,
            [creds["R2_ENDPOINT_URL"]] * n,
            [creds["R2_ACCESS_KEY_ID"]] * n,
            [creds["R2_SECRET_ACCESS_KEY"]] * n,
            [session_token] * n,
        ),
    ):
        print(f"  ✓  {ep_hash} → {dest}")

    print("\nAll downloads complete.")
