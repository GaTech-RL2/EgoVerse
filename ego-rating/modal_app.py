"""Serve ego-rating as a Modal web app (persistent URL, shared by all raters).

    cd ego-rating && modal deploy modal_app.py

Prints the URL (https://<workspace>--ego-rating-web.modal.run). Anyone with the
link can rate; each person sets their own Rater number in the header.

How it holds together:
  * The SQLite DB lives on the `ego-rating-data` Volume (mounted at /data via
    EGO_RATING_DATA_DIR), so the episode pool, comparison log, and axis weights
    persist across container restarts and redeploys. Modal commits volume
    changes in the background.
  * `max_containers=1` keeps a single writer for SQLite; concurrent raters are
    served as concurrent requests within that one container (threadpool +
    busy_timeout), same as the local server.
  * The `egoverse-*` secrets (same ones the training stack attaches) provide
    DATABASE_URL / MONGODB_URI / R2_* — episode resolve + video presign work
    exactly like a local run with ~/.egoverse_env.
  * Embeddings still go through the separately deployed `ego-rating-embed` app
    (pairing looks it up by name; deploy it once with
    `modal deploy backend/modal_embed.py`).
  * config.yaml is baked into the image — changing axes/dataset means
    redeploying (the pinned pool on the volume survives; only an actual change
    to the dataset/spans block re-resolves it).
  * Subtask annotations (`segments`) come from the LEGACY DB. On Modal they're
    fetched LIVE at pool-resolve time via the `egoverse-legacy` secret (a direct
    `LEGACY_DATABASE_URL` to the public legacy RDS — see LEGACY_DB_SECRET below),
    so no cache is needed. If that secret is absent (LEGACY_DB_SECRET = None),
    the app falls back to the bundled backend/segments_cache.json, regenerated
    locally with `python scripts/dump_segments.py`.

The URL is public-but-unguessable. To require auth, set
`requires_proxy_auth=True` on the @modal.asgi_app() decorator (raters then need
Modal-Key/Modal-Secret headers) or put it behind your own proxy.
"""

from pathlib import Path

import modal

APP_NAME = "ego-rating"
REMOTE_ROOT = "/root/ego-rating"
DATA_DIR = "/data"

data_volume = modal.Volume.from_name("ego-rating-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # backend/requirements.txt minus the local-only extras (uvicorn — Modal runs
    # the ASGI app itself; sentence-transformers/torch — embeddings run on the
    # ego-rating-embed app, with sklearn TF-IDF as the in-process fallback).
    .pip_install(
        "fastapi>=0.110",
        "httpx>=0.27",
        "pyyaml>=6.0",
        "numpy>=1.24",
        "glicko2>=2.1",
        "scikit-learn>=1.2",
        "boto3>=1.28",
        "pymongo>=4.5",
        "sqlalchemy>=2.0",
        "psycopg[binary]>=3.1",
        "pandas>=2.0",
    )
    .add_local_dir(
        Path(__file__).parent,
        remote_path=REMOTE_ROOT,
        ignore=[
            "data/**",
            "videos/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.db",
            ".venv/**",
        ],
    )
)

app = modal.App(APP_NAME)

# Base secrets (episode DB / video presign) — always required.
_SECRET_NAMES = ["egoverse-sql", "egoverse-db", "egoverse-mongodb", "egoverse-r2"]

# Subtask annotations: to pull them LIVE from the legacy DB on Modal (instead of
# the bundled backend/segments_cache.json fallback), create a secret with a
# direct connection string — the legacy RDS is a public endpoint, so this needs
# no AWS creds:
#
#   modal secret create egoverse-legacy \
#     LEGACY_DATABASE_URL='postgresql://appuser:<password>@lowuse-pg-east2.cdc8824mase4.us-east-2.rds.amazonaws.com:5432/appdb?sslmode=require'
#
# Then `db._enrich_segments` fetches segments live at pool-resolve time. Set
# LEGACY_DB_SECRET = None to skip it and rely on the bundled cache. If the name
# is set here but the secret doesn't exist, `modal deploy` fails fast (expected).
LEGACY_DB_SECRET = "egoverse-legacy"
if LEGACY_DB_SECRET:
    _SECRET_NAMES.append(LEGACY_DB_SECRET)


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name(n) for n in _SECRET_NAMES],
    max_containers=1,  # single SQLite writer; raters share this container
    scaledown_window=900,  # stay warm between clicks; cold start re-reads the volume
)
@modal.concurrent(max_inputs=25)
@modal.asgi_app()
def web():
    import os
    import sys

    os.environ["EGO_RATING_DATA_DIR"] = DATA_DIR  # SQLite on the persistent volume
    sys.path.insert(0, REMOTE_ROOT)
    from backend.main import app as fastapi_app

    return fastapi_app
