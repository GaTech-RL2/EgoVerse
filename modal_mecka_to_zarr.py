"""
Modal-based subset conversion of Mecka episodes to Zarr format.

Reads raw episode data from MongoDB (`mecka-ai`) and a configurable Cloudflare
R2 bucket (`R2_BUCKET` env var from the `mecka-zarr-conversion` secret),
converts each episode via MeckaDatasetConverter, and writes the resulting
`.zarr` directory + preview `.mp4` into a Modal Volume at::

    /vol/egoverse-zarr-data/processed_v3/<subset-name>/{task_type}/<episode_hash>.zarr
    /vol/egoverse-zarr-data/processed_v3/<subset-name>/{task_type}/<episode_hash>_video.mp4

`task_type` is `flagship` or `freeform`, classified via MongoDB `deliveryBatch`
(freeform = `rl2_freeform_final_jan_20`).

This script is **subset-only**: it does not touch the canonical R2 destination
bucket, AWS RDS `app.episodes`, or the Mongo `delivered` collection. Every run
must supply a `--subset-name` for routing.

Usage:
    # Dry run (3 episodes) via Modal
    modal run modal_mecka_to_zarr.py --subset-name sample --episode-ids-file ids.txt --dry-run

    # Full run, detached
    modal run --detach modal_mecka_to_zarr.py --subset-name pilot --episode-ids-file ids.txt

    # Subset defined by MongoDB filter
    modal run modal_mecka_to_zarr.py --subset-name freeform_jan \\
        --mongo-filter-json '{"deliveryBatch": "rl2_freeform_final_jan_20"}'

    # Local test (no Modal, writes to <output_dir>/<hash>.zarr)
    python modal_mecka_to_zarr.py --local-test --subset-name sample --episode-hash <hash>
"""

import json
import logging
import os
import re
import tempfile
import traceback
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal image – no torch/torchvision needed; preview MP4 uses ffmpeg fallback
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "curl")
    .pip_install(
        "numpy",
        "scipy",
        "pandas",
        "opencv-python-headless",
        "zarr>=3.0",
        "numcodecs",
        "simplejpeg",
        "pymongo",
        "boto3",
        "av",
        "sqlalchemy",
        "psycopg[binary]",
    )
    .add_local_dir(
        "egomimic",
        remote_path="/root/EgoVerse/egomimic",
    )
)

app = modal.App("mecka-zarr-conversion", image=image)

# ---------------------------------------------------------------------------
# Modal Volume — sole output destination
# Override with --volume-name at runtime, or set ZARR_VOLUME_NAME env var.
# ---------------------------------------------------------------------------
EGOVERSE_ZARR_VOLUME_NAME = os.environ.get("ZARR_VOLUME_NAME", "egoverse-zarr-data")
EGOVERSE_ZARR_VOLUME_MOUNT = f"/vol/{EGOVERSE_ZARR_VOLUME_NAME}"
egoverse_zarr_volume = modal.Volume.from_name(EGOVERSE_ZARR_VOLUME_NAME)

# ---------------------------------------------------------------------------
# MongoDB / R2 constants
# ---------------------------------------------------------------------------
R2_ENDPOINT_TEMPLATE = "https://{account_id}.r2.cloudflarestorage.com"
MONGODB_DB = "mecka-ai"
MONGODB_EPISODES_COLLECTION = "episodes"
MONGODB_VLM_SEGMENTS_COLLECTION = "episode_vlm_segments"
FREEFORM_DELIVERY_BATCH = "rl2_freeform_final_jan_20"

# Mongo doc fields whose values are R2 storage keys for episode assets.
MONGO_URL_FIELDS = {
    "video": "video_1",
    "hands": "pipeline_results.post_processing.hands_camera_interpolated",
    "egomotion": "pipeline_results.post_processing.egomotion_client",
    "frames": "framesKey",
}

# Body-pipeline outputs (pelvis-frame hands + camera_transform); used when
# available in preference to hands_camera_interpolated.
MONGO_BODY_FIELDS = {
    "hands_final": "pipeline_results.body.key_outputs.hands_final",
    "body_final": "pipeline_results.body.key_outputs.body_final",
}


# ---------------------------------------------------------------------------
# Subset routing
# ---------------------------------------------------------------------------
_SUBSET_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")


def _validate_subset_name(name: str) -> None:
    if not name or not _SUBSET_NAME_RE.match(name):
        raise ValueError(
            f"subset_name must match [a-zA-Z0-9_.-]{{1,128}}, got: {name!r}"
        )


def _volume_root() -> Path:
    """Filesystem path to the root of the mounted Modal volume."""
    return Path(EGOVERSE_ZARR_VOLUME_MOUNT)


def _write_episode_to_volume(
    local_zarr_dir: str,
    local_mp4: str,
    subset_name: str,
    task_type: str,
    episode_hash: str,
) -> tuple[str, str]:
    """
    Copy the converted episode into the Modal volume root (no subdirectories).

    Returns ``(zarr_dest, mp4_dest)``. Caller must commit the volume after.
    """
    import shutil

    dest_dir = _volume_root()
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_zarr = dest_dir / f"{episode_hash}.zarr"
    if dest_zarr.exists():
        shutil.rmtree(dest_zarr)
    shutil.copytree(local_zarr_dir, dest_zarr)

    dest_mp4 = ""
    if local_mp4 and os.path.exists(local_mp4):
        dest_mp4_path = dest_dir / f"{episode_hash}_video.mp4"
        shutil.copy2(local_mp4, dest_mp4_path)
        dest_mp4 = str(dest_mp4_path)

    return str(dest_zarr), dest_mp4


def _delete_episode_from_volume(
    episode_hash: str, task_type: str, subset_name: str
) -> None:
    """Best-effort cleanup of partial volume outputs for a failed episode."""
    import shutil

    dest_dir = _volume_root()
    dest_zarr = dest_dir / f"{episode_hash}.zarr"
    dest_mp4 = dest_dir / f"{episode_hash}_video.mp4"
    if dest_zarr.exists():
        shutil.rmtree(dest_zarr, ignore_errors=True)
        logger.info(f"[{episode_hash}] Removed stale volume zarr {dest_zarr}")
    if dest_mp4.exists():
        try:
            dest_mp4.unlink()
            logger.info(f"[{episode_hash}] Removed stale volume mp4 {dest_mp4}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mongo + R2 access
# ---------------------------------------------------------------------------

def _get_nested(doc: dict, dotted_key: str):
    """Resolve a dotted key like 'pipeline_results.post_processing.hands'."""
    current = doc
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _parse_storage_key(storage_key: str) -> tuple[str, str]:
    """
    Parse a storage key into ``(bucket, key)``.

    Rules:
      - ``r2://<bucket>/<key>`` → use ``<bucket>`` and ``<key>`` from the URI.
        These keys come from the upstream pipeline and reference the original
        bucket the file was written to (e.g. ``atlas``).
      - Bare path (no ``r2://`` prefix) → use ``R2_BUCKET`` env var as the
        bucket. Mongo lookups like ``device_intrinsics.intrinsics_1080p`` store
        bare paths, so these get routed to the configurable fallback bucket.
    """
    if storage_key.startswith("r2://"):
        without_scheme = storage_key[len("r2://"):]
        bucket, _, key = without_scheme.partition("/")
        if not bucket or not key:
            raise ValueError(f"Malformed r2:// storage key: {storage_key!r}")
        return bucket, key

    bucket = os.environ.get("R2_BUCKET") or os.environ.get("BUCKET")
    if not bucket:
        raise RuntimeError(
            f"Storage key has no r2:// prefix and no bucket env var is set: "
            f"{storage_key!r}. Set R2_BUCKET or BUCKET in the egoverse-r2 secret."
        )
    return bucket, storage_key


def _get_r2_client():
    """
    boto3 S3 client pointed at Cloudflare R2.

    Reads from the ``egoverse-r2`` secret (preferred names) with fallbacks:
      - Endpoint: ``AWS_ENDPOINT_URL_S3`` | ``R2_ENDPOINT_URL`` | ``R2_ENDPOINT``
      - Access key: ``R2_ACCESS_KEY_ID`` | ``R2_ACCESS_KEY``
      - Secret key: ``R2_SECRET_ACCESS_KEY`` | ``R2_SECRET_KEY``
    """
    import boto3

    endpoint_url = (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT")
    )
    if not endpoint_url:
        account_id = os.environ.get("R2_ACCOUNT_ID")
        if not account_id:
            raise RuntimeError(
                "No R2 endpoint configured. Set AWS_ENDPOINT_URL_S3, "
                "R2_ENDPOINT_URL, or R2_ACCOUNT_ID in the egoverse-r2 secret."
            )
        endpoint_url = R2_ENDPOINT_TEMPLATE.format(account_id=account_id)

    access_key = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "R2 access key and secret must be set. Use R2_ACCESS_KEY_ID + "
            "R2_SECRET_ACCESS_KEY in the egoverse-r2 secret."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=boto3.session.Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _sign_url(r2_client, storage_key: str, expiry: int = 3600) -> str:
    bucket, key = _parse_storage_key(storage_key)
    logger.debug(f"sign s3://{bucket}/{key}  (from mongo value: {storage_key})")
    return r2_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry,
    )


def _get_mongo_db():
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGODB_URI"])
    return client[MONGODB_DB]


def _fetch_vlm_annotations(db, episode_hash: str) -> list[dict]:
    """
    Fetch reviewed VLM segments from ``episode_vlm_segments``.

    Prefers ``tierSnapshots.delivery.segments`` (finalized labels) and falls
    back to top-level ``segments``. Closes sub-second gaps between consecutive
    segments by extending the earlier segment's end.

    If the Mongo user has no read permission on ``episode_vlm_segments``, this
    function logs a warning and returns ``[]`` rather than raising. The caller
    decides what to do — flagship episodes proceed without annotations,
    freeform episodes raise a clear error in ``_prepare_episode``.
    """
    from bson import ObjectId
    from pymongo.errors import OperationFailure

    col = db[MONGODB_VLM_SEGMENTS_COLLECTION]
    try:
        doc = col.find_one({"episodeId": ObjectId(episode_hash)})
    except OperationFailure as e:
        if "not authorized" in str(e).lower() or getattr(e, "code", None) == 13:
            logger.warning(
                f"[{episode_hash}] Not authorized to read "
                f"{MONGODB_VLM_SEGMENTS_COLLECTION} (MongoDB user lacks "
                f"permission). Proceeding without VLM annotations."
            )
            return []
        raise
    if doc is None:
        return []

    delivery = doc.get("tierSnapshots", {}).get("delivery", {})
    segments = delivery.get("segments", []) if isinstance(delivery, dict) else []
    if not segments:
        segments = doc.get("segments", [])

    annotations = []
    for seg in segments:
        label = seg.get("label") or seg.get("vlmLabel") or ""
        start = seg.get("start_seconds", 0)
        end = seg.get("end_seconds", 0)
        if label and end > start:
            annotations.append({"Labels": label, "start_time": start, "end_time": end})

    for i in range(len(annotations) - 1):
        gap = annotations[i + 1]["start_time"] - annotations[i]["end_time"]
        if 0 < gap < 1.0:
            annotations[i]["end_time"] = annotations[i + 1]["start_time"]

    return annotations


def _write_annotations_csv(annotations: list[dict], dest_path: str) -> None:
    import pandas as pd

    if annotations:
        df = pd.DataFrame(annotations)
    else:
        df = pd.DataFrame(columns=["Labels", "start_time", "end_time"])
    df.to_csv(dest_path, index=False)


def _convert_hands_final_to_csv(
    hands_final_path: str,
    body_final_path: str,
    output_csv_path: str,
) -> None:
    """
    Convert hands_final.json (pelvis frame) -> hands CSV (camera frame).

    Uses body_final's per-frame camera_transform: cam = inv(camera_transform) @ pelvis.
    Frame numbers are remapped to a local 0-based timeline by subtracting the
    minimum body_final frame.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial.transform import Rotation

    with open(hands_final_path) as f:
        hands_data = json.load(f)
    with open(body_final_path) as f:
        body_data = json.load(f)

    frame_to_cTp = {}
    for pose in body_data["poses"]:
        frame_num = pose["frame"]
        ct = pose["camera_transform"]
        pos, ori = ct["position"], ct["orientation"]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([ori["x"], ori["y"], ori["z"], ori["w"]]).as_matrix()
        T[:3, 3] = [pos["x"], pos["y"], pos["z"]]
        frame_to_cTp[frame_num] = np.linalg.inv(T)

    if not frame_to_cTp:
        raise ValueError("body_final.json contained no camera_transform frames")

    base_frame = min(frame_to_cTp)
    HAND_TYPE_TO_INDEX = {"LEFT": 0, "RIGHT": 1}
    rows = []

    for frame_str, frame_data in hands_data["frames"].items():
        frame_num = int(frame_str)
        cTp = frame_to_cTp.get(frame_num)
        if cTp is None:
            continue
        for hand in frame_data.get("hands", []):
            hand_idx = HAND_TYPE_TO_INDEX.get(hand["hand_type"])
            if hand_idx is None:
                continue
            for lm in hand["landmarks"]:
                pelvis_h = np.array([lm["x"], lm["y"], lm["z"], 1.0])
                cam_xyz = (cTp @ pelvis_h)[:3]
                rows.append({
                    "frame": frame_num - base_frame,
                    "hand_index": hand_idx,
                    "landmark_index": lm["landmark_index"],
                    "world_x": cam_xyz[0],
                    "world_y": cam_xyz[1],
                    "world_z": cam_xyz[2],
                })

    df = pd.DataFrame(rows).sort_values(["frame", "hand_index", "landmark_index"]).reset_index(drop=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(
        f"Converted hands_final -> {output_csv_path} "
        f"({len(df)} rows, {df['frame'].nunique()} frames, base_frame={base_frame})"
    )


def _is_freeform(db, episode_hash: str) -> bool:
    from bson import ObjectId

    doc = db[MONGODB_EPISODES_COLLECTION].find_one(
        {"_id": ObjectId(episode_hash)}, {"deliveryBatch": 1}
    )
    return bool(doc and doc.get("deliveryBatch") == FREEFORM_DELIVERY_BATCH)


def _resolve_intrinsics_key(db, episode_hash: str, mongo_doc: dict) -> str:
    """
    Resolve the intrinsics R2 storage key.

    1. ``intrinsicsKey`` directly on the episode doc.
    2. Device-model chain: episode → user_tasks (userTaskId) → files (meta.fileId)
       → device.modelName → device_intrinsics (intrinsics_1080p).
    """
    from bson import ObjectId

    direct_key = mongo_doc.get("intrinsicsKey")
    if direct_key:
        logger.info(f"[{episode_hash}] Intrinsics resolved via intrinsicsKey")
        return direct_key

    logger.info(f"[{episode_hash}] intrinsicsKey missing, falling back to device model chain")
    user_task_id = mongo_doc.get("userTaskId")
    if not user_task_id:
        raise ValueError(f"Episode {episode_hash} missing both intrinsicsKey and userTaskId")

    user_task = db["user_tasks"].find_one({"_id": user_task_id})
    if not user_task:
        raise ValueError(f"Episode {episode_hash}: userTask {user_task_id} not found")

    file_id = user_task.get("meta", {}).get("fileId")
    if not file_id:
        raise ValueError(
            f"Episode {episode_hash}: userTask {user_task_id} missing meta.fileId"
        )

    file_doc = db["files"].find_one({"_id": ObjectId(str(file_id))})
    if not file_doc:
        raise ValueError(f"Episode {episode_hash}: file {file_id} not found")

    model_name = file_doc.get("device", {}).get("modelName")
    if not model_name:
        raise ValueError(
            f"Episode {episode_hash}: file {file_id} missing device.modelName"
        )

    # Special case: iPhone17,3 → iPhone17,4
    if model_name == "iPhone17,3":
        model_name = "iPhone17,4"

    intrinsics_doc = db["device_intrinsics"].find_one({"device": model_name})
    if not intrinsics_doc:
        raise ValueError(
            f"Episode {episode_hash}: no device_intrinsics for model '{model_name}'"
        )
    key = intrinsics_doc.get("intrinsics_1080p")
    if not key:
        raise ValueError(
            f"Episode {episode_hash}: device_intrinsics for '{model_name}' "
            f"missing intrinsics_1080p"
        )
    logger.info(f"[{episode_hash}] Intrinsics resolved via device model '{model_name}' → {key}")
    return key


def _classify_task_types(episodes_col, episode_hashes: list[str]) -> dict[str, str]:
    """Return {hash: 'freeform' | 'flagship'} by deliveryBatch lookup."""
    from bson import ObjectId

    out = {h: "flagship" for h in episode_hashes}
    batch_size = 10000
    for i in range(0, len(episode_hashes), batch_size):
        batch = episode_hashes[i : i + batch_size]
        for doc in episodes_col.find(
            {
                "_id": {"$in": [ObjectId(h) for h in batch]},
                "deliveryBatch": FREEFORM_DELIVERY_BATCH,
            },
            {"_id": 1},
        ):
            out[str(doc["_id"])] = "freeform"
    return out


# ---------------------------------------------------------------------------
# Postgres SQL writes (records subset conversions into app.episodes)
# ---------------------------------------------------------------------------
# Triggered only when PG_HOST is set in the environment. When unset, the run
# completes successfully but emits a warning that the SQL step was skipped.

SQL_UPSERT_COLUMNS = [
    "episode_hash", "operator", "task", "embodiment", "robot_name",
    "num_frames", "task_description", "scene", "objects",
    "zarr_processed_path", "zarr_mp4_path", "zarr_processing_error",
    "is_deleted", "data_type",
]


def _pg_env_present() -> bool:
    return bool(os.environ.get("DATABASE_URL") or os.environ.get("PG_HOST"))


def _sql_engine_from_pg_env():
    """SQLAlchemy engine from egoverse-sql secret (DATABASE_URL or PG_* vars)."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Swap scheme to psycopg3 driver if needed
        url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        return create_engine(url, poolclass=NullPool)

    from urllib.parse import quote_plus
    user = os.environ["PG_USER"]
    password = quote_plus(os.environ["PG_PASSWORD"])
    host = os.environ["PG_HOST"]
    port = os.environ.get("PG_PORT", "5432")
    database = os.environ.get("PG_DATABASE", "defaultdb")
    return create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}?sslmode=require",
        poolclass=NullPool,
    )


def _build_sql_row(result: dict) -> dict:
    """Map a worker result dict (ok or error) to one app.episodes row."""
    ok = result.get("status") == "ok"
    return {
        "episode_hash": result["episode_hash"],
        "operator": result.get("sql_operator", ""),
        "task": result.get("sql_task", ""),
        "embodiment": "mecka",
        "robot_name": "mecka_bimanual",
        "num_frames": int(result.get("sql_num_frames", -1) or -1),
        "task_description": result.get("task_description", ""),
        "scene": result.get("sql_scene", ""),
        "objects": result.get("sql_objects", "[]"),
        "zarr_processed_path": result.get("zarr_path", "") if ok else "",
        "zarr_mp4_path": result.get("mp4_path", "") if ok else "",
        "zarr_processing_error": "" if ok else (result.get("error", "") or "")[:500],
        "is_deleted": False,
        "data_type": result.get("task_type", ""),
    }


def _fetch_episode_metadata_batch(episodes_col, episode_hashes: list[str]) -> dict:
    """
    Batch-fetch episode metadata from ``episodes`` collection.

    Returns ``{episode_hash: doc}`` containing only the fields needed for SQL
    row construction.
    """
    from bson import ObjectId

    out: dict[str, dict] = {}
    batch_size = 5000
    for i in range(0, len(episode_hashes), batch_size):
        batch = episode_hashes[i : i + batch_size]
        for doc in episodes_col.find(
            {"_id": {"$in": [ObjectId(h) for h in batch]}},
            {
                "userId": 1,
                "task_id": 1,
                "scene_id": 1,
                "objects": 1,
                "video_1_frames": 1,
                "deliveryBatch": 1,
            },
        ):
            out[str(doc["_id"])] = doc
    return out


def _fetch_task_descriptions_batch(vlm_col, episode_hashes: list[str]) -> dict:
    """
    Batch-fetch ``taskDescription`` from ``episode_vlm_segments``.

    Tolerates ``OperationFailure`` (Mongo user lacking read permission on the
    collection) — returns ``{}`` and logs a warning instead of raising.
    """
    from bson import ObjectId
    from pymongo.errors import OperationFailure

    out: dict[str, str] = {}
    batch_size = 5000
    try:
        for i in range(0, len(episode_hashes), batch_size):
            batch = episode_hashes[i : i + batch_size]
            for doc in vlm_col.find(
                {"episodeId": {"$in": [ObjectId(h) for h in batch]}},
                {"episodeId": 1, "taskDescription": 1},
            ):
                desc = (doc.get("taskDescription") or "").strip()
                if desc:
                    out[str(doc["episodeId"])] = desc
    except OperationFailure as e:
        logger.warning(
            f"Could not read taskDescriptions (Mongo user lacks permission on "
            f"{MONGODB_VLM_SEGMENTS_COLLECTION}): {e}. "
            f"task_description will be empty in SQL rows."
        )
    return out


def _build_sql_row_from_mongo(
    episode_hash: str,
    mongo_doc: dict,
    task_type: str,
    task_description: str,
    zarr_path: str,
    mp4_path: str,
    error: str = "",
) -> dict:
    """
    Construct one ``app.episodes`` row dict from Mongo metadata + volume paths.
    Used by the SQL-only mode (which doesn't run a worker).
    """
    objects_raw = mongo_doc.get("objects", [])
    objects_json = (
        json.dumps([str(o) for o in objects_raw]) if objects_raw else "[]"
    )
    return {
        "episode_hash": episode_hash,
        "operator": str(mongo_doc.get("userId", "")),
        "task": str(mongo_doc.get("task_id", "")),
        "embodiment": "mecka",
        "robot_name": "mecka_bimanual",
        "num_frames": int(mongo_doc.get("video_1_frames", -1) or -1),
        "task_description": task_description,
        "scene": str(mongo_doc.get("scene_id", "")),
        "objects": objects_json,
        "zarr_processed_path": zarr_path if not error else "",
        "zarr_mp4_path": mp4_path if not error else "",
        "zarr_processing_error": error[:500],
        "is_deleted": False,
        "data_type": task_type,
    }


def _upsert_episode_records(engine, records: list[dict]) -> tuple[int, int]:
    """
    UPSERT ``records`` into ``app.episodes`` on conflict by ``episode_hash``.

    Returns ``(ok_count, error_count)``.
    """
    from sqlalchemy import text as _text

    if not records:
        return 0, 0

    cols = SQL_UPSERT_COLUMNS
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "episode_hash")
    stmt = _text(
        f"INSERT INTO app.episodes ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (episode_hash) DO UPDATE SET {updates}"
    )

    ok = 0
    err = 0
    # Single transaction for the whole batch; individual rows that fail (e.g.
    # missing data_type column) shouldn't take the rest down.
    with engine.begin() as conn:
        for row in records:
            try:
                conn.execute(stmt, row)
                ok += 1
            except Exception as e:
                err += 1
                logger.warning(
                    f"SQL upsert failed for {row.get('episode_hash')}: {e}"
                )
    return ok, err


def _resolve_mongo_filter_ids(db, mongo_filter_json: str) -> set[str]:
    """Resolve a MongoDB extended-JSON filter to a set of episode hash strings."""
    if not mongo_filter_json or not mongo_filter_json.strip():
        return set()
    from bson import json_util

    filter_doc = json_util.loads(mongo_filter_json)
    ids = {str(d["_id"]) for d in db[MONGODB_EPISODES_COLLECTION].find(filter_doc, {"_id": 1})}
    logger.info(f"MongoDB filter matched {len(ids)} episodes: {mongo_filter_json}")
    return ids


# ---------------------------------------------------------------------------
# Episode preparation: download raw data → run converter → return local paths
# ---------------------------------------------------------------------------

def _prepare_episode(episode_hash: str, task_type: str, tmp_dir: str) -> dict:
    """
    Download raw R2 data for the episode, convert it to zarr+mp4 in ``tmp_dir``,
    and return paths plus metadata. Does NOT write to the volume or anywhere
    durable — the caller (worker or local_test) decides where to put the output.

    Returns a dict with keys:
        episode_hash, task_type, zarr_dir (local path), mp4_path (local path or None),
        num_annotations, task_description.
    """
    import sys
    if "/root/EgoVerse" not in sys.path:
        sys.path.insert(0, "/root/EgoVerse")
    try:
        project_root = str(Path(__file__).resolve().parents[3])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except IndexError:
        pass

    from bson import ObjectId
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from egomimic.scripts.mecka_process.mecka_to_zarr import (
        MeckaDatasetConverter,
        download_with_retry,
    )

    db = _get_mongo_db()
    episodes_col = db[MONGODB_EPISODES_COLLECTION]

    # ---- 1. Episode doc ----
    mongo_doc = episodes_col.find_one({"_id": ObjectId(episode_hash)})
    if mongo_doc is None:
        raise ValueError(f"Episode {episode_hash} not found in MongoDB")

    # ---- 2. Sign R2 URLs ----
    r2 = _get_r2_client()
    urls = {}
    for url_key, mongo_field in MONGO_URL_FIELDS.items():
        storage_key = _get_nested(mongo_doc, mongo_field)
        if storage_key is None:
            raise ValueError(
                f"MongoDB doc missing field '{mongo_field}' for episode {episode_hash}"
            )
        bucket, key = _parse_storage_key(storage_key)
        logger.info(
            f"[{episode_hash}] {url_key}: s3://{bucket}/{key} "
            f"(mongo value: {storage_key})"
        )
        urls[url_key] = _sign_url(r2, storage_key)

    intrinsics_key = _resolve_intrinsics_key(db, episode_hash, mongo_doc)
    urls["intrinsics"] = _sign_url(r2, intrinsics_key)

    # ---- 3. VLM annotations ----
    vlm_annotations = _fetch_vlm_annotations(db, episode_hash)
    if not vlm_annotations and task_type == "freeform":
        raise ValueError(
            f"Freeform episode {episode_hash} has no VLM segments in "
            f"{MONGODB_VLM_SEGMENTS_COLLECTION}"
        )
    logger.info(
        f"[{episode_hash}] VLM annotations: {len(vlm_annotations)} segments "
        f"({'required' if task_type == 'freeform' else 'optional'})"
    )

    # ---- 4. Pre-create download dir + annotations.csv ----
    download_dir = os.path.join(tmp_dir, "temp_download")
    os.makedirs(download_dir, exist_ok=True)
    _write_annotations_csv(vlm_annotations, os.path.join(download_dir, "annotations.csv"))

    # ---- 5. Parallel download of raw assets ----
    hands_final_key = _get_nested(mongo_doc, MONGO_BODY_FIELDS["hands_final"])
    body_final_key = _get_nested(mongo_doc, MONGO_BODY_FIELDS["body_final"])
    use_hands_final = bool(hands_final_key and body_final_key)

    downloads = [
        (urls["video"], os.path.join(download_dir, "video.mp4")),
        (urls["egomotion"], os.path.join(download_dir, "egomotion.txt")),
        (urls["frames"], os.path.join(download_dir, "frames.csv")),
        (urls["intrinsics"], os.path.join(download_dir, "intrinsics.json")),
    ]
    if use_hands_final:
        downloads.append((_sign_url(r2, hands_final_key), os.path.join(download_dir, "hands_final.json")))
        downloads.append((_sign_url(r2, body_final_key), os.path.join(download_dir, "body_final.json")))
    else:
        downloads.append((urls["hands"], os.path.join(download_dir, "hands.csv")))

    logger.info(f"[{episode_hash}] Downloading {len(downloads)} files in parallel...")
    with ThreadPoolExecutor(max_workers=len(downloads)) as pool:
        futures = {pool.submit(download_with_retry, url, path): path for url, path in downloads}
        for fut in as_completed(futures):
            dest_path = futures[fut]
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(
                    f"Download failed for {Path(dest_path).name} "
                    f"(episode {episode_hash}): {e}"
                ) from e

    # Validate JSON files actually contain JSON (catches 404 XML bodies, empties)
    for json_name in ("intrinsics.json",) + (
        ("hands_final.json", "body_final.json") if use_hands_final else ()
    ):
        p = os.path.join(download_dir, json_name)
        try:
            with open(p) as fh:
                json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            size = os.path.getsize(p) if os.path.exists(p) else 0
            raise RuntimeError(
                f"Downloaded {json_name} for episode {episode_hash} is not valid "
                f"JSON ({size} bytes) — the R2 source key likely does not exist "
                f"in bucket '{os.environ.get('R2_BUCKET', '?')}'. Original: {e}"
            ) from e

    if use_hands_final:
        logger.info(f"[{episode_hash}] Converting hands_final → camera-frame hands.csv")
        _convert_hands_final_to_csv(
            os.path.join(download_dir, "hands_final.json"),
            os.path.join(download_dir, "body_final.json"),
            os.path.join(download_dir, "hands.csv"),
        )

    # ---- 6. Synthesize episode.json that MeckaDatasetConverter expects ----
    from pymongo.errors import OperationFailure
    try:
        vlm_doc = db[MONGODB_VLM_SEGMENTS_COLLECTION].find_one(
            {"episodeId": ObjectId(episode_hash)}, {"taskDescription": 1}
        )
    except OperationFailure:
        vlm_doc = None  # already warned above in _fetch_vlm_annotations
    task_description = (vlm_doc or {}).get("taskDescription", "") or ""

    episode_json_path = os.path.join(tmp_dir, "episode.json")
    with open(episode_json_path, "w") as f:
        json.dump(
            {
                "id": episode_hash,
                "urls": urls,
                "user_id": str(mongo_doc.get("userId", "")),
                "duration": mongo_doc.get("duration"),
                "environment_id": str(mongo_doc.get("environment_id", "")),
                "scene_id": str(mongo_doc.get("scene_id", "")),
                "scene_desc": mongo_doc.get("scene_desc", ""),
                "objects": [str(o) for o in mongo_doc.get("objects", [])],
            },
            f,
            default=str,
        )

    # ---- 7. Run conversion ----
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    MeckaDatasetConverter(
        episode_json_path=episode_json_path,
        output_dir=output_dir,
        repo_id="mecka/zarr",
        arm="both",
        task_description=task_description,
    ).extract_episode()

    local_zarr_dir = os.path.join(output_dir, f"{episode_hash}.zarr")
    local_mp4 = os.path.join(output_dir, f"{episode_hash}.mp4")

    # Metadata needed for the SQL row in app.episodes.
    objects_raw = mongo_doc.get("objects", [])
    objects_json = json.dumps([str(o) for o in objects_raw]) if objects_raw else "[]"

    return {
        "episode_hash": episode_hash,
        "task_type": task_type,
        "zarr_dir": local_zarr_dir,
        "mp4_path": local_mp4 if os.path.exists(local_mp4) else None,
        "num_annotations": len(vlm_annotations),
        "task_description": task_description,
        # SQL row fields (worker captures them while Mongo doc is in hand)
        "sql_operator": str(mongo_doc.get("userId", "")),
        "sql_task": str(mongo_doc.get("task_id", "")),
        "sql_num_frames": int(mongo_doc.get("video_1_frames", -1) or -1),
        "sql_scene": str(mongo_doc.get("scene_id", "")),
        "sql_objects": objects_json,
    }


# ---------------------------------------------------------------------------
# Modal worker: convert + write to volume
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("egoverse-r2"),
        modal.Secret.from_name("egoverse-mongodb"),
        modal.Secret.from_name("egoverse-sql"),
    ],
    volumes={EGOVERSE_ZARR_VOLUME_MOUNT: egoverse_zarr_volume},
    timeout=3600,
    memory=8192,
    cpu=2,
    max_containers=4000,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
def convert_episode(episode_hash: str, task_type: str, subset_name: str) -> dict:
    """
    Convert a single episode and write the result to the Modal volume.

    Returns: ``{episode_hash, status: "ok" | "error", zarr_path, mp4_path, error}``.
    """
    import sys
    sys.path.insert(0, "/root/EgoVerse")

    _validate_subset_name(subset_name)

    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix=f"zarr_{episode_hash}_")
        result = _prepare_episode(episode_hash, task_type, tmp_dir)

        zarr_path, mp4_path = _write_episode_to_volume(
            local_zarr_dir=result["zarr_dir"],
            local_mp4=result["mp4_path"] or "",
            subset_name=subset_name,
            task_type=task_type,
            episode_hash=episode_hash,
        )
        egoverse_zarr_volume.commit()
        logger.info(f"[{episode_hash}] Wrote to volume -> {zarr_path}")

        return {
            "episode_hash": episode_hash,
            "task_type": task_type,
            "status": "ok",
            "zarr_path": zarr_path,
            "mp4_path": mp4_path,
            "num_annotations": result["num_annotations"],
            "task_description": result["task_description"],
            # SQL row fields
            "sql_operator": result.get("sql_operator", ""),
            "sql_task": result.get("sql_task", ""),
            "sql_num_frames": result.get("sql_num_frames", -1),
            "sql_scene": result.get("sql_scene", ""),
            "sql_objects": result.get("sql_objects", "[]"),
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"[{episode_hash}] Failed: {error_msg}")
        logger.error(traceback.format_exc())
        try:
            _delete_episode_from_volume(episode_hash, task_type, subset_name)
        except Exception as cleanup_error:
            logger.warning(f"[{episode_hash}] Volume cleanup failed: {cleanup_error}")
        return {
            "episode_hash": episode_hash,
            "task_type": task_type,
            "status": "error",
            "error": error_msg,
        }

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batch dispatcher
# ---------------------------------------------------------------------------

def _run_subset_batch(
    db,
    episodes_col,
    allowed_episode_ids,
    mongo_filter_json: str,
    subset_name: str,
    limit: int,
    dry_run: bool,
    force_task_type: str,
    skip_sql: bool = False,
) -> dict:
    """
    Dispatch episodes to Modal workers, collect results, and upsert one row per
    episode into Postgres ``app.episodes`` (if PG_HOST is set and not skipped).
    """
    import time

    _validate_subset_name(subset_name)

    # ---- 1. Resolve subset (union of ID list + Mongo filter) ----
    subset_ids: set[str] = set()
    if allowed_episode_ids:
        subset_ids |= set(allowed_episode_ids)
    if mongo_filter_json:
        subset_ids |= _resolve_mongo_filter_ids(db, mongo_filter_json)

    if not subset_ids:
        logger.info("Subset is empty (no IDs from list or Mongo filter); nothing to do")
        return {"total": 0, "ok": 0, "errors": 0, "exceptions": 0}

    all_hashes = sorted(subset_ids)
    logger.info(
        f"Subset '{subset_name}': {len(all_hashes)} candidate episodes "
        f"(list={len(allowed_episode_ids) if allowed_episode_ids else 0}, "
        f"filter={'set' if mongo_filter_json else 'unset'})"
    )

    # ---- 2. Classify task_type ----
    if force_task_type:
        task_type_by_hash = {h: force_task_type for h in all_hashes}
        logger.info(f"  Forced task_type={force_task_type} for all episodes")
    else:
        task_type_by_hash = _classify_task_types(episodes_col, all_hashes)
        flagship_count = sum(1 for v in task_type_by_hash.values() if v == "flagship")
        freeform_count = sum(1 for v in task_type_by_hash.values() if v == "freeform")
        logger.info(f"  Flagship: {flagship_count}, Freeform: {freeform_count}")

    # ---- 3. Apply limit / dry_run ----
    work = [(h, task_type_by_hash[h]) for h in all_hashes]
    if dry_run:
        work = work[:3]
        logger.info(f"DRY RUN: processing {len(work)} episodes")
    elif limit > 0:
        work = work[:limit]
        logger.info(f"LIMITED: processing {len(work)} episodes")

    if not work:
        return {"total": 0, "ok": 0, "errors": 0, "exceptions": 0}

    # ---- 4. Dispatch ----
    hashes = [w[0] for w in work]
    task_types = [w[1] for w in work]
    subset_names = [subset_name] * len(work)
    total = len(work)

    logger.info(f"Dispatching {total} episodes to Modal (subset='{subset_name}')...")
    start_time = time.time()

    ok = 0
    errors: list[str] = []
    exceptions = 0
    processed = 0
    records: list[dict] = []  # one row per worker result, used for SQL upsert

    for r in convert_episode.map(
        hashes,
        task_types,
        subset_names,
        order_outputs=False,
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        processed += 1
        if isinstance(r, Exception):
            exceptions += 1
            errors.append(f"EXCEPTION: {r}")
        elif isinstance(r, dict):
            records.append(_build_sql_row(r))
            if r["status"] == "ok":
                ok += 1
            else:
                errors.append(f"{r['episode_hash']}: {r.get('error', 'unknown')}")

        if processed % 100 == 0 or processed == total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_m = (total - processed) / rate / 60 if rate > 0 else 0
            logger.info(
                f"PROGRESS: {processed}/{total} ({ok} ok, {len(errors)} err) "
                f"[{elapsed/60:.1f}m elapsed, ~{eta_m:.0f}m remaining]"
            )

    elapsed = time.time() - start_time

    # ---- 5. SQL upsert into app.episodes (one row per episode) ----
    sql_ok = sql_err = 0
    if skip_sql:
        logger.info("SQL upsert SKIPPED (--skip-sql)")
    elif not _pg_env_present():
        logger.warning(
            "SQL upsert SKIPPED: no SQL credentials found. Add DATABASE_URL (or "
            "PG_HOST/PG_USER/PG_PASSWORD) to the egoverse-sql Modal secret."
        )
    elif not records:
        logger.info("SQL upsert SKIPPED: no records to write")
    else:
        try:
            engine = _sql_engine_from_pg_env()
            sql_ok, sql_err = _upsert_episode_records(engine, records)
            engine.dispose()
            logger.info(f"SQL upsert: {sql_ok} ok, {sql_err} failed")
        except Exception as e:
            logger.error(f"SQL upsert step failed entirely: {e}")
            sql_err = len(records)

    out_prefix = (
        f"modal-volume://{EGOVERSE_ZARR_VOLUME_NAME}/processed_v3/{subset_name}/"
    )
    logger.info("=" * 60)
    logger.info(
        f"SUBSET '{subset_name}' DONE in {elapsed/60:.1f} minutes: "
        f"{ok} ok, {len(errors)} errors, {exceptions} exceptions"
    )
    logger.info(
        f"Outputs landed in Modal volume '{EGOVERSE_ZARR_VOLUME_NAME}' at: "
        f"{out_prefix}{{flagship,freeform}}/<episode_hash>.zarr"
    )
    logger.info("=" * 60)

    if errors:
        for e in errors[:50]:
            logger.info(f"  {e}")
        if len(errors) > 50:
            logger.info(f"  ... and {len(errors) - 50} more")

    return {
        "total": total,
        "ok": ok,
        "errors": len(errors),
        "exceptions": exceptions,
        "elapsed_seconds": elapsed,
        "subset_name": subset_name,
        "output_prefix": out_prefix,
        "sql_ok": sql_ok,
        "sql_errors": sql_err,
    }


# ---------------------------------------------------------------------------
# Remote orchestrator (runs on Modal; holds Mongo connection for the batch)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("egoverse-r2"),
        modal.Secret.from_name("egoverse-mongodb"),
        modal.Secret.from_name("egoverse-sql"),
    ],
    timeout=86400,
    memory=4096,
    cpu=2,
)
def orchestrate_subset_batch(
    episode_ids_csv: str = "",
    subset_name: str = "",
    mongo_filter_json: str = "",
    limit: int = 0,
    dry_run: bool = False,
    force_task_type: str = "",
    skip_sql: bool = False,
) -> dict:
    import sys
    if "/root/EgoVerse" not in sys.path:
        sys.path.insert(0, "/root/EgoVerse")

    _validate_subset_name(subset_name)

    allowed_ids = None
    if episode_ids_csv.strip():
        allowed_ids = {x.strip() for x in episode_ids_csv.split(",") if x.strip()}

    from pymongo import MongoClient

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not set")
    client = MongoClient(mongo_uri)
    db = client[MONGODB_DB]
    episodes_col = db[MONGODB_EPISODES_COLLECTION]

    return _run_subset_batch(
        db=db,
        episodes_col=episodes_col,
        allowed_episode_ids=allowed_ids,
        mongo_filter_json=mongo_filter_json,
        subset_name=subset_name,
        limit=limit,
        dry_run=dry_run,
        force_task_type=force_task_type,
        skip_sql=skip_sql,
    )


# ---------------------------------------------------------------------------
# SQL-only mode (no conversion, no R2 reads; just refresh app.episodes rows)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("egoverse-r2"),
        modal.Secret.from_name("egoverse-mongodb"),
        modal.Secret.from_name("egoverse-sql"),
    ],
    volumes={EGOVERSE_ZARR_VOLUME_MOUNT: egoverse_zarr_volume},
    timeout=3600,
    memory=2048,
    cpu=1,
)
def sql_only_batch(
    episode_ids_csv: str = "",
    subset_name: str = "",
    mongo_filter_json: str = "",
    force_task_type: str = "",
    limit: int = 0,
    verify_volume: bool = True,
) -> dict:
    """
    Backfill / refresh ``app.episodes`` rows WITHOUT running a conversion.

    Use this to:
      - Backfill SQL after a Modal run that was invoked with ``--skip-sql``
      - Re-record rows after manually re-running a subset
      - Refresh stale rows from current MongoDB metadata

    For each episode in the resolved subset:
      1. Pull metadata from MongoDB (one batch query).
      2. Classify task_type via ``deliveryBatch`` (or use ``force_task_type``).
      3. Compute the expected volume path
         (``processed_v3/<subset>/<task_type>/<hash>.zarr``).
      4. If ``verify_volume=True``, check the path exists on the mounted Modal
         volume; missing files get an error row (``zarr_processing_error`` set).
      5. Upsert all rows into ``app.episodes`` in one transaction.
    """
    import sys
    if "/root/EgoVerse" not in sys.path:
        sys.path.insert(0, "/root/EgoVerse")

    _validate_subset_name(subset_name)
    if not _pg_env_present():
        raise RuntimeError(
            "No SQL credentials found in egoverse-sql secret. Add DATABASE_URL "
            "(or PG_HOST/PG_USER/PG_PASSWORD) before running --sql-only."
        )

    from pymongo import MongoClient

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI not set")
    client = MongoClient(mongo_uri)
    db = client[MONGODB_DB]
    episodes_col = db[MONGODB_EPISODES_COLLECTION]
    vlm_col = db[MONGODB_VLM_SEGMENTS_COLLECTION]

    # ---- 1. Resolve subset ----
    subset_ids: set[str] = set()
    if episode_ids_csv.strip():
        subset_ids |= {x.strip() for x in episode_ids_csv.split(",") if x.strip()}
    if mongo_filter_json:
        subset_ids |= _resolve_mongo_filter_ids(db, mongo_filter_json)

    if not subset_ids:
        logger.info("SQL-only: subset is empty; nothing to do")
        return {"total": 0, "sql_ok": 0, "sql_errors": 0, "missing_from_volume": 0}

    all_hashes = sorted(subset_ids)
    if limit > 0:
        all_hashes = all_hashes[:limit]
    logger.info(
        f"SQL-only mode for subset '{subset_name}': {len(all_hashes)} episodes "
        f"(verify_volume={verify_volume})"
    )

    # ---- 2. Batch fetch Mongo metadata ----
    metadata = _fetch_episode_metadata_batch(episodes_col, all_hashes)
    task_descriptions = _fetch_task_descriptions_batch(vlm_col, all_hashes)
    logger.info(
        f"Mongo lookups: {len(metadata)}/{len(all_hashes)} episode docs, "
        f"{len(task_descriptions)} taskDescriptions"
    )

    # ---- 3. Build rows ----
    records: list[dict] = []
    missing_from_volume = 0
    missing_from_mongo = 0

    for episode_hash in all_hashes:
        mongo_doc = metadata.get(episode_hash, {})
        if not mongo_doc:
            missing_from_mongo += 1

        if force_task_type:
            task_type = force_task_type
        else:
            task_type = (
                "freeform"
                if mongo_doc.get("deliveryBatch") == FREEFORM_DELIVERY_BATCH
                else "flagship"
            )

        zarr_path = str(_volume_root() / f"{episode_hash}.zarr")
        mp4_path = str(_volume_root() / f"{episode_hash}_video.mp4")

        error = ""
        if verify_volume:
            if not os.path.exists(zarr_path):
                missing_from_volume += 1
                error = f"zarr not found in volume at {zarr_path}"
            elif not os.path.exists(mp4_path):
                # MP4 missing isn't fatal — keep zarr path, blank mp4 path
                mp4_path = ""

        records.append(
            _build_sql_row_from_mongo(
                episode_hash=episode_hash,
                mongo_doc=mongo_doc,
                task_type=task_type,
                task_description=task_descriptions.get(episode_hash, ""),
                zarr_path=zarr_path,
                mp4_path=mp4_path,
                error=error,
            )
        )

    if missing_from_mongo:
        logger.warning(
            f"{missing_from_mongo} episode IDs not found in MongoDB — their rows "
            f"will be written with empty operator/task/scene/etc."
        )
    if verify_volume and missing_from_volume:
        logger.warning(
            f"{missing_from_volume} episodes missing zarr files on the volume — "
            f"their rows will carry zarr_processing_error='zarr not found in volume…'"
        )

    # ---- 4. Upsert ----
    engine = _sql_engine_from_pg_env()
    try:
        sql_ok, sql_err = _upsert_episode_records(engine, records)
    finally:
        engine.dispose()

    logger.info("=" * 60)
    logger.info(
        f"SQL-ONLY '{subset_name}' DONE: upserted {sql_ok} ok, {sql_err} failed "
        f"({missing_from_volume} missing zarr, {missing_from_mongo} missing mongo)"
    )
    logger.info("=" * 60)

    return {
        "total": len(records),
        "sql_ok": sql_ok,
        "sql_errors": sql_err,
        "missing_from_volume": missing_from_volume,
        "missing_from_mongo": missing_from_mongo,
        "subset_name": subset_name,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
def _load_episode_ids_file(path: str, max_episodes: int = 0) -> list[str]:
    """
    Load episode hashes from a file.

    Accepts:
      - Plain text: one hash per line.
      - JSON: a list of strings, or a list of objects each containing an
        ``episode_hash`` / ``_id`` / ``id`` key.

    ``max_episodes`` caps the result when > 0.
    """
    text = Path(path).read_text().strip()
    if not text:
        return []

    ids: list[str] = []
    if path.endswith(".json") or text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.append(item.strip())
                elif isinstance(item, dict):
                    key = next(
                        (k for k in ("episode_id", "episode_hash", "_id", "id") if k in item), None
                    )
                    if key:
                        ids.append(str(item[key]).strip())
        elif isinstance(data, dict):
            # {"episodes": [...]} wrapper format
            if "episodes" in data and isinstance(data["episodes"], list):
                for item in data["episodes"]:
                    if isinstance(item, str):
                        ids.append(item.strip())
                    elif isinstance(item, dict):
                        key = next(
                            (k for k in ("episode_id", "episode_hash", "_id", "id") if k in item), None
                        )
                        if key:
                            ids.append(str(item[key]).strip())
            else:
                # single object — treat as one episode
                key = next(
                    (k for k in ("episode_id", "episode_hash", "_id", "id") if k in data), None
                )
                if key:
                    ids.append(str(data[key]).strip())
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip()]

    ids = [i for i in ids if i]
    if max_episodes > 0:
        ids = ids[:max_episodes]
    return ids


@app.local_entrypoint()
def main(
    subset_name: str,
    episode_ids_file: str = "",
    mongo_filter_json: str = "",
    mongo_filter_file: str = "",
    limit: int = 0,
    dry_run: bool = False,
    episode_hash: str = "",
    force_task_type: str = "",
    skip_sql: bool = False,
    sql_only: bool = False,
    no_verify_volume: bool = False,
    max_episodes: int = 0,
):
    """
    Dispatch a subset conversion run to Modal.

    Outputs land in Modal volume ``egoverse-zarr-data`` at
    ``processed_v3/<subset_name>/{task_type}/<episode_hash>.zarr``.

    Args:
        subset_name: Required. Routing label for the volume path.
        episode_ids_file: Path to a text file with one episode hash per line.
        mongo_filter_json: MongoDB extended-JSON filter applied to the
            ``episodes`` collection. Matches are unioned with ``episode_ids_file``.
        mongo_filter_file: Path to a file containing the same JSON.
        limit: Max episodes to process (0 = all matched).
        dry_run: Process only 3 episodes.
        episode_hash: Convert a single episode (skips the batch path).
        force_task_type: Override classification ("flagship" or "freeform").
        skip_sql: Don't upsert rows into ``app.episodes`` after conversion.
            When False (default), if ``PG_HOST``/``PG_USER``/``PG_PASSWORD`` are
            present in the orchestrator's secret, one row is upserted per
            episode with columns: episode_hash, operator, task, embodiment,
            robot_name, num_frames, task_description, scene, objects,
            zarr_processed_path, zarr_mp4_path, zarr_processing_error,
            is_deleted, data_type. When the PG_* vars are absent, the step is
            skipped with a warning.
        sql_only: Skip the conversion entirely and only refresh
            ``app.episodes`` rows for the resolved subset. Pulls metadata from
            MongoDB, classifies task_type, computes the expected volume path
            for each episode, optionally verifies the file exists on the
            volume, then upserts.
        no_verify_volume: Only meaningful with ``sql_only``. When True, skips
            the on-volume existence check and writes paths verbatim — useful if
            you trust the prior conversion's outputs.
    """
    _validate_subset_name(subset_name)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    # ---- SQL-only mode: no conversion, just refresh app.episodes rows ----
    if sql_only:
        ids_csv = ""
        if episode_ids_file:
            ids = _load_episode_ids_file(episode_ids_file, max_episodes=max_episodes)
            ids_csv = ",".join(ids)
            logger.info(f"Read {len(ids)} episode IDs from {episode_ids_file}")
        if episode_hash:
            ids_csv = (ids_csv + "," + episode_hash).strip(",") if ids_csv else episode_hash

        filter_json = mongo_filter_json
        if mongo_filter_file and not filter_json:
            filter_json = Path(mongo_filter_file).read_text().strip()
            logger.info(f"Loaded Mongo filter from {mongo_filter_file}")

        logger.info(
            f"SQL-ONLY MODE: subset='{subset_name}', verify_volume="
            f"{not no_verify_volume}. No R2 reads, no conversion — only "
            f"app.episodes will be touched."
        )
        result = sql_only_batch.remote(
            episode_ids_csv=ids_csv,
            subset_name=subset_name,
            mongo_filter_json=filter_json,
            force_task_type=force_task_type,
            limit=limit,
            verify_volume=not no_verify_volume,
        )
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
        return result

    # ---- Single-episode mode ----
    # Route through the remote orchestrator so MongoDB classification runs on
    # Modal (where the secret is available) instead of the local entrypoint.
    if episode_hash:
        logger.info(f"Single-episode mode: {episode_hash} (subset={subset_name})")
        result = orchestrate_subset_batch.remote(
            episode_ids_csv=episode_hash,
            subset_name=subset_name,
            mongo_filter_json="",
            limit=1,
            dry_run=False,
            force_task_type=force_task_type,
            skip_sql=skip_sql,
        )
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
        return result

    # ---- Batch mode ----
    ids_csv = ""
    if episode_ids_file:
        ids = _load_episode_ids_file(episode_ids_file, max_episodes=max_episodes)
        ids_csv = ",".join(ids)
        logger.info(f"Read {len(ids)} episode IDs from {episode_ids_file}")

    filter_json = mongo_filter_json
    if mongo_filter_file and not filter_json:
        filter_json = Path(mongo_filter_file).read_text().strip()
        logger.info(f"Loaded Mongo filter from {mongo_filter_file}")

    logger.info(
        f"SUBSET MODE: subset_name='{subset_name}'. Outputs -> Modal volume "
        f"'{EGOVERSE_ZARR_VOLUME_NAME}' under processed_v3/{subset_name}/"
    )
    logger.info("Dispatching to remote orchestrator on Modal...")

    result = orchestrate_subset_batch.remote(
        episode_ids_csv=ids_csv,
        subset_name=subset_name,
        mongo_filter_json=filter_json,
        limit=limit,
        dry_run=dry_run,
        force_task_type=force_task_type,
        skip_sql=skip_sql,
    )
    logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
    return result


# ---------------------------------------------------------------------------
# Local test (no Modal, no volume — writes to a local output directory)
# ---------------------------------------------------------------------------

def local_test(
    subset_name: str,
    episode_hash: str,
    output_dir: str = "",
    force_task_type: str = "",
):
    """
    Convert a single episode locally for debugging.

    Outputs land at ``<output_dir>/<subset_name>/<task_type>/<episode_hash>.{zarr,mp4}``
    to mirror the Modal volume layout.
    """
    _validate_subset_name(subset_name)

    if not episode_hash:
        raise ValueError("local_test requires --episode-hash")

    import sys
    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[4] / "g_delivery" / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.info(f"Loaded env from {env_path}")
    except ImportError:
        pass

    if not os.environ.get("MONGODB_URI"):
        raise RuntimeError("Missing MONGODB_URI. Set it in the egoverse-mongodb secret.")
    if not (os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")):
        raise RuntimeError("Missing R2 access key. Set R2_ACCESS_KEY_ID in the egoverse-r2 secret.")
    if not (os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY")):
        raise RuntimeError("Missing R2 secret key. Set R2_SECRET_ACCESS_KEY in the egoverse-r2 secret.")
    if not (os.environ.get("R2_BUCKET") or os.environ.get("BUCKET")):
        raise RuntimeError("Missing bucket. Set BUCKET in the egoverse-r2 secret.")
    if not (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT")
        or os.environ.get("R2_ACCOUNT_ID")
    ):
        raise RuntimeError(
            "Missing R2 endpoint. Set AWS_ENDPOINT_URL_S3 in the egoverse-r2 secret."
        )

    db = _get_mongo_db()
    if force_task_type:
        task_type = force_task_type
    else:
        task_type = "freeform" if _is_freeform(db, episode_hash) else "flagship"
    logger.info(f"Local test: episode={episode_hash}, type={task_type}, subset={subset_name}")

    if not output_dir:
        output_dir = str(Path(__file__).resolve().parents[3] / "test_data" / "zarr_test")
    output_path = Path(output_dir) / subset_name / task_type
    output_path.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix=f"zarr_local_{episode_hash}_")
    try:
        result = _prepare_episode(episode_hash, task_type, tmp_dir)

        zarr_dest = output_path / f"{episode_hash}.zarr"
        if zarr_dest.exists():
            shutil.rmtree(zarr_dest)
        shutil.copytree(result["zarr_dir"], zarr_dest)
        logger.info(f"Zarr output: {zarr_dest}")

        if result.get("mp4_path") and os.path.exists(result["mp4_path"]):
            mp4_dest = output_path / f"{episode_hash}.mp4"
            shutil.copy2(result["mp4_path"], mp4_dest)
            logger.info(f"MP4 output: {mp4_dest}")

        logger.info("=" * 60)
        logger.info(f"Episode:      {episode_hash}")
        logger.info(f"Type:         {task_type}")
        logger.info(f"Annotations:  {result['num_annotations']} segments")
        logger.info(f"Task:         {result.get('task_description', '')}")
        logger.info(f"Output:       {zarr_dest}")
        logger.info("=" * 60)

        try:
            import zarr
            store = zarr.open(str(zarr_dest), mode="r")
            logger.info(f"Zarr arrays: {list(store.keys())}")
            logger.info(f"Zarr attrs:  {dict(store.attrs)}")
        except Exception as e:
            logger.warning(f"Could not inspect zarr: {e}")

        return result

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mecka to Zarr (local test)")
    parser.add_argument("--local-test", action="store_true", help="Run local test")
    parser.add_argument("--subset-name", type=str, required=True,
                        help="Subset label (required). Allowed chars: [a-zA-Z0-9_.-].")
    parser.add_argument("--episode-hash", type=str, required=True,
                        help="Episode hash to convert.")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Local output directory (defaults to <repo>/test_data/zarr_test).")
    parser.add_argument("--force-task-type", type=str, default="",
                        choices=["", "flagship", "freeform"],
                        help="Override task_type classification.")
    args = parser.parse_args()

    if args.local_test:
        local_test(
            subset_name=args.subset_name,
            episode_hash=args.episode_hash,
            output_dir=args.output_dir,
            force_task_type=args.force_task_type,
        )
    else:
        print("Use 'modal run' for batch processing, or --local-test for a local single-episode run.")
