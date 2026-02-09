"""
SFS to Egoverse Utilities

Scale API interactions, file downloading, and SFS data loading.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import boto3
import numpy as np
import requests
import scipy.interpolate
from requests.auth import HTTPBasicAuth
from scaleapi import ScaleClient
from scale_sensor_fusion_io.loaders import SFSLoader
from sqlalchemy import MetaData, Table, create_engine, insert, inspect
from sqlalchemy.exc import IntegrityError


# Scale API Configuration
API_KEY = os.environ.get("SCALE_API_KEY", "")
if not API_KEY:
    raise ValueError("SCALE_API_KEY environment variable must be set")

client = ScaleClient(API_KEY)
auth = HTTPBasicAuth(API_KEY, '')


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


def interpolate_arr(v, seq_length):
    """
    v: (B, T, D)
    seq_length: int
    """
    assert len(v.shape) == 3
    if v.shape[1] == seq_length:
        return

    interpolated = []
    for i in range(v.shape[0]):
        index = v[i]

        interp = scipy.interpolate.interp1d(
            np.linspace(0, 1, index.shape[0]), index, axis=0
        )
        interpolated.append(interp(np.linspace(0, 1, seq_length)))

    return np.array(interpolated)


def interpolate_arr_euler(v: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Interpolate 6DoF poses (translation + Euler angles in radians),
    optionally with a 7th gripper dimension, along the time axis.

    v: (B, T, 6) or (B, T, 7)
        [x, y, z, yaw, pitch, roll, (optional) gripper]
    """
    assert (
        v.ndim == 3 and v.shape[2] in (6, 7)
    ), "Input v must be of shape (B, T, 6) or (B, T, 7)"
    B, T, D = v.shape

    new_time = np.linspace(0, 1, seq_length)
    old_time = np.linspace(0, 1, T)

    outputs = []

    for i in range(B):
        seq = v[i]  # (T, D)

        if np.any(seq >= 1e8):
            outputs.append(np.full((seq_length, D), 1e9))
            continue

        trans_seq = seq[:, :3]      # x, y, z
        rot_seq = seq[:, 3:6]       # yaw, pitch, roll

        # Avoid discontinuities in angle interpolation
        rot_seq_unwrapped = np.unwrap(rot_seq, axis=0)

        trans_interp_func = scipy.interpolate.interp1d(
            old_time, trans_seq, axis=0, kind="linear"
        )
        rot_interp_func = scipy.interpolate.interp1d(
            old_time, rot_seq_unwrapped, axis=0, kind="linear"
        )

        trans_interp = trans_interp_func(new_time)  # (seq_length, 3)
        rot_interp = rot_interp_func(new_time)      # (seq_length, 3)

        # Wrap back to [-pi, pi)
        rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

        if D == 6:
            out_seq = np.concatenate([trans_interp, rot_interp], axis=-1)
        else:
            grip_seq = seq[:, 6:7]  # (T, 1)
            grip_interp_func = scipy.interpolate.interp1d(
                old_time, grip_seq, axis=0, kind="linear"
            )
            grip_interp = grip_interp_func(new_time)  # (seq_length, 1)
            out_seq = np.concatenate(
                [trans_interp, rot_interp, grip_interp], axis=-1
            )

        outputs.append(out_seq)

    return np.stack(outputs, axis=0)  # (B, seq_length, D)



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


def get_simple_response_dict_egocentric(task_id: str) -> Optional[Dict[str, Any]]:
    """Get URLs for annotations, SFS, and video streams from a Scale task.
    
    Also returns task metadata like customerId for SQL registration.
    """
    try:
        task = client.get_task(task_id)
        resp = task.response
        
        # Get task dict for metadata
        if hasattr(task, 'as_dict'):
            task_data = task.as_dict()
        else:
            task_data = task.__dict__

        response_dict = {
            "annotations_url": resp["annotations"]["url"],
            "sfs_url": resp["full_recording"]["sfs_url"],
            # Task metadata for SQL
            "customer_id": task_data.get("customerId", ""),
            "project": task_data.get("project", ""),
            "batch_id": task_data.get("batchId", ""),
        }
        
        for video in resp["full_recording"]["video_urls"]:
            if video["sensor_id"] == "left":
                response_dict["left_rectified"] = video["rgb_url"]
            else:
                response_dict["right_rectified"] = video["rgb_url"]
    
        return response_dict
    
    except Exception as e:
        print(f"Error retrieving task {task_id}: {e}")
        return None

    
def _episodes_table(engine):
    md = MetaData()
    return Table("episodes", md, autoload_with=engine, schema="app")


def download_file_in_chunks(url: str, output_path: str, chunk_size: int = 8192) -> str:
    """Download a file in chunks."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    
    return output_path


def download_from_simple_response_dict(
    task_output_path: str,
    simple_response_dict: Dict[str, str],
    verbose: bool = False
) -> Dict[str, str]:
    """Download all files from a response dictionary. Returns local paths.
    
    Only processes keys ending with '_url' or 'rectified' (actual download URLs).
    Skips metadata fields like customer_id, project, batch_id.
    """
    local_path_dict = {}
    
    # Keys that are actual URLs to download
    url_keys = {'annotations_url', 'sfs_url', 'left_rectified', 'right_rectified'}
    
    for key, url in simple_response_dict.items():
        # Skip non-URL metadata fields
        if key not in url_keys:
            continue
            
        parsed = urlparse(url)
        file_extension = Path(parsed.path).suffix
        key_cleaned = key.replace('_url', '')
        local_file_path = os.path.join(task_output_path, key_cleaned + file_extension)
        local_path_dict[key_cleaned] = local_file_path

        if os.path.exists(local_file_path):
            continue
        
        if verbose:
            print(f"Downloading: {key_cleaned}")
        try:
            download_file_in_chunks(url, local_file_path)
        except Exception as e:
            print(f"Error downloading {key}: {e}")
    
    return local_path_dict


def load_scene(file_path: str) -> Optional[Dict[str, Any]]:
    """Load an SFS file."""
    if not os.path.exists(file_path):
        return None

    try:
        loader = SFSLoader(file_path)
        return loader.load_unsafe()
    except Exception:
        return None


def load_annotation_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load an annotation JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = f.read().rstrip('\x00')
            return json.loads(data)
    except Exception:
        return None


def get_posepath(sfs_data: Dict[str, Any], sensor_id: str) -> Optional[Dict[str, Any]]:
    """Get pose path for a sensor."""
    for sensor in sfs_data.get("sensors", []):
        if sensor.get("id") == sensor_id:
            return sensor.get("poses")
    return None


def get_intrinsics(sfs_data: Dict[str, Any], sensor_id: str) -> Optional[Dict[str, float]]:
    """Get camera intrinsics for a sensor."""
    for sensor in sfs_data.get("sensors", []):
        if sensor.get("id") == sensor_id:
            return sensor.get("intrinsics")
    return None
