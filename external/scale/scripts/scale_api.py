"""
Scale API interactions, file downloading, and SFS data loading.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from requests.auth import HTTPBasicAuth
from scaleapi import ScaleClient
from scale_sensor_fusion_io.loaders import SFSLoader

# ---------------------------------------------------------------------------
# Scale API configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("SCALE_API_KEY", "")
if not API_KEY:
    raise ValueError("SCALE_API_KEY environment variable must be set")

client = ScaleClient(API_KEY)
auth = HTTPBasicAuth(API_KEY, "")


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------


def get_simple_response_dict_egocentric(task_id: str) -> Optional[Dict[str, Any]]:
    """Get URLs for annotations, SFS, and video streams from a Scale task.

    Also returns task metadata like customerId for SQL registration.
    """
    try:
        task = client.get_task(task_id)
        resp = task.response

        if hasattr(task, "as_dict"):
            task_data = task.as_dict()
        else:
            task_data = task.__dict__

        response_dict = {
            "annotations_url": resp["annotations"]["url"],
            "sfs_url": resp["full_recording"]["sfs_url"],
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


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------


def download_file_in_chunks(url: str, output_path: str, chunk_size: int = 8192) -> str:
    """Download a file in streaming chunks."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)

    return output_path


def download_from_simple_response_dict(
    task_output_path: str,
    simple_response_dict: Dict[str, str],
    verbose: bool = False,
) -> Dict[str, str]:
    """Download all files from a response dictionary concurrently. Returns local paths."""
    local_path_dict = {}
    to_download: list[tuple[str, str, str]] = []

    url_keys = {"annotations_url", "sfs_url", "left_rectified", "right_rectified"}

    for key, url in simple_response_dict.items():
        if key not in url_keys:
            continue

        parsed = urlparse(url)
        file_extension = Path(parsed.path).suffix
        key_cleaned = key.replace("_url", "")
        local_file_path = os.path.join(task_output_path, key_cleaned + file_extension)
        local_path_dict[key_cleaned] = local_file_path

        if os.path.exists(local_file_path):
            continue

        if verbose:
            print(f"Queued download: {key_cleaned}")
        to_download.append((url, local_file_path, key_cleaned))

    if to_download:
        with ThreadPoolExecutor(max_workers=len(to_download)) as pool:
            futures = {
                pool.submit(download_file_in_chunks, url, path): name
                for url, path, name in to_download
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading {name}: {e}")

    return local_path_dict


# ---------------------------------------------------------------------------
# SFS / annotation file loading
# ---------------------------------------------------------------------------


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
        with open(file_path, "r") as f:
            data = f.read().rstrip("\x00")
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
