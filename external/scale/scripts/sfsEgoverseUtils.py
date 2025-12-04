"""
SFS to Egoverse Utilities

Scale API interactions, file downloading, and SFS data loading.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import requests
from requests.auth import HTTPBasicAuth
from scaleapi import ScaleClient
from scale_sensor_fusion_io.loaders import SFSLoader


# Scale API Configuration
API_KEY = os.environ.get("SCALE_API_KEY", "")
if not API_KEY:
    raise ValueError("SCALE_API_KEY environment variable must be set")

client = ScaleClient(API_KEY)
auth = HTTPBasicAuth(API_KEY, '')


def get_simple_response_dict_egocentric(task_id: str) -> Optional[Dict[str, str]]:
    """Get URLs for annotations, SFS, and video streams from a Scale task."""
    try:
        task = client.get_task(task_id)
        resp = task.response

        response_dict = {
            "annotations_url": resp["annotations"]["url"],
            "sfs_url": resp["full_recording"]["sfs_url"],
        }

        for video in resp["full_recording"]["video_urls"]:
            sensor_id = video["sensor_id"]
            for key, value in video.items():
                if key != "sensor_id":
                    response_dict[f"{sensor_id}_{key}"] = value
        
        return response_dict
    except Exception as e:
        print(f"Error retrieving task {task_id}: {e}")
        return None


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
    simple_response_dict: Dict[str, str]
) -> Dict[str, str]:
    """Download all files from a response dictionary. Returns local paths."""
    local_path_dict = {}
    
    for key, url in simple_response_dict.items():
        parsed = urlparse(url)
        file_extension = Path(parsed.path).suffix
        key_cleaned = key.replace('_url', '')
        local_file_path = os.path.join(task_output_path, key_cleaned + file_extension)
        local_path_dict[key_cleaned] = local_file_path

        if os.path.exists(local_file_path):
            print(f"Exists: {local_file_path}")
            continue
        
        print(f"Downloading: {key_cleaned}")
        try:
            download_file_in_chunks(url, local_file_path)
        except Exception as e:
            print(f"Error downloading {key}: {e}")
    
    return local_path_dict


def load_scene(file_path: str) -> Optional[Dict[str, Any]]:
    """Load an SFS file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        loader = SFSLoader(file_path)
        return loader.load_unsafe()
    except Exception as e:
        print(f"Error loading SFS: {e}")
        return None


def load_annotation_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load an annotation JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = f.read().rstrip('\x00')
            return json.loads(data)
    except Exception as e:
        print(f"Error loading annotations: {e}")
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
