"""
Scale API interactions for robot (ALOHA-style) task data.

Handles fetching task metadata, downloading MCAP files, annotation JSONs,
and MP4 video streams from Scale's platform for robot teleoperation tasks.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from scaleapi import ScaleClient

API_KEY = os.environ.get("SCALE_API_KEY", "")
if not API_KEY:
    raise ValueError("SCALE_API_KEY environment variable must be set")

client = ScaleClient(API_KEY)

CAMERA_SENSOR_MAP = {
    "high-rgb": "cam_high",
    "low-rgb": "cam_low",
    "left-rgb": "cam_left_wrist",
    "right-rgb": "cam_right_wrist",
    "high": "cam_high",
    "low": "cam_low",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}


def get_robot_task_response(task_id: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata for a robot teleoperation task from the Scale API.

    Returns a flat dict with URLs for annotations, MCAP, and per-camera MP4s,
    plus task-level metadata (demonstration label, episode info, etc.).
    """
    try:
        task = client.get_task(task_id)
        resp = task.response
        meta = task.metadata or {}

        full_rec = resp.get("full_recording") or resp.get("fullRecording", {})
        if not full_rec:
            print(f"[{task_id}] No full_recording in response")
            return None

        result: Dict[str, Any] = {
            "annotations_url": resp["annotations"]["url"],
            "mcap_url": full_rec.get("mcap_url") or full_rec.get("mcapUrl", ""),
            "task_id": task_id,
        }

        # Task-level metadata
        episode = meta.get("episode", {})
        if isinstance(episode, dict):
            result["demonstration"] = episode.get("Demonstration", "")
            result["episode_id"] = episode.get("Episode ID", "")
            result["objects"] = episode.get("Objects", "")
        else:
            result["demonstration"] = str(episode) if episode else ""

        if hasattr(task, "as_dict"):
            task_data = task.as_dict()
        else:
            task_data = task.__dict__
        result["customer_id"] = task_data.get("customerId", "")
        result["project"] = task_data.get("project", "")

        video_urls = full_rec.get("video_urls") or full_rec.get("videoUrls", [])
        for video in video_urls:
            sensor_id = video.get("sensor_id", "")
            cam_name = CAMERA_SENSOR_MAP.get(sensor_id)
            if cam_name:
                result[cam_name] = video["rgb_url"]

        return result

    except Exception as e:
        print(f"Error retrieving robot task {task_id}: {e}")
        return None


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> str:
    """Download a file in streaming chunks. Skips if file already exists."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    return output_path


def download_robot_task_files(
    download_dir: str,
    task_response: Dict[str, Any],
) -> Dict[str, str]:
    """Download MCAP, annotations, and all camera MP4s concurrently.

    Returns dict mapping logical names to local file paths.
    """
    task_id = task_response["task_id"]
    task_dir = os.path.join(download_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)

    downloads: List[tuple] = []
    local_paths: Dict[str, str] = {}

    ann_path = os.path.join(task_dir, "annotations.json")
    local_paths["annotations"] = ann_path
    downloads.append((task_response["annotations_url"], ann_path, "annotations"))

    if task_response.get("mcap_url"):
        mcap_path = os.path.join(task_dir, "recording.mcap")
        local_paths["mcap"] = mcap_path
        downloads.append((task_response["mcap_url"], mcap_path, "mcap"))

    for cam_name in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
        url = task_response.get(cam_name)
        if url:
            vid_path = os.path.join(task_dir, f"{cam_name}.mp4")
            local_paths[cam_name] = vid_path
            downloads.append((url, vid_path, cam_name))

    with ThreadPoolExecutor(max_workers=len(downloads)) as pool:
        futures = {
            pool.submit(download_file, url, path): name
            for url, path, name in downloads
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{task_id}] Error downloading {name}: {e}")
                local_paths.pop(name, None)

    return local_paths


def load_annotation_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load an annotation JSON file, stripping trailing null bytes."""
    try:
        with open(file_path, "r") as f:
            data = f.read().rstrip("\x00")
            return json.loads(data)
    except Exception:
        return None
