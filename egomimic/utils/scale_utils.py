import json
import os
import time
from typing import Any

import pandas as pd
import requests
from scaleapi import ScaleClient

# Column name in the Scale annotation CSV that stores episode hashes
_SEQUENCE_ID_COL = "SEQUENCE_ID"
_STATUS_COL = "STATUS"
_ID_COL = "_ID"

REQUEST_TIMEOUT_S = 180  # generous; api.scale.com has been flaky
_MAX_RETRIES = 5
_RETRY_BACKOFF_S = 5.0


def _requests_get_with_retry(*args, **kwargs):
    """``requests.get`` with retries on transient ReadTimeout / ConnectionError.

    api.scale.com routinely 60s-times-out under load. Each task-list page hits
    the API once, so a long-running launch will trip a retry several times.
    """
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return requests.get(*args, **kwargs)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt == _MAX_RETRIES:
                raise
            wait = _RETRY_BACKOFF_S * attempt
            print(
                f"[scale_utils] {type(e).__name__} on attempt {attempt}/{_MAX_RETRIES}; "
                f"retrying in {wait:.1f}s"
            )
            time.sleep(wait)
    raise last_exc  # pragma: no cover


def get_tasks(project_name: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch all completed tasks for a project."""
    headers = {"accept": "application/json"}
    base_url = "https://api.scale.com/v1/tasks"

    next_token = None
    tasks: list[dict[str, Any]] = []

    while True:
        params = {  # TODO: fetch all tasks included non completed tasks after scale explained how this is done
            "project": project_name,
            "include_attachment_url": "true",
            "limit": 100,
        }
        if next_token:
            params["next_token"] = next_token

        response = _requests_get_with_retry(
            base_url,
            headers=headers,
            params=params,
            auth=(api_key, ""),
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        tasks.extend(data.get("docs", []))

        next_token = data.get("next_token")
        if not next_token:
            break

    return tasks


def get_episode_hash(task: dict[str, Any]) -> str:
    """Extract episode hash from task['params']['attachments'][0]."""
    attachment = task["params"]["attachments"][0]
    # Example:
    # s3://scale-sales-uploads/egoverse/2026-03-17-01-42-37-000000/2026-03-17-01-42-37-000000.mp4
    return attachment.rstrip("/").split("/")[-2]


def build_df_from_tasks(
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a lookup from episode_hash -> task."""

    df = pd.DataFrame(columns=["_ID", "STATUS", "S3_ATTACHMENT", "SEQUENCE_ID"])
    for task in tasks:
        attachments = task.get("params", {}).get("attachments", [])
        if not attachments:
            continue

        episode_hash = get_episode_hash(task)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "_ID": task["task_id"],
                            "STATUS": task["status"],
                            "S3_ATTACHMENT": attachments[0],
                            "SEQUENCE_ID": episode_hash,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return df


def download_scale_annotation(client: ScaleClient, tid: str, out_path: str):
    task = client.get_task(tid)
    url = task.response["annotations"]["url"]
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = json.loads(resp.text.rstrip("\x00"))
    path = os.path.join(out_path, f"{tid}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def get_completed_tasks(project_name: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch all completed tasks for a project."""
    headers = {"accept": "application/json"}
    base_url = "https://api.scale.com/v1/tasks"

    next_token = None
    tasks: list[dict[str, Any]] = []

    while True:
        params = {  # TODO: fetch all tasks included non completed tasks after scale explained how this is done
            "status": "completed",
            "project": project_name,
            "include_attachment_url": "true",
            "limit": 100,
        }
        if next_token:
            params["next_token"] = next_token

        response = _requests_get_with_retry(
            base_url,
            headers=headers,
            params=params,
            auth=(api_key, ""),
            timeout=REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()

        tasks.extend(data.get("docs", []))

        next_token = data.get("next_token")
        if not next_token:
            break

    return tasks


# ---------------------------------------------------------------------------
# Annotation CSV helpers
# ---------------------------------------------------------------------------

def load_scale_annotation_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_available_hashes(df: pd.DataFrame) -> list[str]:
    """Return episode hashes that have a completed Scale annotation."""
    return df[df[_STATUS_COL] == "completed"][_SEQUENCE_ID_COL].unique().tolist()


def get_tid_to_episode_hash(df: pd.DataFrame, tid: str) -> str:
    """Return the episode hash for a given Scale task ID."""
    return df[df[_ID_COL] == tid][_SEQUENCE_ID_COL].values[0]


def get_episode_hash_to_tid(df: pd.DataFrame, episode_hash: str) -> str:
    """Return the Scale task ID for a given episode hash."""
    return df[df[_SEQUENCE_ID_COL] == episode_hash][_ID_COL].values[0]


def filter_df_annotations(
    episode_df: pd.DataFrame,
    project_name: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Filter a SQL episode DataFrame to rows that have a completed Scale annotation.

    Args:
        episode_df: DataFrame from episode_table_to_df(engine).
        project_name: Scale project name to fetch completed tasks from.
        api_key: Scale API key. Defaults to SCALE_API_KEY env var.

    Returns:
        Subset of episode_df whose episode_hash has a completed Scale annotation.
    """
    api_key = api_key or os.environ["SCALE_API_KEY"]
    tasks = get_completed_tasks(project_name, api_key)
    annotation_df = build_df_from_tasks(tasks)
    available = set(get_available_hashes(annotation_df))
    return episode_df[episode_df["episode_hash"].isin(available)].reset_index(drop=True)
