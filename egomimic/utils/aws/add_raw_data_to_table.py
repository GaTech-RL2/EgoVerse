import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import boto3
import cloudpathlib

from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    create_default_engine,
    episode_table_to_df,
)

# Runtime
SECRETS_ARN = os.environ["SECRETS_ARN"]

# Local testing
# SECRETS_ARN = "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:rds/appdb/appuser"
# python3 egomimic/utils/aws/add_raw_data_to_table.py s3://rldb/raw_v2


@dataclass
class rawAriaEpisode:
    episode_hash: str
    vrs_path: cloudpathlib.S3Path
    vrs_json_path: cloudpathlib.S3Path
    metadata_json_path: cloudpathlib.S3Path


@dataclass
class rawHdf5Episode:
    episode_hash: str
    hdf5_path: cloudpathlib.S3Path
    metadata_json_path: cloudpathlib.S3Path


def filter_raw_episodes(
    all_files: list[rawAriaEpisode | rawHdf5Episode], current_episodes: set[str]
):
    """
    all_files: list of rawAriaEpisode or rawHdf5Episode
    current_episodes: set of episode_hashes in the database
    Returns: list of rawAriaEpisode or rawHdf5Episode that are not in the database
    """
    filtered_episodes = []
    for file in all_files:
        if file.episode_hash not in current_episodes:
            filtered_episodes.append(file)
    return filtered_episodes


def _get_raw_aria_episodes(all_files, all_file_uris, s3_client):
    all_aria_episodes = []
    for file in all_files:
        if "vrs" in str(file):
            vrs_path = file
            vrs_json_uri = str(file).replace(".vrs", ".json")
            metadata_json_uri = str(file).replace(".vrs", "_metadata.json")
            vrs_json_path = cloudpathlib.S3Path(vrs_json_uri, client=s3_client)
            metadata_json_path = cloudpathlib.S3Path(
                metadata_json_uri, client=s3_client
            )

            if (
                vrs_json_uri not in all_file_uris
                or metadata_json_uri not in all_file_uris
            ):
                print(
                    f"Skipping {file} because it doesn't have a vrs json or metadata json"
                )
                continue

            episode_hash = file.stem
            episode_hash = datetime.fromtimestamp(
                float(episode_hash) / 1000.0, timezone.utc
            ).strftime("%Y-%m-%d-%H-%M-%S-%f")

            raw_aria_episode = rawAriaEpisode(
                episode_hash=episode_hash,
                vrs_path=vrs_path,
                vrs_json_path=vrs_json_path,
                metadata_json_path=metadata_json_path,
            )
            all_aria_episodes.append(raw_aria_episode)

    return all_aria_episodes


def _get_raw_hdf5_episodes(all_files, all_file_uris, s3_client):
    all_hdf5_episodes = []
    for file in all_files:
        if "hdf5" in str(file):
            hdf5_path = file
            metadata_json_uri = str(file).replace(".hdf5", "_metadata.json")
            metadata_json_path = cloudpathlib.S3Path(
                metadata_json_uri, client=s3_client
            )
            if metadata_json_uri not in all_file_uris:
                print(f"Skipping {file} because it doesn't have a metadata json")
                continue

            episode_hash = file.stem
            episode_hash = datetime.fromtimestamp(
                float(episode_hash) / 1000.0, timezone.utc
            ).strftime("%Y-%m-%d-%H-%M-%S-%f")

            raw_hdf5_episode = rawHdf5Episode(
                episode_hash=episode_hash,
                hdf5_path=hdf5_path,
                metadata_json_path=metadata_json_path,
            )
            all_hdf5_episodes.append(raw_hdf5_episode)
    return all_hdf5_episodes


def _add_raw_episode_to_table(
    raw_episodes: list[rawAriaEpisode | rawHdf5Episode], s3_client
):
    engine = create_default_engine()

    for raw_episode in raw_episodes:
        metadata_uri = str(raw_episode.metadata_json_path)
        metadata_path = cloudpathlib.S3Path(metadata_uri, client=s3_client)
        metadata = json.load(metadata_path.open())

        episode = TableRow(
            episode_hash=raw_episode.episode_hash,
            operator=metadata["operator"],
            lab=metadata["lab"],
            task=metadata["task"],
            embodiment=metadata["embodiment"],
            robot_name=metadata.get("robot_name", metadata["embodiment"]),
            task_description=metadata.get("task_description", ""),
            scene=metadata["scene"],
            objects=metadata["objects"],
            processed_path="",
            mp4_path="",
        )

        add_episode(engine, episode)


def main(raw_v2_path_arg: str, endpoint_url: str | None = None):
    r2_access_key_id = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    r2_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    r2_session_token = os.environ.get("R2_SESSION_TOKEN") or os.environ.get(
        "AWS_SESSION_TOKEN"
    )

    if endpoint_url:
        # R2 requires an S3 region of "auto"/provider-specific aliases.
        # Keep AWS_DEFAULT_REGION for Secrets Manager, and override only the S3 client session.
        s3_boto3_session = boto3.session.Session(
            region_name="auto",
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            aws_session_token=r2_session_token,
        )
        s3_client = cloudpathlib.S3Client(
            endpoint_url=endpoint_url,
            boto3_session=s3_boto3_session,
        )
        # Prevent R2 keys from being used against AWS APIs (e.g., Secrets Manager).
        for key in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
        ):
            os.environ.pop(key, None)
    else:
        s3_client = cloudpathlib.S3Client()

    raw_v2_uri = raw_v2_path_arg.strip()
    if "://" in raw_v2_uri and not raw_v2_uri.startswith("s3://"):
        raise ValueError(
            "raw_v2_path must use s3:// scheme when a URI scheme is provided."
        )
    if not raw_v2_uri.startswith("s3://"):
        raw_v2_uri = "s3://" + raw_v2_uri.lstrip("/")
    if not raw_v2_uri.endswith("/"):
        raw_v2_uri += "/"

    raw_v2_path = cloudpathlib.S3Path(raw_v2_uri, client=s3_client)

    # List all files under raw_v2 (recursively)
    all_files = []
    for dirpath, dirnames, filenames in raw_v2_path.walk():
        # Calculate depth relative to raw_v2_path
        rel = str(dirpath.relative_to(raw_v2_path))
        # rel == '.' for root, otherwise by splitting for subdirectories
        depth = 0 if rel == "." else rel.count("/") + 1
        if depth > 1:
            # Prevent descending further by clearing dirnames
            dirnames[:] = []
            continue
        for fname in filenames:
            all_files.append(dirpath / fname)
    all_file_uris = {str(path) for path in all_files}

    engine = create_default_engine()
    episodes_data = episode_table_to_df(engine)
    current_episodes = set(episodes_data["episode_hash"])

    raw_aria_episodes = _get_raw_aria_episodes(all_files, all_file_uris, s3_client)
    raw_aria_episodes = filter_raw_episodes(raw_aria_episodes, current_episodes)
    raw_hdf5_episodes = _get_raw_hdf5_episodes(all_files, all_file_uris, s3_client)
    raw_hdf5_episodes = filter_raw_episodes(raw_hdf5_episodes, current_episodes)

    print(f"Raw Aria episodes: {raw_aria_episodes}")
    print(f"Raw HDF5 episodes: {raw_hdf5_episodes}")

    _add_raw_episode_to_table(raw_aria_episodes, s3_client)
    _add_raw_episode_to_table(raw_hdf5_episodes, s3_client)

    result = {
        "aria_episodes_added": len(raw_aria_episodes),
        "hdf5_episodes_added": len(raw_hdf5_episodes),
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add raw episodes from object store raw_v2 path to SQL table."
    )
    parser.add_argument(
        "raw_v2_path",
        help="Object store path to raw_v2 root (e.g. s3://rldb/raw_v2 or rldb/raw_v2)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="Optional S3-compatible endpoint URL (for R2 or custom object store).",
    )
    args = parser.parse_args()
    main(args.raw_v2_path, endpoint_url=args.endpoint_url)
