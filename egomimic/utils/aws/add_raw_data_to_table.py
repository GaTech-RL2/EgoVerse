import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import boto3
import cloudpathlib
import psycopg

from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    create_default_engine,
    episode_table_to_df,
)

# Lambda
SECRETS_ARN = os.environ["SECRETS_ARN"]
BUCKET = os.environ["BUCKET"]
KEY_PREFIX = os.environ.get("KEY_PREFIX", "")

# Local testing
# SECRETS_ARN = "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:rds/appdb/appuser"
# BUCKET = "rldb"
# KEY_PREFIX = ""

s3 = boto3.client("s3")
secrets = boto3.client("secretsmanager")


@dataclass
class rawAriaEpisode:
    episode_hash: str
    vrs_path: str
    vrs_json_path: str
    metadata_json_path: str


@dataclass
class rawHdf5Episode:
    episode_hash: str
    hdf5_path: str
    metadata_json_path: str


def _get_db_conn():
    sec = secrets.get_secret_value(SecretId=SECRETS_ARN)["SecretString"]
    cfg = json.loads(sec)
    return psycopg.connect(**cfg)


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


def _get_raw_aria_episodes(all_files):
    all_aria_episodes = []
    for file in all_files:
        if "vrs" in str(file):
            vrs_path = file
            vrs_json_path = cloudpathlib.S3Path(str(file).replace(".vrs", ".json"))
            metadata_json_path = cloudpathlib.S3Path(
                str(file).replace(".vrs", "_metadata.json")
            )

            if vrs_json_path not in all_files or metadata_json_path not in all_files:
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


def _get_raw_hdf5_episodes(all_files):
    all_hdf5_episodes = []
    for file in all_files:
        if "hdf5" in str(file):
            hdf5_path = file
            metadata_json_path = cloudpathlib.S3Path(
                str(file).replace(".hdf5", "_metadata.json")
            )
            if metadata_json_path not in all_files:
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


def _add_raw_episode_to_table(raw_episodes: list[rawAriaEpisode | rawHdf5Episode]):
    engine = create_default_engine()

    for raw_episode in raw_episodes:
        metadata = json.load(cloudpathlib.S3Path(raw_episode.metadata_json_path).open())

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


def lambda_handler(event, context):
    # Use environment variables for S3 path
    bucket = os.environ.get("BUCKET", "rldb")
    key_prefix = os.environ.get("KEY_PREFIX", "")
    s3_prefix = f"s3://{bucket}/"
    if key_prefix:
        s3_prefix += key_prefix
    raw_v2_path = cloudpathlib.S3Path(s3_prefix + "raw_v2/")

    # List all files under rldb/raw_v2 (recursively)
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

    engine = create_default_engine()
    episodes_data = episode_table_to_df(engine)
    current_episodes = set(episode["episode_hash"] for episode in episodes_data)

    raw_aria_episodes = _get_raw_aria_episodes(all_files)
    raw_aria_episodes = filter_raw_episodes(raw_aria_episodes, current_episodes)
    raw_hdf5_episodes = _get_raw_hdf5_episodes(all_files)
    raw_hdf5_episodes = filter_raw_episodes(raw_hdf5_episodes, current_episodes)

    print(f"Raw Aria episodes: {raw_aria_episodes}")
    print(f"Raw HDF5 episodes: {raw_hdf5_episodes}")

    _add_raw_episode_to_table(raw_aria_episodes)
    _add_raw_episode_to_table(raw_hdf5_episodes)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "aria_episodes_added": len(raw_aria_episodes),
                "hdf5_episodes_added": len(raw_hdf5_episodes),
            }
        ),
    }


# # Local testing
# if __name__ == "__main__":
#     lambda_handler({}, None)
