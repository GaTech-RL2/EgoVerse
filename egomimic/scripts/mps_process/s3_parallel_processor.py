#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Batch processor for .vrs files using aria_mps CLI.

Downloads VRS files in batches based on size threshold, processes them with
aria_mps, and uploads results back to S3. Uses boto3 for all S3 operations.

Environment Variables Required:
    MPS_USER: MPS username for authentication
    MPS_PASSWORD: MPS password for authentication

Usage:
    # Process all features (default)
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local

    # Process only specific features
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local \
        --features HAND_TRACKING SLAM

    # Pass --force flag to aria_mps
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local \
        --features HAND_TRACKING --force

    # Pass --retry-failed flag to aria_mps
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local \
        --retry-failed

    # Include failed recordings (don't filter them out)
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local \
        --include-failed-recordings

    # Use custom path to recordings_status.json
    python s3_parallel_processor.py --bucket my-bucket --local-dir /local \
        --recordings-status-path custom/path/recordings_status.json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# Version Constants for MPS Features
# ============================================================================
# These versions define what is considered "up to date" for each MPS feature.
# Update these when new MPS versions are released.

EXPECTED_VERSIONS = {
    "HAND_TRACKING": "3.1.1",
    "SLAM": "1.1.0",
    "EYE_GAZE": "3.1.0",
}


class FeatureStatus(Enum):
    """Status of an MPS feature for a recording."""

    OK = "OK"  # Version correct, status SUCCESS/WARNING
    MISSING = "MISSING"  # summary.json doesn't exist
    VERSION_MISMATCH = "VERSION_MISMATCH"  # Wrong version
    FAILED = "FAILED"  # Status is not SUCCESS/WARNING


@dataclass
class FeatureCheckResult:
    """Result of checking a single MPS feature."""

    feature: str
    status: FeatureStatus
    expected_version: str
    actual_version: Optional[str]
    details: Optional[str] = None


@dataclass
class RecordingStatus:
    """Complete status of all MPS features for a recording."""

    vrs_key: str
    features: List[FeatureCheckResult]

    @property
    def is_complete(self) -> bool:
        """True if all features have OK status."""
        return all(f.status == FeatureStatus.OK for f in self.features)

    @property
    def failed_features(self) -> List[str]:
        """List of feature names that are not OK."""
        return [f.feature for f in self.features if f.status != FeatureStatus.OK]


# Configure logging with timestamp
# Only enable DEBUG for our logger, not for boto3/urllib3/etc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ============================================================================
# SAFETY: S3 Delete Operations are FORBIDDEN
# ============================================================================
# This script must NEVER delete any .vrs files or folders from S3.
# All delete operations (shutil.rmtree) are LOCAL ONLY.
# If you need to add delete functionality, DO NOT add it to this script.
# ============================================================================


# ============================================================================
# Boto3 Backend for S3 Operations
# ============================================================================


class Boto3Backend:
    """S3 operations using boto3 with multipart transfers.

    SAFETY: This class intentionally has NO delete operations.
    VRS files and MPS folders on S3 must NEVER be deleted by this script.
    All cleanup operations are LOCAL only.
    """

    def __init__(self, bucket: str):
        import boto3
        from boto3.s3.transfer import TransferConfig

        self.bucket = bucket
        self.client = boto3.client("s3")
        self.config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True,
        )

    def list_vrs_files(self, prefix: str = "") -> List[str]:
        """List all .vrs files under the given S3 prefix."""
        vrs_files = []
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".vrs"):
                    vrs_files.append(key)

        return vrs_files

    def get_file_size_bytes(self, s3_key: str) -> int:
        """Get file size in bytes from S3."""
        response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
        return response["ContentLength"]

    def read_json(self, s3_key: str) -> Optional[dict]:
        """Read and parse a JSON file from S3. Returns None if not found."""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read().decode("utf-8")
            return json.loads(content)
        except self.client.exceptions.NoSuchKey:
            return None
        except (json.JSONDecodeError, Exception):
            return None

    def folder_exists(self, s3_prefix: str) -> bool:
        """Check if a folder (prefix) exists in S3."""
        prefix = s3_prefix.rstrip("/") + "/"
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=prefix, MaxKeys=1
        )
        return response.get("KeyCount", 0) > 0

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download a single file from S3 to local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(
            self.bucket, s3_key, str(local_path), Config=self.config
        )

    def download_folder(self, s3_prefix: str, local_path: Path) -> None:
        """Download a folder from S3 to local path.

        Skips directory marker objects (empty keys that represent folders).
        """
        local_path.mkdir(parents=True, exist_ok=True)
        prefix = s3_prefix.rstrip("/") + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_key = key[len(prefix) :]
                if not relative_key:
                    continue

                # Skip directory marker objects (keys ending with / or empty files
                # that match a directory name)
                if relative_key.endswith("/"):
                    continue

                # Check if this is a directory marker (a key without extension
                # where a "key/" prefix also exists)
                if "/" not in relative_key and obj.get("Size", 0) == 0:
                    # This might be a directory marker, skip it
                    continue

                local_file = local_path / relative_key
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Skip if local path is a directory (directory marker collision)
                if local_file.is_dir():
                    logger.info(f"      Skipping directory marker: {key}")
                    continue

                logger.info(f"      s3://{self.bucket}/{key} -> {local_file}")
                self.client.download_file(
                    self.bucket, key, str(local_file), Config=self.config
                )

    def upload_folder(self, local_path: Path, s3_prefix: str) -> None:
        """Upload a folder to S3."""
        prefix = s3_prefix.rstrip("/") + "/"

        for root, _, files in os.walk(local_path):
            for file in files:
                local_file = Path(root) / file
                relative_path = local_file.relative_to(local_path)
                s3_key = prefix + str(relative_path)
                self.client.upload_file(
                    str(local_file), self.bucket, s3_key, Config=self.config
                )


# ============================================================================
# Helper Functions
# ============================================================================
# NOTE: S3 delete operations are intentionally NOT implemented in this script.
# VRS files and MPS folders must NEVER be deleted.


def get_output_folder_name(vrs_filename: str) -> str:
    """Convert filename.vrs -> mps_filename_vrs"""
    name = vrs_filename.replace(".vrs", "")
    return f"mps_{name}_vrs"


def get_mps_key_from_vrs_key(vrs_key: str) -> str:
    """Convert VRS S3 key to MPS folder S3 key."""
    vrs_name = Path(vrs_key).name
    parent = str(Path(vrs_key).parent)
    mps_folder_name = get_output_folder_name(vrs_name)
    return f"{parent}/{mps_folder_name}" if parent != "." else mps_folder_name


def get_mps_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get MPS credentials from environment variables."""
    mps_user = os.environ.get("MPS_USER")
    mps_password = os.environ.get("MPS_PASSWORD")
    return mps_user, mps_password


# ============================================================================
# Processing Status Check Functions (S3 versions)
# ============================================================================


def all_statuses_success_or_warning(data: dict) -> bool:
    """Check if ALL 'status' fields in the dict are SUCCESS or WARNING.

    This handles multi-stage processing (like SLAM) where each stage
    has its own status field. Returns True only if every status found
    is either SUCCESS or WARNING.
    """
    found_status = False

    def check_recursive(obj):
        nonlocal found_status
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "status":
                    found_status = True
                    if value not in ("SUCCESS", "WARNING"):
                        return False
                elif isinstance(value, (dict, list)):
                    if not check_recursive(value):
                        return False
        elif isinstance(obj, list):
            for item in obj:
                if not check_recursive(item):
                    return False
        return True

    result = check_recursive(data)
    return found_status and result


def get_features_needed_s3(
    backend: Boto3Backend,
    mps_prefix: str,
    requested_features: Optional[List[str]] = None,
) -> List[str]:
    """Get list of features that need processing (S3 version).

    Reuses the detailed status functions to avoid duplicating version checks
    and status validation logic.

    Args:
        backend: S3 backend for reading status files
        mps_prefix: S3 prefix for MPS output folder
        requested_features: List of features to check. If None, checks all features.
    """
    all_features = ["HAND_TRACKING", "SLAM", "EYE_GAZE"]
    features_to_check = requested_features if requested_features else all_features

    features = []

    if "HAND_TRACKING" in features_to_check:
        if get_hand_tracking_status_s3(backend, mps_prefix).status != FeatureStatus.OK:
            features.append("HAND_TRACKING")
    if "SLAM" in features_to_check:
        if get_slam_status_s3(backend, mps_prefix).status != FeatureStatus.OK:
            features.append("SLAM")
    if "EYE_GAZE" in features_to_check:
        if get_eye_gaze_status_s3(backend, mps_prefix).status != FeatureStatus.OK:
            features.append("EYE_GAZE")

    return features


# ============================================================================
# Detailed Status Check Functions (for validation and reporting)
# ============================================================================


def get_hand_tracking_status_s3(
    backend: Boto3Backend, mps_prefix: str
) -> FeatureCheckResult:
    """Get detailed hand tracking status from S3."""
    expected_version = EXPECTED_VERSIONS["HAND_TRACKING"]
    summary_key = f"{mps_prefix}/hand_tracking/summary.json"
    summary = backend.read_json(summary_key)

    if summary is None:
        return FeatureCheckResult(
            feature="HAND_TRACKING",
            status=FeatureStatus.MISSING,
            expected_version=expected_version,
            actual_version=None,
            details="summary.json not found",
        )

    actual_version = summary.get("version")
    if actual_version != expected_version:
        return FeatureCheckResult(
            feature="HAND_TRACKING",
            status=FeatureStatus.VERSION_MISMATCH,
            expected_version=expected_version,
            actual_version=actual_version,
            details=f"Expected {expected_version}, got {actual_version}",
        )

    # Note: hand_tracking summary.json may not have a "status" field like SLAM does.
    # If present, check it; otherwise just check version.
    if "status" in summary and summary.get("status") not in ("SUCCESS", "WARNING"):
        return FeatureCheckResult(
            feature="HAND_TRACKING",
            status=FeatureStatus.FAILED,
            expected_version=expected_version,
            actual_version=actual_version,
            details=f"Status is {summary.get('status')}",
        )

    return FeatureCheckResult(
        feature="HAND_TRACKING",
        status=FeatureStatus.OK,
        expected_version=expected_version,
        actual_version=actual_version,
    )


def get_slam_status_s3(backend: Boto3Backend, mps_prefix: str) -> FeatureCheckResult:
    """Get detailed SLAM status from S3."""
    expected_version = EXPECTED_VERSIONS["SLAM"]
    summary_key = f"{mps_prefix}/slam/summary.json"
    summary = backend.read_json(summary_key)

    if summary is None:
        return FeatureCheckResult(
            feature="SLAM",
            status=FeatureStatus.MISSING,
            expected_version=expected_version,
            actual_version=None,
            details="summary.json not found",
        )

    actual_version = summary.get("version")
    if actual_version != expected_version:
        return FeatureCheckResult(
            feature="SLAM",
            status=FeatureStatus.VERSION_MISMATCH,
            expected_version=expected_version,
            actual_version=actual_version,
            details=f"Expected {expected_version}, got {actual_version}",
        )

    if not all_statuses_success_or_warning(summary):
        return FeatureCheckResult(
            feature="SLAM",
            status=FeatureStatus.FAILED,
            expected_version=expected_version,
            actual_version=actual_version,
            details="One or more stages not SUCCESS/WARNING",
        )

    return FeatureCheckResult(
        feature="SLAM",
        status=FeatureStatus.OK,
        expected_version=expected_version,
        actual_version=actual_version,
    )


def get_eye_gaze_status_s3(
    backend: Boto3Backend, mps_prefix: str
) -> FeatureCheckResult:
    """Get detailed eye gaze status from S3."""
    expected_version = EXPECTED_VERSIONS["EYE_GAZE"]
    summary_key = f"{mps_prefix}/eye_gaze/summary.json"
    summary = backend.read_json(summary_key)

    if summary is None:
        return FeatureCheckResult(
            feature="EYE_GAZE",
            status=FeatureStatus.MISSING,
            expected_version=expected_version,
            actual_version=None,
            details="summary.json not found",
        )

    actual_version = summary.get("version")
    if actual_version != expected_version:
        return FeatureCheckResult(
            feature="EYE_GAZE",
            status=FeatureStatus.VERSION_MISMATCH,
            expected_version=expected_version,
            actual_version=actual_version,
            details=f"Expected {expected_version}, got {actual_version}",
        )

    if not all_statuses_success_or_warning(summary):
        return FeatureCheckResult(
            feature="EYE_GAZE",
            status=FeatureStatus.FAILED,
            expected_version=expected_version,
            actual_version=actual_version,
            details="Status not SUCCESS/WARNING",
        )

    return FeatureCheckResult(
        feature="EYE_GAZE",
        status=FeatureStatus.OK,
        expected_version=expected_version,
        actual_version=actual_version,
    )


def get_recording_status_s3(
    vrs_key: str,
    backend: Boto3Backend,
    requested_features: Optional[List[str]] = None,
) -> RecordingStatus:
    """Get complete status of all MPS features for a recording from S3.

    Args:
        vrs_key: S3 key for the VRS file
        backend: S3 backend for reading status files
        requested_features: List of features to check. If None, checks all features.
    """
    mps_prefix = get_mps_key_from_vrs_key(vrs_key)
    all_features = ["HAND_TRACKING", "SLAM", "EYE_GAZE"]
    features_to_check = requested_features if requested_features else all_features

    features = []
    if "HAND_TRACKING" in features_to_check:
        features.append(get_hand_tracking_status_s3(backend, mps_prefix))
    if "SLAM" in features_to_check:
        features.append(get_slam_status_s3(backend, mps_prefix))
    if "EYE_GAZE" in features_to_check:
        features.append(get_eye_gaze_status_s3(backend, mps_prefix))

    return RecordingStatus(vrs_key=vrs_key, features=features)


# ============================================================================
# Parallel Filtering and Batching
# ============================================================================


def load_failed_recordings_from_s3(
    backend: Boto3Backend,
    s3_prefix: str,
    recordings_status_path: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """Load recordings_status.json from S3 and return a dict of vrs_key -> set of failed features.

    The JSON structure is:
    {
        "vrs_key": {
            "timestamp": "...",
            "HAND_TRACKING": { "status": "...", ... },
            "SLAM": { "status": "...", ... },
            ...
        },
        ...
    }

    Args:
        backend: S3 backend for reading files
        s3_prefix: S3 prefix (used if recordings_status_path is not provided)
        recordings_status_path: Custom S3 path to recordings_status.json. If provided,
                                this path is used directly instead of deriving from s3_prefix.

    Returns:
        Dict mapping vrs_key to set of feature names that failed for that recording.
    """
    if recordings_status_path:
        s3_key = recordings_status_path
    else:
        s3_key = (
            f"{s3_prefix.rstrip('/')}/recordings_status.json"
            if s3_prefix
            else "recordings_status.json"
        )

    logger.debug(f"Loading recordings_status from: s3://{backend.bucket}/{s3_key}")
    failed_recordings_data = backend.read_json(s3_key)
    if not failed_recordings_data:
        logger.debug("No recordings_status.json found or file is empty")
        return {}

    logger.debug(
        f"Loaded {len(failed_recordings_data)} entries from recordings_status.json"
    )

    result: Dict[str, Set[str]] = {}
    all_features = ["HAND_TRACKING", "SLAM", "EYE_GAZE"]

    for vrs_key, recording_data in failed_recordings_data.items():
        if not isinstance(recording_data, dict):
            logger.debug(f"Skipping invalid entry for {vrs_key}: not a dict")
            continue
        failed_features = set()
        for feature in all_features:
            if feature in recording_data:
                feature_data = recording_data[feature]
                if (
                    isinstance(feature_data, dict)
                    and feature_data.get("status") != "OK"
                ):
                    failed_features.add(feature)
        if failed_features:
            result[vrs_key] = failed_features

    logger.debug(f"Total failed recordings loaded: {len(result)}")
    return result


def filter_files_needing_processing(
    vrs_keys: List[str],
    backend: Boto3Backend,
    requested_features: Optional[List[str]] = None,
    failed_recordings: Optional[Dict[str, Set[str]]] = None,
    max_workers: int = 20,
) -> List[str]:
    """Filter VRS files to only those that need processing.

    Uses ThreadPoolExecutor to check multiple files in parallel.

    Logic (S3 is source of truth):
    1. Check S3 to see which features are missing/need processing
    2. For features that need processing, check recordings_status.json to see if they previously failed
    3. Skip features that are known to have failed (unless --include-failed-recordings is used)

    Args:
        vrs_keys: List of VRS file keys to check
        backend: S3 backend for reading status files
        requested_features: List of features to process. If None, processes all features.
        failed_recordings: Dict mapping vrs_key to set of failed features for that recording.
        max_workers: Number of parallel workers for checking files.
    """

    def check_file(vrs_key: str) -> Optional[str]:
        """Check if a single file needs processing."""
        filename = Path(vrs_key).name

        # Determine which features to check
        features_to_check = (
            requested_features
            if requested_features
            else ["HAND_TRACKING", "SLAM", "EYE_GAZE"]
        )

        # Step 1: Check S3 to see which features need processing (source of truth)
        mps_key = get_mps_key_from_vrs_key(vrs_key)
        features_needed_on_s3 = get_features_needed_s3(
            backend, mps_key, features_to_check
        )

        if not features_needed_on_s3:
            logger.debug(f"{filename}: all requested features already OK on S3")
            return None

        # Step 2: For features that need processing, check if they previously failed
        features_to_process = list(features_needed_on_s3)
        if failed_recordings and vrs_key in failed_recordings:
            failed_features = failed_recordings[vrs_key]
            # Filter out features that have previously failed
            features_to_process = [
                f for f in features_needed_on_s3 if f not in failed_features
            ]
            skipped_features = [
                f for f in features_needed_on_s3 if f in failed_features
            ]
            if skipped_features:
                logger.debug(
                    f"{filename}: needs {features_needed_on_s3} on S3, "
                    f"but skipping {skipped_features} (previously failed), "
                    f"will process {features_to_process}"
                )
        else:
            # Recording not in failed_recordings (either not processed before,
            # or all features were OK). Process all features that S3 needs.
            logger.debug(
                f"{filename}: needs {features_needed_on_s3} on S3, "
                f"not in failed_recordings or all previously OK"
            )

        if features_to_process:
            logger.debug(
                f"{filename}: needs processing for features {features_to_process}"
            )
            return vrs_key
        else:
            logger.debug(f"{filename}: all needed features previously failed, skipping")
            return None

    files_needing_work = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(check_file, key): key for key in vrs_keys}

        for future in as_completed(future_to_key):
            try:
                result = future.result()
                if result:
                    files_needing_work.append(result)
            except Exception as e:
                vrs_key = future_to_key[future]
                logger.warning(f"Error checking {vrs_key}: {e}")

    return files_needing_work


def get_file_sizes_parallel(
    vrs_keys: List[str],
    backend: Boto3Backend,
    max_workers: int = 20,
) -> Dict[str, float]:
    """Get file sizes for multiple VRS files in parallel.

    Returns dict mapping vrs_key -> size_in_gb.
    """

    def get_size(vrs_key: str) -> Tuple[str, float]:
        size_bytes = backend.get_file_size_bytes(vrs_key)
        return (vrs_key, size_bytes / (1024**3))

    sizes = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {executor.submit(get_size, key): key for key in vrs_keys}

        for future in as_completed(future_to_key):
            try:
                vrs_key, size_gb = future.result()
                sizes[vrs_key] = size_gb
            except Exception as e:
                vrs_key = future_to_key[future]
                logger.warning(f"Error getting size for {vrs_key}: {e}")
                sizes[vrs_key] = 0.0

    return sizes


def get_file_depth(vrs_key: str, prefix: str) -> int:
    """Get the depth of a file relative to the prefix.

    Files at root level (directly under prefix) have depth 0.
    Files in subdirectories have higher depth values.
    """
    # Remove the prefix from the key
    prefix = prefix.rstrip("/")
    if prefix and vrs_key.startswith(prefix + "/"):
        relative_path = vrs_key[len(prefix) + 1 :]
    else:
        relative_path = vrs_key

    # Count directory separators (depth = number of / in relative path)
    # A file at root has 0 separators before the filename
    parts = relative_path.split("/")
    return len(parts) - 1  # -1 because the last part is the filename


def sort_by_depth(vrs_keys: List[str], prefix: str) -> List[str]:
    """Sort VRS keys by depth, with root-level files first."""
    return sorted(vrs_keys, key=lambda k: (get_file_depth(k, prefix), k))


def create_batches(
    vrs_keys: List[str],
    target_size_gb: float,
    backend: Boto3Backend,
    prefix: str = "",
    max_workers: int = 20,
) -> Tuple[List[List[str]], Dict[str, float]]:
    """Group VRS files into batches by cumulative size.

    Fetches all file sizes in parallel first, then creates batches.
    Files are sorted by depth (root-level first) before batching.
    Returns (batches, sizes_dict) for use in dry-run display.
    """
    logger.info(f"  Fetching file sizes for {len(vrs_keys)} files...")
    sizes = get_file_sizes_parallel(vrs_keys, backend, max_workers)

    # Sort by depth to prioritize root-level files
    logger.info("  Sorting files by depth (root-level first)...")
    sorted_keys = sort_by_depth(vrs_keys, prefix)

    batches = []
    current_batch = []
    current_size = 0.0

    for vrs_key in sorted_keys:
        file_size = sizes.get(vrs_key, 0.0)

        if current_batch and current_size + file_size > target_size_gb:
            batches.append(current_batch)
            current_batch = []
            current_size = 0.0

        current_batch.append(vrs_key)
        current_size += file_size

    if current_batch:
        batches.append(current_batch)

    return batches, sizes


# ============================================================================
# Batch Logging
# ============================================================================


def write_batches_to_file(
    batches: List[List[str]],
    sizes: Dict[str, float],
    output_path: Path,
) -> None:
    """Write batch information to a JSON file for reference.

    Creates a detailed log of all batches with file sizes and totals.
    """
    batch_data = {
        "created_at": datetime.now().isoformat(),
        "total_files": sum(len(batch) for batch in batches),
        "total_size_gb": round(sum(sizes.values()), 2),
        "num_batches": len(batches),
        "batches": [],
    }

    for i, batch in enumerate(batches, 1):
        batch_size = sum(sizes.get(key, 0.0) for key in batch)
        batch_info = {
            "batch_num": i,
            "num_files": len(batch),
            "size_gb": round(batch_size, 2),
            "files": [
                {
                    "vrs_key": vrs_key,
                    "filename": Path(vrs_key).name,
                    "size_gb": round(sizes.get(vrs_key, 0.0), 2),
                }
                for vrs_key in batch
            ],
        }
        batch_data["batches"].append(batch_info)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    logger.info(f"Batch plan saved to {output_path}")


# ============================================================================
# Dry Run Display
# ============================================================================


def display_dry_run(
    batches: List[List[str]],
    sizes: Dict[str, float],
) -> None:
    """Display batch information without processing (dry run mode)."""
    logger.info("\n" + "=" * 60)
    logger.info("DRY RUN MODE - No processing will be performed")
    logger.info("=" * 60)

    total_size = sum(sizes.values())
    total_files = sum(len(batch) for batch in batches)

    logger.info("\nSummary:")
    logger.info(f"  Total files to process: {total_files}")
    logger.info(f"  Total size: {total_size:.2f} GB")
    logger.info(f"  Number of batches: {len(batches)}")

    logger.info("\n" + "-" * 60)
    logger.info("Batch Details:")
    logger.info("-" * 60)

    for i, batch in enumerate(batches, 1):
        batch_size = sum(sizes.get(key, 0.0) for key in batch)
        logger.info(f"\nBatch {i}: {len(batch)} files, {batch_size:.2f} GB")
        logger.info("  Files:")
        for vrs_key in batch:
            file_size = sizes.get(vrs_key, 0.0)
            filename = Path(vrs_key).name
            logger.info(f"    - {filename} ({file_size:.2f} GB)")

    logger.info("\n" + "=" * 60)
    logger.info("To process these batches, run without --dry-run")
    logger.info("=" * 60)


# ============================================================================
# Batch Download/Upload
# ============================================================================


def download_batch(
    vrs_keys: List[str],
    local_dir: Path,
    backend: Boto3Backend,
) -> None:
    """Download all VRS files and their MPS folders to local directory.

    Starts with a clean directory - removes any existing content first.
    """
    # Clean up any existing content to start fresh
    if local_dir.exists():
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    for vrs_key in vrs_keys:
        vrs_name = Path(vrs_key).name
        mps_key = get_mps_key_from_vrs_key(vrs_key)
        mps_folder_name = get_output_folder_name(vrs_name)

        # Download VRS file
        logger.info(f"    Downloading {vrs_name}...")
        local_vrs = local_dir / vrs_name
        backend.download_file(vrs_key, local_vrs)

        # Download MPS folder if exists
        if backend.folder_exists(mps_key):
            logger.info(f"    Downloading {mps_folder_name}/...")
            local_mps = local_dir / mps_folder_name
            backend.download_folder(mps_key, local_mps)


def download_batch_async(
    batch_num: int,
    vrs_keys: List[str],
    local_dir: Path,
    backend: Boto3Backend,
) -> Path:
    """Download a batch in preparation for processing.

    Returns the batch directory path for use with process_batch_no_download.
    """
    batch_dir = local_dir / f"batch_{batch_num}"
    logger.info(f"[Batch {batch_num}] Downloading {len(vrs_keys)} VRS files...")
    download_batch(vrs_keys, batch_dir, backend)
    return batch_dir


def process_batch_no_download(
    batch_num: int,
    vrs_keys: List[str],
    batch_dir: Path,
    backend: Boto3Backend,
    mps_user: Optional[str],
    mps_password: Optional[str],
    s3_prefix: str = "",
    features: Optional[List[str]] = None,
    force: bool = False,
    retry_failed: bool = False,
) -> None:
    """Process a batch (MPS + upload). Assumes files already downloaded.

    Args:
        batch_num: Batch number for logging
        vrs_keys: List of VRS keys in this batch
        batch_dir: Local directory containing downloaded files
        backend: S3 backend for uploads
        mps_user: MPS username
        mps_password: MPS password
        s3_prefix: S3 prefix for uploading failed_recordings.json
        features: List of features to process. If None, processes all features.
        force: Pass --force flag to aria_mps
        retry_failed: Pass --retry-failed flag to aria_mps
    """
    try:
        # Run aria_mps once with all specified options
        features_str = " ".join(features) if features else "all"
        flags_str = []
        if force:
            flags_str.append("--force")
        if retry_failed:
            flags_str.append("--retry-failed")
        flags_display = " ".join(flags_str) if flags_str else "(no flags)"

        logger.info(
            f"[Batch {batch_num}] Running aria_mps with features: {features_str}, flags: {flags_display}"
        )
        result = run_aria_mps(
            batch_dir,
            mps_user,
            mps_password,
            features=features,
            force=force,
            retry_failed=retry_failed,
        )
        if result.returncode != 0:
            logger.warning(
                f"[Batch {batch_num}] aria_mps returned non-zero exit code: {result.returncode}"
            )
            logger.info(
                f"[Batch {batch_num}] Check {batch_dir}/aria_mps_pass1.log and aria_mps_pass2.log for details"
            )

        # Upload results
        logger.info(f"[Batch {batch_num}] Uploading results...")
        upload_batch_results(vrs_keys, batch_dir, backend)

        # Validate results and report failed recordings
        logger.info(f"[Batch {batch_num}] Validating MPS output...")
        failed_recordings_list, passed_recordings_list = validate_batch_results(
            vrs_keys, backend, features
        )

        # Write both failed and passed recordings to update status
        all_recordings = failed_recordings_list + passed_recordings_list
        if all_recordings:
            report_path = batch_dir.parent / "recordings_status.json"
            write_recordings_status_report(all_recordings, report_path)

        if failed_recordings_list:
            logger.warning(
                f"[Batch {batch_num}] {len(failed_recordings_list)} recording(s) "
                f"have incomplete MPS output"
            )
            for rec in failed_recordings_list:
                logger.warning(
                    f"  - {Path(rec.vrs_key).name}: {', '.join(rec.failed_features)}"
                )

        if passed_recordings_list:
            logger.info(
                f"[Batch {batch_num}] {len(passed_recordings_list)} recording(s) "
                f"completed successfully"
            )

        # Upload recordings_status.json to S3 after each batch
        report_path = batch_dir.parent / "recordings_status.json"
        if report_path.exists():
            s3_report_key = (
                f"{s3_prefix.rstrip('/')}/recordings_status.json"
                if s3_prefix
                else "recordings_status.json"
            )
            logger.info(
                f"[Batch {batch_num}] Uploading recordings_status.json to s3://{backend.bucket}/{s3_report_key}"
            )
            try:
                backend.client.upload_file(
                    str(report_path), backend.bucket, s3_report_key
                )
            except Exception as upload_err:
                logger.warning(
                    f"[Batch {batch_num}] Failed to upload recordings_status.json: {upload_err}"
                )

        logger.info(f"[Batch {batch_num}] Complete.")

    except Exception as e:
        logger.error(f"[Batch {batch_num}] {e}")
        try:
            upload_batch_results(vrs_keys, batch_dir, backend)
        except Exception:
            pass

    finally:
        if batch_dir.exists():
            shutil.rmtree(batch_dir, ignore_errors=True)


def upload_batch_results(
    vrs_keys: List[str],
    local_dir: Path,
    backend: Boto3Backend,
) -> None:
    """Upload all MPS result folders back to S3.

    Does NOT upload the VRS files - only the MPS result folders.
    """
    for vrs_key in vrs_keys:
        vrs_name = Path(vrs_key).name
        mps_key = get_mps_key_from_vrs_key(vrs_key)
        mps_folder_name = get_output_folder_name(vrs_name)

        local_mps = local_dir / mps_folder_name
        if local_mps.exists():
            logger.info(f"    Uploading {mps_folder_name}/...")
            backend.upload_folder(local_mps, mps_key)


def validate_batch_results(
    vrs_keys: List[str],
    backend: Boto3Backend,
    requested_features: Optional[List[str]] = None,
) -> Tuple[List[RecordingStatus], List[RecordingStatus]]:
    """Validate MPS results for a batch after upload.

    Checks each recording's MPS output on S3 and returns status for all recordings.

    Args:
        vrs_keys: List of VRS keys to validate
        backend: S3 backend for reading status files
        requested_features: List of features to validate. If None, validates all features.

    Returns:
        Tuple of (failed_recordings, passed_recordings)
    """
    failed_recordings = []
    passed_recordings = []

    for vrs_key in vrs_keys:
        status = get_recording_status_s3(vrs_key, backend, requested_features)
        if status.is_complete:
            passed_recordings.append(status)
        else:
            failed_recordings.append(status)

    return failed_recordings, passed_recordings


def write_recordings_status_report(
    recordings: List[RecordingStatus],
    output_path: Path,
    append: bool = True,
) -> None:
    """Write recordings status report to a JSON file.

    The JSON structure uses vrs_key as the top-level key with features as sub-keys:
    {
        "path/to/file.vrs": {
            "timestamp": "2024-01-15T10:30:00",
            "HAND_TRACKING": {
                "status": "VERSION_MISMATCH",
                "expected_version": "3.1.1",
                "actual_version": "3.0.0",
                "details": "Expected 3.1.1, got 3.0.0"
            },
            "SLAM": {
                "status": "OK",
                "expected_version": "1.1.0",
                "actual_version": "1.1.0",
                "details": null
            },
            ...
        },
        ...
    }

    If append=True and file exists, merges with existing data (updates existing entries).
    """
    # Convert to new format: dict keyed by vrs_key
    new_entries: Dict[str, dict] = {}
    for rec in recordings:
        entry = {
            "timestamp": datetime.now().isoformat(),
        }
        for f in rec.features:
            entry[f.feature] = {
                "status": f.status.value,
                "expected_version": f.expected_version,
                "actual_version": f.actual_version,
                "details": f.details,
            }
        new_entries[rec.vrs_key] = entry

    # Load existing entries if appending
    existing_data: Dict[str, dict] = {}
    if append and output_path.exists():
        try:
            with open(output_path) as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
        except (json.JSONDecodeError, Exception):
            existing_data = {}

    # Merge: new entries overwrite existing ones for the same vrs_key
    # For existing entries, update only the features that were processed
    for vrs_key, new_entry in new_entries.items():
        if vrs_key in existing_data:
            # Update existing entry with new feature data
            existing_data[vrs_key]["timestamp"] = new_entry["timestamp"]
            for key, value in new_entry.items():
                if key != "timestamp":
                    existing_data[vrs_key][key] = value
        else:
            existing_data[vrs_key] = new_entry

    # Sort by vrs_key for consistent output
    sorted_data = dict(sorted(existing_data.items()))

    with open(output_path, "w") as f:
        json.dump(sorted_data, f, indent=2)

    if new_entries:
        logger.info(f"Updated {len(new_entries)} recording(s) in {output_path}")


# ============================================================================
# aria_mps CLI Execution
# ============================================================================


def write_cli_log(
    log_file: Path,
    cmd: List[str],
    result: subprocess.CompletedProcess,
    mps_password: Optional[str],
) -> None:
    """Write CLI command output to a log file, masking the password."""
    cmd_str = " ".join(cmd)
    if mps_password:
        cmd_str = cmd_str.replace(mps_password, "****")

    with open(log_file, "w") as f:
        f.write(f"=== Command ===\n{cmd_str}\n\n")
        f.write(f"=== Return Code ===\n{result.returncode}\n\n")
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
        f.write(f"=== STDERR ===\n{result.stderr}\n")


def _build_aria_mps_cmd(
    folder: Path,
    mps_user: Optional[str],
    mps_password: Optional[str],
    features: Optional[List[str]] = None,
    force: bool = False,
    retry_failed: bool = False,
) -> List[str]:
    """Build aria_mps command with specified options."""
    cmd = [
        "aria_mps",
        "single",
        "-i",
        str(folder),
        "--no-ui",
    ]

    if force:
        cmd.append("--force")

    if retry_failed:
        cmd.append("--retry-failed")

    if features:
        cmd.append("--features")
        cmd.extend(features)

    if mps_user and mps_password:
        cmd.extend(["-u", mps_user, "-p", mps_password])

    return cmd


def _run_aria_mps_cmd(
    cmd: List[str],
    log_file: Path,
    mps_password: Optional[str],
) -> subprocess.CompletedProcess:
    """Execute aria_mps command and log output."""
    # Log command with password masked
    cmd_for_logging = " ".join(cmd)
    if mps_password:
        cmd_for_logging = cmd_for_logging.replace(mps_password, "****")
    logger.info(f"   Running command: {cmd_for_logging}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    write_cli_log(log_file, cmd, result, mps_password)
    return result


def run_aria_mps(
    folder: Path,
    mps_user: Optional[str],
    mps_password: Optional[str],
    features: Optional[List[str]] = None,
    force: bool = False,
    retry_failed: bool = False,
) -> subprocess.CompletedProcess:
    """Run aria_mps on a folder with specified options.

    Always calls aria_mps twice:
    1. First call: with --force and/or --retry-failed flags if specified, with --features
    2. Second call: without --force and --retry-failed, but with --features if specified

    This two-stage approach ensures:
    - First pass handles force reprocessing or retry of failed items (if requested)
    - Second pass processes any remaining features that weren't covered

    Args:
        folder: Path to the folder containing VRS files
        mps_user: MPS username
        mps_password: MPS password
        features: List of features to process (e.g., ["HAND_TRACKING", "SLAM", "EYE_GAZE"]).
                  If None, aria_mps will process all features.
        force: If True, pass --force flag to aria_mps (first call only)
        retry_failed: If True, pass --retry-failed flag to aria_mps (first call only)

    Returns:
        CompletedProcess from the second call
    """
    # First call: with --force and/or --retry-failed if specified
    logger.info("   [Pass 1] Running with force/retry-failed flags (if any)...")
    cmd1 = _build_aria_mps_cmd(
        folder, mps_user, mps_password, features, force, retry_failed
    )
    log_file1 = folder / "aria_mps_pass1.log"
    result1 = _run_aria_mps_cmd(cmd1, log_file1, mps_password)

    if result1.returncode != 0:
        logger.warning(
            f"   [Pass 1] aria_mps returned non-zero exit code: {result1.returncode}"
        )

    # Second call: without --force and --retry-failed, but with --features if specified
    logger.info("   [Pass 2] Running without force/retry-failed flags...")
    cmd2 = _build_aria_mps_cmd(
        folder, mps_user, mps_password, features, force=False, retry_failed=False
    )
    log_file2 = folder / "aria_mps_pass2.log"
    result2 = _run_aria_mps_cmd(cmd2, log_file2, mps_password)

    return result2


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Process .vrs files in batches using aria_mps CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables Required:
  MPS_USER        MPS username for authentication
  MPS_PASSWORD    MPS password for authentication

Examples:
  # Dry run to see batches without processing
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --s3-prefix recordings/ \\
      --local-dir /local \\
      --target-size-gb 100 \\
      --dry-run

  # Process all features (default)
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --s3-prefix recordings/ \\
      --local-dir /local \\
      --target-size-gb 100

  # Process only specific features
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --s3-prefix recordings/ \\
      --local-dir /local \\
      --features HAND_TRACKING SLAM

  # Force reprocessing with --force flag
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --local-dir /local \\
      --features HAND_TRACKING --force

  # Retry failed processing
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --local-dir /local \\
      --retry-failed

  # Include failed recordings (process them again)
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --local-dir /local \\
      --include-failed-recordings

  # Use custom recordings_status.json path
  python s3_parallel_processor.py \\
      --bucket my-s3-bucket \\
      --local-dir /local \\
      --recordings-status-path custom/path/recordings_status.json
        """,
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3-prefix",
        default="",
        help="S3 prefix/folder to search for VRS files",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory for processing",
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=100.0,
        help="Target batch size in GB (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show batches without processing (preview mode)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after processing n batches (useful for testing)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["HAND_TRACKING", "SLAM", "EYE_GAZE"],
        default=None,
        help="Features to process. If not specified, processes all features.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force flag to aria_mps (reprocess even if already done)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Pass --retry-failed flag to aria_mps",
    )
    parser.add_argument(
        "--include-failed-recordings",
        action="store_true",
        help="Include recordings from failed_recordings.json (don't filter them out)",
    )
    parser.add_argument(
        "--recordings-status-path",
        type=str,
        default=None,
        help="Custom S3 path to recordings_status.json (e.g., 'path/to/recordings_status.json')",
    )
    args = parser.parse_args()

    # Get MPS credentials from environment
    mps_user, mps_password = get_mps_credentials()
    if not mps_user or not mps_password:
        logger.warning("MPS credentials not set. Running without authentication.")

    local_dir = Path(args.local_dir)

    # Create boto3 backend
    logger.info(f"Connecting to S3 bucket: {args.bucket}")
    backend = Boto3Backend(bucket=args.bucket)

    # Step 1: Find all VRS files
    logger.info(f"Finding VRS files in s3://{args.bucket}/{args.s3_prefix}...")
    vrs_keys = backend.list_vrs_files(args.s3_prefix)
    logger.info(f"Found {len(vrs_keys)} VRS files")

    if not vrs_keys:
        logger.info("No VRS files found. Exiting.")
        return 0

    # Log requested features
    features_display = (
        args.features if args.features else ["HAND_TRACKING", "SLAM", "EYE_GAZE"]
    )
    logger.info(f"Features to process: {', '.join(features_display)}")

    # Load recordings status from S3 to exclude recordings with failed requested features
    failed_recordings: Dict[str, Set[str]] = {}
    if args.include_failed_recordings:
        logger.info("Including failed recordings (--include-failed-recordings)")
    else:
        recordings_status_source = (
            args.recordings_status_path
            if args.recordings_status_path
            else (
                f"{args.s3_prefix.rstrip('/')}/recordings_status.json"
                if args.s3_prefix
                else "recordings_status.json"
            )
        )
        logger.info(
            f"Loading recordings_status.json from S3: {recordings_status_source}"
        )
        failed_recordings = load_failed_recordings_from_s3(
            backend, args.s3_prefix, recordings_status_path=args.recordings_status_path
        )
        if failed_recordings:
            logger.info(
                f"Found {len(failed_recordings)} recording(s) with failures in recordings_status.json"
            )
        else:
            logger.info("No recordings_status.json found or file is empty")

    # Step 2: Filter to files needing processing (parallel)
    logger.info("Filtering files that need processing (parallel)...")
    files_to_process = filter_files_needing_processing(
        vrs_keys,
        backend,
        requested_features=args.features,
        failed_recordings=failed_recordings,
    )
    logger.info(f"Files needing processing: {len(files_to_process)}")

    if not files_to_process:
        logger.info("All files already processed. Nothing to do.")
        return 0

    # Step 3: Create batches by size (parallel size fetching, root-level files first)
    logger.info(f"Creating batches (target: {args.target_size_gb} GB)...")
    batches, sizes = create_batches(
        files_to_process, args.target_size_gb, backend, prefix=args.s3_prefix
    )
    logger.info(f"Created {len(batches)} batches")

    # Save batch plan to local directory
    batch_plan_path = local_dir / "batch_plan.json"
    write_batches_to_file(batches, sizes, batch_plan_path)

    # Step 4: If dry run, display batches and exit
    if args.dry_run:
        display_dry_run(batches, sizes)
        return 0

    # Step 5: Process each batch with parallel downloading
    batches_to_process = batches
    if args.max_batches is not None:
        batches_to_process = batches[: args.max_batches]
        if args.max_batches < len(batches):
            logger.info(
                f"\nNote: Limited to {args.max_batches} batches (--max-batches)"
            )

    with ThreadPoolExecutor(max_workers=1) as download_executor:
        current_download_future = None
        current_batch_dir: Optional[Path] = None

        for i, batch in enumerate(batches_to_process, 1):
            batch_size = sum(sizes.get(key, 0.0) for key in batch)
            logger.info("")
            logger.info("=" * 60)
            logger.info(
                f"Processing batch {i}/{len(batches_to_process)} "
                f"({len(batch)} files, {batch_size:.2f} GB)"
            )
            logger.info("=" * 60)

            # Wait for current batch download to complete
            if current_download_future is not None:
                logger.info(f"[Batch {i}] Waiting for download to complete...")
                current_batch_dir = current_download_future.result()
            else:
                # First batch - download synchronously
                current_batch_dir = download_batch_async(i, batch, local_dir, backend)

            # Start downloading next batch in background (if exists)
            next_download_future = None
            if i < len(batches_to_process):
                next_batch = batches_to_process[i]
                logger.info(f"[Batch {i + 1}] Starting background download...")
                next_download_future = download_executor.submit(
                    download_batch_async,
                    i + 1,
                    next_batch,
                    local_dir,
                    backend,
                )

            # Process current batch (MPS + upload)
            process_batch_no_download(
                i,
                batch,
                current_batch_dir,
                backend,
                mps_user,
                mps_password,
                args.s3_prefix,
                features=args.features,
                force=args.force,
                retry_failed=args.retry_failed,
            )

            # Move to next iteration
            current_download_future = next_download_future

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total batches processed: {len(batches_to_process)}")
    if args.max_batches is not None and args.max_batches < len(batches):
        logger.info(f"Remaining batches: {len(batches) - args.max_batches}")
    logger.info("\nTo check for any remaining failures, run with --dry-run")

    return 0


if __name__ == "__main__":
    exit(main())
