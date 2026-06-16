"""
Utility script to download .vrs files and their corresponding .json metadata files from an S3 prefix.

Usage (from repo root):
    python -m egomimic.scripts.backfill_script --bucket my-bucket --prefix path/to/prefix --dest /local/path --profile default --dry-run

Behavior:
- Lists objects under the provided prefix (uses pagination).
- Finds objects ending with `.vrs` and attempts to download the `.vrs` and the matching `.json` (same key but
  with `.json` extension) if present.
- Preserves the S3 key relative path under the provided destination directory.
- Provides a `--dry-run` mode that prints what would be downloaded.

Notes:
- Requires AWS credentials available via environment, shared config, or the provided profile.
- Designed to be simple and robust for moderate-sized prefixes. For very large buckets/prefixes, consider
  filtering server-side or running in chunks.
"""
#!/usr/bin/env python3

# Ensure we're running under Python 3 (interactive prompt uses Python 3 semantics)
import sys

if sys.version_info[0] < 3:
    print(
        "This script requires Python 3. Please run with 'python3 backfill_script.py ...'"
    )
    sys.exit(1)

import argparse
import logging
import math
import os
import sys
import textwrap
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def list_objects_with_meta(bucket: str, prefix: str, s3_client) -> List[Dict]:
    """List all objects under an S3 prefix using pagination.

    Returns a list of dicts containing at least Key, Size, LastModified.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objs = []
    for page in page_iterator:
        for obj in page.get("Contents", []) if page is not None else []:
            objs.append(
                {
                    "Key": obj.get("Key"),
                    "Size": obj.get("Size", 0),
                    "LastModified": obj.get("LastModified"),
                    "StorageClass": obj.get("StorageClass", "STANDARD"),
                }
            )
    return objs


def ensure_dir_for_key(dest_dir: str, prefix: str, key: str):
    """Create local directories for a given s3 key, preserving path relative to prefix."""
    # Keep the path relative to the prefix when writing locally
    relative_path = os.path.relpath(key, prefix)
    # If relpath created '..' because prefix is not a true prefix, fallback to basename
    if relative_path.startswith(".."):
        relative_path = os.path.basename(key)
    local_dir = os.path.join(dest_dir, os.path.dirname(relative_path))
    os.makedirs(local_dir, exist_ok=True)
    return os.path.join(dest_dir, relative_path)


def download_key(
    bucket: str, key: str, dest_path: str, s3_client, dry_run: bool = False
):
    if dry_run:
        logger.info("[DRY RUN] Would download s3://%s/%s -> %s", bucket, key, dest_path)
        return True

    try:
        parent = os.path.dirname(dest_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        s3_client.download_file(bucket, key, dest_path)
        logger.info("Downloaded s3://%s/%s -> %s", bucket, key, dest_path)
        return True
    except ClientError as e:
        logger.error("Failed to download s3://%s/%s: %s", bucket, key, str(e))
        return False


def head_object_exists(bucket: str, key: str, s3_client) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in ("404", "NotFound"):
            return False
        # Re-raise unexpected errors
        raise


def human_readable_size(nbytes: int) -> str:
    if nbytes <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = int(math.floor(math.log(nbytes, 1024))) if nbytes > 0 else 0
    idx = min(idx, len(units) - 1)
    return f"{nbytes / (1024**idx):.2f}{units[idx]}"


def interactive_select(
    vrs_objects: List[Dict],
    key_map: Dict[str, Dict],
    bucket: str,
    prefix: str,
    s3_client,
    dry_run: bool = False,
    preview_chars: int = 1000,
) -> List[Dict]:
    """Interactive selection UI for choosing which .vrs objects to download.

    Commands:
      - a or all       : select all
      - n or none      : clear selection
      - <nums/ranges>  : e.g. 1,3-5 to toggle selection
      - pN             : preview JSON for item N (prints first preview_chars)
      - d or download  : finish selection and return selected
      - q or quit      : abort and return empty list
    """
    if not vrs_objects:
        print("No .vrs objects found under prefix")
        return []

    # print list (show vrs + matching json if present)
    print("Found the following .vrs files (and matching .json metadata if present):")
    for i, obj in enumerate(vrs_objects, start=1):
        size = human_readable_size(obj.get("Size", 0))
        lm = obj.get("LastModified")
        sc = obj.get("StorageClass", "STANDARD")
        lm_str = lm.strftime("%b %d, %Y %H:%M:%S") if lm is not None else "N/A"
        print(f"[{i}] {obj['Key']}")
        print(
            f"    type: vrs\n    modified: {lm_str}\n    size: {size}\n    storage: {sc}"
        )
        # matching json
        json_key = obj["Key"] + ".json"
        json_meta = key_map.get(json_key)
        if json_meta:
            jsize = human_readable_size(json_meta.get("Size", 0))
            jlm = json_meta.get("LastModified")
            jlm_str = jlm.strftime("%b %d, %Y %H:%M:%S") if jlm is not None else "N/A"
            jsc = json_meta.get("StorageClass", "STANDARD")
            print(
                f"    {json_key}\n    type: json\n    modified: {jlm_str}\n    size: {jsize}\n    storage: {jsc}"
            )
        print("")

    selected = set()

    def parse_selection(sel_str: str, max_idx: int) -> set:
        out = set()
        parts = [p.strip() for p in sel_str.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                try:
                    a_i = int(a)
                    b_i = int(b)
                    for k in range(max(1, a_i), min(max_idx, b_i) + 1):
                        out.add(k - 1)
                except ValueError:
                    continue
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= max_idx:
                        out.add(idx - 1)
                except ValueError:
                    continue
        return out

    max_idx = len(vrs_objects)
    help_msg = textwrap.dedent("""
    Commands:
      <nums>        Toggle selection by indices (e.g. 1,3-5)
      all           Select all
      none          Clear selection
      pN            Preview JSON for item N (e.g. p3)
      d or download Finish and download selected
      q or quit     Quit without downloading
    """)
    print(help_msg)

    while True:
        try:
            resp = input("Selection> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user")
            return []

        if not resp:
            continue
        if resp.lower() in ("q", "quit"):
            print("Aborting — no files selected")
            return []
        if resp.lower() in ("d", "download"):
            break
        if resp.lower() in ("all", "a"):
            selected = set(range(max_idx))
            print(f"Selected all ({max_idx})")
            continue
        if resp.lower() in ("none", "n"):
            selected.clear()
            print("Cleared selection")
            continue
        if resp.lower().startswith("p"):
            # preview
            body = resp[1:].strip()
            try:
                idx = int(body)
                if 1 <= idx <= max_idx:
                    obj = vrs_objects[idx - 1]
                    json_key = obj["Key"] + ".json"
                    json_meta = key_map.get(json_key)
                    if json_meta:
                        try:
                            r = s3_client.get_object(Bucket=bucket, Key=json_key)
                            text = r["Body"].read().decode("utf-8", errors="replace")
                            print(
                                "\n--- JSON preview (first {n} chars) ---".format(
                                    n=preview_chars
                                )
                            )
                            print(
                                textwrap.shorten(
                                    text, width=preview_chars, placeholder="..."
                                )
                            )
                            print("--- end preview ---\n")
                        except ClientError as e:
                            print(f"Error fetching {json_key}: {e}")
                    else:
                        print(f"No JSON metadata found for {obj['Key']}")
                else:
                    print("Index out of range")
            except ValueError:
                print("Invalid preview command; use pN (e.g. p3)")
            continue

        # otherwise treat as indices/ranges
        inds = parse_selection(resp, max_idx)
        if not inds:
            print("No valid indices found in input")
            continue
        # toggle selection
        for i in inds:
            if i in selected:
                selected.remove(i)
            else:
                selected.add(i)
        sel_list = sorted([i + 1 for i in selected])
        print(f"Currently selected: {sel_list}")

    # build list of selected objects
    selected_objs = [vrs_objects[i] for i in sorted(selected)]
    print(f"Final selection: {len(selected_objs)} files")
    return selected_objs


def download_vrs_and_json(
    bucket: str,
    prefix: str,
    dest_dir: str,
    profile: str = None,
    dry_run: bool = False,
    interactive: bool = False,
    preview_chars: int = 1000,
):
    session_kwargs = {}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs) if session_kwargs else boto3.Session()
    s3_client = session.client("s3")

    logger.info("Listing objects under s3://%s/%s", bucket, prefix)
    objects = list_objects_with_meta(bucket, prefix, s3_client)
    logger.info("Found %d objects under prefix", len(objects))

    key_set = {o["Key"] for o in objects}

    vrs_objects = [o for o in objects if o["Key"].lower().endswith(".vrs")]
    logger.info("Found %d .vrs files", len(vrs_objects))

    total = 0
    downloaded = 0
    # If interactive, show list and allow selection
    # create a key->meta mapping for quick lookups (used by interactive UI)
    key_map = {o["Key"]: o for o in objects}

    if interactive:
        selected_objs = interactive_select(
            vrs_objects,
            key_map,
            bucket,
            prefix,
            s3_client,
            dry_run=dry_run,
            preview_chars=preview_chars,
        )
    else:
        selected_objs = vrs_objects

    for obj in selected_objs:
        vrs_key = obj["Key"]
        total += 1
        json_key = vrs_key + ".json"

        vrs_local = ensure_dir_for_key(dest_dir, prefix, vrs_key)
        json_local = ensure_dir_for_key(dest_dir, prefix, json_key)

        ok_vrs = download_key(bucket, vrs_key, vrs_local, s3_client, dry_run=dry_run)

        if json_key in key_set:
            download_key(bucket, json_key, json_local, s3_client, dry_run=dry_run)
        else:
            try:
                if head_object_exists(bucket, json_key, s3_client):
                    download_key(
                        bucket, json_key, json_local, s3_client, dry_run=dry_run
                    )
                else:
                    logger.warning("No matching .json found for %s", vrs_key)
            except ClientError as e:
                logger.error("Error checking existence of %s: %s", json_key, e)

        if ok_vrs:
            downloaded += 1

    logger.info(
        "Finished. Processed %d .vrs files, downloaded %d (dry_run=%s)",
        total,
        downloaded,
        dry_run,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download .vrs and matching .json files from an S3 prefix"
    )
    p.add_argument(
        "--bucket",
        required=False,
        default=None,
        help="S3 bucket name (will prompt if missing)",
    )
    p.add_argument(
        "--prefix",
        required=False,
        default=None,
        help='S3 prefix to scan (e.g. "path/to/folder/") (will prompt if missing)',
    )
    p.add_argument(
        "--dest",
        required=False,
        default=None,
        help="Local destination directory (will prompt if missing)",
    )
    p.add_argument(
        "--profile", required=False, default=None, help="AWS profile name to use"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, only print what would be downloaded",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode to preview and select files to download",
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=1000,
        help="Number of chars to show when previewing json metadata",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If required args weren't provided, prompt the user for them interactively
    def prompt_for(name: str, current: str = None) -> str:
        """Prompt the user for a value; require a non-empty response."""
        if current:
            return current
        try:
            while True:
                val = input(f"Enter {name}: ").strip()
                if val:
                    return val
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user")
            sys.exit(1)

    bucket = prompt_for("S3 bucket", args.bucket)
    prefix = prompt_for("S3 prefix", args.prefix)
    dest = prompt_for("local destination directory", args.dest)

    download_vrs_and_json(
        bucket,
        prefix,
        dest,
        profile=args.profile,
        dry_run=args.dry_run,
        interactive=args.interactive,
        preview_chars=args.preview_chars,
    )


if __name__ == "__main__":
    main()
