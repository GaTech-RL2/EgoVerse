"""List the episode hashes already stored under an S3 prefix.

``sharpadex_manifest`` must not hand out a name that another episode already
uses. This tool lists every ``<hash>.zarr/`` store under ``--prefix`` (read
only) and writes one line per episode::

    # {"bucket": "rldb", "prefix": "processed_v3/", "listed_at": "...", "count": 2}
    2026-05-06-02-44-02-000000\tprocessed_v3/sharpa_dex/
    2026-05-06-02-44-03-000000\tprocessed_v3/other_lab/kitchen/

The second column is the folder holding the store. Pass the file to
``sharpadex_manifest --s3-hashes``. A typical run::

    python -m egomimic.scripts.sharpa_process.existing_hashes_s3 \\
        --output s3_hashes.tsv

Credentials come from the environment, as for every EgoVerse S3 tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HEADER_PREFIX = "# "
#: The bucket of every lab's processed episodes.
BUCKET = "rldb"
#: Its folder of processed episodes; a lab's episodes sit under ``<root><lab>/``.
S3_ROOT = "processed_v3/"


def listing_covers_root(listing: dict | None) -> bool:
    """Whether a listing's header covers every lab's folder (``rldb/processed_v3/``)."""
    prefix = (listing or {}).get("prefix")
    return (
        (listing or {}).get("bucket") == BUCKET
        and isinstance(prefix, str)
        and S3_ROOT.startswith(prefix)
    )


def list_episodes(client, bucket: str, prefix: str) -> list[tuple[str, str]]:
    """Return ``(episode_hash, folder)`` for every ``.zarr/`` store under ``prefix``.

    Folders are listed with ``Delimiter="/"``; a common prefix ending in
    ``.zarr/`` is an episode and is not listed further (a store holds many
    objects). The detection matches
    ``egomimic.scripts.backfill_scripts.stage_processed_folder.find_zarr_prefixes``.
    """
    paginator = client.get_paginator("list_objects_v2")
    found, stack = [], [prefix]
    while stack:
        folder = stack.pop()
        for page in paginator.paginate(Bucket=bucket, Prefix=folder, Delimiter="/"):
            for common in page.get("CommonPrefixes", []):
                sub = common["Prefix"]
                if sub.rstrip("/").endswith(".zarr"):
                    found.append((sub.rstrip("/").split("/")[-1][:-5], folder))
                else:
                    stack.append(sub)
    return sorted(found, key=lambda pair: (pair[1], pair[0]))


def write_s3_hashes(
    path: Path, bucket: str, prefix: str, listed_at: str, episodes
) -> dict:
    """Write the listing atomically; return its header."""
    path = Path(path)
    header = {
        "bucket": bucket,
        "prefix": prefix,
        "listed_at": listed_at,
        "count": len(episodes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w") as f:
        f.write(HEADER_PREFIX + json.dumps(header) + "\n")
        for name, folder in episodes:
            f.write(f"{name}\t{folder}\n")
    os.replace(partial, path)
    return header


def read_s3_listing(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Return ``(header, [(episode_hash, folder), ...])`` from a listing file.

    Raises:
        ValueError: If the header is missing, a line is malformed, or the
            line count differs from the header's ``count`` (a cut-off file).
    """
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith(HEADER_PREFIX):
        raise ValueError(f"{path}: not an existing_hashes_s3 listing")
    try:
        header = json.loads(lines[0][len(HEADER_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: unreadable header ({exc})") from exc
    episodes = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2 or not all(fields):
            raise ValueError(f"{path}:{number}: expected '<hash>\\t<folder>'")
        episodes.append((fields[0], fields[1]))
    if header.get("count") != len(episodes):
        raise ValueError(
            f"{path}: header counts {header.get('count')} episodes, the file "
            f"lists {len(episodes)}; list again"
        )
    return header, episodes


def read_s3_hashes(path: Path) -> tuple[dict, dict[str, str]]:
    """Return ``(header, {episode_hash: folder})`` from a listing file.

    A hash stored under several folders keeps the first; use
    ``read_s3_listing`` to see every folder.
    """
    header, episodes = read_s3_listing(path)
    folders: dict[str, str] = {}
    for name, folder in episodes:
        folders.setdefault(name, folder)
    return header, folders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.scripts.sharpa_process.existing_hashes_s3",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument(
        "--prefix",
        default=S3_ROOT,
        help="Folder to list recursively; a trailing '/' is added if missing.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # "processed_v3/sharpa_dex" would also match "processed_v3/sharpa_dex_old/".
    prefix = args.prefix.rstrip("/") + "/" if args.prefix else ""
    from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client

    # The listing time is when it started: a store written later may be missing.
    listed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    episodes = list_episodes(get_boto3_s3_client(), args.bucket, prefix)
    header = write_s3_hashes(args.output, args.bucket, prefix, listed_at, episodes)
    logger.info(
        "%d episode(s) under s3://%s/%s -> %s",
        header["count"],
        args.bucket,
        prefix,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
