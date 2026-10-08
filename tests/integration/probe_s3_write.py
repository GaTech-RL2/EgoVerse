from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime, timezone

import boto3
import cloudpathlib

from egomimic.utils.aws.aws_data_utils import _uses_r2_endpoint, load_env


def build_client() -> tuple[cloudpathlib.S3Client, str]:
    load_env()

    endpoint_url = (
        os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
    )
    access_key_id = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    session_token = os.environ.get("R2_SESSION_TOKEN") or os.environ.get(
        "AWS_SESSION_TOKEN"
    )

    if not access_key_id or not secret_access_key:
        raise RuntimeError("Missing R2/AWS access key credentials after load_env().")
    if not endpoint_url:
        raise RuntimeError("Missing R2 endpoint URL after load_env().")
    region_name = os.environ.get("AWS_DEFAULT_REGION", "auto")
    if _uses_r2_endpoint(endpoint_url):
        session_token = None
        region_name = "auto"

    boto3_session = boto3.session.Session(
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
    )
    return (
        cloudpathlib.S3Client(
            endpoint_url=endpoint_url,
            boto3_session=boto3_session,
        ),
        endpoint_url,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attempt to write a small object to S3/R2 using ~/.egoverse_env."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="s3://rldb/test_object",
        help="Destination object URI. Defaults to s3://rldb/test_object",
    )
    args = parser.parse_args()

    client, endpoint_url = build_client()
    target = cloudpathlib.S3Path(args.target, client=client)
    payload = "\n".join(
        [
            "EgoVerse cloudpath write test",
            f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
            f"host={socket.gethostname()}",
            f"target={args.target}",
        ]
    )

    print(f"Attempting cloudpathlib write to {args.target}")
    print(f"Endpoint: {endpoint_url}")
    target.write_text(payload)
    print(f"Write succeeded: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
