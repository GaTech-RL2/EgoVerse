"""Give a new vendor its own upload bucket and a write key for exactly that bucket.

    CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... \\
        python -m egomimic.ingest.onboard scale --dashboard-origin https://ingest.example.org

Run by an admin from their own machine: the token here can create buckets and API
tokens, so it never sits on the ingest box. Steps (Cloudflare API v4):

1. create the bucket ingest-<vendor>;
2. CORS so the dashboard's upload page can PUT parts to it and read back their ETags;
3. a lifecycle rule aborting multipart uploads abandoned for 7 days;
4. an account API token with "Workers R2 Storage Bucket Item Write" on that bucket only,
   whose S3 credentials are its id (access key) and the SHA-256 of its value (secret);
5. the vendor in the ingest bucket's pipeline/vendors.json, so the watcher lists it.

The credentials are printed once, for the vendor; they are not stored anywhere.
--dry-run prints each request instead of sending it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from typing import Any

from egomimic.ingest.store import VENDOR, VENDOR_BUCKET

API = "https://api.cloudflare.com/client/v4"
WRITE_GROUP = "Workers R2 Storage Bucket Item Write"
ABANDONED_UPLOAD_S = 7 * 24 * 3600

Api = Callable[[str, str, dict | None], Any]


def http_api(token: str) -> Api:
    def call(method: str, path: str, body: dict | None) -> dict:
        req = urllib.request.Request(
            API + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        if not out.get("success"):
            raise RuntimeError(f"{method} {path}: {out.get('errors')}")
        return out.get("result") or {}

    return call


def dry_run_api(log: list) -> Api:
    def call(method: str, path: str, body: dict | None) -> dict:
        log.append((method, path, body))
        if path.endswith("/tokens/permission_groups"):
            return [{"id": "<group id>", "name": WRITE_GROUP}]
        if path.endswith("/tokens"):
            return {"id": "<token id>", "value": "<token value>"}
        return {}

    return call


def onboard(
    api: Api,
    account: str,
    vendor: str,
    *,
    dashboard_origins: list[str],
    bucket_pattern: str = VENDOR_BUCKET,
) -> dict:
    """Create the vendor's bucket and key; returns {bucket, access_key_id,
    secret_access_key, endpoint}."""
    if not VENDOR.fullmatch(vendor):
        raise ValueError(f"vendor {vendor!r} is not a lowercase slug")
    bucket = bucket_pattern.format(vendor=vendor)
    base = f"/accounts/{account}/r2/buckets"
    api("POST", base, {"name": bucket})
    if dashboard_origins:
        rule = {
            "allowed": {
                "methods": ["PUT"],
                "origins": dashboard_origins,
                "headers": ["*"],
            },
            "exposeHeaders": ["ETag"],
            "maxAgeSeconds": 3600,
        }
        api("PUT", f"{base}/{bucket}/cors", {"rules": [rule]})
    abort = {
        "id": "abort-abandoned-uploads",
        "enabled": True,
        "conditions": {"prefix": ""},
        "abortMultipartUploadsTransition": {
            "condition": {"type": "Age", "maxAge": ABANDONED_UPLOAD_S}
        },
    }
    api("PUT", f"{base}/{bucket}/lifecycle", {"rules": [abort]})
    groups = api("GET", f"/accounts/{account}/tokens/permission_groups", None)
    group = next(g for g in groups if g["name"] == WRITE_GROUP)
    token = api(
        "POST",
        f"/accounts/{account}/tokens",
        {
            "name": f"ingest upload: {vendor}",
            "policies": [
                {
                    "effect": "allow",
                    "resources": {
                        f"com.cloudflare.edge.r2.bucket.{account}_default_{bucket}": "*"
                    },
                    "permission_groups": [{"id": group["id"]}],
                }
            ],
        },
    )
    return {
        "bucket": bucket,
        "access_key_id": token["id"],
        "secret_access_key": hashlib.sha256(token["value"].encode()).hexdigest(),
        "endpoint": f"https://{account}.r2.cloudflarestorage.com",
    }


def main(argv: list[str] | None = None) -> int:
    from egomimic.ingest.convert_job import add_store_args, store_from_args

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("vendor")
    ap.add_argument(
        "--dashboard-origin",
        action="append",
        default=[],
        help="origin of the dashboard's upload page (repeatable)",
    )
    ap.add_argument("--account", default=os.environ.get("CLOUDFLARE_ACCOUNT_ID"))
    ap.add_argument("--dry-run", action="store_true")
    add_store_args(ap)
    args = ap.parse_args(argv)
    if not args.account:
        ap.error("set CLOUDFLARE_ACCOUNT_ID or pass --account")
    log: list = []
    if args.dry_run:
        api = dry_run_api(log)
    else:
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not token:
            ap.error("set CLOUDFLARE_API_TOKEN (an admin token, kept off the box)")
        api = http_api(token)
    out = onboard(
        api,
        args.account,
        args.vendor,
        dashboard_origins=args.dashboard_origin,
        bucket_pattern=args.vendor_bucket,
    )
    if args.dry_run:
        for method, path, body in log:
            print(method, path, json.dumps(body) if body is not None else "")
        print(
            f"would register {args.vendor} -> {out['bucket']} in pipeline/vendors.json"
        )
        return 0
    store_from_args(args).add_vendor(args.vendor, out["bucket"])
    print(json.dumps(out, indent=1))
    print("Give the vendor these credentials once; they are not stored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
