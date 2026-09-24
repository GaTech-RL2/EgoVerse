"""Fetch small reports using a private presigned-URL catalog, without logging URLs."""

import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog")
    parser.add_argument("destination")
    parser.add_argument("keys", nargs="+")
    parser.add_argument("--archive-receipt", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(Path(args.catalog).read_text())
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(key):
        path = (destination / key).resolve()
        if not path.is_relative_to(destination):
            raise ValueError("Artifact key escapes the destination")
        try:
            if key == "artifacts.tar.gz":
                if not args.archive_receipt:
                    return {"key": key, "status": "skipped_large_archive"}
                request = urllib.request.Request(
                    catalog[key], headers={"Range": "bytes=0-0"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    response.read(1)
                    receipt = {
                        "status": response.status,
                        "content_range": response.headers.get("Content-Range"),
                        "etag": response.headers.get("ETag"),
                    }
                (destination / "archive_receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                return {"key": key, **receipt}
            with urllib.request.urlopen(catalog[key], timeout=30) as response:
                data = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".download")
            temporary.write_bytes(data)
            temporary.replace(path)
            return {"key": key, "status": "downloaded", "bytes": len(data)}
        except Exception as exc:
            # Exception text can contain a presigned URL. Report only its type/status.
            return {
                "key": key,
                "status": "unavailable",
                "exception": type(exc).__name__,
                "http_status": getattr(exc, "code", None),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, args.keys):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
