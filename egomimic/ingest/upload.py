"""Upload a delivery to the EgoVerse dashboard from the command line (for big folders,
or scripts). Standard library only, so a vendor needs nothing beyond Python 3.10+.

    python egoverse_upload.py --url https://<dashboard> --user <account> \\
        <folder> [--delivery-id ID]

(egoverse_upload.py is this file, from the dashboard's /upload.py; in the repo it is
python -m egomimic.ingest.upload.)

The password is asked for, or read from EGOVERSE_UPLOAD_PASSWORD. Every file under the
folder is uploaded (subfolders kept) straight to the bucket in parts through presigned
links, then the delivery is marked complete as exactly those files: uploading to an
existing delivery replaces it. The delivery id defaults to the folder's name.
"""

from __future__ import annotations

import argparse
import getpass
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PARALLEL = 4
RETRIES = 4
BATCH = 100


class UploadError(Exception):
    pass


class Client:
    def __init__(self, url: str, opener=None):
        self.url = url.rstrip("/")
        self.opener = opener or urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.csrf = ""

    def call(self, method: str, path: str, body=None, *, data=None, query=None):
        url = self.url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        headers = {"X-CSRF-Token": self.csrf, "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with self.opener.open(req, timeout=300) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", "")
            except ValueError:
                msg = ""
            raise UploadError(f"{method} {path}: {e.code} {msg or e.reason}") from None

    def login(self, user: str, password: str) -> None:
        self.csrf = self.call(
            "POST", "/api/login", {"user": user, "password": password}
        )["csrf"]


def _retry(fn):
    for i in range(RETRIES):
        try:
            return fn()
        except (UploadError, urllib.error.URLError, OSError):
            if i + 1 == RETRIES:
                raise
            time.sleep(2**i)


def _put(url: str, data: bytes) -> str:
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=600) as r:  # presigned: no cookies
        etag = r.headers.get("ETag")
    if not etag:
        raise UploadError("the bucket returned no ETag")
    return etag  # exactly as given


def upload_file(c: Client, base: str, path: Path, name: str, progress=None) -> None:
    size = path.stat().st_size
    start = c.call("POST", f"{base}/uploads", {"name": name, "size": size})
    ref = {"name": name, "upload_id": start["upload_id"]}
    ps, n_parts = start["part_size"], start["parts"]

    def part(n: int, url: str | None) -> list:
        with open(path, "rb") as f:
            f.seek((n - 1) * ps)
            data = f.read(ps)
        if url:
            etag = _retry(lambda: _put(url, data))
        else:  # the local-directory store: parts go through the dashboard
            q = {**ref, "part": n}
            etag = _retry(
                lambda: c.call("PUT", f"{base}/part", data=data, query=q)["etag"]
            )
        if progress:
            progress(len(data))
        return [n, etag]

    try:
        etags = []
        with ThreadPoolExecutor(PARALLEL) as pool:
            for first in range(1, n_parts + 1, BATCH):
                nums = list(range(first, min(first + BATCH, n_parts + 1)))
                urls = c.call("POST", f"{base}/urls", {**ref, "parts": nums})["urls"]
                etags += pool.map(lambda n: part(n, urls[str(n)]), nums)
        c.call("POST", f"{base}/finish", {**ref, "parts": sorted(etags)})
    except BaseException:
        try:
            c.call("POST", f"{base}/abort", ref)
        except Exception:
            pass  # the dashboard's "unfinished uploads" cleanup catches it
        raise


def upload_delivery(
    c: Client, folder: Path, delivery_id: str, *, clear_stale=False, log=print
) -> dict:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise UploadError(f"{folder} has no files")
    base = "/api/deliveries/" + urllib.parse.quote(delivery_id, safe="")
    now = c.call("GET", base)
    if now["uploads"]:
        if not clear_stale:
            raise UploadError(
                f"{delivery_id} has {len(now['uploads'])} unfinished upload(s) (a "
                "crashed or running upload); pass --clear-stale to drop them"
            )
        for u in now["uploads"]:
            c.call("POST", f"{base}/clear", {"record": u["record"]})
    total = sum(p.stat().st_size for p in files) or 1
    done = [0]

    def progress(n):
        done[0] += n
        log(f"\r{done[0] / total:6.1%}", end="", flush=True)

    names = [p.relative_to(folder).as_posix() for p in files]
    for p, name in zip(files, names):
        log(f"\r{name} ({p.stat().st_size / 1e6:.1f} MB)")
        upload_file(c, base, p, name, progress)
    log("\rmarking complete")
    return c.call("POST", f"{base}/complete", {"files": names})


def url_problem(url: str) -> str | None:
    """Why the password must not be sent to `url`, or None. https only, except plain
    http to this machine (localhost or a loopback address, exactly); never with
    credentials inside the URL."""
    import ipaddress

    try:
        u = urllib.parse.urlsplit(url)
        host = u.hostname or ""
        u.port  # noqa: B018  (raises on a malformed port)
    except ValueError:
        return f"{url!r} is not a URL"
    if u.username is not None or u.password is not None:
        return "the URL must not contain a user or password; use --user"
    if not host:
        return f"{url!r} has no host"
    if u.scheme == "https":
        return None
    if u.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if loopback:
            return None
    return "the dashboard URL must be https:// (plain http only to localhost)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", required=True, help="the dashboard, e.g. https://...")
    ap.add_argument("--user", required=True)
    ap.add_argument("--delivery-id", help="default: the folder's name")
    ap.add_argument(
        "--clear-stale", action="store_true", help="drop unfinished earlier uploads"
    )
    args = ap.parse_args(argv)
    problem = url_problem(args.url)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    password = os.environ.get("EGOVERSE_UPLOAD_PASSWORD") or getpass.getpass(
        f"password for {args.user}: "
    )
    c = Client(args.url)
    try:
        c.login(args.user, password)
        out = upload_delivery(
            c,
            args.folder,
            args.delivery_id or args.folder.resolve().name,
            clear_stale=args.clear_stale,
        )
    except (UploadError, OSError) as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    print(f"\ndone: {len(out['files'])} file(s); the dashboard shows its progress")
    return 0


if __name__ == "__main__":
    sys.exit(main())
