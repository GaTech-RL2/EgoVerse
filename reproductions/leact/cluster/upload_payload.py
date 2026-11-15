"""Transfer one immutable archive through OSMO's PTY when native rsync stalls.

Requires pexpect. Only the named workflow's /osmo/run/workspace inbox is written;
the expected checksum must match locally and remotely before atomic publication.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
import re
import shlex

import pexpect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow")
    parser.add_argument("payload", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--task", default="leact")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.workflow):
        raise ValueError("Invalid workflow name")
    if not re.fullmatch(r"[a-f0-9]{64}", args.sha256):
        raise ValueError("Invalid payload checksum")
    payload = args.payload.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("Local payload checksum does not match the submitted workflow")
    encoded = base64.b64encode(payload).decode("ascii")
    remote = f"""set -euo pipefail
stty -echo -icanon min 1 time 0
trap 'stty sane >/dev/null 2>&1 || true' EXIT
mkdir -p /osmo/run/workspace
printf 'LEACT_UPLOAD_READY\\n'
head -c {len(encoded)} > /osmo/run/workspace/leact-payload.b64
base64 --decode /osmo/run/workspace/leact-payload.b64 > /osmo/run/workspace/payload.tar.gz.tmp
actual=$(sha256sum /osmo/run/workspace/payload.tar.gz.tmp | cut -d' ' -f1)
test "$actual" = {args.sha256}
mv /osmo/run/workspace/payload.tar.gz.tmp /osmo/run/workspace/payload.tar.gz
rm /osmo/run/workspace/leact-payload.b64
printf '\\nLEACT_UPLOAD_OK %s\\n' "$actual"
"""
    child = pexpect.spawn(
        "osmo", ["workflow", "exec", args.workflow, args.task,
                 "--entry", "bash -c " + shlex.quote(remote)],
        encoding="utf-8", timeout=60, maxread=65536, echo=False,
    )
    try:
        child.expect("LEACT_UPLOAD_READY\r?\n", timeout=90)
        print(f"Sending {len(payload)} bytes to {args.workflow}/{args.task}", flush=True)
        offset = 0
        while offset < len(encoded):
            sent = child.send(encoded[offset:offset + 16384])
            if sent <= 0:
                raise RuntimeError("OSMO PTY stopped accepting data")
            offset += sent
        child.expect("LEACT_UPLOAD_OK " + args.sha256, timeout=90)
        print(f"Remote archive SHA256 verified: {args.sha256}", flush=True)
        child.expect(pexpect.EOF, timeout=20)
        child.close()
        if child.exitstatus not in (None, 0):
            raise RuntimeError(f"OSMO exec returned {child.exitstatus}")
    except (pexpect.TIMEOUT, pexpect.EOF) as error:
        # No payload echo is enabled; bounded diagnostics contain control output only.
        print(str(child.before)[-3000:], flush=True)
        raise RuntimeError("OSMO exec archive transfer did not complete") from error
    finally:
        if child.isalive():
            child.close(force=True)


if __name__ == "__main__":
    main()
