"""Stream the frozen audit using the independently verified Linux image renderer."""

import argparse
import json
import time
from pathlib import Path

import audit_worker as frozen_auditor
from audit_worker import Catalog, sha, write_json
from linux_vision import BACKEND, apply_vision_linux


def renderer_source_hashes():
    return {
        "auditor_wrapper_source_sha256": sha(Path(__file__).read_bytes()),
        "linux_vision_source_sha256": sha(
            Path(__file__).with_name("linux_vision.py").read_bytes()
        ),
    }


def audit_worker(catalog, worker, output, assets):
    original = frozen_auditor.apply_vision
    frozen_auditor.apply_vision = apply_vision_linux
    try:
        result = frozen_auditor.audit_worker(catalog, worker, output, assets)
    finally:
        frozen_auditor.apply_vision = original
    result.update(
        vision_redraw_backend=BACKEND,
        **renderer_source_hashes(),
    )
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    try:
        result = audit_worker(
            Catalog(args.catalog), args.worker, args.output, args.assets
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception": type(exc).__name__,
                    "http_status": getattr(exc, "code", None),
                    "check": str(exc) if type(exc) is ValueError else None,
                }
            )
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "status": result["status"],
                "worker": args.worker,
                "cases": len(result["cases"]),
                "archive_sha256": result["archive"]["sha256"],
                "seconds": time.monotonic() - start,
            }
        )
    )


if __name__ == "__main__":
    main()
