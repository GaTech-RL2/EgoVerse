"""One delivery's conversion, as a job: download, prepare, store the stage, record the outcome.

    python -m egomimic.ingest.convert_job --vendor <v> --delivery <id> \\
        --fingerprint <sha256> --lance s3://rldb/lance --work /scratch

The watcher submits one per delivery (egomimic/ingest/launch.py: AWS Batch, or in-process
for tests and local runs) and reads the outcome the job writes under pipeline/jobs/. A
delivery that changed since it was submitted is left alone ("superseded"): the watcher
submits its new version. MANO runs on the job's GPU when it has one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

from egomimic.ingest import formats
from egomimic.ingest.stage import read_pin, stage_digest
from egomimic.ingest.store import DeliveryChanged, Ref, Store


def run_job(
    store: Store,
    ref: Ref,
    fingerprint: str,
    lance_root,
    work: Path,
    **settings,
) -> dict:
    """Convert `ref` as it was at `fingerprint` and record the outcome; returns it."""
    work = Path(work) / ref.vendor / ref.delivery_id / fingerprint[:16]
    local, staged = work / "delivery", work / "staged"
    try:
        d = store.get_delivery(ref)
        if d is None or not d.complete or d.fingerprint != fingerprint:
            outcome = {"status": "superseded"}
        else:
            try:
                store.download(d, local)
            except DeliveryChanged:
                outcome = {"status": "superseded"}
            else:
                p = formats.prepare_delivery(
                    local,
                    staged,
                    lance_root,
                    fingerprint=fingerprint,
                    vendor=ref.vendor,
                    **settings,
                )
                outcome = {
                    "status": p.status,
                    "recording_id": p.recording_id,
                    "problems": p.problems,
                    "error": p.error,
                    "report": p.report,
                    "diff": p.diff,
                }
                if p.status == "staged":
                    digest = stage_digest(p.staged)
                    store.put_staged(ref, digest, p.staged)
                    pin = read_pin(p.staged)
                    outcome["stage"] = digest
                    outcome["lab"] = pin["lab"]
                    outcome["vendor_recording_id"] = pin["vendor_recording_id"]
                    outcome["provenance"] = {
                        k: pin[k] for k in ("converter", "settings")
                    }
    except Exception as e:
        outcome = {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}",
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
    store.put_job(ref, fingerprint, outcome)
    return outcome


def add_store_args(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bucket", default="ingest", help="pipeline bucket on R2")
    src.add_argument("--local-store", type=Path, help="a directory instead of R2")
    ap.add_argument(
        "--vendor-bucket", default="ingest-{vendor}", help="vendor bucket pattern"
    )
    ap.add_argument("--root", default="", help="key prefix in every bucket")


def store_from_args(args) -> Store:
    from egomimic.ingest.store import LocalStore, R2Store

    if args.local_store:
        return LocalStore(args.local_store)
    return R2Store(args.bucket, root=args.root, vendor_bucket=args.vendor_bucket)


def add_convert_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--codec", default="libsvtav1")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mano-iters", type=int, default=2400)
    ap.add_argument(
        "--mano-model-dir", help="MANO_LEFT/RIGHT.pkl (default: external_ckpts/mano)"
    )


def convert_settings(args) -> dict:
    return {
        "codec": args.codec,
        "ffmpeg": args.ffmpeg,
        "mano": args.mano,
        "mano_iters": args.mano_iters,
        "mano_model_dir": args.mano_model_dir,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vendor", required=True)
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--fingerprint", required=True)
    ap.add_argument("--lance", required=True, help="Lance root, read for the diff")
    ap.add_argument("--work", type=Path, required=True, help="local scratch")
    add_store_args(ap)
    add_convert_args(ap)
    args = ap.parse_args(argv)
    outcome = run_job(
        store_from_args(args),
        Ref(args.vendor, args.delivery),
        args.fingerprint,
        args.lance,
        args.work,
        **convert_settings(args),
    )
    print(json.dumps({k: outcome.get(k) for k in ("status", "stage", "error")}))
    return 0 if outcome["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
