"""
Backfill embodiment / intrinsics / extrinsics onto the ABC episodes already in R2.

The `lab='abc'` ingest wrote no camera metadata at all: `zarr.attrs` carries only
embodiment, fps, task_name, task_description, total_frames and features. So the
overlays cannot project, and `embodiment` is `eva_bimanual` although the rig is a
YAM station. This rewrites each episode's top-level `zarr.json` to carry:

    embodiment  -> yam_bimanual
    intrinsics  -> {"front_1": 3x4 K}   scaled to the stored image size
    extrinsics  -> {"front_1": 4x4}     RealSense D405 stations only

Intrinsics are per-episode and are NOT recoverable from the zarr, so they are read
back from the source MCAP on the Hub. That would be a 66 MB download per episode,
except the `<topic>-info` CameraCalibration messages sit in the FIRST chunk (byte
71), so a ~1.2 MB ranged GET is enough -- a ~55x saving. The calibration's
width/height also identifies the station (640x480 RealSense vs 1920x1200 ZED-X),
which is what decides whether the published D405 extrinsics may be applied.

Network-bound, so threads rather than processes.

    # dry run (default): reports what would change, writes nothing
    python -m egomimic.scripts.abc_process.backfill_abc_metadata \
        --task-contains "fold and stack" --manifest /path/manifest.json

    # commit the rewrite
    ... --apply

The manifest records each episode's PREVIOUS attributes, so a run is reversible.
"""

import argparse
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import requests
from botocore.config import Config
from mcap.exceptions import EndOfFile
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_protobuf.decoder import DecoderFactory

from egomimic.rldb.embodiment.yam import Yam
from egomimic.utils.aws.aws_data_utils import _uses_r2_endpoint, load_env
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

logger = logging.getLogger("backfill_abc")

REPO = "XDOF/ABC-130k"
HF_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
TARGET_EMBODIMENT = "yam_bimanual"
# The first MCAP chunk is ~1.0 MB and holds the calibration; take a little more.
PREFIX_BYTES = 1_200_000
PREFIX_RETRY = 4_000_000
MONO_TOP_INFO = "/top-camera-info"

# Which top cameras may take the published D405 extrinsics.
#
# Neither the topic nor the resolution is a sufficient gate. `/top-camera` is
# mono on every RealSense station, and D405 runs at several capture modes
# (640x480, 848x480 and 1280x720 all appear) whose mount pose is identical --
# only K changes, and K is read per episode and rescaled. But the same topic
# ALSO carries a second, much wider camera: across the fold-and-stack family the
# normalised focal length splits into two clean clusters with a wide gap,
#   fx/W ~ 0.678  (2761 eps, ~70 deg HFOV, D405)
#   fx/W ~ 0.445  ( 382 eps, ~97 deg HFOV, a different camera)
# Different camera hardware means a different bracket, so the published D405
# pose must NOT be applied to the wide one. Gate on the optics.
D405_FX_OVER_WIDTH = (0.62, 0.74)

_local = threading.local()


def _task_dir(task: str) -> str:
    """SQL prose task name -> Hub directory name."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", task.lower())).strip("_")


def _s3():
    if not hasattr(_local, "s3"):
        ep = os.environ["R2_ENDPOINT_URL"]
        _local.s3 = boto3.client(
            "s3",
            endpoint_url=ep,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            aws_session_token=None
            if _uses_r2_endpoint(ep)
            else os.environ.get("R2_SESSION_TOKEN"),
            region_name="auto",
            config=Config(max_pool_connections=64, retries={"max_attempts": 5}),
        )
    return _local.s3


def _session():
    if not hasattr(_local, "sess"):
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        _local.sess = s
    return _local.sess


def read_calibration(hf_path: str, nbytes: int = PREFIX_BYTES) -> dict:
    """{topic: {K, width, height}} from the head of an episode.mcap.

    Parsed with the raw StreamReader: NonSeekingReader.iter_messages() sorts the
    whole stream before yielding, so it only ever raises EndOfFile on a prefix.
    """
    r = _session().get(
        f"{HF_BASE}/{hf_path}", headers={"Range": f"bytes=0-{nbytes - 1}"}, timeout=120
    )
    r.raise_for_status()
    schemas, channels, out = {}, {}, {}
    fac = DecoderFactory()
    try:
        for rec in StreamReader(io.BytesIO(r.content)).records:
            if isinstance(rec, Schema):
                schemas[rec.id] = rec
            elif isinstance(rec, Channel):
                channels[rec.id] = rec
            elif isinstance(rec, Message):
                ch = channels.get(rec.channel_id)
                if ch is None or not ch.topic.endswith("-info") or ch.topic in out:
                    continue
                m = fac.decoder_for("protobuf", schemas[ch.schema_id])(rec.data)
                out[ch.topic] = {"K": list(m.K), "width": m.width, "height": m.height}
    except EndOfFile:
        pass
    return out


def build_metadata(calib: dict, stored_hw: tuple[int, int]) -> tuple[dict, dict | None, str]:
    """-> (intrinsics, extrinsics|None, station tag). Scales K to the stored size."""
    top = next(
        (t for t in ("/top-camera-info", "/top-left-camera-info") if t in calib), None
    )
    if top is None:
        raise ValueError(f"no top-camera calibration among {sorted(calib)}")
    c = calib[top]
    K = np.asarray(c["K"], dtype=np.float64).reshape(3, 3).copy()
    h, w = stored_hw
    if c["width"] and c["height"]:
        K[0] *= w / c["width"]
        K[1] *= h / c["height"]
    intr = {"front_1": np.hstack([K, np.zeros((3, 1))]).tolist()}

    fx_over_w = K[0, 0] / w
    lo, hi = D405_FX_OVER_WIDTH
    is_d405 = top == MONO_TOP_INFO and lo < fx_over_w < hi
    if top != MONO_TOP_INFO:
        station = f"stereo_zedx({c['width']}x{c['height']})"
    elif is_d405:
        station = f"realsense_d405({c['width']}x{c['height']})"
    else:
        station = f"wide_angle_unknown_rig(fx/W={fx_over_w:.3f})"
    extr = {"front_1": Yam.TOP_CAMERA_D405.tolist()} if is_d405 else None
    return intr, extr, station


def process(job: tuple) -> dict:
    ep_hash, task, bucket, apply_changes, force = job
    key = f"processed_v3/abc/{ep_hash}.zarr/zarr.json"
    hf_path = f"data/train/{_task_dir(task)}/episode_{ep_hash}/episode.mcap"
    rec = {"episode_hash": ep_hash, "task": task, "key": key}
    t0 = time.time()
    try:
        s3 = _s3()
        doc = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        attrs = doc.setdefault("attributes", {})
        rec["previous"] = {
            k: attrs.get(k) for k in ("embodiment", "intrinsics", "extrinsics")
        }
        done = (
            attrs.get("embodiment") == TARGET_EMBODIMENT
            and attrs.get("intrinsics") is not None
        )
        if done and not force:
            rec.update(status="already_done", seconds=round(time.time() - t0, 1))
            return rec

        shape = attrs.get("features", {}).get("images.front_1", {}).get("shape")
        stored_hw = (int(shape[0]), int(shape[1])) if shape else (480, 640)

        calib = read_calibration(hf_path)
        if len(calib) == 0:
            calib = read_calibration(hf_path, PREFIX_RETRY)
        intr, extr, station = build_metadata(calib, stored_hw)

        attrs["embodiment"] = TARGET_EMBODIMENT
        attrs["intrinsics"] = intr
        attrs["extrinsics"] = extr
        rec.update(station=station, has_extrinsics=extr is not None,
                   fx=round(intr["front_1"][0][0], 3))

        if apply_changes:
            s3.put_object(
                Bucket=bucket, Key=key,
                Body=json.dumps(doc).encode(), ContentType="application/json",
            )
            rec["status"] = "updated"
        else:
            rec["status"] = "would_update"
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:250])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", action="append", help="exact SQL task name (repeatable)")
    g.add_argument("--task-contains", help="substring match on the SQL task name")
    g.add_argument("--all", action="store_true", help="every lab='abc' episode")
    p.add_argument("--manifest", required=True)
    p.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    p.add_argument("--force", action="store_true", help="redo already-complete episodes")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--workers",
        type=int,
        default=min(32, (int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 8) * 2),
    )
    args = p.parse_args()

    load_env()
    if not os.environ.get("HF_TOKEN"):
        tok = os.path.join(os.environ.get("HF_HOME", ""), "token")
        if os.path.exists(tok):
            os.environ["HF_TOKEN"] = open(tok).read().strip()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN not set and $HF_HOME/token missing")

    bucket = os.environ.get("BUCKET", "rldb")
    df = episode_table_to_df(create_default_engine()).query("lab=='abc'")
    if args.task:
        df = df[df.task.isin(args.task)]
    elif args.task_contains:
        df = df[df.task.str.contains(args.task_contains, case=False, na=False)]
    df = df[df.is_deleted == False]  # noqa: E712
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        raise SystemExit("no lab='abc' episodes matched the task filter")

    logger.info(
        "%d episodes across %d tasks | %s | workers=%d",
        len(df), df.task.nunique(), "APPLY" if args.apply else "DRY RUN", args.workers,
    )
    for t, n in df.task.value_counts().items():
        logger.info("    %-46s %5d", t, n)

    jobs = [(r.episode_hash, r.task, bucket, args.apply, args.force)
            for r in df.itertuples()]
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                ok = sum(r["status"] in ("updated", "would_update") for r in results)
                logger.info("  %d/%d  ok=%d  failed=%d",
                            done, len(jobs), ok,
                            sum(r["status"] == "failed" for r in results))

    counts: dict[str, int] = {}
    stations: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if "station" in r:
            stations[r["station"]] = stations.get(r["station"], 0) + 1
    manifest = {
        "applied": args.apply,
        "target_embodiment": TARGET_EMBODIMENT,
        "counts": counts,
        "stations": stations,
        "episodes": sorted(results, key=lambda r: r["episode_hash"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)) or ".", exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("counts: %s", counts)
    logger.info("stations: %s", stations)
    logger.info("manifest -> %s", args.manifest)
    for r in results:
        if r["status"] == "failed":
            logger.warning("  FAILED %s %s", r["episode_hash"][:12], r.get("error"))


if __name__ == "__main__":
    main()
