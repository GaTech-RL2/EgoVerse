"""
Modal-parallelized zero-row scanner for Zarr episodes.

Ports every check from test_data.py and adds:
  - ZERO_FRAME_THRESHOLD: episodes fail only if >5% of frames are all-zeros.
  - Parallel Modal execution: one container per episode via .starmap().
  - Persistent output artifacts on the egoverse-training-outputs volume.

Data access model
-----------------
Episodes live on a Modal Volume named by DATA_VOLUME_NAME (default "egoverse-training-data",
override with the EGOVERSE_DATA_VOLUME env var).  The volume is mounted read-only at
DATA_MOUNT inside every worker container.  Results go to OUTPUT_VOLUME_NAME.

Concurrency
-----------
The hard Modal concurrency limit is set from MAX_CONTAINERS at import time.
To change it, set EGOVERSE_MAX_CONTAINERS before calling modal run:
    EGOVERSE_MAX_CONTAINERS=100 modal run test_data_modal.py -- --dataset-root ...
"""

from __future__ import annotations

import json
import os
import random
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

# ─── Tunables ─────────────────────────────────────────────────────────────────
ZERO_FRAME_THRESHOLD = (
    0.05  # episodes fail only if this fraction of frames are all-zeros
)
MAX_ZERO_INDICES_STORED = 1000  # cap on per-episode stored zero-frame index list

# Modal concurrency — set EGOVERSE_MAX_CONTAINERS env var before `modal run` to override.
MAX_CONTAINERS = int(os.environ.get("EGOVERSE_MAX_CONTAINERS", "50"))

# ─── Volume / image config ────────────────────────────────────────────────────
DATA_VOLUME_NAME = os.environ.get("EGOVERSE_DATA_VOLUME", "mecka_data_v2")
OUTPUT_VOLUME_NAME = "egoverse-training-outputs"
DATA_MOUNT = "/data"
OUTPUT_MOUNT = "/egoverse-training-outputs"

# egomimic must be installed in the local Python env (`pip install -e .` from project root).
# add_local_python_source ships the package source into the image at build/sync time.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "numpy",
            "zarr",
            "tqdm",
            "pandas",
            "requests",
            "scaleapi",
            "simplejpeg",
            "torch",
            "torchvision",
            "boto3",
            "botocore",
            "cloudpathlib",
            "sqlalchemy",
            "projectaria_tools",
            "opencv-python-headless",
            "einops",
            "huggingface-hub",
            "hydra-core",
            "omegaconf",
            "lightning",
            "matplotlib",
            "psutil",
            "pyarrow",
            "pytorch-kinematics",
            "rich",
            "scipy",
        ]
    )
    .add_local_python_source("egomimic")
)

app = modal.App("egoverse-data-health", image=image)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


# ─── Episode discovery ────────────────────────────────────────────────────────


@app.function(
    volumes={DATA_MOUNT: data_vol},
    timeout=600,
)
def discover_episodes(dataset_root: str) -> list[tuple[str, str]]:
    """
    Return list of (path_str, episode_hash) for every candidate zarr store under
    dataset_root (a path within DATA_MOUNT, e.g. "my-dataset/train").

    Single os.listdir — no per-entry .zattrs read. scan_episode validates each
    store and reports errors per-episode, so cheap discovery is fine here.
    """
    root = Path(DATA_MOUNT) / dataset_root.lstrip("/")
    if not root.is_dir():
        print(f"[discover] root does not exist: {root}")
        return []

    out: list[tuple[str, str]] = []
    for name in os.listdir(root):
        if name.startswith("."):
            continue
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        out.append((str(root / name), episode_hash))
    print(f"[discover] listed {len(out)} entries under {root}")
    return out


# ─── Per-episode scanner ──────────────────────────────────────────────────────


@app.function(
    volumes={DATA_MOUNT: data_vol},
    timeout=300,
    max_containers=MAX_CONTAINERS,
)
def scan_episode(ep_path_str: str, episode_hash: str) -> dict:
    """
    Open the zarr store, check every numeric (T, ...) array for all-zero rows.

    Relaxed boundary vs original:
      - PASS  — zero-frame fraction ≤ ZERO_FRAME_THRESHOLD (stats still recorded)
      - FAIL  — zero-frame fraction  > ZERO_FRAME_THRESHOLD
      - ERROR — any exception opening or reading the store

    Returns a result dict; never raises so one bad episode can't abort the run.
    """
    import numpy as np
    import zarr

    try:
        g = zarr.open_group(ep_path_str, mode="r")
        total_frames = int(g.attrs.get("total_frames", 0) or 0)

        zero_rows: dict[str, list[int]] = {}
        for key in g.keys():
            arr = g[key]
            # Skip non-numeric or 1-D-only arrays (annotations, jpeg stores)
            # — same filter as the original _scan_episode
            if arr.ndim < 2 or not np.issubdtype(arr.dtype, np.number):
                continue
            data: np.ndarray = arr[:]  # read whole array once
            T = data.shape[0]
            if total_frames == 0:
                total_frames = T  # fall back to array length if attr absent
            flat = data.reshape(T, -1)  # (T, features)
            zero_mask = (flat == 0).all(axis=1)
            bad = np.where(zero_mask)[0].tolist()
            if bad:
                zero_rows[key] = bad

        # Unique zero-frame indices across all arrays — mirrors original counting logic
        all_bad = sorted(set().union(*zero_rows.values()) if zero_rows else set())
        n_zero = len(all_bad)
        zero_pct = n_zero / total_frames if total_frames > 0 else 0.0
        exceeded = zero_pct > ZERO_FRAME_THRESHOLD

        return {
            "episode_hash": episode_hash,
            "ep_path": ep_path_str,
            "status": "fail" if exceeded else "pass",
            "total_frames": total_frames,
            "zero_frame_count": n_zero,
            "zero_frame_pct": zero_pct,
            "exceeded_threshold": exceeded,
            "zero_frame_indices": all_bad[:MAX_ZERO_INDICES_STORED],
            "zero_rows_by_key": {
                k: v[:MAX_ZERO_INDICES_STORED] for k, v in zero_rows.items()
            },
            "failure_reason": (
                f"zero_frames_exceed_threshold ({zero_pct:.6f} > {ZERO_FRAME_THRESHOLD})"
                if exceeded
                else None
            ),
            "error": None,
            "traceback": None,
        }

    except Exception as exc:
        print(f"[ERROR] ep={episode_hash}: {exc}")
        return {
            "episode_hash": episode_hash,
            "ep_path": ep_path_str,
            "status": "error",
            "total_frames": 0,
            "zero_frame_count": 0,
            "zero_frame_pct": 0.0,
            "exceeded_threshold": False,
            "zero_frame_indices": [],
            "zero_rows_by_key": {},
            "failure_reason": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ─── Artifact writer ──────────────────────────────────────────────────────────


@app.function(
    volumes={OUTPUT_MOUNT: output_vol},
    timeout=600,
)
def save_results(results: list[dict], run_id: str, runtime_seconds: float = 0.0) -> str:
    """
    Aggregate all per-episode results and write three artifacts to the output volume:
      summary.json, errors.jsonl, failures.jsonl
    Calls volume.commit() so results are durable even after the container exits.
    Returns the path to the run directory inside the volume.
    """
    run_dir = Path(OUTPUT_MOUNT) / "data_health" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    errors = [r for r in results if r["status"] == "error"]
    failures = [r for r in results if r["status"] == "fail"]
    passes = [r for r in results if r["status"] == "pass"]

    reason_counts: dict[str, int] = {}
    for r in failures:
        key = r.get("failure_reason") or "unknown"
        reason_counts[key] = reason_counts.get(key, 0) + 1

    valid_pcts = [r["zero_frame_pct"] for r in results if r["status"] != "error"]

    # ── summary.json ──────────────────────────────────────────────────────────
    summary: dict[str, Any] = {
        "run_id": run_id,
        "total_samples": len(results),
        "pass_count": len(passes),
        "fail_count": len(failures),
        "error_count": len(errors),
        "failure_reasons": reason_counts,
        "zero_frame_pct_avg": statistics.mean(valid_pcts) if valid_pcts else 0.0,
        "zero_frame_pct_median": statistics.median(valid_pcts) if valid_pcts else 0.0,
        "zero_frame_pct_max": max(valid_pcts) if valid_pcts else 0.0,
        "threshold_used": ZERO_FRAME_THRESHOLD,
        "runtime_seconds": runtime_seconds,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── errors.jsonl ──────────────────────────────────────────────────────────
    with open(run_dir / "errors.jsonl", "w") as fh:
        for r in errors:
            fh.write(
                json.dumps(
                    {
                        "episode_hash": r["episode_hash"],
                        "ep_path": r["ep_path"],
                        "error": r["error"],
                        "traceback": r.get("traceback") or "",
                    }
                )
                + "\n"
            )

    # ── failures.jsonl ────────────────────────────────────────────────────────
    with open(run_dir / "failures.jsonl", "w") as fh:
        for r in failures:
            fh.write(
                json.dumps(
                    {
                        "episode_hash": r["episode_hash"],
                        "ep_path": r["ep_path"],
                        "failure_reason": r["failure_reason"],
                        "zero_frame_pct": r["zero_frame_pct"],
                        "zero_frame_count": r["zero_frame_count"],
                        "total_frames": r["total_frames"],
                    }
                )
                + "\n"
            )

    output_vol.commit()
    return str(run_dir)


# ─── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    dataset_root: str = ".",
    run_id: str = "",
    pct: float = 100.0,
    seed: int = 42,
    max_containers: int = MAX_CONTAINERS,
):
    """
    Discover zarr episodes on the data volume and fan out scanning across Modal containers.

    dataset_root    path within the data volume, e.g. "my-dataset/train"
    run_id          optional run label (defaults to YYYYMMDD_HHMMSS UTC timestamp)
    pct             percentage of episodes to scan  [default: 100]
    seed            RNG seed for episode sub-sampling
    max_containers  shown in logs; to change the actual Modal concurrency set
                    EGOVERSE_MAX_CONTAINERS before calling modal run (the decorator
                    is evaluated at import time, so the env var must be set first)
    """
    import time

    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    print(f"[data-health] Run ID          : {run_id}")
    print(f"[data-health] Dataset root    : {dataset_root}")
    print(f"[data-health] Max containers  : {max_containers}")
    print("[data-health] Resolving valid episodes...")

    raw: list[tuple[str, str]] = discover_episodes.remote(dataset_root)
    if not raw:
        print("[data-health] No valid zarr episodes found. Exiting.")
        return

    total_found = len(raw)
    print(f"[data-health] Found {total_found} valid episodes.")

    if pct < 100.0:
        k = max(1, int(round(total_found * pct / 100.0)))
        rng = random.Random(seed)
        raw = sorted(rng.sample(raw, k))
        print(f"[data-health] Sampling {k} / {total_found} episodes ({pct:.1f}%).")

    n = len(raw)
    print(f"[data-health] Scanning {n} episodes...")

    results: list[dict] = []
    n_done = n_fail = n_error = 0

    for result in scan_episode.starmap(
        raw, order_outputs=False, return_exceptions=True
    ):
        n_done += 1

        if isinstance(result, BaseException):
            # Unhandled exception escaped the per-sample try/except
            n_error += 1
            print(f"[ERROR] (unhandled container exception #{n_done}): {result}")
            results.append(
                {
                    "episode_hash": f"__unhandled_{n_done}",
                    "ep_path": "",
                    "status": "error",
                    "total_frames": 0,
                    "zero_frame_count": 0,
                    "zero_frame_pct": 0.0,
                    "exceeded_threshold": False,
                    "zero_frame_indices": [],
                    "zero_rows_by_key": {},
                    "failure_reason": None,
                    "error": str(result),
                    "traceback": "",
                }
            )
            continue

        results.append(result)
        if result["status"] == "fail":
            n_fail += 1
        elif result["status"] == "error":
            n_error += 1

        if n_done % 100 == 0 or n_done == n:
            print(
                f"[data-health] Progress: {n_done}/{n} scanned  "
                f"({n_fail} fail, {n_error} error)"
            )

    scan_elapsed = time.time() - t0

    # Write all artifacts to the output volume
    print("\n[data-health] Writing artifacts to volume...")
    run_path = save_results.remote(results, run_id, scan_elapsed)

    print(
        f"[data-health] scanned={len(results)}  fail={n_fail}  error={n_error}  "
        f"runtime={scan_elapsed:.1f}s"
    )
    print(f"[data-health] saved to: {run_path}")
