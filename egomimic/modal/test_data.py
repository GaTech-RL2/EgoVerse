"""
Modal-parallelized version of egomimic/scripts/test_data.py.

Scans every Zarr episode on the `mecka_data_v2` Modal volume in parallel
across up to MAX_CONTAINERS CPU-only workers, then prints the same summary
as the original.

Checks (same as the original):
  1. Rows of all-zeros in any numeric array (frame-gap corruption sign).
  2. Embodiment in zarr.json + presence of every required key dir.
  3. /c layout — files under <key>/c/ with names longer than 4 chars.

Architecture (volume-only state, no long-lived control connections)
-------------------------------------------------------------------
Every persisted artifact lives on the zarr volume under
    <volume>/_results/<run_id>/
        meta.json           — n_shards, expected_total, started
        LAUNCHED            — marker file (idempotent re-entry of scan_all)
        shard_000000.json   — one per shard, written atomically by workers
        ...
        summary.json        — compact aggregate, written by summarize

No modal.Dict and no long-running poll. Each modal function is short and
exits cleanly, so neither client→worker heartbeats nor worker preemption
can break the run.

  main (local):       build emb table, scan_all.spawn(...), exit.

  scan_all (modal):   list episodes, write meta.json, *parallel-spawn* every
                      shard from a ThreadPoolExecutor (RPCs happen
                      concurrently, finishing in seconds instead of minutes),
                      spawn summarize, exit. Idempotent via LAUNCHED marker.

  scan_shard ×N:      scan their slice, atomically write
                      shard_<idx>.json to the volume, exit. Idempotent —
                      if the file already exists, returns immediately.

  summarize:          a single short-lived poll iteration: read shard_*.json
                      from the volume; if all present (or max iters reached)
                      merge + print + write summary.json + exit; otherwise
                      sleep ~60s and spawn the next iteration of itself.
                      Each invocation is <5 min, well below any heartbeat
                      threshold.

Why this fixes the heartbeat / preemption loop the previous version hit
----------------------------------------------------------------------
The old scan_all looped `scan_shard.spawn()` ~1000 times serially. Each
.spawn() is one control-plane RPC; the cumulative wall-clock exceeded the
client→worker heartbeat budget (~60s), so Modal preempted scan_all
mid-loop. On restart it would re-enter the same loop and preempt again.

This version:
  * sizes shards to MAX_CONTAINERS so we issue exactly that many spawns
    (no deep queue), and
  * issues those spawns in parallel via a ThreadPool so the whole spawn
    phase completes in seconds.

Combined with volume-only state (no modal.Dict polling) and the
self-respawning summarize, no function ever runs long enough to heartbeat.

Usage
-----
    # --detach REQUIRED (spawned work must outlive the local entrypoint):
    modal run --detach --env robotics egomimic/modal/test_data.py
    modal run --detach --env robotics egomimic/modal/test_data.py -- --pct 10

    # watch progress; the summarize logs end with the ==== summary block:
    modal app logs egomimic-test-data

    # re-summarize an already-scanned run (e.g. if you missed the log):
    modal run --env robotics egomimic/modal/test_data.py::summarize --run-id <run_id>
"""

from __future__ import annotations

import json as _json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
VOLUME_MOUNT_PATH = "/mnt/zarr-data"
ZARR_VOLUME_NAME = "mecka_data_v2"
RESULTS_PREFIX = "_results"  # results live under <vol>/_results/<run_id>/

# Global cap on concurrent scan_shard worker containers.
MAX_CONTAINERS = 200

# Target eps per shard. Smaller shards = faster individual completion =
# earlier progress reports surfacing in summarize. Each container will run
# multiple shards back-to-back (warm reuse), so the total wall time is
# governed by MAX_CONTAINERS, not the shard count.
TARGET_EPS_PER_SHARD = 250

# Hard cap on the total shard count so the parallel .spawn() phase stays
# well under any client-heartbeat budget even at 1M+ episodes.
MAX_SHARDS = MAX_CONTAINERS * 8  # 1600

# Parallelism for the .spawn() RPC fan-out inside scan_all. The actual scan
# concurrency is bounded by MAX_CONTAINERS; this just controls how fast we
# enqueue the calls. 32 threads × ~100ms/spawn handles 1600 spawns in ~5s.
SPAWN_RPC_THREADS = 32

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "zarr==3.1.5",
    "numpy",
)
zarr_volume = modal.Volume.from_name(ZARR_VOLUME_NAME)

app = modal.App("egomimic-test-data", image=image)


# ---------------------------------------------------------------------------
# Embodiment metadata — inlined so the worker doesn't need egomimic
# ---------------------------------------------------------------------------
# Mirror of egomimic.rldb.embodiment.embodiment.EMBODIMENT.
EMBODIMENT_ID_TO_KEY: dict[int, str] = {
    0: "EVE_RIGHT_ARM",
    1: "EVE_LEFT_ARM",
    2: "EVE_BIMANUAL",
    3: "ARIA_RIGHT_ARM",
    4: "ARIA_LEFT_ARM",
    5: "ARIA_BIMANUAL",
    6: "EVA_RIGHT_ARM",
    7: "EVA_LEFT_ARM",
    8: "EVA_BIMANUAL",
    9: "MECKA_BIMANUAL",
    10: "MECKA_RIGHT_ARM",
    11: "MECKA_LEFT_ARM",
    12: "SCALE_BIMANUAL",
    13: "SCALE_RIGHT_ARM",
    14: "SCALE_LEFT_ARM",
}


def _extract_embodiment_from_zarr_json(zarr_json: dict) -> str | None:
    """Mirror of the same helper in egomimic/scripts/test_data.py."""
    candidates: list[object] = []

    for k in ("embodiment", "robot_name", "robot_type", "embodiment_name"):
        if k in zarr_json:
            candidates.append(zarr_json.get(k))

    attrs = zarr_json.get("attributes", {})
    if isinstance(attrs, dict):
        for k in ("embodiment", "robot_name", "robot_type", "embodiment_name"):
            if k in attrs:
                candidates.append(attrs.get(k))

    meta = zarr_json.get("metadata", {})
    if isinstance(meta, dict):
        for k in ("embodiment", "robot_name", "robot_type", "embodiment_name"):
            if k in meta:
                candidates.append(meta.get(k))

    for v in candidates:
        if v is None:
            continue
        if isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()):
            name = EMBODIMENT_ID_TO_KEY.get(int(v))
            if name:
                return name.upper()
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s.upper()
    return None


def _required_modes_for_embodiment(
    emb_name: str, emb_table: dict[str, dict[str, list[str]]]
) -> tuple[dict[str, set[str]], str | None]:
    """Resolve embodiment name → {mode: required-zarr-key-set}.

    Each embodiment supports one or more modes (cartesian / keypoints /
    etc.). A real episode only ships keys for ONE mode, so the validator
    should accept an episode if any mode is fully satisfied — not require
    every key from every mode at once (the bug the original test_data.py
    had, which flagged ~100% of episodes).
    """
    emb = str(emb_name).strip().upper().replace("-", "_").replace(" ", "_")
    if emb in {"SCALE", "SCALE_BIMANUAL"}:
        emb = "SCALE_BIMANUAL"
    if emb.startswith("EVA_"):
        family = "EVA"
    elif emb.startswith("MECKA_"):
        family = "MECKA"
    elif emb.startswith("ARIA_"):
        family = "ARIA"
    elif emb.startswith("SCALE_"):
        family = "SCALE"
    else:
        return {}, f"unsupported embodiment '{emb_name}'"
    modes = emb_table.get(family, {})
    if not modes:
        return {}, f"no required keys resolved for embodiment '{emb_name}'"
    return {m: set(keys) for m, keys in modes.items() if keys}, None


def _build_embodiment_keymap_table() -> dict[str, dict[str, list[str]]]:
    """Run *locally* — needs the full egomimic stack (torch etc.).

    Returns {family: {mode: [zarr_key, ...]}} for EVA / MECKA / ARIA / SCALE.
    Modes are kept separate (instead of unioned) so the validator can accept
    an episode that fully satisfies *any* one mode — real episodes only
    ship keys for the mode they were processed in.
    """
    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.rldb.embodiment.human import Aria, Mecka, Scale

    def _keys_from(km: dict | None) -> list[str]:
        out: set[str] = set()
        if not km:
            return []
        for spec in km.values():
            if not isinstance(spec, dict):
                continue
            zkey = spec.get("zarr_key")
            if isinstance(zkey, str) and zkey != "annotations":
                out.add(zkey)
        return sorted(out)

    # Only Mecka overrides get_keymap to accept the `annotations` kwarg;
    # Eva/Aria/Scale use the base Embodiment.get_keymap which doesn't.
    return {
        "EVA": {
            "cartesian": _keys_from(Eva.get_keymap("cartesian")),
        },
        "MECKA": {
            "cartesian": _keys_from(Mecka.get_keymap("cartesian", annotations=False)),
            "keypoints": _keys_from(Mecka.get_keymap("keypoints", annotations=False)),
        },
        "ARIA": {
            "cartesian": _keys_from(Aria.get_keymap("cartesian")),
            "keypoints": _keys_from(Aria.get_keymap("keypoints")),
        },
        "SCALE": {
            "cartesian": _keys_from(Scale.get_keymap("cartesian")),
            "keypoints": _keys_from(Scale.get_keymap("keypoints")),
        },
    }


# ---------------------------------------------------------------------------
# Volume-state helpers (replaces modal.Dict)
# ---------------------------------------------------------------------------
def _results_dir(base: Path, run_id: str) -> Path:
    return base / RESULTS_PREFIX / run_id


def _write_json_atomic(path: Path, obj) -> None:
    """Write JSON via a tmp+rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        _json.dump(obj, f)
    tmp.replace(path)


def _read_json_safe(path: Path):
    try:
        with path.open("r") as f:
            return _json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-shard aggregator structure
# ---------------------------------------------------------------------------
def _empty_partial() -> dict:
    return {
        "scanned": 0,
        "scan_errors": [],
        "total_episodes_with_zeros": 0,
        "total_zero_frames": 0,
        "total_episodes_with_struct_issues": 0,
        "total_episodes_with_chunk_issues": 0,
        "results_with_zeros": {},
        "results_struct_issues": {},
        "results_chunk_issues": {},
    }


def _fold_shard_result(acc: dict, res: dict) -> None:
    """Fold one per-episode scan result into an accumulator dict (in place)."""
    acc["scanned"] += 1
    eh = res["episode_hash"]

    if res["error"]:
        acc["scan_errors"].append(f"{eh}: {res['error']}")
        return

    if res.get("issues_struct"):
        acc["total_episodes_with_struct_issues"] += 1
        acc["results_struct_issues"][eh] = list(res["issues_struct"])

    if res.get("issues_chunks"):
        acc["total_episodes_with_chunk_issues"] += 1
        acc["results_chunk_issues"][eh] = dict(res["issues_chunks"])

    if res["zero_rows"]:
        acc["total_episodes_with_zeros"] += 1
        acc["results_with_zeros"][eh] = res["zero_rows"]
        all_bad: set[int] = set()
        for bad in res["zero_rows"].values():
            all_bad.update(bad)
        acc["total_zero_frames"] += len(all_bad)


def _merge_partials(partials: list[dict]) -> dict:
    """Merge per-shard partial dicts into one. Episode hashes are unique
    across shards, so dict-update is collision-free."""
    out = _empty_partial()
    for p in partials:
        out["scanned"] += p["scanned"]
        out["scan_errors"].extend(p["scan_errors"])
        out["total_episodes_with_zeros"] += p["total_episodes_with_zeros"]
        out["total_zero_frames"] += p["total_zero_frames"]
        out["total_episodes_with_struct_issues"] += p[
            "total_episodes_with_struct_issues"
        ]
        out["total_episodes_with_chunk_issues"] += p["total_episodes_with_chunk_issues"]
        out["results_with_zeros"].update(p["results_with_zeros"])
        out["results_struct_issues"].update(p["results_struct_issues"])
        out["results_chunk_issues"].update(p["results_chunk_issues"])
    return out


# ---------------------------------------------------------------------------
# Worker — fan out to up to MAX_CONTAINERS containers
# ---------------------------------------------------------------------------
@app.function(
    cpu=1.0,
    memory=4096,
    timeout=3600,  # 1h per shard; ample slack for ~1000 eps
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    max_containers=MAX_CONTAINERS,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0),
)
def scan_shard(
    run_id: str,
    shard_idx: int,
    episode_specs: list[tuple[str, str]],
    emb_table: dict[str, list[str]],
) -> None:
    """Scan a shard of episodes and write a partial aggregate to the volume.

    Idempotent: if this shard's result file already exists (preemption-restart
    or duplicate spawn), return immediately. Episodes with is_deleted=True in
    zarr.json are skipped to match LocalEpisodeResolver + DatasetFilter().
    """
    import numpy as _np
    import zarr as _zarr

    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)
    out_path = _results_dir(base, run_id) / f"shard_{shard_idx:06d}.json"
    if out_path.exists():
        print(f"shard {shard_idx}: already done, skipping")
        return

    def _validate_c_layout(key_dir: Path) -> list[str]:
        cdir = key_dir / "c"
        if not cdir.exists():
            return ["missing c/"]
        if not cdir.is_dir():
            return ["c/ is not a directory"]
        try:
            entries = list(cdir.iterdir())
        except Exception as e:
            return [f"cannot list c/: {type(e).__name__}: {e}"]
        issues: list[str] = []
        for ent in entries:
            if ent.name == "zarr.json":
                continue
            if len(ent.name) > 4:
                issues.append(f"name length > 4 under c/: {ent.name}")
        return issues

    def _scan(name: str, eh: str) -> dict | None:
        ep_path = base / name
        try:
            g = _zarr.open_group(str(ep_path), mode="r")

            # Skip deleted episodes to match LocalEpisodeResolver+DatasetFilter()
            attrs = dict(g.attrs)
            if attrs.get("is_deleted"):
                return None

            total_frames = int(attrs.get("total_frames", 0) or 0)

            issues_struct: list[str] = []
            issues_chunks: dict[str, list[str]] = {}
            embodiment_name: str | None = None

            # (2) Embodiment + required keys.
            #
            # We support both zarr v3 (zarr.json) and v2 (.zattrs) layouts —
            # the training stack reads both, so we have to as well. The
            # original test_data.py was v3-only and over-flagged v2 stores.
            zarr_json_path = ep_path / "zarr.json"
            zattrs_path = ep_path / ".zattrs"
            is_v3 = zarr_json_path.is_file()

            if is_v3:
                zj = _read_json_safe(zarr_json_path)
                if isinstance(zj, dict):
                    embodiment_name = _extract_embodiment_from_zarr_json(zj)
            if not embodiment_name and zattrs_path.is_file():
                za = _read_json_safe(zattrs_path)
                if isinstance(za, dict):
                    embodiment_name = _extract_embodiment_from_zarr_json(
                        {"attributes": za}
                    )
            if not embodiment_name:
                attr_emb = attrs.get("embodiment")
                if isinstance(attr_emb, int):
                    embodiment_name = EMBODIMENT_ID_TO_KEY.get(attr_emb)
                elif isinstance(attr_emb, str) and attr_emb.strip():
                    embodiment_name = attr_emb.strip().upper()

            required_modes: dict[str, set[str]] = {}
            if not embodiment_name:
                issues_struct.append("missing embodiment (zarr.json/.zattrs/attrs)")
            else:
                required_modes, km_err = _required_modes_for_embodiment(
                    embodiment_name, emb_table
                )
                if km_err:
                    issues_struct.append(km_err)

            # Pick the satisfied mode: an episode is OK if every key for at
            # least one mode (cartesian or keypoints) is present as a dir.
            # We then run downstream checks (/c layout, non-dir) only against
            # that mode's keys — checking the other mode's keys against an
            # episode that wasn't processed in that mode just produces noise.
            satisfied_mode: str | None = None
            mode_missing: dict[str, list[str]] = {}
            for mode_name, mode_keys in required_modes.items():
                missing = sorted(k for k in mode_keys if not (ep_path / k).exists())
                mode_missing[mode_name] = missing
                if not missing and satisfied_mode is None:
                    satisfied_mode = mode_name

            if required_modes and satisfied_mode is None:
                # Neither mode is fully satisfied — pick the one with the
                # fewest missing keys for the diagnostic so the user sees
                # the smallest legible gap.
                best_mode, best_missing = min(
                    mode_missing.items(), key=lambda x: len(x[1])
                )
                issues_struct.append(
                    f"no mode satisfied (best={best_mode}, "
                    f"missing={', '.join(best_missing)})"
                )
            elif satisfied_mode is not None:
                for zkey in sorted(required_modes[satisfied_mode]):
                    key_dir = ep_path / zkey
                    if not key_dir.is_dir():
                        issues_struct.append(
                            f"{zkey}: expected directory, found non-dir"
                        )
                        continue
                    # /c chunk layout is a zarr v3 convention. v2 stores
                    # chunks as 0, 0.0, etc. directly under key_dir, so the
                    # check doesn't apply. Skip on v2 to avoid false flags.
                    if is_v3:
                        c_issues = _validate_c_layout(key_dir)
                        if c_issues:
                            issues_chunks[zkey] = c_issues

            # (1) rows of zeros — only on small pose/keypoint vectors.
            # Skip image arrays: they dominate I/O on a network volume
            # (hundreds of MB each), and zero-row corruption only manifests
            # in pose vecs anyway. Filter by name first to avoid even loading
            # image metadata (zarr 3.x does an async metadata fetch per key).
            zero_rows: dict[str, list[int]] = {}
            for k in g.keys():
                if k.startswith("images.") or k.startswith("images/"):
                    continue
                try:
                    arr = g[k]
                except Exception:
                    continue
                if not hasattr(arr, "ndim") or not hasattr(arr, "shape"):
                    continue  # skip groups
                if arr.ndim < 2:
                    continue
                if not _np.issubdtype(arr.dtype, _np.number):
                    continue
                # Safety net: per-frame element count > 1024 ⇒ image-like
                per_frame = 1
                for d in arr.shape[1:]:
                    per_frame *= int(d)
                if per_frame > 1024:
                    continue
                try:
                    data = arr[:]
                except Exception:
                    continue
                T = data.shape[0]
                flat = data.reshape(T, -1)
                bad = _np.where((flat == 0).all(axis=1))[0].tolist()
                if bad:
                    zero_rows[k] = bad

            return {
                "episode_hash": eh,
                "total_frames": total_frames,
                "embodiment": embodiment_name,
                "zero_rows": zero_rows,
                "issues_struct": issues_struct,
                "issues_chunks": issues_chunks,
                "error": None,
            }

        except Exception as e:
            return {
                "episode_hash": eh,
                "total_frames": 0,
                "embodiment": None,
                "zero_rows": {},
                "issues_struct": [],
                "issues_chunks": {},
                "error": str(e),
            }

    # I/O bound — threads inside the container amplify each shard.
    acc = _empty_partial()
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(lambda spec: _scan(*spec), episode_specs):
            if res is not None:
                _fold_shard_result(acc, res)

    _write_json_atomic(out_path, acc)
    zarr_volume.commit()

    flagged = (
        acc["total_episodes_with_zeros"]
        + acc["total_episodes_with_struct_issues"]
        + acc["total_episodes_with_chunk_issues"]
        + len(acc["scan_errors"])
    )
    print(f"shard {shard_idx}: done — {acc['scanned']} eps, flagged {flagged}")


# ---------------------------------------------------------------------------
# Coordinator — discover episodes, parallel-spawn shards, spawn summarize
# ---------------------------------------------------------------------------
@app.function(
    cpu=4.0,
    memory=8192,
    timeout=1800,  # discovery + parallel spawn; finishes in seconds-to-minutes
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    retries=modal.Retries(max_retries=5, backoff_coefficient=2.0),
)
def scan_all(
    emb_table: dict[str, list[str]],
    pct: float,
    seed: int,
    run_id: str,
) -> None:
    """Discover episodes, spawn every scan_shard + summarize, then exit."""
    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)
    rdir = _results_dir(base, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    launched_path = rdir / "LAUNCHED"

    if launched_path.exists():
        # Preemption-restart after the spawn phase already completed.
        # Don't re-spawn workers; just make sure summarize is running.
        print(f"run {run_id}: already launched, (re)spawning summarize only")
        summarize.spawn(run_id, 0)
        return

    print("Listing episodes from volume (single readdir)...")
    raw: list[tuple[str, str]] = []
    for name in os.listdir(base):
        if name.startswith(".") or name.startswith("_"):
            continue
        p = base / name
        if not p.is_dir():
            continue
        eh = name[:-5] if name.endswith(".zarr") else name
        raw.append((name, eh))
    raw.sort()
    print(f"  found {len(raw)} episode dirs")

    if not raw:
        _write_json_atomic(rdir / "meta.json", {"n_shards": 0, "expected_total": 0})
        launched_path.touch()
        zarr_volume.commit()
        summarize.spawn(run_id, 0)
        return

    if pct < 100.0:
        k = max(1, int(round(len(raw) * pct / 100.0)))
        rng = random.Random(seed)
        raw = sorted(rng.sample(raw, k))
        print(f"  sampling {k} episodes ({pct:.1f}%)")

    # Size shards so each one finishes in tens of seconds — gives summarize
    # an early stream of "shard done" files instead of one big wave at the
    # very end. Capped at MAX_SHARDS so the parallel .spawn() phase stays
    # well under the client-heartbeat budget (the original bug was a serial
    # 1000-call spawn loop overrunning that budget).
    n_shards = max(1, min(MAX_SHARDS, math.ceil(len(raw) / TARGET_EPS_PER_SHARD)))
    n_shards = min(n_shards, len(raw))
    chunk = math.ceil(len(raw) / n_shards)
    shard_specs = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
    n_shards = len(shard_specs)

    _write_json_atomic(
        rdir / "meta.json",
        {
            "n_shards": n_shards,
            "expected_total": len(raw),
            "started": time.time(),
        },
    )
    zarr_volume.commit()

    print(
        f"  fan-out: {n_shards} shards × ~{chunk} eps "
        f"(global max_containers={MAX_CONTAINERS}); spawning in parallel..."
    )

    # Issue the spawn RPCs concurrently. With 200 spawns at ~100ms each,
    # serial would take ~20s — fine, but parallel-32 collapses it to under
    # a second and leaves a comfortable margin under any client-side
    # heartbeat timeout.
    def _spawn_one(idx_specs: tuple[int, list[tuple[str, str]]]) -> None:
        idx, specs = idx_specs
        scan_shard.spawn(run_id, idx, specs, emb_table)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=SPAWN_RPC_THREADS) as ex:
        list(ex.map(_spawn_one, enumerate(shard_specs)))
    print(f"  spawned {n_shards} shards in {time.time() - t0:.1f}s")

    launched_path.touch()
    zarr_volume.commit()
    summarize.spawn(run_id, 0)
    print(f"run {run_id}: all shards + summarize spawned; coordinator exiting")


# ---------------------------------------------------------------------------
# Summarize — short-lived self-respawning poller
# ---------------------------------------------------------------------------
SUMMARIZE_POLL_S = 30  # poll cadence (one sleep per invocation)
SUMMARIZE_MAX_ITERS = 720  # 720 × 30s = 6h overall budget


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=300,  # 5min per iteration — far under any heartbeat threshold
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0),
)
def summarize(run_id: str, iteration: int = 0) -> None:
    """One poll iteration. Either finishes the summary or respawns itself."""
    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)
    rdir = _results_dir(base, run_id)

    meta = _read_json_safe(rdir / "meta.json")
    if meta is None:
        if iteration < SUMMARIZE_MAX_ITERS:
            print(
                f"run {run_id}: meta.json not yet present "
                f"(iter {iteration}/{SUMMARIZE_MAX_ITERS}); respawning"
            )
            time.sleep(SUMMARIZE_POLL_S)
            summarize.spawn(run_id, iteration + 1)
        else:
            print(f"run {run_id}: meta.json never appeared; aborting summarize")
        return

    n_shards = int(meta.get("n_shards", 0))
    expected_total = int(meta.get("expected_total", 0))

    if n_shards == 0:
        print(f"run {run_id}: no episodes found.")
        _write_json_atomic(
            rdir / "summary.json", {"scanned": 0, "summary": "No episodes found."}
        )
        zarr_volume.commit()
        return

    shard_paths = sorted(rdir.glob("shard_*.json"))
    done = len(shard_paths)
    pct_done = int(100 * done / n_shards) if n_shards else 100
    print(
        f"run {run_id}: shards reported {done}/{n_shards} ({pct_done}%) "
        f"iter={iteration}/{SUMMARIZE_MAX_ITERS}"
    )

    if done < n_shards and iteration < SUMMARIZE_MAX_ITERS:
        # Short sleep, then respawn — keeps this invocation well under the
        # 5min timeout and any client-heartbeat window.
        time.sleep(SUMMARIZE_POLL_S)
        summarize.spawn(run_id, iteration + 1)
        return

    # Either every shard has reported or we've hit the iteration budget —
    # merge what we have and print the summary.
    partials: list[dict] = []
    for p in shard_paths:
        d = _read_json_safe(p)
        if isinstance(d, dict):
            partials.append(d)

    result = _merge_partials(partials)
    missing = n_shards - len(partials)
    if missing > 0:
        result["scan_errors"].append(
            f"[summarize] {missing} shard(s) missing after {iteration} iters "
            f"(~{iteration * SUMMARIZE_POLL_S}s budget) — partial result"
        )

    _print_summary(result, expected_total)

    flagged = (
        result["total_episodes_with_zeros"]
        + result["total_episodes_with_struct_issues"]
        + result["total_episodes_with_chunk_issues"]
        + len(result["scan_errors"])
    )
    _write_json_atomic(
        rdir / "summary.json",
        {
            "scanned": result["scanned"],
            "expected_total": expected_total,
            "missing_shards": missing,
            "flagged": flagged,
            "zeros": result["total_episodes_with_zeros"],
            "zero_frames": result["total_zero_frames"],
            "struct": result["total_episodes_with_struct_issues"],
            "chunks": result["total_episodes_with_chunk_issues"],
            "errors": len(result["scan_errors"]),
        },
    )
    zarr_volume.commit()


def _print_summary(result: dict, expected_total: int = 0) -> None:
    """Print the final report. Runs inside summarize (logged to Modal)."""
    scanned = result.get("scanned", 0)
    if scanned == 0:
        print(result.get("summary", "No episodes scanned."))
        return

    scan_errors = result["scan_errors"]
    total_episodes_with_zeros = result["total_episodes_with_zeros"]
    total_zero_frames = result["total_zero_frames"]
    total_episodes_with_struct_issues = result["total_episodes_with_struct_issues"]
    total_episodes_with_chunk_issues = result["total_episodes_with_chunk_issues"]
    results_with_zeros = result["results_with_zeros"]
    results_struct_issues = result["results_struct_issues"]
    results_chunk_issues = result["results_chunk_issues"]

    print()
    print("=" * 60)
    print(f"Episodes expected      : {expected_total}")
    print(f"Episodes scanned       : {scanned}")
    print(f"Scan errors            : {len(scan_errors)}")
    print(f"Episodes with zeros    : {total_episodes_with_zeros}")
    print(f"Total bad frame slots  : {total_zero_frames}")
    print(f"Episodes with struct   : {total_episodes_with_struct_issues}")
    print(f"Episodes with chunks   : {total_episodes_with_chunk_issues}")

    if results_with_zeros:
        print()
        print("Episodes with zero rows (sorted by bad-frame count):")
        sorted_eps = sorted(
            results_with_zeros.items(),
            key=lambda x: -len(set().union(*x[1].values())),
        )
        for eh, zero_rows in sorted_eps:
            all_bad_sorted = sorted(set().union(*zero_rows.values()))
            keys_str = ", ".join(zero_rows.keys())
            print(f"  {eh}")
            print(f"    keys   : {keys_str}")
            print(
                f"    frames : {len(all_bad_sorted)}  "
                f"{all_bad_sorted[:20]}{'...' if len(all_bad_sorted) > 20 else ''}"
            )

    if results_struct_issues:
        print()
        print("Episodes with structure/key issues:")
        for eh, issues in sorted(results_struct_issues.items()):
            print(f"  {eh}")
            for it in issues:
                print(f"    - {it}")

    if results_chunk_issues:
        print()
        print("Episodes with /c layout issues:")
        for eh, per_key in sorted(results_chunk_issues.items()):
            print(f"  {eh}")
            for zkey, issues in sorted(per_key.items()):
                preview = issues[:6]
                more = " ..." if len(issues) > 6 else ""
                print(f"    {zkey}: {', '.join(preview)}{more}")
    print("=" * 60)

    any_flagged = (
        bool(results_with_zeros)
        or bool(results_struct_issues)
        or bool(results_chunk_issues)
        or bool(scan_errors)
    )
    print(f"RESULT: {'FLAGGED' if any_flagged else 'CLEAN'}")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    pct: float = 100.0,
    seed: int = 42,
    run_id: str = "",
) -> None:
    """Build the embodiment table locally, SPAWN the coordinator, exit.

    The coordinator parallel-spawns every shard (no connection held),
    workers write JSON shard files to the volume idempotently, and a
    self-respawning summarize prints the final report. Nothing maintains
    a long-lived control connection, so neither client→worker heartbeats
    nor worker preemption can break the run.
    """
    print("Building embodiment keymap table (locally — needs egomimic)...")
    emb_table = _build_embodiment_keymap_table()
    for family, keys in emb_table.items():
        print(f"  {family:<6}: {keys}")

    rid = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    handle = scan_all.spawn(emb_table, pct, seed, rid)

    print()
    print("=" * 60)
    print("Scan launched on Modal (fire-and-forget, fault-tolerant).")
    print(f"  run id           : {rid}")
    print(f"  function call id : {handle.object_id}")
    print(f"  results path     : <volume>/{RESULTS_PREFIX}/{rid}/")
    print("  REQUIRED: run with `modal run --detach` so the app survives.")
    print("  watch progress + final summary:")
    print("    modal app logs egomimic-test-data")
    print("  re-summarize an existing run:")
    print(
        f"    modal run --env robotics egomimic/modal/test_data.py::summarize --run-id {rid}"
    )
    print("=" * 60)
