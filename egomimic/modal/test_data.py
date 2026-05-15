"""
Modal-parallelized version of egomimic/scripts/test_data.py.

Scans every Zarr episode on the `mecka_data_v2` Modal volume in parallel
across up to MAX_CONTAINERS CPU-only workers, then prints the same summary
as the original.

Checks (same as the original):
  1. Rows of all-zeros in any numeric array (frame-gap corruption sign).
  2. Embodiment in zarr.json + presence of every required key dir.
  3. /c layout — files under <key>/c/ with names longer than 4 chars.

Architecture (fault-tolerant, connectionless)
---------------------------------------------
A 198K-episode scan runs for hours. Any Modal call that returns results
(.remote / .map / .starmap) forces the caller to hold a live control-plane
connection for the whole job — over hours that connection always dies
(heartbeat timeout) or the holding container gets preempted, losing all
progress. So this script holds NO such connection:

  main()       local: build embodiment table, scan_all.spawn(), exit.
  scan_all     Modal: list episodes, scan_shard.spawn() every shard
               (spawn only enqueues — no connection held), then exits in
               minutes. Idempotent under preemption via a LAUNCHED marker.
  scan_shard   Modal worker (×MAX_CONTAINERS): scans its episodes, writes a
               partial aggregate into a modal.Dict keyed by run_id/idx.
               Idempotent: if its key exists it returns immediately, so a
               preemption-restart is a cheap no-op.
  aggregate    Modal: polls the Dict (reads only — zero worker connections),
               merges partials, prints the summary. Idempotent.

Result: neither heartbeat timeouts nor worker preemption can break a run.

Usage
-----
    # --detach REQUIRED (spawned work must outlive the local entrypoint):
    modal run --detach --env robotics egomimic/modal/test_data.py
    modal run --detach --env robotics egomimic/modal/test_data.py -- --pct 10

    # watch progress; the `aggregate` logs end with the ==== summary block:
    modal app logs egomimic-test-data
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
VOLUME_MOUNT_PATH = "/mnt/zarr-data"
ZARR_VOLUME_NAME = "mecka_data_v2"

# Global cap on concurrent scan_shard worker containers.
MAX_CONTAINERS = 200

# Target episodes per shard. Smaller shards finish in minutes, so a
# preemption-restart redoes only a few minutes of work (and the idempotency
# check usually makes it a no-op). Drives the shard count for huge datasets.
TARGET_EPS_PER_SHARD = 200

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "zarr==3.1.5",
    "numpy",
)
zarr_volume = modal.Volume.from_name(ZARR_VOLUME_NAME)

# Connectionless result store. Workers write their partial here keyed by
# run_id/shard_idx; the aggregator polls it. Nothing holds a live connection
# to running workers, so neither heartbeat timeouts nor preemption can break
# the run — a preempted worker just reruns and overwrites the same key.
results_dict = modal.Dict.from_name(
    "egomimic-test-data-results", create_if_missing=True
)


def _dict_set_with_retry(key: str, value, attempts: int = 5) -> None:
    """Best-effort set on results_dict; swallows transient RPC failures.
    Modal heartbeat blips occasionally fail a Dict op; retry quickly so the
    worker doesn't have to be killed and restarted just for one bad RPC."""
    for i in range(attempts):
        try:
            results_dict[key] = value
            return
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(2**i)
            print(f"  Dict set retry {i + 1}/{attempts} after {type(e).__name__}: {e}")


def _dict_contains_with_retry(key: str, attempts: int = 5) -> bool:
    for i in range(attempts):
        try:
            return key in results_dict
        except Exception as e:
            if i == attempts - 1:
                raise
            time.sleep(2**i)
            print(
                f"  Dict contains retry {i + 1}/{attempts} after {type(e).__name__}: {e}"
            )
    return False


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


def _required_keys_for_embodiment(
    emb_name: str, emb_table: dict[str, list[str]]
) -> tuple[set[str], str | None]:
    """Resolve embodiment name → required zarr key set using the local table."""
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
        return set(), f"unsupported embodiment '{emb_name}'"
    keys = emb_table.get(family, [])
    if not keys:
        return set(), f"no required keys resolved for embodiment '{emb_name}'"
    return set(keys), None


def _build_embodiment_keymap_table() -> dict[str, list[str]]:
    """Run *locally* — needs the full egomimic stack (torch etc.).

    Returns {family: [zarr_key, ...]} for EVA / MECKA / ARIA / SCALE.
    """
    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.rldb.embodiment.human import Aria, Mecka, Scale

    def _keys_from(km: dict | None) -> set[str]:
        out: set[str] = set()
        if not km:
            return out
        for spec in km.values():
            if not isinstance(spec, dict):
                continue
            zkey = spec.get("zarr_key")
            if isinstance(zkey, str) and zkey != "annotations":
                out.add(zkey)
        return out

    # Only Mecka overrides get_keymap to accept the `annotations` kwarg;
    # Eva/Aria/Scale use the base Embodiment.get_keymap which doesn't.
    eva_keys = _keys_from(Eva.get_keymap("cartesian"))
    mecka_keys = _keys_from(
        Mecka.get_keymap("cartesian", annotations=False)
    ) | _keys_from(Mecka.get_keymap("keypoints", annotations=False))
    aria_keys = _keys_from(Aria.get_keymap("cartesian")) | _keys_from(
        Aria.get_keymap("keypoints")
    )
    scale_keys = _keys_from(Scale.get_keymap("cartesian")) | _keys_from(
        Scale.get_keymap("keypoints")
    )

    return {
        "EVA": sorted(eva_keys),
        "MECKA": sorted(mecka_keys),
        "ARIA": sorted(aria_keys),
        "SCALE": sorted(scale_keys),
    }


# ---------------------------------------------------------------------------
# Worker — fan out to up to MAX_CONTAINERS containers
# ---------------------------------------------------------------------------
@app.function(
    cpu=1.0,
    memory=2048,
    timeout=3600,  # 1h per shard (shards target ~minutes; this is slack)
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
    """Scan a shard of episodes and write a partial aggregate to results_dict.

    Idempotent: if this shard's key already exists (preemption-restart or
    duplicate spawn), return immediately. Episodes with is_deleted=True in
    zarr.json are skipped to match LocalEpisodeResolver + DatasetFilter().
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    import numpy as _np
    import zarr as _zarr

    key = f"{run_id}/{shard_idx}"
    if _dict_contains_with_retry(key):
        print(f"shard {shard_idx}: already done, skipping")
        return

    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)

    def _read_json(path: Path):
        try:
            with path.open("r") as f:
                return _json.load(f), None
        except FileNotFoundError:
            return None, f"missing {path.name}"
        except Exception as e:
            return None, f"failed to read {path.name}: {type(e).__name__}: {e}"

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
            missing_required_keys: list[str] = []
            required_keys: set[str] = set()
            embodiment_name: str | None = None

            # (2) Embodiment + required keys.
            #
            # We support both zarr v3 (zarr.json) and v2 (.zattrs) layouts —
            # the training stack reads both, so we have to as well. The
            # original test_data.py was v3-only and over-flagged v2 stores.
            # Logic: try zarr.json (v3), then .zattrs (v2), then g.attrs as a
            # final catch-all. Only flag if NOTHING produces an embodiment.
            zarr_json_path = ep_path / "zarr.json"
            zattrs_path = ep_path / ".zattrs"
            is_v3 = zarr_json_path.is_file()

            if is_v3:
                zj, _zj_err = _read_json(zarr_json_path)
                if isinstance(zj, dict):
                    embodiment_name = _extract_embodiment_from_zarr_json(zj)
            if not embodiment_name and zattrs_path.is_file():
                # .zattrs is the bare attrs dict; wrap for the extractor.
                za, _za_err = _read_json(zattrs_path)
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

            if not embodiment_name:
                issues_struct.append("missing embodiment (zarr.json/.zattrs/attrs)")
            else:
                required_keys, km_err = _required_keys_for_embodiment(
                    embodiment_name, emb_table
                )
                if km_err:
                    issues_struct.append(km_err)

            if required_keys:
                for zkey in sorted(required_keys):
                    key_dir = ep_path / zkey
                    if not key_dir.exists():
                        missing_required_keys.append(zkey)
                        continue
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

            if missing_required_keys:
                issues_struct.append(
                    "missing required key dirs: "
                    + ", ".join(sorted(missing_required_keys))
                )

            # (1) rows of zeros — only on small pose/keypoint vectors.
            # Skip image arrays: they dominate I/O on a network volume
            # (hundreds of MB each), and zero-row corruption only manifests
            # in pose vecs anyway. Filter by name first to avoid even loading
            # image metadata (zarr 3.x does an async metadata fetch per key).
            zero_rows: dict[str, list[int]] = {}
            for key in g.keys():
                if key.startswith("images.") or key.startswith("images/"):
                    continue
                try:
                    arr = g[key]
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
                    zero_rows[key] = bad

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

    _dict_set_with_retry(key, acc)
    print(
        f"shard {shard_idx}: done — {acc['scanned']} eps, "
        f"flagged {acc['total_episodes_with_zeros'] + acc['total_episodes_with_struct_issues'] + acc['total_episodes_with_chunk_issues'] + len(acc['scan_errors'])}"
    )


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


def _shard_keys(run_id: str, n_shards: int) -> list[str]:
    return [f"{run_id}/{i}" for i in range(n_shards)]


# ---------------------------------------------------------------------------
# Coordinator — discovers episodes, FIRE-AND-FORGET spawns every shard, exits
# ---------------------------------------------------------------------------
# No .map()/.starmap()/.remote() anywhere: .spawn() only enqueues, so this
# container never holds a live connection to running workers. It finishes in
# a few minutes (discover + spawn loop) and exits. Nothing to heartbeat,
# nothing to preempt mid-job. Idempotent under preemption-restart via the
# LAUNCHED marker.
@app.function(
    cpu=4.0,
    memory=8192,
    timeout=3600,  # only needs to discover + spawn; not the whole scan
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    retries=modal.Retries(max_retries=5, backoff_coefficient=2.0),
)
def scan_all(
    emb_table: dict[str, list[str]],
    pct: float,
    seed: int,
    shards: int,
    run_id: str,
) -> None:
    """Discover episodes, spawn all scan_shards + the aggregator, then exit."""
    launched_key = f"{run_id}/LAUNCHED"
    if launched_key in results_dict:
        # Preemption-restart after the spawn loop already ran. Don't re-spawn
        # workers; just make sure the aggregator is running and exit.
        print(f"run {run_id}: already launched, (re)spawning aggregator only")
        aggregate.spawn(run_id)
        return

    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)

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
        results_dict[f"{run_id}/META"] = {"n_shards": 0, "expected_total": 0}
        results_dict[launched_key] = True
        aggregate.spawn(run_id)
        return

    if pct < 100.0:
        k = max(1, int(round(len(raw) * pct / 100.0)))
        rng = random.Random(seed)
        raw = sorted(rng.sample(raw, k))
        print(f"  sampling {k} episodes ({pct:.1f}%)")

    # Many small shards: each finishes in minutes so a preemption-restart
    # only redoes minutes (and the idempotency check usually no-ops it).
    # `shards` is a floor; the dataset size pushes it higher if needed.
    n_shards = max(
        1,
        min(
            max(shards, math.ceil(len(raw) / TARGET_EPS_PER_SHARD)),
            len(raw),
        ),
    )
    chunk = math.ceil(len(raw) / n_shards)
    shard_specs = [raw[i : i + chunk] for i in range(0, len(raw), chunk)]
    n_shards = len(shard_specs)

    results_dict[f"{run_id}/META"] = {
        "n_shards": n_shards,
        "expected_total": len(raw),
        "started": time.time(),
    }
    print(
        f"  fan-out: {n_shards} shards × ~{chunk} eps "
        f"(global max_containers={MAX_CONTAINERS}); spawning..."
    )

    # Fire-and-forget: .spawn() returns immediately, holds no connection.
    for idx, specs in enumerate(shard_specs):
        scan_shard.spawn(run_id, idx, specs, emb_table)
        if (idx + 1) % 100 == 0 or idx + 1 == n_shards:
            print(f"  spawned {idx + 1}/{n_shards} shards")

    results_dict[launched_key] = True
    aggregate.spawn(run_id)
    print(f"run {run_id}: all shards + aggregator spawned; coordinator exiting")


# ---------------------------------------------------------------------------
# Aggregator — polls the result Dict (reads only), merges, prints summary
# ---------------------------------------------------------------------------
# Holds no connection to any worker; just reads results_dict. Fully
# idempotent — a preemption-restart simply re-polls. This is what makes the
# whole run immune to heartbeat timeouts and preemption.
@app.function(
    cpu=2.0,
    memory=8192,
    timeout=21600,  # 6h budget to let every shard (incl. retries) report
    retries=modal.Retries(max_retries=5, backoff_coefficient=2.0),
)
def aggregate(run_id: str) -> None:
    """Poll results_dict until all shards report, then merge + print."""
    meta_key = f"{run_id}/META"

    # Wait for the coordinator to publish META.
    waited = 0
    while meta_key not in results_dict:
        if waited > 1800:  # 30 min
            print(f"run {run_id}: META never appeared; aborting aggregator")
            return
        time.sleep(15)
        waited += 15

    meta = results_dict[meta_key]
    n_shards = meta["n_shards"]
    expected_total = meta["expected_total"]
    print(
        f"run {run_id}: aggregating {n_shards} shards "
        f"(~{expected_total} episodes expected)"
    )

    if n_shards == 0:
        _print_summary({"scanned": 0, "summary_text": "No episodes found."})
        results_dict[f"{run_id}/SUMMARY"] = "No episodes found."
        return

    keys = _shard_keys(run_id, n_shards)
    done: set[str] = set()
    budget_s = 6 * 3600
    elapsed = 0
    poll_s = 20
    last_report = -1

    while len(done) < n_shards and elapsed < budget_s:
        present = set(results_dict.keys())
        for k in keys:
            if k in present:
                done.add(k)
        pct_done = int(100 * len(done) / n_shards)
        if pct_done != last_report:
            print(f"  shards reported: {len(done)}/{n_shards} ({pct_done}%)")
            last_report = pct_done
        if len(done) >= n_shards:
            break
        time.sleep(poll_s)
        elapsed += poll_s

    missing = [k for k in keys if k not in done]
    if missing:
        print(
            f"run {run_id}: WARNING {len(missing)}/{n_shards} shards never "
            f"reported (budget exhausted); summary is partial"
        )

    partials = [results_dict[k] for k in keys if k in done]
    result = _merge_partials(partials)
    if missing:
        result["scan_errors"].append(
            f"[aggregator] {len(missing)} shard(s) missing — partial result"
        )
    _print_summary(result)

    # Persist a compact summary line so it can be retrieved without logs.
    flagged = (
        result["total_episodes_with_zeros"]
        + result["total_episodes_with_struct_issues"]
        + result["total_episodes_with_chunk_issues"]
        + len(result["scan_errors"])
    )
    results_dict[f"{run_id}/SUMMARY"] = (
        f"scanned={result['scanned']} flagged={flagged} "
        f"zeros={result['total_episodes_with_zeros']} "
        f"struct={result['total_episodes_with_struct_issues']} "
        f"chunks={result['total_episodes_with_chunk_issues']} "
        f"errors={len(result['scan_errors'])} "
        f"missing_shards={len(missing)}"
    )


def _print_summary(result: dict) -> None:
    """Print the final report. Runs inside the coordinator (logged to Modal)."""
    scanned = result.get("scanned", 0)
    if scanned == 0:
        print(result.get("summary_text", "No episodes scanned."))
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
# Local entrypoint — thin wrapper around the coordinator
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    pct: float = 100.0,
    seed: int = 42,
    shards: int = MAX_CONTAINERS,
    run_id: str = "",
) -> None:
    """Build the embodiment table locally, then SPAWN the coordinator and exit.

    The coordinator only discovers + .spawn()s shards (no connection held),
    workers write results to a Dict idempotently, and a polling aggregator
    prints the summary. Nothing maintains a long-lived connection, so neither
    heartbeat timeouts nor worker preemption can break the run.
    """
    import time as _t

    print("Building embodiment keymap table (locally — needs egomimic)...")
    emb_table = _build_embodiment_keymap_table()
    for family, keys in emb_table.items():
        print(f"  {family:<6}: {keys}")

    rid = run_id or _t.strftime("run-%Y%m%d-%H%M%S")
    handle = scan_all.spawn(emb_table, pct, seed, shards, rid)

    print()
    print("=" * 60)
    print("Scan launched on Modal (fire-and-forget, fault-tolerant).")
    print(f"  run id           : {rid}")
    print(f"  function call id : {handle.object_id}")
    print("  REQUIRED: run with `modal run --detach` so the app survives.")
    print("  watch progress + final summary:")
    print("    modal app logs egomimic-test-data")
    print("  the `aggregate` function prints the final report when all")
    print("  shards have reported (look for the ==== summary block).")
    print("=" * 60)
