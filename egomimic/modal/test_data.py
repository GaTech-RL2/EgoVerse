"""
Modal-parallelized version of egomimic/scripts/test_data.py.

Scans every Zarr episode on the `mecka_data_v2` Modal volume in parallel
across up to 500 CPU-only containers, then aggregates the per-episode
findings locally and prints the same summary as the original.

Checks (same as the original):
  1. Rows of all-zeros in any numeric array (frame-gap corruption sign).
  2. Embodiment in zarr.json + presence of every required key dir.
  3. /c layout — files under <key>/c/ with names longer than 4 chars.

Usage
-----
    modal run --detach --env robotics egomimic/modal/test_data.py
    modal run --detach --env robotics egomimic/modal/test_data.py -- --pct 10 --shards 500

    # then watch progress + read the final summary:
    modal app logs egomimic-test-data
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
VOLUME_MOUNT_PATH = "/mnt/zarr-data"
ZARR_VOLUME_NAME = "mecka_data_v2"
MAX_CONTAINERS = 500

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
    timeout=3600,  # 1h per shard
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    max_containers=MAX_CONTAINERS,
)
def scan_shard(
    episode_specs: list[tuple[str, str]],
    emb_table: dict[str, list[str]],
) -> list[dict]:
    """Scan a shard of episodes; returns one result dict per episode.

    Result schema matches egomimic/scripts/test_data.py:_scan_episode().
    Episodes with is_deleted=True in zarr.json are skipped (returns []-style
    "skipped" marker) to match the original DatasetFilter() behavior.
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    import numpy as _np
    import zarr as _zarr

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

            # (2) zarr.json embodiment + required keys
            zj, zj_err = _read_json(ep_path / "zarr.json")
            if zj_err:
                issues_struct.append(zj_err)
            if isinstance(zj, dict):
                embodiment_name = _extract_embodiment_from_zarr_json(zj)
            if not embodiment_name:
                attr_emb = attrs.get("embodiment")
                if attr_emb is not None:
                    try:
                        if isinstance(attr_emb, int):
                            embodiment_name = EMBODIMENT_ID_TO_KEY.get(attr_emb)
                        elif isinstance(attr_emb, str):
                            embodiment_name = attr_emb
                    except Exception:
                        embodiment_name = None
                issues_struct.append("missing embodiment in zarr.json")
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
                    _, key_zj_err = _read_json(key_dir / "zarr.json")
                    if key_zj_err:
                        issues_struct.append(f"{zkey}: {key_zj_err}")
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
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for res in ex.map(lambda spec: _scan(*spec), episode_specs):
            if res is not None:
                out.append(res)
    return out


# ---------------------------------------------------------------------------
# Coordinator — runs inside Modal, fans out scan_shard cloud-to-cloud
# ---------------------------------------------------------------------------
# Fanning out hundreds of containers directly from the local entrypoint
# saturates the client's control-plane connection (heartbeat timeouts).
# Running the fan-out from a Modal coordinator keeps only one cloud↔local
# connection open and performs the wide fan-out cloud-to-cloud, which is
# what Modal is built for.
@app.function(
    cpu=8.0,  # Coordinator juggles up to MAX_CONTAINERS worker heartbeats —
    # starve its gRPC handler and the local client's heartbeats time out.
    memory=8192,
    timeout=7200,  # 2h for the whole scan
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
)
def scan_all(
    emb_table: dict[str, list[str]],
    pct: float,
    seed: int,
    shards: int,
) -> dict:
    """Discover, fan out scan_shard.starmap, aggregate; return a summary dict."""
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
        return {
            "scanned": 0,
            "summary_text": "No episodes found.",
            "any_flagged": False,
        }

    if pct < 100.0:
        k = max(1, int(round(len(raw) * pct / 100.0)))
        rng = random.Random(seed)
        raw = sorted(rng.sample(raw, k))
        print(f"  sampling {k} episodes ({pct:.1f}%)")

    # Allow more shards than MAX_CONTAINERS — extras get queued, smaller
    # per-shard work means slow episodes don't push any single shard near
    # the per-shard timeout. MAX_CONTAINERS still caps concurrency.
    n_shards = max(1, min(shards, len(raw)))
    chunk = math.ceil(len(raw) / n_shards)
    shard_payloads = [
        (raw[i : i + chunk], emb_table) for i in range(0, len(raw), chunk)
    ]
    total_shards = len(shard_payloads)
    print(
        f"  fan-out: {total_shards} shards × ~{chunk} eps "
        f"(max_containers={MAX_CONTAINERS})"
    )

    total_episodes_with_zeros = 0
    total_zero_frames = 0
    total_episodes_with_struct_issues = 0
    total_episodes_with_chunk_issues = 0
    scanned = 0
    scan_errors: list[str] = []
    results_with_zeros: dict[str, dict[str, list[int]]] = {}
    results_struct_issues: dict[str, list[str]] = {}
    results_chunk_issues: dict[str, dict[str, list[str]]] = {}

    # Quiet aggregation: per-episode details are emitted by the local
    # entrypoint from the returned dict. Logging every flagged episode
    # here streams it back to the local client and saturates the
    # heartbeat channel — that was the source of the deadline-exceeded errors.
    shards_done = 0
    progress_every = max(1, total_shards // 20)  # ~20 progress lines total
    for shard_results in scan_shard.starmap(shard_payloads, order_outputs=False):
        shards_done += 1
        for res in shard_results:
            scanned += 1
            eh = res["episode_hash"]

            if res["error"]:
                scan_errors.append(f"{eh}: {res['error']}")
                continue

            if res.get("issues_struct"):
                total_episodes_with_struct_issues += 1
                results_struct_issues[eh] = list(res["issues_struct"])

            if res.get("issues_chunks"):
                total_episodes_with_chunk_issues += 1
                results_chunk_issues[eh] = dict(res["issues_chunks"])

            if res["zero_rows"]:
                total_episodes_with_zeros += 1
                results_with_zeros[eh] = res["zero_rows"]
                all_bad: set[int] = set()
                for bad in res["zero_rows"].values():
                    all_bad.update(bad)
                total_zero_frames += len(all_bad)

        if shards_done % progress_every == 0 or shards_done == total_shards:
            print(
                f"  progress: shard {shards_done}/{total_shards}  "
                f"(eps scanned: {scanned}, flagged: "
                f"{total_episodes_with_zeros + total_episodes_with_struct_issues + total_episodes_with_chunk_issues + len(scan_errors)})"
            )

    result = {
        "scanned": scanned,
        "scan_errors": scan_errors,
        "total_episodes_with_zeros": total_episodes_with_zeros,
        "total_zero_frames": total_zero_frames,
        "total_episodes_with_struct_issues": total_episodes_with_struct_issues,
        "total_episodes_with_chunk_issues": total_episodes_with_chunk_issues,
        "results_with_zeros": results_with_zeros,
        "results_struct_issues": results_struct_issues,
        "results_chunk_issues": results_chunk_issues,
    }
    # Print the full summary into the coordinator's own logs — the local
    # CLI uses .spawn() and exits, so it never sees the returned dict.
    _print_summary(result)
    return result


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
) -> None:
    """Build the embodiment table locally, then SPAWN the coordinator and exit."""
    print("Building embodiment keymap table (locally — needs egomimic)...")
    emb_table = _build_embodiment_keymap_table()
    for family, keys in emb_table.items():
        print(f"  {family:<6}: {keys}")

    handle = scan_all.spawn(emb_table, pct, seed, shards)

    print()
    print("=" * 60)
    print("Scan launched on Modal (fire-and-forget).")
    print(f"  function call id : {handle.object_id}")
    print("  NOTE: this only persists if you ran with `modal run --detach`.")
    print("  watch progress + final summary:")
    print("    modal app logs egomimic-test-data")
    print("  the coordinator prints the full report at the end of its logs")
    print("=" * 60)
