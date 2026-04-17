"""
Zero-row scanner for local Zarr episodes.

For every numeric array in each episode, finds frame indices where all values
are exactly zero.  Zero quaternions (0,0,0,0) embedded in a pose vec are the
classic sign of data corruption this was designed to catch.

Uses LocalEpisodeResolver to discover only valid, readable zarr stores.

Fast by design:
  - ThreadPool parallel reads (mostly I/O-bound, no pickling overhead).
  - Reads each array as a single numpy slice then checks with .all(axis=1).
  - tqdm bar advances per episode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Aria, Mecka, Scale
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver

# ── per-episode worker ────────────────────────────────────────────────────────


def _safe_int(s: str) -> int | None:
    try:
        return int(s)
    except Exception:
        return None


def _read_zarr_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        with path.open("r") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"missing {path.name}"
    except Exception as e:
        return None, f"failed to read {path.name}: {type(e).__name__}: {e}"


def _extract_embodiment_from_zarr_json(zarr_json: dict) -> str | None:
    """
    Attempt to find an embodiment identifier in Zarr v3 JSON metadata.
    Returns a normalized embodiment name like 'MECKA_BIMANUAL' when possible.
    """
    candidates: list[object] = []

    # Common patterns: top-level attributes or nested metadata.
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
        # numeric embodiment id
        if isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()):
            eid = int(v)
            name = get_embodiment(eid)
            if isinstance(name, str):
                return name.upper()
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            return s.upper()
    return None


def _required_zarr_keys_for_embodiment(
    embodiment_name: str,
) -> tuple[set[str], str | None]:
    """
    From embodiment name, return the set of required Zarr array keys (folder names).
    'annotations' is treated as optional and excluded.
    """
    emb = str(embodiment_name).strip().upper().replace("-", "_").replace(" ", "_")
    # Normalize aliases that should map to SCALE bimanual defaults.
    if emb in {"SCALE", "SCALE_BIMANUAL"}:
        emb = "SCALE_BIMANUAL"
    required: set[str] = set()

    def _add_from_keymap(km: dict):
        for _, spec in (km or {}).items():
            if not isinstance(spec, dict):
                continue
            zkey = spec.get("zarr_key", None)
            if not isinstance(zkey, str):
                continue
            if zkey == "annotations":
                continue
            required.add(zkey)

    try:
        if emb.startswith("EVA_"):
            _add_from_keymap(Eva.get_keymap())
        elif emb.startswith("MECKA_"):
            _add_from_keymap(Mecka.get_keymap("cartesian", annotations=False))
            _add_from_keymap(Mecka.get_keymap("keypoints", annotations=False))
        elif emb.startswith("ARIA_"):
            _add_from_keymap(Aria.get_keymap("cartesian", annotations=False))
            _add_from_keymap(Aria.get_keymap("keypoints", annotations=False))
        elif emb.startswith("SCALE_"):
            _add_from_keymap(Scale.get_keymap("cartesian", annotations=False))
            _add_from_keymap(Scale.get_keymap("keypoints", annotations=False))
        else:
            # If it's a valid enum name, we still might not have a keymap class here.
            # Keep it as unsupported for this validator rather than guessing.
            return set(), f"unsupported embodiment '{embodiment_name}'"
    except Exception as e:
        return (
            set(),
            f"failed to resolve keymap for embodiment '{embodiment_name}': {type(e).__name__}: {e}",
        )

    if not required:
        return set(), f"no required keys resolved for embodiment '{embodiment_name}'"
    return required, None


def _validate_c_layout(key_dir: Path) -> list[str]:
    """
    Enforce:
      - key_dir/c exists and is a directory
      - inside c/: any file/folder name longer than 4 chars is flagged
      - ignore zarr.json directly under c/
    """
    issues: list[str] = []
    cdir = key_dir / "c"
    if not cdir.exists():
        issues.append("missing c/")
        return issues
    if not cdir.is_dir():
        issues.append("c/ is not a directory")
        return issues

    try:
        entries = list(cdir.iterdir())
    except Exception as e:
        issues.append(f"cannot list c/: {type(e).__name__}: {e}")
        return issues

    for ent in entries:
        if ent.name == "zarr.json":
            continue
        if len(ent.name) > 4:
            issues.append(f"name length > 4 under c/: {ent.name}")
    return issues


def _scan_episode(ep_path: Path, episode_hash: str) -> dict:
    """
    Open the zarr store, run checks, and return structured findings.
    """
    eh = episode_hash
    try:
        g = zarr.open_group(str(ep_path), mode="r")
        total_frames = int(g.attrs.get("total_frames", 0) or 0)

        issues_struct: list[str] = []
        issues_chunks: dict[str, list[str]] = {}
        missing_required_keys: list[str] = []
        required_keys: set[str] = set()
        embodiment_name: str | None = None

        # (2) Zarr root zarr.json + embodiment + required key presence
        zarr_json_path = ep_path / "zarr.json"
        zarr_json, zarr_json_err = _read_zarr_json(zarr_json_path)
        if zarr_json_err:
            issues_struct.append(zarr_json_err)
        if isinstance(zarr_json, dict):
            embodiment_name = _extract_embodiment_from_zarr_json(zarr_json)
        if not embodiment_name:
            # fallback to zarr attrs (still flag since user asked to use zarr.json)
            attr_emb = g.attrs.get("embodiment", None)
            if attr_emb is not None:
                try:
                    if isinstance(attr_emb, int):
                        embodiment_name = get_embodiment(attr_emb)
                    elif isinstance(attr_emb, str):
                        embodiment_name = attr_emb
                except Exception:
                    embodiment_name = None
            issues_struct.append("missing embodiment in zarr.json")
        else:
            required_keys, km_err = _required_zarr_keys_for_embodiment(embodiment_name)
            if km_err:
                issues_struct.append(km_err)

        if required_keys:
            for zkey in sorted(required_keys):
                key_dir = ep_path / zkey
                if not key_dir.exists():
                    missing_required_keys.append(zkey)
                    continue
                if not key_dir.is_dir():
                    issues_struct.append(f"{zkey}: expected directory, found non-dir")
                    continue
                key_zj, key_zj_err = _read_zarr_json(key_dir / "zarr.json")
                if key_zj_err:
                    issues_struct.append(f"{zkey}: {key_zj_err}")
                # (3) /c layout constraints
                c_issues = _validate_c_layout(key_dir)
                if c_issues:
                    issues_chunks[zkey] = c_issues

        if missing_required_keys:
            issues_struct.append(
                "missing required key dirs: " + ", ".join(sorted(missing_required_keys))
            )

        # (1) Check for rows of zeros (frame gap issue per your note)
        zero_rows: dict[str, list[int]] = {}
        for key in g.keys():
            arr = g[key]
            if arr.ndim < 2 or not np.issubdtype(arr.dtype, np.number):
                continue
            data: np.ndarray = arr[:]
            T = data.shape[0]
            flat = data.reshape(T, -1)
            zero_mask = (flat == 0).all(axis=1)
            bad = np.where(zero_mask)[0].tolist()
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local Zarr episodes (zero rows, key presence, /c layout)."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
        help="Directory containing episode dirs.",
    )
    parser.add_argument(
        "--pct",
        default=100.0,
        type=float,
        help="Percentage of episodes to scan (default 100 = all).",
    )
    parser.add_argument(
        "--workers",
        default=32,
        type=int,
        help="Parallel threads (default 32).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"Error: not a directory: {dataset_root}")
        return 2

    # Use the resolver to enumerate only valid, readable zarr stores.
    print("Resolving valid episodes...")
    raw = LocalEpisodeResolver._get_local_filtered_paths(
        dataset_root,
        filters=DatasetFilter(),
    )
    # raw is a list of (path_str, episode_hash)
    if not raw:
        print("No valid zarr episodes found.")
        return 2

    # Optional episode sampling
    if args.pct < 100.0:
        k = max(1, int(round(len(raw) * args.pct / 100.0)))
        rng = random.Random(args.seed)
        raw = sorted(rng.sample(raw, k))
        print(f"Sampling {k} / {len(raw)} episodes ({args.pct:.1f}%).")

    eps = [(Path(path_str), eh) for path_str, eh in raw]
    print(f"Scanning {len(eps)} episodes with {args.workers} threads...")

    # ── parallel scan ─────────────────────────────────────────────────────────
    total_episodes_with_zeros = 0
    total_zero_frames = 0
    total_episodes_with_struct_issues = 0
    total_episodes_with_chunk_issues = 0
    scan_errors: list[str] = []

    # {episode_hash: {key: [bad_indices]}}
    results_with_zeros: dict[str, dict[str, list[int]]] = {}
    results_struct_issues: dict[str, list[str]] = {}
    results_chunk_issues: dict[str, dict[str, list[str]]] = {}

    workers = max(1, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_scan_episode, ep_path, eh): eh for ep_path, eh in eps}
        pbar = tqdm(
            total=len(eps),
            desc="scanning",
            unit="ep",
            dynamic_ncols=True,
        )
        for fut in concurrent.futures.as_completed(future_map):
            pbar.update(1)
            res = fut.result()
            eh = res["episode_hash"]

            if res["error"]:
                scan_errors.append(f"{eh}: {res['error']}")
                tqdm.write(f"[ERROR] {eh}: {res['error']}")
                continue

            if res.get("issues_struct"):
                total_episodes_with_struct_issues += 1
                results_struct_issues[eh] = list(res["issues_struct"])
                tqdm.write(
                    f"[STRUCT] ep={eh}  "
                    + "; ".join(res["issues_struct"][:6])
                    + (" ..." if len(res["issues_struct"]) > 6 else "")
                )

            if res.get("issues_chunks"):
                total_episodes_with_chunk_issues += 1
                results_chunk_issues[eh] = dict(res["issues_chunks"])
                # compact one-liner: show up to 3 keys with issue-count
                keys_preview = ", ".join(
                    f"{k}({len(v)})" for k, v in list(res["issues_chunks"].items())[:3]
                )
                more = (
                    f" (+{len(res['issues_chunks']) - 3} keys)"
                    if len(res["issues_chunks"]) > 3
                    else ""
                )
                tqdm.write(f"[CHUNKS] ep={eh}  {keys_preview}{more}")

            if res["zero_rows"]:
                total_episodes_with_zeros += 1
                results_with_zeros[eh] = res["zero_rows"]
                # Count unique bad frame indices across all keys
                all_bad = set()
                for bad_idx in res["zero_rows"].values():
                    all_bad.update(bad_idx)
                n_bad = len(all_bad)
                total_zero_frames += n_bad
                keys_str = ", ".join(
                    f"{k}({len(v)})" for k, v in res["zero_rows"].items()
                )
                tqdm.write(
                    f"[ZEROS] ep={eh}  "
                    f"{n_bad} frame(s) with all-zero rows  [{keys_str}]"
                )
    pbar.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Episodes scanned       : {len(eps)}")
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
            all_bad = sorted(set().union(*zero_rows.values()))
            keys_str = ", ".join(zero_rows.keys())
            print(f"  {eh}")
            print(f"    keys   : {keys_str}")
            print(
                f"    frames : {len(all_bad)}  {all_bad[:20]}{'...' if len(all_bad) > 20 else ''}"
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
    return 1 if any_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
