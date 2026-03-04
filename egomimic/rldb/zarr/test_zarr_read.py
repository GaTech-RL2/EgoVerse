"""
Test script for verifying Zarr episodes written by eva_to_zarr.py.

Usage:
    # Test a single episode
    python test_zarr_read.py --zarr-path /path/to/episode.zarr

    # Test all episodes in a directory
    python test_zarr_read.py --zarr-dir /path/to/zarr_dataset/name/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    ZarrDataset,
    ZarrEpisode,
)

# Keys written by eva_to_zarr.py (values from DATASET_KEY_MAPPINGS)
EXPECTED_NUMERIC_KEYS = {"obs_eepose", "obs_joint", "cmd_eepose", "cmd_joint"}
EXPECTED_IMAGE_KEYS = {"front_img_1", "right_wrist_img", "left_wrist_img"}

# key_map for ZarrDataset: maps output names -> zarr key + optional horizon
KEY_MAP = {
    "obs/eepose": {"zarr_key": "obs_eepose"},
    "obs/joint": {"zarr_key": "obs_joint"},
    "cmd/eepose": {"zarr_key": "cmd_eepose"},
    "cmd/joint": {"zarr_key": "cmd_joint"},
    "img/front": {"zarr_key": "front_img_1"},
    "img/right_wrist": {"zarr_key": "right_wrist_img"},
    "img/left_wrist": {"zarr_key": "left_wrist_img"},
}


# ---------------------------------------------------------------------------
# Single-episode checks
# ---------------------------------------------------------------------------

def check_episode(zarr_path: Path) -> bool:
    """
    Run all checks on a single zarr episode. Returns True if all pass.
    """
    print(f"\n{'='*60}")
    print(f"Checking: {zarr_path.name}")
    print(f"{'='*60}")
    ok = True

    # --- 1. ZarrEpisode: metadata and raw keys ---
    try:
        ep = ZarrEpisode(zarr_path)
    except Exception as e:
        print(f"  [FAIL] Could not open ZarrEpisode: {e}")
        return False

    total_frames = ep.metadata.get("total_frames", 0)
    fps = ep.metadata.get("fps", "?")
    embodiment = ep.metadata.get("embodiment", "?")
    task = ep.metadata.get("task", "?")
    features = ep.metadata.get("features", {})

    print(f"  total_frames : {total_frames}")
    print(f"  fps          : {fps}")
    print(f"  embodiment   : {embodiment}")
    print(f"  task         : {task!r}")
    print(f"  stored keys  : {sorted(features.keys())}")

    # Presence checks
    present_keys = set(features.keys())
    for key in EXPECTED_NUMERIC_KEYS | EXPECTED_IMAGE_KEYS:
        if key in present_keys:
            dtype = features[key].get("dtype", "?")
            print(f"    [OK ] {key:30s}  dtype={dtype}")
        else:
            print(f"    [MISSING] {key}")
            ok = False

    # Language annotations (optional — written when --example-language-annotations is set)
    has_lang = "language_annotations" in ep._store
    if has_lang:
        n = ep._store["language_annotations"].shape[0]
        print(f"    [OK ] {'language_annotations':30s}  count={n}")
    else:
        print(f"    [INFO] language_annotations          not present")

    if total_frames == 0:
        print("  [FAIL] total_frames == 0")
        ok = False

    # --- 2. Raw read via ZarrEpisode ---
    print("\n  Raw reads (first frame):")
    for key in sorted(EXPECTED_NUMERIC_KEYS):
        if key not in present_keys:
            continue
        try:
            data = ep.read({key: (0, None)})
            arr = data[key]
            print(f"    [OK ] {key:30s}  shape={arr.shape}  dtype={arr.dtype}")
        except Exception as e:
            print(f"    [FAIL] {key}: {e}")
            ok = False

    for key in sorted(EXPECTED_IMAGE_KEYS):
        if key not in present_keys:
            continue
        try:
            data = ep.read({key: (0, None)})
            raw = data[key]
            # raw is JPEG bytes for single-frame read
            nbytes = len(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else raw.nbytes
            print(f"    [OK ] {key:30s}  bytes={nbytes}")
        except Exception as e:
            print(f"    [FAIL] {key}: {e}")
            ok = False

    # --- 3. ZarrDataset: frame-level __getitem__ ---
    print("\n  ZarrDataset __getitem__(0):")
    # Only include keys actually present; add language_annotations if stored
    key_map_filtered = {k: v for k, v in KEY_MAP.items() if v["zarr_key"] in present_keys}
    if has_lang:
        key_map_filtered["language"] = {"zarr_key": "language_annotations"}
    try:
        ds = ZarrDataset(zarr_path, key_map=key_map_filtered)
        assert len(ds) == total_frames, f"len(ds)={len(ds)} != total_frames={total_frames}"
        frame = ds[0]
        for out_key, val in frame.items():
            if isinstance(val, torch.Tensor):
                finite = torch.isfinite(val).all().item() if val.is_floating_point() else True
                tag = "OK " if finite else "WARN"
                print(f"    [{tag}] {out_key:22s}  shape={tuple(val.shape)}  dtype={val.dtype}")
            elif isinstance(val, np.ndarray):
                print(f"    [OK ] {out_key:22s}  shape={val.shape}  dtype={val.dtype}  (numpy)")
            elif isinstance(val, str):
                preview = val[:60] + "..." if len(val) > 60 else val
                print(f"    [OK ] {out_key:22s}  text={preview!r}")
            else:
                # language_annotations must always resolve to str, never list/bytes
                if out_key == "language":
                    print(f"    [FAIL] {out_key:22s}  expected str, got {type(val).__name__}")
                    ok = False
                else:
                    print(f"    [OK ] {out_key:22s}  type={type(val).__name__}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        import traceback; traceback.print_exc()
        ok = False

    # --- 4. Language annotation span contents + spot-check span matching ---
    if has_lang:
        print(f"\n  Language annotation spans:")
        try:
            spans = ds._load_language_annotations()
            for i, ann in enumerate(spans):
                print(f"    [{i}] text={ann.get('text','')!r}  "
                      f"start={ann.get('start_idx')}  end={ann.get('end_idx')}")

            # Spot-check: for each span, frame at start_idx and end_idx must
            # return a str containing the annotation text.
            print(f"\n  Span matching spot-checks:")
            for ann in spans:
                text = ann.get("text", "")
                s, e = int(ann.get("start_idx", 0)), int(ann.get("end_idx", 0))
                for label, fidx in [("start", s), ("end", e)]:
                    result = ds._annotation_text_for_frame(fidx)
                    assert isinstance(result, str), f"Expected str, got {type(result)}"
                    assert text in result, (
                        f"Frame {fidx} ({label} of span): expected {text!r} in {result!r}"
                    )
                    print(f"    [OK ] frame={fidx:4d} ({label})  ->  {result!r}")

            # A frame beyond all spans should return an empty string
            beyond = total_frames  # one past the last valid index
            result = ds._annotation_text_for_frame(beyond)
            assert result == "", f"Expected '' for out-of-span frame {beyond}, got {result!r}"
            print(f"    [OK ] frame={beyond:4d} (out-of-span)  ->  {result!r}")

        except AssertionError as e:
            print(f"    [FAIL] {e}")
            ok = False
        except Exception as e:
            print(f"    [FAIL] {e}")
            import traceback; traceback.print_exc()
            ok = False

    # --- 5. ZarrDataset: last frame ---
    print(f"\n  ZarrDataset __getitem__({total_frames - 1}) (last frame):")
    try:
        frame_last = ds[total_frames - 1]
        for out_key, val in frame_last.items():
            if isinstance(val, torch.Tensor):
                finite = torch.isfinite(val).all().item() if val.is_floating_point() else True
                status = "OK " if finite else "WARN"
                print(f"    [{status}] {out_key:22s}  shape={tuple(val.shape)}")
            elif isinstance(val, str):
                preview = val[:60] + "..." if len(val) > 60 else val
                print(f"    [OK ] {out_key:22s}  text={preview!r}")
            else:
                print(f"    [OK ] {out_key:22s}  type={type(val).__name__}")
    except Exception as e:
        print(f"    [FAIL] {e}")
        ok = False

    status_str = "PASS" if ok else "FAIL"
    print(f"\n  Result: [{status_str}]")
    return ok


# ---------------------------------------------------------------------------
# Directory-level checks via LocalEpisodeResolver + MultiDataset
# ---------------------------------------------------------------------------

def check_directory(zarr_dir: Path) -> bool:
    """
    Load all zarr episodes in zarr_dir via LocalEpisodeResolver and MultiDataset.
    """
    print(f"\n{'='*60}")
    print(f"Directory check: {zarr_dir}")
    print(f"{'='*60}")

    zarr_paths = sorted(zarr_dir.glob("*.zarr"))
    if not zarr_paths:
        print("  No .zarr files found.")
        return False

    print(f"  Found {len(zarr_paths)} .zarr episode(s)")

    # Per-episode checks
    results = {}
    for p in zarr_paths:
        results[p.name] = check_episode(p)

    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Episode summary: {passed}/{total} passed")
    for name, ok in sorted(results.items()):
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}")

    if total == 0:
        return False

    # LocalEpisodeResolver + MultiDataset (train split)
    print(f"\n  LocalEpisodeResolver + MultiDataset test...")
    try:
        resolver = LocalEpisodeResolver(
            folder_path=zarr_dir,
            key_map=KEY_MAP,
        )
        multi = MultiDataset._from_resolver(
            resolver,
            filters={},          # no extra filters beyond is_deleted=False default
            mode="total",        # use all episodes
        )
        print(f"  [OK ] MultiDataset total items: {len(multi)}")
        sample = multi[0]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"         {k:22s}  shape={tuple(v.shape)}")
            elif isinstance(v, str):
                preview = v[:60] + "..." if len(v) > 60 else v
                print(f"         {k:22s}  text={preview!r}")
            else:
                print(f"         {k:22s}  type={type(v).__name__}")
    except Exception as e:
        print(f"  [FAIL] MultiDataset: {e}")
        import traceback; traceback.print_exc()

    return passed == total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify Zarr episodes from eva_to_zarr.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zarr-path", type=Path, help="Path to a single .zarr episode")
    group.add_argument("--zarr-dir",  type=Path, help="Directory containing .zarr episodes")
    args = parser.parse_args()

    if args.zarr_path:
        ok = check_episode(args.zarr_path)
    else:
        ok = check_directory(args.zarr_dir)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
