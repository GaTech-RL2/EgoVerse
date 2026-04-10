#!/usr/bin/env python3
"""
Validate robot zarr episodes for training compatibility (new per-arm format).

Tests every layer: zarr read -> ZarrEpisode -> ZarrDataset -> transforms -> model input.
Run after robot_to_egoverse_zarr.py to verify data is training-ready.

Usage:
    python validate_robot_zarr.py /tmp/robot_zarr_test/2026-03-12/
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import simplejpeg
import torch

sys.path.insert(0, str(Path(__file__).parent))


NUMERIC_KEYS = {
    "left.obs_ee_pose": {"shape": [7], "dtype": "float32"},
    "right.obs_ee_pose": {"shape": [7], "dtype": "float32"},
    "left.cmd_ee_pose": {"shape": [7], "dtype": "float32"},
    "right.cmd_ee_pose": {"shape": [7], "dtype": "float32"},
    "left.obs_joints": {"shape": [6], "dtype": "float32"},
    "right.obs_joints": {"shape": [6], "dtype": "float32"},
    "left.cmd_joints": {"shape": [6], "dtype": "float32"},
    "right.cmd_joints": {"shape": [6], "dtype": "float32"},
    "left.gripper": {"shape": [1], "dtype": "float32"},
    "right.gripper": {"shape": [1], "dtype": "float32"},
}

IMAGE_KEYS = [
    "observations.images.front_img_1",
    "observations.images.cam_low",
    "observations.images.left_wrist_img",
    "observations.images.right_wrist_img",
]


def validate_episode(zarr_path: str, verbose: bool = True) -> list[str]:
    """Validate a single zarr episode. Returns list of errors (empty = pass)."""
    errors = []
    zarr_path = Path(zarr_path)

    # ---------------------------------------------------------------
    # Layer 1: Raw zarr read
    # ---------------------------------------------------------------
    try:
        import zarr
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        return [f"FATAL: Cannot open zarr store: {e}"]

    attrs = dict(store.attrs)

    for key in ["embodiment", "total_frames", "fps", "features"]:
        if key not in attrs:
            errors.append(f"Missing metadata key: {key}")

    if errors:
        return errors

    total_frames = attrs["total_frames"]
    features = attrs["features"]

    if verbose:
        print(f"  Metadata: embodiment={attrs['embodiment']}, "
              f"frames={total_frames}, fps={attrs['fps']}")

    if attrs["embodiment"] != "scale_aloha_bimanual":
        errors.append(f"Unexpected embodiment: {attrs['embodiment']}")

    # ---------------------------------------------------------------
    # Layer 2: Check per-arm numeric keys exist with correct shapes
    # ---------------------------------------------------------------
    for key, expected in NUMERIC_KEYS.items():
        if key not in features:
            errors.append(f"Missing feature key: {key}")
            continue

        info = features[key]
        if info["shape"] != expected["shape"]:
            errors.append(f"{key}: shape {info['shape']} != expected {expected['shape']}")
        if info["dtype"] != expected["dtype"]:
            errors.append(f"{key}: dtype {info['dtype']} != expected {expected['dtype']}")

        arr = store[key]
        if arr.shape[0] < total_frames:
            errors.append(f"{key}: array length {arr.shape[0]} < total_frames {total_frames}")

        data = arr[0:min(5, total_frames)]
        if np.any(np.isnan(data)):
            errors.append(f"{key}: contains NaN values")
        if np.any(np.isinf(data)):
            errors.append(f"{key}: contains Inf values")

    # Check image keys
    for key in IMAGE_KEYS:
        if key not in features:
            errors.append(f"Missing image key: {key}")
            continue

        info = features[key]
        if info["dtype"] != "jpeg":
            errors.append(f"{key}: dtype should be 'jpeg', got {info['dtype']}")

        arr = store[key]
        batch = arr[0:1]
        jpeg_bytes = batch[0]
        try:
            img = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
            if img.ndim != 3 or img.shape[2] != 3:
                errors.append(f"{key}: decoded image shape {img.shape} is not (H, W, 3)")
            elif verbose:
                print(f"  {key}: image shape {img.shape}")
        except Exception as e:
            errors.append(f"{key}: JPEG decode failed: {e}")

    # Verify old combined keys are NOT present
    for old_key in [
        "observations.state.joint_positions",
        "observations.state.ee_pose",
        "actions_joints",
        "actions_cartesian",
    ]:
        if old_key in features:
            errors.append(f"Old combined key '{old_key}' should not exist in new format")

    # ---------------------------------------------------------------
    # Layer 3: Value range checks (per-arm format)
    # ---------------------------------------------------------------
    for side in ("left", "right"):
        # EE pose: (T, 7) = [xyz_base(3), wxyz_quat(4)]
        ee_key = f"{side}.obs_ee_pose"
        cmd_key = f"{side}.cmd_ee_pose"
        if ee_key in features and cmd_key in features:
            ee = store[ee_key][0:total_frames]
            cmd = store[cmd_key][0:total_frames]

            xyz_ee = ee[:, :3]
            xyz_cmd = cmd[:, :3]
            if np.abs(xyz_ee).max() > 5.0:
                errors.append(f"{ee_key} xyz max {np.abs(xyz_ee).max():.3f} > 5m")
            if np.abs(xyz_cmd).max() > 5.0:
                errors.append(f"{cmd_key} xyz max {np.abs(xyz_cmd).max():.3f} > 5m")

            # Quaternion unit norm check
            quat_ee = ee[:, 3:7]
            norms = np.linalg.norm(quat_ee, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                errors.append(
                    f"{ee_key} quaternion norms not unit: "
                    f"range [{norms.min():.6f}, {norms.max():.6f}]"
                )

            quat_cmd = cmd[:, 3:7]
            norms_cmd = np.linalg.norm(quat_cmd, axis=-1)
            if not np.allclose(norms_cmd, 1.0, atol=1e-4):
                errors.append(
                    f"{cmd_key} quaternion norms not unit: "
                    f"range [{norms_cmd.min():.6f}, {norms_cmd.max():.6f}]"
                )

            # Obs vs cmd should be close (leader-follower)
            diff = np.abs(ee - cmd).mean()
            if verbose:
                print(f"  {side} ee_pose: xyz=[{xyz_ee.min():.3f}, {xyz_ee.max():.3f}], "
                      f"obs-cmd diff={diff:.4f}")

        # Joints: (T, 6)
        obs_j_key = f"{side}.obs_joints"
        cmd_j_key = f"{side}.cmd_joints"
        if obs_j_key in features and cmd_j_key in features:
            obs_j = store[obs_j_key][0:total_frames]
            cmd_j = store[cmd_j_key][0:total_frames]
            if np.abs(obs_j).max() > 10:
                errors.append(f"{obs_j_key}: max abs value {np.abs(obs_j).max():.2f} > 10")
            jdiff = np.abs(obs_j - cmd_j).mean()
            if verbose:
                print(f"  {side} joints: range=[{obs_j.min():.3f}, {obs_j.max():.3f}], "
                      f"obs-cmd diff={jdiff:.4f}")

        # Gripper: (T, 1)
        grip_key = f"{side}.gripper"
        if grip_key in features:
            grip = store[grip_key][0:total_frames]
            if verbose:
                print(f"  {side} gripper: range=[{grip.min():.3f}, {grip.max():.3f}]")

    # ---------------------------------------------------------------
    # Layer 4: ZarrEpisode reader
    # ---------------------------------------------------------------
    try:
        from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode
        ep = ZarrEpisode(str(zarr_path))

        if len(ep) != total_frames:
            errors.append(f"ZarrEpisode len {len(ep)} != total_frames {total_frames}")

        read_result = ep.read({
            "left.obs_ee_pose": (0, None),
            "right.cmd_ee_pose": (0, None),
            "observations.images.front_img_1": (0, None),
        })

        if read_result["left.obs_ee_pose"].shape != (7,):
            errors.append(
                f"ZarrEpisode left.obs_ee_pose shape "
                f"{read_result['left.obs_ee_pose'].shape} != (7,)"
            )

        img_frame = read_result["observations.images.front_img_1"]
        decoded = simplejpeg.decode_jpeg(img_frame, colorspace="RGB")
        if verbose:
            print(f"  ZarrEpisode: OK (read single frame, image {decoded.shape})")

    except Exception as e:
        errors.append(f"ZarrEpisode failed: {e}")

    # ---------------------------------------------------------------
    # Layer 5: ZarrDataset with Scale Aloha keymap + transforms
    # ---------------------------------------------------------------
    try:
        from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
        from egomimic.rldb.embodiment.scale_aloha import ScaleAloha

        keymap = ScaleAloha.get_keymap()
        transforms = ScaleAloha.get_transform_list()

        ds = ZarrDataset(
            Episode_path=zarr_path,
            key_map=keymap,
            transform_list=transforms,
        )

        if len(ds) != total_frames:
            errors.append(f"ZarrDataset len {len(ds)} != total_frames {total_frames}")

        sample = ds[0]

        # After transforms, must have concatenated keys
        if "actions_cartesian" not in sample:
            errors.append("Post-transform 'actions_cartesian' missing")
        elif sample["actions_cartesian"].shape != (100, 14):
            errors.append(
                f"actions_cartesian shape {tuple(sample['actions_cartesian'].shape)} "
                f"!= (100, 14)"
            )
        elif torch.isnan(sample["actions_cartesian"]).any():
            errors.append("actions_cartesian contains NaN after transforms")

        if "observations.state.ee_pose" not in sample:
            errors.append("Post-transform 'observations.state.ee_pose' missing")
        elif sample["observations.state.ee_pose"].shape != (14,):
            errors.append(
                f"ee_pose shape {tuple(sample['observations.state.ee_pose'].shape)} "
                f"!= (14,)"
            )

        for img_key in IMAGE_KEYS:
            if img_key in sample:
                val = sample[img_key]
                if val.ndim != 3 or val.shape[0] != 3:
                    errors.append(f"{img_key} shape {tuple(val.shape)}, expected (3, H, W)")

        # Intermediate keys should be deleted
        for gone_key in ["left.cmd_ee_pose", "right.cmd_ee_pose",
                         "left_extrinsics_pose", "right_extrinsics_pose"]:
            if gone_key in sample:
                errors.append(f"Intermediate key '{gone_key}' should be deleted")

        # Test multiple samples
        test_indices = [0, min(5, total_frames - 1), min(total_frames - 1, 20)]
        for idx in test_indices:
            try:
                s = ds[idx]
                ac = s.get("actions_cartesian")
                if ac is not None and torch.isnan(ac).any():
                    errors.append(f"Sample {idx}: actions_cartesian has NaN")
            except Exception as e:
                errors.append(f"Sample {idx} failed: {e}")

        if verbose:
            shapes = {
                k: tuple(v.shape)
                for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            }
            print(f"  ZarrDataset+transforms: OK")
            for k, s in sorted(shapes.items()):
                print(f"    {k}: {s}")

    except Exception as e:
        errors.append(f"ZarrDataset+transforms failed: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------------
    # Layer 6: Annotations
    # ---------------------------------------------------------------
    if "annotations" in features:
        try:
            ann_arr = store["annotations"]
            n_ann = ann_arr.shape[0]
            for i in range(n_ann):
                raw = ann_arr[i : i + 1][0]
                parsed = json.loads(raw)
                required_ann_keys = {"text", "start_idx", "end_idx"}
                if not required_ann_keys.issubset(parsed.keys()):
                    errors.append(
                        f"Annotation[{i}] missing keys: "
                        f"{required_ann_keys - set(parsed.keys())}"
                    )

            if verbose:
                print(f"  Annotations: {n_ann} valid entries")
        except Exception as e:
            errors.append(f"Annotation read failed: {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate robot zarr episodes (new per-arm format)")
    parser.add_argument("zarr_dir", help="Directory containing .zarr episodes")
    parser.add_argument("--quiet", action="store_true", help="Only print errors")
    args = parser.parse_args()

    zarr_dir = Path(args.zarr_dir)
    episodes = sorted(zarr_dir.glob("*.zarr"))

    if not episodes:
        episodes = sorted(
            p for d in zarr_dir.iterdir() if d.is_dir() for p in d.glob("*.zarr")
        )

    if not episodes:
        print(f"No .zarr episodes found in {zarr_dir}")
        return 1

    print(f"Validating {len(episodes)} episodes in {zarr_dir}\n")

    total_errors = 0
    for ep_path in episodes:
        if not args.quiet:
            print(f"[{ep_path.name}]")
        errs = validate_episode(str(ep_path), verbose=not args.quiet)
        if errs:
            for e in errs:
                print(f"  ERROR: {e}")
            total_errors += len(errs)
        elif not args.quiet:
            print(f"  PASS\n")

    print(f"\n{'=' * 60}")
    if total_errors == 0:
        print(f"ALL {len(episodes)} EPISODES PASSED VALIDATION")
    else:
        print(f"FAILED: {total_errors} errors across {len(episodes)} episodes")
    print(f"{'=' * 60}")

    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
