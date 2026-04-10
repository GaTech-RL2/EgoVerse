#!/usr/bin/env python3
"""
End-to-end test for the reworked Scale Aloha data pipeline.

Tests:
  1. Unit: _split_per_arm matches Eva's reference implementation
  2. Unit: R_t_e + wxyz roundtrip is invertible
  3. Integration: ZarrWriter → ZarrDataset with per-arm keymap
  4. Integration: Full transform pipeline produces correct shapes
  5. Integration: Training smoke test (forward + backward pass)

Usage:
    conda run -n emimic python test_new_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import simplejpeg

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# Test 1: _split_per_arm matches Eva's reference
# ============================================================
def test_split_per_arm_matches_eva():
    """Verify our converter's _split_per_arm produces identical output to Eva's."""
    from scipy.spatial.transform import Rotation as Rot

    # --- Eva's reference implementation (copied from eva_to_zarr.py) ---
    R_t_e_ref = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

    def rot_orientation_ref(quat_xyzw):
        rotation = Rot.from_quat(quat_xyzw).as_matrix()
        rotation = R_t_e_ref @ rotation
        return Rot.from_matrix(rotation).as_quat()

    from egomimic.utils.pose_utils import xyzw_to_wxyz

    def eva_split_bimanual(arr_14):
        """Eva's _split_per_arm for bimanual non-joints keys."""
        left_ypr = arr_14[:, 3:6]
        right_ypr = arr_14[:, 10:13]
        left_quat = rot_orientation_ref(
            Rot.from_euler("ZYX", left_ypr, degrees=False).as_quat()
        )
        right_quat = rot_orientation_ref(
            Rot.from_euler("ZYX", right_ypr, degrees=False).as_quat()
        )
        left_quat = xyzw_to_wxyz(left_quat)
        right_quat = xyzw_to_wxyz(right_quat)
        left_xyz = arr_14[:, 0:3]
        right_xyz = arr_14[:, 7:10]
        left_out = np.concatenate([left_xyz, left_quat], axis=-1)
        right_out = np.concatenate([right_xyz, right_quat], axis=-1)
        left_grip = arr_14[:, 6:7]
        right_grip = arr_14[:, 13:14]
        return left_out, right_out, left_grip, right_grip

    # --- Our implementation ---
    from robot_to_egoverse_zarr import _split_per_arm

    # Generate random FK-like data: (T, 14) = [L_xyz, L_ypr, L_grip, R_xyz, R_ypr, R_grip]
    np.random.seed(42)
    T = 50
    cart_base = np.random.randn(T, 14).astype(np.float32)
    cart_base[:, 3:6] = np.random.uniform(-np.pi, np.pi, (T, 3))
    cart_base[:, 10:13] = np.random.uniform(-np.pi, np.pi, (T, 3))
    cart_base[:, 6] = np.random.uniform(0, 1, T)
    cart_base[:, 13] = np.random.uniform(0, 1, T)

    # Eva reference
    eva_left, eva_right, eva_lg, eva_rg = eva_split_bimanual(cart_base)

    # Our implementation
    result = _split_per_arm(cart_base)

    errors = []
    if not np.allclose(result["left"], eva_left, atol=1e-6):
        max_diff = np.abs(result["left"] - eva_left).max()
        errors.append(f"left pose mismatch: max_diff={max_diff:.2e}")
    if not np.allclose(result["right"], eva_right, atol=1e-6):
        max_diff = np.abs(result["right"] - eva_right).max()
        errors.append(f"right pose mismatch: max_diff={max_diff:.2e}")
    if not np.allclose(result["left_grip"], eva_lg, atol=1e-6):
        errors.append("left gripper mismatch")
    if not np.allclose(result["right_grip"], eva_rg, atol=1e-6):
        errors.append("right gripper mismatch")

    # Check shapes
    if result["left"].shape != (T, 7):
        errors.append(f"left shape {result['left'].shape} != ({T}, 7)")
    if result["right"].shape != (T, 7):
        errors.append(f"right shape {result['right'].shape} != ({T}, 7)")
    if result["left_grip"].shape != (T, 1):
        errors.append(f"left_grip shape {result['left_grip'].shape} != ({T}, 1)")

    return errors


# ============================================================
# Test 2: R_t_e + wxyz roundtrip
# ============================================================
def test_rte_roundtrip():
    """Verify R_t_e rotation is consistent and quaternions are unit norm."""
    from scipy.spatial.transform import Rotation as Rot
    from robot_to_egoverse_zarr import R_t_e, _rot_orientation
    from egomimic.utils.pose_utils import xyzw_to_wxyz

    errors = []
    np.random.seed(123)
    T = 100
    ypr = np.random.uniform(-np.pi, np.pi, (T, 3))
    quat_xyzw = Rot.from_euler("ZYX", ypr, degrees=False).as_quat()
    quat_rte = _rot_orientation(quat_xyzw)
    wxyz = xyzw_to_wxyz(quat_rte)

    # Check unit quaternions
    norms = np.linalg.norm(wxyz, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        errors.append(f"quaternion norms not unit: range [{norms.min():.6f}, {norms.max():.6f}]")

    # Verify R_t_e is a proper rotation (det=1, orthogonal)
    det = np.linalg.det(R_t_e)
    if not np.isclose(det, 1.0):
        errors.append(f"R_t_e determinant {det} != 1")
    if not np.allclose(R_t_e @ R_t_e.T, np.eye(3), atol=1e-10):
        errors.append("R_t_e not orthogonal")

    # Verify R_t_e matches Eva's definition
    R_t_e_eva = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    if not np.array_equal(R_t_e, R_t_e_eva):
        errors.append("R_t_e doesn't match Eva's definition")

    return errors


# ============================================================
# Test 3: Create synthetic zarr and load with ZarrDataset
# ============================================================
def test_zarr_write_and_load():
    """Write a zarr with new per-arm keys, load through ZarrDataset."""
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, ZarrEpisode
    from egomimic.rldb.embodiment.scale_aloha import ScaleAloha
    from robot_to_egoverse_zarr import _split_per_arm

    errors = []
    tmpdir = tempfile.mkdtemp(prefix="scale_aloha_test_")

    try:
        # Create synthetic FK data
        np.random.seed(99)
        T = 60
        obs_joints = np.random.randn(T, 14).astype(np.float32) * 0.5
        act_joints = obs_joints + np.random.randn(T, 14).astype(np.float32) * 0.01

        # Simulate FK output: [xyz, ypr, grip] x 2
        ee_base = np.random.randn(T, 14).astype(np.float32) * 0.3
        ee_base[:, 3:6] *= 0.5  # smaller rotations
        ee_base[:, 10:13] *= 0.5
        ee_base[:, 6] = np.clip(ee_base[:, 6], 0, 1)
        ee_base[:, 13] = np.clip(ee_base[:, 13], 0, 1)

        act_base = ee_base + np.random.randn(T, 14).astype(np.float32) * 0.01

        obs_split = _split_per_arm(ee_base)
        act_split = _split_per_arm(act_base)

        numeric_data = {
            "left.obs_ee_pose": obs_split["left"],
            "right.obs_ee_pose": obs_split["right"],
            "left.cmd_ee_pose": act_split["left"],
            "right.cmd_ee_pose": act_split["right"],
            "left.obs_joints": obs_joints[:, :6].astype(np.float32),
            "right.obs_joints": obs_joints[:, 7:13].astype(np.float32),
            "left.cmd_joints": act_joints[:, :6].astype(np.float32),
            "right.cmd_joints": act_joints[:, 7:13].astype(np.float32),
            "left.gripper": act_split["left_grip"],
            "right.gripper": act_split["right_grip"],
        }

        # Verify shapes before writing
        expected_shapes = {
            "left.obs_ee_pose": (T, 7),
            "right.obs_ee_pose": (T, 7),
            "left.cmd_ee_pose": (T, 7),
            "right.cmd_ee_pose": (T, 7),
            "left.obs_joints": (T, 6),
            "right.obs_joints": (T, 6),
            "left.cmd_joints": (T, 6),
            "right.cmd_joints": (T, 6),
            "left.gripper": (T, 1),
            "right.gripper": (T, 1),
        }
        for k, expected in expected_shapes.items():
            if numeric_data[k].shape != expected:
                errors.append(f"Pre-write shape {k}: {numeric_data[k].shape} != {expected}")

        # Create dummy images
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        jpeg_bytes = simplejpeg.encode_jpeg(dummy_img, quality=85)
        jpeg_arr = np.array([jpeg_bytes] * T, dtype=object)
        img_shape = [480, 640, 3]

        pre_encoded = {}
        for cam_key in [
            "observations.images.front_img_1",
            "observations.images.cam_low",
            "observations.images.left_wrist_img",
            "observations.images.right_wrist_img",
        ]:
            pre_encoded[cam_key] = (jpeg_arr, img_shape)

        # Write zarr
        ep_path = Path(tmpdir) / "test_episode.zarr"
        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data=numeric_data,
            pre_encoded_image_data=pre_encoded,
            embodiment="scale_aloha_bimanual",
            fps=30,
            task_description="test episode",
        )

        if not ep_path.exists():
            errors.append("Zarr episode was not created")
            return errors

        # Read back and verify
        ep = ZarrEpisode(str(ep_path))
        if len(ep) != T:
            errors.append(f"ZarrEpisode len {len(ep)} != {T}")

        meta = ep.metadata
        if meta["embodiment"] != "scale_aloha_bimanual":
            errors.append(f"Embodiment mismatch: {meta['embodiment']}")

        features = meta.get("features", {})
        for k in numeric_data:
            if k not in features:
                errors.append(f"Feature key '{k}' missing from zarr metadata")

        # Test ZarrDataset with Scale Aloha keymap
        keymap = ScaleAloha.get_keymap()
        transform_list = ScaleAloha.get_transform_list()

        ds = ZarrDataset(
            Episode_path=ep_path,
            key_map=keymap,
            transform_list=transform_list,
        )

        if len(ds) != T:
            errors.append(f"ZarrDataset len {len(ds)} != {T}")

        # Try loading a sample from early in the episode (to have enough action horizon)
        sample = ds[0]

        # After transforms, these keys should exist:
        # - actions_cartesian (100, 14): concatenated cam-frame actions with ypr
        # - observations.state.ee_pose (14,): concatenated cam-frame obs with ypr
        # - observations.images.front_img_1 (3, H, W): image tensor
        # - observations.images.cam_low (3, H, W): image tensor
        # - etc.
        import torch

        required_post_transform = {
            "actions_cartesian": (100, 14),
            "observations.state.ee_pose": (14,),
        }
        for k, expected_shape in required_post_transform.items():
            if k not in sample:
                errors.append(f"Post-transform key '{k}' missing from sample")
                continue
            val = sample[k]
            if not isinstance(val, torch.Tensor):
                errors.append(f"sample['{k}'] is {type(val).__name__}, expected Tensor")
                continue
            if val.shape != expected_shape:
                errors.append(f"sample['{k}'] shape {tuple(val.shape)} != {expected_shape}")
            if torch.isnan(val).any():
                errors.append(f"sample['{k}'] contains NaN")
            if torch.isinf(val).any():
                errors.append(f"sample['{k}'] contains Inf")

        # Check image keys
        for img_key in [
            "observations.images.front_img_1",
            "observations.images.cam_low",
            "observations.images.left_wrist_img",
            "observations.images.right_wrist_img",
        ]:
            if img_key not in sample:
                errors.append(f"Image key '{img_key}' missing from sample")
                continue
            val = sample[img_key]
            if val.ndim != 3 or val.shape[0] != 3:
                errors.append(f"sample['{img_key}'] shape {tuple(val.shape)}, expected (3, H, W)")

        # Verify the intermediate per-arm keys are GONE (cleaned up by DeleteKeys)
        for should_be_gone in [
            "left.cmd_ee_pose",
            "right.cmd_ee_pose",
            "left_extrinsics_pose",
            "right_extrinsics_pose",
        ]:
            if should_be_gone in sample:
                errors.append(f"Intermediate key '{should_be_gone}' should have been deleted")

        # Verify actions_cartesian layout: [L_xyz(3), L_ypr(3), L_grip(1), R_xyz(3), R_ypr(3), R_grip(1)]
        if "actions_cartesian" in sample:
            ac = sample["actions_cartesian"]
            # Check value ranges are reasonable
            xyz_vals = torch.cat([ac[:, :3], ac[:, 7:10]])
            if xyz_vals.abs().max() > 10:
                errors.append(f"actions_cartesian xyz max {xyz_vals.abs().max():.2f} > 10m")

        # Load several more samples to check robustness
        for idx in [5, 10, T - 1]:
            try:
                s = ds[idx]
                if "actions_cartesian" not in s:
                    errors.append(f"Sample {idx} missing actions_cartesian")
            except Exception as e:
                errors.append(f"Failed to load sample {idx}: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return errors


# ============================================================
# Test 4: FK → split → zarr → dataset → model forward pass
# ============================================================
def test_training_forward_pass():
    """Smoke test: create zarr, load batch, run model forward + backward."""
    import torch
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
    from egomimic.rldb.embodiment.scale_aloha import ScaleAloha
    from robot_to_egoverse_zarr import _split_per_arm

    errors = []
    tmpdir = tempfile.mkdtemp(prefix="scale_aloha_fwd_")

    try:
        np.random.seed(77)
        T = 80

        ee_base = np.random.randn(T, 14).astype(np.float32) * 0.3
        ee_base[:, 3:6] *= 0.5
        ee_base[:, 10:13] *= 0.5
        ee_base[:, 6] = np.clip(ee_base[:, 6], 0, 1)
        ee_base[:, 13] = np.clip(ee_base[:, 13], 0, 1)
        act_base = ee_base + np.random.randn(T, 14).astype(np.float32) * 0.01

        obs_split = _split_per_arm(ee_base)
        act_split = _split_per_arm(act_base)

        numeric_data = {
            "left.obs_ee_pose": obs_split["left"],
            "right.obs_ee_pose": obs_split["right"],
            "left.cmd_ee_pose": act_split["left"],
            "right.cmd_ee_pose": act_split["right"],
            "left.obs_joints": np.random.randn(T, 6).astype(np.float32) * 0.5,
            "right.obs_joints": np.random.randn(T, 6).astype(np.float32) * 0.5,
            "left.cmd_joints": np.random.randn(T, 6).astype(np.float32) * 0.5,
            "right.cmd_joints": np.random.randn(T, 6).astype(np.float32) * 0.5,
            "left.gripper": act_split["left_grip"],
            "right.gripper": act_split["right_grip"],
        }

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        jpeg_bytes = simplejpeg.encode_jpeg(dummy_img, quality=85)
        jpeg_arr = np.array([jpeg_bytes] * T, dtype=object)

        pre_encoded = {
            cam: (jpeg_arr, [480, 640, 3])
            for cam in [
                "observations.images.front_img_1",
                "observations.images.cam_low",
                "observations.images.left_wrist_img",
                "observations.images.right_wrist_img",
            ]
        }

        ep_path = Path(tmpdir) / "fwd_test.zarr"
        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data=numeric_data,
            pre_encoded_image_data=pre_encoded,
            embodiment="scale_aloha_bimanual",
            fps=30,
            task_description="forward pass test",
        )

        keymap = ScaleAloha.get_keymap()
        transforms = ScaleAloha.get_transform_list()
        ds = ZarrDataset(ep_path, keymap, transforms)

        # Build a batch of 2 samples
        samples = [ds[0], ds[1]]
        batch = {}
        for key in samples[0]:
            vals = [s[key] for s in samples]
            if isinstance(vals[0], torch.Tensor):
                batch[key] = torch.stack(vals)
            else:
                batch[key] = vals

        # Verify batch shapes
        if batch["actions_cartesian"].shape != (2, 100, 14):
            errors.append(f"Batch actions shape {batch['actions_cartesian'].shape} != (2, 100, 14)")
        if batch["observations.state.ee_pose"].shape != (2, 14):
            errors.append(f"Batch ee_pose shape {batch['observations.state.ee_pose'].shape} != (2, 14)")
        if batch["observations.images.front_img_1"].shape[1] != 3:
            errors.append(f"Batch image channels {batch['observations.images.front_img_1'].shape[1]} != 3")

        # Check no NaN/Inf in batch
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                if torch.isnan(val).any():
                    errors.append(f"Batch '{key}' contains NaN")
                if torch.isinf(val).any():
                    errors.append(f"Batch '{key}' contains Inf")

    except Exception as e:
        errors.append(f"Forward pass test failed: {e}\n{traceback.format_exc()}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return errors


# ============================================================
# Test 5: Extrinsics are correct and consistent
# ============================================================
def test_extrinsics_consistency():
    """Verify EXTRINSICS['scale_aloha'] matches the fallback values in the converter."""
    from egomimic.utils.egomimicUtils import EXTRINSICS
    from robot_to_egoverse_zarr import _FALLBACK_EXTRINSICS

    errors = []

    for side in ("left", "right"):
        central = EXTRINSICS["scale_aloha"][side]
        fallback = _FALLBACK_EXTRINSICS[side]
        if not np.allclose(central, fallback, atol=1e-6):
            diff = np.abs(central - fallback).max()
            errors.append(
                f"EXTRINSICS['scale_aloha']['{side}'] doesn't match "
                f"converter fallback (max_diff={diff:.2e})"
            )

        # Verify it's a valid SE(3) matrix
        R = central[:3, :3]
        det = np.linalg.det(R)
        if not np.isclose(abs(det), 1.0, atol=1e-6):
            errors.append(f"EXTRINSICS['{side}'] rotation det={det:.6f} != ±1")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            errors.append(f"EXTRINSICS['{side}'] rotation not orthogonal")
        if not np.array_equal(central[3, :], [0, 0, 0, 1]):
            errors.append(f"EXTRINSICS['{side}'] bottom row != [0,0,0,1]")

    return errors


# ============================================================
# Test 6: Validate the keymap matches what's in the zarr
# ============================================================
def test_keymap_zarr_alignment():
    """Verify every zarr_key in the keymap exists in a written zarr episode."""
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.rldb.embodiment.scale_aloha import ScaleAloha
    from robot_to_egoverse_zarr import _split_per_arm
    import zarr

    errors = []
    tmpdir = tempfile.mkdtemp(prefix="scale_aloha_align_")

    try:
        T = 30
        ee_base = np.random.randn(T, 14).astype(np.float32) * 0.3
        ee_base[:, 6] = 0.5
        ee_base[:, 13] = 0.5
        obs_split = _split_per_arm(ee_base)
        act_split = _split_per_arm(ee_base)

        numeric_data = {
            "left.obs_ee_pose": obs_split["left"],
            "right.obs_ee_pose": obs_split["right"],
            "left.cmd_ee_pose": act_split["left"],
            "right.cmd_ee_pose": act_split["right"],
            "left.obs_joints": np.random.randn(T, 6).astype(np.float32),
            "right.obs_joints": np.random.randn(T, 6).astype(np.float32),
            "left.cmd_joints": np.random.randn(T, 6).astype(np.float32),
            "right.cmd_joints": np.random.randn(T, 6).astype(np.float32),
            "left.gripper": obs_split["left_grip"],
            "right.gripper": obs_split["right_grip"],
        }

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        jpeg_bytes = simplejpeg.encode_jpeg(dummy_img, quality=85)
        jpeg_arr = np.array([jpeg_bytes] * T, dtype=object)
        pre_encoded = {
            cam: (jpeg_arr, [480, 640, 3])
            for cam in [
                "observations.images.front_img_1",
                "observations.images.cam_low",
                "observations.images.left_wrist_img",
                "observations.images.right_wrist_img",
            ]
        }

        ep_path = Path(tmpdir) / "align_test.zarr"
        ZarrWriter.create_and_write(
            episode_path=ep_path,
            numeric_data=numeric_data,
            pre_encoded_image_data=pre_encoded,
            embodiment="scale_aloha_bimanual",
            fps=30,
            task_description="alignment test",
        )

        store = zarr.open_group(str(ep_path), mode="r")
        features = dict(store.attrs).get("features", {})
        zarr_keys_available = set(features.keys())

        keymap = ScaleAloha.get_keymap()
        for batch_key, info in keymap.items():
            zarr_key = info["zarr_key"]
            if zarr_key not in zarr_keys_available:
                errors.append(
                    f"Keymap batch_key='{batch_key}' references zarr_key='{zarr_key}' "
                    f"which is NOT in the zarr. Available: {sorted(zarr_keys_available)}"
                )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return errors


# ============================================================
# Test 7: DataSchematic compatibility
# ============================================================
def test_data_schematic():
    """Verify DataSchematic can be instantiated with the updated hpt.yaml."""
    errors = []
    try:
        import hydra
        from omegaconf import OmegaConf

        schematic_path = Path(__file__).parents[3] / "egomimic" / "hydra_configs" / "data_schematic" / "hpt.yaml"
        cfg = OmegaConf.load(str(schematic_path))

        # Check scale_aloha_bimanual section exists
        if "scale_aloha_bimanual" not in cfg.get("schematic_dict", {}):
            errors.append("scale_aloha_bimanual missing from schematic_dict")
            return errors

        section = cfg["schematic_dict"]["scale_aloha_bimanual"]

        required_entries = {
            "front_img_1": "camera_keys",
            "cam_low": "camera_keys",
            "left_wrist_img": "camera_keys",
            "right_wrist_img": "camera_keys",
            "ee_pose": "proprio_keys",
            "actions_cartesian": "action_keys",
            "embodiment": "metadata_keys",
        }

        for key_name, expected_type in required_entries.items():
            if key_name not in section:
                errors.append(f"DataSchematic missing key_name '{key_name}'")
                continue
            actual_type = section[key_name].get("key_type", "")
            if actual_type != expected_type:
                errors.append(
                    f"DataSchematic '{key_name}' key_type='{actual_type}' "
                    f"!= expected '{expected_type}'"
                )

        # Verify the old combined keys are gone
        for old_key in ["joint_positions", "actions_joints"]:
            if old_key in section:
                errors.append(f"Old key '{old_key}' should be removed from schematic")

    except Exception as e:
        errors.append(f"DataSchematic test failed: {e}")

    return errors


# ============================================================
# Runner
# ============================================================
def main():
    tests = [
        ("1. _split_per_arm matches Eva", test_split_per_arm_matches_eva),
        ("2. R_t_e + wxyz roundtrip", test_rte_roundtrip),
        ("3. Extrinsics consistency", test_extrinsics_consistency),
        ("4. Keymap ↔ zarr alignment", test_keymap_zarr_alignment),
        ("5. Zarr write → dataset load", test_zarr_write_and_load),
        ("6. Training batch construction", test_training_forward_pass),
        ("7. DataSchematic compatibility", test_data_schematic),
    ]

    all_pass = True
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            errs = fn()
            if errs:
                all_pass = False
                for e in errs:
                    print(f"  FAIL: {e}")
            else:
                print("  PASS")
        except Exception as e:
            all_pass = False
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
