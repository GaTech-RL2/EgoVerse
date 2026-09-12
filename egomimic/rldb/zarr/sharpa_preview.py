"""Create an explicit analysis copy of the 65-column Sharpa structural sample.

This optional preview producer is separate from ingestion and training. It
reconstructs palm poses/keypoints from the delivered URDFs and estimates an
optical-camera mount from manually identified image landmarks. Original arrays
and metadata remain in the source; estimates never qualify as vendor calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import zarr
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.embodiment.hand_kinematics import fk_keypoints, load_chain
from egomimic.rldb.embodiment.urdf import load_urdf
from egomimic.rldb.zarr.overlay import decode_frame, episode_length
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz


def reconstruct_tracks(group):
    """Return canonical arrays and observed head-link poses in robot_base."""
    resolved = Embodiment.resolve("dexmate_bimanual")
    platform = load_urdf(resolved.platform.asset_path(verify=True))
    total = episode_length(group)
    output = {}
    observed_head = []
    for prefix, source in (
        ("obs", "sharpa.observation_state"),
        ("cmd", "sharpa.action"),
    ):
        raw = np.asarray(group[source][:total], dtype=float)
        if raw.shape != (total, 65) or not np.isfinite(raw).all():
            raise ValueError(f"{source} must be a finite (T, 65) array")
        output[f"{prefix}_aux_joints"] = raw[:, 58:65]
        values = {
            name: raw[:, 58 + i]
            for i, name in enumerate(resolved.platform.aux.joint_names)
        }
        for side, offset in (("left", 0), ("right", 29)):
            for j in range(7):
                values[f"{side}_arm_joint_{j + 1}"] = raw[:, offset + j]
            output[f"{side}.{prefix}_joints"] = raw[:, offset : offset + 7]
            output[f"{side}.{prefix}_hand_joints"] = raw[:, offset + 7 : offset + 29]
        platform_poses = [
            platform.link_transforms({k: v[t] for k, v in values.items()})
            for t in range(total)
        ]
        if prefix == "obs":
            observed_head = np.stack(
                [poses["head_base_link"] for poses in platform_poses]
            )
        for side in ("left", "right"):
            spec = resolved.end_effectors[side]
            chain = load_chain(spec)
            joints = output[f"{side}.{prefix}_hand_joints"]
            # The platform mount leaf and hand flange are composed directly.
            # Record that mechanical-frame assumption in the copy provenance.
            palm = np.stack(
                [
                    poses[f"{side}_hand_base_link"]
                    @ chain.link_transforms(dict(zip(spec.joint_names, row)))[
                        spec.ee_pose_link
                    ]
                    for poses, row in zip(platform_poses, joints, strict=True)
                ]
            )
            output[f"{side}.{prefix}_ee_pose"] = _matrix_to_xyzwxyz(palm)
            if prefix == "obs":
                local = fk_keypoints(spec, joints)
                points = (
                    np.einsum("tij,tkj->tki", palm[:, :3, :3], local)
                    + palm[:, None, :3, 3]
                )
                output[f"{side}.obs_hand_keypoints"] = points.reshape(total, 63)
    return output, np.asarray(observed_head)


def fit_camera(tracks, head_poses, landmarks, resolution):
    """Fit a constant head-link optical mount and focal lengths to 2D picks.

    Principal point is fixed at image centre; distortion is assumed zero.
    Bounds and weak mount priors constrain this poorly observed calibration.
    The residual reports agreement with approximate picks, not metric accuracy.
    """
    entries = landmarks["landmarks"]
    if len(entries) < 8:
        raise ValueError("at least eight image landmarks are needed")
    width, height = resolution
    xyz, uv = [], []
    for item in entries:
        frame, side, slot = item["frame"], item["side"], item["slot"]
        if (
            side not in ("left", "right")
            or not 0 <= slot < 21
            or not 0 <= frame < len(head_poses)
        ):
            raise ValueError(f"invalid landmark {item}")
        point = tracks[f"{side}.obs_hand_keypoints"][frame].reshape(21, 3)[slot]
        head_T_ref = np.linalg.inv(head_poses[frame])
        xyz.append((head_T_ref @ np.r_[point, 1])[:3])
        uv.append(item["uv"])
    xyz, uv = np.asarray(xyz), np.asarray(uv)
    if uv.shape != (len(xyz), 2) or not np.isfinite(uv).all():
        raise ValueError("landmark pixels must be finite pairs")
    # Optical x right, y down, z forward; head link nominally x forward, z up.
    optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    initial_angles = landmarks.get("initial_rpy", [0, 0.7, 0])
    initial = np.r_[
        initial_angles,
        landmarks.get("initial_translation", [0.05, 0, 0.03]),
        [350, 350],
    ]

    def model(parameters):
        transform = np.eye(4)
        transform[:3, :3] = (
            Rotation.from_euler("xyz", parameters[:3]).as_matrix() @ optical
        )
        transform[:3, 3] = parameters[3:6]
        camera_points = (xyz - transform[:3, 3]) @ transform[:3, :3]
        pixels = camera_points[:, :2] / np.maximum(
            camera_points[:, 2:], 0.01
        ) * parameters[6:8] + [width / 2, height / 2]
        return transform, camera_points, pixels

    def residual(parameters):
        _, points, pixels = model(parameters)
        return np.r_[
            (pixels - uv).ravel(),
            np.minimum(points[:, 2] - 0.02, 0) * 1000,
            (parameters[3:6] - initial[3:6]) * 40,
            parameters[:3] - initial[:3],
        ]

    fit = least_squares(
        residual,
        initial,
        bounds=(
            [-0.8, -0.3, -0.8, -0.20, -0.20, -0.20, 100, 100],
            [0.8, 1.5, 0.8, 0.30, 0.20, 0.30, 900, 900],
        ),
        loss="soft_l1",
        f_scale=8,
        max_nfev=3000,
    )
    mount, points, pixels = model(fit.x)
    if not fit.success or (points[:, 2] <= 0).any():
        raise ValueError(f"camera fitting failed: {fit.message}")
    errors = np.linalg.norm(pixels - uv, axis=1)
    K = np.array(
        [[fit.x[6], 0, width / 2, 0], [0, fit.x[7], height / 2, 0], [0, 0, 1, 0]]
    )
    return {
        "head_link_T_cam": mount.tolist(),
        "K": K.tolist(),
        "resolution": list(resolution),
        "rms_landmark_error_px": float(np.sqrt(np.mean(errors**2))),
        "max_landmark_error_px": float(errors.max()),
        "landmark_errors_px": errors.tolist(),
        "fitted_rpy": fit.x[:3].tolist(),
        "fitted_translation": fit.x[3:6].tolist(),
        "parameters_at_bounds": [
            name
            for name, active in zip(
                ("roll", "pitch", "yaw", "tx", "ty", "tz", "fx", "fy"),
                fit.active_mask,
                strict=True,
            )
            if active
        ],
        "assumptions": [
            "manually identified approximate MANO link origins",
            "centred principal point",
            "zero distortion / pinhole delivered images",
            "constant head_base_link optical mount",
            "platform hand_base_link equals hand flange mount",
        ],
    }


def prepare_preview(source, destination, landmarks_path):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("analysis output must be a new directory outside the source")
    group = zarr.open_group(source, mode="r")
    tracks, head_poses = reconstruct_tracks(group)
    first = decode_frame(group, 0)
    landmark_bytes = Path(landmarks_path).read_bytes()
    landmarks = json.loads(landmark_bytes)
    estimate = fit_camera(
        tracks, head_poses, landmarks, (first.shape[1], first.shape[0])
    )
    tracks["obs_head_pose"] = _matrix_to_xyzwxyz(
        head_poses @ np.asarray(estimate["head_link_T_cam"])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    copy = zarr.open_group(destination, mode="a")
    original_attrs = dict(copy.attrs)
    features = dict(original_attrs.get("features", {}))
    for key, values in tracks.items():
        if key in copy:
            del copy[key]
        copy.create_array(
            key, data=values, chunks=(min(100, len(values)), values.shape[1])
        )
        features[key] = {
            "dtype": str(values.dtype),
            "shape": [values.shape[1]],
            "names": ["derived_preview"],
        }
    resolved = Embodiment.resolve("dexmate_bimanual")
    copy.attrs.update(
        {
            "embodiment": "dexmate_bimanual",
            "schema_version": "v3.1",
            "data_status": "structural_sample",
            "morphology": {
                "platform": resolved.platform.name,
                "end_effector": {
                    s: ee.name for s, ee in resolved.end_effectors.items()
                },
            },
            "action_semantics": "absolute_joint_position_setpoint",
            "features": features,
            "calibration": {
                "reference_frame": "robot_base",
                "cameras": {
                    "front_1": {
                        "K": estimate["K"],
                        "model": "PINHOLE",
                        "rectified": True,
                        "resolution": estimate["resolution"],
                    }
                },
            },
            "intrinsics": {"front_1": estimate["K"]},
            "extrinsics": {},
            "pose_status": "FK-derived palm poses; observed optical trajectory uses estimated mount",
            "intrinsics_status": "estimated for analysis; not vendor calibration",
            "extrinsics_status": "estimated head mount; wrist calibration unavailable",
            "do_not_use_for_training": True,
            "preview_provenance": {
                "source": str(source),
                "source_metadata_sha256": hashlib.sha256(
                    (source / "zarr.json").read_bytes()
                ).hexdigest(),
                "source_attrs": original_attrs,
                "estimated_cameras": ["front_1"],
                "camera_estimate": estimate,
                "landmarks": landmarks,
                "landmarks_sha256": hashlib.sha256(landmark_bytes).hexdigest(),
                "derived_arrays": list(tracks),
                "keypoint_method": "delivered URDF FK, MANO21 link origins in robot_base",
                "model_hashes": {
                    "platform": resolved.platform.urdf_sha256,
                    **{s: ee.urdf_sha256 for s, ee in resolved.end_effectors.items()},
                },
                "limitations": [
                    "not independently measured keypoints",
                    "camera fit uses approximate manual image picks",
                    "original timestamp and annotation defects retained",
                    "wrist camera calibration unavailable",
                ],
            },
        }
    )
    return estimate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    args = parser.parse_args(argv)
    estimate = prepare_preview(args.source, args.out, args.landmarks)
    print(json.dumps(estimate, indent=2))


if __name__ == "__main__":
    main()
