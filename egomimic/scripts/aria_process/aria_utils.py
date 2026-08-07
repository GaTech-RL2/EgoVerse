import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import projectaria_tools.core.sophus as sp
import torch
import torch.nn.functional as F
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps.utils import (
    get_nearest_eye_gaze,
    get_nearest_hand_tracking_result,
    get_nearest_pose,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

from egomimic.utils.pose_utils import T_rot_orientation

ROTATION_MATRIX = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
T_ROT_CAM = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

# ---------------------------------------------------------------------------
# Aria 21-keypoint -> MANO 21-keypoint conversion (per-frame batched fit).
# Single source of truth for the correspondence tables; the tutorial script
# egomimic/scripts/tutorials/aria_to_mano_convert.py imports from here.
#
# otaheri MANO joint order with return_tips=True (21 joints):
#   0 wrist, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb,
#   16 thumb_tip, 17 index_tip, 18 middle_tip, 19 ring_tip, 20 pinky_tip
# Canonical MANO order used by Human.FINGER_EDGES:
#   0 wrist, 1-4 thumb (CMC, MCP, IP, tip), 5-8 index, 9-12 middle,
#   13-16 ring, 17-20 pinky
OTAHERI_TO_CANONICAL = [
    0,              # wrist
    13, 14, 15, 16, # thumb (3 joints + tip)
    1, 2, 3, 17,    # index
    4, 5, 6, 18,    # middle
    10, 11, 12, 19, # ring
    7, 8, 9, 20,    # pinky
]

# Aria's 21-keypoint layout: 0-4 fingertips (thumb..pinky), 5 palm root/wrist,
# 6-7 thumb intermediates, 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky,
# 20 palm center (dropped from the fit loss).
# Aria idx -> otaheri MANO target idx; MANO's thumb CMC has no Aria source and
# is filled in by MANO's kinematic prior.
ARIA_TO_OTAHERI = {
    5: 0,    # wrist
    6: 14,   # thumb MCP
    7: 15,   # thumb IP
    0: 16,   # thumb tip
    8: 1,    # index1
    9: 2,    # index2
    10: 3,   # index3
    1: 17,   # index tip
    11: 4,   # middle1
    12: 5,   # middle2
    13: 6,   # middle3
    2: 18,   # middle tip
    14: 10,  # ring1
    15: 11,  # ring2
    16: 12,  # ring3
    3: 19,   # ring tip
    17: 7,   # pinky1
    18: 8,   # pinky2
    19: 9,   # pinky3
    4: 20,   # pinky tip
}
ARIA_IDX_USED = list(ARIA_TO_OTAHERI.keys())  # length 20
MANO_IDX_TARGET = [ARIA_TO_OTAHERI[a] for a in ARIA_IDX_USED]  # length 20

# The extractor marks missing hand data with 1e9; clean_data() drops rows
# >= 1e8, but keep the guard here so the converter is safe on raw inputs too.
_ARIA_INVALID_SENTINEL = 1e8

ARIA_WRIST_IDX = 5  # palm-root/wrist index in Aria's 21-kp layout

# Canonical MANO-21 landmarks used to build the hand cartesian frame:
# 0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky,
# so the MCP (knuckle) of each finger is the first joint of its run.
MANO_WRIST_IDX = 0
MANO_INDEX_MCP_IDX = 5
MANO_PINKY_MCP_IDX = 17
MANO_MCP_IDX = [5, 9, 13, 17]  # index, middle, ring, pinky knuckles
# Palm origin = centroid of the wrist + the index/middle/ring knuckles.
# The PINKY MCP is deliberately excluded: Aria's palm landmark sits toward the
# radial (thumb) side, and including the pinky drags the centroid ulnar.
# Measured against Aria's own palm landmark (raw kp 20), both hands:
#     MCP centroid, no wrist   19.8mm   (lands ON the knuckle row, ~17mm distal
#                                        of the palm -- the MCPs are ~coplanar)
#     wrist + 4 MCPs           13.8mm
#     wrist + idx/mid/ring     11.3mm   <- this one
# It also improves orientation vs the aria-derived frame (4.65/5.30 -> 3.84/4.50
# deg), since the origin sets the x-axis via (wrist - palm).
# NB an *equidistant* (circumcentre) origin is far worse: the knuckles lie on a
# shallow arc of ~139mm radius, so their circumcentre flies ~131mm off the hand.
MANO_PALM_IDX = [0, 5, 9, 13]


def _default_mano_model_dir() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    d = repo_root / "external_ckpts" / "mano"
    if d.exists():
        return str(d)
    # Fallback: pkls bundled inside the mano package itself. Needed on ray
    # workers — runtime_env working_dir packaging honors .gitignore, so the
    # gitignored external_ckpts/ never ships, while py_modules (the mano
    # package, with pkls copied into _bundled_models/) does.
    import mano as _mano

    return str(Path(_mano.__file__).resolve().parent / "_bundled_models")


def _default_mano_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_mano_to_aria_batched(
    aria_kp: torch.Tensor,
    is_rhand: bool,
    model_dir: str | None = None,
    device: torch.device | str | None = None,
    n_iters: int = 2400,
    lr: float = 0.05,
    beta_reg: float = 0.01,
    verbose: bool = False,
) -> torch.Tensor:
    """Batched MANO fit against Aria keypoints, in whatever frame they're in.

    aria_kp: (N, 21, 3) Aria-layout keypoints (NaN = invalid point).
    Returns (N, 21, 3) canonical-MANO-ordered keypoints on CPU, in the same
    frame as the input; frames with an invalid wrist are all-NaN.
    """
    import mano  # lazy: optional dep + license-gated model files

    model_dir = model_dir or _default_mano_model_dir()
    device = torch.device(device) if device is not None else _default_mano_device()

    N = aria_kp.shape[0]
    aria_kp = aria_kp.to(device)

    valid_mask = torch.isfinite(aria_kp).all(dim=-1)  # (N, 21)

    init_translation = torch.where(
        valid_mask[:, ARIA_WRIST_IDX : ARIA_WRIST_IDX + 1, None],
        aria_kp[:, ARIA_WRIST_IDX : ARIA_WRIST_IDX + 1, :],
        torch.zeros_like(aria_kp[:, ARIA_WRIST_IDX : ARIA_WRIST_IDX + 1, :]),
    ).squeeze(1)  # (N, 3)

    model = mano.load(
        model_path=model_dir,
        is_rhand=is_rhand,
        num_pca_comps=45,
        batch_size=N,
        flat_hand_mean=False,
    ).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    theta = torch.zeros(N, 45, device=device, requires_grad=True)
    beta = torch.zeros(N, 10, device=device, requires_grad=True)
    R_global = torch.zeros(N, 3, device=device, requires_grad=True)
    t_global = init_translation.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([theta, beta, R_global, t_global], lr=lr)
    # Cosine-decay the LR to 5% of peak over the budget. A constant high LR
    # oscillates at long iteration counts and degrades the fit (~5.2mm vs 4.7mm
    # corresponded-finger err); the decay is load-bearing, not optional.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=lr * 0.05)

    aria_idx_t = torch.tensor(ARIA_IDX_USED, device=device, dtype=torch.long)
    mano_idx_t = torch.tensor(MANO_IDX_TARGET, device=device, dtype=torch.long)

    targets = aria_kp.index_select(1, aria_idx_t)  # (N, 20, 3)
    point_valid = torch.isfinite(targets).all(dim=-1)  # (N, 20)
    targets = torch.nan_to_num(targets, nan=0.0)

    for step in range(n_iters):
        out = model(
            betas=beta, global_orient=R_global, hand_pose=theta,
            transl=t_global, return_verts=True, return_tips=True,
        )
        pred = out.joints.index_select(1, mano_idx_t)  # (N, 20, 3)

        per_point_err = ((pred - targets) ** 2).sum(dim=-1)  # (N, 20)
        masked = per_point_err * point_valid.float()
        denom = point_valid.sum().clamp(min=1).float()
        pos_loss = masked.sum() / denom
        reg_loss = beta_reg * (beta**2).sum() / N
        loss = pos_loss + reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 50 == 0 or step == n_iters - 1):
            print(f"    iter {step:4d}  pos={pos_loss.item():.5f}  reg={reg_loss.item():.5f}")

    with torch.no_grad():
        out = model(
            betas=beta, global_orient=R_global, hand_pose=theta,
            transl=t_global, return_verts=True, return_tips=True,
        )
        mano_kp = out.joints.detach().cpu()  # (N, 21, 3) otaheri order

    perm = torch.tensor(OTAHERI_TO_CANONICAL, dtype=torch.long)
    canonical = mano_kp.index_select(1, perm)  # (N, 21, 3) canonical order

    frame_valid = valid_mask[:, ARIA_WRIST_IDX].cpu()
    canonical = torch.where(
        frame_valid[:, None, None], canonical, torch.full_like(canonical, float("nan"))
    )
    return canonical


def convert_aria_keypoints_to_mano(
    keypoints_t63: np.ndarray,
    is_rhand: bool,
    model_dir: str | None = None,
    device: torch.device | str | None = None,
    n_iters: int = 2400,
    lr: float = 0.05,
    beta_reg: float = 0.01,
    chunk_size: int = 512,
    verbose: bool = False,
) -> np.ndarray:
    """Convert one hand's (T, 63) Aria keypoints to (T, 63) canonical MANO.

    Output is in the same coordinate frame as the input. Fits are batched in
    chunks of `chunk_size` frames to bound memory.
    """
    kp = torch.as_tensor(np.asarray(keypoints_t63), dtype=torch.float32).reshape(-1, 21, 3)
    if kp.shape[0] == 0:
        return np.asarray(keypoints_t63, dtype=np.float64).reshape(-1, 63)
    # Sentinel (1e9) -> NaN so the fit's validity masks apply.
    kp = torch.where(kp.abs() < _ARIA_INVALID_SENTINEL, kp, torch.full_like(kp, float("nan")))
    outs = []
    for s in range(0, kp.shape[0], chunk_size):
        outs.append(
            fit_mano_to_aria_batched(
                kp[s : s + chunk_size],
                is_rhand=is_rhand,
                model_dir=model_dir,
                device=device,
                n_iters=n_iters,
                lr=lr,
                beta_reg=beta_reg,
                verbose=verbose,
            )
        )
    return torch.cat(outs, dim=0).reshape(-1, 63).numpy().astype(np.float64)


def undistort_to_linear(
    provider, stream_ids, raw_image, camera_label="rgb", height=480, width=640
):
    camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    focal_length = 133.25430222 * (height / 240)
    warped = calibration.get_linear_camera_calibration(
        height, width, focal_length, camera_label, calib.get_transform_device_camera()
    )
    warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
    warped_rot = np.rot90(warped_image, k=3)
    return warped_rot


def slam_to_rgb(provider, height=480, width=640):
    """
    Get slam camera to rgb camera transform
    provider: vrs data provider
    """
    focal_length = 133.25430222 * (height / 240)
    device_calibration = provider.get_device_calibration()

    slam_id = StreamId("1201-1")
    slam_label = provider.get_label_from_stream_id(slam_id)
    slam_calib = device_calibration.get_camera_calib(slam_label)
    slam_camera = calibration.get_linear_camera_calibration(
        height,
        width,
        focal_length,
        slam_label,
        slam_calib.get_transform_device_camera(),
    )
    T_device_slam_camera = (
        slam_camera.get_transform_device_camera()
    )  # slam to device frame

    rgb_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_id)
    rgb_calib = device_calibration.get_camera_calib(rgb_label)
    rgb_camera = calibration.get_linear_camera_calibration(
        height, width, focal_length, rgb_label, rgb_calib.get_transform_device_camera()
    )
    T_device_rgb_camera = (
        rgb_camera.get_transform_device_camera()
    )  # rgb to device frame

    transform = T_device_rgb_camera.inverse() @ T_device_slam_camera

    return transform


def compute_coordinate_frame(palm_pose, wrist_pose, palm_normal):
    x_axis = wrist_pose - palm_pose
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)
    z_axis = np.ravel(palm_normal) / np.linalg.norm(palm_normal)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = np.ravel(y_axis) / np.linalg.norm(y_axis)

    x_axis = np.cross(z_axis, y_axis)
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)

    return -1 * x_axis, y_axis, z_axis


def compute_orientation_rotation_matrix(palm_pose, wrist_pose, palm_normal):
    x_axis = wrist_pose - palm_pose
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)
    z_axis = np.ravel(palm_normal) / np.linalg.norm(palm_normal)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = np.ravel(y_axis) / np.linalg.norm(y_axis)

    x_axis = np.cross(z_axis, y_axis)
    x_axis = np.ravel(x_axis) / np.linalg.norm(x_axis)

    rot_matrix = np.column_stack([-1 * x_axis, y_axis, z_axis])
    return rot_matrix


def mano_keypoints_to_cartesian(
    mano_kp_t63: np.ndarray,
    is_rhand: bool,
) -> np.ndarray:
    """Derive the hand cartesian EE pose from fitted canonical-MANO keypoints.

    The MANO fit is a much better-conditioned estimate of hand orientation than
    Aria's raw MPS palm/wrist/normal triple: measured over 10 episodes, the MPS
    pose intermittently produced near-flipped frames (orientation error p99 138
    deg, max 178 deg) while the MANO-derived pose stays at p99 9.2 deg, max 21
    deg. The EE pose is therefore built from the fitted keypoints.

    Frame construction (mirrors `get_ee_pose`'s convention so the output is
    drop-in compatible with the MPS-derived pose it replaces):
      wrist       = canonical MANO joint 0
      palm        = centroid of the wrist + index/middle/ring MCPs (pinky
                    excluded); MANO has no palm-center joint, as Aria's
                    palm-center landmark is dropped by the fit. The MCPs alone
                    are ~coplanar along the knuckle row so their centroid sits on
                    that row, ~17mm distal of the palm centre, and the pinky
                    drags it ulnar. See MANO_PALM_IDX for the measured ranking.
      palm_normal = (index_MCP - wrist) x (pinky_MCP - wrist), sign-flipped for
                    the left hand so it points out of the palm for BOTH hands
                    (the cross product mirrors between left and right)
    `compute_orientation_rotation_matrix` then builds R exactly as the MPS path
    does, T_ROT_CAM is applied, and the pose is returned as [xyz, quat wxyz].

    Parameters
    ----------
    mano_kp_t63 : np.ndarray
        (T, 63) canonical-MANO keypoints in WORLD frame — i.e. exactly what is
        stored in the `<side>.obs_keypoints` zarr key.
    is_rhand : bool
        Right hand if True; selects the palm-normal sign.

    Returns
    -------
    np.ndarray
        (T, 7) EE pose per frame as [x, y, z, qw, qx, qy, qz]. Frames with
        invalid/NaN keypoints keep the 1e9 sentinel used by the MPS extractors.
    """
    kp = np.asarray(mano_kp_t63, dtype=np.float64).reshape(-1, 21, 3)
    n_frames = kp.shape[0]
    out = np.full((n_frames, 7), 1e9, dtype=np.float64)
    needed = MANO_PALM_IDX
    normal_sign = 1.0 if is_rhand else -1.0

    for t in range(n_frames):
        pts = kp[t]
        if not np.isfinite(pts[needed]).all():
            continue
        wrist = pts[MANO_WRIST_IDX]
        palm = pts[MANO_PALM_IDX].mean(axis=0)
        normal = normal_sign * np.cross(
            pts[MANO_INDEX_MCP_IDX] - wrist, pts[MANO_PINKY_MCP_IDX] - wrist
        )
        # Degenerate (collapsed) hand -> leave the sentinel rather than emit a
        # garbage frame; compute_orientation_rotation_matrix would divide by ~0.
        if np.linalg.norm(normal) < 1e-9 or np.linalg.norm(wrist - palm) < 1e-9:
            continue
        rot_matrix = compute_orientation_rotation_matrix(
            palm_pose=palm, wrist_pose=wrist, palm_normal=normal
        )
        if not np.isfinite(rot_matrix).all():
            continue
        pose_T = np.eye(4)
        pose_T[:3, :3] = rot_matrix
        pose_T[:3, 3] = palm
        pose_T = T_rot_orientation(pose_T, T_ROT_CAM)
        quat_and_translation = quat_translation_swap(
            sp.SE3.from_matrix(pose_T).to_quat_and_translation()
        )
        if quat_and_translation.ndim == 2:
            quat_and_translation = quat_and_translation[0]
        out[t] = quat_and_translation
    return out


def downsample_hwc_uint8_in_chunks(
    images: np.ndarray,  # (T,H,W,3) uint8
    out_hw=(240, 320),
    chunk: int = 256,
) -> np.ndarray:
    assert images.dtype == np.uint8 and images.ndim == 4 and images.shape[-1] == 3
    T, H, W, C = images.shape
    outH, outW = out_hw

    out = np.empty((T, outH, outW, 3), dtype=np.uint8)

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        x = (
            torch.from_numpy(images[s:e]).permute(0, 3, 1, 2).to(torch.float32) / 255.0
        )  # (B,3,H,W)
        x = F.interpolate(x, size=(outH, outW), mode="bilinear", align_corners=False)
        x = (x * 255.0).clamp(0, 255).to(torch.uint8)  # (B,3,outH,outW)
        out[s:e] = x.permute(0, 2, 3, 1).cpu().numpy()
        del x

    return out


def quat_translation_swap(quat_translation: np.ndarray) -> np.ndarray:
    """
    Swap the quaternion and translation in a (N, 7) array.
    Parameters
    ----------
    quat_translation : np.ndarray
        (N, 7) array of quaternion and translation
    Returns
    -------
    np.ndarray:
        (N, 7) array of translation and quaternion
    """
    return np.concatenate(
        (quat_translation[..., 4:7], quat_translation[..., 0:4]), axis=-1
    )


class AriaVRSExtractor:
    TAGS = ["aria", "robotics", "vrs"]

    @staticmethod
    def process_episode(
        episode_path,
        arm: str,
        low_res=False,
        height=480,
        width=640,
        convert_mano: bool = True,
        mano_model_dir: str | None = None,
        mano_device: str | None = None,
        mano_n_iters: int = 400,
        mano_lr: float = 0.02,
        mano_beta_reg: float = 0.01,
        mano_chunk_size: int = 512,
    ):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the VRS file containing the episode data.
        arm : str
            String for which arm to add data for
        convert_mano : bool
            If True (default), fit MANO to the raw Aria keypoints and emit:
              <side>.obs_keypoints      = MANO-converted keypoints (canonical
                                          MANO joint order, world frame) —
                                          the key downstream consumers read.
              <side>.obs_aria_keypoints = raw Aria keypoints (Aria layout),
                                          preserved for reference.
            If False, legacy schema: raw Aria keypoints under
            <side>.obs_keypoints, no <side>.obs_aria_keypoints.
        mano_model_dir / mano_device / mano_n_iters / mano_lr / mano_beta_reg /
        mano_chunk_size :
            MANO fit knobs; defaults match the validated tutorial settings
            (400 Adam iters @ lr 0.02, beta_reg 0.01, cuda>mps>cpu).
        Returns
        -------
        episode_feats : dict
            Dictionary mapping keys in the episode to episode features, for example:
                hand.<cartesian>   : (world frame) (6D per arm)
                hand.<keypoints>   : (world frame) (3 cartesian + 4 quaternion + 63 dim (21 keypoints) per arm)
                images.<camera_key>    :
                head_pose              : (world frame)

            #TODO: Add metadata to be a nested dict

        """
        episode_feats = dict()

        # file setup and opening
        root_dir = episode_path.parent

        mps_sample_path = os.path.join(root_dir, ("mps_" + episode_path.stem + "_vrs"))

        hand_tracking_results_path = os.path.join(
            mps_sample_path, "hand_tracking", "hand_tracking_results.csv"
        )

        closed_loop_pose_path = os.path.join(
            mps_sample_path, "slam", "closed_loop_trajectory.csv"
        )

        eye_gaze_path = os.path.join(
            mps_sample_path, "eye_gaze", "general_eye_gaze.csv"
        )
        use_eye_gaze = os.path.exists(eye_gaze_path)
        # TODO: in the future might write to sql on the failure due to mps processing failures
        if not os.path.exists(hand_tracking_results_path):
            raise FileNotFoundError(
                f"Hand tracking results file not found at {hand_tracking_results_path}"
            )
        if not os.path.exists(closed_loop_pose_path):
            raise FileNotFoundError(
                f"Closed loop pose file not found at {closed_loop_pose_path}"
            )

        vrs_reader = data_provider.create_vrs_data_provider(str(episode_path))

        hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
            hand_tracking_results_path
        )

        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_pose_path)
        if use_eye_gaze:
            eye_gaze_results = mps.read_eyegaze(eye_gaze_path)

        time_domain: TimeDomain = TimeDomain.DEVICE_TIME

        stream_ids: Dict[str, StreamId] = {
            "rgb": StreamId("214-1"),
            "slam-left": StreamId("1201-1"),
            "slam-right": StreamId("1201-2"),
        }
        stream_timestamps_ns: Dict[str, List[int]] = {
            key: vrs_reader.get_timestamps_ns(stream_id, time_domain)
            for key, stream_id in stream_ids.items()
        }

        rgb_to_device_T = slam_to_rgb(
            vrs_reader, height=height, width=width
        )  # aria sophus SE3

        hand_cartesian_pose = AriaVRSExtractor.get_ee_pose(
            world_device_T=closed_loop_traj,
            stream_timestamps_ns=stream_timestamps_ns,
            hand_tracking_results=hand_tracking_results,
            arm=arm,
        )

        hand_keypoints_pose = AriaVRSExtractor.get_hand_keypoints(
            world_device_T=closed_loop_traj,
            stream_timestamps_ns=stream_timestamps_ns,
            hand_tracking_results=hand_tracking_results,
            arm=arm,
        )

        head_pose = AriaVRSExtractor.get_head_pose(
            world_device_T=closed_loop_traj,
            device_rgb_T=rgb_to_device_T.inverse(),
            stream_timestamps_ns=stream_timestamps_ns,
        )
        if use_eye_gaze:
            eye_gaze = AriaVRSExtractor.get_eye_gaze(
                eye_gaze_results=eye_gaze_results,
                stream_timestamps_ns=stream_timestamps_ns,
            )

        images = AriaVRSExtractor.get_images(
            vrs_reader=vrs_reader,
            stream_ids=stream_ids,
            stream_timestamps_ns=stream_timestamps_ns,
            height=height,
            width=width,
        )

        if low_res:
            images = downsample_hwc_uint8_in_chunks(
                images, out_hw=(240, 320), chunk=256
            )

        rgb_timestamps_ns = np.array(stream_timestamps_ns["rgb"])
        print(f"[DEBUG] LENGTH BEFORE CLEANING: {len(hand_cartesian_pose)}")
        mask_data = [images, rgb_timestamps_ns]
        filter_mask_data = [hand_cartesian_pose, hand_keypoints_pose, head_pose]
        if use_eye_gaze:
            mask_data.append(eye_gaze)

        (
            output_filter_mask_data,
            output_mask_data,
        ) = AriaVRSExtractor.clean_data(
            filter_mask_data=filter_mask_data,
            mask_data=mask_data,
        )

        hand_cartesian_pose = output_filter_mask_data[0]
        hand_keypoints_pose = output_filter_mask_data[1]
        head_pose = output_filter_mask_data[2]

        if use_eye_gaze:
            eye_gaze = output_mask_data[2]

        images = output_mask_data[0]
        rgb_timestamps_ns = output_mask_data[1]

        print(f"[DEBUG] LENGTH AFTER CLEANING: {len(hand_cartesian_pose)}")

        use_left_hand = arm == "left" or arm == "both"
        use_right_hand = arm == "right" or arm == "both"

        mano_cfg = dict(
            model_dir=mano_model_dir,
            device=mano_device,
            n_iters=mano_n_iters,
            lr=mano_lr,
            beta_reg=mano_beta_reg,
            chunk_size=mano_chunk_size,
        )

        if use_left_hand:
            episode_feats["left.obs_ee_pose"] = hand_cartesian_pose[..., :7]
            left_raw_kp = hand_keypoints_pose[..., 7 : 7 + 21 * 3]
            episode_feats["left.obs_wrist_pose"] = hand_keypoints_pose[..., :7]
            if convert_mano:
                episode_feats["left.obs_aria_keypoints"] = left_raw_kp
                print(f"[MANO] Fitting LEFT hand ({left_raw_kp.shape[0]} frames)...")
                left_mano_kp = convert_aria_keypoints_to_mano(
                    left_raw_kp, is_rhand=False, **mano_cfg
                )
                episode_feats["left.obs_keypoints"] = left_mano_kp
                # The cartesian EE pose is derived from the fitted MANO
                # keypoints, not Aria's MPS palm/wrist/normal triple (which
                # intermittently emitted ~180deg-flipped frames).
                episode_feats["left.obs_ee_pose"] = mano_keypoints_to_cartesian(
                    left_mano_kp, is_rhand=False
                )
            else:
                episode_feats["left.obs_keypoints"] = left_raw_kp

        if use_right_hand:
            if arm == "both":
                episode_feats["right.obs_ee_pose"] = hand_cartesian_pose[..., 7:]
                right_raw_kp = hand_keypoints_pose[
                    ..., 7 + 21 * 3 + 7 : 7 + 21 * 3 + 7 + 21 * 3
                ]
                episode_feats["right.obs_wrist_pose"] = hand_keypoints_pose[
                    ..., 7 + 21 * 3 : 7 + 21 * 3 + 7
                ]
            elif arm == "right":
                episode_feats["right.obs_ee_pose"] = hand_cartesian_pose[..., :7]
                right_raw_kp = hand_keypoints_pose[..., 7 : 7 + 21 * 3]
                episode_feats["right.obs_wrist_pose"] = hand_keypoints_pose[..., :7]
            if convert_mano:
                episode_feats["right.obs_aria_keypoints"] = right_raw_kp
                print(f"[MANO] Fitting RIGHT hand ({right_raw_kp.shape[0]} frames)...")
                right_mano_kp = convert_aria_keypoints_to_mano(
                    right_raw_kp, is_rhand=True, **mano_cfg
                )
                episode_feats["right.obs_keypoints"] = right_mano_kp
                # See the left-hand note: cartesian comes from the MANO fit.
                episode_feats["right.obs_ee_pose"] = mano_keypoints_to_cartesian(
                    right_mano_kp, is_rhand=True
                )
            else:
                episode_feats["right.obs_keypoints"] = right_raw_kp
        episode_feats["images.front_1"] = images
        episode_feats["obs_head_pose"] = head_pose
        if use_eye_gaze:
            episode_feats["obs_eye_gaze"] = eye_gaze
        episode_feats["obs_rgb_timestamps_ns"] = rgb_timestamps_ns

        return episode_feats

    @staticmethod
    def clean_data(filter_mask_data, mask_data):
        """
        Clean data
        Parameters
        ----------
        actions : np.arrayoses
        pose : np.array
        images : np.array
        Returns
        -------
        actions, pose, images : tuple of np.array
            cleaned data
        """
        mask = np.ones(len(filter_mask_data[0]), dtype=bool)
        for pose in filter_mask_data:
            bad_data_mask = np.any(pose >= 1e8, axis=1)
            mask = mask & ~bad_data_mask

        for i in range(len(filter_mask_data)):
            filter_mask_data[i] = filter_mask_data[i][mask]
        for i in range(len(mask_data)):
            mask_data[i] = mask_data[i][mask]

        return filter_mask_data, mask_data

    @staticmethod
    def get_images(
        vrs_reader,
        stream_ids: dict,
        stream_timestamps_ns: dict,
        height=480,
        width=640,
    ):
        """
        Get RGB Image from VRS
        Parameters
        ----------
        vrs_reader : VRS Data Provider
            Object that reads and obtains data from VRS
        stream_ids : dict
            maps sensor keys to a list of ids for Aria
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        Returns
        -------
        images : np.array
            rgb images undistorted to 480x640x3
        """
        images = []
        frame_length = len(stream_timestamps_ns["rgb"])

        time_domain = TimeDomain.DEVICE_TIME
        time_query_closest = TimeQueryOptions.CLOSEST

        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]

            sample_frame = vrs_reader.get_image_data_by_time_ns(
                stream_ids["rgb"],
                query_timestamp,
                time_domain,
                time_query_closest,
            )

            image_t = undistort_to_linear(
                vrs_reader,
                stream_ids,
                raw_image=sample_frame[0].to_numpy_array(),
                height=height,
                width=width,
            )

            images.append(image_t)
        images = np.array(images)
        return images

    @staticmethod
    def get_hand_keypoints(
        world_device_T,
        stream_timestamps_ns: dict,
        hand_tracking_results,
        arm: str,
    ):
        """
        Get Hand Keypoints from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
        hand_tracking_results : dict
        arm : str
            arm to get hand keypoints for
        Returns
        -------
        hand_keypoints : np.array
            hand_keypoints
        """
        frame_length = len(stream_timestamps_ns["rgb"])

        keypoints = []

        use_left_hand = arm == "left" or arm == "both"
        use_right_hand = arm == "right" or arm == "both"
        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            hand_tracking_result_t = get_nearest_hand_tracking_result(
                hand_tracking_results, query_timestamp
            )
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device

            right_confidence = getattr(
                getattr(hand_tracking_result_t, "right_hand", None), "confidence", -1
            )
            left_confidence = getattr(
                getattr(hand_tracking_result_t, "left_hand", None), "confidence", -1
            )
            left_obs_t = np.full(7 + 21 * 3, 1e9)
            if (
                use_left_hand
                and not left_confidence < 0
                and world_device_T_t is not None
            ):
                left_hand_keypoints = np.stack(
                    hand_tracking_result_t.left_hand.landmark_positions_device, axis=0
                )
                wrist_T = (
                    hand_tracking_result_t.left_hand.transform_device_wrist
                )  # Sophus SE3

                world_wrist_T = world_device_T_t @ wrist_T
                world_keypoints = (
                    world_device_T_t @ left_hand_keypoints.T
                ).T  # keypoints are in device frame

                world_wrist_T = sp.SE3.from_matrix(
                    T_rot_orientation(world_wrist_T.to_matrix(), T_ROT_CAM)
                )
                wrist_quat_and_translation = quat_translation_swap(
                    world_wrist_T.to_quat_and_translation()
                )
                if wrist_quat_and_translation.ndim == 2:
                    wrist_quat_and_translation = wrist_quat_and_translation[0]
                left_obs_t[:7] = wrist_quat_and_translation
                left_obs_t[7:] = world_keypoints.flatten()

            right_obs_t = np.full(7 + 21 * 3, 1e9)
            if (
                use_right_hand
                and not right_confidence < 0
                and world_device_T_t is not None
            ):
                right_hand_keypoints = np.stack(
                    hand_tracking_result_t.right_hand.landmark_positions_device, axis=0
                )
                wrist_T = (
                    hand_tracking_result_t.right_hand.transform_device_wrist
                )  # Sophus SE3

                world_wrist_T = world_device_T_t @ wrist_T
                world_keypoints = (
                    world_device_T_t @ right_hand_keypoints.T
                ).T  # keypoints are in device frame

                world_wrist_T = sp.SE3.from_matrix(
                    T_rot_orientation(
                        world_wrist_T.to_matrix(),
                        T_ROT_CAM,
                    )
                )
                wrist_quat_and_translation = quat_translation_swap(
                    world_wrist_T.to_quat_and_translation()
                )
                if wrist_quat_and_translation.ndim == 2:
                    wrist_quat_and_translation = wrist_quat_and_translation[0]
                right_obs_t[:7] = wrist_quat_and_translation
                right_obs_t[7:] = world_keypoints.flatten()

            if use_left_hand and use_right_hand:
                keypoints_obs_t = np.concatenate((left_obs_t, right_obs_t), axis=-1)
            elif use_left_hand:
                keypoints_obs_t = left_obs_t
            elif use_right_hand:
                keypoints_obs_t = right_obs_t
            else:
                raise ValueError(f"Incorrect arm provided: {arm}")
            keypoints.append(np.ravel(keypoints_obs_t))
        keypoints = np.array(keypoints)
        return keypoints

    @staticmethod
    def get_head_pose(
        world_device_T,
        device_rgb_T,
        stream_timestamps_ns: dict,
    ):
        """
        Get Head Pose from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time

        Returns
        -------
        head_pose : np.array
            head_pose
        """
        head_pose = []
        frame_length = len(stream_timestamps_ns["rgb"])

        rgb_to_rgbprime_rot = np.eye(4)
        rgb_to_rgbprime_rot[:3, :3] = ROTATION_MATRIX.T
        rgb_to_rgbprime_T = sp.SE3.from_matrix(rgb_to_rgbprime_rot)
        rgbprime_to_rgb_T = rgb_to_rgbprime_T.inverse()
        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device
            head_pose_obs_t = np.full(7, 1e9)
            if world_device_T_t is not None:
                world_rgb_T_t = world_device_T_t @ device_rgb_T @ rgbprime_to_rgb_T
                head_pose_quat_and_translation = quat_translation_swap(
                    world_rgb_T_t.to_quat_and_translation()
                )
                if head_pose_quat_and_translation.ndim == 2:
                    head_pose_quat_and_translation = head_pose_quat_and_translation[0]
                head_pose_obs_t[:7] = head_pose_quat_and_translation
            head_pose.append(np.ravel(head_pose_obs_t))
        head_pose = np.array(head_pose)
        return head_pose

    @staticmethod
    def get_eye_gaze(
        eye_gaze_results,
        stream_timestamps_ns: dict,
    ):
        gaze = []
        frame_length = len(stream_timestamps_ns["rgb"])
        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            gaze_info = get_nearest_eye_gaze(eye_gaze_results, query_timestamp)
            if gaze_info is None:
                gaze.append([1e9, 1e9, 1e9])
            else:
                gaze.append([gaze_info.yaw, gaze_info.pitch, gaze_info.depth])
        gaze = np.array(gaze)
        return gaze

    @staticmethod
    def get_ee_pose(
        world_device_T,
        stream_timestamps_ns: dict,
        hand_tracking_results,
        arm: str,
    ):
        """
        Get EE Pose from VRS
        Parameters
        ----------
        world_device_T : np.array
            Transform from world coordinates to ARIA camera frame
        stream_timestamps_ns : dict
            dict that maps sensor keys to a list of nanosecond timestamps in device time
        hand_tracking_results : dict
            dict that maps sensor keys to a list of hand tracking results
        arm : str
            arm to get hand keypoints for
        Returns
        -------
        ee_pose : np.array
            ee_pose (6D per arm)
            -1 if no hand tracking data is available
        """
        ee_pose = []
        frame_length = len(stream_timestamps_ns["rgb"])

        use_left_hand = arm == "left" or arm == "both"
        use_right_hand = arm == "right" or arm == "both"

        for t in range(frame_length):
            query_timestamp = stream_timestamps_ns["rgb"][t]
            hand_tracking_result_t = get_nearest_hand_tracking_result(
                hand_tracking_results, query_timestamp
            )
            world_device_T_t = get_nearest_pose(world_device_T, query_timestamp)
            if world_device_T_t is not None:
                world_device_T_t = world_device_T_t.transform_world_device

            right_confidence = getattr(
                getattr(hand_tracking_result_t, "right_hand", None), "confidence", -1
            )
            left_confidence = getattr(
                getattr(hand_tracking_result_t, "left_hand", None), "confidence", -1
            )

            left_obs_t = np.full(7, 1e9)
            if (
                use_left_hand
                and not left_confidence < 0
                and world_device_T_t is not None
            ):
                left_palm_pose = (
                    hand_tracking_result_t.left_hand.get_palm_position_device()
                )
                left_wrist_pose = (
                    hand_tracking_result_t.left_hand.get_wrist_position_device()
                )
                left_palm_normal = hand_tracking_result_t.left_hand.wrist_and_palm_normal_device.palm_normal_device

                left_rot_matrix = compute_orientation_rotation_matrix(
                    palm_pose=left_palm_pose,
                    wrist_pose=left_wrist_pose,
                    palm_normal=left_palm_normal,
                )
                left_T_t = np.eye(4)
                left_T_t[:3, :3] = left_rot_matrix
                left_T_t[:3, 3] = left_palm_pose
                left_T_t = sp.SE3.from_matrix(left_T_t)
                left_T_t = world_device_T_t @ left_T_t
                left_T_t = sp.SE3.from_matrix(
                    T_rot_orientation(left_T_t.to_matrix(), T_ROT_CAM)
                )

                left_quat_and_translation = quat_translation_swap(
                    left_T_t.to_quat_and_translation()
                )
                if left_quat_and_translation.ndim == 2:
                    left_quat_and_translation = left_quat_and_translation[0]
                left_obs_t[:7] = left_quat_and_translation

            right_obs_t = np.full(7, 1e9)
            if (
                use_right_hand
                and not right_confidence < 0
                and world_device_T_t is not None
            ):
                right_palm_pose = (
                    hand_tracking_result_t.right_hand.get_palm_position_device()
                )
                right_wrist_pose = (
                    hand_tracking_result_t.right_hand.get_wrist_position_device()
                )
                right_palm_normal = hand_tracking_result_t.right_hand.wrist_and_palm_normal_device.palm_normal_device

                right_rot_matrix = compute_orientation_rotation_matrix(
                    palm_pose=right_palm_pose,
                    wrist_pose=right_wrist_pose,
                    palm_normal=right_palm_normal,
                )
                right_T_t = np.eye(4)
                right_T_t[:3, :3] = right_rot_matrix
                right_T_t[:3, 3] = right_palm_pose
                right_T_t = sp.SE3.from_matrix(right_T_t)
                right_T_t = world_device_T_t @ right_T_t
                right_T_t = sp.SE3.from_matrix(
                    T_rot_orientation(
                        right_T_t.to_matrix(), T_ROT_CAM
                    )
                )
                right_quat_and_translation = quat_translation_swap(
                    right_T_t.to_quat_and_translation()
                )
                if right_quat_and_translation.ndim == 2:
                    right_quat_and_translation = right_quat_and_translation[0]
                right_obs_t[:7] = right_quat_and_translation

            if use_left_hand and use_right_hand:
                ee_pose_obs_t = np.concatenate((left_obs_t, right_obs_t), axis=-1)
            elif use_left_hand:
                ee_pose_obs_t = left_obs_t
            elif use_right_hand:
                ee_pose_obs_t = right_obs_t
            else:
                raise ValueError(f"Incorrect arm provided: {arm}")
            ee_pose.append(np.ravel(ee_pose_obs_t))
        ee_pose = np.array(ee_pose)
        return ee_pose
