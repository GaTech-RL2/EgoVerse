import logging
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

from egomimic.scripts.aria_process.gen2_filter import (
    Gen2FilterConfig,
    Gen2FrameSignals,
    classify_frames,
    demo_segments,
)
from egomimic.utils.pose_utils import T_rot_orientation

logger = logging.getLogger(__name__)

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
    0,  # wrist
    13,
    14,
    15,
    16,  # thumb (3 joints + tip)
    1,
    2,
    3,
    17,  # index
    4,
    5,
    6,
    18,  # middle
    10,
    11,
    12,
    19,  # ring
    7,
    8,
    9,
    20,  # pinky
]

# Aria's 21-keypoint layout: 0-4 fingertips (thumb..pinky), 5 palm root/wrist,
# 6-7 thumb intermediates, 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky,
# 20 palm center (dropped from the fit loss).
# Aria idx -> otaheri MANO target idx; MANO's thumb CMC has no Aria source and
# is filled in by MANO's kinematic prior.
ARIA_TO_OTAHERI = {
    5: 0,  # wrist
    6: 14,  # thumb MCP
    7: 15,  # thumb IP
    0: 16,  # thumb tip
    8: 1,  # index1
    9: 2,  # index2
    10: 3,  # index3
    1: 17,  # index tip
    11: 4,  # middle1
    12: 5,  # middle2
    13: 6,  # middle3
    2: 18,  # middle tip
    14: 10,  # ring1
    15: 11,  # ring2
    16: 12,  # ring3
    3: 19,  # ring tip
    17: 7,  # pinky1
    18: 8,  # pinky2
    19: 9,  # pinky3
    4: 20,  # pinky tip
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=lr * 0.05
    )

    aria_idx_t = torch.tensor(ARIA_IDX_USED, device=device, dtype=torch.long)
    mano_idx_t = torch.tensor(MANO_IDX_TARGET, device=device, dtype=torch.long)

    targets = aria_kp.index_select(1, aria_idx_t)  # (N, 20, 3)
    point_valid = torch.isfinite(targets).all(dim=-1)  # (N, 20)
    targets = torch.nan_to_num(targets, nan=0.0)

    for step in range(n_iters):
        out = model(
            betas=beta,
            global_orient=R_global,
            hand_pose=theta,
            transl=t_global,
            return_verts=True,
            return_tips=True,
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
            print(
                f"    iter {step:4d}  pos={pos_loss.item():.5f}  reg={reg_loss.item():.5f}"
            )

    with torch.no_grad():
        out = model(
            betas=beta,
            global_orient=R_global,
            hand_pose=theta,
            transl=t_global,
            return_verts=True,
            return_tips=True,
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
    kp = torch.as_tensor(np.asarray(keypoints_t63), dtype=torch.float32).reshape(
        -1, 21, 3
    )
    if kp.shape[0] == 0:
        return np.asarray(keypoints_t63, dtype=np.float64).reshape(-1, 63)
    # Sentinel (1e9) -> NaN so the fit's validity masks apply.
    kp = torch.where(
        kp.abs() < _ARIA_INVALID_SENTINEL, kp, torch.full_like(kp, float("nan"))
    )
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


def is_aria_gen2(provider) -> bool:
    return provider.get_device_version() == calibration.DeviceVersion.Gen2


def undistort_to_linear(
    provider,
    stream_ids,
    raw_image,
    camera_label="rgb",
    height=480,
    width=640,
    rotate=True,
):
    """Undistort to a pinhole image of size (height, width).

    Gen 1 mounts the RGB sensor sideways, so the linear image is rendered
    portrait and rotated upright (rotate=True). Gen 2 RGB is already upright
    landscape (rotate=False). Both produce the same focal length, so
    ARIA_INTRINSICS holds for either generation.
    """
    camera_label = provider.get_label_from_stream_id(stream_ids[camera_label])
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    focal_length = 133.25430222 * (height / 240)
    # get_linear_camera_calibration takes (width, height).
    linear_w, linear_h = (height, width) if rotate else (width, height)
    warped = calibration.get_linear_camera_calibration(
        linear_w,
        linear_h,
        focal_length,
        camera_label,
        calib.get_transform_device_camera(),
    )
    warped_image = calibration.distort_by_calibration(raw_image, warped, calib)
    if rotate:
        warped_image = np.rot90(warped_image, k=3)
    return warped_image


# Gen 2 on-device machine perception (MP) streams.
GEN2_HAND_STREAM = StreamId("371-1")
GEN2_VIO_STREAM = StreamId("371-2")
GEN2_VIO_HIGH_FREQ_STREAM = StreamId("371-3")
GEN2_EYE_GAZE_STREAM = StreamId("373-1")


def _tracking_ts_ns(items) -> np.ndarray:
    return np.array([int(x.tracking_timestamp.total_seconds() * 1e9) for x in items])


def load_gen2_on_device_gaze(provider) -> list:
    """On-device eye gaze (empty list if the VRS has no gaze stream)."""
    if str(GEN2_EYE_GAZE_STREAM) not in {str(s) for s in provider.get_all_streams()}:
        return []
    return [
        provider.get_eye_gaze_data_by_index(GEN2_EYE_GAZE_STREAM, i)
        for i in range(provider.get_num_data(GEN2_EYE_GAZE_STREAM))
    ]


def select_gen2_eye_gaze(mps_gaze: list, on_device_gaze: list) -> tuple[list, str]:
    """Eye gaze to write for a Gen 2 recording converted from MPS.

    projectaria-tools < 2.3 cannot parse the Gen 2 MPS general_eye_gaze.csv
    (new columns) and returns no samples; fall back to on-device gaze then.
    Returns (gaze results, source name).
    """
    if len(mps_gaze):
        return mps_gaze, "mps"
    if len(on_device_gaze):
        return on_device_gaze, "on_device"
    return [], "none"


def load_gen2_on_device_mp(provider, rgb_timestamps_ns):
    """Read Gen 2 on-device hand tracking, VIO and eye gaze from the VRS.

    Returns objects shaped like the Gen 1 MPS outputs so the existing get_*
    helpers can consume them unchanged:
      hand_tracking_results : list[mps.hand_tracking.HandTrackingResult]
      world_device_T        : list[mps.ClosedLoopTrajectoryPose], one per RGB
                              frame from the high-frequency VIO pose ("world"
                              is the VIO odometry frame)
      eye_gaze_results      : list[mps.EyeGaze] (empty if no gaze stream)
      vio_ok                : (T,) bool, VIO status VALID with GOOD quality
    """
    time_domain = TimeDomain.DEVICE_TIME
    closest = TimeQueryOptions.CLOSEST
    streams = {str(s) for s in provider.get_all_streams()}
    for sid in (GEN2_HAND_STREAM, GEN2_VIO_STREAM, GEN2_VIO_HIGH_FREQ_STREAM):
        if str(sid) not in streams:
            raise ValueError(f"Gen 2 VRS is missing on-device stream {sid}")

    hand_tracking_results = [
        provider.get_hand_pose_data_by_index(GEN2_HAND_STREAM, i)
        for i in range(provider.get_num_data(GEN2_HAND_STREAM))
    ]
    world_device_T = []
    vio_ok = np.ones(len(rgb_timestamps_ns), bool)
    for t, ts in enumerate(rgb_timestamps_ns):
        vio = provider.get_vio_data_by_time_ns(
            GEN2_VIO_STREAM, ts, time_domain, closest
        )
        hf = provider.get_vio_high_freq_data_by_time_ns(
            GEN2_VIO_HIGH_FREQ_STREAM, ts, time_domain, closest
        )
        vio_ok[t] = vio.status.name == "VALID" and vio.pose_quality.name == "GOOD"
        pose = mps.ClosedLoopTrajectoryPose()
        pose.tracking_timestamp = hf.tracking_timestamp
        pose.transform_world_device = hf.transform_odometry_device
        world_device_T.append(pose)
    return (
        hand_tracking_results,
        world_device_T,
        load_gen2_on_device_gaze(provider),
        vio_ok,
    )


def gen2_utc_timestamps_ns(provider, rgb_timestamps_ns) -> np.ndarray:
    """UTC (epoch ns) of each RGB frame, from the VRS's device-to-UTC sync.

    Falls back to the recording's start_time_epoch_sec plus the device-time
    offset (second resolution) when the VRS has no UTC sync.
    """
    rgb_timestamps_ns = np.asarray(rgb_timestamps_ns, dtype=np.int64)
    rgb_id = StreamId("214-1")
    if provider.supports_time_domain(rgb_id, TimeDomain.UTC):
        device = np.asarray(
            provider.get_timestamps_ns(rgb_id, TimeDomain.DEVICE_TIME), dtype=np.int64
        )
        utc = np.asarray(
            provider.get_timestamps_ns(rgb_id, TimeDomain.UTC), dtype=np.int64
        )
        # Integer median: float64 would lose ~100 ns at epoch-ns magnitudes.
        diffs = np.sort(utc - device)
        offset = int(diffs[len(diffs) // 2])
    else:
        logger.warning("VRS has no UTC sync; demo times are second-resolution")
        offset = provider.get_metadata().start_time_epoch_sec * 1_000_000_000 - int(
            provider.get_timestamps_ns(rgb_id, TimeDomain.DEVICE_TIME)[0]
        )
    return rgb_timestamps_ns + offset


def _nearest_index(sorted_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Index of the nearest element of sorted_ts for every query (-1 if empty)."""
    if not len(sorted_ts):
        return np.full(len(query_ts), -1)
    hi = np.clip(np.searchsorted(sorted_ts, query_ts), 1, len(sorted_ts) - 1)
    lo = hi - 1
    if len(sorted_ts) == 1:
        return np.zeros(len(query_ts), dtype=int)
    return np.where(
        np.abs(sorted_ts[lo] - query_ts) <= np.abs(sorted_ts[hi] - query_ts), lo, hi
    )


def gen2_frame_signals(
    rgb_timestamps_ns,
    world_device_T,
    hand_tracking_results,
    gaze_results,
    tracking_ok,
    cfg: Gen2FilterConfig,
) -> Gen2FrameSignals:
    """Per-RGB-frame inputs for gen2_filter.classify_frames.

    world_device_T : sorted poses with .tracking_timestamp and
                     .transform_world_device (MPS closed loop or on-device VIO)
    hand_tracking_results : sorted HandTrackingResult list (MPS or on-device)
    gaze_results : sorted on-device EyeGaze list; combined_gaze_valid is the
                   glasses-on-face signal used to find resets
    tracking_ok : (T,) bool, extra per-frame validity from the source
    """
    rgb = np.asarray(rgb_timestamps_ns, dtype=np.int64)
    T = len(rgb)
    pose_ts = _tracking_ts_ns(world_device_T)
    hand_ts = _tracking_ts_ns(hand_tracking_results)
    gaze_ts = _tracking_ts_ns(gaze_results)
    pose_i = _nearest_index(pose_ts, rgb)
    hand_i = _nearest_index(hand_ts, rgb)
    gaze_i = _nearest_index(gaze_ts, rgb)

    def close(ts, idx, max_dt_s):
        if not len(ts):
            return np.zeros(T, bool)
        return np.abs(ts[idx] - rgb) <= max_dt_s * 1e9

    tracking_ok = (
        np.array(tracking_ok, bool)
        & close(pose_ts, pose_i, cfg.max_pose_dt_s)
        & close(hand_ts, hand_i, cfg.max_hand_dt_s)
    )
    gaze_available = bool(len(gaze_ts))
    if gaze_available:
        gaze_ok = close(gaze_ts, gaze_i, cfg.max_gaze_dt_s)
        gaze_valid = gaze_ok & np.array(
            [bool(gaze_results[i].combined_gaze_valid) for i in gaze_i]
        )
    else:
        logger.warning(
            "No on-device eye gaze: resets (glasses off the face) cannot be "
            "detected, so demos will not be split at resets"
        )
        gaze_valid = np.ones(T, bool)

    forward = np.zeros((T, 3))
    hand_min_dist = np.full(T, np.nan)
    left_tracked = np.zeros(T, bool)
    right_tracked = np.zeros(T, bool)
    for t in range(T):
        if pose_i[t] >= 0:
            T_wd = world_device_T[pose_i[t]].transform_world_device.to_matrix()
            forward[t] = T_wd.reshape(-1, 4, 4)[0][:3, 2]
        if hand_i[t] >= 0:
            hands = hand_tracking_results[hand_i[t]]
            left, right = hands.left_hand, hands.right_hand
            left_tracked[t] = left is not None and not left.confidence < 0
            right_tracked[t] = right is not None and not right.confidence < 0
            dists = [
                np.linalg.norm(np.stack(h.landmark_positions_device), axis=1).min()
                for h in (left, right)
                if h is not None
            ]
            if dists:
                hand_min_dist[t] = min(dists)
    return Gen2FrameSignals(
        t_ns=rgb,
        tracking_ok=tracking_ok,
        gaze_valid=gaze_valid,
        device_forward_world=forward,
        hand_min_dist_m=hand_min_dist,
        left_tracked=left_tracked,
        right_tracked=right_tracked,
        gaze_available=gaze_available,
    )


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


def aria_keypoints_to_canonical_layout(aria_kp_t63: np.ndarray) -> np.ndarray:
    """Reorder raw Aria 21-keypoints into canonical MANO joint order (no fit).

    The thumb CMC has no Aria landmark and is NaN; rows holding the 1e9
    missing-data sentinel become NaN. Enough for mano_keypoints_to_cartesian,
    which only reads the wrist and the four finger MCPs.
    """
    kp = np.asarray(aria_kp_t63, dtype=np.float64).reshape(-1, 21, 3)
    canonical_from_otaheri = {o: c for c, o in enumerate(OTAHERI_TO_CANONICAL)}
    out = np.full_like(kp, np.nan)
    for aria_idx, otaheri_idx in ARIA_TO_OTAHERI.items():
        out[:, canonical_from_otaheri[otaheri_idx]] = kp[:, aria_idx]
    out[np.any(np.abs(kp) >= _ARIA_INVALID_SENTINEL, axis=(1, 2))] = np.nan
    return out.reshape(-1, 63)


def gen2_ee_frame_ok(hand_keypoints_pose: np.ndarray, arm: str) -> np.ndarray:
    """(T,) bool: the keypoint-based EE frame exists for every hand the arm
    uses. hand_keypoints_pose is AriaVRSExtractor.get_hand_keypoints' output."""
    sides = {"left": ("left",), "right": ("right",), "both": ("left", "right")}[arm]
    ok = np.ones(len(hand_keypoints_pose), bool)
    for k, side in enumerate(sides):
        raw_kp = hand_keypoints_pose[..., k * 70 + 7 : k * 70 + 70]
        ee = mano_keypoints_to_cartesian(
            aria_keypoints_to_canonical_layout(raw_kp), is_rhand=side == "right"
        )
        ok &= np.all(ee < 1e8, axis=1)
    return ok


def _set_gen2_hand(episode_feats: dict, side: str, raw_kp: np.ndarray) -> None:
    """Gen 2 without a MANO fit: keep raw keypoints under obs_aria_keypoints
    (obs_keypoints is reserved for canonical MANO) and build obs_ee_pose with
    the same frame convention as the MANO path."""
    episode_feats[f"{side}.obs_aria_keypoints"] = raw_kp
    episode_feats[f"{side}.obs_ee_pose"] = mano_keypoints_to_cartesian(
        aria_keypoints_to_canonical_layout(raw_kp), is_rhand=side == "right"
    )


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
        gen2_source: str = "mps",
        split_demos: bool = True,
        intra_demo_drop: bool = True,
        gen2_filter: Gen2FilterConfig | None = None,
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
        gen2_source : str
            Gen 2 only: "mps" reads mps_<stem>_vrs/ like Gen 1, "on_device"
            reads the VRS's own hand tracking / VIO / eye gaze, "auto" uses
            MPS when its hand_tracking and slam outputs exist.
        split_demos : bool
            Gen 2 only: mark demos (stretches between resets / the opening
            calibration) so the writer can save one episode per demo. Adds
            "_demo_index" (per kept frame) and "_gen2_demos" (per-demo info)
            to the returned dict. Gen 2 recordings also get "_attrs" (zarr
            attrs: device, sources, keypoint layout); callers must pop every
            "_" key before writing.
        intra_demo_drop : bool
            Gen 2 only: drop short in-demo errors (tracking gaps, head sweeps,
            hand at headset, untracked hands) as Gen 1 does. If False, keep
            those frames, write NaN where a pose is missing, and add a
            per-frame "obs_frame_valid" flag.
        gen2_filter : Gen2FilterConfig, optional
            Gen 2 filter thresholds (defaults if None); see gen2_filter.py.
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
        vrs_reader = data_provider.create_vrs_data_provider(str(episode_path))
        gen2 = is_aria_gen2(vrs_reader)

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

        if gen2_source not in ("auto", "mps", "on_device"):
            raise ValueError(f"Unknown gen2_source: {gen2_source!r}")
        has_mps = os.path.exists(hand_tracking_results_path) and os.path.exists(
            closed_loop_pose_path
        )
        gen2_masks = None
        gen2_filter = gen2_filter or Gen2FilterConfig()
        if not gen2:
            split_demos, intra_demo_drop = False, True
        if not intra_demo_drop and convert_mano:
            raise ValueError("intra_demo_drop=False needs convert_mano=False")
        rgb_ts = np.asarray(stream_timestamps_ns["rgb"])
        if gen2 and (
            gen2_source == "on_device" or (gen2_source == "auto" and not has_mps)
        ):
            # Gen 2 also records hand tracking, VIO and eye gaze on device.
            logger.info("Aria Gen 2 recording: using on-device MP streams")
            gen2_source_used, gaze_source = "on_device", "on_device"
            (
                hand_tracking_results,
                closed_loop_traj,
                eye_gaze_results,
                vio_ok,
            ) = load_gen2_on_device_mp(vrs_reader, rgb_ts)
            use_eye_gaze = len(eye_gaze_results) > 0
            gen2_signals = gen2_frame_signals(
                rgb_ts,
                closed_loop_traj,
                hand_tracking_results,
                eye_gaze_results,
                vio_ok,
                gen2_filter,
            )
            gen2_masks = classify_frames(gen2_signals, arm, gen2_filter)
            if not vio_ok.any():
                raise ValueError(
                    "Gen 2 recording has no frames with valid on-device VIO "
                    "(headset never started tracking)"
                )
        else:
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

            hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(
                hand_tracking_results_path
            )

            closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_pose_path)
            if use_eye_gaze:
                eye_gaze_results = mps.read_eyegaze(eye_gaze_path)
            if gen2:
                # MPS poses/hands, but the reset rule still needs the
                # on-device gaze validity (glasses on the face or not).
                logger.info("Aria Gen 2 recording: using MPS outputs")
                gen2_source_used, gaze_source = "mps", "mps"
                on_device_gaze = load_gen2_on_device_gaze(vrs_reader)
                gen2_signals = gen2_frame_signals(
                    rgb_ts,
                    closed_loop_traj,
                    hand_tracking_results,
                    on_device_gaze,
                    np.ones(len(rgb_ts), bool),
                    gen2_filter,
                )
                gen2_masks = classify_frames(gen2_signals, arm, gen2_filter)
                if use_eye_gaze:
                    eye_gaze_results, gaze_source = select_gen2_eye_gaze(
                        eye_gaze_results, on_device_gaze
                    )
                    if gaze_source != "mps":
                        logger.warning(
                            "MPS eye gaze unreadable; using %s eye gaze", gaze_source
                        )
                if use_eye_gaze and not len(eye_gaze_results):
                    logger.warning(
                        "No eye gaze parsed from %s; skipping", eye_gaze_path
                    )
                    use_eye_gaze = False
        if gen2_masks is not None:
            fps = len(rgb_ts) / max((rgb_ts[-1] - rgb_ts[0]) / 1e9, 1e-6)
            for name in ("boundary", "error"):
                n = int(gen2_masks[name].sum())
                logger.info("Gen 2 filter: %s %d frames (%.1f s)", name, n, n / fps)

        if gen2:
            # T_device_rgb straight from the RGB calibration; slam_to_rgb
            # assumes the device frame is the left SLAM camera (Gen 1).
            rgb_calib = vrs_reader.get_device_calibration().get_camera_calib(
                vrs_reader.get_label_from_stream_id(stream_ids["rgb"])
            )
            rgb_to_device_T = rgb_calib.get_transform_device_camera().inverse()
        else:
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
            rgb_rotated=not gen2,
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
            rotate=not gen2,
        )

        if low_res:
            images = downsample_hwc_uint8_in_chunks(
                images, out_hw=(240, 320), chunk=256
            )

        rgb_timestamps_ns = np.array(stream_timestamps_ns["rgb"])
        print(f"[DEBUG] LENGTH BEFORE CLEANING: {len(hand_cartesian_pose)}")
        data = {
            "images": images,
            "rgb_ts": rgb_timestamps_ns,
            "cart": hand_cartesian_pose,
            "kp": hand_keypoints_pose,
            "head": head_pose,
        }
        if use_eye_gaze:
            data["gaze"] = eye_gaze
        filter_keys = ["cart", "kp", "head"]
        demos = []
        if gen2_masks is not None:
            outside = gen2_masks["boundary"]
            # A frame is an in-demo error if a gen2 rule or a missing pose
            # (untracked hand, no head pose) flags it.
            missing = np.zeros(len(rgb_timestamps_ns), bool)
            for k in ("cart", "kp", "head"):
                missing |= np.any(data[k] >= 1e8, axis=1)
            errors = gen2_masks["error"] | missing
            data["error"] = errors[:, None].astype(float)
            if split_demos:
                demo_index, demos, removed = demo_segments(
                    rgb_timestamps_ns,
                    outside,
                    gen2_filter.min_demo_s,
                    usable=~errors,
                )
                outside = demo_index < 0
                data["demo"] = demo_index[:, None]
                data["utc"] = gen2_utc_timestamps_ns(vrs_reader, rgb_timestamps_ns)[
                    :, None
                ]
                t_s = (rgb_timestamps_ns - rgb_timestamps_ns[0]) / 1e9
                for s, e in removed:
                    logger.info(
                        "Gen 2 demos: dropped %.1f-%.1f s, < %.1f s usable",
                        t_s[s],
                        t_s[e - 1],
                        gen2_filter.min_demo_s,
                    )
            if intra_demo_drop:
                drop = outside | (data["error"][:, 0] > 0)
            else:
                drop = outside
                filter_keys = []
            data["drop"] = np.where(drop, 1e9, 0.0)[:, None]
            filter_keys.append("drop")
        mask_keys = [k for k in data if k not in filter_keys]
        kept_filter, kept_mask = AriaVRSExtractor.clean_data(
            filter_mask_data=[data[k] for k in filter_keys],
            mask_data=[data[k] for k in mask_keys],
        )
        data = dict(zip(filter_keys, kept_filter)) | dict(zip(mask_keys, kept_mask))
        hand_cartesian_pose = data["cart"]
        hand_keypoints_pose = data["kp"]
        head_pose = data["head"]
        images = data["images"]
        rgb_timestamps_ns = data["rgb_ts"]
        if use_eye_gaze:
            eye_gaze = data["gaze"]
        if gen2_masks is not None and not intra_demo_drop:
            for arr in (hand_cartesian_pose, hand_keypoints_pose, head_pose):
                arr[arr >= 1e8] = np.nan
            episode_feats["obs_frame_valid"] = data["error"][:, 0] == 0
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
            elif gen2:
                _set_gen2_hand(episode_feats, "left", left_raw_kp)
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
            elif gen2:
                _set_gen2_hand(episode_feats, "right", right_raw_kp)
            else:
                episode_feats["right.obs_keypoints"] = right_raw_kp
        episode_feats["images.front_1"] = images
        episode_feats["obs_head_pose"] = head_pose
        if use_eye_gaze:
            episode_feats["obs_eye_gaze"] = eye_gaze
        episode_feats["obs_rgb_timestamps_ns"] = rgb_timestamps_ns
        if gen2 and intra_demo_drop and not convert_mano:
            # The keypoint-based EE frame can still fail on a degenerate hand.
            bad = ~gen2_ee_frame_ok(hand_keypoints_pose, arm)
            if bad.any():
                logger.info("Gen 2: dropping %d frames with no EE frame", bad.sum())
                for key in list(episode_feats):
                    episode_feats[key] = episode_feats[key][~bad]
                rgb_timestamps_ns = rgb_timestamps_ns[~bad]
                if split_demos:
                    data["demo"] = data["demo"][~bad]
                    data["utc"] = data["utc"][~bad]
        if not len(rgb_timestamps_ns):
            raise ValueError(f"No usable frames left in {episode_path} after filtering")
        if split_demos:
            demo_kept = data["demo"][:, 0]
            utc_kept = data["utc"][:, 0]
            all_ts = np.array(stream_timestamps_ns["rgb"])
            episode_feats["_demo_index"] = demo_kept
            episode_feats["_gen2_demos"] = [
                {
                    "demo_index": d,
                    "start_ns": int(all_ts[s]),
                    "end_ns": int(all_ts[e - 1]),
                    "start_s": float((all_ts[s] - all_ts[0]) / 1e9),
                    "end_s": float((all_ts[e - 1] - all_ts[0]) / 1e9),
                    "span_frames": int(e - s),
                    "error_frames": int(errors[s:e].sum()),
                    "kept_frames": int((demo_kept == d).sum()),
                    # UTC of the demo's first kept frame (None: nothing kept)
                    "first_kept_utc_ns": (
                        int(utc_kept[demo_kept == d][0])
                        if (demo_kept == d).any()
                        else None
                    ),
                }
                for d, (s, e) in enumerate(demos)
            ]
            for info in episode_feats["_gen2_demos"]:
                logger.info(
                    "Gen 2 demo %d: %.1f-%.1f s, %d frames, %d in-demo error "
                    "frames, %d kept",
                    info["demo_index"],
                    info["start_s"],
                    info["end_s"],
                    info["span_frames"],
                    info["error_frames"],
                    info["kept_frames"],
                )

        if gen2:
            if not intra_demo_drop:
                for key, value in episode_feats.items():
                    if key.endswith(("ee_pose", "wrist_pose")):
                        value[value >= 1e8] = np.nan
            episode_feats["_attrs"] = {
                "device": "aria_gen2",
                "mp_source": gen2_source_used,
                "eye_gaze_source": gaze_source if use_eye_gaze else "none",
                "keypoint_layout": (
                    "mano_canonical" if convert_mano else "aria21_obs_aria_keypoints"
                ),
                "ee_pose_source": (
                    "mano_fit" if convert_mano else "aria_keypoints_canonical_frame"
                ),
                "reset_detection": (
                    "eye_gaze" if gen2_signals.gaze_available else "unavailable"
                ),
            }

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
        rotate=True,
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
                rotate=rotate,
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
        rgb_rotated=True,
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

        # Gen 1 images are rotated upright (see undistort_to_linear); Gen 2 are not.
        rgb_to_rgbprime_rot = np.eye(4)
        if rgb_rotated:
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
                    T_rot_orientation(right_T_t.to_matrix(), T_ROT_CAM)
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
