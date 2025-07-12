import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
from scipy.spatial.transform import Rotation
import pytorch_kinematics as pk
import egomimic
import os
import torchvision.transforms.v2.functional as TVTF
import scipy
from numbers import Number
from enum import Enum
import torch.nn as nn
import einops
import pandas as pd
import pyarrow.parquet as pq
import huggingface_hub
import math
import argparse

STD_SCALE = 0.02

ARIA_INTRINSICS = np.array(
    [
        [133.25430222 * 2, 0.0, 320, 0],
        [0.0, 133.25430222 * 2, 240, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)

ARIA_INTRINSICS_HALF = np.array(
    [
        [133.25430222, 0.0, 320 / 2, 0],
        [0.0, 133.25430222, 240 / 2, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)


# Cam to base extrinsics
EXTRINSICS = {
    "ariaJul29": {
        "left": np.array([[-0.02701913, -0.77838164,  0.62720969,  0.1222102 ],
       [ 0.99958387, -0.01469678,  0.02482135,  0.17666979],
       [-0.01010252,  0.62761934,  0.77845482,  0.00423704],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),

       "right": np.array([[ 0.07280155, -0.81760187,  0.57116295,  0.12038065],
       [ 0.9973441 ,  0.05843903, -0.04346979, -0.31690207],
       [ 0.00216277,  0.57281067,  0.81968485, -0.03742754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    },
    "ariaJul29L": np.array([[-0.02701913, -0.77838164,  0.62720969,  0.1222102 ],
       [ 0.99958387, -0.01469678,  0.02482135,  0.17666979],
       [-0.01010252,  0.62761934,  0.77845482,  0.00423704],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    "ariaJul29R": np.array([[ 0.07280155, -0.81760187,  0.57116295,  0.12038065],
       [ 0.9973441 ,  0.05843903, -0.04346979, -0.31690207],
       [ 0.00216277,  0.57281067,  0.81968485, -0.03742754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    "ariaFeb17": {
        "left": np.array([[ 0.17238669, -0.87525225,  0.45190301,  0.17410326],
       [ 0.98220447,  0.11801606, -0.14610472,  0.16975309],
       [ 0.07454667,  0.46904766,  0.88002107, -0.14999786],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),

       "right": np.array([[-0.05745408, -0.7655291 ,  0.64083089,  0.08510947],
       [ 0.99760645, -0.01928462,  0.06640383, -0.37591887],
       [-0.03847589,  0.6431122 ,  0.76480475, -0.24081715],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    },
    "ariaMar4": {
        "left": np.array([[ 0.17238669, -0.87525225,  0.45190301,  0.17410326],
       [ 0.98220447,  0.11801606, -0.14610472,  0.16975309],
       [ 0.07454667,  0.46904766,  0.88002107, -0.14999786],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),

       "right": np.array([[-0.01705736, -0.76701011,  0.64140825,  0.1040253 ],
       [ 0.99936749,  0.00694131,  0.03487736, -0.31958691],
       [-0.0312035 ,  0.64159747,  0.76640657, -0.14678789],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    },    
    "ariaJun7": {
        "left": np.array([[-0.18832572, -0.65324461,  0.73335183,  0.09628296],
       [ 0.98061435, -0.08392577,  0.17706487,  0.13806555],
       [-0.05411956,  0.7524812 ,  0.65638641, -0.03473177],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
        "right": np.array([[ 0.09025644, -0.73272786,  0.67450994,  0.02224729],
       [ 0.985515  , -0.03192781, -0.1665557 , -0.36016571],
       [ 0.14357562,  0.67977239,  0.71923261, -0.36191491],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    },
}

INTRINSICS = {
    "base": ARIA_INTRINSICS,
    "base_half": ARIA_INTRINSICS_HALF
}

class CameraTransforms:
    def __init__(self, intrinsics_key, extrinsics_key):
        self.intrinsics = INTRINSICS[intrinsics_key]
        self.extrinsics = EXTRINSICS[extrinsics_key]

## HPT Utils
def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    # Create position tensor
    positions = torch.arange(position_start, position_end, dtype=torch.float32)

    # Create division term for angles
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))

    # Create empty table
    sinusoid_table = torch.zeros((position_end - position_start, d_hid))

    # Fill even indices with sin and odd indices with cos
    sinusoid_table[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
    sinusoid_table[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term[:d_hid//2])

    return sinusoid_table.unsqueeze(0)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)

def download_from_huggingface(huggingface_repo_id: str):

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder

def fmt(v):
    # Convert to flat list of floats no matter the input shape/type
    if isinstance(v, torch.Tensor):
        v = v.flatten().tolist()
    elif isinstance(v, np.ndarray):
        v = v.flatten().tolist()
    return ", ".join(f"{f:.2f}" for f in v)

def draw_rotation_text(
    image: np.ndarray,
    gt_rot: torch.Tensor,
    pred_rot: torch.Tensor,
    position: tuple = (340, 20),
    font_scale: float = 0.45,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws ground truth and predicted rotation vectors on an image.

    Args:
        image (np.ndarray): Image of shape (H, W, 3) in uint8 format.
        gt_rot (torch.Tensor): Rotation vector, shape (3,) or (6,) (dual arm).
        pred_rot (torch.Tensor): Same shape as gt_rot.
        position (tuple): Top-left corner (x, y) for drawing text.
        font_scale (float): Font size.
        color (tuple): Text color (B, G, R).
        thickness (int): Line thickness.

    Returns:
        image (np.ndarray): Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    image = np.ascontiguousarray(image.copy())

    if gt_rot.shape[-1] == 3:
        image = cv2.putText(image, f"GT rot:    [{fmt(gt_rot)}]", (x, y), font, font_scale, color, thickness)
        image = cv2.putText(image, f"Pred rot:  [{fmt(pred_rot)}]", (x, y + 20), font, font_scale, color, thickness)
    elif gt_rot.shape[-1] == 6:
        image = cv2.putText(image, f"L GT rot:  [{fmt(gt_rot[0:3])}]", (x, y), font, font_scale, color, thickness)
        image = cv2.putText(image, f"L Pred rot:[{fmt(pred_rot[0:3])}]", (x, y + 20), font, font_scale, color, thickness)
        image = cv2.putText(image, f"R GT rot:  [{fmt(gt_rot[3:6])}]", (x, y + 40), font, font_scale, color, thickness)
        image = cv2.putText(image, f"R Pred rot:[{fmt(pred_rot[3:6])}]", (x, y + 60), font, font_scale, color, thickness)
    else:
        raise ValueError(f"Unsupported rotation shape: {gt_rot.shape}")

    return image

def draw_actions(im, type, color, actions, extrinsics, intrinsics, arm="both"):
    """
    args:
        im: (H, W, C)
        type: "joints" or "xyz"
        color: ex) "Purples", "Blues", "Greens"
        actions: (N, 6) or (N, 3) if type is "xyz" or (N, 7) or (N, 14) if type is "joints"
        extrinsics: dict with keys "left" and "right" with values (4, 4)
        intrinsics: (3, 4)
        arm: "both", "left", "right"
    returns
        im: (H, W, C)
    """
    aloha_fk = AlohaFK()
    if type == "joints": 
        if arm == "both":
            right_actions = aloha_fk.fk(actions[:, 7:13])
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, extrinsics["right"])
            left_actions = aloha_fk.fk(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, extrinsics["left"])
            actions_drawable = np.concatenate((left_actions_drawable, right_actions_drawable), axis=0)
        elif arm == "right":
            right_actions = aloha_fk.fk(actions[:, :6])
            right_actions_drawable = ee_pose_to_cam_frame(right_actions, extrinsics["right"])
            actions_drawable = right_actions_drawable
        elif arm == "left":
            left_actions = aloha_fk.fk(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(left_actions, extrinsics["left"])
            actions_drawable = left_actions_drawable
    else:
        actions = actions.reshape(-1, 3)
        actions_drawable = actions
    
    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, intrinsics)
    im = draw_dot_on_frame(
        im, actions_drawable, show=False, palette=color
    )

    return im


def is_key(x):
    return hasattr(x, "keys") and callable(x.keys)


def is_listy(x):
    return isinstance(x, list)

def nds_pq(file_path):
    """
    Open a .parquet file and explore its structure, including nested datasets.
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        print(f"File Schema:\n{parquet_file.schema}\n")

        df = pd.read_parquet(file_path)

        print(f"Headers (Columns): {list(df.columns)}")
        print(f"Shape (Rows, Columns): {df.shape}")

        nested_columns = []
        for column in df.columns:
            # Check for nested data
            if isinstance(df[column].iloc[0], (dict, list)):
                nested_columns.append(column)

        if nested_columns:
            print(f"Nested Headers: {nested_columns}")
        else:
            print("No nested headers found.")
    except Exception as e:
        print(f"An error occurred: {e}")


nested_ds_pq = nds_pq
nds_parquet = nds_pq
nested_ds_parquet = nds_pq

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def nds(nested_ds, tab_level=0):
    """
    Print the structure of a nested dataset.
    nested_ds: a series of nested dictionaries and iterables.  If a dictionary, print the key and recurse on the value.  If a list, print the length of the list and recurse on just the first index.  For other types, just print the shape.
    """
    # print('--' * tab_level, end='')
    if is_key(nested_ds):
        print("dict with keys: ", nested_ds.keys())
    elif is_listy(nested_ds):
        print("list of len: ", len(nested_ds))
    elif nested_ds is None:
        print("None")
    elif isinstance(nested_ds, Number):
        print("Number: ", nested_ds)
    else:
        # print('\t' * (tab_level), end='')
        print(nested_ds.shape)

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print("\t" * (tab_level), end="")
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif isinstance(nested_ds, list):
        print("\t" * tab_level, end="")
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level + 1)


def ee_pose_to_cam_frame(ee_pose_base, T_cam_base):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)

    returns ee_pose_cam: (N, 3)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T
    return ee_pose_grip_cam.T[:, :3]

def ee_orientation_to_cam_frame(ee_orientation_base, T_cam_base):
    """
    ee_orientation_base: (N, 3, 3) rotation matrices representing orientations in the base frame.
    T_cam_base: (4, 4) transformation matrix from base to camera.
    returns ee_orientation_cam: (N, 3, 3) orientations in the camera frame, and Euler Angles (yaw, pitch, roll) - (N, 3)
    """
    T_base_cam = np.linalg.inv(T_cam_base)  # Inverse transformation
    # Transform orientations
    R_cam_base = T_base_cam[:3, :3]  # Extract rotation matrix from transformation
    ee_orientation_cam = np.array([R_cam_base @ R for R in ee_orientation_base.cpu().numpy()])  # Loop over each orientation
    ## get yaw, pitch, roll
    batched_ypr = batched_rotation_matrices_to_euler_angles(torch.tensor(ee_orientation_cam))
    return ee_orientation_cam, batched_ypr

def batched_rotation_matrices_to_euler_angles(batch_R):
    """
    Convert batched rotation matrices to Euler angles (ZYX order).
    Parameters:
        batch_R (torch.Tensor): A batched tensor of shape [batch_size, 3, 3]
    Returns:
        torch.Tensor: A tensor of Euler angles of shape [batch_size, 3] (yaw, pitch, roll)
    """
    # Reshape the tensor to merge batch and sequence dimensions for processing
    batch_size, _, _ = batch_R.shape
    is_torch_tensor = isinstance(batch_R, torch.Tensor)
    if is_torch_tensor:
        # PyTorch reshape
        reshaped_R = batch_R.view(-1, 3, 3).cpu().numpy()
    else:
        # NumPy reshape
        reshaped_R = batch_R.reshape(-1, 3, 3)
    # reshaped_R = batch_R.view(-1, 3, 3).cpu().numpy()
    # Use scipy's Rotation to convert rotation matrices to Euler angles
    rotation_objects = R.from_matrix(reshaped_R)
    euler_angles = rotation_objects.as_euler('zyx', degrees=False)  # Shape [batch_size * seq_len, 3]
    # Convert back to torch and reshape to original batch dimensions
    euler_angles = torch.tensor(euler_angles, device=batch_R.device)
    euler_angles = euler_angles.view(batch_size, 3)
    return euler_angles

def pose_transform(a_pose, T_a_b):
    """
    a_pose: (N, 3) series of poses in frame a
    T_a_b: (4, 4) transformation matrix from frame a to frame b

    returns b_pose: (N, 3) series of poses in frame b
    """
    orig_shape = list(a_pose.shape)
    a_pose = a_pose.reshape(-1, 3)
    N, _ = a_pose.shape
    a_pose = np.concatenate([a_pose, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = T_a_b @ a_pose.T
    orig_shape[-1] += 1
    return ee_pose_grip_cam.T.reshape(orig_shape)


def ee_pose_to_cam_pixels(ee_pose_base, T_cam_base, intrinsics):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)
    intrinsics: (3, 4)


    returns ee_pose_cam_pixels (N, 2)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T

    px_val = intrinsics @ ee_pose_grip_cam
    px_val = px_val / px_val[2, :]

    return px_val.T
    
def pose_to_transform(pose):
    """
    Convert a 6D pose [x, y, z, yaw, pitch, roll] into a 4x4 homogeneous transform.
    Assumes Euler angles are in radians and follow ZYX (yaw-pitch-roll) order.
    """
    x, y, z, yaw, pitch, roll = pose

    # Compute individual rotation matrices
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch),0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,            0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Combined rotation: note the multiplication order
    R = Rz @ Ry @ Rx

    # Assemble homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T
    
def transform_to_pose(T):
    """
    Convert a 4x4 homogeneous transform back to a 6D pose [x, y, z, yaw, pitch, roll].
    Uses the ZYX (yaw-pitch-roll) convention.
    """
    x, y, z = T[:3, 3]
    R = T[:3, :3]

    # Extract pitch from the (3,1) element of R
    pitch = np.arcsin(-R[2, 0])
    # To avoid numerical issues, check for gimbal lock:
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock: arbitrarily set yaw=0
        yaw = 0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([x, y, z, yaw, pitch, roll])

def cam_frame_to_cam_pixels(ee_pose_cam, intrinsics):
    """
    camera frame 3d coordinates to pixels in camera frame
    ee_pose_cam: (N, 3)
    intrinsics: 3x4 matrix
    """
    N, _ = ee_pose_cam.shape
    ee_pose_cam = np.concatenate([ee_pose_cam, np.ones((N, 1))], axis=1)
    # print("3d pos in cam frame: ", ee_pose_cam)

    # print("intrinsics: ", intrinsics.shape, ee_pose_cam.shape)
    px_val = intrinsics @ ee_pose_cam.T
    px_val = px_val / px_val[2, :]
    # print("2d pos cam frame: ", px_val)

    return px_val.T


def draw_dot_on_frame(frame, pixel_vals, show=True, palette="Purples", dot_size=5):
    """
    frame: (H, W, C) numpy array
    pixel_vals: (N, 2) numpy array of pixel values to draw on frame
    Drawn in light to dark order
    """
    frame = frame.astype(np.uint8).copy()
    if isinstance(pixel_vals, tuple):
        pixel_vals = [pixel_vals]

    # get purples color palette, and color the circles accordingly
    color_palette = plt.get_cmap(palette)
    color_palette = color_palette(np.linspace(0, 1, len(pixel_vals)))
    color_palette = (color_palette[:, :3] * 255).astype(np.uint8)
    color_palette = color_palette.tolist()

    for i, pixel_val in enumerate(pixel_vals):
        try:
            frame = cv2.circle(
                frame,
                (int(pixel_val[0]), int(pixel_val[1])),
                dot_size,
                color_palette[i],
                -1,
            )
        except:
            print("Got bad pixel_val: ", pixel_val)
        if show:
            plt.imshow(frame)
            plt.show()

    return frame


def general_norm(array, min_val, max_val, arr_min=None, arr_max=None):
    if arr_min is None:
        arr_min = array.min()
    if arr_max is None:
        arr_max = array.max()

    return (max_val - min_val) * ((array - arr_min) / (arr_max - arr_min)) + min_val


def general_unnorm(array, orig_min, orig_max, min_val, max_val):
    return ((array - min_val) / (max_val - min_val)) * (orig_max - orig_min) + orig_min


def miniviewer(frame, goal_frame, location="top_right"):
    """
    overlay goal_frame in a corner of frame
    frame: (H, W, C) numpy array
    goal_frame: (H, W, C) numpy array
    location: "top_right", "top_left", "bottom_left", "bottom_right"

    return frame with goal_frame in top right corner (1/4 original size)

    resize using TF
    """
    frame = frame.copy()
    goal_frame = goal_frame.copy()
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if isinstance(goal_frame, np.ndarray):
        goal_frame = torch.from_numpy(goal_frame)

    goal_frame = goal_frame.permute((2, 0, 1))
    frame = frame.permute((2, 0, 1))

    goal_frame = TF.resize(goal_frame, (frame.shape[1] // 4, frame.shape[2] // 4))
    if location == "top_right":
        frame[:, : goal_frame.shape[1], -goal_frame.shape[2] :] = goal_frame
    elif location == "top_left":
        frame[:, : goal_frame.shape[1], : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_left":
        frame[:, -goal_frame.shape[1] :, : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_right":
        frame[:, -goal_frame.shape[1] :, -goal_frame.shape[2] :] = goal_frame
    # frame[:, :goal_frame.shape[1], -goal_frame.shape[2]:] = goal_frame
    return frame.permute((1, 2, 0)).numpy()


def transformation_matrix_to_pose(T):
    R = T[:3, :3]
    p = T[:3, 3]
    rotation_quaternion = Rotation.from_matrix(R).as_quat()
    pose_array = np.concatenate((p, rotation_quaternion))
    return pose_array

def interpolate_arr_euler(v: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Interpolate 6DoF poses (translation + Euler angles in radians) along the time axis.
    If any values in a sequence are >= 1e9 (used as a NaN sentinel), skip interpolation and return filled value.
    """
    assert v.ndim == 3 and v.shape[2] == 6, "Input v must be of shape (B, T, 6)"
    B, T, _ = v.shape

    translations_interp = []
    rotations_interp = []

    new_time = np.linspace(0, 1, seq_length)
    old_time = np.linspace(0, 1, T)

    for i in range(B):
        seq = v[i]
        if np.any(seq >= 1e8):
            # Invalid sequence — fill with dummy values
            translations_interp.append(np.full((seq_length, 3), 1e9))
            rotations_interp.append(np.full((seq_length, 3), 1e9))
            continue

        trans_seq = seq[:, :3]
        rot_seq = seq[:, 3:]

        # Avoid discontinuities in angle interpolation
        rot_seq_unwrapped = np.unwrap(rot_seq, axis=0)

        trans_interp_func = scipy.interpolate.interp1d(old_time, trans_seq, axis=0, kind='linear')
        rot_interp_func = scipy.interpolate.interp1d(old_time, rot_seq_unwrapped, axis=0, kind='linear')

        trans_interp = trans_interp_func(new_time)
        rot_interp = rot_interp_func(new_time)
        rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

        translations_interp.append(trans_interp)
        rotations_interp.append(rot_interp)

    translations_interp = np.array(translations_interp)
    rotations_interp = np.array(rotations_interp)

    return np.concatenate([translations_interp, rotations_interp], axis=-1)

class AlohaFK:
    def __init__(self):
        urdf_path = os.path.join(
            os.path.dirname(egomimic.__file__), "resources/model.urdf"
        )
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf_path).read(), "vx300s/ee_gripper_link"
        )

    def fk(self, qpos):
        if isinstance(qpos, np.ndarray):
            qpos = torch.from_numpy(qpos)

        return self.chain.forward_kinematics(qpos, end_only=True).get_matrix()[:, :3, 3]


def robo_to_aria_imstyle(im):
    im = TVTF.adjust_hue(im, -0.05)
    im = TVTF.adjust_saturation(im, 1.2)
    im = apply_vignette(im, exponent=1)

    return im


def create_vignette_mask(height, width, exponent=2):
    """
    Create a vignette mask with the given height and width.
    The exponent controls the strength of the vignette effect.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
    )
    radius = torch.sqrt(x**2 + y**2) / 2
    mask = 1 - torch.pow(radius, exponent)
    mask = torch.clamp(mask, 0, 1)
    return mask


def apply_vignette(image_tensor, exponent=2):
    """
    Apply a vignette effect to a batch of image tensors.
    """
    N, C, H, W = image_tensor.shape
    vignette_mask = create_vignette_mask(H, W, exponent)
    vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dimensions
    vignette_mask = vignette_mask.expand(
        N, C, H, W
    )  # Expand to match the batch of images
    vignette_mask = vignette_mask.to(image_tensor.device)
    return image_tensor * vignette_mask

def add_extra_train_splits(data, split_percentages):
    """
    data: hdf5 file in robomimic format
    split_percentages: list of percentages for each split, e.g. [0.7, 0.1, 0.2]
    add key "mask/train_{split_name}" which subsamples "mask/train" by split_percentages
    """
    N = len(data["mask/train"][:])
    random_order = np.random.permutation(N)
    mask = data["mask/train"][:]
    splits = []
    for split in split_percentages:
        # data[f"mask/train_{split_percentages:.2f}"] = random_order[:int(N*split)]
        sorted_order = np.sort(random_order[:int(N*split)])
        print(sorted_order)
        splits.append(sorted_order)
        print(mask[sorted_order])
        data[f"mask/train_{int(split*100)}%"] = mask[sorted_order]
    
    for i in range(4):
        print(i)
        assert set(splits[i]).issubset(set(splits[i+1]))

def interpolate_arr(v, seq_length):
    """
    v: (B, T, D)
    seq_length: int
    """
    assert len(v.shape) == 3
    if v.shape[1] == seq_length:
        return
    
    interpolated = []
    for i in range(v.shape[0]):
        index = v[i]

        interp = scipy.interpolate.interp1d(
            np.linspace(0, 1, index.shape[0]), index, axis=0
        )
        interpolated.append(interp(np.linspace(0, 1, seq_length)))

    return np.array(interpolated)

def interpolate_keys(obs, keys, seq_length):
    """
    obs: dict with values of shape (T, D)
    keys: list of keys to interpolate
    seq_length: int changes shape (T, D) to (seq_length, D)
    """
    for k in keys:
        v = obs[k]
        L = v.shape[0]
        if L == seq_length:
            continue

        if k == "pad_mask":
            # interpolate it by simply copying each index (seq_length / seq_length_to_load) times
            obs[k] = np.repeat(v, (seq_length // L), axis=0)
        elif k != "pad_mask":
            interp = scipy.interpolate.interp1d(
                np.linspace(0, 1, L), v, axis=0
            )
            try:
                obs[k] = interp(np.linspace(0, 1, seq_length))
            except:
                raise ValueError(f"Interpolation failed for key: {k} with shape{k.shape}")


def ypr_to_matrix(ypr):
    """Convert yaw-pitch-roll (zyx) to rotation matrix. ypr: (..., 3) → (..., 3, 3)"""
    yaw, pitch, roll = ypr.unbind(-1)

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)

    # Build rotation matrix (Z-Y-X / yaw-pitch-roll)
    Rz = torch.stack([
        torch.stack([cy, -sy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([sy,  cy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1)
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([cp, torch.zeros_like(pitch), sp], dim=-1),
        torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)], dim=-1),
        torch.stack([-sp, torch.zeros_like(pitch), cp], dim=-1)
    ], dim=-2)

    Rx = torch.stack([
        torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)], dim=-1),
        torch.stack([torch.zeros_like(roll), cr, -sr], dim=-1),
        torch.stack([torch.zeros_like(roll), sr,  cr], dim=-1)
    ], dim=-2)

    return Rz @ Ry @ Rx


def matrix_to_ypr(R):
    """Convert rotation matrix to yaw-pitch-roll (zyx). R: (..., 3, 3) → (..., 3)"""
    # Safe conversion for all angles
    pitch = torch.asin(-R[..., 2, 0])
    cos_pitch = torch.cos(pitch)

    yaw = torch.atan2(R[..., 1, 0] / cos_pitch, R[..., 0, 0] / cos_pitch)
    roll = torch.atan2(R[..., 2, 1] / cos_pitch, R[..., 2, 2] / cos_pitch)

    return torch.stack([yaw, pitch, roll], dim=-1)


def convert_to_cam_frame(pose_base: torch.Tensor, T_cam_base: torch.Tensor):
    """
    pose_base: (B, T, 6) — [x, y, z, yaw, pitch, roll] in base frame
    T_cam_base: (4, 4) — camera-to-base transform (same for all B)

    Returns:
    pose_cam: (B, T, 6) — in camera frame
    """
    B, T, _ = pose_base.shape
    device = pose_base.device

    # Positions
    pos_base = pose_base[..., :3]  # (B, T, 3)
    ypr_base = pose_base[..., 3:]  # (B, T, 3)

    # Homogeneous transform
    ones = torch.ones((B, T, 1), device=device, dtype=pose_base.dtype)
    pos_homo = torch.cat([pos_base, ones], dim=-1)  # (B, T, 4)

    T_base_cam = torch.linalg.inv(T_cam_base).to(device)  # (4, 4)
    pos_cam_homo = torch.einsum("ij,btj->bti", T_base_cam.float(), pos_homo.float())  # (B, T, 4)
    pos_cam = pos_cam_homo[..., :3]  # (B, T, 3)

    # Orientation
    R_base = ypr_to_matrix(ypr_base)  # (B, T, 3, 3)
    R_cam_base = T_base_cam[:3, :3]   # (3, 3)

    R_cam = torch.einsum("ij,btjk->btik", R_cam_base.float(), R_base.float())  # (B, T, 3, 3)
    ypr_cam = matrix_to_ypr(R_cam)  # (B, T, 3)

    return torch.cat([pos_cam, ypr_cam], dim=-1)  # (B, T, 6)

def transform_matrix_to_pose(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert a (B, T, 4, 4) homogeneous transform matrix to (B, T, 6) [xyz + ypr]

    Args:
        mat: (B, T, 4, 4) torch tensor — full transform matrix

    Returns:
        pose: (B, T, 6) — xyz + yaw-pitch-roll
    """
    assert mat.shape[-2:] == (4, 4), "Input must be (B, T, 4, 4)"
    B, T = mat.shape[:2]
    device = mat.device

    # Translation part: (B, T, 3)
    xyz = mat[..., :3, 3]

    # Rotation matrix: (B, T, 3, 3)
    R = mat[..., :3, :3]

    # Convert to yaw-pitch-roll: (B, T, 3)
    ypr = matrix_to_ypr(R)

    # Return pose: (B, T, 6)
    return torch.cat([xyz, ypr], dim=-1)

