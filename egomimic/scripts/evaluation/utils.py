import torch

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

def save_image(image, path):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def transformation_matrix_to_pose(T):
    R = T[:3, :3]
    p = T[:3, 3]
    rotation_quaternion = Rotation.from_matrix(R).as_quat()
    pose_array = np.concatenate((p, rotation_quaternion))
    return pose_array

def batched_euler_to_rot_matrix(actions_ypr: torch.Tensor) -> torch.Tensor:
    """
    Convert batched Euler angles (yaw-pitch-roll) to rotation matrices.

    Args:
        actions_ypr (torch.Tensor): shape (B, 3) in radians as (yaw, pitch, roll).

    Returns:
        torch.Tensor: shape (B, 3, 3) rotation matrices, where each matrix is
                    R = Rz(yaw) · Ry(pitch) · Rx(roll).
    """
    yaw, pitch, roll = actions_ypr.unbind(-1)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # First column
    r00 = cy * cp
    r10 = sy * cp
    r20 = -sp

    # Second column
    r01 = cy * sp * sr - sy * cr
    r11 = sy * sp * sr + cy * cr
    r21 = cp * sr

    # Third column
    r02 = cy * sp * cr + sy * sr
    r12 = sy * sp * cr - cy * sr
    r22 = cp * cr

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    if row0.ndim == 2:              # batched case (B,3)
        rot_mats = torch.stack([row0, row1, row2], dim=-2)
    else:                           # single vector case (3,)
        rot_mats = torch.stack([row0, row1, row2], dim=0)
    return rot_mats

class TemporalAgg:
    def __init__(self):
        self.recent_actions = []
    
    def add_action(self, action):
        """
            actions: (100, 7) tensor
        """
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            del self.recent_actions[0]

    def smoothed_action(self):
        """
            returns smooth action (100, 7)
        """
        mask = []
        count = 0

        shifted_actions = []
        # breakpoint()

        for ac in self.recent_actions[::-1]:
            basic_mask = np.zeros(100)
            basic_mask[:100-count] = 1
            mask.append(basic_mask)
            shifted_ac = ac[count:]
            shifted_ac = np.concatenate([shifted_ac, np.zeros((count, 7))], axis=0)
            shifted_actions.append(shifted_ac)
            count += 25

        mask = mask[::-1]
        mask = ~(np.array(mask).astype(bool))
        recent_actions = shifted_actions[::-1]
        recent_actions = np.array(recent_actions)
        # breakpoint()
        mask = np.repeat(mask[:, :, None], 7, axis=2)
        smoothed_action = np.ma.array(recent_actions, mask=mask).mean(axis=0)

        # PLOT_JOINT = 0
        # for i in range(recent_actions.shape[0]):
        #     plt.plot(recent_actions[i, :, PLOT_JOINT], label=f"index{i}")
        # plt.plot(smoothed_action[:, PLOT_JOINT], label="smooth")
        # plt.legend()
        # plt.savefig("smoothing.png")
        # plt.close()
        # breakpoint()

        return smoothed_action