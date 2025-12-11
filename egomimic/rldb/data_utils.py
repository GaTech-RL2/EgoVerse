from collections.abc import Sequence
import torch.nn.functional as F
import torch

def _slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternion sequences.

    q0, q1: (..., 4)
    t:      (..., 1) in [0, 1]
    """
    # Normalize input quaternions
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # (..., 1)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)

    theta = torch.acos(dot)        # (..., 1)
    sin_theta = torch.sin(theta)   # (..., 1)

    small = sin_theta.abs() < 1e-6

    # Standard SLERP weights
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    # Replace with LERP weights when angle is tiny
    w0 = torch.where(small, 1.0 - t, w0)
    w1 = torch.where(small, t, w1)

    out = w0 * q0 + w1 * q1
    return F.normalize(out, dim=-1)

def _ypr_to_quat(ypr):
    """
    Convert yaw-pitch-roll (radians) to quaternion.
    ypr: (..., 3) with [yaw, pitch, roll]
    Returns: (..., 4) quaternion [w, x, y, z]
    """

    yaw, pitch, roll = ypr.unbind(dim=-1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    # Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(quat, dim=-1)

def _quat_to_ypr(quat):
    """
    Convert quaternion [w, x, y, z] to yaw-pitch-roll (radians).
    quat: (..., 4)
    Returns: (..., 3) [yaw, pitch, roll]
    """

    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp for numerical stability
    sinp_clamped = sinp.clamp(-1.0, 1.0)
    pitch = torch.asin(sinp_clamped)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([yaw, pitch, roll], dim=-1)