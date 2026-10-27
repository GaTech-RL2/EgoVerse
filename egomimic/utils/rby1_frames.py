"""RB-Y1 body frames for multi-frame point-cloud views (2026-10-17 mixture-of-frames study).

A point cloud in the robot camera (glass) frame is re-expressed in a body frame F of the SAME row with T_F_glass, from FK of
the row's 22-D proprio [torso 6 | r_arm 7 | l_arm 7 | head 2] on model_v1.3_xhand_act.xml (the dataset builder's and the
robot client's FK). The training loader (RLDBDataset frame_views) and the policy server build identical views from the same
joints, so nothing extra is stored or sent.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/coc/flash7/czhang883/Documents/SEW-Geometric-Teleop-hw")

FRAMES = ("base", "eefL", "eefR")


class FrameFK:
    def __init__(self):
        from projects.rby1_teleop.utils.rby1_eef_frames import RBY1EefFK

        self.fk = RBY1EefFK()

    def T_frame_glass(self, q22, frames=FRAMES):
        """{frame: 4x4 T_F_glass} for one 22-D joint vector"""
        q = np.asarray(q22, np.float64).reshape(22)
        T_base_glass = self.fk.eef_geometry(q)["T_base_glass"]
        f = self.fk.frames(q)
        T = {"base": T_base_glass, "eefL": np.linalg.inv(f["T_base_eefL"]) @ T_base_glass,
             "eefR": np.linalg.inv(f["T_base_eefR"]) @ T_base_glass}
        return {k: T[k] for k in frames}

    def batch(self, Q, frames=FRAMES):
        """(N, 22) joints -> {frame: (N, 4, 4) float32}"""
        Q = np.asarray(Q, np.float64).reshape(-1, 22)
        out = {k: np.zeros((len(Q), 4, 4), np.float32) for k in frames}
        for i, q in enumerate(Q):
            for k, T in self.T_frame_glass(q, frames).items():
                out[k][i] = T
        return out


def apply_frame(cloud, T):
    """cloud (..., N*6) or (..., N, 6) xyzrgb in the glass frame; T (..., 4, 4) T_F_glass -> the same layout in frame F
    (xyz rotated + translated, rgb unchanged). Works batched on torch tensors (any device) and on numpy arrays."""
    if isinstance(cloud, np.ndarray):
        cloud, T, back = torch.from_numpy(np.asarray(cloud)), torch.from_numpy(np.asarray(T)), True
    else:
        back = False
    shape = cloud.shape
    P = cloud.reshape(*shape[:-1], -1, 6) if shape[-1] != 6 else cloud
    R, t = T[..., :3, :3].to(P.dtype), T[..., :3, 3].to(P.dtype)
    xyz = torch.einsum("...ij,...nj->...ni", R, P[..., :3]) + t[..., None, :]
    out = torch.cat([xyz, P[..., 3:]], dim=-1).reshape(shape)
    return out.numpy() if back else out
