"""Virtual human-camera view of the robot's own point cloud (2026-10-25).

A GR1-view policy was trained on clouds seen from the demonstrator's camera, which sits ~18 cm higher and ~11 cm further forward
than the robot's camera. At deploy we only have the robot camera, so: (1) predict where the demonstrator's camera would be from the
robot's joints (a ridge model fitted on the paired v12 data, runs/lowdim/virtual_view/fit_virtual_cam.py), then (2) reproject the
robot's cloud into that virtual camera and keep only the points it can see: inside its image, and not hidden behind a nearer
surface (a z-buffer with a small splat). Unobserved surfaces cannot be added; points the virtual camera could not see are dropped.
All poses are rect camera -> robot glass frame; points are in the robot glass frame."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def load_model(path):
    z = np.load(path)
    m = {k: z[k] for k in ("mu", "sd", "W", "R_DR")}
    m["zclip"] = float(z["zclip"]) if "zclip" in z.files else np.inf
    for k in ("t_lo", "t_hi", "r_max"):   # the offset range seen in the paired data (older models: no clamp)
        m[k] = z[k] if k in z.files else None
    return m


def predict_virtual_T(q22, model):
    """4x4 virtual (demonstrator) rect camera -> robot glass, from the robot's 22 joints (torso 6, right arm 7, left arm 7, head 2)."""
    z = np.clip((np.asarray(q22, np.float64) - model["mu"]) / model["sd"], -model["zclip"], model["zclip"])  # bounded extrapolation
    y = np.concatenate([z, z ** 2, [1.0]]) @ model["W"]
    if model["t_lo"] is not None:  # clamp to the observed range: the robot can be in states the demonstrations never had
        y[:3] = np.clip(y[:3], model["t_lo"], model["t_hi"])
        n = np.linalg.norm(y[3:6])
        if n > model["r_max"]:
            y[3:6] *= model["r_max"] / n
    R_DR = model["R_DR"]
    T = np.eye(4)
    T[:3, :3] = R_DR @ Rot.from_rotvec(y[3:6]).as_matrix()
    T[:3, 3] = R_DR @ y[:3]
    return T


def visible_from(P_glass, T_v, KH=134.45979925, C0=112.0, size=224, splat=1, tol0=0.02, tol_per_m=0.02, occlusion=True):
    """Mask of points (robot glass frame) the camera T_v (rect -> glass) can see: in front, inside its size x size image, and (with
    occlusion) within tol0 + tol_per_m * depth of the nearest point splatted to the same pixel."""
    X = (np.asarray(P_glass, np.float64) - T_v[:3, 3]) @ T_v[:3, :3]
    z = X[:, 2]
    ok = z > 0.05
    u = np.full(len(z), -1.0)
    v = np.full(len(z), -1.0)
    u[ok] = KH * X[ok, 0] / z[ok] + C0
    v[ok] = KH * X[ok, 1] / z[ok] + C0
    ok &= (u >= 0) & (u < size) & (v >= 0) & (v < size)
    if not occlusion or not ok.any():
        return ok
    ui, vi, zi = u[ok].astype(np.int64), v[ok].astype(np.int64), z[ok]
    zbuf = np.full((size, size), np.inf)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = np.clip(ui + du, 0, size - 1), np.clip(vi + dv, 0, size - 1)
            np.minimum.at(zbuf, (vv, uu), zi)
    vis = np.zeros(len(z), bool)
    vis[np.flatnonzero(ok)] = zi <= zbuf[vi, ui] + tol0 + tol_per_m * zi
    return vis
