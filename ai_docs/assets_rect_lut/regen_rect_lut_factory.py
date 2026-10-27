"""
Regenerate robot_rect224_lut.npz from the ROBOT device's FACTORY calibration.

Why: the current LUT uses an empirically-solved pinhole+2-radial model. It is
accurate (<=1.6 px) to ~40 deg off-axis but extrapolates badly past 45 deg --
+6.8 px @48, +14.8 @50, +52 @54 -- so it declares pixels "off-sensor" at 48.2 deg
instead of the true 50.1 deg. That is the extra ~5.9% black band at the bottom
of the live rectified frame.

The recipe below is the one the TRAINING data was built with
(build_human_rgbd.sbatch): rect ray -> 3D -> camera-rgb via the factory FISHEYE624
model -> raw sensor bounds. Validated on the human device: predicted invalid
region vs actual training frames IoU 0.944-0.966 (rows 206-223 vs 207-223).

Run ON THE ROBOT LAPTOP (needs the device calibration).
"""
import numpy as np, cv2, json
from projectaria_tools.core import data_provider

OUT_SZ   = 224
F512     = 307.336684          # robot rect focal @512 (device constant)
NOM_D    = 1.5                 # nominal depth for the parallax term (m)
SRC_W, SRC_H = 2016, 1512      # camera-rgb full readout
SQUEEZE  = 640 / 1512.0        # R_640x640 y-scale; LUT target = unsqueezed frame

# --- inputs -----------------------------------------------------------------
# Either a live VRS/stream provider, or the saved device-calibration JSON.
cal = data_provider.create_vrs_data_provider("<robot_recording>.vrs").get_device_calibration()
rgbc = cal.get_camera_calib("camera-rgb")
# R_DEV_RECT: device -> rect (pure rotation), same matrix already in depth_store attrs
R_DEV_RECT = np.array(json.load(open("aria_transforms.json"))["R_DEV_RECT"], float).reshape(3, 3)

# --- rect grid rays ---------------------------------------------------------
fr = F512 * OUT_SZ / 512.0
cr = 256.0 * OUT_SZ / 512.0
u, v = np.meshgrid(np.arange(OUT_SZ), np.arange(OUT_SZ))
rays_rect = np.stack([(u - cr) / fr, (v - cr) / fr, np.ones_like(u, float)], -1)

# rect -> device -> camera-rgb (full extrinsic, translation INCLUDED: the training
# warp is parallax-correct; ignoring t costs ~1 px @1.5 m, ~3 px @0.5 m)
P_dev = (rays_rect.reshape(-1, 3) * NOM_D) @ R_DEV_RECT          # rect -> device
T_dc  = np.asarray(rgbc.get_transform_device_camera().inverse().to_matrix())
P_cam = P_dev @ T_dc[:3, :3].T + T_dc[:3, 3]

# --- project through the FACTORY fisheye model ------------------------------
map_x = np.full(OUT_SZ * OUT_SZ, -1.0, np.float32)
map_y = np.full(OUT_SZ * OUT_SZ, -1.0, np.float32)
for i, q in enumerate(P_cam):
    if q[2] <= 0.05:
        continue
    uv = rgbc.project(q)
    if uv is None:
        continue
    if 0 <= uv[0] < SRC_W and 0 <= uv[1] < SRC_H:
        map_x[i] = uv[0] * SQUEEZE       # into the unsqueezed 853x640 frame
        map_y[i] = uv[1] * SQUEEZE
map_x = map_x.reshape(OUT_SZ, OUT_SZ); map_y = map_y.reshape(OUT_SZ, OUT_SZ)

invalid = (map_x < 0) | (map_y < 0)
print(f"invalid region: {invalid.mean()*100:.2f}%   (old LUT 9.46%, training arc ~4.1%)")
rows = np.where(invalid.any(1))[0]
print(f"invalid rows {rows.min()}-{rows.max()}   (training arc rows 206-223)")

# --- gate against the OLD LUT where it is trustworthy (rows 0-180) ----------
old = np.load("robot_rect224_lut.npz")
m = (~invalid) & (old["map_x"] >= 0)
m[181:, :] = False
d = np.hypot(map_x[m] - old["map_x"][m], map_y[m] - old["map_y"][m])
print(f"agreement with old LUT, rows 0-180: median {np.median(d):.2f} px, p95 {np.percentile(d,95):.2f} px")
assert np.median(d) < 4.0, "rotation disagreement -- check R_DEV_RECT / extrinsic convention"
assert invalid.mean() < 0.06, "invalid region still too large -- model or squeeze convention wrong"

np.savez("robot_rect224_lut_factory.npz", map_x=map_x, map_y=map_y,
         note="cv2.remap(unsqueezed_rgb_BGR, map_x, map_y, INTER_LINEAR, borderValue=0) -> 224 rect; then [...,::-1]",
         solved_from=f"FACTORY camera-rgb FISHEYE624 {SRC_W}x{SRC_H} + T_device_camera, nominal depth {NOM_D} m")
print("wrote robot_rect224_lut_factory.npz")
