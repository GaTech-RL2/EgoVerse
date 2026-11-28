"""RGB-D shard: COLOUR image_rect + depth in the depth-native rect frame. Identical logic to
the chuye_basket build (validated colour<->depth alignment)."""
import sys, os, glob
import numpy as np, cv2, h5py
sys.path.insert(0, "/coc/flash7/czhang883/Documents/aria_gen2_scripts")
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.image import InterpolationMethod
from src.utils.stereo_utils import create_scanline_rectified_cameras, fisheye_to_linear_calib

SRC  = os.environ["TOY_SRC"]
MAN  = os.environ["TOY_MAN"]
VRSD = os.environ["TOY_VRSD"]
DEPD = os.environ["DEPD"]
OUTD = os.environ["OUTD"]
SH, NS = int(os.environ["SHARD"]), int(os.environ["NSHARD"])
W = H = 512; RW = RH = 1408; OUT_SZ = 224
CARRY = ["obs/hand_left_qpos", "obs/hand_right_qpos", "obs/robot0_joint_pos",
         "actions/joint", "actions/joint_arm", "actions/joint_arm_hand",
         "actions/joint_arm_head_torso_base"]

t2v = {}
for ln in open(MAN):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 3: t2v[p[0]] = p[2][:-4]

src = h5py.File(SRC, "r")
demos = sorted(src["data"].keys(), key=lambda s: int(s.split("_")[1]))
mine = demos[SH::NS]
print(f"shard {SH}/{NS}: {len(mine)} demos", flush=True)

dst = h5py.File(f"{OUTD}/shard_{SH}.hdf5", "w")
g = dst.create_group("data")
RGBSID = StreamId("214-1")
worst = 0.0

for dn in mine:
    sd = src["data"][dn]
    tag = sd.attrs["source_tag"]; rec = t2v[tag]
    ts = sd["obs/aria_ts"][:].astype(np.int64)
    N = len(ts)

    dp = data_provider.create_vrs_data_provider(f"{VRSD}/{rec}.vrs")
    cal = dp.get_device_calibration()
    lc, rgbc = cal.get_camera_calib("slam-front-left"), cal.get_camera_calib("camera-rgb")
    rc = cal.get_camera_calib("slam-front-right")
    lin = fisheye_to_linear_calib(lc, 1.25, W, H)
    Rl, _ = create_scanline_rectified_cameras(lc.get_transform_device_camera().inverse(),
                                              rc.get_transform_device_camera().inverse())
    p = lin.get_projection_params()
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], float)
    rgb_lin = fisheye_to_linear_calib(rgbc, 0.35, RW, RH)
    rp = rgb_lin.get_projection_params()
    Kr = np.array([[rp[0], 0, rp[2]], [0, rp[1], rp[3]], [0, 0, 1]], float)
    M = np.asarray((rgbc.get_transform_device_camera().inverse()
                    @ lc.get_transform_device_camera()).to_matrix())
    R_rl, t_rl = M[:3, :3], M[:3, 3:4]
    Rl_m = np.asarray(Rl.to_matrix())
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.linalg.inv(K) @ np.stack([uu.ravel(), vv.ravel(), np.ones(W * H)])

    dts = np.array(sorted(int(f[:-4]) for f in os.listdir(f"{DEPD}/{rec}") if f.endswith(".npy")))
    nrgb = dp.get_num_data(RGBSID)
    rts = np.array([dp.get_image_data_by_index(RGBSID, i)[1].capture_timestamp_ns for i in range(nrgb)])

    def nearest(arr, t):
        i = int(np.clip(np.searchsorted(arr, t), 1, len(arr) - 1))
        return i if abs(arr[i] - t) < abs(arr[i - 1] - t) else i - 1

    imgs = np.zeros((N, OUT_SZ, OUT_SZ, 3), np.uint8)
    deps = np.zeros((N, OUT_SZ, OUT_SZ), np.float32)
    stale = np.zeros(N, np.float32)
    ck_d = ck_r = None; cur_d = cur_und = None; ck_pair = None; cur_img = cur_dep = None

    for r in range(N):
        di, ri = nearest(dts, ts[r]), nearest(rts, ts[r])
        stale[r] = abs(dts[di] - ts[r]) / 1e6
        if (di, ri) != ck_pair:
            if di != ck_d:
                cur_d = np.load(f"{DEPD}/{rec}/{dts[di]}.npy").astype(np.float64); ck_d = di
                cur_dep = cv2.resize(cur_d.astype(np.float32), (OUT_SZ, OUT_SZ), interpolation=cv2.INTER_NEAREST)
            if ri != ck_r:
                raw = dp.get_image_data_by_index(RGBSID, ri)[0].to_numpy_array()
                cur_und = np.asarray(calibration.distort_by_calibration(
                    raw, rgb_lin, rgbc, InterpolationMethod.BILINEAR)); ck_r = ri
            valid = (cur_d > 0.05).ravel()
            P = rays * cur_d.ravel()[None, :]
            Prgb = R_rl @ (Rl_m @ P) + t_rl
            z = Prgb[2]
            u = Kr[0, 0] * Prgb[0] / np.where(z > 1e-6, z, 1) + Kr[0, 2]
            v = Kr[1, 1] * Prgb[1] / np.where(z > 1e-6, z, 1) + Kr[1, 2]
            ok = valid & (z > 0.05) & (u >= 0) & (u < RW) & (v >= 0) & (v < RH)
            col = np.zeros((H * W, 3), np.uint8)
            col[ok] = cur_und[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
            cur_img = cv2.resize(col.reshape(H, W, 3), (OUT_SZ, OUT_SZ), interpolation=cv2.INTER_AREA)
            ck_pair = (di, ri)
        imgs[r] = cur_img; deps[r] = cur_dep

    do = g.create_group(dn)
    do.attrs["num_samples"] = N
    do.attrs["source_tag"] = tag
    do.attrs["rec"] = rec
    do.attrs["depth_staleness_ms_max"] = float(stale.max())
    do.create_dataset("obs/aria_image", data=imgs, compression="gzip", compression_opts=4)
    do.create_dataset("obs/aria_depth", data=deps.reshape(N, -1), compression="gzip", compression_opts=4)
    for k in CARRY:
        if k in sd: do.create_dataset(k, data=sd[k][:], compression="gzip", compression_opts=4)
    worst = max(worst, float(stale.max()))
    print(f"  {dn} ({tag}/{rec}): N={N} stale_max={stale.max():.1f}ms "
          f"depth_valid={(deps > 0.05).mean() * 100:.0f}%", flush=True)
    del dp

dst.attrs["frame"] = "slam_rect_v1 (rectified left SLAM, 1.25 focal scale, 512->224)"
dst.attrs["image"] = "COLOUR: camera-rgb undistorted to linear then warped into the rect frame via /depth"
dst.attrs["depth"] = "metres float32, 224 NEAREST, flattened; 0 = invalid (outside stereo overlap)"
dst.close()
print(f"SHARD_{SH}_DONE demos={len(mine)} worst_staleness_ms={worst:.1f}", flush=True)