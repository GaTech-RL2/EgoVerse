"""Hand-eye calibration of the real_mug rig from AprilTag 36h11 id5 seen by the wrist cam
(eye-in-hand) + Panda FK, then front-camera pose from the same tag. Outputs T_base_front."""
import h5py, glob, os, sys, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
exec(open(os.path.expanduser("~/claude_fk/fk_check.py")).read().split("files = sorted")[0])  # panda_fk
TOOL_OFFSET = np.array([0.1325, -0.0049, 0.0735])

def parse_intr(v):
    K = v[2:11].reshape(3, 3); D = v[11:25].astype(np.float64); return K, D
def inv(T): Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]; return Ti
def T_from(rvec, tvec): T = np.eye(4); T[:3,:3] = cv2.Rodrigues(np.asarray(rvec, float))[0]; T[:3,3] = np.asarray(tvec, float).ravel(); return T

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), cv2.aruco.DetectorParameters())
def detect(img, K, D):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or 5 not in ids.ravel(): return None
    c = corners[list(ids.ravel()).index(5)].reshape(4, 2).astype(np.float32)
    c = cv2.cornerSubPix(gray, c.copy(), (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return cv2.undistortPoints(c.reshape(-1, 1, 2), K, D, P=K).reshape(4, 2)   # undistorted pixel coords

def pnp(c, K, s):
    obj = np.array([[-s/2, s/2, 0], [s/2, s/2, 0], [s/2, -s/2, 0], [-s/2, -s/2, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return T_from(rvec, tvec) if ok else None

files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
sel = files[::10][:12]
W, F = [], []   # wrist obs: (T_base_flange, corners) ; front obs: corners
Kw = Dw = Kf = Df = None
for fn in sel:
    with h5py.File(fn) as f:
        if Kw is None:
            Kw, Dw = parse_intr(f["camera/intrinsics/wrist"][:]); Kf, Df = parse_intr(f["camera/intrinsics/front"][:])
        q = f["arm/current_joint"][:, :7]; T = len(q)
        for i in range(0, T, 5):
            cw = detect(f["camera/wrist"][i], Kw, Dw)
            if cw is not None: W.append((panda_fk(q[i]), cw))
            if i % 25 == 0:
                cf = detect(f["camera/front"][i], Kf, Df)
                if cf is not None: F.append(cf)
print(f"wrist-cam tag observations: {len(W)}  front-cam tag observations: {len(F)}  (from {len(sel)} episodes)")
assert len(W) > 30, "too few wrist detections"
R_g2b = [T[:3,:3] for T, _ in W]; t_g2b = [T[:3,3] for T, _ in W]

def solve(s):
    tc = [pnp(c, Kw, s) for _, c in W]
    R_t2c = [T[:3,:3] for T in tc]; t_t2c = [T[:3,3] for T in tc]
    Rx, tx = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    X = np.eye(4); X[:3,:3] = Rx; X[:3,3] = tx.ravel()                      # T_flange_cam
    tags = np.stack([Tb @ X @ Tc for (Tb, _), Tc in zip(W, tc)])            # T_base_tag per obs
    pos_std = tags[:, :3, 3].std(0); rot_spread = np.degrees(np.linalg.norm(R.from_matrix(tags[:, :3, :3]).as_rotvec() - R.from_matrix(tags[:, :3, :3]).mean().as_rotvec(), axis=1)).mean()
    return X, tags, np.linalg.norm(pos_std), rot_spread

print("tag-size search (consistency of T_base_tag across wrist observations):")
best = None
for s in np.arange(0.02, 0.121, 0.005):
    X, tags, ps, rs = solve(s)
    print(f"  s={s:.3f} m  pos-std {ps*1000:6.1f} mm  rot-spread {rs:5.2f} deg")
    if best is None or ps < best[1]: best = (s, ps, X, tags)
s, ps, X, tags = best
T_base_tag = np.eye(4); T_base_tag[:3,:3] = R.from_matrix(tags[:,:3,:3]).mean().as_matrix(); T_base_tag[:3,3] = tags[:,:3,3].mean(0)
print(f"\nbest tag size {s:.3f} m, T_base_tag pos {T_base_tag[:3,3].round(4)} (std {ps*1000:.1f} mm)")
print("T_flange_cam (wrist cam in flange frame): t =", X[:3,3].round(4), " euler xyz deg =", R.from_matrix(X[:3,:3]).as_euler("xyz", degrees=True).round(1))

# front camera pose in base frame
fronts = np.stack([T_base_tag @ inv(pnp(c, Kf, s)) for c in F])
T_base_front = np.eye(4); T_base_front[:3,:3] = R.from_matrix(fronts[:,:3,:3]).mean().as_matrix(); T_base_front[:3,3] = fronts[:,:3,3].mean(0)
print(f"T_base_front: pos {T_base_front[:3,3].round(4)}  std over {len(F)} obs {(fronts[:,:3,3].std(0)*1000).round(1)} mm")
print("T_base_front euler xyz deg:", R.from_matrix(T_base_front[:3,:3]).as_euler("xyz", degrees=True).round(1))
np.set_printoptions(precision=8, suppress=True)
print("T_base_front =\n", T_base_front); print("K_front =\n", Kf); print("D_front =", Df)
np.savez(os.path.expanduser("~/claude_fk/mug_calib.npz"), T_base_front=T_base_front, T_flange_wristcam=X, T_base_tag=T_base_tag, tag_size=s, K_front=Kf, D_front=Df, K_wrist=Kw, D_wrist=Dw)

# validation: project the FK tool point into the (undistorted) front image on a few frames
T_front_base = inv(T_base_front)
from PIL import Image
rows = []
with h5py.File(files[3]) as f:
    q = f["arm/current_joint"][:, :7]; T = len(q)
    for i in (0, T//4, T//2, 3*T//4, T-1):
        Tf = panda_fk(q[i]); p_base = Tf[:3,3] + Tf[:3,:3] @ TOOL_OFFSET; p_fl = Tf[:3,3]
        img = cv2.undistort(f["camera/front"][i], Kf, Df, None, Kf)
        for p, col in ((p_base, (0,255,0)), (p_fl, (255,0,0))):
            pc = T_front_base[:3,:3] @ p + T_front_base[:3,3]; px = Kf @ (pc / pc[2])
            cv2.circle(img, (int(px[0]), int(px[1])), 6, col, -1)
        # tag corners re-projected from base (should sit on the tag)
        for corner in np.array([[-s/2, s/2, 0], [s/2, s/2, 0], [s/2, -s/2, 0], [-s/2, -s/2, 0]]):
            pb = T_base_tag[:3,:3] @ corner + T_base_tag[:3,3]; pc = T_front_base[:3,:3] @ pb + T_front_base[:3,3]; px = Kf @ (pc/pc[2])
            cv2.circle(img, (int(px[0]), int(px[1])), 3, (255,255,0), -1)
        rows.append(img)
Image.fromarray(np.concatenate(rows, 0)).save(os.path.expanduser("~/claude_fk/mug_proj_check.png")); print("saved mug_proj_check.png (green=tool point, red=flange, yellow=tag corners from base)")
