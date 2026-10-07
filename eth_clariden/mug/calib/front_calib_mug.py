"""Front camera <-> robot base directly from 3D-3D correspondences: yellow mug centroid
(RGB mask + aligned depth, in cam frame) vs FK tool point (base frame) during the carry
phase (mug moving => in hand). No tag, no hand-eye. Kabsch + outlier rejection."""
import h5py, glob, os, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
exec(open(os.path.expanduser("~/claude_fk/fk_check.py")).read().split("files = sorted")[0])
TOOL_OFFSET = np.array([0.1325, -0.0049, 0.0735])
C = np.load(os.path.expanduser("~/claude_fk/mug_calib.npz")); Kf, Df = C["K_front"], C["D_front"]
def inv(T): Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]; return Ti
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); H = (P-pc).T @ (Q-qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1,1,d]) @ U.T; T = np.eye(4); T[:3,:3] = Rm; T[:3,3] = qc - Rm @ pc; return T
def mug_centroid(rgb, dep):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); m = (hsv[...,0] > 15) & (hsv[...,0] < 40) & (hsv[...,1] > 120) & (hsv[...,2] > 90) & (dep > 0.3) & (dep < 1.5)
    if m.sum() < 250: return None
    vv, uu = np.nonzero(m); z = dep[m]; keep = np.abs(z - np.median(z)) < 0.06; uu, vv, z = uu[keep], vv[keep], z[keep]
    pts = np.stack([(uu-Kf[0,2])/Kf[0,0]*z, (vv-Kf[1,2])/Kf[1,1]*z, z], 1); return pts.mean(0), (uu.mean(), vv.mean()), m.sum()
files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
P_base, P_cam, meta = [], [], []
for fn in files[::4][:36]:
    with h5py.File(fn) as f:
        q = f["arm/current_joint"][:, :7]; T = len(q)
        if T < 60: continue
        tool = np.array([panda_fk(qi)[:3,3] + panda_fk(qi)[:3,:3] @ TOOL_OFFSET for qi in q])
        cents = []
        for i in range(0, T, 3):
            r = mug_centroid(f["camera/front"][i], f["depth/front"][i].astype(np.float64)/1000.0); cents.append((i, r))
        first = next((r[0] for _, r in cents if r is not None), None)
        if first is None: continue
        for i, r in cents:
            if r is None: continue
            moved = np.linalg.norm(r[0] - first) > 0.04          # mug displaced >4 cm from its start => being carried
            if moved: P_base.append(tool[i]); P_cam.append(r[0]); meta.append((os.path.basename(fn), i))
P_base, P_cam = np.array(P_base), np.array(P_cam); print(f"carry-phase correspondences: {len(P_base)} from {len(set(m[0] for m in meta))} episodes")
keep = np.ones(len(P_base), bool)
for it in range(6):
    T_cb = kabsch(P_base[keep], P_cam[keep]); res = np.linalg.norm((T_cb[:3,:3] @ P_base.T).T + T_cb[:3,3] - P_cam, axis=1)
    thr = max(0.03, np.percentile(res[keep], 80)); keep = res < thr
    print(f"  iter {it}: inliers {keep.sum()} / {len(res)}  rms {np.sqrt((res[keep]**2).mean())*1000:.1f} mm  median {np.median(res[keep])*1000:.1f} mm  thr {thr*1000:.0f} mm")
T_base_front = inv(T_cb)
np.set_printoptions(precision=8, suppress=True)
print("T_base_front pos", T_base_front[:3,3].round(4), " euler xyz deg", R.from_matrix(T_base_front[:3,:3]).as_euler("xyz", degrees=True).round(1)); print("T_base_front =\n", T_base_front)
# systematic mug-vs-tool offset in base frame (should be small; residual bias of the grasp point)
bias = ((inv(T_cb)[:3,:3] @ P_cam[keep].T).T + inv(T_cb)[:3,3] - P_base[keep]).mean(0); print("mean (mug - tool) offset in base frame (m):", bias.round(3))
np.savez(os.path.expanduser("~/claude_fk/mug_calib_final.npz"), T_base_front=T_base_front, K_front=Kf, D_front=Df)
# ---- validation overlay: FK tool point (red) + next 1.5 s (green) + mug centroid (yellow) on frames NOT used for fitting
T_fb = inv(T_base_front); rows = []; dz = []
for fn in (files[1], files[50], files[101]):
    with h5py.File(fn) as f:
        q = f["arm/current_joint"][:, :7]; T = len(q); tool = np.array([panda_fk(qi)[:3,3] + panda_fk(qi)[:3,:3] @ TOOL_OFFSET for qi in q])
        for i in (T//8, T//2, 7*T//8):
            img = f["camera/front"][i].copy(); dep = f["depth/front"][i]/1000.0
            pc = (T_fb[:3,:3] @ tool[i:i+45].T).T + T_fb[:3,3]
            px = cv2.projectPoints(pc.reshape(-1,1,3), np.zeros(3), np.zeros(3), Kf, Df)[0].reshape(-1,2)
            for j,(u,v) in enumerate(px):
                if 0 <= u < 432 and 0 <= v < 240: cv2.circle(img, (int(u), int(v)), 3, (0,255,0) if j else (255,0,0), -1)
            r = mug_centroid(f["camera/front"][i], dep)
            if r is not None: cv2.circle(img, (int(r[1][0]), int(r[1][1])), 5, (255,255,0), 1)
            u, v = px[0]; zc = pc[0,2]; zd = dep[int(np.clip(v,0,239)), int(np.clip(u,0,431))]
            if zd > 0: dz.append(round(zc - zd, 3))
            cv2.putText(img, f"{os.path.basename(fn)[:15]} t={i}  tool z_cam {zc:.2f}  depth@px {zd:.2f}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,0), 1); rows.append(img)
print("tool z_cam - depth@pixel (m) on held-out frames:", dz)
Image.fromarray(np.concatenate(rows, 0)).save(os.path.expanduser("~/claude_fk/mug_proj_final.png")); print("saved mug_proj_final.png")
