"""Drive cvg-dex-teleop's combined FR3+ORCA URDFs with the recorded joints (7 arm + 17 hand per
arm) and project arm links + hand wireframes onto the top camera of the v2 demos. Also
cross-checks the URDF chain against the hand-coded path (Panda DH + TCP + HAND_MOUNTS) and
against the recorded ee_target. Run on Euler (needs ~/claude_fk/{fk_check.py, orca_hand_fk.py,
orcahand_{right,left}.urdf, fr3_mount_orcahand{,_left}.urdf})."""
import glob, os, sys, h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
HERE = os.path.expanduser("~/claude_fk"); sys.path.insert(0, HERE)
for k, f in (("ORCA_URDF", "orcahand_right.urdf"), ("ORCA_URDF_LEFT", "orcahand_left.urdf"), ("FR3_ORCA_URDF_RIGHT", "fr3_mount_orcahand.urdf"), ("FR3_ORCA_URDF_LEFT", "fr3_mount_orcahand_left.urdf")):
    os.environ[k] = os.path.join(HERE, f)
import orca_hand_fk as ofk
exec(open(os.path.join(HERE, "fk_check.py")).read().split("files = sorted")[0])   # panda_fk (global D = DH d's)
TCP = np.array([0.13, 0.0, 0.07])
T_right_cam = np.array([[-0.00160789, -0.90637180, 0.42247790, 0.10703568], [-0.99985423, 0.00863841, 0.01472729, 0.35064200], [-0.01699794, -0.42239264, -0.90625357, 0.72825985], [0, 0, 0, 1]])
T_left_cam = T_right_cam.copy(); T_left_cam[1, 3] -= 0.7
CAM = {"right": T_right_cam, "left": T_left_cam}
files = sorted(glob.glob("/cluster/scratch/agavryushin/egoverse/v2/recordings/sanity_check/*.h5"))
with h5py.File(files[0]) as f:
    K = f["camera/intrinsics/top"][:][2:11].reshape(3, 3); DIST = f["camera/intrinsics/top"][:][11:25]


def proj(P, T_base_cam):
    Tcb = np.linalg.inv(T_base_cam); pc = (Tcb[:3, :3] @ P.T).T + Tcb[:3, 3]
    px = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)[0].reshape(-1, 2); px[pc[:, 2] <= 0.05] = np.nan; return px


# ---- numeric cross-check on a few episodes
d_hand, d_tcp, d_tcp_dh = [], [], []
for fn in files[::5]:
    with h5py.File(fn) as g:
        for side in ("left", "right"):
            fr3 = ofk.get_fr3_fk(side); hand_fk = ofk.get_fk(side); mount = ofk.mount_for(side, "mug_tcp")
            qa = g[f"arm/current_joint/{side}"][:, :7]; qh = g[f"hand/current_joint/{side}"][:]; e = g[f"arm/ee_target/{side}"][:, :3]
            for i in range(0, len(qa), 40):
                hand_u, arm_u, T8 = fr3.fk(qa[i], qh[i])
                Tt = panda_fk(qa[i]); Tt[:3, 3] += Tt[:3, :3] @ TCP
                hand_c = hand_fk.keypoints(qh[i], T_ref_world=Tt @ mount)
                d_hand.append(np.linalg.norm(hand_u - hand_c, axis=1).max())
                tcp_u = T8[:3, 3] + T8[:3, :3] @ TCP
                d_tcp.append(np.linalg.norm(tcp_u - e[i])); d_tcp_dh.append(np.linalg.norm(Tt[:3, 3] - e[i]))
print(f"URDF chain vs DH+TCP+mount hand keypoints: max {np.max(d_hand)*1000:.2f} mm (mean {np.mean(d_hand)*1000:.2f} mm)")
print(f"URDF link8+TCP vs ee_target: mean {np.mean(d_tcp)*1000:.1f} mm | DH+TCP vs ee_target: mean {np.mean(d_tcp_dh)*1000:.1f} mm")

# ---- render: arm links (white) + hand wireframe (per finger), both arms
def overlay(rgb, g, i):
    for side in ("left", "right"):
        fr3 = ofk.get_fr3_fk(side); hand, arm, T8 = fr3.fk(g[f"arm/current_joint/{side}"][i, :7], g[f"hand/current_joint/{side}"][i])
        pa = proj(arm, CAM[side])
        for a, b in zip(pa[:-1], pa[1:]):
            if np.isfinite(a).all() and np.isfinite(b).all(): cv2.line(rgb, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (230, 230, 230), 2, cv2.LINE_AA)
        for p in pa:
            if np.isfinite(p).all() and 0 <= p[0] < 432 and 0 <= p[1] < 240: cv2.circle(rgb, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
        ofk.draw_wireframe(rgb, proj(hand, CAM[side]), fr3.edges, fr3.finger_of_edge, thickness=1, radius=2, skip=("tower",))
    return rgb


rows = []
for fn in (files[2], files[10], files[17]):
    with h5py.File(fn) as g:
        T = g["timestamps_ns"].shape[0]
        for i in (T // 8, T // 2, 7 * T // 8):
            rgb = overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i)
            cv2.putText(rgb, f"{os.path.basename(fn)[:15]} t={i}  FR3+ORCA URDF driven by H5 joints", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1); rows.append(rgb)
big = np.concatenate(rows, 0); big = cv2.resize(big, (big.shape[1] * 2, big.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
Image.fromarray(big).save(os.path.join(HERE, "v2_fr3_urdf_overlay.png")); print("saved png")
with h5py.File(files[10]) as g:
    T = g["timestamps_ns"].shape[0]; vw = None
    for i in range(T):
        rgb = cv2.resize(overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i), (864, 480), interpolation=cv2.INTER_CUBIC)
        if vw is None: vw = cv2.VideoWriter(os.path.join(HERE, "v2_fr3_urdf_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (864, 480))
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    vw.release(); print("saved mp4", T, "frames")
