"""Render the ORCA wireframe on held-out real_mug frames with a given TCP->hand mount
(default: the exact URDF mount in orca_hand_fk.HAND_MOUNTS["mug_tcp"]). Also writes a
short H.264-friendly mp4 (mp4v; re-encode with ffmpeg for browsers).
Run on Euler: python3 render_hand_overlay.py [mount_key]
"""
import glob, os, sys, h5py, numpy as np, cv2
from PIL import Image

HERE = os.path.expanduser("~/claude_fk")
sys.path.insert(0, HERE); os.environ["ORCA_URDF"] = os.path.join(HERE, "orcahand_right.urdf")
import orca_hand_fk as ofk
exec(open(os.path.join(HERE, "fk_check.py")).read().split("files = sorted")[0])   # panda_fk
TCP = np.array([0.13, 0.0, 0.07])
T_base_cam = np.array([[0.00063391, -0.90908461, 0.41661106, 0.10781140],
                       [-0.99977636, 0.00823022, 0.01948035, 0.34446003],
                       [-0.02113809, -0.41653024, -0.90887609, 0.72878461], [0, 0, 0, 1]])
K = np.array([[345.60421753, 0, 213.33850098], [0, 345.57580566, 120.38240051], [0, 0, 1]])
DIST = np.array([6.12003279   # NOTE: not 'D' - fk_check.py's panda_fk reads a global D (DH d-params)
    , -57.19741058, -0.00002767, -0.00046653, 240.82293701, 5.93924999, -56.1584549, 236.36027527, 0, 0, 0, 0, -0.00281097, 0.00253649])
Tcb = np.linalg.inv(T_base_cam)
fk = ofk.get_fk()
key = sys.argv[1] if len(sys.argv) > 1 else "mug_tcp"
T_tcp_hand = ofk.HAND_MOUNTS[key]


def T_tcp(q7):
    T = panda_fk(q7); T[:3, 3] = T[:3, 3] + T[:3, :3] @ TCP; return T


def proj(P_base):
    pc = (Tcb[:3, :3] @ P_base.T).T + Tcb[:3, 3]
    px = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)[0].reshape(-1, 2)
    px[pc[:, 2] <= 0.05] = np.nan; return px


def overlay(rgb, q7, hj):
    Tb = T_tcp(q7) @ T_tcp_hand
    px = proj(fk.keypoints(hj, T_ref_world=Tb)); ofk.draw_wireframe(rgb, px, fk.edges, fk.finger_of_edge, thickness=1, radius=2)
    tp = proj(T_tcp(q7)[:3, 3][None])[0]; cv2.circle(rgb, (int(tp[0]), int(tp[1])), 4, (255, 0, 0), -1); return rgb


files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
rows = []
for fn in (files[3], files[52], files[103]):
    with h5py.File(fn) as f:
        q7 = f["arm/current_joint"][:, :7]; hj = f["hand/current_joint"][:]; T = len(q7)
        for i in (T // 8, T // 2, 7 * T // 8):
            rgb = overlay(np.ascontiguousarray(f["camera/front"][i][..., ::-1]), q7[i], hj[i])
            cv2.putText(rgb, f"{os.path.basename(fn)[:15]} t={i}  mount={key}", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1); rows.append(rgb)
big = np.concatenate(rows, 0); big = cv2.resize(big, (big.shape[1] * 2, big.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
Image.fromarray(big).save(os.path.join(HERE, f"hand_overlay_{key}.png")); print("saved png")

with h5py.File(files[52]) as f:
    q7 = f["arm/current_joint"][:, :7]; hj = f["hand/current_joint"][:]; T = len(q7)
    vw = None
    for i in range(T):
        rgb = overlay(np.ascontiguousarray(f["camera/front"][i][..., ::-1]), q7[i], hj[i])
        rgb = cv2.resize(rgb, (864, 480), interpolation=cv2.INTER_CUBIC)
        if vw is None: vw = cv2.VideoWriter(os.path.join(HERE, f"hand_overlay_{key}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (864, 480))
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    vw.release(); print("saved mp4", T, "frames")
