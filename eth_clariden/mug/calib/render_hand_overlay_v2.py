"""Bimanual v2 check: project BOTH ORCA wireframes (left mount + left URDF, right mount + right
URDF) onto the top camera using the per-arm mug2_front extrinsics (left base = right base +
0.7 m Y), on held-out frames of the sanity_check demos. Run on Euler:
    python3 render_hand_overlay_v2.py   (needs ~/claude_fk/{fk_check.py,orca_hand_fk.py,orcahand_{right,left}.urdf})
"""
import glob, os, sys, h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
HERE = os.path.expanduser("~/claude_fk"); sys.path.insert(0, HERE)
os.environ["ORCA_URDF"] = os.path.join(HERE, "orcahand_right.urdf"); os.environ["ORCA_URDF_LEFT"] = os.path.join(HERE, "orcahand_left.urdf")
import orca_hand_fk as ofk
exec(open(os.path.join(HERE, "fk_check.py")).read().split("files = sorted")[0])   # panda_fk (reads global D = DH d's!)
TCP = np.array([0.13, 0.0, 0.07])
T_right_cam = np.array([[-0.00160789, -0.90637180, 0.42247790, 0.10703568], [-0.99985423, 0.00863841, 0.01472729, 0.35064200], [-0.01699794, -0.42239264, -0.90625357, 0.72825985], [0, 0, 0, 1]])
T_left_cam = T_right_cam.copy(); T_left_cam[1, 3] -= 0.7
files = sorted(glob.glob("/cluster/scratch/agavryushin/egoverse/v2/recordings/sanity_check/*.h5"))
with h5py.File(files[0]) as f:
    K = f["camera/intrinsics/top"][:][2:11].reshape(3, 3); DIST = f["camera/intrinsics/top"][:][11:25]


def proj(P, T_base_cam):
    Tcb = np.linalg.inv(T_base_cam); pc = (Tcb[:3, :3] @ P.T).T + Tcb[:3, 3]
    px = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)[0].reshape(-1, 2); px[pc[:, 2] <= 0.05] = np.nan; return px


def overlay(rgb, g, i):
    for side, T_base_cam in (("left", T_left_cam), ("right", T_right_cam)):
        fk = ofk.get_fk(side); q7 = g[f"arm/current_joint/{side}"][i, :7]; hj = g[f"hand/current_joint/{side}"][i]
        Tt = panda_fk(q7); Tt[:3, 3] += Tt[:3, :3] @ TCP; Tb = Tt @ ofk.mount_for(side, "mug_tcp")
        px = proj(fk.keypoints(hj, T_ref_world=Tb), T_base_cam); ofk.draw_wireframe(rgb, px, fk.edges, fk.finger_of_edge, thickness=1, radius=2)
        tp = proj(Tt[:3, 3][None], T_base_cam)[0]
        if np.isfinite(tp).all(): cv2.circle(rgb, (int(tp[0]), int(tp[1])), 4, (255, 0, 0) if side == "right" else (0, 255, 255), -1)
    return rgb


rows = []
for fn in (files[2], files[10], files[17]):
    with h5py.File(fn) as g:
        T = g["timestamps_ns"].shape[0]
        for i in (T // 8, T // 2, 7 * T // 8):
            rgb = overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i)
            cv2.putText(rgb, f"{os.path.basename(fn)[:15]} t={i}  L+R wireframes (URDF mounts)", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1); rows.append(rgb)
big = np.concatenate(rows, 0); big = cv2.resize(big, (big.shape[1] * 2, big.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
Image.fromarray(big).save(os.path.join(HERE, "v2_hand_overlay.png")); print("saved png")
with h5py.File(files[10]) as g:
    T = g["timestamps_ns"].shape[0]; vw = None
    for i in range(T):
        rgb = cv2.resize(overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i), (864, 480), interpolation=cv2.INTER_CUBIC)
        if vw is None: vw = cv2.VideoWriter(os.path.join(HERE, "v2_hand_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (864, 480))
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    vw.release(); print("saved mp4", T, "frames")
