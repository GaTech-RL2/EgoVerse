"""Zoomed left-hand mesh overlays under candidate joint-sign conventions, to find how the
recorded left-hand joints map onto orcahand_left.urdf. Writes left_hand_candidates.png."""
import glob, os, sys, h5py, numpy as np, cv2
from PIL import Image
HERE = os.path.expanduser("~/claude_fk"); sys.path.insert(0, HERE)
for k, f in (("ORCA_URDF", "orcahand_right.urdf"), ("ORCA_URDF_LEFT", "orcahand_left.urdf"), ("FR3_ORCA_URDF_RIGHT", "fr3_mount_orcahand.urdf"), ("FR3_ORCA_URDF_LEFT", "fr3_mount_orcahand_left.urdf")):
    os.environ[k] = os.path.join(HERE, f)
import orca_hand_fk as ofk
import render_mesh_overlay as rmo   # reuses VIS, CAM, K, DIST, silhouette()
J = ofk.ORCA_V1_JOINT_ORDER
ABD = [i for i, n in enumerate(J) if n.endswith("_abd")]; THUMB = [i for i, n in enumerate(J) if n.startswith("thumb")]; WRIST = [J.index("wrist")]
NONTHUMB_ABD = [i for i in ABD if not J[i].startswith("thumb")]
CANDS = {
    "as-is": lambda q: q,
    "neg abd": lambda q: _neg(q, ABD),
    "neg finger abd": lambda q: _neg(q, NONTHUMB_ABD),
    "neg thumb": lambda q: _neg(q, THUMB),
    "neg wrist": lambda q: _neg(q, WRIST),
    "neg abd+wrist": lambda q: _neg(q, ABD + WRIST),
    "neg all": lambda q: -q,
}
def _neg(q, idx):
    q = q.copy(); q[idx] *= -1; return q

files = sorted(glob.glob("/cluster/scratch/agavryushin/egoverse/v2/recordings/sanity_check/*.h5"))
tiles_rows = []
for fn, i in ((files[2], 243), (files[10], 219), (files[17], 281), (files[5], 150)):
    with h5py.File(fn) as g:
        rgb0 = np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]); qa = g["arm/current_joint/left"][i, :7]; qh = g["hand/current_joint/left"][i].astype(float)
        # crop window around the left TCP
        poses = ofk.get_fr3_fk("left").poses(qa, qh); tcp = poses["fr3_link8"][:3, 3] + poses["fr3_link8"][:3, :3] @ np.array([0.13, 0, 0.07])
        Tcb = np.linalg.inv(rmo.CAM["left"]); pc = Tcb[:3, :3] @ tcp + Tcb[:3, 3]; px = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), rmo.K, rmo.DIST)[0].reshape(2)
        cx, cy = int(np.clip(px[0], 80, 352)), int(np.clip(px[1], 60, 180))
        row = [cv2.resize(rgb0[cy - 60:cy + 60, cx - 80:cx + 80], (320, 240), interpolation=cv2.INTER_CUBIC)]
        cv2.putText(row[0], f"real  t={i}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        for name, fn_ in CANDS.items():
            m = rmo.silhouette("left", qa, fn_(qh)); out = rgb0.copy(); sel = m > 0
            out[sel] = (out[sel] * 0.5 + np.array((40, 220, 40)) * 0.5).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cnts, -1, (120, 255, 120), 1)
            crop = cv2.resize(out[cy - 60:cy + 60, cx - 80:cx + 80], (320, 240), interpolation=cv2.INTER_CUBIC); cv2.putText(crop, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1); row.append(crop)
        tiles_rows.append(np.concatenate(row, 1))
Image.fromarray(np.concatenate(tiles_rows, 0)).save(os.path.join(HERE, "left_hand_candidates.png")); print("saved")
