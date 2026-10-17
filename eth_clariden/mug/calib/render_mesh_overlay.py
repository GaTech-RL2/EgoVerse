"""Sim-asset overlay: rasterize the FR3 + ORCA visual meshes (from the combined URDFs, posed
by the recorded 7 arm + 17 hand joints per arm) into the top camera and blend the silhouette
onto the real frame (green fill + contour), like a segmentation-mask overlay.

Mesh sources (not in the cvg-dex-teleop git tree): FR3 visual meshes from frankaemika/
franka_description (visual_link{i}.dae -> STL via trimesh, no axis remap), ORCA visual meshes
from orcahand_description v1/assets/urdf/{right,left}/visual. The Gavin mount STL is not
available -> the mount link is skipped. Run on Euler:
    python3 render_mesh_overlay.py   (needs ~/claude_fk/meshes/{franka,orca}, the URDFs, orca_hand_fk.py)
"""
import glob, os, sys, h5py, numpy as np, cv2, trimesh
import xml.etree.ElementTree as ET
from PIL import Image
HERE = os.path.expanduser("~/claude_fk"); sys.path.insert(0, HERE)
for k, f in (("ORCA_URDF", "orcahand_right.urdf"), ("ORCA_URDF_LEFT", "orcahand_left.urdf"), ("FR3_ORCA_URDF_RIGHT", "fr3_mount_orcahand.urdf"), ("FR3_ORCA_URDF_LEFT", "fr3_mount_orcahand_left.urdf")):
    os.environ[k] = os.path.join(HERE, f)
import orca_hand_fk as ofk
T_right_cam = np.array([[-0.00160789, -0.90637180, 0.42247790, 0.10703568], [-0.99985423, 0.00863841, 0.01472729, 0.35064200], [-0.01699794, -0.42239264, -0.90625357, 0.72825985], [0, 0, 0, 1]])
T_left_cam = T_right_cam.copy(); T_left_cam[1, 3] -= 0.7
CAM = {"right": T_right_cam, "left": T_left_cam}
files = sorted(glob.glob("/cluster/scratch/agavryushin/egoverse/v2/recordings/sanity_check/*.h5"))
with h5py.File(files[0]) as f:
    K = f["camera/intrinsics/top"][:][2:11].reshape(3, 3); DIST = f["camera/intrinsics/top"][:][11:25]
W, H = 432, 240


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def resolve(fn):
    b = os.path.basename(fn)
    for sub in ("franka", "orca"):
        p = os.path.join(HERE, "meshes", sub, b)
        if os.path.exists(p): return p
    return None


def load_visuals(urdf):
    """link -> list of (vertices in link frame (N,3), faces (M,3))."""
    out = {}; skipped = []
    for l in ET.parse(urdf).getroot().findall("link"):
        for v in l.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None: continue
            p = resolve(m.get("filename"))
            if p is None: skipped.append(m.get("filename")); continue
            mesh = trimesh.load(p, force="mesh"); V = np.asarray(mesh.vertices, float)
            if m.get("scale"): V = V * np.array([float(s) for s in m.get("scale").split()])
            o = v.find("origin"); T = np.eye(4)
            if o is not None:
                T[:3, :3] = rpy_mat(*[float(x) for x in o.get("rpy", "0 0 0").split()]); T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
            V = (T[:3, :3] @ V.T).T + T[:3, 3]
            out.setdefault(l.get("name"), []).append((V, np.asarray(mesh.faces)))
    return out, skipped


VIS = {}
for side in ("right", "left"):
    VIS[side], sk = load_visuals(ofk.DEFAULT_FR3_URDF[side] if False else os.environ[f"FR3_ORCA_URDF_{side.upper()}"])
    print(side, "links with meshes:", len(VIS[side]), "faces:", sum(len(f) for vs in VIS[side].values() for _, f in vs), "| skipped:", sorted(set(os.path.basename(s) for s in sk)))


def silhouette(side, q_arm, q_hand):
    """Rasterize all posed visual meshes of one arm into a (H, W) uint8 mask, nearest-first via z-sorted triangles."""
    poses = ofk.get_fr3_fk(side).poses(q_arm, q_hand); Tcb = np.linalg.inv(CAM[side]); mask = np.zeros((H, W), np.uint8)
    tris, depth = [], []
    for link, vis in VIS[side].items():
        if link not in poses: continue
        Tl = Tcb @ poses[link]
        for V, F in vis:
            Vc = (Tl[:3, :3] @ V.T).T + Tl[:3, 3]
            px = cv2.projectPoints(Vc.reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, DIST)[0].reshape(-1, 2)
            ok = Vc[:, 2] > 0.05
            keep = ok[F].all(1)
            if keep.any():
                tris.append(px[F[keep]]); depth.append(Vc[F[keep], 2].mean(1))
    if not tris: return mask
    tris = np.concatenate(tris); depth = np.concatenate(depth)
    tris = np.clip(tris, -2 * W, 3 * W).astype(np.int32)
    cv2.fillPoly(mask, list(tris), 255)
    return mask


def overlay(rgb, g, i, alpha=0.45):
    out = rgb.copy()
    for side, col in (("right", (40, 220, 40)), ("left", (40, 220, 40))):
        m = silhouette(side, g[f"arm/current_joint/{side}"][i, :7], g[f"hand/current_joint/{side}"][i])
        sel = m > 0
        out[sel] = (out[sel] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cnts, -1, (120, 255, 120), 1)
    return out



if __name__ == "__main__":
    rows = []
    for fn in (files[2], files[10], files[17]):
        with h5py.File(fn) as g:
            T = g["timestamps_ns"].shape[0]
            for i in (T // 8, T // 2, 7 * T // 8):
                rgb = overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i)
                cv2.putText(rgb, f"{os.path.basename(fn)[:15]} t={i}  FR3+ORCA meshes posed by H5 joints", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1); rows.append(rgb)
    big = np.concatenate(rows, 0); big = cv2.resize(big, (big.shape[1] * 2, big.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    Image.fromarray(big).save(os.path.join(HERE, "v2_mesh_overlay.png")); print("saved png", flush=True)
    with h5py.File(files[10]) as g:
        T = g["timestamps_ns"].shape[0]; vw = None
        for i in range(0, T, 2):
            rgb = cv2.resize(overlay(np.ascontiguousarray(g["camera/image/top"][i][..., ::-1]), g, i), (864, 480), interpolation=cv2.INTER_CUBIC)
            if vw is None: vw = cv2.VideoWriter(os.path.join(HERE, "v2_mesh_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (864, 480))
            vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        vw.release(); print("saved mp4", T // 2, "frames @15fps (every 2nd frame)")
