import h5py, glob, numpy as np, os
from scipy.spatial.transform import Rotation as R
exec(open(os.path.expanduser("~/claude_fk/fk_check.py")).read().split("files = sorted")[0])  # reuse panda_fk
files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
sel = [files[i] for i in (3, 40, 80, 120)]
Q, E = [], []
for fn in sel:
    with h5py.File(fn) as f: Q.append(f["arm/current_joint"][:, :7]); E.append(f["arm/ee_target"][:])
Q = np.concatenate(Q); E = np.concatenate(E)
R_fl = np.stack([panda_fk(q)[:3,:3] for q in Q])
def rotres(R_t, side):
    if side == "flange":  off = np.einsum("nji,njk->nik", R_fl, R_t)   # R_fl^T R_t  (R_t = R_fl @ off)
    else:                 off = np.einsum("nij,nkj->nik", R_t, R_fl)   # R_t R_fl^T  (R_t = off @ R_fl)
    rv = R.from_matrix(off).as_rotvec()
    Rm = R.from_rotvec(rv).mean().as_matrix()
    res = np.degrees(np.linalg.norm(R.from_matrix(np.einsum("ji,njk->nik", Rm, off)).as_rotvec(), axis=1))
    return res, R.from_matrix(Rm).as_euler("xyz", degrees=True)
print("order  conj  side    | resid rot mean / p95 (deg) | const offset euler xyz (deg)")
for order, idx in (("wxyz", [4,5,6,3]), ("xyzw", [3,4,5,6])):
    for conj in (False, True):
        q = E[:, idx].copy()
        if conj: q[:, :3] *= -1
        R_t = R.from_quat(q).as_matrix()
        for side in ("flange", "base"):
            res, eul = rotres(R_t, side)
            print(f"{order:6s} {str(conj):5s} {side:7s} | {res.mean():6.2f} / {np.percentile(res,95):6.2f}        | {eul.round(1)}")
