import os
import h5py, glob, numpy as np
from scipy.spatial.transform import Rotation as R

# Franka Panda modified-DH (Craig), flange frame (d8 = 0.107), no gripper offset
A     = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
D     = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
ALPHA = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
def tf(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a],[st*ca, ct*ca, -sa, -sa*d],[st*sa, ct*sa, ca, ca*d],[0,0,0,1]])
def panda_fk(q):
    T = np.eye(4)
    for i in range(7): T = T @ tf(A[i], D[i], ALPHA[i], q[i])
    return T @ tf(A[7], D[7], ALPHA[7], 0.0)

files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
sel = [files[i] for i in (3, 40, 80, 120)]
allP, allR = [], []
for fn in sel:
    with h5py.File(fn) as f:
        q = f["arm/current_joint"][:, :7]; e = f["arm/ee_target"][:]
    Ts = np.stack([panda_fk(qi) for qi in q])
    p_fl, R_fl = Ts[:, :3, 3], Ts[:, :3, :3]
    p_t = e[:, :3]
    R_t = R.from_quat(e[:, [4,5,6,3]]).as_matrix()   # wxyz -> xyzw
    raw = np.linalg.norm(p_t - p_fl, axis=1)
    # tool offset expressed in flange frame
    t_local = np.einsum("nji,nj->ni", R_fl, p_t - p_fl)   # R_fl^T (p_t - p_fl)
    R_off = np.einsum("nji,njk->nik", R_fl, R_t)          # R_fl^T R_t
    allP.append(t_local); allR.append(R_off)
    print(f"{os.path.basename(fn)}: raw |p_t - p_flange| mean {raw.mean():.4f} std {raw.std():.4f} m | t_local mean {t_local.mean(0).round(4)} std {t_local.std(0).round(4)}")
t = np.concatenate(allP).mean(0)
Rm = R.from_matrix(np.concatenate(allR)).mean().as_matrix()
print("\nestimated constant tool offset in flange frame (m):", t.round(4), " |t| =", np.linalg.norm(t).round(4))
print("estimated constant rot offset (euler xyz deg):", R.from_matrix(Rm).as_euler("xyz", degrees=True).round(2))
# residual after applying the constant offset
res_p, res_r = [], []
for fn in sel:
    with h5py.File(fn) as f:
        q = f["arm/current_joint"][:, :7]; e = f["arm/ee_target"][:]
    Ts = np.stack([panda_fk(qi) for qi in q])
    p_fk = Ts[:, :3, 3] + np.einsum("nij,j->ni", Ts[:, :3, :3], t)
    R_fk = Ts[:, :3, :3] @ Rm
    R_t = R.from_quat(e[:, [4,5,6,3]]).as_matrix()
    res_p.append(np.linalg.norm(e[:, :3] - p_fk, axis=1))
    res_r.append(np.degrees(np.linalg.norm(R.from_matrix(np.einsum("nji,njk->nik", R_fk, R_t)).as_rotvec(), axis=1)))
res_p = np.concatenate(res_p); res_r = np.concatenate(res_r)
print(f"\nresidual vs commanded ee_target after offset: pos mean {res_p.mean()*1000:.2f} mm, p95 {np.percentile(res_p,95)*1000:.2f} mm, max {res_p.max()*1000:.2f} mm | rot mean {res_r.mean():.2f} deg, p95 {np.percentile(res_r,95):.2f} deg")
# sanity: is a no-offset or z-only offset hypothesis as good?
print("quat sign check: w mean", e[:,3].mean().round(3))
