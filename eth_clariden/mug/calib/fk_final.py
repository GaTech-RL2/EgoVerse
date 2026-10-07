import h5py, glob, numpy as np, os
from scipy.spatial.transform import Rotation as R
exec(open(os.path.expanduser("~/claude_fk/fk_check.py")).read().split("files = sorted")[0])
T_OFF = np.array([0.1325, -0.0049, 0.0735])
def ee_from_joints(q):           # -> (N,7) xyz + xyzw, same frame as arm/ee_target
    Ts = np.stack([panda_fk(qi) for qi in q])
    p = Ts[:,:3,3] + np.einsum("nij,j->ni", Ts[:,:3,:3], T_OFF)
    return np.concatenate([p, R.from_matrix(Ts[:,:3,:3]).as_quat()], 1)
files = sorted(glob.glob("/cluster/scratch/agovryushin/ben_icra/demos/*.h5".replace("agovryushin","agavryushin")))
per_ep, lagres = [], {k: [] for k in (0,1,2,3,4,6)}
for fn in files:
    with h5py.File(fn) as f:
        q = f["arm/current_joint"][:, :7]; e = f["arm/ee_target"][:]
    if len(q) < 60: continue
    fk = ee_from_joints(q)
    dp = np.linalg.norm(fk[:,:3]-e[:,:3], axis=1)
    dr = np.degrees(np.linalg.norm((R.from_quat(fk[:,3:]).inv()*R.from_quat(e[:,3:])).as_rotvec(), axis=1))
    per_ep.append((os.path.basename(fn), dp.mean()*1000, dp.max()*1000, dr.mean(), dr.max()))
    for k in lagres: lagres[k].append(np.linalg.norm(fk[k:,:3]-e[:len(e)-k,:3], axis=1).mean()*1000)
a = np.array([p[1:] for p in per_ep])
print(f"episodes checked: {len(per_ep)}")
print(f"pos residual (mm): mean-of-means {a[:,0].mean():.2f}, worst episode mean {a[:,0].max():.2f}, worst single frame {a[:,1].max():.1f}")
print(f"rot residual (deg): mean-of-means {a[:,2].mean():.2f}, worst episode mean {a[:,2].max():.2f}, worst single frame {a[:,3].max():.1f}")
print("lag test (mean mm when comparing FK(q[t+k]) to target[t]):", {k: round(float(np.mean(v)),2) for k,v in lagres.items()})
bad = [p for p in per_ep if p[1] > 15 or p[3] > 6]
print("episodes with mean pos>15mm or rot>6deg:", [(p[0], round(p[1],1), round(p[3],1)) for p in bad][:10])
