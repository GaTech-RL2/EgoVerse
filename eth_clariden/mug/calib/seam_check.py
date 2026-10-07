import h5py, glob, numpy as np
from scipy.spatial.transform import Rotation as R
files = sorted(glob.glob("/cluster/scratch/agavryushin/ben_icra/demos/*.h5"))
Q = []
for fn in files:
    with h5py.File(fn) as f:
        e = f["arm/ee_target"][:]
    if len(e) >= 60: Q.append(e[:, 3:])
Q = np.concatenate(Q); Rm = R.from_quat(Q)          # xyzw
print("frames:", len(Q))
Rx180 = R.from_euler("x", 180, degrees=True); Ry180 = R.from_euler("y", 180, degrees=True)
for name, rot in (("as recorded", Rm), ("tool * Rx(180)", Rm * Rx180), ("tool * Ry(180)", Rm * Ry180)):
    for seq in ("xyz", "ZYX", "zyx", "XYZ"):
        a = rot.as_euler(seq, degrees=True)
        near = (np.abs(a) > 170).any(1).mean() * 100
        print(f"  {name:16s} seq={seq:4s}  range min {a.min(0).round(0)} max {a.max(0).round(0)}  frames with |angle|>170deg: {near:5.1f}%")
