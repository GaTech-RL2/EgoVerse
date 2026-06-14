import zarr, numpy as np, glob, os
ds = "/coc/flash7/paphiwetsa3/datasets/pushshapes_paper"
eps = sorted(glob.glob(os.path.join(ds, "*.zarr")))[:5]
for p in eps:
    g = zarr.open(p, mode="r")
    gp = np.asarray(g["goal_pose"][:])[0]
    print(os.path.basename(p), "goal_pose[0] full precision =", repr(gp.tolist()))
print("pi/4 =", np.pi/4)
