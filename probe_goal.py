import sys, zarr, numpy as np, glob, os
ds = sys.argv[1]
eps = sorted(glob.glob(os.path.join(ds, "*.zarr")))[:8]
goals = []
for p in eps:
    try:
        g = zarr.open(p, mode="r")
        gp = np.asarray(g["goal_pose"][:])
        goals.append(gp[0])
        print(os.path.basename(p), "goal[0]=", np.round(gp[0], 2), "n_unique_in_ep=", len(np.unique(gp.reshape(len(gp), -1), axis=0)))
    except Exception as e:
        print(p, "ERR", e)
goals = np.array(goals)
print("=== across-episode goal spread (std):", np.round(goals.std(axis=0), 3), " -> FIXED" if goals.std() < 1.0 else " -> VARIES")
