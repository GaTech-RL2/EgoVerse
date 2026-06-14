import zarr, numpy as np, glob, os
ds = "/coc/flash7/paphiwetsa3/datasets/circle_950norm_plus_new400"
allp = sorted(glob.glob(os.path.join(ds, "*.zarr")))
# sample some normalized-origin (no prefix) and some new400_ ones
norm = [p for p in allp if not os.path.basename(p).startswith("new400_")][:5]
new4 = [p for p in allp if os.path.basename(p).startswith("new400_")][:5]
goals = []
for label, lst in [("orig950", norm), ("new400", new4)]:
    for p in lst:
        g = zarr.open(p, mode="r")
        gp = np.asarray(g["goal_pose"][:])[0]
        goals.append(gp)
        print(label, os.path.basename(p)[:40], "goal[0]=", np.round(gp, 2))
goals = np.array(goals)
print("=== goal std across episodes:", np.round(goals.std(axis=0), 2), "(0 = FIXED, >0 = VARIES/random-goal)")
