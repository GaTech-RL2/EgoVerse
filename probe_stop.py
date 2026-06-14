import zarr, numpy as np, glob, os
D = "/coc/flash7/paphiwetsa3/datasets/circle_3000"
eps = sorted(glob.glob(os.path.join(D, "*.zarr")))
print("n_episodes", len(eps))
for p in eps[:3]:
    g = zarr.open(p, mode="r")
    act = np.asarray(g["actions"][:])
    st = np.asarray(g["observations.state"][:])
    pc = np.asarray(g["observations.pusher_cmd_pose"][:])
    gp = np.asarray(g["goal_pose"][:])
    rw = np.asarray(g["reward"][:]).reshape(-1)
    print("===", os.path.basename(p), "T=", len(act))
    print("  actions shape", act.shape, "ex0", np.round(act[0],1), "ex_last", np.round(act[-1],1))
    print("  state   shape", st.shape, "ex0", np.round(st[0],1))
    print("  pusher_cmd shape", pc.shape, "ex0", np.round(pc[0],1))
    print("  goal", np.round(gp[0],1))
    print("  reward[:4]", np.round(rw[:4],3), "reward[-4:]", np.round(rw[-4:],3), "max", round(float(rw.max()),3))
    idx = np.where(rw >= 0.95)[0]
    if len(idx):
        ts = int(idx[0]); print("  t*(reward>=0.95)=", ts, "of", len(rw))
        print("  at t*: state[:2]", np.round(st[ts][:2],1), "action", np.round(act[ts],1), "pusher_cmd", np.round(pc[ts][:2],1))
        print("  AFTER t* | action vs state[:2] (cursor vs pusher) mean|diff|:",
              round(float(np.abs(act[ts:,:2]-st[ts:,:2]).mean()),2))
        print("  reward after t*: min", round(float(rw[ts:].min()),3), "last", round(float(rw[-1]),3),
              "-> drops below 0.95?", bool((rw[ts:] < 0.95).any()))
    else:
        print("  never reaches 0.95 (max", round(float(rw.max()),3), ")")
