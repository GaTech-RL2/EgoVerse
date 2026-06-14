import zarr, numpy as np, glob, os, json
D = "/coc/flash7/paphiwetsa3/datasets/circle_3000"
eps = sorted(glob.glob(os.path.join(D, "*.zarr")))
print("n_eps", len(eps))
real_drift = 0   # episodes that solve then drop below 0.95 WITHIN the real (non-pad) frames
solve_last = 0   # episodes that first hit 0.95 on their last real frame
never = 0
checked = 0
for p in eps[:200]:
    g = zarr.open(p, mode="r")
    rw = np.asarray(g["reward"][:]).reshape(-1)
    act = np.asarray(g["actions"][:])
    # find real length: attrs total_frames, else first trailing [0,0] action run
    tf = None
    try:
        a = dict(g.attrs)
        for k in ("total_frames","num_frames","n_frames","length","valid_len"):
            if k in a: tf = int(a[k]); break
    except Exception: pass
    if tf is None:
        # infer: last frame whose action is not [0,0]
        nz = np.where(np.abs(act).sum(1) > 1e-6)[0]
        tf = int(nz[-1]) + 1 if len(nz) else len(act)
    real = rw[:tf]
    checked += 1
    idx = np.where(real >= 0.95)[0]
    if len(idx) == 0:
        never += 1; continue
    t0 = int(idx[0])
    if t0 >= tf - 1:
        solve_last += 1
    # does it drop below 0.95 AFTER first solving, within real frames?
    if (real[t0:] < 0.95).any():
        real_drift += 1
print("checked", checked, "| array_len_vs_real: solve_on_last_realframe", solve_last,
      "| REAL post-solve drift (the premise)", real_drift, "| never_solve", never)
# show 3 examples with array len vs real len + reward tail in real region
for p in eps[:3]:
    g = zarr.open(p, mode="r"); rw=np.asarray(g["reward"][:]).reshape(-1); act=np.asarray(g["actions"][:])
    nz=np.where(np.abs(act).sum(1)>1e-6)[0]; tf=int(nz[-1])+1 if len(nz) else len(act)
    print(os.path.basename(p)[:30], "array_len", len(rw), "real_len", tf,
          "reward[real_last-3:real_last]", np.round(rw[max(0,tf-3):tf],3),
          "attrs", list(dict(g.attrs).keys())[:6])
