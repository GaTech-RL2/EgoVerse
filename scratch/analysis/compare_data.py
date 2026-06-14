import glob, numpy as np, zarr
def summ(root, n=60):
    eps=sorted(glob.glob(root+"/*.zarr"))
    if not eps: return f"NO episodes at {root}"
    use=eps[:n]
    tf_lt=0; zr_in=0; zr_tot=0; alla=[]; alls=[]; goal=0; tfs=[]; lens=[]; keys=set()
    for ep in use:
        st=zarr.open_group(ep,mode="r"); ks=set(st.keys()); keys|=ks
        a=np.asarray(st["actions"][:],dtype=np.float64)
        s=np.asarray(st["observations.state"][:],dtype=np.float64)
        tf=int(dict(st.attrs).get("total_frames",len(a)))
        tfs.append(tf); lens.append(len(a))
        if tf<len(a): tf_lt+=1
        z=(np.abs(a[:,0])<1)&(np.abs(a[:,1])<1)
        zr_in+=int(z[:tf].sum()); zr_tot+=tf
        alla.append(a[:tf]); alls.append(s[:tf])
        if "goal_pose" in ks: goal+=1
    A=np.concatenate(alla); S=np.concatenate(alls)
    md=np.median(np.linalg.norm(np.diff(A,axis=0),axis=1))
    return (f"episodes(total)={len(eps)}  has_goal_pose={goal}/{len(use)}\n"
            f"  total_frames<len: {tf_lt}/{len(use)}  mean_tf={np.mean(tfs):.0f} mean_len={np.mean(lens):.0f}\n"
            f"  in-range (0,0) zeros: {100*zr_in/zr_tot:.2f}%\n"
            f"  action x[{A[:,0].min():.0f},{A[:,0].max():.0f}] mean={A[:,0].mean():.0f} std={A[:,0].std():.0f}\n"
            f"  action y[{A[:,1].min():.0f},{A[:,1].max():.0f}] mean={A[:,1].mean():.0f} std={A[:,1].std():.0f}\n"
            f"  state mean={np.round(S.mean(0),1)} std={np.round(S.std(0),1)}\n"
            f"  per-step action move median={md:.1f}px")
for name,root in [("circle (WORKS)","/coc/cedarp-dxu345-0/Tsim_datasets2/circle"),
                  ("circle3","/coc/cedarp-dxu345-0/Tsim_datasets2/circle3"),
                  ("new_circle_3 (FAILS)","/coc/flash7/paphiwetsa3/datasets/new_circle_3")]:
    print(f"===== {name} =====\n{summ(root)}\n")
