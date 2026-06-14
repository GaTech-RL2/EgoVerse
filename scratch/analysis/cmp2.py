import glob, numpy as np, zarr
def summ(root, n=60):
    eps=sorted(glob.glob(root+"/*.zarr"))
    if not eps: return f"NO episodes at {root}"
    use=eps[:n]; zr=0; zt=0; alla=[]; goal=0; tfs=[]; lens=[]; keys=set()
    for ep in use:
        st=zarr.open_group(ep,mode="r"); keys|=set(st.keys())
        a=np.asarray(st["actions"][:],dtype=np.float64)
        tf=int(dict(st.attrs).get("total_frames",len(a))); tfs.append(tf); lens.append(len(a))
        z=(np.abs(a[:,0])<1)&(np.abs(a[:,1])<1); zr+=int(z[:tf].sum()); zt+=tf
        alla.append(a[:tf])
        if "goal_pose" in st.keys(): goal+=1
    A=np.concatenate(alla)
    mv=np.linalg.norm(np.diff(A,axis=0),axis=1)
    return (f"episodes={len(eps)} has_goal={goal}/{len(use)} keys={sorted(keys)}\n"
            f"  mean_tf={np.mean(tfs):.0f} mean_len={np.mean(lens):.0f} tail_trim={sum(1 for t,l in zip(tfs,lens) if t<l)}/{len(use)}\n"
            f"  in-range zeros={100*zr/zt:.2f}%  action x[{A[:,0].min():.0f},{A[:,0].max():.0f}] y[{A[:,1].min():.0f},{A[:,1].max():.0f}]\n"
            f"  per-step move: median={np.median(mv):.1f} mean={mv.mean():.1f} max={mv.max():.0f}  >50px-jumps={100*(mv>50).mean():.1f}%")
for nm,rt in [("circle_750 (HNet WORKS)","/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle"),
              ("new_circle_3 (HPT FAILS)","/coc/flash7/paphiwetsa3/datasets/new_circle_3")]:
    print(f"== {nm} ==\n{summ(rt)}\n")
