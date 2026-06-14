import glob, numpy as np, zarr
ROOT="/coc/flash7/paphiwetsa3/datasets/new_circle_3"
eps=sorted(glob.glob(ROOT+"/*.zarr"))[:60]
A=[]; tail_zero=[]; mid_zero=[]
for ep in eps:
    a=np.asarray(zarr.open_group(ep,mode="r")["actions"][:],dtype=np.float64)
    A.append(a)
    z=(np.abs(a[:,0])<1)&(np.abs(a[:,1])<1)
    T=len(a)
    # how many zero frames are in the last 20% (idle tail) vs scattered mid
    tail=z[int(T*0.8):].mean() if T>5 else 0
    mid=z[:int(T*0.8)].mean() if T>5 else 0
    tail_zero.append(tail); mid_zero.append(mid)
A=np.concatenate(A)
z=(np.abs(A[:,0])<1)&(np.abs(A[:,1])<1)
print(f"total frames={len(A)}  zero(0,0) frames={z.sum()} ({z.mean()*100:.1f}%)")
print(f"  fraction of (0,0) in idle TAIL (last 20%): {np.mean(tail_zero)*100:.1f}%")
print(f"  fraction of (0,0) SCATTERED in body (first 80%): {np.mean(mid_zero)*100:.1f}%")
nz=~z
print(f"\nNON-zero actions: x mean={A[nz,0].mean():.0f} std={A[nz,0].std():.0f} | y mean={A[nz,1].mean():.0f} std={A[nz,1].std():.0f}")
print(f"WITH zeros (what model trains on): x mean={A[:,0].mean():.0f} | y mean={A[:,1].mean():.0f}  <- pulled toward 0 by garbage")
q1=np.percentile(A,1,axis=0); q1nz=np.percentile(A[nz],1,axis=0)
print(f"\nnorm quantile_1 WITH zeros={q1}  vs WITHOUT zeros={q1nz}  (zeros corrupt the normalization floor)")
# stuck: consecutive near-zero movement
d=np.linalg.norm(np.diff(A,axis=0),axis=1)
print(f"\nconsecutive |delta action|<1px: {(d<1).mean()*100:.1f}% of steps (many = cursor parked / repeated)")
