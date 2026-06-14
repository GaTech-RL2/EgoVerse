import os, glob, numpy as np, zarr
ROOT="/coc/flash7/paphiwetsa3/datasets/new_circle_3"
eps=sorted(glob.glob(ROOT+"/*.zarr"))[:60]
all_a=[]; all_s=[]; zero_frac=[]; stuck_frac=[]
for ep in eps:
    st=zarr.open_group(ep,mode="r")
    a=np.asarray(st["actions"][:],dtype=np.float64)      # (T,2) cursor pos
    s=np.asarray(st["observations.state"][:],dtype=np.float64)  # (T,5)
    all_a.append(a); all_s.append(s)
    # garbage (0,0) frames
    zero_frac.append(((np.abs(a[:,0])<1)&(np.abs(a[:,1])<1)).mean())
    # "stuck": consecutive action barely moves
    d=np.linalg.norm(np.diff(a,axis=0),axis=1)
    stuck_frac.append((d<1.0).mean())
A=np.concatenate(all_a); S=np.concatenate(all_s)
print(f"episodes={len(eps)}  total_frames={len(A)}")
print(f"action x: min={A[:,0].min():.1f} max={A[:,0].max():.1f} mean={A[:,0].mean():.1f} std={A[:,0].std():.1f}")
print(f"action y: min={A[:,1].min():.1f} max={A[:,1].max():.1f} mean={A[:,1].mean():.1f} std={A[:,1].std():.1f}")
print(f"(0,0)-garbage frame fraction: mean={np.mean(zero_frac)*100:.1f}%  max-in-an-ep={np.max(zero_frac)*100:.1f}%")
print(f"'stuck' (consecutive |Δaction|<1px) fraction: mean={np.mean(stuck_frac)*100:.1f}%")
# histogram of action positions -> is there a dominant cluster / fixed point?
print("\naction 2D histogram (12x12 over 0..512, % of frames per cell, top cells):")
H,xe,ye=np.histogram2d(A[:,0],A[:,1],bins=12,range=[[0,512],[0,512]])
H=H/H.sum()*100
flat=[(H[i,j],xe[i],ye[j]) for i in range(12) for j in range(12)]
for v,x,y in sorted(flat,reverse=True)[:6]:
    print(f"  cell x~{x:3.0f} y~{y:3.0f}: {v:4.1f}% of all action frames")
# quantile norm stats (what preprocessing uses)
q1=np.percentile(A,1,axis=0); q99=np.percentile(A,99,axis=0)
print(f"\nquantile_1 (norm)={q1}  quantile_99={q99}")
print(f"action span used by norm: x={q99[0]-q1[0]:.0f}px y={q99[1]-q1[1]:.0f}px (world=512)")
