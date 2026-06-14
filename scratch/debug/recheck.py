import glob, numpy as np, zarr
eps=sorted(glob.glob("/coc/flash7/paphiwetsa3/datasets/new_circle_3/*.zarr"))[:80]
in_range_zero=0; in_range_tot=0; full_zero=0; full_tot=0; tf_lt_len=0
for ep in eps:
    st=zarr.open_group(ep,mode="r")
    a=np.asarray(st["actions"][:],dtype=np.float64)
    tf=int(dict(st.attrs).get("total_frames", len(a)))
    z=(np.abs(a[:,0])<1)&(np.abs(a[:,1])<1)
    full_zero+=int(z.sum()); full_tot+=len(a)
    in_range_zero+=int(z[:tf].sum()); in_range_tot+=tf
    if tf<len(a): tf_lt_len+=1
print(f"episodes={len(eps)}")
print(f"FULL array:      {full_zero}/{full_tot} = {100*full_zero/full_tot:.1f}% zero")
print(f"WITHIN total_frames (what training uses): {in_range_zero}/{in_range_tot} = {100*in_range_zero/in_range_tot:.1f}% zero")
print(f"episodes where total_frames < len(actions): {tf_lt_len}/{len(eps)}")
# norm quantile within range
alla=[]
for ep in eps:
    st=zarr.open_group(ep,mode="r"); a=np.asarray(st["actions"][:],dtype=np.float64)
    tf=int(dict(st.attrs).get("total_frames", len(a))); alla.append(a[:tf])
alla=np.concatenate(alla)
print(f"\nnorm within-range: quantile_1={np.percentile(alla,1,axis=0)} (0 => still garbage; ~32 => clean)")
