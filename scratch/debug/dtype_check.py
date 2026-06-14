import glob, numpy as np, zarr
eps=sorted(glob.glob("/coc/flash7/paphiwetsa3/datasets/new_circle_3/*.zarr"))[:50]
maxerr_a=0; maxerr_s=0
for ep in eps:
    st=zarr.open_group(ep,mode="r")
    a64=np.asarray(st["actions"][:],dtype=np.float64)
    s64=np.asarray(st["observations.state"][:],dtype=np.float64)
    a32=a64.astype(np.float32).astype(np.float64)   # round-trip like the loader
    s32=s64.astype(np.float32).astype(np.float64)
    maxerr_a=max(maxerr_a, np.abs(a64-a32).max())
    maxerr_s=max(maxerr_s, np.abs(s64-s32).max())
print(f"actions: dtype on disk = {zarr.open_group(eps[0],mode='r')['actions'].dtype}")
print(f"max |float64 - float32| over {len(eps)} episodes:")
print(f"  actions (cursor px, range 0-512): {maxerr_a:.2e} pixels")
print(f"  state   (pos px / angle rad):     {maxerr_s:.2e}")
print(f"  (1 pixel = 1.0; world is 512x512) -> error is ~{maxerr_a/512*100:.6f}% of the arena")
# also: after quantile-normalization to [-1,1], what's the float32 resolution there?
print(f"\nfloat32 resolution near value 256 (mid-arena): {np.spacing(np.float32(256)):.2e} px")
print(f"float32 resolution near value 1.0 (normalized): {np.spacing(np.float32(1.0)):.2e}")
