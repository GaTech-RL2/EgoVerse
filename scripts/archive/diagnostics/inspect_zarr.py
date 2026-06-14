import zarr
import numpy as np

z = zarr.open_group("/coc/cedarp-dxu345-0/Tsim_datasets2/new_circle_2/new_circle_2/episode_T_circle_obs0_000000.zarr", mode="r")
print("Keys:", list(z.keys()))
print("Attrs:", dict(z.attrs))
for k in z.keys():
    arr = z[k]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.shape[0] > 0:
        print(f"    first: {arr[0]}")
