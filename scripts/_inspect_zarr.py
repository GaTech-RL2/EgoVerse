import zarr, numpy as np
p = "/coc/flash7/paphiwetsa3/datasets/circle_3000/episode_T_circle_obs0_000000.zarr"
z = zarr.open(p, mode="r")
print("zarr", zarr.__version__)
try:
    print(z.tree())
except Exception as e:
    print("tree-fail", e)
def walk(g, pre=""):
    for k in list(g):
        v = g[k]
        if isinstance(v, zarr.Group):
            walk(v, pre + k + "/")
        else:
            print(f"{pre}{k}: shape={v.shape} dtype={v.dtype}")
walk(z)
print("ATTRS:", dict(z.attrs))
