import os, glob
os.environ["EGOVERSE_TRIM_IDLE"]="1"
from egomimic.rldb.embodiment.pushshapes import get_keymap_causal
from egomimic.rldb.zarr.zarr_dataset_inmem import InMemoryZarrDataset
ep=sorted(glob.glob("/coc/flash7/paphiwetsa3/datasets/new_circle_3/*.zarr"))[0]
ds=InMemoryZarrDataset(ep, key_map=get_keymap_causal(action_horizon=32))
print("episode:", ep.split("/")[-1])
print("total_frames(attr):", ds.total_frames)
print("len(ds):", len(ds))
print("has _valid_indices:", getattr(ds,"_valid_indices",None) is not None)
vi=getattr(ds,"_valid_indices",None)
print("n valid:", len(vi) if vi is not None else "None")
print("_mem keys:", list(ds._mem.keys()))
import numpy as np
a=np.asarray(ds._mem.get("actions")).reshape(-1,2) if "actions" in ds._mem else None
if a is not None:
    z=(np.abs(a[:,0])<1)&(np.abs(a[:,1])<1)
    print(f"actions shape={a.shape} zero_frames={int(z.sum())}/{len(a)}")
