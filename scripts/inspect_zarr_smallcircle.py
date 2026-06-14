import sys, json
import zarr
import numpy as np

def dump(path, label):
    print(f"\n========== {label}: {path} ==========")
    g = zarr.open(path, mode="r")
    print("--- root attrs ---")
    print(json.dumps(dict(g.attrs), indent=2, default=str))
    print("--- arrays/groups ---")
    def walk(grp, prefix=""):
        for k in grp.keys():
            item = grp[k]
            if isinstance(item, zarr.hierarchy.Group):
                print(f"{prefix}{k}/ (group) attrs={dict(item.attrs)}")
                walk(item, prefix + k + "/")
            else:
                print(f"{prefix}{k}  shape={item.shape} dtype={item.dtype} attrs={dict(item.attrs)}")
    walk(g)

for path, label in [
    (sys.argv[1], "SMALL"),
    (sys.argv[2], "BIG"),
]:
    try:
        dump(path, label)
    except Exception as e:
        import traceback
        print(f"ERR {label}: {e}\n{traceback.format_exc()}")
