import sys, zarr, numpy as np
def stats(p):
    try:
        g = zarr.open(p, mode="r")
        a = np.asarray(g["actions"][:])
        s = np.asarray(g["observations.state"][:])
        return f"actions shape={a.shape} min={a.min():.3f} max={a.max():.3f} mean={a.mean():.3f} | state min={s.min():.3f} max={s.max():.3f}"
    except Exception as e:
        return f"ERR {e}"
for label, p in [
    ("NORMALIZED 950 src", sys.argv[1]),
    ("RAW orig untouched", sys.argv[2]),
    ("NEW 400 batch raw ", sys.argv[3]),
]:
    print(label, "->", stats(p))
