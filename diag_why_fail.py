"""Is the 1.5x failure a tracking bug or physics?

For each episode compare, at matched progress (t/T):
  * PUSHER path  variant vs source  -- tests whether my time-scaling makes the
    pusher retrace the recorded path. If this is ~0, the compensation works.
  * OBJECT path  variant vs source  -- if the pusher matches but the object does
    not, the divergence is momentum/friction, i.e. the task is not time-scale
    invariant, not a code error.
state layout (circle pusher): [pusher_x, pusher_y, obj_x, obj_y, obj_theta]
"""
import json
from pathlib import Path

import numpy as np
import zarr

R = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2")
SRC = R / "circle_4500_plus_gen_v2"
RES = Path("/coc/flash7/scratch/paphiwetsa3/respeed")


def states(p):
    g = zarr.open_group(str(p), mode="r")
    T = int(dict(g.attrs).get("total_frames", 0)) or None
    return np.asarray(g["observations.state"][:T])


def at_progress(a, n=100):
    """Resample rows of `a` at n evenly spaced progress fractions."""
    idx = np.clip((np.linspace(0, 1, n) * (len(a) - 1)).round().astype(int),
                  0, len(a) - 1)
    return a[idx]


src_eps = sorted(p for p in SRC.iterdir() if p.name.endswith(".zarr"))[:100]

for fac, ds in (("0.5", "circle4500gen_v2_pusher0.5x_pilot100"),
                ("1.5", "circle4500gen_v2_pusher1.5x_pilot100")):
    var_eps = sorted(p for p in (R / ds).iterdir() if p.name.endswith(".zarr"))
    peaks = [e["peak_coverage"] for e in
             json.load(open(RES / ("respeed_%sx_pilot100.json" % fac)))["episodes"]]

    push_d, obj_d, rows = [], [], []
    for i, (s, v) in enumerate(zip(src_eps, var_eps)):
        S, V = at_progress(states(s)), at_progress(states(v))
        pd = np.linalg.norm(S[:, :2] - V[:, :2], axis=1)      # pusher xy
        od = np.linalg.norm(S[:, 2:4] - V[:, 2:4], axis=1)    # object xy
        push_d.append(pd.mean())
        obj_d.append(od.mean())
        rows.append((i, peaks[i], pd.mean(), od.mean()))

    push_d, obj_d = np.array(push_d), np.array(obj_d)
    pk = np.array(peaks)
    good, bad = pk >= 0.95, pk < 0.95

    print("=" * 74)
    print("%sx  (n=%d)   arena is 512 units wide" % (fac, len(pk)))
    print("  mean pusher-path deviation vs source : %7.2f units" % push_d.mean())
    print("  mean object-path deviation vs source : %7.2f units" % obj_d.mean())
    if bad.any() and good.any():
        print("  --- split by outcome ---")
        print("  pass (peak>=0.95, n=%3d): pusher %6.2f   object %6.2f"
              % (good.sum(), push_d[good].mean(), obj_d[good].mean()))
        print("  fail (peak< 0.95, n=%3d): pusher %6.2f   object %6.2f"
              % (bad.sum(), push_d[bad].mean(), obj_d[bad].mean()))
    worst = sorted(rows, key=lambda r: r[1])[:3]
    for i, pkv, pdv, odv in worst:
        print("  worst ep%03d peak=%.3f  pusher_dev=%6.2f  object_dev=%6.2f"
              % (i, pkv, pdv, odv))
