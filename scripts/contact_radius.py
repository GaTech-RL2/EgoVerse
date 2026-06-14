"""Physics-based pusher-radius determination, resolution-independent.

The pusher is a rigid disk that cannot penetrate the object. At a firm-contact
frame, the distance from the pusher CENTER to the object's polygon SURFACE
equals the pusher radius (disk tangent to the object boundary). Across an
episode the MINIMUM such center-to-surface distance is a tight lower bound that
the pusher actually attains while pushing -> ~= the true radius.

We have: state[0:2]=pusher center (agent_pos), state[2:5]=object (x,y,theta).
Object geometry = T-shape rects (from shapes.py SHAPES["T"]). Build the object
world polygon at each frame, compute shapely distance(point, polygon_boundary),
take a low percentile over the episode (the firm-contact floor).

Validate on BIG (true radius 15.0) then report SMALL.
"""
import os, glob
import numpy as np
import zarr
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate, translate

# gym-pusht T (from Tsimulation/pushshapes/shapes.py): rects (cx,cy,w,h) body-local
T_RECTS = [(0.0, -30.0, 120.0, 30.0), (0.0, 30.0, 30.0, 90.0)]

def object_polygon(x, y, theta):
    polys = []
    for cx, cy, w, h in T_RECTS:
        hw, hh = w/2.0, h/2.0
        p = Polygon([(cx-hw, cy-hh), (cx+hw, cy-hh), (cx+hw, cy+hh), (cx-hw, cy+hh)])
        polys.append(p)
    body = unary_union(polys)
    body = rotate(body, theta, origin=(0, 0), use_radians=True)
    body = translate(body, xoff=x, yoff=y)
    return body

def episode_min_gap(path):
    g = zarr.open(path, mode="r")
    st = np.asarray(g["observations.state"][:], dtype=np.float64)  # (T,5)
    gaps = []
    for t in range(st.shape[0]):
        px, py, ox, oy, oth = st[t]
        if px == 0 and py == 0:  # off-screen garbage (data-quality note)
            continue
        poly = object_polygon(ox, oy, oth)
        pt = Point(px, py)
        if poly.contains(pt):
            d = -poly.boundary.distance(pt)  # center inside (penetration) -> negative
        else:
            d = poly.boundary.distance(pt)   # center outside -> dist to surface
        gaps.append(d)
    return np.array(gaps)

def dataset_radius(folder, n_eps=40):
    eps = sorted(glob.glob(os.path.join(folder, "*.zarr")))[:n_eps]
    mins = []; all_gaps = []
    for e in eps:
        gp = episode_min_gap(e)
        if len(gp) == 0:
            continue
        # firm-contact floor: the smallest positive gap the pusher attains
        pos = gp[gp > 0]
        if len(pos):
            mins.append(np.percentile(pos, 1))  # 1st percentile of positive gaps
        all_gaps.append(gp)
    mins = np.array(mins)
    allg = np.concatenate(all_gaps)
    return mins, allg

for label, folder in [("BIG (true r=15.0)", "/coc/flash7/paphiwetsa3/datasets/new_circle_3"),
                      ("SMALL", "/coc/flash7/paphiwetsa3/datasets/new_circle_small__3")]:
    mins, allg = dataset_radius(folder)
    print(f"\n===== {label} =====  n_eps_used={len(mins)}  n_frames={len(allg)}")
    print(f"  per-episode 1pct-positive-gap: median={np.median(mins):.3f}  min={mins.min():.3f}  mean={mins.mean():.3f}")
    pos = allg[allg > 0]
    print(f"  global positive-gap percentiles 0.5/1/2/5: {np.percentile(pos,[0.5,1,2,5])}")
    print(f"  global gap min (incl. penetration): {allg.min():.3f}   frac penetrating(<0): {(allg<0).mean():.3f}")
