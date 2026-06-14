"""Measure the pusher disk radius (in world units) from rendered frames of the
small-circle vs big-circle datasets, by isolating the red PUSHER_COLOR=(210,60,60)
pixels and estimating the disk radius. The big dataset's true radius is KNOWN
(15.0 world, hardcoded PUSHER_RADIUS), so we calibrate: a measured area->radius
relation on the big set anchors the small-set radius.

Strategy per frame:
  - decode jpeg -> (96,96,3) uint8 RGB
  - red mask: R high, G/B low, close to (210,60,60)
  - keep frames where the red blob is a single compact connected component
    (pusher not occluded/merged with anything red — nothing else is red)
  - radius estimate from area: r_px = sqrt(area_px / pi)
  - also a max-extent estimate (half of bounding-box max side)
  - convert px->world via 512/96
Aggregate median over many frames/episodes.
"""
import os, sys, json, glob
import numpy as np
import zarr
import simplejpeg

SCALE = 512.0 / 96.0  # world units per image pixel

def red_mask(img):
    R = img[..., 0].astype(np.int32); G = img[..., 1].astype(np.int32); B = img[..., 2].astype(np.int32)
    # PUSHER_COLOR (210,60,60). object is (60,100,200) blue, goal (60,180,90) green, bg (240,240,240).
    return (R > 150) & (G < 110) & (B < 110) & (R - G > 70) & (R - B > 70)

def largest_cc(mask):
    # simple flood-fill connected components (4-conn) without scipy
    from collections import deque
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None; best_n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                q = deque([(i, j)]); seen[i, j] = True; pts = []
                while q:
                    y, x = q.popleft(); pts.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(pts) > best_n:
                    best_n = len(pts); best = pts
    return best

def measure_episode(path, max_frames=40):
    g = zarr.open(path, mode="r")
    frames = g["observations.images.front_img_1"]
    n = frames.shape[0]
    idxs = np.linspace(0, n-1, min(max_frames, n)).astype(int)
    area_r = []; ext_r = []; ncomp = []
    for t in idxs:
        raw = frames[t]
        img = simplejpeg.decode_jpeg(bytes(raw.tobytes() if hasattr(raw, "tobytes") else raw)) if isinstance(raw, (bytes, bytearray)) or getattr(raw, "dtype", None)==np.uint8 and raw.ndim==1 else np.asarray(raw)
        if img.ndim == 1:
            img = simplejpeg.decode_jpeg(bytes(img.tobytes()))
        m = red_mask(img)
        if m.sum() == 0:
            continue
        cc = largest_cc(m)
        # fraction of red pixels in the largest component: if <0.85 the blob is
        # split/occluded -> skip (unreliable)
        frac = len(cc) / m.sum()
        if frac < 0.85:
            continue
        ys = np.array([p[0] for p in cc]); xs = np.array([p[1] for p in cc])
        area = len(cc)
        area_r.append(np.sqrt(area / np.pi))
        ext = 0.5 * max(ys.max()-ys.min()+1, xs.max()-xs.min()+1)
        ext_r.append(ext)
        ncomp.append(frac)
    return area_r, ext_r, ncomp

def measure_dataset(folder, n_eps=12, pattern="*.zarr"):
    eps = sorted(glob.glob(os.path.join(folder, pattern)))[:n_eps]
    A = []; E = []
    for e in eps:
        try:
            a, x, f = measure_episode(e)
            A += a; E += x
        except Exception as ex:
            print(f"  skip {os.path.basename(e)}: {ex}")
    A = np.array(A); E = np.array(E)
    return A, E

for label, folder, pat in [
    ("BIG(circle=15world)", "/coc/flash7/paphiwetsa3/datasets/new_circle_3", "*.zarr"),
    ("SMALL(circle_small)", "/coc/flash7/paphiwetsa3/datasets/new_circle_small__3", "*.zarr"),
]:
    A, E = measure_dataset(folder, n_eps=12)
    print(f"\n===== {label} =====  n_frames_used={len(A)}")
    if len(A):
        print(f"  area-radius px: median={np.median(A):.3f} mean={A.mean():.3f} std={A.std():.3f}  -> world median={np.median(A)*SCALE:.2f}")
        print(f"  extent-radius px: median={np.median(E):.3f} mean={E.mean():.3f}  -> world median={np.median(E)*SCALE:.2f}")
        print(f"  area-radius px percentiles 25/50/75: {np.percentile(A,[25,50,75])}")
