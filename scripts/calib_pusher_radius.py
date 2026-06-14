"""Definitively determine the small-circle pusher radius by CALIBRATION.

1. Render reference red disks at known world radii through the EXACT same
   pygame-draw + cv2.INTER_AREA downscale (512->96) pipeline the env uses, and
   measure each disk's red-pixel area with a fixed red mask. -> area(radius) curve.
2. Measure the red-pixel area of the actual pusher in the BIG dataset (true
   radius 15.0, known from PUSHER_RADIUS) and the SMALL dataset.
3. Invert the calibration curve: small_radius_world = f^{-1}(area_small).
   Cross-check: the big dataset's measured area should invert to ~15.0 (sanity).

We only use frames where the red blob is a single compact component (pusher not
touching the goal/object, nothing else is red), measured at the disk's max
extent over many frames (extent is more stable than area at small sizes, but we
report both).
"""
import os, glob
import numpy as np
import zarr, simplejpeg, cv2, pygame
from collections import deque

WORLD = 512
IMG = 96
PUSHER_COLOR = (210, 60, 60)

def render_disk(radius_world):
    surf = pygame.Surface((WORLD, WORLD))
    surf.fill((240, 240, 240))
    pygame.draw.circle(surf, PUSHER_COLOR, (WORLD // 2, WORLD // 2), int(radius_world))
    arr = pygame.surfarray.array3d(surf)
    rgb = np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    small = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return small

def red_mask(img):
    R = img[..., 0].astype(np.int32); G = img[..., 1].astype(np.int32); B = img[..., 2].astype(np.int32)
    return (R > 150) & (G < 120) & (B < 120) & (R - G > 60) & (R - B > 60)

def largest_cc(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []; best_n = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                q = deque([(i, j)]); seen[i, j] = True; pts = []
                while q:
                    y, x = q.popleft(); pts.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; q.append((ny, nx))
                if len(pts) > best_n:
                    best_n = len(pts); best = pts
    return best

def disk_stats(img):
    m = red_mask(img)
    tot = int(m.sum())
    if tot == 0:
        return None
    cc = largest_cc(m)
    frac = len(cc) / tot
    ys = np.array([p[0] for p in cc]); xs = np.array([p[1] for p in cc])
    area = len(cc)
    ext = 0.5 * max(ys.max()-ys.min()+1, xs.max()-xs.min()+1)
    r_area = np.sqrt(area / np.pi)
    return dict(area=area, r_area=r_area, ext=ext, frac=frac, tot=tot)

print("pygame init...")
pygame.init()

# ---- 1. calibration curve ----
print("\n=== CALIBRATION (rendered reference disks) ===")
ref = {}
for rw in [4,5,6,7,8,9,10,11,12,13,14,15,16,18,20]:
    s = disk_stats(render_disk(rw))
    if s is None:
        print(f"  r_world={rw:5.1f}  -> no red pixels (too small), skip")
        continue
    ref[rw] = s
    print(f"  r_world={rw:5.1f}  area_px={s['area']:4d}  r_area_px={s['r_area']:.3f}  ext_px={s['ext']:.2f}")

ref_r = np.array(sorted(ref.keys()), dtype=float)
ref_area = np.array([ref[r]['area'] for r in ref_r])
ref_ext = np.array([ref[r]['ext'] for r in ref_r])

def invert(meas, xs_area):
    # monotone interpolation area->radius
    order = np.argsort(xs_area)
    xa = xs_area[order]; rr = ref_r[order]
    return float(np.interp(meas, xa, rr))
def invert_ext(meas):
    order = np.argsort(ref_ext)
    xe = ref_ext[order]; rr = ref_r[order]
    return float(np.interp(meas, xe, rr))

# ---- 2. measure datasets ----
def measure_ds(folder, n_eps=15, max_frames=30):
    eps = sorted(glob.glob(os.path.join(folder, "*.zarr")))[:n_eps]
    areas=[]; exts=[]
    for e in eps:
        g = zarr.open(e, mode="r")
        raw = g["observations.images.front_img_1"][:]  # object array of jpeg bytes
        n = raw.shape[0]
        idxs = np.linspace(5, n-5, max_frames).astype(int)
        for t in idxs:
            buf = raw[t]            # element == jpeg bytes (same as dataset's `for b in raw`)
            img = simplejpeg.decode_jpeg(buf, colorspace="RGB")
            s = disk_stats(img)
            if s is None: continue
            if s['frac'] < 0.9: continue   # single compact blob only
            areas.append(s['area']); exts.append(s['ext'])
    return np.array(areas), np.array(exts)

for label, folder in [
    ("BIG(circle, true r=15.0)", "/coc/flash7/paphiwetsa3/datasets/new_circle_3"),
    ("SMALL(circle_small)",      "/coc/flash7/paphiwetsa3/datasets/new_circle_small__3"),
]:
    A, E = measure_ds(folder)
    print(f"\n=== {label} ===  n_frames={len(A)}")
    if len(A)==0:
        print("  NO CLEAN FRAMES"); continue
    ma = float(np.median(A)); me = float(np.median(E))
    print(f"  median area_px={ma:.1f}  median ext_px={me:.2f}")
    print(f"  area percentiles 25/50/75: {np.percentile(A,[25,50,75])}")
    print(f"  ext  percentiles 25/50/75: {np.percentile(E,[25,50,75])}")
    print(f"  -> radius_world via AREA curve: {invert(ma, ref_area):.2f}")
    print(f"  -> radius_world via EXT  curve: {invert_ext(me):.2f}")
