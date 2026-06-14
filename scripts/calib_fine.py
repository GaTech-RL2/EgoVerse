"""Finer calibration of the small pusher radius. The 96px area is heavily
quantized below r~6 world. To squeeze a tighter bound, average the measured
red-pixel area over MANY sub-pixel disk CENTERS (the antialiased footprint
depends on where the true center lands within a 512->96 cell), giving a
smooth mean-area(radius) curve. Then compare to the small dataset's MEAN area.
"""
import os, glob
import numpy as np
import zarr, simplejpeg, cv2, pygame
from collections import deque

WORLD=512; IMG=96; PUSHER_COLOR=(210,60,60)

def render_disk(radius_world, cx, cy):
    surf=pygame.Surface((WORLD,WORLD)); surf.fill((240,240,240))
    # pygame.draw.circle takes int center; emulate sub-pixel by drawing on a
    # supersampled surface then INTER_AREA down -- matches env exactly (env draws
    # at int(px),int(py) on 512 then INTER_AREA to 96). So replicate int() too.
    pygame.draw.circle(surf,PUSHER_COLOR,(int(cx),int(cy)),int(radius_world))
    arr=pygame.surfarray.array3d(surf); rgb=np.transpose(arr,(1,0,2)).astype(np.uint8)
    return cv2.resize(rgb,(IMG,IMG),interpolation=cv2.INTER_AREA)

def red_area(img):
    R=img[...,0].astype(int);G=img[...,1].astype(int);B=img[...,2].astype(int)
    m=(R>150)&(G<120)&(B<120)&(R-G>60)&(R-B>60)
    return int(m.sum())

pygame.init()
print("=== FINE CALIBRATION: mean red-area over 64 sub-pixel centers ===")
centers=[(256+dx,256+dy) for dx in np.linspace(0,3.99,8) for dy in np.linspace(0,3.99,8)]
ref={}
for rw in [2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9,10,12,15]:
    areas=[red_area(render_disk(rw,cx,cy)) for cx,cy in centers]
    ref[rw]=np.mean(areas)
    print(f"  r_world={rw:5.2f}  mean_area_px={np.mean(areas):6.3f}  (min={min(areas)} max={max(areas)})")

rr=np.array(sorted(ref.keys())); aa=np.array([ref[r] for r in rr])
def invert(meas):
    return float(np.interp(meas, aa, rr))

def measure_ds(folder,n_eps=20,max_frames=40):
    eps=sorted(glob.glob(os.path.join(folder,"*.zarr")))[:n_eps]
    areas=[]
    for e in eps:
        raw=zarr.open(e,mode="r")["observations.images.front_img_1"][:]
        n=raw.shape[0]; idxs=np.linspace(5,n-5,max_frames).astype(int)
        for t in idxs:
            img=simplejpeg.decode_jpeg(raw[t],colorspace="RGB")
            a=red_area(img)
            if a>0: areas.append(a)
    return np.array(areas)

for label,folder in [("BIG(true 15.0)","/coc/flash7/paphiwetsa3/datasets/new_circle_3"),
                     ("SMALL","/coc/flash7/paphiwetsa3/datasets/new_circle_small__3")]:
    A=measure_ds(folder)
    print(f"\n=== {label} === n={len(A)} mean_area={A.mean():.3f} median={np.median(A):.1f}")
    print(f"  area histogram counts (px:count): "+", ".join(f"{v}:{c}" for v,c in zip(*np.unique(A,return_counts=True))))
    print(f"  -> radius via MEAN-area curve: {invert(A.mean()):.2f}")
