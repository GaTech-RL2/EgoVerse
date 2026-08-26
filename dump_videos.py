"""Dump mp4s from the stored frames of the respeed datasets (+ source baseline).

Reads observations.images.front_img_1 out of the zarr and writes it as-is, so
what you watch is literally the data that would be trained on. 96x96 is upscaled
4x (nearest) purely for viewing.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

ROOT = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2")
OUT = Path("/coc/flash7/scratch/paphiwetsa3/respeed/videos")
OUT.mkdir(parents=True, exist_ok=True)
SCALE, FPS = 4, 30

RES = Path("/coc/flash7/scratch/paphiwetsa3/respeed")
peaks = {}
for f in ("0.5", "1.5"):
    d = json.load(open(RES / ("respeed_%sx_pilot100.json" % f)))
    peaks[f] = [e["peak_coverage"] for e in d["episodes"]]


def eps(ds):
    return sorted(p for p in (ROOT / ds).iterdir() if p.name.endswith(".zarr"))


def dump(zpath, out_name, label):
    g = zarr.open_group(str(zpath), mode="r")
    md = dict(g.attrs)
    T = int(md.get("total_frames", 0)) or None
    raw = g["observations.images.front_img_1"][:T]
    imgs = []
    for b in raw:
        if b is None or len(b) == 0:
            continue
        # frames are JPEG-encoded bytes (object dtype), not a raw HWC array
        bgr = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            imgs.append(bgr)
    if not imgs:
        print("  %-34s NO DECODABLE FRAMES" % out_name)
        return
    h, w = imgs[0].shape[:2]
    vw = cv2.VideoWriter(str(OUT / out_name), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (w * SCALE, h * SCALE))
    for fr in imgs:
        big = cv2.resize(fr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(big)
    vw.release()
    print("  %-34s T=%-5d %s" % (out_name, len(imgs), label))


src = eps("circle_4500_plus_gen_v2")[:100]
e05 = eps("circle4500gen_v2_pusher0.5x_pilot100")
e15 = eps("circle4500gen_v2_pusher1.5x_pilot100")

# ep 0: clean case at all three speeds; ep 84 / 61: the failures
dump(src[0], "ep000_source_1.0x.mp4", "SOURCE 1.0x  peak 0.962")
dump(e05[0], "ep000_pusher_0.5x.mp4", "0.5x  peak %.3f" % peaks["0.5"][0])
dump(e15[0], "ep000_pusher_1.5x.mp4", "1.5x  peak %.3f" % peaks["1.5"][0])
dump(e15[84], "ep084_pusher_1.5x_WORST.mp4", "1.5x WORST  peak %.3f" % peaks["1.5"][84])
dump(src[84], "ep084_source_1.0x.mp4", "SOURCE 1.0x (same ep)")
dump(e05[61], "ep061_pusher_0.5x_fail.mp4", "0.5x fail  peak %.3f" % peaks["0.5"][61])
print("\nwrote to", OUT)
