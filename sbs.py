"""Side-by-side: 1.0x source vs 0.25x variant, for the worst failures.

The 0.25x episode is 4x longer, so frames are sampled at matched PROGRESS
(t/T) rather than matched index -- otherwise the slow clip would merely look
"behind" instead of showing where the outcome differs.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import zarr

SRC = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2")
VAR = Path("/coc/flash7/scratch/paphiwetsa3/respeed/fails025")
OUT = Path("/coc/flash7/scratch/paphiwetsa3/respeed/sbs")
OUT.mkdir(parents=True, exist_ok=True)
N, SCALE = 260, 4

PEAKS = {}
import glob
for f in glob.glob("/coc/flash7/scratch/paphiwetsa3/respeed/full025/shard*.json"):
    for e in json.load(open(f))["episodes"]:
        PEAKS[e["src"]] = e["peak_coverage"]


def frames(zpath):
    g = zarr.open_group(str(zpath), mode="r")
    T = int(dict(g.attrs).get("total_frames", 0)) or None
    raw = g["observations.images.front_img_1"][:T]
    out = []
    for b in raw:
        if b is None or len(b) == 0:
            continue
        im = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def at_progress(fr, n):
    idx = np.clip((np.linspace(0, 1, n) * (len(fr) - 1)).round().astype(int),
                  0, len(fr) - 1)
    return [fr[i] for i in idx]


# map variant episodes by (obs level, reset_seed) so we pair the right ones
var_by_key = {}
for p in sorted(VAR.iterdir()):
    if not p.name.endswith(".zarr"):
        continue
    a = dict(zarr.open_group(str(p), mode="r").attrs)
    ini = json.loads(a["episode_init"])
    var_by_key[(int(ini.get("obstacle_level", 0)), ini.get("reset_seed"))] = p

for sname in sorted(PEAKS):
    sp = SRC / sname
    if not sp.exists():
        continue
    a = dict(zarr.open_group(str(sp), mode="r").attrs)
    ini = json.loads(a["episode_init"])
    key = (int(ini.get("obstacle_level", 0)), ini.get("reset_seed"))
    vp = var_by_key.get(key)
    if vp is None:
        continue

    sf, vf = frames(sp), frames(vp)
    if not sf or not vf:
        continue
    S, V = at_progress(sf, N), at_progress(vf, N)
    h, w = S[0].shape[:2]
    H, W = h * SCALE, w * SCALE
    gap, bar = 8, 34
    canvas_w, canvas_h = W * 2 + gap, H + bar

    out = OUT / ("sbs_%s.mp4" % sname.replace(".zarr", ""))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                         (canvas_w, canvas_h))
    lvl = key[0]
    lab_l = "1.0x SOURCE  (%d frames)" % len(sf)
    lab_r = "0.25x  peak %.3f  (%d frames)" % (PEAKS[sname], len(vf))
    for i, (a_, b_) in enumerate(zip(S, V)):
        cv = np.full((canvas_h, canvas_w, 3), 20, np.uint8)
        cv[bar:bar + H, 0:W] = cv2.resize(a_, (W, H), interpolation=cv2.INTER_NEAREST)
        cv[bar:bar + H, W + gap:] = cv2.resize(b_, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(cv, lab_l, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(cv, lab_r, (W + gap + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (110, 170, 245), 1, cv2.LINE_AA)
        cv2.putText(cv, "obs%d   t=%d%%" % (lvl, int(100 * i / (N - 1))),
                    (canvas_w // 2 - 52, canvas_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(cv)
    vw.release()
    print("  %-46s src=%-5d var=%-5d peak=%.3f -> %s"
          % (sname, len(sf), len(vf), PEAKS[sname], out.name))
