"""Side-by-side 1.0x source vs 0.25x variant for a FIXED list of episodes.

Pairs on the initial state (agent_pos + object_pose + goal_pose + level), which
is unique per episode -- reset_seed is not. Frames are sampled at matched
PROGRESS so the 4x-longer clip lines up with the source.
"""
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import zarr

SRC = Path("/coc/flash7/paphiwetsa3/datasets/Tsim_v2/circle_4500_plus_gen_v2")
VAR = Path("/coc/flash7/scratch/paphiwetsa3/respeed/fails025")
OUT = Path("/coc/flash7/scratch/paphiwetsa3/respeed/sbs")
OUT.mkdir(parents=True, exist_ok=True)
N, SCALE = 260, 4

WANT = ["episode_T_circle_obs0_001940.zarr",
        "episode_T_circle_obs0_003072.zarr",
        "episode_T_circle_obs1_000007.zarr"]

PEAKS = {}
for f in glob.glob("/coc/flash7/scratch/paphiwetsa3/respeed/full025/shard*.json"):
    for e in json.load(open(f))["episodes"]:
        PEAKS[e["src"]] = e["peak_coverage"]


def init_key(z):
    i = json.loads(dict(z.attrs)["episode_init"])
    r = lambda v: tuple(round(float(x), 4) for x in v)
    return (int(i.get("obstacle_level", 0)), r(i["agent_pos"]),
            r(i["object_pose"]), r(i["goal_pose"]))


def frames(zpath):
    g = zarr.open_group(str(zpath), mode="r")
    T = int(dict(g.attrs).get("total_frames", 0)) or None
    out = []
    for b in g["observations.images.front_img_1"][:T]:
        if b is None or len(b) == 0:
            continue
        im = cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


var_map = {}
for p in sorted(VAR.iterdir()):
    if p.name.endswith(".zarr"):
        var_map[init_key(zarr.open_group(str(p), mode="r"))] = p
print("variant episodes indexed: %d" % len(var_map))

for sname in WANT:
    sp = SRC / sname
    k = init_key(zarr.open_group(str(sp), mode="r"))
    vp = var_map.get(k)
    if vp is None:
        print("  NO MATCH for %s" % sname)
        continue

    sf, vf = frames(sp), frames(vp)
    S = [sf[i] for i in np.clip((np.linspace(0, 1, N) * (len(sf) - 1)).round().astype(int), 0, len(sf) - 1)]
    V = [vf[i] for i in np.clip((np.linspace(0, 1, N) * (len(vf) - 1)).round().astype(int), 0, len(vf) - 1)]

    h, w = S[0].shape[:2]
    H, W = h * SCALE, w * SCALE
    gap, bar = 8, 34
    cw, ch = W * 2 + gap, H + bar
    out = OUT / ("sbs_%s.mp4" % sname.replace(".zarr", ""))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (cw, ch))
    ll = "1.0x SOURCE   %d frames" % len(sf)
    lr = "0.25x   peak %.3f   %d frames" % (PEAKS.get(sname, float("nan")), len(vf))
    for i in range(N):
        c = np.full((ch, cw, 3), 20, np.uint8)
        c[bar:bar + H, 0:W] = cv2.resize(S[i], (W, H), interpolation=cv2.INTER_NEAREST)
        c[bar:bar + H, W + gap:] = cv2.resize(V[i], (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, ll, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(c, lr, (W + gap + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 170, 245), 1, cv2.LINE_AA)
        cv2.putText(c, "obs%d    t = %d%%" % (k[0], int(100 * i / (N - 1))),
                    (cw // 2 - 58, ch - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(c)
    vw.release()
    print("  %-44s src=%-5d var=%-5d peak=%.3f -> %s"
          % (sname, len(sf), len(vf), PEAKS.get(sname, -1), out.name))
