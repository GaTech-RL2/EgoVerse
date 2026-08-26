"""Confirm respeed output stores PUSHER poses as actions (honouring total_frames)."""
import json
import sys
from pathlib import Path

import numpy as np
import zarr

D = Path(sys.argv[1])
for ep in sorted(p for p in D.iterdir() if p.name.endswith(".zarr"))[:2]:
    g = zarr.open_group(str(ep), mode="r")
    md = dict(g.attrs)
    T = int(md["total_frames"])
    a = np.asarray(g["actions"][:T])
    cmd = np.asarray(g["observations.pusher_cmd_pose"][:T])
    st = np.asarray(g["observations.state"][:T])

    n = T - 1
    d_next = np.abs(a[:n, :2] - st[1:n + 1, :2]).max()
    d_cursor = np.abs(a[:T, :2] - cmd[:T, :2]).max()
    d_pre = np.abs(a[:T, :2] - st[:T, :2]).max()

    print("%s  T=%d" % (ep.name, T))
    print("   |action - next_state_pusher| max = %.6f   (~0 => actions ARE pusher poses)" % d_next)
    print("   |action - cursor_cmd|        max = %.4f   (>0 => not the cursor)" % d_cursor)
    print("   |action - pre_state_pusher|  max = %.4f   (= per-step travel cap)" % d_pre)
    print("   speed=%s action_space=%s compensated=%s"
          % (md.get("speed_factor"), md.get("action_space"), md.get("time_compensated")))
