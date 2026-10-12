"""Per-episode camera-tracking noise from every raw hand_pose.json -> scaling/cam_jitter.csv.

python -m egomimic.scripts.mecka_process.mono.cam_jitter [workers]
"""

import csv
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.scripts.mecka_process.mono.private_ids import private_dir

D = private_dir()


def one(row):
    h = json.load(open(D / "raw" / row["episode_id"] / "hand_pose.json"))
    fr = h["frames"]
    ks = sorted(fr, key=int)
    t = np.array(
        [[fr[k]["camera_transform"]["position"][a] for a in "xyz"] for k in ks]
    )
    q = np.array(
        [[fr[k]["camera_transform"]["orientation"][a] for a in "xyzw"] for k in ks]
    )
    step = np.linalg.norm(np.diff(t, axis=0), axis=1) * 100
    rot = np.degrees((R.from_quat(q[1:]) * R.from_quat(q[:-1]).inv()).magnitude())
    # second difference: frame-to-frame jitter, small for smooth head motion however fast
    acc = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=1) * 100
    return {
        **{k: row[k] for k in ("episode_id", "user_id", "category", "off_task")},
        "frames": len(t),
        "glitch_frames": int(((step > 5) | (rot > 10)).sum()),
        "step_med_cm": float(np.median(step)),
        "rot_med_deg": float(np.median(rot)),
        "acc_med_cm": float(np.median(acc)),
        "acc_p99_cm": float(np.percentile(acc, 99)),
    }


if __name__ == "__main__":
    rows = list(csv.DictReader(open(D / "manifest.csv")))
    rows = [
        r for r in rows if (D / "raw" / r["episode_id"] / "hand_pose.json").exists()
    ]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as p:
        out = p.map(one, rows, chunksize=8)
    with open(D / "scaling" / "cam_jitter.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out))
