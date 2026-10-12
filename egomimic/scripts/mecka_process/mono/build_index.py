"""Write zarr/_episode_index.json ({episode_hash: attrs}) for LocalEpisodeResolver
from the master JSON, the manifest and the conversion logs (plus one directory
listing), so training never opens every episode on NFS to filter it.

  python -m egomimic.scripts.mecka_process.mono.build_index <master.json>
"""

import csv
import glob
import json
import os
import sys

from egomimic.scripts.mecka_process.mono.private_ids import private_dir

D = private_dir()
QA = (
    "glitch_frames",
    "max_cam_step_m",
    "max_cam_rot_deg",
    "frozen_cam_frames",
    "max_wrist_step_m",
    "zero_hand_frames",
)

master = {e["episode_id"]: e for e in json.load(open(sys.argv[1]))["episodes"]}
manifest = {r["episode_id"]: r for r in csv.DictReader(open(D / "manifest.csv"))}
logged = {}
for f in sorted(glob.glob(str(D / "zarr_logs" / "convert_log*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r.get("ok") and "frames" in r:
            logged[r["episode_id"]] = r
present = {
    n.removesuffix(".zarr") for n in os.listdir(D / "zarr") if n.endswith(".zarr")
}

index = {}
for eid in sorted(present & set(logged)):
    e, m, r = master[eid], manifest[eid], logged[eid]
    index[eid] = {
        "embodiment": "human_bimanual",
        "total_frames": r["frames"],
        "fps": 30,
        "task_name": m["category"],
        "task_description": e["task_description"],
        **{
            k: e[k]
            for k in (
                "episode_id",
                "user_id",
                "duration",
                "environment_id",
                "scene_id",
                "task_id",
            )
        },
        "task": m["category"],
        "lab": "mecka",
        "source": "mecka_mono",
        "qa": {k: r[k] for k in QA},
    }
out = D / "zarr" / "_episode_index.json"
tmp = out.with_suffix(".tmp")
tmp.write_text(json.dumps(index))
tmp.replace(out)
print(
    f"{len(index)} episodes indexed; {len(present - set(logged))} zarrs without a log row, "
    f"{len(set(logged) - present)} logged but missing"
)
