"""Build eva_episode_objects.json + the anchored D0 lists from the annotations.

Single source of truth for object parsing is ``validate_object_labels`` (PICK /
canon / canonical_segments) so the committed artifacts always match what the
validator considers clean. Re-run after changing ``canon()`` there.

    python -m egomimic.ricl.scripts.build_episode_objects \
        [--zarr-root '/storage/project/r-dxu345-0/agao81/pick_place/{hash}'] \
        [--anchor-object 'blue marker'] [--anchor-scene pick_place_diverse]
"""

from __future__ import annotations

import argparse
import json
import os

import zarr

from egomimic.ricl.scripts.validate_object_labels import PICK, canon, canonical_segments
from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

# episode_lists/ lives in the parent ricl/ dir, one level up from scripts/.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "episode_lists")


def episode_objects(store) -> tuple[list[str], int]:
    objs, npick = [], 0
    if "annotations" in store and store["annotations"].shape[0] > 0:
        for seg in canonical_segments(store):
            m = PICK.match(seg.strip())
            if m:
                objs.append(canon(m.group(1)))
                npick += 1
    return sorted(set(objs)), npick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zarr-root", default="/storage/project/r-dxu345-0/agao81/pick_place/{hash}"
    )
    ap.add_argument("--anchor-object", default="blue marker")
    ap.add_argument("--anchor-scene", default="pick_place_diverse")
    args = ap.parse_args()

    df = episode_table_to_df(create_default_engine())
    df["episode_hash"] = df["episode_hash"].astype(str)
    eva = df[(df["task"] == "pick_place") & (df["embodiment"] == "eva_bimanual")]

    records = {}
    for r in eva.itertuples():
        path = args.zarr_root.format(hash=r.episode_hash)
        if not os.path.isdir(path):
            continue
        objs, npick = episode_objects(zarr.open(path, mode="r"))
        records[r.episode_hash] = {
            "scene": r.scene,
            "task_description": r.task_description,
            "objects": objs,
            "num_picks": npick,
        }
    records = dict(sorted(records.items()))

    master = os.path.join(LISTS, "eva_episode_objects.json")
    with open(master, "w") as f:
        json.dump(
            {
                "anchor": {"scene": args.anchor_scene, "object": args.anchor_object},
                "episodes": records,
            },
            f,
            indent=2,
        )
    print(f"wrote {master}  ({len(records)} eva episodes)")

    obj, scn = args.anchor_object, args.anchor_scene
    lists = {
        "d0_anchor_1_same_scene_same_object": [
            h for h, v in records.items() if v["scene"] == scn and obj in v["objects"]
        ],
        "d0_anchor_2_same_scene": [h for h, v in records.items() if v["scene"] == scn],
        "d0_anchor_3_same_object": [
            h for h, v in records.items() if obj in v["objects"]
        ],
    }
    for name, lst in lists.items():
        path = os.path.join(LISTS, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(sorted(lst)) + "\n")
        print(f"wrote {path}  ({len(lst)})")


if __name__ == "__main__":
    main()
