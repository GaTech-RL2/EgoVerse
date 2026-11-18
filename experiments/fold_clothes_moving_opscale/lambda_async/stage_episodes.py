"""Stage one cell's episodes from R2 onto node-local disk for evaluation.

For clusters without the Pace copy (e.g. ICE). `<private_dir>/canonical_zarr`
should be a symlink to DEST, so the unchanged LocalEpisodeResolver sees the
staged episodes. Every file is checked against the inventory's size; episodes
already complete on this node are skipped. A lock keeps concurrent jobs on one
node from staging the same episode twice.

Usage: python stage_episodes.py <private_dir> <cell> <dest> <index_dir>
Needs s5cmd on PATH and R2_ENDPOINT_URL plus R2 keys in AWS_* env vars.
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def missing(dest, inventory, episodes):
    out = []
    for h in episodes:
        for rel, meta in inventory[h].items():
            p = dest / f"{h}.zarr" / rel
            if not p.is_file() or p.stat().st_size != meta["size"]:
                out.append((h, rel))
    return out


def main(private_dir, cell, dest, index_dir):
    private_dir, dest, index_dir = Path(private_dir), Path(dest), Path(index_dir)
    split = json.loads(
        (private_dir / "splits" / f"fold_moving_{cell}.json").read_text()
    )
    inv = json.loads((private_dir / "destination_inventory.json").read_text())
    prefix, inventory = inv["s3_prefix"].rstrip("/"), inv["episodes"]
    keys = ("train", "seen_matched", "seen_perlevel", "unseen")
    episodes = sorted(set().union(*(split[k] for k in keys)))
    absent = [h for h in episodes if h not in inventory]
    if absent:
        sys.exit(f"{len(absent)} episodes missing from inventory, e.g. {absent[:3]}")
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / ".stage.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = sorted({h for h, _ in missing(dest, inventory, episodes)})
        size = sum(m["size"] for h in todo for m in inventory[h].values())
        print(
            json.dumps(
                {
                    "cell": cell,
                    "episodes": len(episodes),
                    "to_download": len(todo),
                    "gb": round(size / 1e9, 1),
                }
            ),
            flush=True,
        )
        if todo:
            cmds = dest / f".stage-{os.getpid()}.txt"
            cmds.write_text(
                "".join(f'cp "{prefix}/{h}.zarr/*" "{dest}/{h}.zarr/"\n' for h in todo)
            )
            subprocess.run(
                [
                    "s5cmd",
                    "--endpoint-url",
                    os.environ["R2_ENDPOINT_URL"],
                    "--log",
                    "error",
                    "--numworkers",
                    "128",
                    "run",
                    str(cmds),
                ],
                check=True,
            )
            cmds.unlink()
        bad = missing(dest, inventory, episodes)
        if bad:
            sys.exit(
                f"{len(bad)} files missing or wrong size after staging, e.g. {bad[:3]}"
            )
        for name in ("_episode_index.json", "_episode_index_all_available.json"):
            shutil.copyfile(index_dir / name, dest / name)
    print(json.dumps({"cell": cell, "staged": len(episodes)}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
