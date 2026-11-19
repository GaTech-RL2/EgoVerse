"""Stage one cell's episodes from R2 onto node-local disk for evaluation.

For clusters without the Pace copy (e.g. ICE). `<private_dir>/canonical_zarr`
should be a symlink to DEST, so the unchanged LocalEpisodeResolver sees the
staged episodes.

The stage inventory maps episode -> relative file -> [s3 url, md5, size]. It
must describe the exact bytes the checkpoints were trained and scored on: the
2026-11-07 R2 episode copies differ from Pace's canonical_zarr in
left.obs_wrist_pose, so non-image arrays come from a Pace overlay. Every staged
file is MD5-checked; equal sizes are not enough.

Usage: python stage_episodes.py <stage_inventory> <split_json> <dest> <index_dir>
Needs s5cmd on PATH and R2_ENDPOINT_URL plus R2 keys in AWS_* env vars.
"""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def bad_files(dest, inventory, episodes, check_md5):
    files = [
        (h, rel, meta) for h in episodes for rel, meta in sorted(inventory[h].items())
    ]

    def bad(item):
        h, rel, (_, digest, size) = item
        p = dest / f"{h}.zarr" / rel
        if not p.is_file() or p.stat().st_size != size:
            return True
        return check_md5 and md5(p) != digest

    with ThreadPoolExecutor(32) as pool:
        flags = list(pool.map(bad, files))
    return [(h, rel, meta[0]) for (h, rel, meta), f in zip(files, flags) if f]


def main(stage_inventory, split_json, dest, index_dir):
    dest, index_dir = Path(dest), Path(index_dir)
    inventory = json.loads(Path(stage_inventory).read_text())["episodes"]
    split = json.loads(Path(split_json).read_text())
    keys = ("train", "seen_matched", "seen_perlevel", "unseen")
    episodes = sorted(set().union(*(split[k] for k in keys)))
    absent = [h for h in episodes if h not in inventory]
    if absent:
        sys.exit(f"{len(absent)} episodes missing from inventory, e.g. {absent[:3]}")
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / ".stage.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = bad_files(dest, inventory, episodes, check_md5=False)
        size = sum(inventory[h][rel][2] for h, rel, _ in todo)
        print(
            json.dumps(
                {
                    "episodes": len(episodes),
                    "files_to_download": len(todo),
                    "gb": round(size / 1e9, 1),
                }
            ),
            flush=True,
        )
        for attempt in range(3):
            if todo:
                cmds = dest / f".stage-{os.getpid()}.txt"
                cmds.write_text(
                    "".join(
                        f'cp "{url}" "{dest}/{h}.zarr/{rel}"\n' for h, rel, url in todo
                    )
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
                    check=False,
                )
                cmds.unlink()
            todo = bad_files(dest, inventory, episodes, check_md5=True)
            print(json.dumps({"attempt": attempt, "bad_after": len(todo)}), flush=True)
            if not todo:
                break
        if todo:
            sys.exit(f"{len(todo)} files failed MD5 after staging, e.g. {todo[:3]}")
        for name in ("_episode_index.json", "_episode_index_all_available.json"):
            shutil.copyfile(index_dir / name, dest / name)
    print(json.dumps({"staged_md5_verified": len(episodes)}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
