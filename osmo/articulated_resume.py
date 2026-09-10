"""Resume this run's contact jobs after a verified controller/uploader fix.

Executed inside an owned collection container. The original coordinator stays
paused until its replacement has persisted every episode. Atomic committed
episodes survive; only unfinished staging directories are discarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path("/workspace/demos")
REPO = Path("/workspace/EgoVerse")


def process_table():
    result = {}
    for f in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = f.read_bytes().split(b"\0")
            status = (f.parent / "status").read_text()
            parent = int(
                next(x.split()[1] for x in status.splitlines() if x.startswith("PPid:"))
            )
            result[int(f.parent.name)] = (parent, argv)
        except (OSError, ValueError, StopIteration):
            pass
    return result


def main():
    marker = ROOT / "controller_resume.json"
    if marker.exists():
        raise RuntimeError(
            "This container was already resumed; inspect its recorded status"
        )
    processes = process_table()
    matching = {
        pid: (parent, argv)
        for pid, (parent, argv) in processes.items()
        if b"Tsimulation.sim_v2.collect.articulation_batch" in argv
    }
    roots = [
        (pid, argv)
        for pid, (parent, argv) in matching.items()
        if parent not in matching
    ]
    assert len(roots) == 1, roots
    pid, argv = roots[0]
    argv = [x.decode() for x in argv if x]
    assert argv[argv.index("--out") + 1] == str(ROOT)
    emb = argv[argv.index("--embodiments") + 1]
    assert emb in ("triangle", "flipper"), emb
    prefix = argv[argv.index("--s3-prefix") + 1]
    assert prefix == "staged/pushshapes_articulated/articulated-20260909/" + emb
    os.kill(pid, signal.SIGSTOP)
    # Stop only our old upload helper, which would otherwise race the new one.
    for other, (_, args) in processes.items():
        if b"/workspace/articulated_upload_recovery.py" in args:
            os.kill(other, signal.SIGTERM)
    workers = [child for child in matching if child != pid]
    for child in workers:
        os.kill(child, signal.SIGTERM)
    time.sleep(2)
    for child in workers:
        status = Path("/proc") / str(child) / "status"
        try:
            state = next(
                x for x in status.read_text().splitlines() if x.startswith("State:")
            )
            if "Z (zombie)" not in state:
                os.kill(child, signal.SIGKILL)
        except (ProcessLookupError, FileNotFoundError):
            pass
    time.sleep(1)
    original = Path("/workspace/source.tar.gz")
    with original.open("rb") as f:
        original_sha = hashlib.file_digest(f, "sha256").hexdigest()
    patches = json.loads(Path("/workspace/controller_patches.json").read_text())
    allowed = {
        "Tsimulation/sim_v2/collect/contact_controller.py",
        "Tsimulation/sim_v2/collect/articulation_batch.py",
    }
    assert set(patches) == allowed
    for name, content in patches.items():
        (REPO / name).write_text(content)
    updated = Path("/workspace/source_after_resume.tar.gz")
    with tarfile.open(original, "r:gz") as old:
        names = [m.name for m in old.getmembers() if m.isfile()]
    with tarfile.open(updated, "w:gz") as archive:
        for name in sorted(names):
            archive.add(REPO / name, arcname=name, recursive=False)
    with updated.open("rb") as f:
        updated_sha = hashlib.file_digest(f, "sha256").hexdigest()
    boundary = {}
    for folder in ROOT.glob("*/*/shard*"):
        if not folder.is_dir():
            continue
        for pending in folder.glob(".pending-*.zarr"):
            shutil.rmtree(pending)
        existing = sorted(folder.glob("episode_*.zarr"))
        last = max((int(p.stem.rsplit("_", 1)[1]) for p in existing), default=-1)
        record = dict(
            original_source_sha256=original_sha,
            resumed_source_sha256=updated_sha,
            original_last_episode_index=last,
            original_episodes=len(existing),
        )
        (folder / "generation_sources.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        boundary[str(folder.relative_to(ROOT))] = record
    record = dict(
        original_pid=pid,
        embodiment=emb,
        prefix=prefix,
        started_at=time.time(),
        original_source_sha256=original_sha,
        resumed_source_sha256=updated_sha,
        preserved_episodes=sum(v["original_episodes"] for v in boundary.values()),
        source_boundaries=boundary,
        status="resuming",
    )
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print("PRESERVED_EPISODES", record["preserved_episodes"], flush=True)
    import sys

    sys.path.insert(0, str(REPO))
    from Tsimulation.sim_v2.collect.articulation_batch import r2_client

    client = r2_client()
    for capsule, digest in ((original, original_sha), (updated, updated_sha)):
        client.upload_file(
            str(capsule), "rldb", prefix + "/sources/" + digest + ".tar.gz"
        )
    client.upload_file(str(updated), "rldb", prefix + "/source.tar.gz")
    client.upload_file(str(marker), "rldb", prefix + "/controller_resume.json")
    client.upload_file(__file__, "rldb", prefix + "/controller_resume.py")
    for f in Path("/workspace/preflight").glob("*.json"):
        client.upload_file(str(f), "rldb", prefix + "/preflight/" + f.name)
    # A new interpreter loads the patched controller; the simulator is unchanged.
    argv[0] = str(REPO / "emimic/bin/python")
    with open("/workspace/resumed_collection.log", "a") as log:
        result = subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    record["exit_code"] = result.returncode
    record["finished_at"] = time.time()
    if result.returncode:
        record["status"] = "needs_attention"
        marker.write_text(json.dumps(record, indent=2) + "\n")
        client.upload_file(str(marker), "rldb", prefix + "/controller_resume.json")
        raise RuntimeError(
            "Resumed collector needs attention; original coordinator remains paused"
        )
    summary = json.loads((ROOT / "batch_result.json").read_text())
    assert summary["complete"] and summary["episodes"] == 18000
    record["status"] = "complete"
    marker.write_text(json.dumps(record, indent=2) + "\n")
    client.upload_file(str(marker), "rldb", prefix + "/controller_resume.json")
    summary.update(
        controller_resumed=True,
        source_capsule_sha256=updated_sha,
        original_source_capsule_sha256=original_sha,
    )
    (ROOT / "batch_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    client.upload_file(
        str(ROOT / "batch_result.json"), "rldb", prefix + "/batch_result.json"
    )
    print("ALL_EPISODES_DURABLE", 18000, prefix, flush=True)
    os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
