"""Relocate only completed ExIt weight shards, preserving old paths and exact bytes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    old = Path("/mnt/amlfs-07/shared/egoverse/rpunamiya/leact")
    study = old / "studies/concise-reproduction-v1"
    new = Path("/mnt/amlfs-03/home/rpunamiya/leact/offloaded-completed-exit")
    if not (study / "exit/complete.json").exists() or not os.path.ismount("/mnt/amlfs-03/home/rpunamiya"):
        raise RuntimeError("Completed-role proof and the personal mount are required")
    new.mkdir(parents=True, exist_ok=True)
    audit = new / "offload.json"
    rows = json.loads(audit.read_text())["files"] if audit.exists() else []
    for source in sorted((study / "exit/exit").glob("round-*/sft/checkpoints/**/*.safetensors")):
        target = new / source.relative_to(study / "exit/exit")
        if source.is_symlink():
            if source.resolve() != target.resolve() or not target.is_file():
                raise RuntimeError("Unexpected preexisting checkpoint symlink")
            continue
        stat = source.stat()
        if shutil.disk_usage(new).free < stat.st_size + 5 * 1024**3:
            raise RuntimeError("Not enough personal storage")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(source) != digest(target):
                raise RuntimeError("Preexisting offload content differs")
            sha = digest(target)
        else:
            temporary = target.with_suffix(".partial")
            if temporary.exists():
                raise RuntimeError("Interrupted copy needs inspection")
            h = hashlib.sha256()
            with source.open("rb") as incoming, temporary.open("xb") as outgoing:
                for chunk in iter(lambda: incoming.read(16 * 1024**2), b""):
                    h.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            sha = h.hexdigest()
            now = source.stat()
            if digest(temporary) != sha or (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
                raise RuntimeError("Checkpoint copy failed verification or source changed")
            shutil.copystat(source, temporary)
            temporary.rename(target)
        link = source.with_suffix(".offload-link")
        link.symlink_to(target)
        link.replace(source)  # Exact bytes already fsynced and checksum-verified on the personal mount.
        rows.append({"source": str(source), "destination": str(target), "sha256": sha,
                     "bytes": stat.st_size, "completed_unix": time.time()})
        temporary_audit = audit.with_suffix(".tmp")
        temporary_audit.write_text(json.dumps({"files": rows, "old_paths_preserved": True,
                                              "completed_role": str(study / "exit/complete.json")}, indent=2) + "\n")
        temporary_audit.replace(audit)
        print(json.dumps({"offloaded_bytes": stat.st_size, "source": str(source),
                          "free_on_shared": shutil.disk_usage(old).free}), flush=True)


if __name__ == "__main__":
    main()
