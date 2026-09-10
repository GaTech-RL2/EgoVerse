"""Read progress from this run's dedicated OSMO collection containers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import shlex
import struct
import subprocess
import termios
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EMBODIMENTS = (
    "u_socket",
    "gripper",
    "chain_gripper",
    "suction",
    "umi",
    "triangle",
    "scoop",
    "flipper",
    "spring",
)
REMOTE = r"""
import json, pathlib, collections, os
root=pathlib.Path('/workspace/demos')
cells=collections.defaultdict(lambda: dict(kept=0,attempts=0,started=0,complete=0))
for p in root.glob('*/*/shard*/progress.json'):
    try: r=json.loads(p.read_text())
    except (OSError,ValueError): continue
    c=cells[r['gap']]
    c['kept']+=r['kept']; c['attempts']+=r['attempts']
    c['started']+=1; c['complete']+=int(r['complete'])
extras={}
for filename in ('upload_recovery.json','batch_progress.json','batch_result.json','controller_resume.json','checkpoint_complete.json','restored_checkpoint.json'):
    p=root/filename
    if p.exists():
        try: extras[filename]={k:v for k,v in json.loads(p.read_text()).items() if k not in ('shards','source_boundaries')}
        except (OSError,ValueError): pass
if 'upload_recovery.json' in extras:
    pid=extras['upload_recovery.json']['parent_pid']
    p=pathlib.Path('/proc')/str(pid)/'status'
    if p.exists(): extras['coordinator_state']=next(x for x in p.read_text().splitlines() if x.startswith('State:'))
log=pathlib.Path('/workspace/upload_recovery.log')
if log.exists():
    with log.open() as f: extras['recovery_log_tail']=list(collections.deque(f,maxlen=3))
for filename in ('controller_resume.log','resumed_collection.log','checkpoint.log'):
    p=pathlib.Path('/workspace')/filename
    if p.exists():
        with p.open() as f: extras[filename]=list(collections.deque(f,maxlen=3))
fs=os.statvfs('/workspace')
extras['free_gib']=round(fs.f_bavail*fs.f_frsize/2**30,1)
extras['configured_shards']=len(list(root.glob('*/*/shard*/collection.json')))
for filename in ('cpu.max','cpu.stat'):
    p=pathlib.Path('/sys/fs/cgroup')/filename
    if p.exists(): extras[filename]=p.read_text().strip()
print('INSPECTION_JSON '+json.dumps(dict(cells=dict(cells),**extras)),flush=True)
"""


def inspect(workflow):
    if not workflow.startswith("articulated-20260909-"):
        raise ValueError("Inspection is restricted to this collection run")
    entry = "bash -lc " + shlex.quote(
        "cd /workspace/EgoVerse && source emimic/bin/activate && python -c "
        + shlex.quote(REMOTE)
    )
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 200, 0, 0))
    process = subprocess.Popen(
        [
            "osmo",
            "workflow",
            "exec",
            workflow,
            "collect",
            "--entry",
            entry,
            "--connect-timeout",
            "30",
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    output = b""
    deadline = time.monotonic() + 40
    try:
        while time.monotonic() < deadline and process.poll() is None:
            ready, _, _ = select.select([master], [], [], 1.0)
            if ready:
                try:
                    output += os.read(master, 65536)
                except OSError:
                    break
            if b"INSPECTION_JSON " in output and output.endswith(b"\n"):
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        os.close(master)
    lines = output.decode(errors="replace").splitlines()
    for line in lines:
        if line.startswith("INSPECTION_JSON "):
            return dict(
                workflow=workflow, **json.loads(line[len("INSPECTION_JSON ") :])
            )
    return dict(workflow=workflow, error="; ".join(lines[-6:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--embodiments", nargs="+", choices=EMBODIMENTS, default=EMBODIMENTS
    )
    ap.add_argument(
        "--workflows", nargs="+", help="Explicit workflow IDs from this collection run"
    )
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    jobs = a.workflows or [
        "articulated-20260909-" + e.replace("_", "-") + "-1" for e in a.embodiments
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(inspect, jobs))
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
