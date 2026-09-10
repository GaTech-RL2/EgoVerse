"""Start a durable checkpoint of this run's three slow collection jobs."""

import argparse
import base64
import fcntl
import os
import pty
import re
import select
import shlex
import struct
import subprocess
import termios
import time
import zlib
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("embodiment", choices=["scoop", "spring", "flipper"])
ap.add_argument("--workflow")
ap.add_argument("--checkpoint-suffix", default="")
a = ap.parse_args()
if a.checkpoint_suffix and not re.fullmatch(r"[a-z0-9-]+", a.checkpoint_suffix):
    ap.error("invalid checkpoint suffix")
source = base64.b64encode(
    zlib.compress(Path(__file__).with_name("articulated_checkpoint.py").read_bytes(), 9)
).decode()
workflow = a.workflow or "articulated-20260909-" + a.embodiment + "-1"
if not workflow.startswith("articulated-20260909-" + a.embodiment + "-"):
    ap.error("workflow does not belong to this embodiment and run")
code = f"""import base64,pathlib,subprocess,zlib,os
p=pathlib.Path('/workspace/articulated_checkpoint.py')
assert not pathlib.Path('/workspace/demos/checkpoint_complete.json').exists(), 'Already checkpointed'
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: active=str(p).encode() in f.read_bytes().split(b'\\0')
    except OSError: continue
    assert not active, 'Checkpoint already running'
p.write_bytes(zlib.decompress(base64.b64decode('{source}')))
log=open('/workspace/checkpoint.log','a')
child=subprocess.Popen(['/workspace/EgoVerse/emimic/bin/python','-u',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,ARTICULATED_CHECKPOINT_SUFFIX='{a.checkpoint_suffix}'))
print('CHECKPOINT_PROCESS',child.pid,flush=True)
"""
entry = "bash -lc " + shlex.quote(
    "cd /workspace/EgoVerse && source emimic/bin/activate && python -c "
    + shlex.quote(code)
)
master, slave = pty.openpty()
fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
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
while time.monotonic() < deadline and process.poll() is None:
    ready, _, _ = select.select([master], [], [], 1.0)
    if ready:
        try:
            output += os.read(master, 65536)
        except OSError:
            break
    if b"CHECKPOINT_PROCESS " in output:
        break
process.terminate()
try:
    process.wait(timeout=5)
except subprocess.TimeoutExpired:
    process.kill()
    process.wait()
os.close(master)
lines = output.decode(errors="replace").splitlines()
markers = [line for line in lines if line.startswith("CHECKPOINT_PROCESS ")]
if not markers:
    print("\n".join(lines[-10:]))
    raise SystemExit(1)
print(workflow, markers[-1])
