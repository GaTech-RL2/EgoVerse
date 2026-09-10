"""Install the upload recovery into one of our own workflow containers."""

import argparse
import base64
import fcntl
import hashlib
import os
import pty
import select
import shlex
import struct
import subprocess
import termios
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("workflow")
a = ap.parse_args()
if not a.workflow.startswith("articulated-20260909-"):
    ap.error("This recovery is restricted to this collection run")
raw = Path(__file__).with_name("articulated_upload_recovery.py").read_bytes()
payload = base64.b64encode(raw).decode()
digest = hashlib.sha256(raw).hexdigest()
code = f"""import base64,pathlib,subprocess,os,signal,hashlib,sys
p=pathlib.Path('/workspace/articulated_upload_recovery.py')
existing=[]
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        if str(p).encode() in f.read_bytes().split(b'\\0'): existing.append(int(f.parent.name))
    except OSError: pass
if existing and p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()=='{digest}':
    print('RECOVERY_PROCESS',existing[0],flush=True)
    sys.exit(0)
for pid in existing: os.kill(pid,signal.SIGTERM)
p.write_bytes(base64.b64decode('{payload}'))
log=open('/workspace/upload_recovery.log','a')
child=subprocess.Popen(['/workspace/EgoVerse/emimic/bin/python','-u',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('RECOVERY_PROCESS',child.pid,flush=True)
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
        a.workflow,
        "collect",
        "--entry",
        entry,
        "--connect-timeout",
        "30",
    ],
    stdin=slave,
    stdout=slave,
    stderr=slave,
    close_fds=True,
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
    if b"RECOVERY_PROCESS " in output:
        break
process.terminate()
try:
    process.wait(timeout=5)
except subprocess.TimeoutExpired:
    process.kill()
    process.wait()
os.close(master)
lines = output.decode(errors="replace").splitlines()
markers = [line for line in lines if line.startswith("RECOVERY_PROCESS ")]
if markers:
    print(a.workflow, markers[-1])
else:
    print("\n".join(lines[-12:]))
    raise SystemExit(1)
