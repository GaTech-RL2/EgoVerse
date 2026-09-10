"""Install the verified sticky-clearance fix into this run's two contact jobs."""

import argparse
import base64
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
import zlib
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("embodiment", choices=["triangle", "flipper"])
a = ap.parse_args()
repo = Path(__file__).resolve().parents[1]
patches = {
    str(p.relative_to(repo)): p.read_text()
    for p in (
        repo / "Tsimulation/sim_v2/collect/contact_controller.py",
        repo / "Tsimulation/sim_v2/collect/articulation_batch.py",
    )
}
package = dict(
    patches=patches,
    source=Path(__file__).with_name("articulated_resume.py").read_text(),
)
payload = base64.b64encode(zlib.compress(json.dumps(package).encode(), 9)).decode()
workflow = "articulated-20260909-" + a.embodiment + "-1"
code = f"""import base64,pathlib,subprocess,sys,json,zlib
p=pathlib.Path('/workspace/articulated_resume.py')
assert not pathlib.Path('/workspace/demos/controller_resume.json').exists(), 'Already resumed'
for f in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: active=str(p).encode() in f.read_bytes().split(b'\\0')
    except OSError: continue
    assert not active, 'Resume already running'
package=json.loads(zlib.decompress(base64.b64decode('{payload}')))
p.write_text(package['source'])
pathlib.Path('/workspace/controller_patches.json').write_text(json.dumps(package['patches']))
log=open('/workspace/controller_resume.log','a')
child=subprocess.Popen(['/workspace/EgoVerse/emimic/bin/python','-u',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('RESUME_PROCESS',child.pid,flush=True)
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
    if b"RESUME_PROCESS " in output:
        break
process.terminate()
try:
    process.wait(timeout=5)
except subprocess.TimeoutExpired:
    process.kill()
    process.wait()
os.close(master)
lines = output.decode(errors="replace").splitlines()
markers = [line for line in lines if line.startswith("RESUME_PROCESS ")]
if not markers:
    print("\n".join(lines[-10:]))
    raise SystemExit(1)
print(workflow, markers[-1])
