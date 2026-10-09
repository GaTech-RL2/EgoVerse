"""Own one guarded allocation, its tunnel, relay and final result recovery."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OPS = Path(__file__).resolve().parent
REPO = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/astra_reversal/workspaces/complex-manipulation-20261006')
launch = OPS / 'launch-robocasa-xiaomi-selector-v4'
port = 18777
children = []
logs = []
workflow = None


def log_process(name, argv, env=None):
    path = OPS / f'xiaomi-selector-v4-{name}.log'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(fd, 'w')
    logs.append(stream)
    child = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
    children.append(child)
    return child


try:
    guard = log_process('monitor', [sys.executable, str(OPS/'submit_xiaomi_selector_low_guarded.py'), str(launch)])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            workflow = json.loads((launch/'submission.json').read_text())['name']
            break
        except (OSError, ValueError, KeyError):
            if guard.poll() is not None:
                raise RuntimeError('Submission guard stopped before a known workflow')
            time.sleep(2)
    if not workflow or not workflow.startswith('astra-complex-20261006-robocasa-xiaomi-selector-v4-'):
        raise RuntimeError('No validated owned workflow identity')
    print(json.dumps({'workflow':workflow,'stage':'submitted'}), flush=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            state = json.loads((launch/'latest-status.json').read_text())
        except (OSError, ValueError):
            state = {}
        if state.get('end_time') or guard.poll() is not None:
            raise RuntimeError('Workflow ended before worker readiness')
        if state.get('start_time'):
            break
        time.sleep(5)
    else:
        raise RuntimeError('Worker startup deadline exceeded')

    tunnel = log_process('port-forward', [
        sys.executable,str(OPS/'reconnect_xiaomi_selector_v4_tunnel.py'),workflow,str(port)])
    relay = log_process('relay', [
        sys.executable,'-u','-m','astra_reversal.codex_relay',
        '--url',f'http://127.0.0.1:{port}',
        '--directory',str(OPS/'codex-xiaomi-selector-v4'),
        '--token-file',str(launch/'astra-relay.token'),
        '--max-requests','160','--idle-timeout','2500'])
    print(json.dumps({'workflow':workflow,'stage':'relay_started','port':port}), flush=True)
    (launch/'owned_processes.json').write_text(json.dumps({
        'controller_pid':os.getpid(),'guard_pid':guard.pid,
        'tunnel_pid':tunnel.pid,'relay_pid':relay.pid,'workflow':workflow},indent=2)+'\n')
    deadline = time.monotonic() + 23400
    service_dead_since = None
    while time.monotonic() < deadline and guard.poll() is None:
        if relay.poll() is not None and relay.returncode != 0 or tunnel.poll() is not None:
            service_dead_since = service_dead_since or time.monotonic()
            if time.monotonic()-service_dead_since > 45:
                raise RuntimeError('An owned relay service stopped before workflow completion')
        time.sleep(5)
    if guard.poll() is None or guard.returncode:
        raise RuntimeError('Allocation guard did not complete successfully')
    state = json.loads((launch/'final-status.json').read_text())
    print(json.dumps({'workflow':workflow,'stage':'allocation_closed','status':state['status']}), flush=True)
except BaseException as exc:
    print(json.dumps({'controller_error':type(exc).__name__,'message':str(exc)[:250]}), flush=True)
    if workflow and not (launch/'final-status.json').exists():
        subprocess.run(['/usr/local/bin/osmo','workflow','cancel',workflow],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
    raise
finally:
    # Guard is left alive if needed to account for a cancellation to completion.
    for child in children[1:]:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    for stream in logs: stream.close()

for stage, argv in (
    ('catalog',[sys.executable,str(OPS/'recover_catalog.py'),str(launch),'--pool','groot-l40s-01','--timeout','600']),
    ('download',[sys.executable,str(OPS/'fetch_result_archive.py'),str(launch),str(OPS/'robocasa-xiaomi-selector-v4-results')]),
):
    with (OPS/f'xiaomi-selector-v4-{stage}.log').open('x') as stream:
        result = subprocess.run(argv,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps({'workflow':workflow,'stage':stage,'returncode':result.returncode}),flush=True)
    if result.returncode: raise SystemExit(result.returncode)
