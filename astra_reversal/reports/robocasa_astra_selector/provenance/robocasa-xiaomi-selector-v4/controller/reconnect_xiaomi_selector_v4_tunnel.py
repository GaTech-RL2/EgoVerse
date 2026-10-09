"""Reconnect the owned OSMO port-forward after bounded transient failures."""
import json
import signal
import subprocess
import sys
import time

workflow, port = sys.argv[1:]
if not workflow.startswith('astra-complex-20261006-robocasa-xiaomi-selector-'):
    raise SystemExit('Unexpected workflow')
if int(port) not in (18774, 18775, 18776, 18777):
    raise SystemExit('Unexpected owned loopback port')
child = None
def stop(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
failures, unavailable_since = 0, time.monotonic()
while failures < 40 and time.monotonic() - unavailable_since < 1320:
    started = time.monotonic()
    child = subprocess.Popen(['/usr/local/bin/osmo', 'workflow', 'port-forward', workflow,
        'evaluate', '--host', '127.0.0.1', '--port', f'{port}:8769', '--connect-timeout', '60'])
    code = child.wait()
    if time.monotonic() - started > 120:
        failures, unavailable_since = 0, time.monotonic()
    failures += 1
    print(json.dumps({'tunnel_disconnected': code, 'reconnect_attempt': failures}), flush=True)
    time.sleep(min(5 * failures, 30))
raise SystemExit('Owned tunnel failed bounded reconnects')
