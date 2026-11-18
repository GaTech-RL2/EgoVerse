"""Collect one bounded runtime preflight and publish its measured outcome."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zlib

import pexpect

BASE = Path(__file__).resolve().parents[1]


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def remote(workflow, script,task='preflight'):
    payload = base64.b64encode(zlib.compress(script.encode())).decode()
    entry = 'python3 -c ' + shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(payload)+')))')
    child = pexpect.spawn('osmo', ['workflow','exec',workflow,task,'--entry',entry],
                          encoding='utf-8', timeout=35, maxread=65536, echo=False)
    try:
        child.expect(r'EFE_CAPTURE (\{[^\r\n]+\})\r?\n')
        record = json.loads(child.match.group(1))
        child.expect(pexpect.EOF, timeout=10)
        return record
    finally:
        child.close(force=True)


def publish(workflow, state, receipt):
    with (BASE/'cluster/.build/builderbench-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status_path=BASE/'artifacts/embodied-first-exam/status.json'
        status=json.loads(status_path.read_text())
        passed=bool(receipt and receipt.get('nested_container_passed'))
        status['runtime_followup']={'workflow':workflow,'scheduler_status':state['status'],
                                    'nested_container_passed':passed,'receipt_available':receipt is not None}
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        write(status_path,status)
        path=BASE/'artifacts/experiment-plan.json'; plan=json.loads(path.read_text())
        plan['embodied_first_exam']['runtime_followup']=status['runtime_followup']
        if receipt:
            note=('H100 nested-container prerequisite passed; native simulator oracle/nop and VL training pilots still pending'
                  if passed else 'L40S graphics passed; H100 nested runtime probe also failed; supported simulator deployment still needs integration')
        else:
            note='L40S graphics passed; H100 nested runtime follow-up '+state['status']+'; no EFE SFT yet'
        plan['embodied_first_exam']['status']=note
        for item in plan['sequence']:
            if item['id']=='embodied_first_exam': item['status']=note
        plan['updated_utc']=status['updated_utc'];write(path,plan)
        from ..builderbench.queue import publish as publish_dashboard
        publish_dashboard()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args(); build=args.build.resolve()
    lock=(build/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    workflow=json.loads((build/'submission.json').read_text())['name']
    receipt=None
    for _ in range(600):
        try:
            process=subprocess.run(['osmo','workflow','query',workflow,'--format-type','json'],capture_output=True,text=True,check=True,timeout=35)
            state=json.loads(process.stdout[process.stdout.index('{'):]);write(build/'status-current.json',state)
            if state['status']=='RUNNING' and receipt is None:
                capture=remote(workflow,"""from pathlib import Path
import base64,hashlib,json
p=Path('/tmp/efe-preflight/result.json')
d=p.read_bytes() if p.exists() else None
print('EFE_CAPTURE '+json.dumps({'ready':d is not None,'sha256':hashlib.sha256(d).hexdigest() if d else None,'data':base64.b64encode(d).decode() if d else None}),flush=True)
""")
                if capture['ready']:
                    data=base64.b64decode(capture['data'])
                    if hashlib.sha256(data).hexdigest()!=capture['sha256']: raise ValueError('Result checksum mismatch')
                    receipt=json.loads(data)
                    output=BASE/'artifacts/embodied-first-exam/preflight'/f'{workflow}.json'
                    output.write_bytes(data)
                    write(build/'collection-receipt.json',{'path':str(output),'sha256':capture['sha256']})
                    remote(workflow,"from pathlib import Path;Path('/tmp/efe-preflight/collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
            publish(workflow,state,receipt)
            write(build/'monitor-status.json',{'pid':os.getpid(),'workflow':workflow,'scheduler_status':state['status'],
                                               'updated_utc':datetime.now(timezone.utc).isoformat(),'receipt_collected':receipt is not None})
            if state['status']=='COMPLETED' or state['status'].startswith('FAILED'):
                return
        except Exception as error:
            write(build/'monitor-error.json',{'error':repr(error),'updated_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(20)
    raise TimeoutError('Preflight monitor reached its bounded lifetime')


if __name__=='__main__':
    main()
