"""Publish measured Astra pilot progress; retain raw episodes on shared storage."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zlib

from .monitor_efe_preflight import remote
from ..embodied_first_exam import report
from ..embodied_first_exam.report import ROOT, publish, read, write


def collection_worker(state, active):
    """Read a finished job's shared archive through another owned allocation."""
    if state['status']=='RUNNING':return state['name']
    owner=state.get('submitted_by')
    return next((row['name'] for row in active
                 if owner and row.get('user')==owner and row.get('status')=='RUNNING'),None)


def restore_local_export(log,dest,names):
    """Recover small node-local diagnostics from durable OSMO logs."""
    matches=re.findall(r'EFE_LOCAL_RESULTS ([A-Za-z0-9+/=]+)',log)
    if not matches:return {}
    files=json.loads(zlib.decompress(base64.b64decode(matches[-1])))
    decoded={}
    for name,item in files.items():
        if name not in names:raise ValueError('Unexpected exported result filename')
        data=base64.b64decode(item['data'])
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Exported result checksum mismatch')
        decoded[name]=data
    for name,data in decoded.items():
        temporary=dest/(name+'.download.tmp');temporary.write_bytes(data);temporary.replace(dest/name)
    return {name:files[name]['sha256'] for name in decoded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    lock = (build / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workflow = read(build / 'submission.json')['name']
    spec = read(build / 'spec.json')
    archive = str(Path(spec['archive_root']) / workflow)
    dest = ROOT / spec.get('local_results_directory', 'astra-pilot')
    if ROOT not in dest.resolve().parents:
        raise ValueError('Result directory must remain within the EFE artifacts')
    dest.mkdir(parents=True, exist_ok=True)
    label = spec.get('agent_label', 'Astra')
    collected = (dest / 'complete.json').exists()
    missing_completion_reads=0
    names = ['progress.json', 'results.json', 'complete.json', 'setup-error.json',
             'model-routing.json', 'native-config.json', 'student-serving.json', 'student-preflight.json',
             'student-serving-failure.json']
    names.append('behavior-audit.json')
    for _ in range(4500):
        try:
            result = subprocess.run(['osmo', 'workflow', 'query', workflow, '--format-type', 'json'],
                                    capture_output=True, text=True, timeout=35, check=True)
            state = json.loads(result.stdout[result.stdout.index('{'):])
            write(build / 'status-current.json', state)
            via=collection_worker(state,[])
            terminal=state['status']=='COMPLETED' or state['status'].startswith('FAILED')
            if not collected and terminal and spec.get('node_local_results'):
                logs=subprocess.run(['osmo','workflow','logs',workflow,'-n','100'],capture_output=True,text=True,check=True,timeout=35)
                receipt=restore_local_export(logs.stdout,dest,names)
                if receipt:write(build/'collection-receipt.json',receipt)
                collected='complete.json' in receipt
                if not collected:missing_completion_reads+=1
            elif not collected and terminal:
                from .scale_efe import command
                active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','--format-type','json'])['workflows']
                via=collection_worker(state,active)
                if via is None and state['status']=='COMPLETED':
                    from .launch_efe_training import recovery_reader
                    via=recovery_reader(build)
            if via and not collected:
                script = """from pathlib import Path
import base64,hashlib,json,time,zlib
root=Path(ARCHIVE);files={}
for name in NAMES:
 p=root/name
 if p.exists():
  d=p.read_bytes();files[name]={'sha256':hashlib.sha256(d).hexdigest(),'data':base64.b64encode(zlib.compress(d)).decode()}
requests=[]
for p in Path(ARCHIVE).glob('requests/*/*.json'):
 d=json.loads(p.read_text());requests.append(d)
transport={'requests':len(requests),'completed':sum(d.get('status')=='completed' for d in requests),
 'http_200':sum(d.get('http_status')==200 for d in requests),
 'http_errors':sum(d.get('http_status',0)>=400 for d in requests if d.get('http_status') is not None),
 'caps':sorted({d['max_output_tokens'] for d in requests}),
 'returned_models':sorted({d['returned_model'] for d in requests if d.get('returned_model')}),
 'public_tool_calls':sum(x.get('type') in {'function_call','custom_tool_call'} for d in requests for x in d.get('public_output',[])),
 'input_tokens':sum((d.get('usage') or {}).get('input_tokens',0) for d in requests),
 'output_tokens':sum((d.get('usage') or {}).get('output_tokens',0) for d in requests),
 'reasoning_tokens':sum(((d.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens',0) for d in requests)}
s='EFE_CAPTURE '+json.dumps({'files':files,'transport':transport})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.005)
""".replace('NAMES', repr(names)).replace('ARCHIVE', repr(archive))
                capture = remote(via, script)
                receipt = {}
                for name, item in capture['files'].items():
                    if name not in names:
                        raise ValueError('Unexpected pilot result file')
                    data = zlib.decompress(base64.b64decode(item['data']))
                    if hashlib.sha256(data).hexdigest() != item['sha256']:
                        raise ValueError('Pilot result checksum mismatch')
                    temporary=dest/(name+'.download.tmp')
                    temporary.write_bytes(data)
                    temporary.replace(dest/name)
                    receipt[name] = item['sha256']
                write(build / 'collection-receipt.json', receipt)
                transport = capture['transport']
                transport.update(workflow=workflow, updated_utc=datetime.now(timezone.utc).isoformat())
                write(dest / 'transport.json', transport)
                progress = read(dest / 'progress.json', {})
                if transport['requests']:
                    progress['benchmark_rollouts_started'] = True
                if progress.get('phase', '').startswith('episode_'):
                    index = int(progress['phase'].split('_')[1].split('-')[0])
                    progress['current_task'] = spec['tasks'][index]['task']
                    progress['detail'] = (f'{label} task {index + 1}/{len(spec["tasks"])}: '
                                          f'{progress["current_task"]}; {transport["completed"]} model requests completed across the pilot. '
                                          'Native evaluation; training updates are tracked separately.')
                progress['observed_utc']=transport['updated_utc']
                write(dest / 'progress.json', progress)
                if 'complete.json' in receipt:
                    collected = True
                    if via==workflow and state['status']=='RUNNING':
                        remote(via, "from pathlib import Path;Path('/tmp/efe-native/collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    elif via==read(build/'completion-reader/submission.json',{}).get('name'):
                        remote(via, "from pathlib import Path;Path('/tmp/efe-reader-collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                elif state['status']=='COMPLETED':
                    missing_completion_reads+=1
            elif state['status'] in {'PENDING','WAITING'} or not (dest / 'progress.json').exists():
                write(dest / 'progress.json', {'workflow': workflow, 'phase': state['status'],
                      'detail': label + ' native evaluation ' + state['status'].lower() + '; training updates are tracked separately',
                      'training_started': False, 'benchmark_rollouts_started': False, 'planned_trials': len(spec['tasks'])})
            if state['status'].startswith('FAILED'):
                progress = read(dest / 'progress.json', {})
                progress.update(phase=state['status'], detail=label+' workflow failed; recorded trial receipts remain available')
                write(dest / 'progress.json', progress)
            progress = read(dest / 'progress.json', {})
            write(dest / 'scheduler.json', {'workflow': workflow, 'status': state['status'],
                                           'updated_utc': datetime.now(timezone.utc).isoformat()})
            if dest == ROOT / 'astra-pilot':
                status = read(ROOT / 'status.json')
                status.update(workflow=workflow, workflow_scheduler_status=state['status'],
                              phase=progress.get('phase', state['status']),
                              benchmark_rollouts_started=progress.get('benchmark_rollouts_started', False),
                              training_started=False, updated_utc=datetime.now(timezone.utc).isoformat())
                write(ROOT / 'status.json', status)
            importlib.reload(report).publish()
            write(build / 'monitor-status.json', {'pid': os.getpid(), 'workflow': workflow,
                  'scheduler_status': state['status'], 'complete_collected': collected,
                  'collection_needs_review':missing_completion_reads>=3,
                  'updated_utc': datetime.now(timezone.utc).isoformat()})
            if state['status'].startswith('FAILED') or (state['status']=='COMPLETED' and (collected or missing_completion_reads>=3)):
                return
        except Exception as exc:
            write(build / 'monitor-error.json', {'error': repr(exc), 'updated_utc': datetime.now(timezone.utc).isoformat()})
        time.sleep(20)
    raise TimeoutError('Astra pilot monitor reached its bounded lifetime')


if __name__ == '__main__':
    main()
