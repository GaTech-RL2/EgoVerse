"""Collect the simulator-control workflow without importing oracle traces."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .monitor_efe_preflight import remote
from .efe_native_pilot import control_verdict
from ..embodied_first_exam.report import publish,write,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve()
    lock=(build/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    workflow=json.loads((build/'submission.json').read_text())['name'];dest=ROOT/'native-pilot';dest.mkdir(exist_ok=True)
    expected_controls=2*len(json.loads((build/'spec.json').read_text())['tasks'])
    collected=False
    for _ in range(1500):
        try:
            p=subprocess.run(['osmo','workflow','query',workflow,'--format-type','json'],capture_output=True,text=True,timeout=35,check=True)
            state=json.loads(p.stdout[p.stdout.index('{'):]);write(build/'status-current.json',state)
            if state['status']=='RUNNING' and not collected:
                capture=remote(workflow,"""from pathlib import Path
import base64,hashlib,json,time
root=Path('/tmp/efe-native');files={}
for name in ['progress.json','results.json','complete.json','setup-error.json']:
 p=root/name
 if p.exists():
  d=p.read_bytes();files[name]={'sha256':hashlib.sha256(d).hexdigest(),'data':base64.b64encode(d).decode()}
controls=[]
if 'results.json' in files:
 summary=json.loads(base64.b64decode(files['results.json']['data']))
 archive=Path('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-native-controls-20261026-v1')/summary['workflow']
 for index,record in enumerate(summary.get('trials',[])):
  job=f'{index//2:02d}-{record["agent"]}'
  matches=[]
  for p in (archive/'jobs'/job).glob('*/result.json'):
   raw=p.read_bytes();d=json.loads(raw)
   if d.get('task_name') in {record['task'],'embodied-first-exam/'+record['task']}:
    matches.append({'task_name':d['task_name'],'verifier_result':d.get('verifier_result'),'exception_info':d.get('exception_info'),'file_sha256':hashlib.sha256(raw).hexdigest()})
  controls.append({'task':record['task'],'agent':record['agent'],'matches':matches})
s='EFE_CAPTURE '+json.dumps({'files':files,'controls':controls})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.02)
""")
                receipt={}
                for name,item in capture['files'].items():
                    if name not in {'progress.json','results.json','complete.json','setup-error.json'}:raise ValueError('Unexpected output name')
                    data=base64.b64decode(item['data'])
                    if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Output checksum mismatch')
                    (dest/name).write_bytes(data);receipt[name]=item['sha256']
                write(build/'collection-receipt.json',receipt)
                # Harbor may prefix task_name with the dataset namespace. Reconcile
                # from the sealed trial result; preserve the original runner receipt.
                if 'results.json' in receipt:
                    summary=json.loads((dest/'results.json').read_text())
                    write(dest/'runner-results.json',summary)
                    for record,control in zip(summary['trials'],capture['controls'],strict=True):
                        if (record['task'],record['agent'])!=(control['task'],control['agent']):raise ValueError('Control order changed')
                        if len(control['matches'])>1:raise ValueError('Ambiguous native trial result')
                        if control['matches']:
                            trial=control['matches'][0];verdict,score=control_verdict(record['agent'],trial)
                            if record['returncode']!=0:verdict='infrastructure_failure'
                            record.update(status=verdict,score=score,trial_result_file_sha256=trial['file_sha256'],
                                          detail=f'Expected {int(record["agent"]=="oracle")}; actual {score}. Diagnostic only.',
                                          resource_metrics=(trial.get('verifier_result') or {}).get('rewards',{}))
                            if verdict=='passed':record['error_tail']=None
                    summary['all_controls_passed']=len(summary['trials'])==expected_controls and all(r['status']=='passed' for r in summary['trials'])
                    write(dest/'results.json',summary)
                    if 'complete.json' in receipt:
                        complete=json.loads((dest/'complete.json').read_text());write(dest/'runner-complete.json',complete)
                        complete.update(all_controls_passed=summary['all_controls_passed'],reconciled_from_trial_results=True)
                        write(dest/'complete.json',complete)
                    if 'complete.json' in receipt and summary['all_controls_passed']:
                        progress=json.loads((dest/'progress.json').read_text())
                        progress.update(phase='controls_passed',detail=f'All {expected_controls} development oracle/nop checks passed. Teacher collection and resource-enforcement validation are next; no EFE model score yet.')
                        write(dest/'progress.json',progress)
                if 'complete.json' in receipt:
                    remote(workflow,"from pathlib import Path;Path('/tmp/efe-native/collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    collected=True
            elif not (dest/'progress.json').exists():
                write(dest/'progress.json',{'workflow':workflow,'phase':state['status'],'detail':'Native development-task simulator controls '+state['status'].lower()+'; no EFE SFT or model result yet'})
            if state['status'].startswith('FAILED'):
                write(dest/'progress.json',{'workflow':workflow,'phase':state['status'],'detail':'OSMO simulator-control workflow ended before completing its controls; recorded results remain diagnostic only'})
            publish()
            write(build/'monitor-status.json',{'workflow':workflow,'pid':os.getpid(),'scheduler_status':state['status'],'updated_utc':datetime.now(timezone.utc).isoformat(),'complete_collected':collected})
            if state['status']=='COMPLETED' or state['status'].startswith('FAILED'):return
        except Exception as error:
            write(build/'monitor-error.json',{'error':repr(error),'updated_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(20)
    raise TimeoutError('Native simulator-control monitor reached its bounded lifetime')


if __name__=='__main__':main()
