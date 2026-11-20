"""Validate the complete independent native YAM bank in small RTX allocations."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .launch_efe_pear_scaled import DEST
from .launch_efe_pear_warmstart import ARM,prepare_native_yam_check
from .launch_efe_training import freeze_sources
from .scale_efe import command,read,write,submit


def tick():
    from .launch_efe_coverage import infra_retryable
    from ..tracking.reasoning_live import command as output
    queue=read(DEST/'native-yam-bank-queue.json');rows=[]
    for item in queue['shards']:
        state=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
        row={**item,'scheduler_status':state['status']}
        if state.get('start_time')and state['status']not in item.get('terminal_collected',[]):
            text=output(['osmo','workflow','logs',item['workflow'],'-n','15'],timeout=5)
            for line in text.splitlines():
                if'PEAR_NATIVE_YAM_CHECK 'in line:
                    try:item['worker']=json.JSONDecoder().raw_decode(line.split('PEAR_NATIVE_YAM_CHECK ',1)[1])[0]
                    except ValueError:pass
            if state['status']=='COMPLETED'or state['status'].startswith('FAILED'):
                item['terminal_collected']=[state['status']]
        row.update(worker=item.get('worker',{}))
        if state['status'].startswith('FAILED')and infra_retryable(state)and item.get('attempt',0)<8:
            original=Path(item['build']);attempt=item.get('attempt',0)+1
            target=original.with_name(original.name.rsplit('-allocation-r',1)[0]+'-allocation-r'+str(attempt))
            if not(target/'source-manifest.json').exists():
                target.mkdir(exist_ok=True)
                for name,digest in read(original/'source-manifest.json').items():
                    if hashlib.sha256((original/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen native bank worker changed')
                    shutil.copyfile(original/name,target/name)
                spec=yaml.safe_load((target/'workflow.yaml').read_text());spec['workflow']['name']='leact-'+target.name
                (target/'workflow.yaml').write_text(yaml.safe_dump(spec,sort_keys=False));freeze_sources(target)
            receipt=submit(target,priority='LOW',pool=item['pool'],nodes=1)
            item.setdefault('history',[]).append({'workflow':item['workflow'],'scheduler_status':state['status']})
            item.update(workflow=receipt['name'],build=str(target),attempt=attempt,worker={})
            item.pop('terminal_collected',None);row={**item,'scheduler_status':'PENDING'}
        rows.append(row)
    write(DEST/'native-yam-bank-queue.json',queue)
    worker={'phase':'independent_native_reset_bank_validation','checked':sum(r.get('worker',{}).get('checked',0)for r in rows),
        'passed':sum(r.get('worker',{}).get('passed',0)for r in rows),'total':sum(len(r['tasks'])for r in rows),
        'training_admitted':False,'primary_rl_updates':0}
    finished=all(r['scheduler_status']=='COMPLETED'or r['scheduler_status'].startswith('FAILED')for r in rows)
    value={'updated_utc':datetime.now(timezone.utc).isoformat(),'shards':rows,'worker':worker,
        'scheduler_status':'COMPLETED'if all(r['scheduler_status']=='COMPLETED'for r in rows)else'RUNNING'if any(r['scheduler_status']=='RUNNING'for r in rows)else'PENDING',
        'phase':'native_yam_reset_bank_checked'if finished else'native_yam_reset_bank_validation',
        'primary_rl_updates':0}
    if finished and value['scheduler_status']!='COMPLETED':value['scheduler_status']='FAILED'
    write(DEST/'native-yam-bank-status.json',value)
    return finished,value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if args.watch:
        with(DEST/'native-yam-bank-watch.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while ARM in read(DEST/'control.json',{}).get('enabled_training_arms',[]):
                try:
                    done,status=tick()
                    if done:return
                except Exception as exc:write(DEST/'native-yam-bank-observer-error.json',{'error':str(exc)[-1000:]})
                time.sleep(30)
        return
    tasks=sorted(read(DEST/'native-yam-generated-resets/generation-report.json')['tasks'])
    runtime=read(DEST/'native-yam-runtime-config.json')
    if runtime.get('validation_gpus')!=2:raise ValueError('This bank launcher requires two-GPU RTX allocations')
    queue=read(DEST/'native-yam-bank-queue.json',{'shards':[]})
    existing={tuple(row['tasks'])for row in queue['shards']}
    if len({task for row in queue['shards']for task in row['tasks']})!=sum(len(row['tasks'])for row in queue['shards']):
        raise ValueError('A native task has duplicate shard ownership')
    for offset in range(0,len(tasks),4):
        if tuple(tasks[offset:offset+4])not in existing:
            chunk=tasks[offset:offset+4];build=prepare_native_yam_check(chunk,generated_resets=True)
            row={'build':str(build),'tasks':chunk,'pool':runtime['pool'],'priority':'LOW','gpu':2,'primary_rl_updates':0}
            if args.submit:
                receipt=submit(build,priority='LOW',pool=runtime['pool'],nodes=1);row['workflow']=receipt['name']
            queue['shards'].append(row)
            if args.submit:write(DEST/'native-yam-bank-queue.json',queue)
            print(json.dumps(row),flush=True)
    if args.submit:
        with(DEST/'native-yam-bank-observer.log').open('ab',buffering=0)as log:
            process=subprocess.Popen([sys.executable,'-u','-m',__package__+'.launch_efe_pear_yam_bank','--watch'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(DEST/'native-yam-bank-observer.json',{'pid':process.pid})


if __name__=='__main__':main()
