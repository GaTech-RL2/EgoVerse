"""Launch native curriculum integration checks without changing training admission."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .launch_efe_pear_scaled import DEST, REMOTE, TEMPLATE, now, require_unpaused
from .scale_efe import BASE, BUILD_ROOT, ROOT, command, read, submit, write
from .launch_efe_training import freeze_sources


SHARDS=[['metaworldplus','libero-max','abc-sim'],
        ['robocasa','dexjoco','vlabench'],['robodojo'],['robolab'],['yamlab']]
BUILD_VERSION=2


def prepare(shard,index=0,only_suites=None):
    require_unpaused()
    build=BUILD_ROOT/f'efe-pear-catalog-20261117-v{BUILD_VERSION}-s{shard}-r{index}'
    if (build/'source-manifest.json').exists():return build
    build.mkdir(parents=True,exist_ok=True)
    rows=read(DEST/'proposed-candidate-membership-v2.json')['rows']
    selected=[]
    preferred={'metaworldplus':'metaworldplus-door-lock-i00-standard',
               'robodojo':'robodojo-general-pickup-i00-standard'}
    for suite in SHARDS[shard]:
        if only_suites is not None and suite not in only_suites:continue
        choices=[r for r in rows if r['suite']==suite]
        row=next((r for r in choices if r['task']==preferred.get(suite)),choices[0])
        selected.append(dict(row,partition='train_candidates'))
    split=read(ROOT/'generalization-v2/splits.json')
    assert all(r['task'] in split['train_candidates'] and r['task'] not in split['protected_core_all_modes']
               and r['task'] not in split['development'] for r in selected)
    template=read(BUILD_ROOT/'efe-pear-native-qualification-20261114-v1/spec.json')
    spec={k:template[k] for k in ['repository','revision','podman','compose','fixed_action_config']}
    if not selected:raise ValueError('No native tasks selected')
    spec.update(tasks=selected,archive_root=REMOTE+'/catalog-checks',primary_rl_updates=0,
                enable_gpu_cdi=True,adapter_revision='native22-cdi-v1')
    spec['fixed_action_config']={**spec['fixed_action_config'],'model':'external-pear-policy'}
    write(build/'spec.json',spec)
    for name in ['efe_pear_bridge.py','efe_action_loop.py','efe_action_agent.py','efe_native_pilot.py','efe_pear_gpu_cdi.py']:
        shutil.copyfile(BASE/'cluster'/name,build/name)
    shutil.copyfile(BASE/'cluster/efe_pear_catalog_probe.py',build/'worker.py')
    workflow=yaml.safe_load(TEMPLATE.read_text())
    workflow['workflow']['name']=f'leact-efe-pear-catalog-v{BUILD_VERSION}-s{shard}-r{index}'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'6h'}
    # The first combined Isaac-family shard exceeded 400 GiB while unpacking
    # simulator images under nested Podman's VFS driver. Isolate those families
    # and size the job's scratch space; native agent resource limits stay intact.
    storage='800Gi' if shard>=2 else '400Gi'
    workflow['workflow']['resources']['default'].update(gpu=8,cpu=48,memory='512Gi',storage=storage)
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None);task['privileged']=True
    task['args']=['-u','/tmp/efe-pear-catalog/worker.py']
    task['files']=[{'localpath':p.name,'path':'/tmp/efe-pear-catalog/'+p.name} for p in build.iterdir() if p.is_file()]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    return build


def launch():
    require_unpaused()
    with (DEST/'catalog-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queues=read(DEST/'catalog-queues.json',{})
        for shard in range(len(SHARDS)):
            key=str(shard)
            if queues.get(key,{}).get('attempts'):continue
            build=prepare(shard);require_unpaused();receipt=submit(build,priority='HIGH')
            queues[key]={'attempts':[{'workflow':receipt['name'],'build':str(build),
                'pool':'groot-h100-01','priority':'HIGH','submitted_utc':now()}]}
            write(DEST/'catalog-queues.json',queues)
        return queues


def tick():
    if read(DEST/'control.json',{}).get('paused'):return True,{'phase':'paused_by_user'}
    from .launch_efe_coverage import infra_retryable
    from .efe_coverage_retry import retry_due
    queues=read(DEST/'catalog-queues.json');statuses=[];done=True
    contracts=read(DEST/'catalog-contracts.json',{})
    for shard,queue in queues.items():
        attempt=queue['attempts'][-1];name=attempt['workflow']
        scheduler=command(['osmo','workflow','query',name,'--format-type','json'])
        write(Path(attempt['build'])/'scheduler.json',scheduler)
        state={'shard':int(shard),'workflow':name,'scheduler_status':scheduler['status']}
        terminal=scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED')
        if terminal or scheduler['status']=='RUNNING':
            try:
                logs=subprocess.run(['osmo','workflow','logs',name,'-n','30'],capture_output=True,text=True,timeout=15)
                lines=logs.stdout.splitlines()
            except subprocess.TimeoutExpired:
                state['log_error']='OSMO log request timed out; scheduler state remains authoritative.'
                lines=[]
            for line in lines:
                if 'PEAR_CATALOG_PROBE ' in line:
                    try:state['worker']=json.loads(line.split('PEAR_CATALOG_PROBE ',1)[1])
                    except json.JSONDecodeError:pass
        for task in state.get('worker',{}).get('tasks',[]):
            contracts[task['suite']]={**task,'workflow':name}
        previous_spec=read(Path(attempt['build'])/'spec.json')
        storage_repair=(scheduler['status']=='FAILED_EVICTED' and int(shard)==2
                        and 'local ephemeral storage limit' in json.dumps(scheduler)
                        and '400Gi' in json.dumps(scheduler))
        source_repair=(scheduler['status'].startswith('FAILED') and state.get('worker')
                       and previous_spec.get('adapter_revision')!='native22-cdi-v1')
        if source_repair or storage_repair:
            terminal=False
            # A terminated shard may never have reached its last suite. Keep
            # those unchecked suites in the retry alongside observed failures.
            failed=[s for s in SHARDS[int(shard)] if not contracts.get(s,{}).get('success')]
            if storage_repair:failed=SHARDS[int(shard)]
            # Two observed integration errors: missing NVIDIA CDI, and native
            # eai-standard/2.2 Hz/nested-bound fields. One revised attempt only.
            build=prepare(int(shard),len(queue['attempts']),only_suites=failed or None)
            require_unpaused();receipt=submit(build,priority='HIGH')
            queue['attempts'].append({'workflow':receipt['name'],'build':str(build),
                'pool':'groot-h100-01','priority':'HIGH','submitted_utc':now(),
                'source_repair':'native22-cdi-v1','retry_suites':failed or SHARDS[int(shard)]})
            state.update(workflow=receipt['name'],scheduler_status='PENDING',source_repair='native22-cdi-v1')
            state.pop('worker',None)
        elif scheduler['status'].startswith('FAILED') and infra_retryable(scheduler):
            terminal=False
            if retry_due(queue,name):
                suites=[r['suite'] for r in previous_spec['tasks']]
                build=prepare(int(shard),len(queue['attempts']),only_suites=suites);require_unpaused()
                receipt=submit(build,priority='LOW')
                queue['attempts'].append({'workflow':receipt['name'],'build':str(build),'pool':'groot-h100-01',
                    'priority':'LOW','submitted_utc':now()})
                state.update(workflow=receipt['name'],scheduler_status='PENDING')
        # A prior repair may have selected only its observed failure and missed
        # an unexecuted trailing suite. Reconcile against every frozen payload,
        # then submit only suites that never had a corrected attempt at all.
        corrected_attempts=set()
        for old in queue['attempts']:
            old_spec=read(Path(old['build'])/'spec.json')
            if old_spec.get('adapter_revision')=='native22-cdi-v1':
                corrected_attempts.update(r['suite'] for r in old_spec['tasks'])
        missing=[s for s in SHARDS[int(shard)] if s not in corrected_attempts
                 and not contracts.get(s,{}).get('success')]
        if terminal and missing:
            build=prepare(int(shard),len(queue['attempts']),only_suites=missing)
            require_unpaused();receipt=submit(build,priority='HIGH')
            queue['attempts'].append({'workflow':receipt['name'],'build':str(build),
                'pool':'groot-h100-01','priority':'HIGH','submitted_utc':now(),
                'reason':'Complete suites omitted by an interrupted earlier shard','retry_suites':missing})
            state.update(workflow=receipt['name'],scheduler_status='PENDING',missing_suites=missing)
            state.pop('worker',None);terminal=False
        statuses.append(state);done=done and terminal
        write(DEST/'catalog-queues.json',queues)
        write(DEST/'catalog-contracts.json',contracts)
    result={'phase':'checks_finished' if done else 'native_catalog_checks','updated_utc':now(),
            'shards':statuses,'contracts':contracts,'primary_rl_updates':0,'training_admission':False}
    write(DEST/'catalog-status.json',result)
    return done,result


def monitor():
    from ..embodied_first_exam.report import publish
    with (DEST/'catalog-monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                done,state=tick();print(json.dumps(state),flush=True)
                publish()
                if done:return
            except Exception as exc:print(json.dumps({'utc':now(),'error':repr(exc)}),flush=True)
            time.sleep(45)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','monitor']);args=p.parse_args()
    if args.action=='launch':print(json.dumps(launch(),indent=2))
    elif args.action=='monitor':monitor()
    else:print(json.dumps(tick()[1],indent=2))
