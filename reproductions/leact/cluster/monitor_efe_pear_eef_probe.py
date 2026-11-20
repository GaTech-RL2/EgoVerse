"""Read-only native EEF qualification census, separate from teacher collection."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time

from .scale_efe import BASE,command,read,write
from .launch_efe_pear_scaled import DEST
from .efe_pear_allocation_jobs import running_tasks
from .monitor_efe_preflight import remote


ROOT=DEST/'eef-interface-v1'


def capture(label,state):
    workflow=state.get('workflow',state.get('name'))
    query=command(['osmo','workflow','query',workflow,'--format-type','json'])
    prior=read(ROOT/(label+'-native-probe-latest.json'),{})
    result={**prior,'label':label,'workflow':workflow,'scheduler_status':query['status'],
        'probe_root':state['root']}
    if prior.get('probe_root')not in (None,state['root']):result.pop('status',None)
    reader=workflow
    if query['status']=='RUNNING':
        running_tasks(query)
    else:
        # Completed checks live on the shared volume even if their GPU
        # allocation is reclaimed later. Never erase verified results because
        # the scheduler no longer permits exec into the original pod.
        retained=read(ROOT/'yam-native-probe-launch.json',{})
        reader=retained.get('workflow')
        if not reader:raise RuntimeError('No retained reader for durable EEF proof')
        running_tasks(command(['osmo','workflow','query',reader,'--format-type','json']))
    script='''from pathlib import Path
import json
root=Path(__ROOT__);logs={}
for name in ['probe.log','server.log']:
 p=root/name
 if p.exists():logs[name]=p.read_text()[-8000:]
for p in (root/'native').glob('*/*/native.log'):
 logs[str(p.relative_to(root))]=p.read_text()[-4000:]
p=root/'status.json'
print('EFE_CAPTURE '+json.dumps({'status':json.loads(p.read_text())if p.exists()else{},'logs':logs}),flush=True)
'''.replace('__ROOT__',repr(state['root']))
    observed=remote(reader,script,task='native')
    if observed.get('status'):result.update(observed)
    write(ROOT/(label+'-native-probe-latest.json'),result)
    return result


def tick():
    records=[]
    for label in ['yam','dojo','robolab','yamlab','core']:
        state=read(ROOT/(label+'-native-probe-launch.json'))
        if state:records.append((label,state))
    rows={}
    with ThreadPoolExecutor(max_workers=5)as pool:
        futures={pool.submit(capture,*item):item[0]for item in records}
        for future in as_completed(futures):
            label=futures[future]
            try:
                value=future.result();status=value.get('status',{});checks=status.get('checks',[])
                rows[label]={'workflow':value['workflow'],'scheduler_status':value['scheduler_status'],
                    'phase':status.get('phase','awaiting_native_probe_status'),
                    'diagnostic_checks':len(checks),'diagnostic_checks_passed':sum(bool(c.get('passed'))for c in checks),
                    'api_motion_passed':status.get('api_motion_passed',status.get('passed',False)),
                    'error':status.get('error'),'updated_unix':status.get('updated_unix')}
            except Exception as exc:rows[label]={'observer_error':repr(exc)}
    result={'scope':'pear_rl_only','control_interface':'pear-eef-native-v1','updated_unix':time.time(),
        'teacher_collection_changed':False,'rl_optimizer_updates':0,'native_probes':rows,
        'all_new_native_adapters_qualified':len(rows)==5 and all(v.get('api_motion_passed')for v in rows.values())}
    if(ROOT/'qualification.json').exists():
        from .efe_pear_eef_admission import qualify
        from .efe_pear_full_admission import PendingAdmission
        try:
            qualify(DEST,BASE/'cluster')
            result.update(all_new_native_adapters_qualified=True,
                qualification_basis='Composed source-pinned, passed per-suite native receipts; failed attempts remain separately visible.',
                qualification_path=str(ROOT/'qualification.json'))
        except(PendingAdmission,ValueError):pass
    write(ROOT/'status.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        print(json.dumps(tick()),flush=True)
        if not args.watch:break
        time.sleep(90)
