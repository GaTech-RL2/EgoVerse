"""Reassign only ungraded teacher tasks after the October 30 quota reclaim."""
import shutil
import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write
from .launch_efe_training import freeze_sources
from .launch_efe_direct_control import DEST, DURABLE, now


def build(source, name, tasks, directory, recovery=None):
    target=BUILD_ROOT/('efe-direct-control-20261101-v2-'+name)
    if (target/'source-manifest.json').exists():return target
    target.mkdir(exist_ok=True)
    for filename in read(source/'source-manifest.json'):
        shutil.copyfile(source/filename,target/filename)
    spec=read(target/'spec.json')
    spec.update(tasks=[t for t in spec['tasks'] if t['task'] in tasks],
        local_results_directory='direct-control-v1/'+directory,durable_root=DURABLE)
    if len(spec['tasks'])!=len(tasks):raise ValueError('Recovery membership differs')
    if recovery:
        spec.update(recovery_of=recovery,recovery_reason='Quota reclaim interrupted this allocation. Restart only tasks without collected native grades; preserve all graded successes and failures.')
    write(target/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_direct_control_worker.py',target/'efe_direct_control_worker.py')
    workflow=yaml.safe_load((target/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-direct-'+name+'-v2'
    (target/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for p in target.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(target)
    return target


def prepare():
    receipt=ROOT/'recovery-20261101/teacher-reassignment.json'
    if receipt.exists():return read(receipt)
    plan=read(DEST/'plan.json');write(ROOT/'recovery-20261101/teacher-plan-before.json',plan)
    entries=[];changes=[]
    for item in plan['builds']:
        folder=DEST/item['name'];source=BUILD_ROOT/item['build'].split('/')[-1]
        if item['name'] in {'a1-03','a2-01'}:
            results=read(folder/'results.json');trials=results['trials']
            if not trials or any(r['status']!='graded' for r in trials):raise ValueError('Review non-graded partial records before reassignment')
            retained=read(folder/'durable-retained.json')
            manifest=read(folder/'public-episodes/manifest.json')
            for row in manifest['episodes']:
                if retained.get(row['retained_file'],{}).get('sha256')!=row['retained_sha256']:
                    raise ValueError('Partial teacher bundle not durably retained')
            graded={r['task'] for r in trials};remaining=[t for t in item['tasks'] if t not in graded]
            if len(graded)!=4 or len(remaining)!=2:raise ValueError('Unexpected recovery membership')
            original_tasks=item['tasks'];item={**item,'tasks':[t for t in original_tasks if t in graded],
                'original_tasks':original_tasks,'remaining_reassigned_to':item['name']+'-resume'}
            entries.append(item)
            name=item['name']+'-resume'
            target=build(source,name,remaining,name,results['workflow'])
            entries.append({'name':name,'attempt':item['attempt'],'build':str(target),'tasks':remaining,
                'recovery_of':results['workflow']})
            record={'workflow':results['workflow'],'retained_graded':len(graded),'reassigned_ungraded':remaining,
                'completion_scope':'Collected partial outcomes only; unfinished tasks are assigned to a new workflow.',
                'updated_utc':now()}
            write(folder/'collection-complete.json',record)
            changes.append(record)
        elif not read(source/'submission.json'):
            target=build(source,item['name'],item['tasks'],item['name'])
            entries.append({**item,'build':str(target),'previous_unsubmitted_build':str(source)})
        else:entries.append(item)
    assert sum(len(i['tasks']) for i in entries)==72
    pairs=[(item['attempt'],task) for item in entries for task in item['tasks']]
    assert len(pairs)==len(set(pairs))==72
    plan.update(builds=entries,recovery_updated_utc=now())
    write(DEST/'plan.json',plan)
    result={'changes':changes,'total_planned_attempts':72,'unique_task_attempt_pairs':72,
        'new_workers_persist_completed_episodes_without_local_collector':True,'updated_utc':now()}
    write(receipt,result);return result


if __name__=='__main__':
    import json
    print(json.dumps(prepare()))
