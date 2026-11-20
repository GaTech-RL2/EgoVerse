"""Validate native float32 JSON, then add compatible simulator lanes in parallel."""
import hashlib
from pathlib import Path

import yaml

from .launch_efe_pear_scaled import DEST
from .scale_efe import BASE,read,write,command
from .efe_pear_submission import submit_reconciled,infrastructure_failure
from .efe_pear_float32_json import VERSION

SOURCES=['efe_pear_float32_json.py','efe_pear_yam_api_policy.py','efe_pear_yam_session_policy.py','efe_pear_yam_server.py']


def source_pins():return{name:hashlib.sha256((BASE/'cluster'/name).read_bytes()).hexdigest()for name in SOURCES}


def qualified(receipt):
    return bool(receipt.get('passed')and receipt.get('model_file_relay_passed')
        and receipt.get('native_float32_json_passed')and receipt.get('numeric_representation')==VERSION
        and receipt.get('native_numeric_sources')==source_pins()
        and receipt.get('executed_control_steps')==1 and receipt.get('cameras')
        and isinstance(receipt.get('native_termination_terms'),dict))


def tick(state):
    control=read(DEST/'control.json',{})
    if not control.get('native_float32_recovery')or control.get('paused'):return None
    if state.get('scheduler_status')!='RUNNING'or state.get('run_id')not in control.get('enabled_training_arms',[]):return None
    progress=read(DEST/'native-numeric-recovery.json',{})
    if progress.get('workflow')==state['workflow']and progress.get('phase')=='numeric_workers_submitted':return progress
    live=read(DEST/'native-numeric-probe.json',{})
    if not live.get('jobs'):return None
    progress={'workflow':state['workflow'],'phase':'validating_native_float32_json',
        'probe_workflow':live['jobs']['probe']['workflow'],
        'model_and_existing_simulators_preserved':True,'validation_jobs':{}}
    queries={};retried=False
    for role,row in live['jobs'].items():
        query=command(['osmo','workflow','query',row['workflow'],'--format-type','json']);queries[role]=query
        progress['validation_jobs'][role]={'workflow':row['workflow'],'status':query['status']}
        if not query['status'].startswith('FAILED'):continue
        if infrastructure_failure(query)and row.get('attempt',0)<8:
            from .monitor_efe_pear_full_qualification import retry
            row['scheduler_status']=query['status'];retry(row);retried=True
            write(DEST/'native-numeric-probe.json',live)
        else:progress['phase']='numeric_validation_requires_review'
    row=live['jobs']['probe'];query=queries['probe'];progress['probe_status']=query['status']
    if retried or progress['phase']=='numeric_validation_requires_review':
        write(DEST/'native-numeric-recovery.json',progress);return progress
    receipt={}
    if query.get('start_time'):
        from .monitor_efe_pear_l40s01 import records
        from ..tracking.reasoning_live import command as logs
        parsed=records(logs(['osmo','workflow','logs',row['workflow'],'-n','50'],timeout=8),'PEAR_CROSS_POOL_PROBE ')
        if parsed:receipt=parsed[-1]
    if not qualified(receipt):
        write(DEST/'native-numeric-recovery.json',progress);return progress
    write(DEST/'native-float32-proof.json',{**receipt,'validation_workflow':row['workflow']})
    from .launch_efe_pear_primary import worker_build
    from .recover_efe_pear_graphics import frozen_files
    queue=read(DEST/'primary-native-queue.json',{});originals=[]
    for item in queue.values():
        if(item.get('owner_model_workflow')!=state['workflow']or item.get('graphics_recovery_wire')
            or not item.get('pool','').startswith('groot-l40')):continue
        spec=read(Path(item['build'])/'spec.json')
        if 'generated_reset_tasks'in spec:originals.append(item)
    if len(originals)!=8:raise ValueError('The complete original native YAM task coverage is required')
    jobs=[]
    for item in originals:
        old=Path(item['build']);spec=read(old/'spec.json')
        spec.update(numeric_representation=VERSION,allocation_lane='native-float32-shadow-v1')
        files=frozen_files(old)
        files.update({name:BASE/'cluster'/name for name in SOURCES})
        workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow'];task=workflow['tasks'][0]
        build=worker_build(spec,files,'efe_pear_yam_server.py',task['image'],task['command'][0],
            **workflow['resources']['default'])
        if build.name not in queue:
            result=submit_reconciled(build,priority='LOW',pool='groot-l40-02',nodes=1)
            queue[build.name]={**result,'workflow':result['name'],'build':str(build),'pool':'groot-l40-02',
                'priority':'LOW','allocation_priority':'LOW','owner_model_workflow':state['workflow'],
                'graphics_recovery_wire':'native-float32-shadow-v1','allow_amlfs07_pool_fallback':True,
                'gpu':workflow['resources']['default']['gpu'],'scheduler_status':'PENDING',
                'native_numeric_sources':receipt['native_numeric_sources']}
            write(DEST/'primary-native-queue.json',queue)
        jobs.append(queue[build.name]['workflow'])
    progress.update(phase='numeric_workers_submitted',jobs=jobs,native_numeric_sources=source_pins(),
        numeric_representation=VERSION,validation_workflow=row['workflow'])
    write(DEST/'native-numeric-recovery.json',progress)
    return progress
