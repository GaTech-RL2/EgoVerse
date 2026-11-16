"""Frozen matched train-scene and development panels after corpus admission."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,command
from .launch_efe_training import freeze_sources
from .compute_budget import reserved_nodes,node_limit
from .efe_coverage_retry import retry_due,submission_priority,native_disconnect_retry_tasks,reviewed_preflight_retry

PREFIX='leact-efe-coverage-eval-'


def register_intermediates(frozen,queue,registry,panel_sha256):
    """Add separately named fixed-update screens only on the frozen panel."""
    ready={}
    for run in registry.get('runs',[]):
        model=run['id'];step=run['step']
        if model!=f'warmstart-step{step}' or type(step) is not int or step<1:
            raise ValueError('Invalid intermediate checkpoint identity')
        if run.get('panel_sha256')!=panel_sha256 or run.get('planned_tasks')!=len(frozen['tasks']):
            raise ValueError('Intermediate evaluation panel differs from the frozen baselines')
        for index in range(0,len(frozen['tasks']),4):
            ident=f'b{index//4+1:02d}';tasks=frozen['tasks'][index:index+4]
            existing=next((b for b in queue['batches'] if (b['model'],b['id'])==(model,ident)),None)
            if existing is None:queue['batches'].append({'id':ident,'model':model,'tasks':tasks,'attempts':[]})
            elif existing['tasks']!=tasks:raise ValueError('Intermediate batch tasks changed')
        value=run.get('ready')
        if value is None:continue
        replay=value.get('loss_replay',{});error=replay.get('absolute_error')
        if not (value.get('step')==step and value.get('final') is False
                and value.get('pin_sha256')==run.get('pin_sha256')
                and isinstance(run.get('pin_sha256'),str) and re.fullmatch('[0-9a-f]{64}',run['pin_sha256'])
                and value.get('additional_optimizer_updates')==0 and replay.get('passed') is True
                and replay.get('optimizer_updates')==0 and isinstance(error,(int,float)) and 0<=error<=.0005
                and all(value.get(k) for k in ['checkpoint','manifest','revision'])):
            raise ValueError('Intermediate checkpoint has not passed exact export verification')
        ready[model]=value
    return ready


def register_branches(frozen,queue,runs,training,panel_sha256):
    ready={}
    for run in runs:
        model=run['id']
        if not re.fullmatch(r'leact-from-step[1-9][0-9]*-r[1-9][0-9]*',model):raise ValueError('Invalid LeAct branch identity')
        if run['panel_sha256']!=panel_sha256 or run['planned_tasks']!=len(frozen['tasks']):
            raise ValueError('LeAct branch panel differs from its warmstart comparison')
        for index in range(0,len(frozen['tasks']),4):
            ident=f'b{index//4+1:02d}';tasks=frozen['tasks'][index:index+4]
            old=next((b for b in queue['batches'] if (b['model'],b['id'])==(model,ident)),None)
            if old is None:queue['batches'].append({'id':ident,'model':model,'tasks':tasks,'attempts':[]})
            elif old['tasks']!=tasks:raise ValueError('LeAct branch tasks changed')
        stage=training.get(model,{})
        if stage.get('complete',{}).get('success') and not stage['complete'].get('no_admissions'):
            value=stage.get('ready',{})
            if value.get('final') is not True or value.get('checkpoint')!=run['archive_root']+'/model':
                raise ValueError('LeAct branch export does not match its completed training run')
            ready[model]=value
    return ready


def balanced(rows,count):
    groups={}
    for r in sorted(rows,key=lambda r:hashlib.sha256(('coverage-eval-v1/'+r['task']).encode()).hexdigest()):
        groups.setdefault(r['suite'],[]).append(r)
    result=[]
    while len(result)<count and any(groups.values()):
        for suite in sorted(groups):
            if groups[suite] and len(result)<count:result.append(groups[suite].pop(0))
    return result


def panel(plan,admission):
    from .launch_efe_coverage import DEST,now
    path=DEST/'evaluation-panel.json'
    if path.exists():return read(path)
    inventory=read(ROOT/'generalization-v2/inventory.json')
    splits=read(ROOT/'generalization-v2/splits.json')
    admitted={}
    for row in admission['episodes']:admitted.setdefault(row['task'],row)
    suites={r['suite'] for r in admission['episodes']}
    train=[dict(r,partition='training_evaluation',evaluation_cohort='seen_training_scene',collection_attempt=1,
        teacher_initial_state_sha256=admitted[r['task']]['initial_state_sha256']) for r in inventory if r['task'] in admitted]
    represented={(r['simulator_runtime'],r['semantic_task']) for r in train}
    development=[dict(r,partition='development',evaluation_cohort=(
        'same_task_new_variant' if (r['simulator_runtime'],r['semantic_task']) in represented else 'same_simulator_new_task'),collection_attempt=1)
        for r in inventory if r['task'] in splits['development'] and r['suite'] in suites and r['mode']=='standard']
    tasks=balanced(train,24)+balanced(development,12)
    value={'created_utc':now(),'tasks':tasks,'cohort_counts':dict(Counter(r['evaluation_cohort'] for r in tasks)),
        'conditions':['astra','base','warmstart','leact-r1'],'policy_attempts_per_condition_task':1,
        'selection':'Task names are frozen from the admitted corpus and original development partition before model evaluations; identical tasks/budgets across conditions.',
        'limits':'Initial capability screen, one stochastic rollout per task/condition. Classify same-task new variants versus same-simulator new tasks from actual admitted task families; report only populated cohorts. Held-out Genesis/Sapien runtime/graphics qualification is separate and their scores remain unmeasured. No official full-core claim.'}
    write(path,value)
    return value


def build(model,batch,ready,retry):
    from .launch_efe_coverage import DEST,DURABLE,TEMPLATE,LANE,sha,harden_gateway_shutdown
    suffix=f'-r{retry}' if retry else ''
    folder=BUILD_ROOT/f'efe-coverage-eval-20261105-v1-{model}-{batch["id"]}{suffix}'
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(exist_ok=True)
    source=TEMPLATE if model=='astra' else BUILD_ROOT/'efe-fixed-action-20261104-v4-qwen36-task'
    for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,folder/name)
    spec=read(folder/'spec.json')
    spec.update(tasks=batch['tasks'],model_id='coverage-'+model,agent_slug='coverage-'+model,
        agent_label='Expanded coverage · '+model,export_role='coverage_evaluation_'+model,
        coverage_evaluation=True,coverage_root=DURABLE,coverage_plan_sha256=sha(DEST/'plan.json'),
        protocol_version='efe-constrained-action-v4-task',
        action_durable_root=DURABLE+'/evaluation/'+model,
        local_results_directory='fixed-action-v1/coverage-scale-v1/evaluation/'+model+'/'+batch['id'],
        training_admission='Evaluation only; never admitted to training.',
        purpose='Matched native capability panel after expanded training-corpus admission.',
        scope='Fixed training scenes and untouched same-simulator development tasks, reported separately. Same constrained interface, native task seeds, and budgets for teacher and students.')
    spec['gateway_cleanup_policy']='Terminate for 20 seconds, then kill and wait for 10 seconds; continue native result recording after forced shutdown.'
    spec.pop('training_parent',None)
    if model!='astra':
        spec.update(student_checkpoint=ready['checkpoint'],student_manifest=ready['manifest'],student_revision=ready['revision'])
        if ready.get('final') is False:
            spec['intermediate_checkpoint']={k:ready[k] for k in ['step','final','selection','pin_sha256','additional_optimizer_updates']}
        boot=(folder/'native-bootstrap.py').read_text()
        line="subprocess.run([python, '-u', str(lane / 'efe_prepare_qualification_student.py')], env=env, check=True)\n"
        if boot.count(line)!=1:raise ValueError('Qualification bootstrap changed')
        (folder/'native-bootstrap.py').write_text(boot.replace(line,''))
        student=folder/'native-student.py';text=student.read_text()
        marker="    calls=[x for x in first.get('output',[]) if x['type']=='function_call']"
        if text.count(marker)!=1:raise ValueError('Synthetic preflight diagnostic hook changed')
        student.write_text(text.replace(marker,"    runtime.write('student-preflight-response.json',first)\n"+marker))
    write(folder/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_coverage_teacher.py',folder/'native-worker.py')
    shutil.copyfile(BASE/'cluster/efe_action_artifacts.py',folder/'efe_action_artifacts.py')
    for name in ['efe_action_loop.py','efe_action_agent.py']:shutil.copyfile(BASE/'cluster'/name,folder/name)
    p=folder/'native-pilot.py';p.write_text(harden_gateway_shutdown(p.read_text().replace('for _ in range(4320):','for _ in range(12):')))
    workflow=yaml.safe_load((folder/'workflow.yaml').read_text())
    workflow['workflow']['name']=PREFIX+model+'-'+batch['id']+'-v1'+suffix
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'12h'}
    if model!='astra':workflow['workflow']['tasks'][0].pop('credentials',None)
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in folder.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(folder)
    return folder


def tick(plan,evidence,active):
    from .launch_efe_coverage import DEST,PREFIX as COLLECTION_PREFIX,now,infra_retryable
    admission=evidence.get('admission',{})
    if not admission.get('gate_passed'):return
    frozen=panel(plan,admission)
    queue=read(DEST/'evaluation-queue.json',{'batches':[]})
    if not queue['batches']:
        for model in frozen['conditions']:
            for index in range(0,len(frozen['tasks']),4):
                queue['batches'].append({'id':f'b{index//4+1:02d}','model':model,
                    'tasks':frozen['tasks'][index:index+4],'attempts':[]})
    training=evidence.get('training',{})
    ready={'astra':{},'base':evidence.get('base_ready')}
    ready.update(register_intermediates(frozen,queue,read(DEST/'intermediate-evaluations.json',{}),
        hashlib.sha256((DEST/'evaluation-panel.json').read_bytes()).hexdigest()))
    branches=read(DEST/'branch-runs.json',{}).get('runs',[])
    ready.update(register_branches(frozen,queue,branches,training,
        hashlib.sha256((DEST/'evaluation-panel.json').read_bytes()).hexdigest()))
    for stage in ['warmstart','leact-r1']:
        if training.get(stage,{}).get('complete',{}).get('success') and training[stage].get('ready'):
            ready[stage]=training[stage]['ready']
    running=sum(r['name'].startswith(PREFIX) for r in active)
    authorization=read(ROOT/'compute-authorization.json',{})
    max_nodes=(1 if any(r['name'].startswith(COLLECTION_PREFIX) for r in active)
               else authorization.get('max_parallel_evaluation_nodes',3))
    slots=max(0,min(max_nodes-running,node_limit()-reserved_nodes(active)))
    results=evidence.get('evaluations',{})
    states=[]
    aggregate={}
    for batch in queue['batches']:
        model=batch['model']
        if (model=='leact-r1' or model in {r['id'] for r in branches}) and training.get(model,{}).get('complete',{}).get('no_admissions'):
            states.append({'model':model,'id':batch['id'],'status':'NO_ADMISSIONS_USE_WARMSTART',
                           'graded':0,'planned':len(batch['tasks']),'workflow':None})
            continue
        graded={}
        for attempt in batch['attempts']:
            evidence_row=results.get(attempt['workflow'],{})
            public={r['task']:r for r in evidence_row.get('public_manifest',{}).get('episodes',[])}
            expected={r['task'] for r in batch['tasks'] if r.get('teacher_initial_state_sha256')}
            for row in evidence_row.get('results',{}).get('trials',[]):
                if row['status']=='graded':
                    if row['task'] in expected and not public.get(row['task'],{}).get('initial_state_matches_teacher'):
                        continue
                    if row['task'] in graded:raise ValueError('Duplicate graded evaluation attempt')
                    graded[row['task']]=row
        tasks={r['task']:r for r in batch['tasks']}
        for task,row in graded.items():
            cohort=tasks[task]['evaluation_cohort'];bucket=aggregate.setdefault((model,cohort),[]);bucket.append(row)
        left=[r for r in batch['tasks'] if r['task'] not in graded]
        retry=False
        if not left:state='COLLECTED'
        elif batch['attempts']:
            attempt=batch['attempts'][-1]
            scheduler=command(['osmo','workflow','query',attempt['workflow'],'--format-type','json'])
            state=scheduler['status'];write(Path(attempt['build'])/'scheduler.json',scheduler)
            ended=scheduler.get('end_time')
            after_end=ended and evidence.get('captured_utc') and datetime.fromisoformat(evidence['captured_utc']).timestamp()>=datetime.fromisoformat(ended).replace(tzinfo=timezone.utc).timestamp()
            review=read(DEST/'preflight-reviews'/(attempt['workflow']+'.json'),{})
            if bool(after_end) and reviewed_preflight_retry(scheduler,results.get(attempt['workflow'],{}),review,
                    hashlib.sha256((Path(attempt['build'])/'spec.json').read_bytes()).hexdigest()):
                retry=True;state='WAITING_FOR_PREFLIGHT_RETRY'
            elif bool(after_end) and infra_retryable(scheduler):
                retry=retry_due(batch,attempt['workflow'])
                if not retry:state='RETRY_BACKOFF'
            elif bool(after_end) and state=='COMPLETED':
                row=results.get(attempt['workflow'],{})
                retry_tasks=native_disconnect_retry_tasks(read(Path(attempt['build'])/'spec.json'),row)
                if retry_tasks and set(retry_tasks)=={r['task'] for r in left}:
                    batch['native_infrastructure_review']={
                        'workflow':attempt['workflow'],'reviewed_utc':now(),'tasks':retry_tasks,
                        'evidence_sha256':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
                        'reason':'Recorded ungraded RemoteDisconnected interruption followed only by untouched tasks; preserve every graded outcome.'}
                    retry=retry_due(batch,attempt['workflow'])
                    if not retry:state='RETRY_BACKOFF'
        else:state='WAITING_FOR_CHECKPOINT'
        if left and model in ready and ready[model] is not None and (not batch['attempts'] or retry) and slots>0:
            folder=build(model,{**batch,'tasks':left},ready[model],len(batch['attempts']))
            priority=submission_priority(batch)
            receipt=submit(folder,priority=priority)
            batch['attempts'].append({'workflow':receipt['name'],'build':str(folder),'submitted_utc':now(),'priority':priority})
            write(DEST/'evaluation-queue.json',queue)
            state='PENDING';slots-=1
        states.append({'model':model,'id':batch['id'],'status':state,'graded':len(graded),'planned':len(tasks),
            'workflow':batch['attempts'][-1]['workflow'] if batch['attempts'] else None})
    write(DEST/'evaluation-queue.json',queue)
    scores=[{'model':model,'cohort':cohort,'graded':len(rows),'successes':sum(r['score'] for r in rows),
             'success_rate':sum(r['score'] for r in rows)/len(rows)} for (model,cohort),rows in aggregate.items()]
    write(DEST/'evaluation-status.json',{'updated_utc':now(),'jobs':states,'scores':scores,'panel':frozen['cohort_counts'],
        'limits':frozen['limits']})
