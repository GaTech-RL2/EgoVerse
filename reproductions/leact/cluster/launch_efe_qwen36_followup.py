"""Bounded diagnostic and repeat cohort after the first constrained LeAct round."""
import argparse
from datetime import datetime,timezone
import fcntl
import importlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import yaml

from .scale_efe import BASE,ROOT,BUILD_ROOT,read,write,submit,command
from .launch_efe_training import freeze_sources,recovery_reader
from .launch_efe_leact_warm96 import collect
from .launch_efe_native_followups import collect_arm,verified_native_completion
from .monitor_efe_preflight import remote

DEST=ROOT/'fixed-action-v1/leact-followup-v1'
ARCHIVE='/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-qwen36-leact-followup-20261105-v1'
PARENTS={'warmstart':ROOT/'fixed-action-v1/warmstart-lora-v2','leact':ROOT/'fixed-action-v1/leact-lora-v1'}


def now():return datetime.now(timezone.utc).isoformat()


def quota_reclaimed(state):
    """Retry scheduler reclamation; never reinterpret a user's cancellation."""
    return (state.get('status')=='FAILED_CANCELED'
        and state.get('cancelled_by')=='groot-idle-job-shutdown'
        and any('GPU quota enforcement' in (group.get('failure_message') or '')
            for group in state.get('groups',[])))


def control_failed(state):
    return (state.get('status')=='FAILED' and any(
        task.get('exit_code')==2001 and 'OSMO Control failure' in (task.get('failure_message') or '')
        for group in state.get('groups',[]) for task in group.get('tasks',[])))


def prepare_retry(original,number,kind):
    retry=Path(str(original)+'-low'+str(number));retry.mkdir(exist_ok=True)
    if not (retry/'source-manifest.json').exists():
        for name in read(original/'source-manifest.json'):shutil.copyfile(original/name,retry/name)
        workflow=yaml.safe_load((retry/'workflow.yaml').read_text())
        workflow['workflow']['name']+='-low'+str(number)
        workflow['workflow']['timeout']['queue_timeout']='12h'
        (retry/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
        if kind=='diagnostic':
            spec=read(retry/'spec.json');spec['archive_root']=ARCHIVE+'/diagnostic-low'+str(number)
            write(retry/'spec.json',spec)
        freeze_sources(retry)
    return retry


def reasoning_comparison(records):
    """Compare likelihoods within each state, retaining missing-score counts."""
    result={'states':len(records),'planned':43,
        'valid_outputs':sum(bool(r.get('valid')) for r in records),
        'exact_commands':sum(bool(r.get('exact_command')) for r in records),
        'stops':sum(bool(r.get('stop')) for r in records),
        'unfinished_outputs':sum(r.get('finish_reason')!='stop' for r in records)}
    for label,left,right in [('forward_gain','forward','empty'),('backward_gain','backward','empty'),
                             ('backward_minus_forward','backward','forward')]:
        deltas=[r[left+'_command_logp']-r[right+'_command_logp'] for r in records
            if r.get(left+'_command_logp') is not None and r.get(right+'_command_logp') is not None]
        result[label]={'states':len(deltas),'mean_nats_per_command_token':statistics.mean(deltas) if deltas else None}
    return result


def parent_ready(arm):
    directory=PARENTS[arm]
    if not read(directory/'evaluation/verified-completion.json',{}).get('success'):raise ValueError('Original comparison is not verified')
    return read(directory/('warmstart-ready.json' if arm=='warmstart' else 'leact-ready.json'))


def prepare_diagnostic():
    build=BUILD_ROOT/'efe-qwen36-action-diagnostic-20261105-v1'
    if (build/'source-manifest.json').exists():return build
    build.mkdir(exist_ok=True);source=BUILD_ROOT/'efe-qwen36-constrained-leact-20261105-v1';old=read(source/'spec.json')
    spec={'archive_root':ARCHIVE+'/diagnostic','local_models':'/tmp/efe-qwen36-diagnostic-models','seed':20261105,
        'checkpoints':{arm:parent_ready(arm) for arm in PARENTS},'decisions':old['parent_corpus']+'/decisions.jsonl.gz',
        'decisions_sha256':old['decisions_sha256'],'candidate_report':old['archive_root']+'/candidate-report.json',
        'purpose':'Greedy 4096-token support-set diagnostic only; no optimizer updates or simulator-success claims. Compare empty, forward-generated and selected backward reasoning on identical command tokens.'}
    write(build/'spec.json',spec)
    shutil.copyfile(BASE/'cluster/efe_qwen36_action_diagnostic.py',build/'diagnostic.py')
    for name in ['efe_training_data.py','efe_budget_gateway.py','efe_leact_vl.py']:shutil.copyfile(source/name,build/name)
    shutil.copyfile(source/'pipeline.py',build/'leact_reference.py')
    shutil.copyfile(BUILD_ROOT/'efe-qwen36-constrained-leact-eval-20261105-v1/efe_action_loop.py',build/'efe_action_loop.py')
    lane='/tmp/efe-qwen36-diagnostic'
    bootstrap=(source/'bootstrap.py').read_text().replace('/tmp/efe-qwen36-leact',lane).replace("lane/'pipeline.py'","lane/'diagnostic.py'")
    (build/'bootstrap.py').write_text(bootstrap)
    workflow=yaml.safe_load((source/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-qwen36-action-diagnostic-v1'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'8h'}
    task=workflow['workflow']['tasks'][0];task['args']=['-u',lane+'/bootstrap.py']
    names=['spec.json','bootstrap.py','diagnostic.py','efe_training_data.py','efe_budget_gateway.py','efe_leact_vl.py','leact_reference.py','efe_action_loop.py']
    task['files']=[{'localpath':name,'path':lane+'/'+name} for name in names]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in build.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(build);return build


def prepare_repeat(arm,seed):
    ident=f'{arm}-seed{seed}';build=BUILD_ROOT/('efe-qwen36-'+ident+'-20261105-v2')
    if (build/'source-manifest.json').exists():return build
    source=BUILD_ROOT/'efe-qwen36-constrained-leact-eval-20261105-v1';build.mkdir(exist_ok=True)
    for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,build/name)
    ready=parent_ready(arm);spec=read(build/'spec.json')
    spec['tasks']=[task for task in spec['tasks'] if task['suite']=='metaworldplus']
    if len(spec['tasks'])!=3:raise ValueError('Repeat cohort must contain exactly three trained MetaWorld tasks')
    spec.update(student_checkpoint=ready['checkpoint'],student_manifest=ready['manifest'],student_revision=ready['revision'],
        inference_seed=seed,model_id='qwen36-'+ident,agent_slug='qwen36-'+ident,agent_label='Qwen3.6 '+arm+' · seed '+str(seed),
        export_role='constrained_'+arm+'_seed'+str(seed),local_results_directory='fixed-action-v1/leact-followup-v1/'+ident,
        action_durable_root=ARCHIVE+'/native',training_parent=read(PARENTS[arm]/'status.json')['workflow'],
        purpose='Matched decoding-seed repetition, no weight updates. Separate three-task training-scene cohort.',
        preflight_protocol='constrained-image-action-preflight-v1',preflight_semantics_policy='record_and_evaluate',
        preflight_change='Exercise completed image-to-action Responses through the same gateway, with no tools. Record incorrect colors/commands and grade them on native tasks; do not exclude seeds using an unused generic-tool capability.',
        scope='Two predeclared additional engine seeds (101, 202), same three trained MetaWorld scenes and native budgets for both warmstart and LeAct. The untrained faucet task is excluded from this separate cohort. Original four-task results remain unchanged.')
    spec['sampling']+=' Explicit vLLM engine seed '+str(seed)+'. Requests retain their original sampling settings.'
    write(build/'spec.json',spec)
    p=build/'native-student.py';text=p.read_text();hook='    transport_args,transport_env=serving_transport_options(spec)\n'
    if text.count(hook)!=1:raise ValueError('Serving seed insertion point changed')
    text=text.replace(hook,"    command.extend(['--seed',str(spec['inference_seed'])])\n"+hook)
    old="        return vision_tool_preflight({**spec,'upstream':'http://127.0.0.1:19862/v1'},token)"
    if text.count(old)!=1:raise ValueError('Preflight insertion point changed')
    text=text.replace(old,"        from efe_constrained_preflight import check\n        return check({**spec,'upstream':'http://127.0.0.1:19862/v1'},token,response)")
    text=text.replace('Exercise the same HTTP gateway and tool events used by native Codex.','Exercise the same streamed image/action interface used by the constrained policy.')
    text=text.replace('Qwen server ready; testing streamed image recognition and tool-result continuation.','Qwen server ready; testing the streamed constrained image/action interface.')
    text=text.replace('three native training-scene trials follow image/tool qualification.','three native training-scene trials follow the image/action transport probe.')
    p.write_text(text)
    for name in ['efe_constrained_preflight.py','efe_action_artifacts.py']:shutil.copyfile(BASE/'cluster'/name,build/name)
    workflow=yaml.safe_load((build/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-qwen36-'+ident+'-v2'
    workflow['workflow']['tasks'][0]['files'].append({'localpath':'efe_constrained_preflight.py',
        'path':'/tmp/efe-training-evaluation-v1/warm96/efe_constrained_preflight.py'})
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in build.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(build);return build


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'plan.json').exists():return read(DEST/'plan.json')
    jobs=[{'id':'diagnostic','kind':'diagnostic','build':str(prepare_diagnostic()),
        'directory':'fixed-action-v1/leact-followup-v1/diagnostic'}]
    for seed in [101,202]:
        for arm in ['warmstart','leact']:
            ident=f'{arm}-seed{seed}'
            jobs.append({'id':ident,'kind':'native','arm':arm,'seed':seed,'build':str(prepare_repeat(arm,seed)),
                'directory':'fixed-action-v1/leact-followup-v1/'+ident})
    plan={'created_utc':now(),'maximum_concurrent_nodes':2,'priority':'HIGH','capacity_fallback':'LOW','jobs':jobs,
        'parent_results':{'warmstart':{'successes':1,'graded':4},'leact':{'successes':0,'graded':4}},
        'scope':'Diagnostic, not another training round: fixed-observation imitation/likelihood for both checkpoints; two additional engine seeds on the same three teacher-successful training tasks.',
        'repeat_analysis':'Report every seed/task result. Keep the new three-task cohort separate from the original four-task score; no best-seed or best-checkpoint selection.',
        'next_decision':'Use support-set action errors, forward-vs-backward reasoning likelihood gaps and rollout repeatability to choose data correction, more training support, or a new LeAct configuration. No blind extra optimizer round.'}
    for job in jobs:(ROOT/job['directory']).mkdir(parents=True,exist_ok=True)
    write(DEST/'plan.json',plan);return plan


def publish(status):
    from ..embodied_first_exam import report
    write(DEST/'status.json',status)
    importlib.reload(report).publish()


def monitor():
    plan=prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    capacity={'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'}
    while True:
        try:
            jobs=[];live=[];active_count=0
            for original in plan['jobs']:
                folder=ROOT/original['directory'];active=read(folder/'active-build.json',{});build=Path(active.get('build',original['build']))
                receipt=read(build/'submission.json',{})
                item={**original,'build':str(build),'workflow':receipt.get('name'),'status':'NOT_SUBMITTED',
                    'priority':active.get('priority','HIGH'),'retry_number':active.get('retry_number',0)}
                if receipt:
                    state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(folder/'scheduler.json',state)
                    item['status']=state['status']
                    if state['status'] in {'RUNNING','PENDING','WAITING'}:active_count+=1
                    if state['status']=='RUNNING':live.append(receipt['name'])
                    done=read(folder/'verified-completion.json',{})
                    if not done and (quota_reclaimed(state) or control_failed(state) or (state['status'] in capacity and not state.get('start_time'))):
                        # Keep every interrupted attempt's evidence and its durable
                        # archive locator before retrying the identical recipe.
                        attempts=folder/'attempts'/receipt['name'];attempts.mkdir(parents=True,exist_ok=True)
                        for path in list(folder.iterdir()):
                            if path.name in {'attempts','active-build.json'}:continue
                            if path.is_file() or path.name in {'public-episodes','student-prompt-requests','retained'}:
                                shutil.move(str(path),str(attempts/path.name))
                        write(attempts/'interruption.json',{'updated_utc':now(),'build':str(build),
                            'workflow':receipt['name'],'scheduler':state,
                            'reason':'scheduler_quota_reclamation' if quota_reclaimed(state) else 'osmo_control_failure' if control_failed(state) else 'allocation_failure',
                            'durable_archive':read(build/'spec.json')['archive_root'] if original['kind']=='diagnostic' else ARCHIVE+'/native/'+receipt['name'],
                            'retry_priority':'LOW','not_a_completed_task_score':True})
                        number=active.get('retry_number',0)+1;retry=prepare_retry(Path(original['build']),number,original['kind'])
                        write(folder/'active-build.json',{'build':str(retry),'retry_number':number,'priority':'LOW'})
                        item.update(build=str(retry),workflow=None,status='NOT_SUBMITTED',priority='LOW',retry_number=number)
                jobs.append(item)
            # A running cohort node can read all shared-storage artifacts. Only
            # allocate a receipt reader after every primary job has been submitted.
            reader=live[0] if live else None
            if not reader and all(j['workflow'] for j in jobs) and not active_count:
                pending=[j for j in jobs if j['status']=='COMPLETED' and not read(ROOT/j['directory']/'verified-completion.json',{})]
                if pending:reader=recovery_reader(Path(pending[0]['build']))
            for item in jobs:
                folder=ROOT/item['directory'];workflow=item['workflow'];done=read(folder/'verified-completion.json',{})
                if workflow and not done:
                    host=workflow if item['status']=='RUNNING' else reader if item['status'] in {'COMPLETED','FAILED'} else None
                    if host and item['kind']=='diagnostic':
                        archive=read(Path(item['build'])/'spec.json')['archive_root']
                        collect(host,archive,folder,['worker-start.json','progress.json','complete.json',
                            'warmstart-records.json','warmstart-summary.json','warmstart-complete.json','warmstart.log',
                            'leact-records.json','leact-summary.json','leact-complete.json','leact.log','bootstrap-error.json'])
                        done=read(folder/'complete.json',{})
                        if done:
                            write(folder/'verified-completion.json',done)
                            remote(host,"from pathlib import Path;Path("+repr(archive+'/collected')+").touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                    elif host:
                        arm={'directory':item['directory'],'require_retained':True,'retained_storage':'durable'}
                        if host!=workflow or item['status']!='RUNNING':arm.update(host_workflow=host,remote_archive=ARCHIVE+'/native/'+workflow)
                        collect_arm(arm,workflow);done=verified_native_completion(arm)
                item.update(complete=done,progress=read(folder/'progress.json',{}))
                if item['kind']=='native':item['behavior']=read(folder/'behavior.json',{})
                else:
                    item['summaries']={a:read(folder/(a+'-summary.json'),{}) for a in PARENTS}
                    item['reasoning_comparison']={a:reasoning_comparison(read(folder/(a+'-records.json'),{}).get('records',[])) for a in PARENTS}
            for item in jobs:
                if active_count>=plan['maximum_concurrent_nodes']:break
                if item['status']!='NOT_SUBMITTED':continue
                folder=ROOT/item['directory'];active=read(folder/'active-build.json',{})
                receipt=submit(Path(item['build']),priority=active.get('priority','HIGH'))
                item.update(workflow=receipt['name'],status='PENDING');active_count+=1
            for item in jobs:
                if item['status']=='NOT_SUBMITTED':
                    item['progress']={'phase':'waiting_for_cohort_slot','detail':
                        f'Waiting for a slot within the two-node limit; {item["priority"]} priority'
                        +(' retry after infrastructure interruption.' if item['retry_number'] else '.')}
            completed=sum(bool(j['complete'].get('success')) for j in jobs)
            repeats={a:{'successes':sum(j.get('behavior',{}).get('successes',0) for j in jobs if j.get('arm')==a),
                'graded':sum(j.get('behavior',{}).get('graded',0) for j in jobs if j.get('arm')==a),'planned':6} for a in PARENTS}
            summary=(f'LeAct follow-up: {completed}/5 jobs complete. Two additional decoding seeds compare both checkpoints on three trained MetaWorld tasks; '
                f'warmstart repeats {repeats["warmstart"]["successes"]}/{repeats["warmstart"]["graded"]} graded, '
                f'LeAct repeats {repeats["leact"]["successes"]}/{repeats["leact"]["graded"]} graded. '
                'The 43-observation diagnostic tests action imitation and forward/backward reasoning gaps without updating weights.')
            status={'updated_utc':now(),'summary':summary,'jobs':jobs,'completed_jobs':completed,'total_jobs':len(jobs),
                'maximum_concurrent_nodes':2,'repeats':repeats,'all_complete':completed==len(jobs)}
            publish(status)
            if any(j.get('behavior',{}).get('graded') for j in jobs):
                from ..embodied_first_exam.direct_control_viewer import build as viewer
                viewer()
            if status['all_complete']:
                if reader and not live:remote(reader,"from pathlib import Path;Path('/tmp/efe-reader-collected').touch();print('EFE_CAPTURE {\"ack\":true}',flush=True)")
                return
            if any(j['complete'] and not j['complete'].get('success') for j in jobs):return
        except Exception as error:write(DEST/'monitor-error.json',{'updated_utc':now(),'error':str(error)[-2000:]})
        time.sleep(30)


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,'monitor'],cwd=BASE.parents[1],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':process.pid,'updated_utc':now()});return process.pid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','monitor','start']);args=p.parse_args()
    if args.action=='monitor':monitor()
    elif args.action=='start':print(json.dumps({'pid':start()}))
    else:print(json.dumps(prepare(),indent=2))
