"""Submit the admitted EFE warmstart pilot and advance to native development eval."""
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
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BUILD_ROOT, BASE, ROOT, read, write, submit, monitor as native_monitor, command
from .monitor_efe_preflight import remote
from ..embodied_first_exam import report


def freeze_sources(build):
    write(build/'source-manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in build.iterdir() if p.is_file()})


def recovery_reader(build):
    """Recover persisted receipts if a local outage outlives the original worker."""
    reader=build/'completion-reader';reader.mkdir(exist_ok=True)
    if not (reader/'source-manifest.json').exists():
        parent=read(build/'submission.json')['name']
        suffix=hashlib.sha256(parent.encode()).hexdigest()[:12]
        (reader/'reader.py').write_text(
            "from pathlib import Path\nimport time\n"
            "print('Bounded persisted-artifact recovery; no model work',flush=True)\n"
            "for _ in range(60):\n"
            " if Path('/tmp/efe-reader-collected').exists():break\n"
            " time.sleep(5)\n")
        write(reader/'spec.json',{'parent':parent,'lifetime_seconds':300,
              'resource_note':'Pool validation requires a full 8-GPU node even for this short receipt read.'})
        workflow={'workflow':{'name':'leact-efe-receipt-'+suffix,
            'resources':{'default':{'gpu':8,'cpu':8,'memory':'32Gi','storage':'16Gi','platform':'dgx-h100'}},
            'timeout':{'queue_timeout':'2h','exec_timeout':'10m'},'tasks':[{
                'name':'preflight','image':'docker.io/vllm/vllm-openai@sha256:4801151759655c57606c844662e5213403c032a62d149c7ce61d615759a821ef',
                'command':['python3'],'args':['/tmp/reader.py'],
                'files':[{'localpath':'reader.py','path':'/tmp/reader.py'}]}]}}
        (reader/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(reader)
    receipt=submit(reader)
    state=command(['osmo','workflow','query',receipt['name'],'--format-type','json'])
    if state['status']=='RUNNING':return receipt['name']
    if state['status']=='COMPLETED' or state['status'].startswith('FAILED'):
        raise RuntimeError('Bounded receipt reader ended before collection; inspect '+receipt['name'])
    return None


def build_run():
    build=BUILD_ROOT/'efe-training-pilot-20261027-v1';build.mkdir(exist_ok=True)
    if (build/'source-manifest.json').exists():return build
    audit=ROOT/'training-pilot/scene-audit-v1.json'
    if not audit.exists():raise ValueError('Scene audit is required before submission')
    shutil.copyfile(audit,build/'scene-audit.json')
    spec={'archive_root':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-training-pilot-20261027-v1',
          'base_model':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1/base-model',
          'base_manifest':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-student-v1/checkpoint-manifest.json',
          'base_revision':'92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b',
          'scene_audit':'/tmp/efe-scene-audit.json','scene_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
          'student_prompt_requests':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-qwen-base-20261026-v4/leact-efe-qwen-base-v4-1/requests/00-qwen-base',
          'pilot_decisions':256,'context':32768,'effective_batch':32,'lr':1e-6,'seed':20261027,
          'teacher_model':'azure/openai/gpt-6-astra','teacher_concurrency':4,'rationale_max_tokens':8192,
          'local_results_directory':'training-pilot/run-v1',
          'recipe_sha256':hashlib.sha256((ROOT/'training-recipe-v2.json').read_bytes()).hexdigest(),
          'scope':'Three-suite development pilot at initial LR=1e-6; fixed 0.25/0.5/1 pass snapshots. Not the selected broad recipe or official core score.',
          'next':'Automatically submit final fixed-pass native development evaluation; select/freeze the production recipe only on development results.'}
    write(build/'spec.json',spec)
    mapping=[('efe_training_pilot.py','runner.py','/tmp/efe_training_pilot.py'),
             ('efe_training_data.py','data.py','/tmp/efe_training_data.py'),
             ('efe_train_vl.py','trainer.py','/tmp/efe_train_vl.py'),
             ('efe_budget_gateway.py','gateway.py','/tmp/efe_budget_gateway.py'),
             ('text_capability_eval.py','cuda_helpers.py','/tmp/efe_cuda_helpers.py')]
    for source,target,_ in mapping:shutil.copyfile(BASE/'cluster'/source,build/target)
    (build/'bootstrap.sh').write_text('set -euo pipefail\nexport TOKENIZERS_PARALLELISM=false\npython3 -u /tmp/efe_training_pilot.py\n')
    teacher=yaml.safe_load((BUILD_ROOT/'efe-astra-pilot-resume-20261026-v3/workflow.yaml').read_text())['workflow']['tasks'][0]
    workflow={'workflow':{'name':'leact-efe-training-pilot','resources':{'default':{
        'gpu':8,'cpu':32,'memory':'256Gi','storage':'512Gi','platform':'dgx-h100'}},
        'timeout':{'queue_timeout':'12h','exec_timeout':'12h'},'tasks':[{
        'name':'preflight','image':'docker.io/vllm/vllm-openai@sha256:4801151759655c57606c844662e5213403c032a62d149c7ce61d615759a821ef',
        'privileged':True,'command':['bash'],'args':['/tmp/efe-training-bootstrap.sh'],
        'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','NVIDIA_DRIVER_CAPABILITIES':'all','OMP_NUM_THREADS':'2'},
        'credentials':teacher['credentials'],
        'files':[{'localpath':target,'path':remote_path} for _,target,remote_path in mapping]+[
            {'localpath':'bootstrap.sh','path':'/tmp/efe-training-bootstrap.sh'},
            {'localpath':'spec.json','path':'/tmp/efe-training-spec.json'},
            {'localpath':'scene-audit.json','path':'/tmp/efe-scene-audit.json'}]}]}}
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build);return build


def eval_build(ready, parent_workflow, stage='warmstart'):
    build=BUILD_ROOT/f'efe-{stage}-pilot-dev-20261027-v1';build.mkdir(exist_ok=True)
    if (build/'source-manifest.json').exists():return build
    source=BUILD_ROOT/'efe-qwen-base-20261027-v5';spec=read(source/'spec.json')
    spec.update(student_checkpoint=ready['checkpoint'],student_manifest=ready['manifest'],student_revision=ready['revision'],
        agent_label='Qwen '+stage+' pilot',agent_slug='qwen-'+stage+'-pilot',
        archive_root=f'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-{stage}-pilot-dev-20261027-v1',
        local_results_directory=f'training-pilot/{stage}-evaluation',training_parent=parent_workflow,
        purpose='Fixed '+stage+' development pilot on the same seven development tasks as tool-aligned base Qwen. No core evaluation.',
        local_episode_outputs=True,
        preflight_semantics_policy='record_and_evaluate',
        preflight_change='Record wrong synthetic answers after valid HTTP and native function-call transport; task prompts, sampling and scoring are unchanged.',
        retry_of={})
    if stage=='warm-continue32-96':
        transport=read(ROOT/'training-pilot/native-step32-transport-review.json',{})
        if transport.get('validated_native_tasks')==7 and transport.get('mode')=='nccl_no_p2p_v1':
            spec['serving_transport']=transport
            spec['comparison_to_step32']='Same seven development tasks and checkpoint serving recipe as the completed step-32 evaluation, including its recorded communication workaround.'
    write(build/'spec.json',spec)
    for name in ['runtime.py','pilot.py','gateway.py','bootstrap.sh']:shutil.copyfile(source/name,build/name)
    shutil.copyfile(BASE/'cluster/efe_astra_pilot.py',build/'pilot.py')
    # Synthetic answer correctness is recorded separately from task performance;
    # the verified continuation transport workaround is explicit in its spec.
    shutil.copyfile(BASE/'cluster/efe_qwen_baseline.py',build/'student.py')
    workflow=yaml.safe_load((source/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-'+stage+'-pilot-dev'
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build);return build


def leact_build(ready, warm_build):
    build=BUILD_ROOT/'efe-leact-pilot-20261027-v1';build.mkdir(exist_ok=True)
    if (build/'source-manifest.json').exists():return build
    warm=read(warm_build/'spec.json');parent=read(warm_build/'submission.json')['name']
    spec={**warm,'stage':'leact','ready_name':'leact-ready.json','balanced_passes':.5,
          'base_model':ready['checkpoint'],'base_manifest':ready['manifest'],'base_revision':ready['revision'],
          'tokenizer_base':warm['base_model'],'parent_corpus':warm['archive_root']+'/'+parent,
          'archive_root':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-leact-pilot-20261027-v1',
          'local_results_directory':'training-pilot/leact-r1-run-v1','leact_decisions':128,'candidate_tokens':16384,
          'scope':'One half-pass N=8/top-2-positive LeAct development pilot with 50% warmstart replay. Frozen warmstart states isolate the change in reasoning source.',
          'next':'Native development evaluation on the same frozen seven tasks; no automatic additional LeAct rounds or core selection.'}
    write(build/'spec.json',spec)
    mapping=[('efe_leact_vl.py','runner.py','/tmp/efe_leact_vl.py'),
             ('efe_training_pilot.py','pilot_helpers.py','/tmp/efe_training_pilot.py'),
             ('efe_training_data.py','data.py','/tmp/efe_training_data.py'),
             ('efe_train_vl.py','trainer.py','/tmp/efe_train_vl.py'),
             ('efe_budget_gateway.py','gateway.py','/tmp/efe_budget_gateway.py'),
             ('text_capability_eval.py','cuda_helpers.py','/tmp/efe_cuda_helpers.py')]
    for source,target,_ in mapping:shutil.copyfile(BASE/'cluster'/source,build/target)
    (build/'bootstrap.sh').write_text('set -euo pipefail\nexport TOKENIZERS_PARALLELISM=false\npython3 -u /tmp/efe_leact_vl.py\n')
    workflow=yaml.safe_load((warm_build/'workflow.yaml').read_text());workflow['workflow']['name']='leact-efe-leact-pilot'
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None)
    task['files']=[{'localpath':target,'path':path} for _,target,path in mapping]+[
        {'localpath':'bootstrap.sh','path':'/tmp/efe-training-bootstrap.sh'},
        {'localpath':'spec.json','path':'/tmp/efe-training-spec.json'}]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build);return build


def start_native_recovery(build,stage):
    """Keep the final continuation evaluation retryable independently of training."""
    marker=build/'native-recovery-started.json'
    if marker.exists():return
    plan_path=ROOT/f'training-pilot/{stage}-native-recovery.json'
    state_path=plan_path.with_name(plan_path.stem+'-status.json')
    if not plan_path.exists():
        spec=read(build/'spec.json')
        write(plan_path,{'enabled':True,'attempt':1,'build':str(build),'priority':'HIGH',
            'evaluation_stage':stage,'continuation_arm_id':'qwen-warm-continue32-96',
            'build_prefix':f'efe-{stage}-native-retry-',
            'workflow_prefix':f'leact-efe-{stage}-native-r',
            'local_results_prefix':f'training-pilot/{stage}-evaluation-retry-',
            'serving_transport':spec.get('serving_transport'),
            'policy':'One native evaluation node; HIGH initially, LOW after confirmed capacity interruption. Keep every graded result, including failures, and retry only unfinished tasks after verified infrastructure interruptions.'})
    subprocess.run([sys.executable,'-m','reproductions.leact.cluster.recover_efe_native',
        'start','--plan',str(plan_path),'--state',str(state_path)],cwd=BASE.parents[1],
        capture_output=True,text=True,check=True,timeout=30)
    write(marker,{'plan':str(plan_path),'state':str(state_path),'started_utc':datetime.now(timezone.utc).isoformat()})


def capture(workflow, archive):
    names=['progress.json','complete.json','trace-admission.json','corpus-manifest.json','pilot-membership.json',
           'warmstart-ready.json','leact-ready.json','membership.json','selection.json','reference-policy-audit.json',
           'versions.json','processor.json','cuda-preflight.json','smoke/complete.json','train/complete.json',
           'smoke/metrics.jsonl','train/metrics.jsonl','train/configuration.json','train/schedule.json',
           'smoke/scoring-parity.json','train/scoring-parity.json','smoke.log','train.log']
    names+=['optimizer-resume.json','restart-ready.json','memory-preflight-reuse.json','train/checkpoint-timings.jsonl','storage-cleanup.json','storage-capacity-verification.json']
    script="""from pathlib import Path
import base64,hashlib,json,time
root=Path(ARCHIVE);files={}
for name in NAMES:
 p=root/name
 if p.exists():
  data=p.read_bytes();files[name]={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
s='EFE_CAPTURE '+json.dumps({'files':files})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.005)
""".replace('ARCHIVE',repr(archive)).replace('NAMES',repr(names))
    return remote(workflow,script)


def register_retention(ready, stage):
    from ..embodied_first_exam.capability_retention import DEST
    registry=read(DEST/'checkpoints.json');ident='qwen-'+stage
    checkpoint=next(r for r in registry['checkpoints'] if r['id']==ident)
    if checkpoint['status']=='waiting_for_checkpoint':
        checkpoint.update(path=ready['checkpoint'],manifest=ready['manifest'],status='ready',
                          selection='Fixed final EFE development pilot checkpoint; no capability-score selection')
        write(DEST/'checkpoints.json',registry)
    elif checkpoint.get('path')!=ready['checkpoint'] or checkpoint.get('manifest')!=ready['manifest']:
        raise ValueError('Capability checkpoint registration is already frozen to different weights')


def watch(build):
    lock=(build/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    submission=read(build/'submission.json');workflow=submission['name'];spec=read(build/'spec.json')
    dest=ROOT/spec['local_results_directory'];dest.mkdir(parents=True,exist_ok=True)
    archive=spec['archive_root']+'/'+workflow
    stage=spec.get('stage','warmstart')
    comparison=bool(spec.get('schedule_comparison'))
    deadline=time.monotonic()+36*3600
    while time.monotonic()<deadline:
        try:
            state=command(['osmo','workflow','query',workflow,'--format-type','json']);write(dest/'scheduler.json',state)
            status=read(ROOT/'status.json')
            pending_detail=('Warmstart optimizer preflight '+state['status'].lower()+'; the saved 512-example corpus is ready.'
                            if spec.get('prepared_dataset') else 'Training pilot '+state['status'].lower()+': trace admission, public backward explanations, optimizer smoke, then warmstart.')
            status['training_pilot']={'workflow':workflow,'scheduler_status':state['status'],
                 'progress':read(dest/'progress.json',{'detail':pending_detail}),
                 'directory':str(dest)}
            if not comparison:write(ROOT/'status.json',status)
            via=workflow if state['status']=='RUNNING' else None
            if via is None and state['status']!='PENDING':
                active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','--format-type','json'])['workflows']
                via=next((r['name'] for r in active if r['user']=='rpunamiya@nvidia.com'),None)
            if via is None and state['status']=='COMPLETED' and not (dest/'complete.json').exists():
                via=recovery_reader(build)
            if via:
                result=capture(via,archive)
                for name,row in result['files'].items():
                    data=base64.b64decode(row['data'])
                    if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Remote summary checksum mismatch')
                    path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
                    if name.endswith(('.jsonl','.log')):path.write_bytes(data)
                    else:write(path,json.loads(data))
                corpus=read(dest/'corpus-manifest.json')
                if corpus and not comparison:write(ROOT/'training-pilot/corpus-manifest.json',corpus)
                status=read(ROOT/'status.json');progress=read(dest/'progress.json',{})
                status['training_started']=status.get('training_started',False) or progress.get('training_started',False) or (dest/'train/complete.json').exists()
                status['training_pilot']={'workflow':workflow,'scheduler_status':state['status'],'progress':progress,'directory':str(dest)}
                if not comparison:write(ROOT/'status.json',status)
                if (dest/'complete.json').exists() and state['status']=='RUNNING':
                    remote(workflow,'from pathlib import Path;Path('+repr(archive+'/collected')+').touch();print(\'EFE_CAPTURE {"ack":true}\',flush=True)')
                elif (dest/'complete.json').exists() and via==read(build/'completion-reader/submission.json',{}).get('name'):
                    remote(via,'from pathlib import Path;Path("/tmp/efe-reader-collected").touch();print(\'EFE_CAPTURE {"ack":true}\',flush=True)')
            # A completed worker may no longer be reachable. Its collected,
            # checksummed readiness receipt still authorizes the next job.
            ready=read(dest/spec.get('ready_name','warmstart-ready.json'))
            if ready and ready.get('final') and read(dest/'complete.json',{}).get('success'):
                if spec.get('register_capability',True):register_retention(ready,stage)
                evaluation=eval_build(ready,workflow,stage)
                if not (evaluation/'submission.json').exists():
                    receipt=submit(evaluation,priority='HIGH' if spec.get('continuation') else 'LOW');native_monitor(evaluation)
                    write(dest/'evaluation-submission.json',{'workflow':receipt['name'],'build':str(evaluation)})
                if spec.get('continuation'):start_native_recovery(evaluation,stage)
                # The fixed first LeAct round does not select data, weights, or
                # hyperparameters from development outcomes. Launch it beside
                # evaluation so a grading/transport failure cannot idle training.
                if stage=='warmstart':
                    next_build=leact_build(ready,build)
                    if not (next_build/'submission.json').exists():
                        receipt=submit(next_build);launch_monitor(next_build)
                        write(dest/'leact-submission.json',{'workflow':receipt['name'],'build':str(next_build),
                              'scheduling_note':'Fixed LeAct round runs independently of development grading.'})
            importlib.reload(report).publish()
            write(build/'monitor-status.json',{'workflow':workflow,'scheduler_status':state['status'],'updated_utc':datetime.now(timezone.utc).isoformat()})
            if state['status']=='COMPLETED' or state['status'].startswith('FAILED'):
                complete=read(dest/'complete.json')
                if not complete and state['status'].startswith('FAILED'):
                    write(dest/'continuation-status.json',{'status':'needs_review','reason':'Worker ended without a completion receipt: '+state['status']})
                    return
                if complete and (not complete.get('success') or complete.get('no_admissions')):return
                if complete and complete.get('success'):
                    eval_directory=ROOT/read(ROOT/f'training-pilot/{stage}-evaluation-active.json',{}).get('directory',f'training-pilot/{stage}-evaluation')
                    evaluation=read(eval_directory/'results.json',{})
                    eval_complete=read(eval_directory/'complete.json')
                    if eval_complete:
                        rows=evaluation.get('trials',[])
                        valid=len(rows)==7 and all(r['status']=='graded' for r in rows)
                        write(dest/'development-verdict.json',{'valid':valid,'graded':len(rows),
                              'successes':sum(r.get('score',0) or 0 for r in rows),'tasks':[r['task'] for r in rows]})
                        if not valid:
                            write(dest/'continuation-status.json',{'status':'needs_review','reason':'Development evaluation has missing/infrastructure outcomes; inspect before continuing.'})
                            return
                        return
        except Exception as exc:
            write(build/'monitor-error.json',{'error':str(exc)[-1600:],'updated_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(30)


def launch_monitor(build):
    with (build/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_training','monitor','--build',str(build)],
           cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(build/'monitor-launch.json',{'pid':p.pid,'updated_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','submit','monitor']);p.add_argument('--build',type=Path);a=p.parse_args()
    if a.action=='monitor':watch(a.build.resolve())
    else:
        b=build_run()
        if a.action=='submit':
            receipt=submit(b);launch_monitor(b);print(json.dumps({'workflow':receipt['name'],'build':str(b)}))
        else:print(b)
