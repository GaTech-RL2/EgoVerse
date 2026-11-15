"""Launch/observe a discarded actual-update PEAR qualification on one H100 node."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import yaml

from .scale_efe import BASE,BUILD_ROOT,read,write,submit,command
from .launch_efe_training import freeze_sources
from .launch_efe_pear import DEST
from .recover_efe_coverage_storage import remote

VERSION=2
LOCAL=DEST/f'optimizer-update-v{VERSION}'
LANE='/tmp/efe-pear-optimizer'
ROOT=f'/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-20261114-v1/optimizer-qualification-v{VERSION}'
TEMPLATE=BUILD_ROOT/'efe-coverage-20261105-v1-warmstart-p7-allocation-retry-24'


def now():return datetime.now(timezone.utc).isoformat()


def result_from_logs(logs):
    """Read the worker's final receipt even after its allocation has ended."""
    marker='PEAR_OPTIMIZER_RESULT '
    results=[]
    for line in logs.splitlines():
        if marker not in line:continue
        try:value=json.loads(line.split(marker,1)[1])
        except json.JSONDecodeError:continue
        if isinstance(value,dict):results.append(value)
    if not results:return None
    result=results[-1]
    if (result.get('success') is True and result.get('optimizer_updates')==1
        and result.get('primary_rl_updates')==0
        and result.get('native_episodes')==8 and result.get('generated_tokens')==1418034
        and result.get('exact_adam_and_rng_restored') is True
        and result.get('fresh_process_replay_max_error')==0.0):
        return result
    return None


def prepare():
    LOCAL.mkdir(parents=True,exist_ok=True)
    build=BUILD_ROOT/f'efe-pear-optimizer-qualification-20261115-v{VERSION}'
    if (build/'source-manifest.json').exists():
        if not (LOCAL/'plan.json').exists():
            saved=read(build/'spec.json');audit=saved['input']
            write(LOCAL/'plan.json',{'build':str(build),'qualification_root':saved['qualification_root'],
                'optimizer_updates':1,'generated_tokens':audit['generated_tokens'],'requests':audit['requests'],
                'scope':saved['scope'],'retained_groups':1,'nodes':1,'gpus_per_node':8,'source_group':audit['group_root']})
        return build
    audit=read(DEST/'optimizer-input-audit.json')
    if not audit['group']['admitted'] or audit['episodes']!=8 or audit['sampler_revision']!=read(TEMPLATE/'spec.json')['base_revision']:
        raise ValueError('Actual optimizer qualification requires the audited informative base-policy group')
    native=read(BUILD_ROOT/'efe-pear-native-qualification-20261114-v1/spec.json')
    build.mkdir(parents=True,exist_ok=True)
    mapping={'efe_pear_update_worker.py':'worker.py','efe_train_pear_qualification.py':'rl_trainer.py',
        'efe_train_qwen36_lora.py':'trainer.py'}
    names=['efe_pear_update_data.py','efe_pear_sampler.py','efe_pear_groups.py','efe_pear_loss.py',
        'efe_training_data.py','efe_budget_gateway.py','efe_visual_tensor_store.py','efe_model_cache.py',
        'efe_qwen36_restart.py','efe_qwen36_adapter_io.py','efe_training_state.py','efe_qwen36_health.py']
    mapping.update({name:name for name in names})
    for source,target in mapping.items():shutil.copyfile(BASE/'cluster'/source,build/target)
    shutil.copyfile(DEST/'training-membership.json',build/'training-membership.json')
    spec={'role':'discarded_base_policy_optimizer_qualification','qualification_root':ROOT,
        'base_model':native['student_checkpoint'],'base_manifest':native['student_manifest'],
        'base_revision':native['student_revision'],'input':audit,
        'base_health_reference':'/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-coverage-20261105-v1/warmstart',
        'membership':LANE+'/training-membership.json',
        'membership_sha256':hashlib.sha256((build/'training-membership.json').read_bytes()).hexdigest(),
        'image_processor_size':native['image_processor_size'],'context':131072,'seed':20261115,
        'lr':1e-6,'betas':[.9,.999],'weight_decay':.01,'grad_clip':1.,'score_centering':True,
        'optimizer_recipe_note':'PEAR LR and token-mean score-centered REINFORCE; AdamW betas, decay, clipping and FP32 LoRA use our validated EFE trainer adaptation.',
        'validate_each_restart_probe':False,'validate_final_probe':False,
        'checkpoint_revision_prefix':f'discarded-efe-pear-base-optimizer-v{VERSION}',
        'ready_name':'qualification-adapter-ready.json','restart_keep_recent':3,
        'scope':'One discarded base-policy update on all eight door-lock episodes and every captured generated token. No primary RL update or post-training success score. Primary arms still start from the same final 737-update expanded warmstart.',
        'source_hashes':{name:hashlib.sha256((build/name).read_bytes()).hexdigest() for name in mapping.values()}}
    lineage=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    spec['durable_restart']={'lineage_sha256':lineage,'state_directory':ROOT+'/restart-states','target_step':1,'restart_interval':1}
    write(build/'spec.json',spec)
    workflow=yaml.safe_load((TEMPLATE/'workflow.yaml').read_text())
    workflow['workflow']['name']=f'leact-efe-pear-optimizer-v{VERSION}'
    workflow['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'12h'}
    task=workflow['workflow']['tasks'][0];task.pop('credentials',None)
    task['args']=['-u',LANE+'/worker.py']
    task['files']=[{'localpath':name,'path':LANE+'/'+name} for name in [*mapping.values(),'spec.json','training-membership.json']]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for path in build.glob('*.py'):compile(path.read_text(),str(path),'exec')
    freeze_sources(build)
    write(LOCAL/'plan.json',{'build':str(build),'qualification_root':ROOT,'optimizer_updates':1,
        'generated_tokens':audit['generated_tokens'],'requests':audit['requests'],'scope':spec['scope'],
        'retained_groups':1,'nodes':1,'gpus_per_node':8,'source_group':audit['group_root']})
    return build


def publish(state):
    write(LOCAL/'status.json',state)
    with (DEST/'optimizer-qualification-plan.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=read(DEST/'plan.json');plan['optimizer_update_qualification']=state;write(DEST/'plan.json',plan)
    from ..embodied_first_exam import report
    report.publish()


def tick():
    queue=read(LOCAL/'queue.json');attempt=queue['attempts'][-1];name=attempt['workflow'];build=Path(attempt['build'])
    scheduler=command(['osmo','workflow','query',name,'--format-type','json']);write(build/'scheduler.json',scheduler)
    complete=scheduler['status']=='COMPLETED' or scheduler['status'].startswith('FAILED')
    host=name if scheduler['status']=='RUNNING' else None
    if complete:
        active=command(['osmo','workflow','list','--name','leact-efe-coverage-warmstart','--status','RUNNING','--format-type','json'])['workflows']
        host=active[0]['name'] if active else None
    evidence=read(LOCAL/'evidence.json',{})
    state={'workflow':name,'scheduler_status':scheduler['status'],'phase':'optimizer_qualification_'+scheduler['status'].lower(),
        'optimizer_updates':0,'primary_rl_updates':0,'updated_utc':now(),'scope':read(LOCAL/'plan.json')['scope']}
    if host:
        script='''from pathlib import Path
import json
p=Path(ROOT)
r={n:json.loads((p/n).read_text()) for n in ['progress.json','update-progress.json','untouched-base-replay.json',
 'update-complete.json','result.json','optimizer-resume.json'] if (p/n).exists()}
a=p/'attempts'/WORKFLOW
if (a/'error.json').exists():r['error']=json.loads((a/'error.json').read_text())
for n in ['prepare','update','verify']:
 if (a/(n+'.log')).exists():r[n+'_tail']=(a/(n+'.log')).read_text(errors='replace')[-3500:]
print('EFE_CAPTURE '+json.dumps(r),flush=True)
'''.replace('ROOT',repr(ROOT)).replace('WORKFLOW',repr(name))
        try:
            evidence={'workflow':name,'captured_utc':now(),**remote(host,'preflight',script)}
            write(LOCAL/'evidence.json',evidence)
        except Exception as error:state['capture_error']=repr(error)
    if complete and not evidence.get('result.json'):
        final=subprocess.run(['osmo','workflow','logs',name,'-n','120'],
            capture_output=True,text=True,timeout=35)
        if final.returncode==0:
            receipt=result_from_logs(final.stdout)
            if receipt:
                (LOCAL/'final-scheduler.log').write_text(final.stdout)
                evidence={**evidence,'workflow':name,'captured_utc':now(),
                    'result.json':receipt,'result_source':'terminal OSMO worker log',
                    'terminal_log_sha256':hashlib.sha256(final.stdout.encode()).hexdigest()}
                write(LOCAL/'evidence.json',evidence)
    if evidence.get('workflow')==name:
        state.update({k:v for k,v in evidence.get('progress.json',{}).items() if k not in {'workflow','utc'}})
        state['accumulation']=evidence.get('update-progress.json',{})
        state['base_replay']=evidence.get('untouched-base-replay.json',{})
        if evidence.get('update-complete.json'):state['optimizer_updates']=1
        if evidence.get('result.json',{}).get('success'):
            state.update(phase='optimizer_qualification_passed',optimizer_updates=1,result=evidence['result.json'])
            write(LOCAL/'result.json',evidence['result.json']);complete=True
        if evidence.get('error'):
            state.update(phase='optimizer_qualification_needs_review',error=evidence['error']['error'])
    state['summary']=(f"Real PEAR optimizer qualification: {state['phase'].replace('_',' ')}; "
        f"{state['optimizer_updates']}/1 discarded base-policy update. All 512 requests / 1,418,034 generated tokens from the admitted door-lock group are retained. "
        'A fresh-process likelihood, Adam and RNG replay gates completion. Primary RL arms still wait for the final common warmstart.')
    if state['phase']=='optimizer_qualification_passed':
        state['summary']=('PEAR optimizer qualification passed: one discarded base-policy update over all 512 requests / '
            '1,418,034 generated tokens. A fresh process restored Adam and RNG exactly and reproduced the saved probes '
            'with zero maximum error. Primary RL updates remain zero; both comparison arms wait for the common final warmstart.')
    if scheduler['status']=='COMPLETED' and state['phase']!='optimizer_qualification_passed' and not evidence.get('error'):
        complete=False
        state['phase']='awaiting_terminal_evidence'
        state['summary']='Scheduler completed; awaiting the final optimizer/reload receipt. No qualification pass or primary RL result is claimed.'
    from .launch_efe_coverage import infra_retryable
    from .efe_coverage_retry import retry_due
    if complete and not evidence.get('error') and state['phase']!='optimizer_qualification_passed' and infra_retryable(scheduler):
        complete=False
        if len(queue['attempts'])<8 and retry_due(queue,name):
            index=len(queue['attempts']);original=Path(queue['attempts'][0]['build'])
            retry=original.parent/(original.name+f'-allocation-retry-{index}')
            if not (retry/'source-manifest.json').exists():
                retry.mkdir(exist_ok=True)
                for member in read(original/'source-manifest.json'):shutil.copyfile(original/member,retry/member)
                workflow=yaml.safe_load((retry/'workflow.yaml').read_text())
                workflow['workflow']['name']=f'leact-efe-pear-optimizer-v{VERSION}-r{index}'
                (retry/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(retry)
            receipt=submit(retry,priority='LOW')
            queue['attempts'].append({'workflow':receipt['name'],'build':str(retry),'priority':'LOW','submitted_utc':now()})
            write(LOCAL/'queue.json',queue)
            state['summary']+=' Exact pinned inputs queued on LOW priority after a verified infrastructure interruption.'
    publish(state);return complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--monitor',action='store_true');p.add_argument('--tick',action='store_true');a=p.parse_args()
    if a.monitor:
        LOCAL.mkdir(parents=True,exist_ok=True)
        lock=(LOCAL/'observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if tick():return
            except Exception as error:print(now()+' '+repr(error),flush=True)
            time.sleep(60)
    elif a.tick:tick()
    else:
        build=prepare()
        if a.submit:
            receipt=submit(build,priority='HIGH')
            if not (LOCAL/'queue.json').exists():
                write(LOCAL/'queue.json',{'attempts':[{'workflow':receipt['name'],'build':str(build),'priority':'HIGH','submitted_utc':now()}]})
            publish({'workflow':receipt['name'],'scheduler_status':'SUBMITTED','phase':'optimizer_qualification_submitted',
                'optimizer_updates':0,'primary_rl_updates':0,'updated_utc':now(),
                'summary':'One H100 node submitted for one discarded real PEAR update and exact restart qualification. No primary RL update or trained task score yet.'})
            print(json.dumps({'workflow':receipt['name'],'build':str(build)}),flush=True)
        else:print(build)


if __name__=='__main__':main()
