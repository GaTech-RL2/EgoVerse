"""Launch only the newly requested warmstart-496 retention check, not PEAR."""
import argparse
import ast
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.request import urlopen

import yaml

from .scale_efe import BASE,BUILD_ROOT,read,write,submit
from .launch_efe_training import freeze_sources

ROOT=BASE/'artifacts/reasoning-workspace/retention-496'
REMOTE='/mnt/amlfs-07/shared/rpunamiya/leact/studies/rp_jhh_warm496_realworldqa_v2'
IMAGE='docker.io/vllm/vllm-openai@sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90'


def prepare(attempt=1,pool='groot-l40s-01'):
    ROOT.mkdir(parents=True,exist_ok=True)
    suffix=''if attempt==1 else'-r'+str(attempt)
    folder=BUILD_ROOT/('rp-jhh-warm496-realworldqa-v2'+suffix);folder.mkdir(exist_ok=True)
    if (folder/'source-manifest.json').exists():return folder
    ready=read(BASE/'artifacts/embodied-first-exam/fixed-action-v1/coverage-scale-v1/intermediate-496/status.json')['result']['ready']
    if ready['step']!=496 or ready['additional_optimizer_updates']!=0 or not ready['loss_replay']['passed']:
        raise ValueError('The requested fixed warmstart checkpoint is not qualified')
    url='https://raw.githubusercontent.com/modelscope/evalscope/v1.13.0/evalscope/benchmarks/real_world_qa/real_world_qa_adapter.py'
    with urlopen(url,timeout=30)as response:adapter=response.read()
    tree=ast.parse(adapter);prompt=next(ast.literal_eval(n.value)for n in tree.body
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='OPEN_PROMPT'for t in n.targets))
    with urlopen('https://huggingface.co/api/datasets/lmms-lab/RealWorldQA',timeout=30)as response:dataset=json.load(response)
    parent='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1'
    export=read(BASE/'artifacts/embodied-first-exam/fixed-action-v1/coverage-scale-v1/intermediate-496/plan.json')
    export_spec=read(Path(export['build'])/'spec.json')
    base_manifest=export_spec['base_manifest']
    base_path=str(Path(base_manifest).parent/'base-model')
    gpu_count=4 if pool=='groot-l40s-01'else 8
    spec={'id':'rp_jhh_warm496_realworldqa_v2','study_version':2,'purpose':'Explicit user-requested pre-PEAR retention check; does not enable the older capability screen or PEAR training.',
        'shared_root':REMOTE,'gpu_count':gpu_count,'batch_size':1,'seed':20261118,'max_output_tokens':32768,
        'serving_profile':{'max_num_batched_tokens':2048,'gpu_memory_utilization':.75,
            'enable_prefix_caching':False,'async_scheduling':False,
            'reason':'Serial synchronous serving after v1 batch-8 async CUDA misaligned-address failure. Benchmark and sampling unchanged.'},
        'runtime_image':IMAGE,
        'runner_sha256':hashlib.sha256((BASE/'cluster/efe_realworldqa_retention.py').read_bytes()).hexdigest(),
        'dataset':{'id':'lmms-lab/RealWorldQA','revision':dataset['sha'],'split':'test','expected_questions':765},
        'prompt':prompt,'prompt_source':url,'prompt_source_sha256':hashlib.sha256(adapter).hexdigest(),
        'protocol_reference':'https://arxiv.org/html/2611.18562#A6',
        'protocol_note':'EvalScope 1.13.0 prompt, deterministic answer extraction and paper sampling; custom offline vLLM runner. The paper does not specify this local 32768 output / 65536 context ceiling.',
        'sampling':{'temperature':1.,'top_p':.95,'top_k':20,'n':1,'dtype':'bfloat16'},
        'image_processor_size':{'longest_edge':1048576,'shortest_edge':65536},
        'models':[{'id':'base','path':base_path,'manifest':base_manifest,
            'revision':'995ad96eacd98c81ed38be0c5b274b04031597b0'},
            {'id':'warmstart496','path':ready['checkpoint'],'manifest':ready['manifest'],'revision':ready['revision'],
             'checkpoint_pin':parent+'/intermediate/warmstart-step496/checkpoint-pin.json',
             'checkpoint_pin_sha256':ready['pin_sha256']}],
        'optimizer_updates':0,'primary_pear_launch_authorized':False}
    write(folder/'spec.json',spec)
    for source,target in [('efe_realworldqa_retention.py','runner.py'),('efe_model_cache.py','efe_model_cache.py')]:
        shutil.copyfile(BASE/'cluster'/source,folder/target)
    workflow={'workflow':{'name':'leact-rp-jhh-warm496-realworldqa-v2'+suffix,
        'resources':{'default':{'gpu':gpu_count,'cpu':48,'memory':'512Gi','storage':'256Gi',
            'platform':'ovx-l40s'if gpu_count==4 else'dgx-h100'}},
        'timeout':{'queue_timeout':'30m','exec_timeout':'8h'},'tasks':[{'name':'preflight','image':IMAGE,
            'privileged':False,'command':['python3'],'args':['-u','/tmp/rp-retention/runner.py','--spec','/tmp/rp-retention/spec.json'],
            'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','NVIDIA_DRIVER_CAPABILITIES':'all'},
            'files':[{'localpath':name,'path':'/tmp/rp-retention/'+name}for name in ['runner.py','efe_model_cache.py','spec.json']]}]}}
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(folder)
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    parser.add_argument('--monitor',action='store_true')
    parser.add_argument('--attempt',type=int,default=1)
    parser.add_argument('--priority',choices=['HIGH','NORMAL','LOW'],default='LOW')
    parser.add_argument('--pool',choices=['groot-l40s-01','groot-h100-01','groot-h100-02'],default='groot-l40s-01');args=parser.parse_args()
    if args.monitor:return monitor()
    if not 1<=args.attempt<=32:raise ValueError('Invalid bounded evaluation attempt')
    if args.attempt>1:
        from .scale_efe import command
        previous=read(ROOT/'launch.json',{})
        if previous.get('name'):
            state=command(['osmo','workflow','query',previous['name'],'--format-type','json'])
            if state['status']in {'RUNNING','PENDING','WAITING'}:raise RuntimeError('Previous evaluation still owns the allocation')
    build=prepare(args.attempt,args.pool);record={'build':str(build),'attempt':args.attempt,'pool':args.pool,'priority':args.priority,
        'scope':'retention evaluation only','pear_launched':False}
    if args.submit:record.update(submit(build,priority=args.priority,pool=args.pool,nodes=1))
    write(ROOT/'launch.json',record);print(json.dumps(record))


def monitor():
    """Retry only scheduler interruptions; never retry scoring/code failures."""
    from .scale_efe import command
    from .launch_efe_coverage import infra_retryable
    with (ROOT/'monitor.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+86400
        while time.monotonic()<deadline:
            try:
                launch=read(ROOT/'launch.json');state=command(['osmo','workflow','query',launch['name'],'--format-type','json'])
                progress={'workflow':launch['name'],'scheduler_status':state['status'],
                          'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_pear_updates':0}
                if state['status']=='COMPLETED':
                    write(ROOT/'monitor.json',{**progress,'phase':'evaluation_completed'});return
                if state['status'].startswith('FAILED'):
                    if not infra_retryable(state):
                        write(ROOT/'monitor.json',{**progress,'phase':'needs_source_review'});return
                    attempt=launch.get('attempt')
                    if attempt is None:
                        match=re.search(r'-r(\d+)$',Path(launch['build']).name)
                        attempt=int(match.group(1))if match else 1
                    attempt+=1
                    if attempt>32:raise RuntimeError('Reached the bounded retention retry limit')
                    pool=launch.get('pool','groot-h100-02')
                    # Infrastructure retries use the previous immutable runner,
                    # prompt and specification, even if the checkout changed.
                    source=Path(launch['build']);version=read(source/'spec.json').get('study_version',1)
                    build=BUILD_ROOT/('rp-jhh-warm496-realworldqa-v'+str(version)+'-r'+str(attempt))
                    if not (build/'source-manifest.json').exists():
                        build.mkdir(exist_ok=True)
                        for name,digest in read(source/'source-manifest.json').items():
                            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen retry source changed')
                            shutil.copyfile(source/name,build/name)
                        workflow=yaml.safe_load((build/'workflow.yaml').read_text())
                        workflow['workflow']['name']=re.sub(r'-r\d+$','',workflow['workflow']['name'])+'-r'+str(attempt)
                        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
                    priority='LOW'if state['status']=='FAILED_QUEUE_TIMEOUT'else launch.get('priority','LOW')
                    receipt=submit(build,priority=priority,pool=pool,nodes=1)
                    write(ROOT/'launch.json',{'build':str(build),'attempt':attempt,'pool':pool,'priority':priority,
                        'scope':launch.get('scope','retention evaluation only'),'pear_launched':False,
                        **{key:launch[key]for key in ['questions_per_model','preliminary']if key in launch},**receipt})
                    progress.update(workflow=receipt['name'],scheduler_status='PENDING',phase='infrastructure_retry_submitted')
                write(ROOT/'monitor.json',progress)
            except Exception as exc:
                write(ROOT/'monitor.json',{'error':str(exc)[-500:],'phase':'retry_monitor',
                                          'updated_utc':datetime.now(timezone.utc).isoformat()})
            time.sleep(45)


if __name__=='__main__':main()
