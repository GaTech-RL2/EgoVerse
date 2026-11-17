"""Freeze selection-worker payloads separately from the scientific run spec."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import yaml

from .scale_efe import BASE,BUILD_ROOT,read,write
from .launch_efe_training import freeze_sources


def build(source,name,execution):
    source=Path(source);folder=BUILD_ROOT/name
    if (folder/'source-manifest.json').exists():
        if read(folder/'execution.json')!=execution:raise ValueError('Frozen execution plan differs')
        return folder
    folder.mkdir(exist_ok=True)
    for filename in read(source/'source-manifest.json'):
        shutil.copyfile(source/filename,folder/filename)
    # The original spec is byte-for-byte unchanged, including seeds and corpus.
    shutil.copyfile(BASE/'cluster/efe_qwen36_leact.py',folder/'leact.py')
    shutil.copyfile(BASE/'cluster/efe_leact_selection_worker.py',folder/'selection-worker.py')
    write(folder/'execution.json',execution)
    bootstrap=(source/'bootstrap.py').read_text()
    old="os.execve(sys.executable,[sys.executable,'-u',str(lane/'pipeline.py'),'--spec',str(lane/'spec.json')],env)"
    new="env.update({'VLLM_WORKER_MULTIPROC_METHOD':'spawn','TOKENIZERS_PARALLELISM':'false',**spec.get('nccl_collective_workaround',{})})\nos.execve(sys.executable,[sys.executable,'-u',str(lane/'selection-worker.py'),'--spec',str(lane/'spec.json'),'--execution',str(lane/'execution.json')],env)"
    if old not in bootstrap:raise ValueError('Unexpected bootstrap')
    (folder/'bootstrap.py').write_text(bootstrap.replace(old,new))
    workflow=yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-efe-'+name
    task=workflow['workflow']['tasks'][0]
    task['files'] += [{'localpath':n,'path':'/tmp/efe-coverage-training/'+n} for n in ['selection-worker.py','execution.json']]
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    for p in folder.glob('*.py'):compile(p.read_text(),str(p),'exec')
    freeze_sources(folder)
    return folder


def prepare_benchmark(source,version=1,fixed_scores=False):
    spec=read(Path(source)/'spec.json')
    root=spec['archive_root']+'/performance/selection-20261116-v'+str(version)
    return build(source,'coverage-select-benchmark-20261116-v'+str(version),{
        'benchmark':True,'indices':[0,3,120],'output_root':root,'worker_root':root,
        'candidate_only':True,'max_num_seqs':1,
        **({'score_reference_root':spec['archive_root']+'/performance/selection-20261116-v1/seq1'} if fixed_scores else {})})


def prepare_coordinator(source,name='efe-coverage-leact496-parallel-v1'):
    source=Path(source);folder=BUILD_ROOT/name
    if (folder/'source-manifest.json').exists():return folder
    folder.mkdir(exist_ok=True)
    for filename in read(source/'source-manifest.json'):
        shutil.copyfile(source/filename,folder/filename)
    shutil.copyfile(BASE/'cluster/efe_qwen36_leact.py',folder/'leact.py')
    workflow=yaml.safe_load((source/'workflow.yaml').read_text())
    workflow['workflow']['name']='leact-'+name
    (folder/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
    freeze_sources(folder)
    original=read(source/'source-manifest.json');updated=read(folder/'source-manifest.json')
    changes={name for name in set(original)|set(updated) if original.get(name)!=updated.get(name)}
    if changes-{'leact.py','workflow.yaml'}:raise ValueError('Coordinator upgrade changed non-selection payloads')
    return folder


def prepare_workers(source,start=200,total=2048,count=3,max_sequences=1,version=1):
    """Reserve the first chunk for the coordinator and three later chunks."""
    source=Path(source);spec=read(source/'spec.json');root=spec['archive_root']
    width=(total-start+count)//(count+1)
    jobs=[]
    for worker in range(1,count+1):
        indices=list(range(start+worker*width,min(total,start+(worker+1)*width)))
        worker_root=root+'/selection-workers/worker-'+str(worker)
        execution={'candidate_only':True,'shared_candidates':True,'indices':indices,
            'max_num_seqs':max_sequences,'output_root':root,'worker_root':worker_root}
        folder=build(source,'coverage-select-worker-'+str(worker)+'-20261116-v'+str(version),execution)
        jobs.append({'id':'worker-'+str(worker),'build':str(folder),'worker_root':worker_root,'attempts':[]})
    return jobs


def helper_assignments(missing_indices,total,count):
    """Partition unfinished fixed states; existing state locks prevent duplicates."""
    if type(count) is not int or count<1 or type(total) is not int or total<1:
        raise ValueError('Positive worker and state counts required')
    indices=sorted(missing_indices)
    if len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<total for i in indices):
        raise ValueError('Invalid unfinished state membership')
    return [indices[offset::count] for offset in range(count) if indices[offset::count]]


def prepare_helpers(source,missing_indices,*,total=2048,count=4,version=1):
    source=Path(source);spec=read(source/'spec.json');root=spec['archive_root']
    jobs=[]
    for worker,indices in enumerate(helper_assignments(missing_indices,total,count),1):
        ident=f'helper-{version}-{worker}'
        worker_root=root+'/selection-workers/'+ident
        execution={'candidate_only':True,'shared_candidates':True,'indices':indices,
            'max_num_seqs':8,'output_root':root,'worker_root':worker_root}
        folder=build(source,f'coverage-select-{ident}-20261117',execution)
        jobs.append({'id':ident,'build':str(folder),'worker_root':worker_root,'attempts':[]})
    return jobs


def tick():
    from .scale_efe import command,submit
    from .launch_efe_coverage import DEST,infra_retryable
    from .efe_coverage_retry import retry_due,submission_priority
    from .compute_budget import reserved_nodes,node_limit
    path=DEST/'selection-workers.json';registry=read(path,{})
    if not registry.get('enabled'):return
    active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    slots=node_limit()-reserved_nodes(active)
    for job in registry.get('jobs',[]):
        if job['attempts']:
            previous=job['attempts'][-1]
            state=command(['osmo','workflow','query',previous['workflow'],'--format-type','json'])
            job['status']=state['status']
            if not state['status'].startswith('FAILED'):continue
            upgrade=(state['status']=='FAILED_CANCELED' and job.get('reviewed_execution_upgrade_from_workflow')==previous['workflow'])
            if not upgrade and not infra_retryable(state):
                job['status']='NEEDS_REVIEW';continue
            if not upgrade and not retry_due(job,previous['workflow']):continue
        if slots<=0:break
        folder=Path(job['build'])
        if job['attempts']:
            retry=folder.parent/(folder.name+'-retry-'+str(len(job['attempts'])))
            if not (retry/'source-manifest.json').exists():
                retry.mkdir(exist_ok=True)
                for filename in read(folder/'source-manifest.json'):shutil.copyfile(folder/filename,retry/filename)
                workflow=yaml.safe_load((retry/'workflow.yaml').read_text())
                workflow['workflow']['name']+='-r'+str(len(job['attempts']))
                (retry/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(retry)
            folder=retry
        priority=submission_priority(job)
        pool=read(DEST/'execution-pools.json',{}).get('selection_pool','groot-h100-01')
        receipt=submit(folder,priority=priority,pool=pool)
        job['attempts'].append({'workflow':receipt['name'],'build':str(folder),'priority':priority,
            'pool':pool,
            'submitted_utc':datetime.now(timezone.utc).isoformat()})
        job['status']='PENDING';slots-=1
        write(path,registry)
    registry['updated_utc']=datetime.now(timezone.utc).isoformat();write(path,registry)
