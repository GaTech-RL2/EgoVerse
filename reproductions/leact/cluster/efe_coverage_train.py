"""Data-scaled Qwen3.6 warmstart and LeAct with durable stage boundaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from efe_training_data import encode_example,sha,write
import efe_training_pilot as pilot


def now():
    return datetime.now(timezone.utc).isoformat()


def backward(state):
    messages=copy.deepcopy(state['messages'])
    messages[0]['content']+='\nFor the following recorded-action explanation request, explain the supplied command without executing it, then finish with DONE.'
    messages.append({'role':'user','content':'The expert chose this exact action for the preceding observation. Explain the visible evidence, intended effect and relevant constraints. Do not invent future observations. Finish with DONE.\nExpert action:\n'+json.dumps(state['action'])})
    return messages


def prefetch_warm_rationales(spec,root,admission):
    """Cache explanations for audited partial data without authorizing updates."""
    if not admission.get('native_execution_verified') or not admission.get('training_partition_only'):
        raise ValueError('Explanation prefetch requires audited training decisions')
    source=Path(spec['coverage_root'])/'corpus/decisions.jsonl.gz.tmp'
    (root/'rationales').mkdir(exist_ok=True)
    count=accepted=0;rejected=[]
    receipt={'corpus_sha256':sha(source),'verified_episodes':admission['successful_episodes'],
        'verified_decisions':admission['decisions'],'started_utc':now(),
        'training_started':False,'optimizer_updates':0,
        'scope':'Cache public explanations on verified partial training data; the full corpus gate still controls optimizer training.'}
    pilot.progress('warmstart_explanations_prefetch',
        f"Preparing explanations for {admission['decisions']} verified decisions while collection continues.",
        completed_states=0,total_states=admission['decisions'],training_started=False)
    with gzip.open(source,'rt') as stream,ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            states=[]
            for _ in range(16):
                line=stream.readline()
                if not line:break
                states.append(json.loads(line))
            if not states:break
            explanations=list(pool.map(lambda state:pilot.rationale(state,spec),states))
            for state,explanation in zip(states,explanations):
                count+=1
                if explanation['accepted']:accepted+=1
                else:rejected.append({'state_id':state['id'],'error':explanation.get('error')})
            write(root/'explanation-prefetch.json',{**receipt,'completed_states':count,
                'accepted':accepted,'rejected':rejected,'complete':False,'updated_utc':now()})
            pilot.progress('warmstart_explanations_prefetch',
                f'{count}/{admission["decisions"]} explanations checked; {accepted} accepted. No optimizer updates.',
                completed_states=count,total_states=admission['decisions'],accepted_states=accepted,
                rejected_states=len(rejected),training_started=False)
    if count!=admission['decisions']:raise ValueError('Partial corpus decision count changed')
    write(root/'explanation-prefetch.json',{**receipt,'completed_states':count,'accepted':accepted,
        'rejected':rejected,'complete':True,'success':not rejected,'completed_utc':now()})
    if rejected:raise ValueError(f'{len(rejected)} prefetched explanations require review; acceptance rules are unchanged')


def prepare_warm_data(spec,root):
    from transformers import AutoProcessor
    from efe_visual_tensor_store import save
    source=Path(spec['coverage_root'])/'corpus'
    admission=json.loads((source/'admission.json').read_text())
    if not admission['gate_passed'] or sha(source/'decisions.jsonl.gz')!=admission['decisions_sha256']:
        raise ValueError('Corpus is not admitted and frozen')
    for name in ['decisions.jsonl.gz','admission.json']:
        if not (root/name).exists():os.link(source/name,root/name)
    if (root/'dataset-manifest.json').exists():
        return json.loads((root/'dataset-manifest.json').read_text())
    if importlib.metadata.version('transformers')!='4.57.6':
        raise ValueError('Prepare tensors using the pinned serving processor before HF5 training')
    processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
    processor.image_processor.size.update(shortest_edge=65536,longest_edge=1048576)
    write(root/'processor.json',{'size':dict(processor.image_processor.size),'transformers':'4.57.6',
        'chat_template_sha256':sha(Path(spec['base_model'])/'chat_template.jinja'),
        'prefix_validation':'Exact processor-token prefix equality; no target truncation'})
    for name in ['rationales','tensors','example-receipts']:(root/name).mkdir(exist_ok=True)
    rows=[]
    def chunks(stream):
        pending=[]
        for line in stream:
            pending.append(json.loads(line))
            if len(pending)==16:
                yield pending
                pending=[]
        if pending:yield pending
    with gzip.open(source/'decisions.jsonl.gz','rt') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for states in chunks(stream):
            rationales=list(pool.map(lambda state:pilot.rationale(state,spec),states))
            for state,rationale in zip(states,rationales):
                if not rationale['accepted']:raise ValueError('Astra explanation failed: '+state['id']+' '+str(rationale.get('error')))
                receipt=root/'example-receipts'/(state['id']+'.json')
                if receipt.exists():
                    saved=json.loads(receipt.read_text())
                    if saved['rationale_sha256']!=sha(root/'rationales'/(state['id']+'.json')):
                        raise ValueError('Cached explanation changed')
                    for row in saved['rows']:
                        if sha(root/row['tensor_file'])!=row['tensor_sha256']:raise ValueError('Cached tensor changed')
                    rows.extend(saved['rows'])
                    continue
                pair=[]
                for direction,messages,answer in [('forward',state['messages'],json.dumps(state['action'],ensure_ascii=False)),('backward',backward(state),'DONE')]:
                    target={'role':'assistant','reasoning_content':rationale['rationale'],'content':answer}
                    batch,meta=encode_example(processor,messages,[],target,spec['context'])
                    if batch is None:raise ValueError('Complete example exceeds context: '+state['id'])
                    name='tensors/'+state['id']+'-'+direction+'.pt'
                    save(batch,root/name)
                    pair.append({'id':state['id']+'-'+direction,'state_id':state['id'],'decision_id':state['id'],
                        'suite':state['suite'],'task':state['task'],'episode':state['episode'],'source':'warmstart',
                        'direction':direction,'tensor_file':name,'tensor_sha256':sha(root/name),**meta})
                write(receipt,{'rows':pair,'rationale_sha256':sha(root/'rationales'/(state['id']+'.json'))})
                rows.extend(pair)
            pilot.progress('backward_explanations_and_tokenization',f'{len(rows)//2}/{admission["decisions"]} states prepared.',
                           completed_states=len(rows)//2,total_states=admission['decisions'],training_started=False)
    if len(rows)!=2*admission['decisions']:raise ValueError('Prepared pair count changed')
    data={'examples':rows,'states':admission['decisions'],'pairs':admission['decisions'],'examples_count':len(rows),
        'source_sha256':admission['decisions_sha256'],'teacher_episodes':admission['successful_episodes'],
        'unique_tasks':admission['unique_tasks'],'unique_scenes':admission['unique_scenes'],
        'target_truncation':False,'tensor_storage':'shared-visual-tensors-v1',
        'rationale':'Brief public Astra action explanations; no provider-private reasoning.'}
    write(root/'dataset-manifest.json',data)
    return data


def training_runtime(spec,root):
    from trainer import training_plan
    import efe_qwen36_restart as restart
    data=json.loads((root/'dataset-manifest.json').read_text())
    steps,_,_,digest=training_plan(data['examples'],spec)
    identity={'dataset_sha256':sha(root/'dataset-manifest.json'),'base_model':spec['base_model'],
        'base_revision':spec['base_revision'],'schedule_sha256':digest,'steps':steps,
        'lr':spec['lr'],'rank':spec['lora_rank'],'alpha':spec['lora_alpha'],'world_size':8}
    lineage=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=root/'training-lineage.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Training lineage changed')
    write(path,identity)
    (root/'restart-states').mkdir(exist_ok=True)
    runtime={**spec,'expected_optimizer_steps':steps,'adapter_restart_only':True,
        'durable_restart':{'state_directory':str(root/'restart-states'),'lineage_sha256':lineage,
            'target_step':steps,'restart_interval':4,'origin_step':0,'initial_checkpoint':spec['base_model']}}
    runtime['active_restore']=restart.choose_resume(runtime)
    metrics=root/'train/metrics.jsonl'
    start=runtime['active_restore']['step']
    attempt=root/'attempts'/os.environ['EFE_WORKFLOW_ID']
    attempt.mkdir(parents=True,exist_ok=True)
    if metrics.exists() and not (root/'train/complete.json').exists():
        shutil.copyfile(metrics,attempt/'metrics-before-resume.jsonl')
        records=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        metrics.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=start))
    if start==steps and not (root/'train/complete.json').exists():
        receipt=runtime['active_restore']['receipt']
        if receipt.get('post_update_full_target_ce') is None or not receipt.get('portable',{}).get('final'):
            raise ValueError('Final restart lacks the fixed loss probe or final adapter receipt')
        write(root/'adapter-ready.json',receipt['portable'])
        write(root/'train/complete.json',{'success':True,'optimizer_steps':steps,
            'post_update_full_target_ce':receipt['post_update_full_target_ce'],'recovered_from_final_restart':True})
    write(root/'spec.json',runtime)
    return runtime


def run(spec,lane):
    root=Path(spec['archive_root']);root.mkdir(parents=True,exist_ok=True);pilot.ROOT=root
    lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','NCCL_P2P_DISABLE':'1',
         'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
         'VLLM_WORKER_MULTIPROC_METHOD':'spawn'}
    # The default transport corrupted frozen weights on the reviewed P6 host.
    # Ring/Simple passed the same full-model replay to 5.96e-8 absolute CE.
    # Keep the temporary workaround explicit in the frozen run specification.
    transport=spec.get('nccl_collective_workaround',{})
    if transport and transport!={'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple'}:
        raise ValueError('Unvalidated distributed communication workaround')
    env.update(transport)
    write(root/'worker-start.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'started_utc':now()})
    def logged(name,argv,timeout=86400):
        pilot.run_logged(name,argv,env,timeout)
    try:
        if spec['stage']=='warmstart':
            from efe_coverage_admission import prepare
            plan=json.loads((lane/'coverage-plan.json').read_text())
            pilot.progress('corpus_admission','Auditing all selected native successes and executed commands.',training_started=False)
            admitted=prepare(plan,Path(spec['coverage_root']))
            if not admitted['gate_passed'] and spec.get('prefetch_explanations_before_corpus_ready'):
                prefetch_warm_rationales(spec,root,admitted)
                # Collection may have crossed the original gate while public
                # explanations were being cached. Reaudit before proceeding.
                admitted=prepare(plan,Path(spec['coverage_root']))
            if not admitted['gate_passed']:
                write(root/'waiting-for-corpus.json',admitted)
                pilot.progress('waiting_for_corpus','Verified explanations cached; optimizer training still waits for the full corpus gate.',
                    verified_episodes=admitted['successful_episodes'],training_started=False)
                return
            (root/'waiting-for-corpus.json').unlink(missing_ok=True)
            from huggingface_hub import snapshot_download
            pilot.progress('base_model','Preparing the immutable base Qwen3.6 checkpoint.',training_started=False)
            snapshot_download(spec['model'],revision=spec['base_revision'],local_dir=spec['base_model'],max_workers=4,
                allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE*'])
            base_ready=Path(spec['coverage_root'])/'base-ready.json'
            if not base_ready.exists():
                files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in Path(spec['base_model']).iterdir() if p.is_file()}
                manifest=Path(spec['coverage_root'])/'base-checkpoint-manifest.json'
                write(manifest,{'files':files,'revision':spec['base_revision'],'model':spec['model'],'untuned_base':True})
                write(base_ready,{'checkpoint':spec['base_model'],'manifest':str(manifest),'revision':spec['base_revision']})
            prepare_warm_data(spec,root)
        else:
            manifest=json.loads(Path(spec['parent_manifest']).read_text())
            if manifest['revision']!=spec['base_revision']:raise ValueError('Warmstart parent revision changed')
            for name,row in manifest['files'].items():
                if sha(Path(spec['base_model'])/name)!=row['sha256']:raise ValueError('Warmstart parent file changed')
            if not (root/'selection-complete.json').exists():
                pilot.progress('leact_selection','Generating and scoring student explanations on balanced teacher states.',training_started=False)
                logged('selection',[sys.executable,str(lane/'leact.py'),'--spec',str(lane/'spec.json'),'--mode','select'],48*3600)
            selected=json.loads((root/'selection-complete.json').read_text())
            if selected['no_admissions']:
                write(root/'complete.json',{'success':True,'no_admissions':True,'optimizer_steps':0,'completed_utc':now()})
                return
        if spec.get('node_local_model_cache'):
            from efe_model_cache import stage as stage_model
            manifest=(Path(spec['coverage_root'])/'base-checkpoint-manifest.json'
                      if spec['stage']=='warmstart' else Path(spec['parent_manifest']))
            pilot.progress('model_cache','Copying and verifying the pinned base on node-local disk before distributed model loading.',training_started=False)
            def cache_progress(value):
                write(root/'model-cache-progress.json',{**value,'updated_utc':now()})
            cache=stage_model(spec['base_model'],manifest,lane/'model-cache',spec['base_revision'],cache_progress)
            write(root/'model-cache.json',cache)
            # Keep the durable base path in the lineage and checkpoints. This
            # separate path changes only where identical bytes are loaded.
            spec={**spec,'local_base_model':cache['local_model']}
        training=lane/'training-env'
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(training)],check=True)
        python=str(training/'bin/python')
        logged('training-environment',[python,'-m','pip','install','--disable-pip-version-check',
            'transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],900)
        env.update(VIRTUAL_ENV=str(training),PATH=str(training/'bin')+':'+env['PATH'])
        # Lossless visual storage and processor parity are verified before any update.
        if not (root/'modality-parity.json').exists():
            logged('modality-parity',[python,str(lane/'leact.py'),'--spec',str(lane/'spec.json'),'--mode','parity'],7200)
        probe=Path(spec['coverage_root'])/'restart-probe/result.json'
        if not probe.exists() or not json.loads(probe.read_text()).get('success'):
            raise ValueError('Exact optimizer-resume probe must pass before large-scale training')
        if spec.get('require_empty_shard_restart_probe'):
            probe_root=root/'restart-empty-shard-probe'/os.environ['EFE_WORKFLOW_ID']
            pilot.progress('restart_validation','Checking exact Adam/RNG restart with empty optimizer shards.',training_started=False)
            logged('restart-empty-shard-probe',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(lane/'probe.py'),'--root',str(probe_root),'--empty-shards'],900)
            result=json.loads((probe_root/'result.json').read_text())
            if not result.get('success') or not result.get('empty_shards_tested'):
                raise ValueError('Empty-shard optimizer-resume probe did not pass')
        runtime=training_runtime(spec,root)
        command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(lane/'trainer.py'),
                 '--spec',str(root/'spec.json'),'--root',str(root)]
        def replay(mode):
            ready=json.loads((root/('smoke-adapter-ready.json' if mode=='smoke' else 'adapter-ready.json')).read_text())
            completed=json.loads((root/mode/'complete.json').read_text())
            path=root/(mode+'-verification-spec.json')
            write(path,{**runtime,'checkpoint':ready['checkpoint'],'training_root':str(root),'verification_mode':mode,
                        'expected_post_update_ce':completed['post_update_full_target_ce']})
            pilot.progress(mode+'_reload','Reloading the saved adapter and checking its loss against the recorded value.',
                           training_started=mode=='train',optimizer_steps_completed=runtime['expected_optimizer_steps'] if mode=='train' else 0)
            logged(mode+'-reload',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                str(lane/'verify.py'),'--spec',str(path),'--mode','verify'],3600)
        if not (root/'smoke-loss-replay.json').exists():
            if not (root/'smoke/complete.json').exists():
                # A reclaimed incomplete smoke has no production updates to preserve.
                if (root/'smoke').exists():
                    shutil.move(str(root/'smoke'),str(root/'attempts'/os.environ['EFE_WORKFLOW_ID']/'incomplete-smoke'))
                smoke={k:v for k,v in runtime.items() if k not in {'durable_restart','active_restore','adapter_restart_only'}}
                write(root/'smoke-spec.json',smoke)
                smoke_command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(lane/'trainer.py'),
                    '--spec',str(root/'smoke-spec.json'),'--root',str(root),'--mode','smoke']
                pilot.progress('optimizer_smoke','Testing two discarded updates on p95 and longest complete examples.',training_started=False)
                logged('smoke',smoke_command,7200)
            replay('smoke')
        if not (root/'train/complete.json').exists():
            pilot.progress('training',f"{spec['stage']}: resume {runtime['active_restore']['step']} → {runtime['expected_optimizer_steps']} updates.",training_started=True)
            logged('train',command+['--mode','train'],60*3600)
        if not (root/'train-loss-replay.json').exists():replay('train')
        if not (root/spec['export_ready_name']).exists():
            if (root/'model').exists():
                shutil.move(str(root/'model'),str(root/'attempts'/os.environ['EFE_WORKFLOW_ID']/'incomplete-export'))
            logged('merge',[python,str(lane/'merge.py'),'--spec',str(root/'spec.json')],7200)
        ready=json.loads((root/spec['export_ready_name']).read_text())
        write(root/'complete.json',{'success':True,'optimizer_steps':runtime['expected_optimizer_steps'],
            'checkpoint':ready,'completed_utc':now(),'dataset_sha256':sha(root/'dataset-manifest.json')})
    except Exception as error:
        write(root/'worker-error.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'error':str(error),'utc':now()})
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);args=p.parse_args()
    run(json.loads(args.spec.read_text()),Path(__file__).parent)
