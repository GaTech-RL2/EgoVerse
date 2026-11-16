"""Constrained-action LeAct on the verified Qwen3.6 warmstart."""
import argparse
import copy
import errno
import fcntl
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from .efe_training_data import balanced_order,encode_example,sha,write
except ImportError:
    from efe_training_data import balanced_order,encode_example,sha,write


def now():return datetime.now(timezone.utc).isoformat()


def progress(root,phase,detail,**extra):
    write(root/'progress.json',{'phase':phase,'detail':detail,'updated_utc':now(),
        'workflow':os.environ.get('EFE_WORKFLOW_ID'),**extra})


def backward(state):
    messages=copy.deepcopy(state['messages'])
    messages[0]['content']+='\nFor the following recorded-action explanation request, explain the supplied command without executing it, then finish with DONE.'
    messages.append({'role':'user','content':'The expert chose this exact action for the preceding observation. Explain the visible evidence, intended effect and relevant constraints. Do not invent future observations. Finish with DONE.\nExpert action:\n'+json.dumps(state['action'])})
    return messages


def candidate_reasoning(text,finish):
    if finish!='stop':return None,'generation_did_not_finish'
    if text.count('</think>')!=1:return None,'missing_or_ambiguous_reasoning_boundary'
    rationale,answer=text.split('</think>');rationale=rationale.removeprefix('<think>').strip()
    if not rationale:return None,'empty_explanation'
    if any(k in rationale for k in ['```','<tool_call>','<|im_','<think>']):return None,'non_explanation_output'
    if answer.strip()!='DONE':return None,'backward_request_did_not_end_with_DONE'
    return rationale,None


def physical_positions(tokenizer,ids,start):
    """Score every executable command field, excluding the free-text summary."""
    tail=tokenizer.decode(ids[start:],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    boundary=tail.index(', "summary":')
    encoded=tokenizer(tail,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ids[start:]:raise ValueError('Action-tail re-encoding changed token IDs')
    positions=[start+i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<boundary and b<=boundary and b>a]
    if not positions or max(encoded['offset_mapping'][i-start][1] for i in positions)!=boundary:
        raise ValueError('Executable command boundary is not token aligned')
    return positions


def link_immutable(source,destination,expected_sha256=None):
    """Share frozen replay files across mounts without copying their visual payloads."""
    source=Path(source).resolve(strict=True);destination=Path(destination)
    if expected_sha256 and sha(source)!=expected_sha256:raise ValueError('Replay source checksum changed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if destination.resolve()!=source and sha(destination)!=sha(source):raise ValueError('Replay destination differs')
        return
    try:os.link(source,destination)
    except OSError as error:
        if error.errno!=errno.EXDEV:raise
        destination.symlink_to(source)


@contextmanager
def candidate_guard(root,state_id):
    """One writer per immutable state, including across selection nodes."""
    if len(state_id)!=64 or any(c not in '0123456789abcdef' for c in state_id):
        raise ValueError('Invalid candidate state identifier')
    folder=Path(root)/'candidate-locks';folder.mkdir(parents=True,exist_ok=True)
    with (folder/(state_id+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def selection_indices(count,execution):
    indices=execution.get('indices',list(range(count)))
    if len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<count for i in indices):
        raise ValueError('Invalid selection worker membership')
    return set(indices)


def select(spec,root,execution=None):
    import torch
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from efe_leact_vl import prompt,images_of
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    config=root/'selection-execution.json'
    execution=execution if execution is not None else json.loads(config.read_text()) if config.exists() else {}
    producer=execution.get('candidate_only',False)
    max_sequences=execution.get('max_num_seqs',1)
    if max_sequences not in (1,8):raise ValueError('Unreviewed selection concurrency')
    worker_root=Path(execution.get('worker_root',str(root)))
    worker_root.mkdir(parents=True,exist_ok=True)
    timings=defaultdict(float)
    source=Path(spec['parent_corpus'])/'decisions.jsonl.gz'
    if sha(source)!=spec['decisions_sha256']:raise ValueError('Teacher state membership changed')
    states=balanced_order([json.loads(s) for s in gzip.decompress(source.read_bytes()).splitlines()],spec['seed'])
    if len(states)!=spec.get('expected_teacher_states',43):raise ValueError('The frozen teacher-state count changed')
    states=states[:spec.get('max_balanced_states',len(states))]
    indices=selection_indices(len(states),execution)
    if not producer and indices!=set(range(len(states))):raise ValueError('Only producers may select a partial membership')
    spec_digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    protocol=root/'selection-lock-protocol.json'
    if producer and execution.get('shared_candidates'):
        deadline=time.monotonic()+900
        while not protocol.exists():
            if time.monotonic()>deadline:raise TimeoutError('Coordinated selector has not started')
            time.sleep(5)
        receipt=json.loads(protocol.read_text())
        if receipt.get('version')!=1 or receipt.get('spec_sha256')!=spec_digest:
            raise ValueError('Shared candidate coordinator protocol differs')
    elif not producer:
        write(protocol,{'version':1,'spec_sha256':spec_digest,'workflow':os.environ.get('EFE_WORKFLOW_ID'),'utc':now()})
    if spec.get('reuse_parent_visual_blobs') and not producer:
        parent=Path(spec['parent_corpus'])
        if sha(parent/'dataset-manifest.json')!=spec['parent_dataset_sha256']:raise ValueError('Warmstart replay corpus changed')
        for blob in (parent/'tensor-blobs').glob('*.pt'):
            link_immutable(blob,root/'tensor-blobs'/blob.name)
    for state in states:
        if tuple(state['action'])!=('action_type','action','repeat','summary') or state['tools']:
            raise ValueError('Unexpected constrained command interface')
    processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
    processor.image_processor.size.update(shortest_edge=65536,longest_edge=1048576)
    llm=LLM(model=spec['base_model'],tokenizer=spec['base_model'],dtype='bfloat16',tensor_parallel_size=8,
        distributed_executor_backend='mp',disable_custom_all_reduce=True,enforce_eager=True,
        max_model_len=spec['context'],max_num_seqs=max_sequences,max_num_batched_tokens=2048,gpu_memory_utilization=.75,
        enable_prefix_caching=False,limit_mm_per_prompt={'image':40,'video':0},
        mm_processor_kwargs={'size':dict(processor.image_processor.size)},seed=spec['seed'])
    for name in ['candidates','probes','tensors']:(root/name).mkdir(exist_ok=True)
    write((worker_root if producer else root)/'membership.json',{'states':[s['id'] for i,s in enumerate(states) if i in indices],'source':'Balanced subset of verified warmstart teacher states; no evaluation trajectory enters training.',
        'candidates_per_state':8,'top_positive':2,'candidate_tokens':spec['candidate_tokens'],
        'admission_target':'Every token through the complete action_type, action vector, and repeat fields. Free-text summary excluded; full JSON likelihood also retained as a diagnostic.'})
    def score(state,rationale,seed):
        started=time.monotonic()
        target={'role':'assistant','reasoning_content':rationale,'content':json.dumps(state['action'],ensure_ascii=False)}
        batch,meta=encode_example(processor,state['messages'],[],target,spec['context'],action_only=True)
        if batch is None:return None,None
        ids=batch['input_ids'][0].tolist();start=meta['target_start_token'];positions=physical_positions(processor.tokenizer,ids,start)
        request=prompt(processor,state['messages'],[],target=target)
        timings['score_preprocessing_seconds']+=time.monotonic()-started;started=time.monotonic()
        response=llm.generate([request],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=seed),use_tqdm=False)[0]
        timings['score_engine_seconds']+=time.monotonic()-started;timings['score_calls']+=1
        if list(response.prompt_token_ids)!=ids:raise ValueError('Scoring HF/vLLM image-token parity failed')
        values={}
        for i in range(start,len(ids)):
            row=response.prompt_logprobs[i]
            if row is None or ids[i] not in row:raise ValueError('Missing action-token likelihood')
            values[i]=float(row[ids[i]].logprob)
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite action score')
        labels=torch.full_like(batch['input_ids'],-100);labels[0,positions]=batch['input_ids'][0,positions];batch['labels']=labels
        selected=[ids[i] for i in positions]
        result={'mean_target_logp':sum(values[i] for i in positions)/len(positions),'target_tokens':len(positions),
            'target_ids_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
            'full_json_mean_logp':sum(values.values())/len(values),'full_json_tokens':len(values),
            'complete_prompt_tokens':len(ids)}
        return result,batch
    rows=[];reports=[];probes=[]
    for index,state in enumerate(states):
        if index not in indices:continue
        path=root/'candidates'/(state['id']+'.json')
        condition=hashlib.sha256(json.dumps({'spec':spec,'state':state},sort_keys=True).encode()).hexdigest()
        seed=int(hashlib.sha256((str(spec['seed'])+'/'+state['id']).encode()).hexdigest()[:8],16)%(2**31)
        if producer and execution.get('score_reference_root'):
            reference=json.loads((Path(execution['score_reference_root'])/'candidates'/(state['id']+'.json')).read_text())
            if reference['condition_sha256']!=condition:raise ValueError('Fixed scoring reference provenance differs')
            values={};started=time.monotonic()
            for name,rationale in [('base_first',''),('base_repeat','')]+[
                    ('candidate_'+str(c['candidate']),c['rationale']) for c in reference['candidates'] if c.get('rationale')]:
                value,_=score(state,rationale,seed)
                if value is None:raise ValueError('Fixed reference does not fit scoring context')
                values[name]=value
            reports.append({'state_id':state['id'],'scores':values,'seconds':time.monotonic()-started})
            write(worker_root/'fixed-scores.json',{'states':reports,'execution':execution})
            progress(worker_root,'fixed_scoring_check',f'{len(reports)}/{len(indices)} fixed states rescored.',training_started=False)
            continue
        with candidate_guard(root,state['id']):
            if path.exists():
                record=json.loads(path.read_text())
                if record['condition_sha256']!=condition:raise ValueError('Existing candidate provenance differs')
            else:
                timings.clear()
                tick=time.monotonic();base,base_batch=score(state,'',seed)
                if base is None:raise ValueError('Teacher command no longer fits complete scoring context')
                prep_started=time.monotonic()
                messages=backward(state);request=prompt(processor,messages,[])
                expected=processor(text=[request['prompt']],images=images_of(messages),return_tensors='pt')['input_ids'][0].tolist()
                if len(expected)+spec['candidate_tokens']>spec['context']:raise ValueError('Complete candidate budget does not fit')
                timings['generation_preprocessing_seconds']=time.monotonic()-prep_started
                generated_started=time.monotonic()
                output=llm.generate([request],SamplingParams(n=8,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,
                    max_tokens=spec['candidate_tokens'],seed=seed),use_tqdm=False)[0]
                timings['generation_engine_seconds']=time.monotonic()-generated_started
                if list(output.prompt_token_ids)!=expected:raise ValueError('Generation HF/vLLM image-token parity failed')
                candidates=[];seen=set()
                for answer in output.outputs:
                    rationale,reason=candidate_reasoning(answer.text,answer.finish_reason)
                    candidate={'candidate':answer.index,'text':answer.text,'tokens':len(answer.token_ids),'finish_reason':answer.finish_reason,'eligible':False,'reason':reason}
                    if rationale is not None:
                        if rationale in seen:candidate['reason']='duplicate_explanation'
                        else:
                            seen.add(rationale);conditioned,_=score(state,rationale,seed)
                            if conditioned is None:candidate['reason']='complete_example_exceeds_context'
                            else:
                                if any(conditioned[k]!=base[k] for k in ['target_tokens','target_ids_sha256']):raise ValueError('Conditional and baseline command tokens differ')
                                delta=conditioned['mean_target_logp']-base['mean_target_logp']
                                candidate.update(rationale=rationale,score=conditioned,delta=delta,
                                    full_json_delta=conditioned['full_json_mean_logp']-base['full_json_mean_logp'],
                                    eligible=delta>0,reason=None if delta>0 else 'nonpositive_command_delta')
                    candidates.append(candidate)
                if len(candidates)!=8:raise ValueError('Incomplete candidate set')
                chosen=sorted([c for c in candidates if c['eligible']],key=lambda c:(-c['delta'],c['candidate']))[:2]
                record={'state_id':state['id'],'task':state['task'],'suite':state['suite'],'condition_sha256':condition,
                    'base':base,'candidates':candidates,'selected':[c['candidate'] for c in chosen],'seconds':time.monotonic()-tick,
                    'generation_prompt_tokens':len(expected),'exact_generation_tokens_verified':True,
                    'timings':dict(timings),'execution':{'max_num_seqs':max_sequences,'workflow':os.environ.get('EFE_WORKFLOW_ID')}}
                write(path,record)
        if producer:
            reports.append({'state_id':state['id'],'seconds':record['seconds'],'timings':record.get('timings',{}),'selected':len(record['selected'])})
            progress(worker_root,'selection_worker',f'{len(reports)}/{len(indices)} assigned states available.',completed=len(reports),total=len(indices),training_started=False)
            write(worker_root/'timings.json',{'states':reports,'execution':execution})
            continue
        if index<2:
            base,batch=score(state,'',seed);p=root/'probes'/f'base-{index}.pt';torch.save(batch,p)
            probes.append({'condition':'base-'+str(index),'tensor_file':str(p.relative_to(root)),'tensor_sha256':sha(p),'mean_target_logp':base['mean_target_logp']})
        for candidate in record['candidates']:
            if candidate['candidate'] not in record['selected']:continue
            pair=state['id']+'-leact-'+str(candidate['candidate']);rationale=candidate['rationale']
            for direction,messages,answer in [('forward',state['messages'],json.dumps(state['action'],ensure_ascii=False)),('backward',backward(state),'DONE')]:
                batch,meta=encode_example(processor,messages,[],{'role':'assistant','reasoning_content':rationale,'content':answer},spec['context'])
                if batch is None:raise ValueError('Admitted complete training pair exceeds context')
                name='tensors/'+pair+'-'+direction+'.pt'
                if spec.get('tensor_storage')=='shared-visual-tensors-v1':
                    from efe_visual_tensor_store import save
                    save(batch,root/name)
                else:torch.save(batch,root/name)
                rows.append({'id':pair+'-'+direction,'state_id':pair,'decision_id':state['id'],'direction':direction,
                    'suite':state['suite'],'task':state['task'],'episode':state['episode'],'source':'leact','tensor_file':name,
                    'tensor_sha256':sha(root/name),**meta})
        reports.append(record)
        write(root/'candidate-report.json',{'states':reports,'completed':len(reports),'total':len(states)})
        progress(root,'leact_generation',f'{index+1}/{len(states)} states scored; {sum(bool(r["selected"]) for r in reports)} have admitted explanations.',
            completed=index+1,total=len(states),admitted_states=sum(bool(r['selected']) for r in reports),training_started=False)
    if producer:
        write(worker_root/'complete.json',{'success':True,'states':len(reports),'candidate_only':True,'completed_utc':now()})
        return
    selection={'states_attempted':len(states),'candidate_count':8,'max_positive_candidates':2,
        'admitted_candidates':len(rows)//2,'admitted_states':len({r['decision_id'] for r in rows}),
        'rule':'Strictly positive mean log-likelihood gain on identical executable-command tokens; top two per state. Summary tokens cannot determine admission.'}
    write(root/'selection.json',selection)
    if not rows:write(root/'selection-complete.json',{'success':True,'no_admissions':True});return
    parent=Path(spec['parent_corpus']);manifest=json.loads((parent/'dataset-manifest.json').read_text())
    if sha(parent/'dataset-manifest.json')!=spec['parent_dataset_sha256']:raise ValueError('Warmstart replay corpus changed')
    for row in manifest['examples']:
        source=parent/row['tensor_file'];dest=root/row['tensor_file']
        link_immutable(source,dest,row['tensor_sha256'])
        rows.append(dict(row,source='warmstart'))
    if spec.get('tensor_storage')=='shared-visual-tensors-v1':
        (root/'tensor-blobs').mkdir(exist_ok=True)
        for source in (parent/'tensor-blobs').glob('*.pt'):
            destination=root/'tensor-blobs'/source.name
            link_immutable(source,destination)
    write(root/'dataset-manifest.json',{'examples':rows,'selection':selection,'replay_fraction':.5,'target_truncation':False,
        'teacher_episodes':spec.get('teacher_episodes',3),'native_episodes':0,'parent_dataset_sha256':spec['parent_dataset_sha256'],
        'source_sha256':spec['decisions_sha256'],'private_reasoning_exported':False})
    write(root/'scoring-probes.json',probes)
    write(root/'selection-complete.json',{'success':True,'no_admissions':False,'examples':len(rows)})


def parity(spec,root):
    import torch
    from transformers import AutoConfig,AutoProcessor
    from trainer import load_tensor,modality_inputs
    config=AutoConfig.from_pretrained(spec['base_model'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
    rows=json.loads((root/'dataset-manifest.json').read_text())['examples']
    for row in rows:
        path=root/row['tensor_file']
        if sha(path)!=row['tensor_sha256']:raise ValueError('Frozen training tensor changed')
        batch=load_tensor(path);adapted=modality_inputs(batch,config)
        expected=torch.tensor(processor.create_mm_token_type_ids(batch['input_ids'].tolist()),dtype=batch['input_ids'].dtype)
        if not torch.equal(adapted['mm_token_type_ids'],expected):raise ValueError('HF5 modality labels differ')
        if any(not torch.equal(batch[k],adapted[k]) for k in batch):raise ValueError('Frozen tensor changed during modality adaptation')
    write(root/'modality-parity.json',{'examples_checked':len(rows),'matches_official_processor':True,'frozen_inputs_and_targets_unchanged':True})


def run(spec,root,lane):
    import importlib.metadata
    root.mkdir(parents=True,exist_ok=True);write(root/'spec.json',spec)
    write(root/'worker-start.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'started_utc':now()})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','NCCL_P2P_DISABLE':'1','OMP_NUM_THREADS':'2',
        'TOKENIZERS_PARALLELISM':'false','VLLM_WORKER_MULTIPROC_METHOD':'spawn','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}
    def logged(name,cmd,timeout):
        with (root/(name+'.log')).open('w') as out:
            completed=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
        if completed.returncode:raise RuntimeError(name+' failed; see '+name+'.log')
    try:
        progress(root,'preparing_parent','Copying and verifying the final warmstart checkpoint for LeAct.',training_started=False)
        original=Path(spec['parent_checkpoint']);manifest=json.loads(Path(spec['parent_manifest']).read_text())
        if manifest['revision']!=spec['parent_revision']:raise ValueError('Unexpected warmstart revision')
        model=Path(spec['base_model']);model.mkdir(parents=True,exist_ok=True)
        for name,row in manifest['files'].items():
            if Path(name).name!=name:raise ValueError('Invalid checkpoint member')
            destination=model/name
            if not destination.exists():shutil.copyfile(original/name,destination)
            if destination.stat().st_size!=row['bytes'] or sha(destination)!=row['sha256']:raise ValueError('Warmstart copy checksum mismatch')
        write(root/'parent-verification.json',{'revision':manifest['revision'],'all_files_verified':True,'files':len(manifest['files'])})
        write(root/'selection-runtime.json',{k:importlib.metadata.version(k) for k in ['torch','transformers','vllm']})
        progress(root,'leact_generation','Sampling eight student explanations on each of the 43 fixed teacher states.',training_started=False)
        logged('selection',[sys.executable,str(lane/'pipeline.py'),'--spec',str(root/'spec.json'),'--mode','select'],4*3600)
        selected=json.loads((root/'selection-complete.json').read_text())
        if selected['no_admissions']:
            write(root/'complete.json',{'success':True,'no_admissions':True,'optimizer_steps':0,'completed_utc':now()})
            progress(root,'no_admissions','No positive command-likelihood candidates. Warmstart retained without another update or duplicate eval.');return
        training=lane/'training-env';subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(training)],check=True)
        python=str(training/'bin/python')
        logged('training-environment',[python,'-m','pip','install','--disable-pip-version-check','transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'],600)
        env.update(VIRTUAL_ENV=str(training),PATH=str(training/'bin')+':'+env['PATH'])
        logged('modality-parity',[python,str(lane/'pipeline.py'),'--spec',str(root/'spec.json'),'--mode','parity'],600)
        train=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(lane/'trainer.py'),'--spec',str(root/'spec.json'),'--root',str(root)]
        def replay(mode):
            ready=json.loads((root/('smoke-adapter-ready.json' if mode=='smoke' else 'adapter-ready.json')).read_text())
            completed=json.loads((root/mode/'complete.json').read_text())
            path=root/(mode+'-verification-spec.json')
            write(path,{**spec,'checkpoint':ready['checkpoint'],'training_root':str(root),'verification_mode':mode,'expected_post_update_ce':completed['post_update_full_target_ce']})
            progress(root,'checkpoint_loss_replay',f'Verifying a fresh-model reload of the {mode} adapter.',training_started=mode=='train')
            logged(mode+'-reload',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(lane/'verify.py'),'--spec',str(path),'--mode','verify'],1800)
        progress(root,'optimizer_smoke','Checking complete-command likelihood parity and two discarded optimizer updates.',training_started=False)
        logged('smoke',train+['--mode','smoke'],3600);replay('smoke')
        progress(root,'leact_training','Training 32 LeAct LoRA updates with 50% warmstart replay and equal forward/backward mass.',training_started=True)
        logged('train',train+['--mode','train'],4*3600);replay('train')
        progress(root,'merging_checkpoint','Exporting the verified LeAct adapter for the same four native tasks.',training_started=True)
        logged('merge',[python,str(lane/'merge.py'),'--spec',str(root/'spec.json')],3600)
        write(root/'complete.json',{'success':True,'optimizer_steps':32,'completed_utc':now(),
            'selection':json.loads((root/'selection.json').read_text()),'leact':json.loads((root/'leact-ready.json').read_text())})
        progress(root,'leact_ready','Verified LeAct checkpoint ready for matched native evaluation.',training_started=True)
    except Exception as error:
        write(root/'complete.json',{'success':False,'error':str(error),'completed_utc':now()})
        progress(root,'needs_review',str(error),training_started=(root/'train/metrics.jsonl').exists());raise
    finally:
        for _ in range(120):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--mode',choices=['run','select','parity'],default='run');args=parser.parse_args()
    spec=json.loads(args.spec.read_text());root=Path(spec['archive_root'])
    if args.mode=='run':run(spec,root,Path(__file__).parent)
    elif args.mode=='select':select(spec,root)
    else:parity(spec,root)
