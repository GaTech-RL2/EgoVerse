"""One EFE LeAct pilot: N=8 backward candidates, complete-action delta, top K=2."""
from __future__ import annotations

import argparse
import base64
import errno
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

try:
    from .efe_training_data import balanced_order, backward_messages, encode_example, forward_target, sha, write
except ImportError:
    from efe_training_data import balanced_order, backward_messages, encode_example, forward_target, sha, write

SPEC=Path('/tmp/efe-training-spec.json')


def link_replay(source, destination):
    """Keep immutable replay tensors usable when outputs are on node-local disk."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source)!=sha(destination):raise ValueError('Existing replay file differs')
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)


def images_of(messages):
    from PIL import Image
    return [Image.open(io.BytesIO(base64.b64decode(part['image'].split(',',1)[1]))).convert('RGB')
            for message in messages if isinstance(message.get('content'),list)
            for part in message['content'] if part['type']=='image']


def prompt(processor, messages, tools, *, target=None):
    text=processor.apply_chat_template(messages+([target] if target else []),tools=tools,
                                       tokenize=False,add_generation_prompt=target is None)
    images=images_of(messages)
    return {'prompt':text,**({'multi_modal_data':{'image':images}} if images else {})}


def action_score(llm, processor, state, rationale, cutoff, seed):
    """Score every exact action token, including multi-line code/tool arguments."""
    import torch
    from vllm import SamplingParams
    target=forward_target(state['action'],rationale)
    encoded,receipt=encode_example(processor,state['messages'],state['tools'],target,cutoff,action_only=True)
    if encoded is None:return None,None,None
    ids=encoded['input_ids'][0].tolist()
    start=receipt['target_start_token']
    request=prompt(processor,state['messages'],state['tools'],target=target)
    result=llm.generate([request],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=seed),use_tqdm=False)[0]
    if list(result.prompt_token_ids)!=ids:raise ValueError('vLLM/Transformers multimodal prompt tokens differ')
    values=[]
    for i in range(start,len(ids)):
        row=result.prompt_logprobs[i]
        if row is None or ids[i] not in row:raise ValueError('Incomplete target log probabilities')
        values.append(float(row[ids[i]].logprob))
    if not values or not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite/empty complete-target score')
    return {'mean_target_logp':sum(values)/len(values),'target_tokens':len(values),
            'target_ids_sha256':hashlib.sha256(json.dumps(ids[start:]).encode()).hexdigest(),
            'complete_prompt_tokens':len(ids)},encoded,receipt


def reasoning_from_output(text, finish_reason, action):
    if finish_reason!='stop':return None,'generation_did_not_finish'
    if '</think>' not in text:return None,'missing_complete_reasoning_delimiter'
    reasoning,final=text.split('</think>',1)
    reasoning=reasoning.removeprefix('<think>').strip()
    if not reasoning or any(marker in reasoning for marker in ('```','<tool_call>','<|im_','</think>')):
        return None,'empty_or_executable_reasoning'
    # Reject verbatim executable fragments even without markdown fences.
    for call in action:
        for value in call['function']['arguments'].values():
            if isinstance(value,str):
                for line in value.splitlines():
                    if len(line.strip())>=24 and line.strip() in reasoning:
                        return None,'copied_executable_fragment'
    return reasoning,None


def worker(spec, root, shard):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import torch
    processor=AutoProcessor.from_pretrained(spec['tokenizer_base'],local_files_only=True)
    processor.image_processor.max_pixels=1048576
    if isinstance(getattr(processor.image_processor,'size',None),dict):processor.image_processor.size['longest_edge']=1048576
    llm=LLM(model=spec['base_model'],tokenizer=spec['tokenizer_base'],dtype='bfloat16',tensor_parallel_size=1,
            max_model_len=spec.get('generation_context',65536),max_num_seqs=spec.get('generation_max_sequences',2),max_num_batched_tokens=2048,gpu_memory_utilization=.65,
            enable_prefix_caching=False,enforce_eager=True,limit_mm_per_prompt={'image':40,'video':0},
            # Qwen3-VL reads size, whereas the legacy max_pixels override left
            # vLLM image tokens different from the admitted HF training tensors.
            # Reserve room for full-vocabulary prompt log probabilities too.
            mm_processor_kwargs={'size':dict(processor.image_processor.size)},seed=spec['seed'])
    states=[json.loads(line) for line in (root/'states.jsonl').read_text().splitlines()]
    for index,state in enumerate(states):
        if index%8!=shard:continue
        path=root/'candidates'/(state['id']+'.json')
        condition=hashlib.sha256(json.dumps({'spec':spec,'state':state},sort_keys=True).encode()).hexdigest()
        if path.exists():
            prior=json.loads(path.read_text())
            if not spec.get('resume_completed_candidates') or prior.get('condition_sha256')!=condition or len(prior.get('candidates',[]))!=8:
                raise ValueError('Candidate output exists without matching immutable provenance')
            if index//8 or (root/'probes'/f'base-{shard}.json').exists():continue
        seed=int(hashlib.sha256((str(spec['seed'])+'/'+state['id']).encode()).hexdigest()[:8],16)%(2**31)
        start=time.monotonic()
        base,base_tensor,_=action_score(llm,processor,state,'',spec['context'],seed)
        if base is None:raise ValueError('Previously admitted expert action no longer fits scoring context')
        messages=backward_messages(state['messages'],state['action'])
        request=prompt(processor,messages,state['tools'])
        if path.exists():
            probe_path=root/'probes'/f'base-{shard}.pt';torch.save(base_tensor,probe_path)
            write(root/'probes'/f'base-{shard}.json',{'condition':'base','tensor_file':str(probe_path.relative_to(root)),
                'tensor_sha256':sha(probe_path),'mean_target_logp':base['mean_target_logp']})
            continue
        expected_generation=None
        if spec.get('verify_generation_tokens'):
            encoded=processor(text=[request['prompt']],images=request.get('multi_modal_data',{}).get('image'),return_tensors='pt')
            expected_generation=encoded['input_ids'][0].tolist()
            if len(expected_generation)+spec['candidate_tokens']>spec.get('generation_context',65536):
                raise ValueError('Complete backward prompt plus declared candidate budget exceeds generation context')
            del encoded
        output=llm.generate([request],SamplingParams(n=8,temperature=1.,top_p=.95,top_k=20,repetition_penalty=1.,
                            max_tokens=spec['candidate_tokens'],seed=seed),use_tqdm=False)[0]
        if expected_generation is not None and list(output.prompt_token_ids)!=expected_generation:
            raise ValueError('Generation HF/vLLM multimodal tokens differ')
        candidates=[];seen=set()
        for completion in output.outputs:
            r,reason=reasoning_from_output(completion.text,completion.finish_reason,state['action'])
            record={'candidate':completion.index,'text':completion.text,'finish_reason':completion.finish_reason,
                    'tokens':len(completion.token_ids),'eligible':False,'reason':reason}
            if r is not None:
                if r in seen:record['reason']='duplicate_reasoning'
                else:
                    seen.add(r)
                    score,_,_=action_score(llm,processor,state,r,spec['context'],seed)
                    if score is None:record['reason']='complete_scoring_example_exceeds_context'
                    else:
                        if score['target_ids_sha256']!=base['target_ids_sha256'] or score['target_tokens']!=base['target_tokens']:
                            raise ValueError('Conditional and baseline scores use different action targets')
                        delta=score['mean_target_logp']-base['mean_target_logp']
                        record.update(rationale=r,score=score,delta=delta,eligible=math.isfinite(delta) and delta>0,
                                      reason=None if delta>0 else 'nonpositive_delta')
            candidates.append(record)
        if len(candidates)!=8:raise ValueError('LeAct must retain all eight attempted candidates')
        selected=sorted([c for c in candidates if c['eligible']],key=lambda c:(-c['delta'],c['candidate']))[:2]
        write(path,{'state_id':state['id'],'task':state['task'],'suite':state['suite'],'base':base,'candidates':candidates,
                    'condition_sha256':condition,'exact_generation_tokens_verified':expected_generation is not None,
                    'selected':[c['candidate'] for c in selected],'seconds':time.monotonic()-start,
                    'generation_prompt_tokens':len(output.prompt_token_ids),'candidate_count':8,'max_positive_candidates':2})
        # First state on every shard supplies an independent likelihood parity
        # probe for the distributed HF trainer before its optimizer update.
        if index//8==0:
            probe_path=root/'probes'/f'base-{shard}.pt';torch.save(base_tensor,probe_path)
            write(root/'probes'/f'base-{shard}.json',{'condition':'base','tensor_file':str(probe_path.relative_to(root)),
                   'tensor_sha256':sha(probe_path),'mean_target_logp':base['mean_target_logp']})
        print(json.dumps({'shard':shard,'state':state['id'],'selected':len(selected),'seconds':time.monotonic()-start}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    spec=json.loads(SPEC.read_text());root=Path(spec['archive_root'])/spec.get('run_id',os.environ['EFE_WORKFLOW_ID'])
    if args.worker is not None:worker(spec,root,args.worker);return
    root.mkdir(parents=True,exist_ok=True)
    if spec.get('checkpoint_durability') and (root/'spec.json').exists() and json.loads((root/'spec.json').read_text())!=spec:
        raise ValueError('Durable LeAct configuration changed')
    write(root/'spec.json',spec)
    def progress(phase,detail,**extra):
        value={'phase':phase,'detail':detail,'workflow':os.environ['EFE_WORKFLOW_ID'],'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
        write(root/'progress.json',value);print(json.dumps(value),flush=True)
    processes=[];logs=[]
    try:
        parent=Path(spec['parent_corpus']);manifest=json.loads((parent/'dataset-manifest.json').read_text())
        if sha(parent/'warmstart-states.jsonl')!=manifest['states_sha256']:raise ValueError('Warmstart state corpus changed')
        checkpoint=json.loads(Path(spec['base_manifest']).read_text())
        if checkpoint['revision']!=spec['base_revision']:
            raise ValueError('Wrong frozen LeAct starting checkpoint revision')
        for name,row in checkpoint['files'].items():
            file=Path(spec['base_model'])/name
            if file.stat().st_size!=row['bytes'] or sha(file)!=row['sha256']:raise ValueError('Warmstart checkpoint changed')
        if spec.get('checkpoint_durability') and (root/'dataset-manifest.json').exists():
            import efe_training_pilot as pilot
            import efe_durable_training as durable
            pilot.ROOT=root
            if not (root/'smoke/complete.json').exists() or not json.loads((root/'smoke/complete.json').read_text()).get('success'):
                raise ValueError('Saved LeAct corpus needs a completed optimizer smoke')
            durable.run(pilot,spec)
            write(root/'complete.json',{'success':True,'leact':json.loads((root/'leact-ready.json').read_text()),
                **json.loads((root/'selection.json').read_text())})
            return
        states=[json.loads(line) for line in (parent/'warmstart-states.jsonl').read_text().splitlines()]
        selected=balanced_order(states,spec['seed'])[:spec['leact_decisions']]
        (root/'states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in selected))
        for name in ['candidates','probes','tensors']:(root/name).mkdir(exist_ok=True)
        write(root/'membership.json',{'states':[s['id'] for s in selected],'source':'Fixed warmstart teacher states; new student backward reasoning, no new simulator trajectories',
             'candidate_count':8,'top_positive':2,'candidate_tokens':spec['candidate_tokens'],'scoring_context':spec['context']})
        import efe_cuda_helpers as helpers
        helpers.ROOT=root;helpers.ARCHIVE=root;env=helpers.driver_environment()
        progress('leact_generation','Sampling eight student backward candidates per frozen teacher state and scoring the complete expert action.',completed=0,total=len(selected))
        for shard in range(8):
            stream=(root/f'worker-{shard}.log').open('w');logs.append(stream)
            processes.append(subprocess.Popen([sys.executable,'/tmp/efe_leact_vl.py','--worker',str(shard)],
                  env={**env,'CUDA_VISIBLE_DEVICES':str(shard),'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
                       'VLLM_PORT':str(31000+shard*100)},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        deadline=time.monotonic()+8*3600
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('LeAct worker failed; inspect per-shard logs')
            if time.monotonic()>deadline:raise TimeoutError('LeAct generation exceeded eight hours')
            records=[json.loads(p.read_text()) for p in (root/'candidates').glob('*.json')]
            progress('leact_generation',f'{len(records)}/{len(selected)} states scored; {sum(bool(r["selected"]) for r in records)} have positive candidates.',
                     completed=len(records),total=len(selected),admitted_states=sum(bool(r['selected']) for r in records))
            time.sleep(30)
        if any(p.returncode for p in processes):raise RuntimeError('LeAct generation worker failed')
        from transformers import AutoProcessor
        import torch
        processor=AutoProcessor.from_pretrained(spec['tokenizer_base'],local_files_only=True)
        processor.image_processor.max_pixels=1048576
        if isinstance(getattr(processor.image_processor,'size',None),dict):processor.image_processor.size['longest_edge']=1048576
        rows=[];admitted=0;exclusions=[]
        for state in selected:
            record=json.loads((root/'candidates'/(state['id']+'.json')).read_text())
            for candidate in record['candidates']:
                if candidate['candidate'] not in record['selected']:continue
                r=candidate['rationale'];pair=state['id']+'-leact-'+str(candidate['candidate'])
                pairs=[('forward',state['messages'],forward_target(state['action'],r)),
                       ('backward',backward_messages(state['messages'],state['action']),{'role':'assistant','content':'<think>\n'+r+'\n</think>\n\nDONE'})]
                encoded=[(direction,*encode_example(processor,messages,state['tools'],target,spec['context'])) for direction,messages,target in pairs]
                if any(data is None for _,data,_ in encoded):
                    exclusions.append({'state_id':state['id'],'candidate':candidate['candidate'],'reason':'Complete training pair exceeds context'});continue
                admitted+=1
                for direction,data,receipt in encoded:
                    ident=pair+'-'+direction;path=root/'tensors'/(ident+'.pt')
                    if spec.get('tensor_storage')=='shared-visual-tensors-v1':
                        from efe_visual_tensor_store import save
                        save(data,path)
                    else:torch.save(data,path)
                    rows.append({'id':ident,'state_id':pair,'decision_id':state['id'],'source':'leact','direction':direction,
                         'suite':state['suite'],'task':state['task'],'episode':state['episode'],'tensor_file':str(path.relative_to(root)),
                         'tensor_sha256':sha(path),**receipt})
        selection={'states_attempted':len(selected),'candidate_count':8,'max_positive_candidates':2,'admitted_candidates':admitted,
                   'admitted_states':len({r['decision_id'] for r in rows}),'exclusions':exclusions,'rule':'Finite positive mean complete-action log-likelihood delta, exact same target tokens'}
        write(root/'selection.json',selection)
        if not admitted:
            progress('no_admissions','No positive complete-target candidates; retain warmstart and report the zero-admission result.')
            write(root/'complete.json',{'success':True,'no_admissions':True,**selection});return
        if manifest.get('tensor_storage')=='shared-visual-tensors-v1':
            for source in (parent/'tensor-blobs').glob('*.pt'):
                link_replay(source,root/'tensor-blobs'/source.name)
        for row in manifest['examples']:
            source=parent/row['tensor_file'];destination=root/row['tensor_file']
            if sha(source)!=row['tensor_sha256']:raise ValueError('Warmstart replay tensor changed')
            link_replay(source,destination)
            rows.append({**row,'source':'warmstart','decision_id':row['state_id']})
        write(root/'dataset-manifest.json',{'examples':rows,'decisions':selection['admitted_states'],
             'teacher_episodes':manifest['teacher_episodes'],'native_episodes':0,'replay_fraction':.5,
             'states_file':'states.jsonl','tensor_storage':spec.get('tensor_storage','inline-tensors-v1'),
             'states_sha256':sha(root/'states.jsonl'),'selection':selection,'private_reasoning_exported':False})
        write(root/'scoring-probes.json',[json.loads(p.read_text()) for p in sorted((root/'probes').glob('*.json'))[:2]])
        import efe_training_pilot as pilot
        pilot.ROOT=root
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(8))),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        train=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','/tmp/efe_train_vl.py','--spec',str(SPEC),'--root',str(root)]
        progress('leact_optimizer_preflight','Checking complete-target scoring parity and longest-example optimizer memory.')
        pilot.run_logged('smoke',train+['--mode','smoke'],env,3600)
        progress('leact_training','Training one half-pass LeAct round with equal new/replay and forward/backward mass.',training_started=True)
        if spec.get('checkpoint_durability'):
            import efe_durable_training as durable
            durable.run(pilot,spec,env)
        else:pilot.run_logged('train',train+['--mode','train'],env,8*3600)
        write(root/'complete.json',{'success':True,'leact':json.loads((root/'leact-ready.json').read_text()),**selection})
        progress('leact_complete','LeAct checkpoint saved; native development evaluation is next.',training_started=True)
    except Exception as exc:
        progress('needs_review',str(exc)[-1600:]);write(root/'complete.json',{'success':False,'error':str(exc)[-1600:]});raise
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for stream in logs:stream.close()
        for _ in range(spec.get('collection_wait_seconds',150)//5):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':main()
