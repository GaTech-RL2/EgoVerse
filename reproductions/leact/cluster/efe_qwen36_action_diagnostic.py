"""Compare warmstart and LeAct on the same frozen teacher observations."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from efe_training_data import encode_example,sha,write


def now():return datetime.now(timezone.utc).isoformat()


def aggregate(records):
    def mean(key):
        values=[r[key] for r in records if r.get(key) is not None]
        return statistics.mean(values) if values else None
    return {'states':len(records),'valid_outputs':sum(r['valid'] for r in records),
        'stop_outputs':sum(r.get('stop',False) for r in records),
        'exact_executable_commands':sum(r.get('exact_command',False) for r in records),
        'numeric_error_states':sum(r.get('xyz_mae') is not None for r in records),
        'mean_xyz_component_absolute_error':mean('xyz_mae'),'mean_repeat_absolute_error':mean('repeat_error'),
        'mean_translation_cosine':mean('translation_cosine'),'gripper_sign_matches':sum(r.get('gripper_sign_match',False) for r in records),
        'mean_empty_reasoning_command_logp':mean('empty_command_logp'),
        'mean_forward_reasoning_command_logp':mean('forward_command_logp'),
        'mean_backward_reasoning_command_logp':mean('backward_command_logp'),
        'mean_generation_tokens':mean('generation_tokens'),
        'warning':'Teacher-command agreement and likelihood are support-set diagnostics, not native task success. Other commands may also be valid.'}


def worker(spec,root,arm):
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from efe_leact_vl import prompt,images_of
    from leact_reference import physical_positions
    from efe_action_loop import parse_action
    model=Path(spec['local_models'])/arm
    processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    processor.image_processor.size.update(shortest_edge=65536,longest_edge=1048576)
    llm=LLM(model=str(model),tokenizer=str(model),dtype='bfloat16',tensor_parallel_size=8,
        distributed_executor_backend='mp',disable_custom_all_reduce=True,enforce_eager=True,
        max_model_len=65536,max_num_seqs=1,max_num_batched_tokens=2048,gpu_memory_utilization=.75,
        enable_prefix_caching=False,limit_mm_per_prompt={'image':40,'video':0},
        mm_processor_kwargs={'size':dict(processor.image_processor.size)},seed=spec['seed'])
    source=Path(spec['decisions'])
    if sha(source)!=spec['decisions_sha256']:raise ValueError('Frozen teacher states changed')
    states=[json.loads(s) for s in gzip.decompress(source.read_bytes()).splitlines()]
    reports={r['state_id']:r for r in json.loads(Path(spec['candidate_report']).read_text())['states']}
    if len(states)!=43 or set(reports)!={s['id'] for s in states}:raise ValueError('Teacher-state membership differs')
    def score(state,reasoning):
        target={'role':'assistant','reasoning_content':reasoning,'content':json.dumps(state['action'],ensure_ascii=False)}
        batch,meta=encode_example(processor,state['messages'],[],target,65536,action_only=True)
        if batch is None:raise ValueError('Complete diagnostic score exceeds context')
        ids=batch['input_ids'][0].tolist();positions=physical_positions(processor.tokenizer,ids,meta['target_start_token'])
        result=llm.generate([prompt(processor,state['messages'],[],target=target)],
            SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=spec['seed']),use_tqdm=False)[0]
        if list(result.prompt_token_ids)!=ids:raise ValueError('Diagnostic score HF/vLLM image-token mismatch')
        values=[float(result.prompt_logprobs[i][ids[i]].logprob) for i in positions]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite complete-command score')
        return {'mean':statistics.mean(values),'tokens':len(values),
            'target_sha256':hashlib.sha256(json.dumps([ids[i] for i in positions]).encode()).hexdigest()}
    records=[]
    for state in states:
        tick=time.monotonic();request=prompt(processor,state['messages'],[])
        expected=processor(text=[request['prompt']],images=images_of(state['messages']),return_tensors='pt')['input_ids'][0].tolist()
        result=llm.generate([request],SamplingParams(temperature=0,max_tokens=4096,seed=spec['seed']),use_tqdm=False)[0]
        if list(result.prompt_token_ids)!=expected:raise ValueError('Diagnostic generation HF/vLLM image-token mismatch')
        out=result.outputs[0];reasoning,answer=(out.text.split('</think>',1) if '</think>' in out.text else ('',out.text))
        reasoning=reasoning.removeprefix('<think>').strip()
        contract=None
        for message in state['messages']:
            if not isinstance(message.get('content'),list):continue
            for part in message['content']:
                if part['type']!='text':continue
                try:visible=json.loads(part['text'])
                except (ValueError,TypeError):continue
                if isinstance(visible,dict) and 'action_contract' in visible:contract=visible['action_contract']
        if contract is None:raise ValueError('Missing archived native action contract')
        row={'state_id':state['id'],'task':state['task'],'arm':arm,'teacher_command':state['action'],
            'forward_reasoning':reasoning,'public_output':answer,'finish_reason':out.finish_reason,
            'generation_tokens':len(out.token_ids),'valid':False,'exact_command':False,'stop':False}
        try:
            if out.finish_reason!='stop':raise ValueError('Generation did not finish within the fixed 4096-token diagnostic budget')
            if out.text.count('</think>')!=1:raise ValueError('Missing or ambiguous reasoning/final boundary')
            predicted=parse_action(answer,contract=contract,allow_stop=True)
            row.update(valid=True,predicted_command=predicted,stop=bool(predicted.get('stop')))
            if not row['stop']:
                target=state['action'];a=predicted['action'];b=target['action']
                row['exact_command']=(predicted['action_type']==target['action_type'] and predicted['repeat']==target['repeat']
                    and len(a)==len(b) and all(abs(x-y)<=1e-6 for x,y in zip(a,b)))
                if predicted['action_type']==target['action_type'] and len(a)==len(b)==4:
                    row.update(xyz_mae=statistics.mean(abs(x-y) for x,y in zip(a[:3],b[:3])),
                        repeat_error=abs(predicted['repeat']-target['repeat']),gripper_sign_match=(a[3]>0)==(b[3]>0))
                    denominator=math.sqrt(sum(x*x for x in a[:3])*sum(x*x for x in b[:3]))
                    row['translation_cosine']=sum(x*y for x,y in zip(a[:3],b[:3]))/denominator if denominator>0 else None
        except (ValueError,TypeError) as error:row['output_error']=str(error)
        prior=reports[state['id']];chosen=next(c for c in prior['candidates'] if c['candidate']==prior['selected'][0])
        scores={'empty':score(state,''),'backward':score(state,chosen['rationale'])}
        if out.finish_reason=='stop' and out.text.count('</think>')==1:scores['forward']=score(state,reasoning)
        if len({v['target_sha256'] for v in scores.values()})!=1:raise ValueError('Compared conditions have different command tokens')
        for name,value in scores.items():row[name+'_command_logp']=value['mean']
        row.update(scores=scores,backward_reasoning=chosen['rationale'],seconds=time.monotonic()-tick)
        records.append(row);write(root/(arm+'-records.json'),{'records':records})
        write(root/(arm+'-summary.json'),aggregate(records))
        write(root/'progress.json',{'phase':'fixed_observation_diagnostic','arm':arm,'completed':len(records),'total':43,
            'detail':f'{arm}: {len(records)}/43 fixed teacher observations measured; no optimizer updates.',
            'updated_utc':now(),'training_started':False})
    write(root/(arm+'-complete.json'),{'success':True,'states':len(records),'summary':aggregate(records),
        'by_task':{task:aggregate([r for r in records if r['task']==task]) for task in sorted({r['task'] for r in records})}})


def run(spec,root,lane):
    root.mkdir(parents=True,exist_ok=True);write(root/'worker-start.json',{'workflow':os.environ['EFE_WORKFLOW_ID'],'started_utc':now()})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','NCCL_P2P_DISABLE':'1','OMP_NUM_THREADS':'2',
        'TOKENIZERS_PARALLELISM':'false','VLLM_WORKER_MULTIPROC_METHOD':'spawn'}
    try:
        for arm,item in spec['checkpoints'].items():
            model=Path(spec['local_models'])/arm;model.mkdir(parents=True,exist_ok=True)
            manifest=json.loads(Path(item['manifest']).read_text())
            if manifest['revision']!=item['revision']:raise ValueError('Unexpected diagnostic checkpoint')
            write(root/'progress.json',{'phase':'preparing_diagnostic','detail':'Copying and verifying '+arm+' weights.','updated_utc':now()})
            for name,row in manifest['files'].items():
                if Path(name).name!=name:raise ValueError('Invalid checkpoint file')
                destination=model/name
                if not destination.exists():shutil.copyfile(Path(item['checkpoint'])/name,destination)
                if destination.stat().st_size!=row['bytes'] or sha(destination)!=row['sha256']:raise ValueError('Diagnostic checkpoint changed')
            with (root/(arm+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,str(lane/'diagnostic.py'),'--spec',str(lane/'spec.json'),'--arm',arm],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3*3600)
            if result.returncode:raise RuntimeError(arm+' diagnostic failed; see '+arm+'.log')
        write(root/'complete.json',{'success':True,'completed_utc':now(),'optimizer_updates':0,
            'arms':{a:json.loads((root/(a+'-complete.json')).read_text()) for a in spec['checkpoints']}})
        write(root/'progress.json',{'phase':'diagnostic_complete','detail':'Both checkpoints measured on all 43 teacher observations. No optimizer updates.','updated_utc':now()})
    except Exception as error:
        write(root/'complete.json',{'success':False,'error':str(error),'completed_utc':now()});raise
    finally:
        for _ in range(120):
            if (root/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--arm',choices=['warmstart','leact']);args=p.parse_args()
    spec=json.loads(args.spec.read_text());root=Path(spec['archive_root'])
    if args.arm:worker(spec,root,args.arm)
    else:run(spec,root,args.spec.parent)
