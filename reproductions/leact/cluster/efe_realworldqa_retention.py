"""Matched base / warmstart-496 evaluation on PEAR's RealWorldQA benchmark.

Evaluation only. No optimizer or robot environment is created. The prompt and
answer extraction follow EvalScope v1.13.0's RealWorldQA adapter (Apache-2.0).
"""
import argparse
from datetime import datetime,timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,allow_nan=False)+'\n');temporary.replace(path)


def protocol_digest(spec):
    return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def save_row(path,row):
    path=Path(path);temporary=path.with_suffix('.partial')
    with gzip.open(temporary,'wt')as stream:json.dump(row,stream,allow_nan=False)
    temporary.replace(path)


def grade(text,target,*,thinking_required=True):
    incomplete=(thinking_required or '<think>'in text)and '</think>'not in text
    final=text.rsplit('</think>',1)[-1].strip()if not incomplete else''
    matches=re.findall(r'ANSWER:\s*(.*)',final)
    answer=matches[-1].strip().lower()if matches else''
    return {'answer':answer,'final':final,'correct':answer==str(target).lower(),
            'format_ok':bool(matches),'unfinished_reasoning':incomplete}


def compare(base,warm,*,resamples=2000,seed=20261118):
    if not base or set(base)!=set(warm):raise ValueError('Both checkpoints must finish the identical questions')
    ids=sorted(base);deltas=[]
    for key in ids:
        if base[key]['input_sha256']!=warm[key]['input_sha256']or base[key]['seed']!=warm[key]['seed']:
            raise ValueError('Paired questions differ in model inputs or random seed')
        deltas.append(int(warm[key]['correct'])-int(base[key]['correct']))
    rng=random.Random(seed);n=len(ids)
    bootstrap=sorted(sum(rng.choices(deltas,k=n))/n for _ in range(resamples))
    def summary(rows):
        values=list(rows.values());return {'questions':len(values),
            'accuracy':sum(r['correct']for r in values)/n,
            'format_failures':sum(not r['format_ok']for r in values),
            'truncated':sum(r['finish_reason']=='length'for r in values),
            'mean_output_tokens':sum(r['output_tokens']for r in values)/n}
    return {'benchmark':'RealWorldQA','questions':n,'base':summary(base),'warmstart496':summary(warm),
        'warm_minus_base':sum(deltas)/n,
        'paired_95ci':[bootstrap[int(.025*resamples)],bootstrap[min(resamples-1,int(.975*resamples))]],
        'base_only_correct':deltas.count(-1),'warm_only_correct':deltas.count(1),
        'bootstrap_resamples':resamples,'scope':'Single benchmark retention check; not a claim about all general capabilities.',
        'optimizer_updates':0,'success':True}


def cuda_environment(count):
    version=re.search(r'\b(\d{3}\.\d+(?:\.\d+)?)\b',Path('/proc/driver/nvidia/version').read_text())
    if not version:raise RuntimeError('Cannot identify NVIDIA host driver')
    candidates=[Path(p)/('libcuda.so.'+version.group(1))for p in ['/usr/lib/x86_64-linux-gnu','/usr/local/nvidia/lib64']]
    library=next((p for p in candidates if p.is_file()),None)
    if library is None:raise RuntimeError('Host-matching CUDA library missing')
    libraries=[library,*[p for name in ['libnvidia-ptxjitcompiler','libnvidia-nvvm']
        if (p:=library.with_name(name+'.so.'+version.group(1))).is_file()]]
    return {**os.environ,'LD_PRELOAD':':'.join(map(str,libraries)),
        'LD_LIBRARY_PATH':str(library.parent)+':'+os.environ.get('LD_LIBRARY_PATH',''),
        'CUDA_VISIBLE_DEVICES':','.join(map(str,range(count))),
        'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','NCCL_P2P_DISABLE':'1','TOKENIZERS_PARALLELISM':'false',
        'VLLM_WORKER_MULTIPROC_METHOD':'spawn'}


def cached_rows(root,*,protocol_sha256=None,model_revision=None):
    result={}
    for path in sorted(Path(root).glob('*.json.gz')):
        with gzip.open(path,'rt')as stream:row=json.load(stream)
        if path.name!=row['id']+'.json.gz' or row['id']in result:raise ValueError('Duplicate or mislabeled cached answer')
        if protocol_sha256 is not None and row.get('protocol_sha256')!=protocol_sha256:
            raise ValueError('Cached answer has a different evaluation protocol')
        if model_revision is not None and row.get('model_revision')!=model_revision:
            raise ValueError('Cached answer has a different checkpoint')
        result[row['id']]=row
    return result


def run_model(spec,ident):
    import datasets
    import importlib.metadata
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    from efe_model_cache import stage
    root=Path(spec['shared_root']);root.mkdir(parents=True,exist_ok=True)
    model=next(m for m in spec['models']if m['id']==ident)
    digest_protocol=protocol_digest(spec)
    def progress(phase,**extra):
        value={'phase':phase,'model':ident,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'workflow':os.environ['EFE_WORKFLOW_ID'],'optimizer_updates':0,**extra}
        write(root/'progress.json',value);print('EFE_RETENTION_PROGRESS '+json.dumps(value),flush=True)
    progress('staging_model')
    write(root/'environment.json',{name:importlib.metadata.version(name)for name in ['torch','vllm','transformers','datasets']})
    if model.get('checkpoint_pin') and sha(model['checkpoint_pin'])!=model['checkpoint_pin_sha256']:
        raise ValueError('The fixed warmstart checkpoint pin changed')
    local=stage(model['path'],model['manifest'],'/tmp/rp_jhh_retention/models',model['revision'])
    base=spec['models'][0]
    processor=AutoProcessor.from_pretrained(base['path'],local_files_only=True)
    processor.image_processor.size.update(spec['image_processor_size'])
    progress('loading_dataset',dataset_revision=spec['dataset']['revision'])
    dataset=datasets.load_dataset(spec['dataset']['id'],revision=spec['dataset']['revision'],split='test')
    if len(dataset)!=765:raise ValueError('The frozen full RealWorldQA split must have 765 rows')
    llm=LLM(model=local['local_model'],tokenizer=base['path'],dtype='bfloat16',
        tensor_parallel_size=spec['gpu_count'],distributed_executor_backend='mp',
        disable_custom_all_reduce=True,enforce_eager=True,max_model_len=65536,max_num_seqs=spec['batch_size'],
        max_num_batched_tokens=spec['serving_profile']['max_num_batched_tokens'],
        gpu_memory_utilization=spec['serving_profile']['gpu_memory_utilization'],
        enable_prefix_caching=spec['serving_profile']['enable_prefix_caching'],
        async_scheduling=spec['serving_profile']['async_scheduling'],disable_log_stats=False,
        limit_mm_per_prompt={'image':1,'video':0},
        mm_processor_kwargs={'size':spec['image_processor_size']},generation_config='auto',seed=spec['seed'])
    destination=root/'responses'/ident;destination.mkdir(parents=True,exist_ok=True)
    rows=cached_rows(destination,protocol_sha256=digest_protocol,model_revision=model['revision'])
    progress('evaluating',completed=len(rows),total=len(dataset),correct=sum(r['correct']for r in rows.values()))
    for start in range(0,len(dataset),spec['batch_size']):
        inputs=[];parameters=[];metadata=[]
        for index in range(start,min(start+spec['batch_size'],len(dataset))):
            row=dataset[index];key=f'{index:04d}';path=destination/(key+'.json.gz')
            if path.exists():continue
            image=row['image']
            if isinstance(image,dict):image=Image.open(io.BytesIO(image['bytes']))
            image=image.convert('RGB');pixel_hash=hashlib.sha256(image.tobytes()).hexdigest()
            text=spec['prompt'].format(question=row['question'])
            messages=[{'role':'user','content':[{'type':'text','text':text},{'type':'image','image':image}]}]
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            encoded=processor(text=[prompt],images=[image],return_tensors='pt')['input_ids'][0].tolist()
            if len(encoded)+spec['max_output_tokens']>65536:raise ValueError('Retention input would exceed context')
            digest=hashlib.sha256(json.dumps({'tokens':encoded,'pixels':pixel_hash,'size':image.size},sort_keys=True).encode()).hexdigest()
            seed=(spec['seed']+index)%(2**31)
            inputs.append({'prompt':prompt,'multi_modal_data':{'image':image}})
            parameters.append(SamplingParams(temperature=1.,top_p=.95,top_k=20,
                max_tokens=spec['max_output_tokens'],seed=seed))
            metadata.append({'id':key,'target':str(row['answer']).lower(),'question':row['question'],
                'seed':seed,'input_sha256':digest,'prompt_token_ids':encoded,
                'thinking_required':prompt.rstrip().endswith('<think>')})
        if inputs:
            before=time.monotonic();outputs=llm.generate(inputs,parameters,use_tqdm=False)
            for meta,out in zip(metadata,outputs,strict=True):
                if list(out.prompt_token_ids)!=meta['prompt_token_ids']:raise ValueError('HF/vLLM multimodal input token mismatch')
                answer=out.outputs[0]
                if answer.finish_reason not in {'stop','length'}:raise RuntimeError('Incomplete inference result')
                result={**meta,**grade(answer.text,meta['target'],thinking_required=meta['thinking_required']),
                    'text':answer.text,'finish_reason':answer.finish_reason,'output_tokens':len(answer.token_ids),
                    'model_revision':model['revision'],'manifest_sha256':local['source_manifest_sha256'],
                    'protocol_sha256':digest_protocol,'batch_seconds':time.monotonic()-before}
                save_row(destination/(meta['id']+'.json.gz'),result)
                rows[meta['id']]=result
        progress('evaluating',completed=len(rows),total=len(dataset),correct=sum(r['correct']for r in rows.values()),
            truncated=sum(r['finish_reason']=='length'for r in rows.values()))
    rows=cached_rows(destination)
    if len(rows)!=765:raise RuntimeError('Incomplete evaluation panel')
    write(root/(ident+'-complete.json'),{'success':True,'questions':765,'revision':model['revision'],
        'optimizer_updates':0,'versions':{'torch':torch.__version__,'datasets':datasets.__version__}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--model')
    args=parser.parse_args();spec=json.loads(args.spec.read_text());root=Path(spec['shared_root']);root.mkdir(parents=True,exist_ok=True)
    if args.model:return run_model(spec,args.model)
    lock=(root/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import importlib.util
    if importlib.util.find_spec('datasets')is None:
        subprocess.run([sys.executable,'-m','pip','install','datasets==4.4.1'],check=True,timeout=600)
    write(root/'protocol.json',spec)
    try:
        for model in spec['models']:
            subprocess.run([sys.executable,'-u',__file__,'--spec',str(args.spec),'--model',model['id']],
                env=cuda_environment(spec['gpu_count']),check=True)
        base,warm=[cached_rows(root/'responses'/m['id'])for m in spec['models']]
        result=compare(base,warm);result['completed_utc']=datetime.now(timezone.utc).isoformat()
        result['models']=spec['models'];result['dataset']=spec['dataset'];write(root/'results.json',result)
        write(root/'progress.json',{'phase':'completed','updated_utc':result['completed_utc'],'optimizer_updates':0})
        print('EFE_RETENTION_RESULT '+json.dumps(result),flush=True)
    except Exception as exc:
        write(root/'error.json',{'error':str(exc),'updated_utc':datetime.now(timezone.utc).isoformat(),'optimizer_updates':0})
        raise


if __name__=='__main__':main()
