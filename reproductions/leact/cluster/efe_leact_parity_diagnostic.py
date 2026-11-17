"""Measure the stopped R1 scorer mismatch without changing candidates or weights."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def vllm_check(spec,out):
    import torch
    from transformers import AutoProcessor
    from transformers.feature_extraction_utils import BatchFeature
    from vllm import LLM,SamplingParams
    from efe_leact_vl import prompt,images_of
    from efe_training_data import balanced_order
    from leact import physical_positions
    root=Path(spec['archive_root'])
    source=Path(spec['parent_corpus'])/'decisions.jsonl.gz'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['decisions_sha256']:
        raise ValueError('Teacher membership changed')
    states=balanced_order([json.loads(s)for s in gzip.decompress(source.read_bytes()).splitlines()],spec['seed'])
    processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
    processor.image_processor.size.update(shortest_edge=65536,longest_edge=1048576)
    model=LLM(model=spec['base_model'],tokenizer=spec['base_model'],dtype='bfloat16',tensor_parallel_size=8,
        distributed_executor_backend='mp',disable_custom_all_reduce=True,enforce_eager=True,
        max_model_len=spec['context'],max_num_seqs=8,max_num_batched_tokens=2048,gpu_memory_utilization=.75,
        enable_prefix_caching=False,limit_mm_per_prompt={'image':40,'video':0},
        mm_processor_kwargs={'size':dict(processor.image_processor.size)},seed=spec['seed'])
    result=[]
    probes=json.loads((root/'scoring-probes.json').read_text())
    for index,probe in enumerate(probes):
        with torch.serialization.safe_globals([BatchFeature]):
            batch=dict(torch.load(root/probe['tensor_file'],weights_only=True,map_location='cpu'))
        if hashlib.sha256((root/probe['tensor_file']).read_bytes()).hexdigest()!=probe['tensor_sha256']:
            raise ValueError('Scoring probe changed')
        state=states[index]
        target={'role':'assistant','reasoning_content':'','content':json.dumps(state['action'],ensure_ascii=False)}
        request=prompt(processor,state['messages'],[],target=target)
        expected=batch['input_ids'][0].tolist();positions=(batch['labels'][0]!=-100).nonzero().flatten().tolist()
        seed=int(hashlib.sha256((str(spec['seed'])+'/'+state['id']).encode()).hexdigest()[:8],16)%(2**31)
        for repeat in range(3):
            response=model.generate([request],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=1,seed=seed),use_tqdm=False)[0]
            if list(response.prompt_token_ids)!=expected:raise ValueError('Frozen probe and current vLLM inputs differ')
            values=[float(response.prompt_logprobs[i][expected[i]].logprob)for i in positions]
            result.append({'condition':probe['condition'],'repeat':repeat,'state_id':state['id'],
                'original_mean_target_logp':probe['mean_target_logp'],'mean_target_logp':sum(values)/len(values),
                'target_tokens':len(values),'target_positions':positions,'target_token_ids':[expected[i]for i in positions],
                'token_logprobs':values,'input_tokens':len(expected)})
            write(out/'vllm.json',result)
    print('EFE_PARITY_VLLM '+json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['vllm','hf-processor-check','hf-worker-input-check']);args=parser.parse_args()
    lane=Path(__file__).parent;request=json.loads((lane/'diagnostic.json').read_text())
    spec=json.loads((lane/'spec.json').read_text());out=Path(request['output'])/os.environ['EFE_WORKFLOW_ID']
    out.mkdir(parents=True,exist_ok=True)
    if args.phase=='vllm':vllm_check(spec,out);return
    if args.phase in {'hf-processor-check','hf-worker-input-check'}:
        from efe_leact_hf_rescore import processor_for_frozen_inputs
        execution=spec['hf_rescore_execution'];worker=Path(execution['output_root'])/f"worker-{execution['shard']}"
        tensor_root=worker if args.phase=='hf-worker-input-check'else Path(spec['archive_root'])
        _,receipt=processor_for_frozen_inputs(spec,tensor_root)
        write(out/'processor-preflight.json',receipt)
        print('EFE_HF_PROCESSOR '+json.dumps(receipt),flush=True);return
    if not os.environ.get('EFE_PARITY_CUDA_READY'):
        version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
        libs=[Path('/usr/lib/x86_64-linux-gnu')/(n+'.so.'+version)for n in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
        if not all(p.is_file()for p in libs):raise ValueError('Matching host CUDA libraries required')
        env={**os.environ,'EFE_PARITY_CUDA_READY':'1','LD_PRELOAD':':'.join(map(str,libs)),
            'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),
            'NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','NCCL_P2P_DISABLE':'1','OMP_NUM_THREADS':'2',
            'VLLM_WORKER_MULTIPROC_METHOD':'spawn','TOKENIZERS_PARALLELISM':'false'}
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    def run(name,command,timeout=3600):
        write(out/'status.json',{'phase':name,'utc':datetime.now(timezone.utc).isoformat(),'optimizer_updates':0})
        print('EFE_PARITY_PHASE '+name,flush=True)
        with (out/(name+'.log')).open('w')as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,cwd=lane)
        if p.returncode:
            print((out/(name+'.log')).read_text(errors='replace')[-7000:],flush=True)
            raise RuntimeError(name+' failed')
    environment=lane/'diagnostic-env'
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(environment)],check=True)
    python=str(environment/'bin/python')
    run('install',[python,'-m','pip','install','--disable-pip-version-check','transformers==5.11.0','peft==0.21.0','accelerate==1.11.0'])
    if request.get('kind')=='hf-rescore':
        run('processor-preflight',[python,'-u',__file__,'--phase','hf-processor-check'])
    from efe_model_cache import stage
    cache=stage(spec['base_model'],spec['parent_manifest'],lane/'model-cache',spec['base_revision'],
        progress=lambda p:write(out/'cache-progress.json',p))
    runtime={**spec,'local_base_model':cache['local_model']}
    path=out/'runtime.json';write(path,runtime)
    if request.get('kind')=='hf-rescore':
        execution=runtime['hf_rescore_execution'];worker=Path(execution['output_root'])/f"worker-{execution['shard']}"
        worker.mkdir(parents=True,exist_ok=True)
        for name in ['dataset-manifest.json','scoring-probes.json','membership.json','tensors','tensor-blobs','candidates','probes']:
            source=Path(spec['archive_root'])/name;target=worker/name
            if target.is_symlink():
                if target.resolve()!=source.resolve():raise ValueError('HF rescore reference changed')
            elif target.exists():raise ValueError('Unexpected HF rescore reference path')
            else:target.symlink_to(source,target_is_directory=source.is_dir())
        run('worker-input-preflight',[python,'-u',__file__,'--phase','hf-worker-input-check'])
        run('hf-rescore',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(lane/'trainer.py'),'--spec',str(path),'--root',str(worker),'--mode','hf-rescore'],12*3600)
        write(out/'complete.json',{'success':True,'optimizer_updates':0,'kind':'hf-rescore'})
        print('EFE_HF_RESCORE_COMPLETE '+json.dumps({'output':execution['output_root'],'shard':execution['shard']}),flush=True)
        return
    run('hf-scoring',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
        str(lane/'trainer.py'),'--spec',str(path),'--root',spec['archive_root'],'--mode','scoring-check'])
    hf=json.loads((Path(spec['archive_root'])/'scoring-check/scoring-parity.json').read_text())
    write(out/'hf-scoring.json',hf)
    run('vllm-scoring',[sys.executable,'-u',__file__,'--phase','vllm'])
    write(out/'complete.json',{'success':True,'optimizer_updates':0,'hf_scoring':hf,
        'vllm':json.loads((out/'vllm.json').read_text())})
    print('EFE_PARITY_COMPLETE '+json.dumps({'output':str(out),'optimizer_updates':0,'hf_parity_passed':hf['passed']}),flush=True)


if __name__=='__main__':main()
