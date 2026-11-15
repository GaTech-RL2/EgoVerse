"""Bounded H100 qualification of the EFE PEAR sampler/loss data contract.

Uses frozen TRAINING observations only. No simulator score or optimizer step
is claimed: this checks numerical gradients and exact sampler probabilities
before spending native rollout budget on RL.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import traceback


def write(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    p=root/name;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');tmp.replace(p)


def main(root, spec):
    from efe_pear_loss import validate_sampling
    from efe_model_cache import stage
    # A CUDA context retained by the parent can prevent vLLM workers from
    # acquiring a GPU in exclusive-process mode. Run and exit the numerical
    # check before any inference worker starts.
    check=subprocess.run([sys.executable,'-c',
        "import json;from efe_pear_loss import qualify;print(json.dumps(qualify('cuda')))"],
        capture_output=True,text=True,check=True,timeout=90,cwd=Path(__file__).parent)
    numerical=json.loads(check.stdout.strip().splitlines()[-1])
    write(root,'loss-qualification.json',numerical)
    print('PEAR_PROGRESS '+json.dumps({'phase':'loss_qualified','checks':numerical}),flush=True)
    source=Path(spec['coverage_root'])
    ready=json.loads((source/'base-ready.json').read_text())
    local=stage(ready['checkpoint'],ready['manifest'],'/tmp/efe-pear-model',ready['revision'],
                progress=lambda r:write(root,'cache-progress.json',r))
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from PIL import Image
    processor=AutoProcessor.from_pretrained(local['local_model'],local_files_only=True)
    processor.image_processor.size.update(shortest_edge=65536,longest_edge=1048576)
    admission=json.loads((source/'corpus/admission.json').read_text())
    data=(source/'corpus/decisions.jsonl.gz').read_bytes()
    if hashlib.sha256(data).hexdigest()!=admission['decisions_sha256']:
        raise ValueError('Frozen corpus checksum differs')
    states=[json.loads(x) for x in gzip.decompress(data).splitlines()]
    # First deterministic observed training state; no expert command is included
    # in the model input. Native rollout integration is a separate gate.
    state=sorted(states,key=lambda s:s['id'])[0]
    messages=state['messages']
    images=[Image.open(io.BytesIO(base64.b64decode(part['image'].split(',',1)[1]))).convert('RGB')
            for m in messages if isinstance(m.get('content'),list)
            for part in m['content'] if part['type']=='image']
    text=processor.apply_chat_template(messages,tools=[],tokenize=False,add_generation_prompt=True)
    expected=processor(text=[text],images=images,return_tensors='pt')['input_ids'][0].tolist()
    if len(expected)+spec['max_tokens']>spec['context']:
        raise ValueError('Complete observation exceeds the declared context')
    request={'prompt':text,'multi_modal_data':{'image':images}}
    llm=LLM(model=local['local_model'],tokenizer=local['local_model'],dtype='bfloat16',
        tensor_parallel_size=8,distributed_executor_backend='mp',disable_custom_all_reduce=True,
        enforce_eager=True,max_model_len=spec['context'],max_num_seqs=1,max_num_batched_tokens=2048,
        gpu_memory_utilization=.75,enable_prefix_caching=False,max_logprobs=spec['capture_top_k'],
        limit_mm_per_prompt={'image':40,'video':0},
        mm_processor_kwargs={'size':dict(processor.image_processor.size)},seed=spec['seed'])
    rows=[]
    for sample_index in range(spec['samples']):
        out=llm.generate([request],SamplingParams(temperature=1.,top_p=1.,top_k=-1,
            max_tokens=spec['max_tokens'],logprobs=spec['capture_top_k'],seed=spec['seed']+sample_index),use_tqdm=False)[0]
        if list(out.prompt_token_ids)!=expected:
            raise ValueError('Sampler/HF input token mismatch')
        completion=out.outputs[0];heads=[];values=[];selected=[]
        for token,distribution in zip(completion.token_ids,completion.logprobs):
            selected.append(float(distribution[token].logprob))
            top=sorted(distribution,key=lambda t:distribution[t].logprob,reverse=True)[:spec['capture_top_k']]
            heads.append(top);values.append([float(distribution[t].logprob) for t in top])
        record={'token_ids':list(completion.token_ids),'prompt_token_ids':expected,
            'sampled_logprobs':selected,'topk_token_ids':heads,'topk_logprobs':values,
            'sampler_revision':ready['revision'],'temperature':1.,'top_p':1.,'top_k':-1,
            'finish_reason':completion.finish_reason,'text':completion.text,'state_id':state['id'],
            'task':state['task'],'partition':'train_candidates','role':'sampler_qualification',
            'native_rollout':False,'success_reward':None}
        count=validate_sampling(record,len(processor.tokenizer),spec['capture_top_k'])
        with gzip.open(root/f'sample-{sample_index}.json.gz','wt') as f:json.dump(record,f)
        rows.append({'sample':sample_index,'tokens':count,'finish_reason':completion.finish_reason,
                     'sampler_capture_complete':True,'prompt_tokens_match':True})
    result={'success':True,'phase':'sampler_and_loss_qualified','updated_utc':datetime.now(timezone.utc).isoformat(),
        'model':spec['model'],'revision':ready['revision'],'samples':rows,'numerical':numerical,
        'optimizer_updates':0,'native_rollouts':0,'task_success_rate':None,
        'next_gate':'Integrate grouped native EFE rollouts and verify one actual policy update from the final common warmstart.'}
    write(root,'result.json',result)
    print('PEAR_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    # Match the validated EFE host-driver transport workaround.
    if not os.environ.get('EFE_PEAR_DRIVER_READY'):
        version=re.search(r'([0-9]{3}[.][0-9]+(?:[.][0-9]+)?)',Path('/proc/driver/nvidia/version').read_text()).group(1)
        libs=[Path('/usr/lib/x86_64-linux-gnu')/(n+'.so.'+version) for n in ['libcuda','libnvidia-ptxjitcompiler','libnvidia-nvvm']]
        if not all(x.is_file() for x in libs):raise RuntimeError('Matching host libraries missing')
        env={**os.environ,'EFE_PEAR_DRIVER_READY':'1','LD_PRELOAD':':'.join(map(str,libs)),
            'LD_LIBRARY_PATH':'/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH',''),
            'NCCL_P2P_DISABLE':'1','NCCL_ALGO':'Ring','NCCL_PROTO':'Simple','OMP_NUM_THREADS':'2'}
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    try:main(args.output,json.loads(args.spec.read_text()))
    except Exception as error:
        result={'success':False,'error':repr(error),'traceback':traceback.format_exc(),
                'optimizer_updates':0,'native_rollouts':0,'updated_utc':datetime.now(timezone.utc).isoformat()}
        write(args.output,'error.json',result);print('PEAR_RESULT '+json.dumps(result),flush=True);raise
