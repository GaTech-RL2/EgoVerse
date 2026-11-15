"""Bounded EFE training-example generation and native action-format audit."""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import zlib


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    spec=json.loads(Path('/tmp/efe-behavior-spec.json').read_text())
    workflow=os.environ['EFE_WORKFLOW_ID'];root=Path(spec['archive_root'])/workflow
    root.mkdir(parents=True,exist_ok=True);Path('/tmp/efe-native').mkdir(exist_ok=True)
    def write(name,value):
        p=root/name;temporary=p.with_suffix(p.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(p)
    def progress(detail):
        write('progress.json',{'workflow':workflow,'phase':'efe_behavior_audit','detail':detail,
              'benchmark_rollouts_started':False,'training_started':False,
              'updated_utc':datetime.now(timezone.utc).isoformat()})
    result={'scope':'Diagnostic only: existing EFE training examples and completed native traces. No new training or general-capability benchmark.',
            'native_actions':[],'training_examples':[],'generations':[]}
    if spec.get('prior_result_file'):
        prior=json.loads(Path(spec['prior_result_file']).read_text())
        result['generations']=prior['generations']
        result['prior_result_sha256']=sha(spec['prior_result_file'])
    write('spec.json',spec)
    try:
        progress('Auditing completed step-96 tool calls and frozen training target tokens.')
        native=Path(spec['native_archive']);trials=json.loads((native/'results.json').read_text())['trials']
        for trial in trials:
            calls=Counter();reasoning=0;requests=0;statuses=Counter()
            for p in (native/'requests'/trial['job']).glob('*.json'):
                record=json.loads(p.read_text());requests+=1;statuses[record.get('status')]+=1
                reasoning+=((record.get('usage') or {}).get('output_tokens_details') or {}).get('reasoning_tokens',0)
                for item in record.get('public_output',[]):
                    if item.get('type')=='function_call':calls[(item.get('name',''),item.get('arguments',''))]+=1
            result['native_actions'].append({'task':trial['task'],'score':trial['score'],'requests':requests,
                'statuses':dict(statuses),'reported_reasoning_tokens':reasoning,
                'top_repeated_actions':[{'name':key[0],'arguments':key[1][:1200],'count':count}
                                        for key,count in calls.most_common(5)]})
        write('behavior-audit.json',result)
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from efe_train_vl import load_tensor,target_loss
        corpus=Path(spec['corpus']);manifest=corpus/'dataset-manifest.json'
        if sha(manifest)!=spec['dataset_sha256']:raise ValueError('Frozen corpus manifest changed')
        rows=json.loads(manifest.read_text())['examples']
        selected=[];suites=set()
        for row in sorted((r for r in rows if r['direction']=='forward'),key=lambda r:(r['tokens'],r['id'])):
            if row['suite'] in suites:continue
            selected.append(row);suites.add(row['suite'])
            if len(selected)==2:break
        processor=AutoProcessor.from_pretrained(spec['models'][0]['path'],local_files_only=True)
        tokenizer=processor.tokenizer;batches=[]
        for row in selected:
            path=corpus/row['tensor_file']
            if sha(path)!=row['tensor_sha256']:raise ValueError('Frozen example checksum changed')
            batch=load_tensor(path);labels=batch['labels'][0];start=int(torch.nonzero(labels!=-100)[0].item())
            target=tokenizer.decode(labels[start:],skip_special_tokens=False)
            result['training_examples'].append({'id':row['id'],'suite':row['suite'],'task':row['task'],
                'prompt_tokens':start,'target_tokens':int((labels!=-100).sum()),
                'prompt_suffix':tokenizer.decode(batch['input_ids'][0,max(0,start-80):start],skip_special_tokens=False),
                'target_prefix':target[:1600],'target_has_think_close':'</think>' in target,
                'tensor_sha256':row['tensor_sha256']})
            batches.append((row,batch,start))
        write('behavior-audit.json',result)
        for entry in spec['models']:
            progress('Testing '+entry['label']+' on the same two frozen EFE training prompts; no optimizer updates.')
            model=Qwen3VLForConditionalGeneration.from_pretrained(entry['path'],dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
            model.config.use_cache=True
            for row,cpu,start in batches:
                batch={k:v.to(device='cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in cpu.items()}
                with torch.inference_mode():ce=float(target_loss(model,batch))
                inputs={k:(v[:,:start] if k in {'input_ids','attention_mask','token_type_ids'} else v)
                        for k,v in batch.items() if k!='labels'}
                torch.manual_seed(0);started=time.monotonic()
                with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=spec['max_new_tokens'],do_sample=False,use_cache=True)
                generated=tokenizer.decode(output[0,start:],skip_special_tokens=False)
                result['generations'].append({'model':entry['label'],'state_id':row['state_id'],
                    'training_target_ce':ce,'generated_tokens':output.shape[1]-start,'seconds':time.monotonic()-started,
                    'greedy_diagnostic_not_benchmark_sampling':True,
                    'has_think_close':'</think>' in generated,'has_tool_call':'<tool_call>' in generated,
                    'generated_text':generated[:12000]})
                write('behavior-audit.json',result)
                del output,inputs,batch
            del model;gc.collect();torch.cuda.empty_cache()
        result['success']=True
    except Exception as exc:
        result.update(success=False,error=str(exc),traceback=traceback.format_exc())
    write('behavior-audit.json',result)
    write('complete.json',{'completed_trials':0,'planned_trials':0,'error':result.get('error'),
          'diagnostic_success':result['success'],'completed_utc':datetime.now(timezone.utc).isoformat()})
    progress('EFE behavior audit complete; results are in behavior-audit.json.')
    if spec.get('node_local_results'):
        files={}
        for name in ['behavior-audit.json','complete.json','progress.json']:
            data=(root/name).read_bytes()
            files[name]={'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()}
        print('EFE_LOCAL_RESULTS '+base64.b64encode(zlib.compress(json.dumps(files).encode())).decode(),flush=True)
    print(json.dumps({'workflow':workflow,'diagnostic_success':result['success'],'error':result.get('error')}),flush=True)
    for _ in range(60):
        if Path('/tmp/efe-native/collected').exists():break
        time.sleep(5)
    if not result['success']:raise RuntimeError(result['error'])


if __name__=='__main__':main()
