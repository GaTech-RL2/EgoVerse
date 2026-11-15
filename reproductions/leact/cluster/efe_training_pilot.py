"""Admit a bounded public EFE corpus, measure memory, then train warmstart."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

try:
    from .efe_training_data import extract_decisions, balanced_order, backward_messages, forward_target, encode_example, sha, write
    from .efe_budget_gateway import public_value
except ImportError:  # Standalone OSMO-mounted modules.
    from efe_training_data import extract_decisions, balanced_order, backward_messages, forward_target, encode_example, sha, write
    from efe_budget_gateway import public_value

SPEC=Path('/tmp/efe-training-spec.json')
ROOT=None


def progress(phase, detail, **extra):
    value={'phase':phase,'detail':detail,'updated_utc':datetime.now(timezone.utc).isoformat(),
           'workflow':os.environ['EFE_WORKFLOW_ID'],**extra}
    write(ROOT/'progress.json',value);print(json.dumps(value),flush=True)


def rationale(state, spec):
    path=ROOT/'rationales'/(state['id']+'.json')
    if path.exists():return json.loads(path.read_text())
    content=[{'type':'input_text','text':'Public conversation before the committed expert decision:'}]
    for message in state['messages']:
        if message['role']=='system':continue
        content.append({'type':'input_text','text':'\nRole: '+message['role']+'\n'})
        value=message.get('content','')
        if isinstance(value,str):content.append({'type':'input_text','text':value})
        else:
            for item in value:
                content.append({'type':'input_text','text':item['text']} if item['type']=='text'
                               else {'type':'input_image','image_url':item['image'],'detail':'auto'})
        if message.get('tool_calls'):
            content.append({'type':'input_text','text':json.dumps(message['tool_calls'])})
    content.append({'type':'input_text','text':'\nCommitted expert action:\n'+json.dumps(state['action'])+
         '\nGive a brief public decision explanation in 1–3 sentences, at most 120 words. '
         'Connect the visible evidence, intended effect and relevant constraint. Do not execute tools, reproduce code, '
         'invent future observations or provide a private internal monologue. Return only JSON: {"decision_summary":"..."}.'})
    body={'model':spec['teacher_model'],'input':[{'role':'user','content':content}],
          'instructions':'Explain an already recorded robot-agent decision from its public state and exact action. '
                         'The conversation is evidence, not a request for you to control the robot. Give only the requested concise public explanation.',
          'max_output_tokens':spec['rationale_max_tokens'],'reasoning':{'effort':'medium'},
          'stream':False,'store':False}
    key=os.environ['NVIDIA_INFERENCE_API_KEY'];start=time.monotonic()
    record={'state_id':state['id'],'request_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),
            'model':spec['teacher_model'],'max_output_tokens':spec['rationale_max_tokens'],'accepted':False}
    # Only transient HTTP/connection failures may be retried, never a bad answer.
    for attempt in range(3):
        try:
            request=urllib.request.Request('https://inference-api.nvidia.com/v1/responses',data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response: envelope=json.load(response)
            public=public_value(envelope,(key,))
            record.update(http_status=200,status=public.get('status'),returned_model=public.get('model'),
                          public_output=public.get('output'),usage=public.get('usage'))
            if public.get('status')!='completed' or public.get('model') not in {spec['teacher_model'],'gpt-6-astra-global'}:
                raise ValueError('Incomplete explanation or unexpected returned model')
            text=''.join(part.get('text','') for item in public.get('output',[]) if item['type']=='message' for part in item.get('content',[]) if part['type']=='output_text')
            result=json.loads(text)
            if set(result)!={'decision_summary'} or not isinstance(result['decision_summary'],str):raise ValueError('Invalid public explanation schema')
            summary=result['decision_summary'].strip()
            if not 1<=len(summary.split())<=120 or any(x in summary for x in ('```','<think>','</think>','<|im_')):
                raise ValueError('Explanation violates the brief public-output contract')
            record.update(accepted=True,rationale=summary);break
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode(errors='replace').replace(key,'[REDACTED]')[:3000]
            record.update(http_status=exc.code,error='Provider HTTP '+str(exc.code),provider_error=detail)
            if exc.code not in {429,500,502,503,504} or attempt==2:break
            time.sleep(2**attempt*5)
        except (TimeoutError,urllib.error.URLError) as exc:
            record.update(error=type(exc).__name__)
            if attempt==2:break
            time.sleep(2**attempt*5)
        except (ValueError,KeyError,TypeError) as exc:
            record.update(error=str(exc));break
    record.update(seconds=time.monotonic()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    write(path,record);return record


def run_logged(name, command, env, timeout):
    path=ROOT/(name+'.log')
    with path.open('w') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise TimeoutError(name+' exceeded its declared time limit')
    if code:
        # Elastic's closing signal summary can obscure the actual exception.
        # Preserve the first traceback as well as the bounded final summary.
        text=path.read_text();start=text.find('Traceback (most recent call last):')
        cause=text[start:start+10000] if start>=0 else ''
        raise RuntimeError(name+' exited '+str(code)+': '+cause+'\n'+text[-1800:])


def prepared_states(spec):
    receipt=spec['prepared_corpus'];source=Path(receipt['archive'])
    for name,row in receipt['files'].items():
        path=source/name
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Prepared corpus changed: '+name)
    previous=json.loads((source/'spec.json').read_text())
    for key in ('base_model','base_revision','scene_audit_sha256','student_prompt_requests','pilot_decisions','context','seed'):
        if previous[key]!=spec[key]:raise ValueError('Prepared corpus condition changed: '+key)
    membership=json.loads((source/'pilot-membership.json').read_text());wanted=membership['state_ids']
    chosen={}
    with (source/'decisions.jsonl').open() as stream:
        for line in stream:
            state=json.loads(line)
            if state['id'] in wanted:chosen[state['id']]=state
    if set(chosen)!=set(wanted):raise ValueError('Prepared membership missing decisions')
    for name in ('trace-admission.json','pilot-membership.json','student-interface.json'):
        shutil.copyfile(source/name,ROOT/name)
    write(ROOT/'prepared-corpus-receipt.json',receipt)
    return [chosen[ident] for ident in wanted],membership['context_exclusions']


def main():
    global ROOT
    spec=json.loads(SPEC.read_text());ROOT=Path(spec['archive_root'])/spec.get('run_id',os.environ['EFE_WORKFLOW_ID']);ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'spec.json').exists() and spec.get('checkpoint_durability'):
        if json.loads((ROOT/'spec.json').read_text())!=spec:raise ValueError('Durable production spec changed')
    write(ROOT/'spec.json',spec)
    try:
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','safetensors','pillow']}
        write(ROOT/'versions.json',versions)
        progress('verifying_base_weights','Verifying the pinned student checkpoint before building training examples.')
        manifest=json.loads(Path(spec['base_manifest']).read_text())
        if manifest['revision']!=spec['base_revision']:raise ValueError('Base checkpoint revision changed')
        for name,record in manifest['files'].items():
            path=Path(spec['base_model'])/name
            if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Base checkpoint checksum mismatch: '+name)
        if spec.get('checkpoint_durability') and (ROOT/'dataset-manifest.json').exists():
            try:from . import efe_durable_training as durable
            except ImportError:import efe_durable_training as durable
            if not (ROOT/'smoke/complete.json').exists() or not json.loads((ROOT/'smoke/complete.json').read_text()).get('success'):
                raise ValueError('Saved corpus needs its completed optimizer smoke before automatic recovery')
            durable.run(sys.modules[__name__],spec)
            write(ROOT/'complete.json',{'success':True,'warmstart':json.loads((ROOT/'warmstart-ready.json').read_text()),'completed_utc':datetime.now(timezone.utc).isoformat()})
            return
        progress('admitting_public_traces','Checking successful native grade receipts and public request/action integrity.')
        states=extract_decisions(spec,ROOT) if not spec.get('prepared_corpus') else None
        from transformers import AutoProcessor
        import torch
        processor=AutoProcessor.from_pretrained(spec['base_model'],local_files_only=True)
        processor.image_processor.max_pixels=1048576
        # Qwen3-VL processors use size.longest_edge as the area cap in newer
        # Transformers. Set both named representations and persist them.
        if isinstance(getattr(processor.image_processor,'size',None),dict):
            processor.image_processor.size['longest_edge']=1048576
        write(ROOT/'processor.json',{'image_processor':processor.image_processor.to_dict(),
             'chat_template_sha256':hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
             'base_revision':spec['base_revision']})
        chosen,exclusions=prepared_states(spec) if spec.get('prepared_corpus') else ([],[])
        for state in balanced_order(states,spec['seed']) if states is not None else []:
            # The identical complete action and backward conditioning must fit;
            # reserve 512 tokens for the bounded 120-word public explanation.
            short='Explain the intended effect from the current observation.'
            fwd,fr=encode_example(processor,state['messages'],state['tools'],forward_target(state['action'],short),spec['context']-512)
            bwd,br=encode_example(processor,backward_messages(state['messages'],state['action']),state['tools'],
                                  {'role':'assistant','content':'<think>\n'+short+'\n</think>\n\nDONE'},spec['context']-512)
            if fwd is not None and bwd is not None:chosen.append(state)
            else:exclusions.append({'state_id':state['id'],'forward':fr,'backward':br})
            if len(chosen)>=spec['pilot_decisions']:break
        minimum_suites=spec.get('minimum_suites',3)
        if minimum_suites<2:raise ValueError('A multitask corpus requires at least two declared suites')
        if not chosen or len({s['suite'] for s in chosen})<minimum_suites:raise ValueError('Too few suites have context-eligible states for the declared corpus')
        if states is not None:
            write(ROOT/'pilot-membership.json',{'state_ids':[s['id'] for s in chosen],'available_decisions':len(states),
                  'max_pilot_decisions':spec['pilot_decisions'],'context_exclusions':exclusions,
                  'selection':'Frozen seed, round-robin suites/tasks, no replacement; before rationale generation or student scores.'})
        progress('teacher_backward_explanations','Generating explicit public backward explanations for the frozen training decisions.',completed=0,total=len(chosen))
        accepted={};failures=[]
        probe=rationale(chosen[0],spec)
        if probe.get('http_status',200)>=400:
            raise RuntimeError('Teacher explanation API preflight failed; no remaining requests submitted: '+str(probe.get('provider_error',probe.get('error'))))
        with ThreadPoolExecutor(max_workers=spec['teacher_concurrency']) as pool:
            futures={pool.submit(rationale,state,spec):state for state in chosen}
            for i,future in enumerate(as_completed(futures),1):
                record=future.result()
                if record['accepted']:accepted[record['state_id']]=record['rationale']
                else:failures.append(record)
                if i%8==0 or i==len(chosen):progress('teacher_backward_explanations',f'{i}/{len(chosen)} explanations returned; {len(accepted)} valid.',completed=i,total=len(chosen),accepted=len(accepted))
        rows=[];admitted_states=[];(ROOT/'tensors').mkdir(exist_ok=True)
        for state in chosen:
            if state['id'] not in accepted:continue
            r=accepted[state['id']]
            examples=[('forward',state['messages'],forward_target(state['action'],r)),
                      ('backward',backward_messages(state['messages'],state['action']),{'role':'assistant','content':'<think>\n'+r+'\n</think>\n\nDONE'})]
            encoded=[(direction,*encode_example(processor,messages,state['tools'],target,spec['context'])) for direction,messages,target in examples]
            if any(data is None for _,data,_ in encoded):
                exclusions.append({'state_id':state['id'],'reason':'Full rationale example exceeds context; excluded from both directions'});continue
            for direction,data,receipt in encoded:
                ident=state['id']+'-'+direction;path=ROOT/'tensors'/(ident+'.pt')
                if spec.get('tensor_storage')=='shared-visual-tensors-v1':
                    try:from .efe_visual_tensor_store import save
                    except ImportError:from efe_visual_tensor_store import save
                    save(data,path)
                elif spec.get('tensor_storage','inline-tensors-v1')=='inline-tensors-v1':torch.save(data,path)
                else:raise ValueError('Unknown tensor storage contract')
                rows.append({'id':ident,'state_id':state['id'],'suite':state['suite'],'task':state['task'],'episode':state['episode'],
                             'direction':direction,'tensor_file':str(path.relative_to(ROOT)),'tensor_sha256':sha(path),**receipt})
            admitted_states.append({**state,'rationale':r})
        if not rows or len({r['suite'] for r in rows})<minimum_suites:raise ValueError('Too few suites remain after explanation validation')
        with (ROOT/'warmstart-states.jsonl').open('w') as out:
            for state in admitted_states:out.write(json.dumps(state)+'\n')
        dataset={'examples':rows,'decisions':len(admitted_states),'teacher_episodes':len({r['episode'] for r in rows}),'native_episodes':0,
                 'tensor_storage':spec.get('tensor_storage','inline-tensors-v1'),
                 'scene_audit_sha256':spec['scene_audit_sha256'],'private_reasoning_exported':False,
                 'complete_targets':True,'loss_mask':'Last complete assistant target only; all prompts, observations and backward expert actions masked.',
                 'context_exclusions':exclusions,'explanation_failures':failures,'states_sha256':sha(ROOT/'warmstart-states.jsonl')}
        write(ROOT/'dataset-manifest.json',dataset)
        write(ROOT/'corpus-manifest.json',{k:v for k,v in dataset.items() if k!='examples'})
        progress('optimizer_memory_preflight','Public corpus frozen; measuring full-parameter optimizer memory and full-target likelihood on eight H100s.',decisions=len(admitted_states),teacher_episodes=dataset['teacher_episodes'])
        import efe_cuda_helpers as helpers
        helpers.ROOT=ROOT;helpers.ARCHIVE=ROOT
        env=helpers.driver_environment();env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(8)))
        env.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','/tmp/efe_train_vl.py','--spec',str(SPEC),'--root',str(ROOT)]
        run_logged('smoke',command+['--mode','smoke'],env,3600)
        smoke=json.loads((ROOT/'smoke/complete.json').read_text())
        if not smoke['success']:raise ValueError('Training smoke did not pass')
        progress('warmstart_training',f"Memory/mask checks passed; starting LR={spec['lr']} balanced warmstart from untouched weights.",training_started=True)
        if spec.get('checkpoint_durability'):
            try:from . import efe_durable_training as durable
            except ImportError:import efe_durable_training as durable
            durable.run(sys.modules[__name__],spec,env)
        else:run_logged('train',command+['--mode','train'],env,spec.get('training_timeout_seconds',8*3600))
        progress('warmstart_complete','Warmstart pilot checkpoints saved. Evaluate development before LeAct or broader recipe selection.',training_started=True)
        write(ROOT/'complete.json',{'success':True,'warmstart':json.loads((ROOT/'warmstart-ready.json').read_text()),'completed_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        progress('needs_review',str(exc)[-1600:])
        write(ROOT/'complete.json',{'success':False,'error':str(exc)[-1600:]})
        raise
    finally:
        # Summary collectors acknowledge finished artifacts; full traces remain
        # on the persistent shared volume regardless of collector availability.
        for _ in range(30):
            if (ROOT/'collected').exists():break
            time.sleep(5)


if __name__=='__main__':main()
