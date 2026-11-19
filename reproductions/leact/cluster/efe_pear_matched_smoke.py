"""Replay previously successful training cases against a running rollout API.

This is diagnostic inference only. Results never enter the learner's queue.
Native simulators and their official success verifiers remain authoritative.
"""
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.request

from efe_action_loop import SYSTEM,parse_action
from efe_pear_file_rpc import FileEpisodeClient,atomic_json
from efe_pear_action_json import final_action_json


def decode_action(text,contract):
    return parse_action(final_action_json(text),contract=contract,allow_stop=True)


def public_content(blocks):
    content=[]
    for block in blocks:
        if block['type']=='text':content.append({'type':'text','text':block['text']})
        elif block['type']=='image':
            data=io.BytesIO();block['image'].save(data,format='PNG')
            content.append({'type':'image_url','image_url':{'url':
                'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()}})
        else:raise ValueError('Unexpected public observation block')
    return content


def run_case(spec,case,index):
    root=Path(spec['output']);name=case['task'];folder=root/name;folder.mkdir(exist_ok=True)
    api=spec['rollout_apis'][index%len(spec['rollout_apis'])]
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with http.open(api+'/v1/models',timeout=15)as response:models=json.load(response)
    if not any(m['id']==spec['model']for m in models['data']):
        raise ValueError('The rollout API serves a different checkpoint')
    client=FileEpisodeClient(spec['transport_root'],'efe/'+case['suite'],
        {'task':name,'seed':case['seed']},timeout=1800)
    result={'task':name,'status':'opening','model':spec['model'],'model_revision':spec['model_revision'],
        'episode_id':client.uid,'primary_rl_updates':0,'generated_tokens':0,'valid_actions':0,
        'invalid_actions':0,'official_success':None,'response_budget':160000,'max_calls':96,
        'action_decoder':'production final_action_json followed by native parse_action',
        'inference_path':'Existing PEAR vLLM HTTP API; no new model allocation.',
        'scope':'HTTP case replay using the current tool prompt and parser; separate from primary training metrics. Exact training-sampler tokenization parity is not asserted.'}
    def save():
        result['updated_unix']=time.time();atomic_json(folder/'result.json',result)
        print('PEAR_MATCHED_SMOKE '+json.dumps(result),flush=True)
    save()
    try:
        opened=client.open();contract=opened['action_contract'];value=opened
        started=time.monotonic();result.update(status='running',opened_unix=time.time());save()
        prompt=SYSTEM.replace('Return only one JSON object with action_type',
            'End your response with one json code block containing action_type')+(
            '\nGround your reasoning in the action and observation guide supplied with each observation, '
            'including its coordinate frames, units and named sensor layout.')
        messages=[{'role':'system','content':prompt}]
        for turn in range(96):
            if value['done']:break
            if time.monotonic()-started>=1500 or result['generated_tokens']>=160000:
                result['end_reason']='diagnostic_budget'
                value=client.step({'stop':True,'summary':'Finish the diagnostic at its declared budget.'});break
            messages.append({'role':'user','content':public_content(value['public_blocks'])})
            body={'model':spec['model'],'messages':messages,'temperature':1.,
                'max_tokens':min(32768,160000-result['generated_tokens'])}
            if spec.get('chat_template_kwargs'):body['chat_template_kwargs']=spec['chat_template_kwargs']
            request=urllib.request.Request(api+'/v1/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            before=time.monotonic()
            with http.open(request,timeout=600)as response:reply=json.load(response)
            message=reply['choices'][0]['message'];text=message.get('content')or''
            result['generated_tokens']+=reply['usage']['completion_tokens']
            with(folder/'turns.jsonl').open('a')as log:
                log.write(json.dumps({'turn':turn,'text':text,'finish_reason':reply['choices'][0]['finish_reason'],
                    'usage':reply['usage'],'model_seconds':time.monotonic()-before})+'\n')
            messages.append({'role':'assistant','content':text})
            try:action=decode_action(text,contract)
            except(ValueError,TypeError,KeyError)as exc:
                result['invalid_actions']+=1;result['end_reason']='action_validation'
                result['action_error']=str(exc)[:500]
                value=client.step({'stop':True,'summary':'Finish the diagnostic after an invalid policy command.'});break
            result['valid_actions']+=1;value=client.step(action);save()
        if not value['done']:
            value=client.step({'stop':True,'summary':'Finish the diagnostic at the declared action count.'})
        if not value['done']:raise ValueError('Native simulator has no finalized diagnostic outcome')
        result.update(status='completed',official_success=value['rewards'][-1],
            native_trial_path=value.get('native_trial_path'),native_termination_terms=value.get('native_termination_terms'))
    except Exception as exc:
        result.update(status='infrastructure_or_runtime_failure',error_type=type(exc).__name__,
            error=str(exc)[:700],official_success=None)
    finally:client.close();save()
    return result


def main():
    spec=json.loads(Path(__file__).with_name('spec.json').read_text())
    if not spec.get('qualification_only')or spec['response_budget']!=160000:
        raise ValueError('Only the declared inference diagnostic is supported')
    root=Path(spec['output']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    from concurrent.futures import as_completed
    results=[]
    with ThreadPoolExecutor(max_workers=4)as pool:
        jobs={pool.submit(run_case,spec,row,index):row for index,row in enumerate(spec['cases'])}
        for future in as_completed(jobs):
            try:result=future.result()
            except Exception as exc:
                result={'task':jobs[future]['task'],'status':'infrastructure_or_runtime_failure',
                    'error_type':type(exc).__name__,'error':str(exc)[:700],'official_success':None}
            results.append(result);atomic_json(root/'summary.json',{'cases':results,
                'primary_rl_updates':0,'complete':len(results)==len(spec['cases'])})


if __name__=='__main__':main()
