"""Serve the frozen Qwen VL checkpoint and evaluate native EFE development tasks."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

import efe_native_pilot as runtime


def serving_transport_options(spec):
    """Keep the reviewed interconnect workaround explicit and opt-in."""
    mode=spec.get('serving_transport',{}).get('mode','default')
    if mode=='default':return [],{}
    if mode=='nccl_no_p2p_v1':
        return ['--disable-custom-all-reduce'],{'NCCL_P2P_DISABLE':'1'}
    raise ValueError('Unknown serving transport: '+str(mode))


def serving_failure(path, returncode, token):
    data=path.read_bytes()
    return {'returncode':returncode,'log_bytes':len(data),'log_sha256':hashlib.sha256(data).hexdigest(),
            'log_tail':data[-65536:].decode(errors='replace').replace(token,'[REDACTED]'),
            'updated_utc':runtime.now()}


def response(url, body, token):
    request=urllib.request.Request(url+'/responses',data=json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    try:
        reply = urllib.request.urlopen(request,timeout=900)
    except urllib.error.HTTPError as exc:
        # These are synthetic test inputs, never benchmark observations or provider credentials.
        detail = exc.read(1024*1024).decode(errors='replace').replace(token, '[REDACTED]')
        runtime.write('student-preflight-http-error.json',{'status':exc.code,'body':detail})
        raise RuntimeError(f'Student Responses HTTP {exc.code}; full synthetic validation error saved in student-preflight-http-error.json. '+detail[:500]) from None
    with reply:
        final=None
        emitted_calls=[]
        for line in reply:
            if line.startswith(b'data: '):
                if line[6:].strip()==b'[DONE]':continue
                event=json.loads(line[6:])
                if event.get('type')=='response.output_item.done' and event.get('item',{}).get('type')=='function_call':
                    emitted_calls.append(event['item'])
                if event.get('type')=='response.completed':final=event['response']
                if event.get('type') in {'error','response.failed','response.incomplete'}:
                    error = event.get('error') or event.get('response', {}).get('error')
                    raise RuntimeError('Student Responses preflight did not complete: '+event['type']+' '+str(error))
        if final is None:raise RuntimeError('Student did not emit response.completed')
        calls=[x for x in final.get('output',[]) if x.get('type')=='function_call']
        if emitted_calls!=calls:
            raise RuntimeError('Student stream did not deliver the final tool calls as executable output_item.done events')
        return final


def preflight(spec, token):
    """Exercise the same HTTP gateway and tool events used by native Codex."""
    command=[sys.executable,spec.get('gateway_script','/tmp/efe-budget-gateway.py'),'--root',str(runtime.ARCHIVE/'preflight-requests'),
             '--port','18862','--model',spec['request_model'],'--max-output-tokens',str(spec['max_output_tokens']),
             '--max-images',str(spec.get('max_images',40)),'--upstream',spec['upstream'],
             '--flatten-namespaces','--terminal-output-events']
    log=(runtime.ARCHIVE/'logs/preflight-gateway.log').open('w')
    gateway=subprocess.Popen(command,env={**os.environ,'NVIDIA_INFERENCE_API_KEY':token,'EFE_GATEWAY_TOKEN':token},
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(30):
            if gateway.poll() is not None:raise RuntimeError('Synthetic preflight gateway exited')
            try:
                with socket.create_connection(('127.0.0.1',18862),timeout=1):break
            except OSError:time.sleep(1)
        else:raise RuntimeError('Synthetic preflight gateway did not become ready')
        return vision_tool_preflight({**spec,'upstream':'http://127.0.0.1:18862/v1'},token)
    finally:
        gateway.terminate()
        try:gateway.wait(timeout=10)
        except subprocess.TimeoutExpired:gateway.kill();gateway.wait()
        log.close()


def vision_tool_preflight(spec, token):
    from PIL import Image
    semantics_policy=spec.get('preflight_semantics_policy','stop')
    if semantics_policy not in {'stop','record_and_evaluate'}:
        raise ValueError('Unknown synthetic preflight semantics policy')
    semantic_checks=[]
    def check_answer(name,passed,error):
        semantic_checks.append({'name':name,'passed':passed,'error':None if passed else error})
        if not passed and semantics_policy=='stop':raise RuntimeError(error)
    image=Image.new('RGB',(96,96),(0,230,0));buf=io.BytesIO();image.save(buf,format='PNG')
    inputs=[{'role':'user','content':[{'type':'input_text','text':'Call report_color with the dominant color of this image. Use exactly one tool call.'},
             {'type':'input_image','detail':'auto','image_url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}]}]
    tools=[{'type':'namespace','name':'probe','description':'Synthetic vision and tool transport check.', 'tools':[
           {'type':'function','name':'report_color','description':'Report the dominant image color.',
            'parameters':{'type':'object','properties':{'color':{'type':'string','enum':['red','green','blue']}},
                          'required':['color'],'additionalProperties':False},'strict':True}]}]
    body={'model':spec['request_model'],'input':inputs,'tools':tools,'stream':True,'store':False,
          'max_output_tokens':4096,'temperature':1.0,'top_p':0.95}
    first=response(spec['upstream'],body,token)
    calls=[x for x in first.get('output',[]) if x['type']=='function_call']
    if len(calls)!=1 or calls[0]['name']!='report_color' or calls[0].get('namespace')!='probe':
        raise RuntimeError('Student vision/function-call preflight failed; do not run benchmark tasks')
    call=calls[0]
    arguments=json.loads(call['arguments'])
    if not isinstance(arguments,dict) or set(arguments)!={'color'} or arguments['color'] not in ('red','green','blue'):
        raise RuntimeError('Student preflight returned invalid function arguments')
    check_answer('initial_image_color',arguments['color']=='green','Student initial image-color preflight failed')
    # Codex retains assistant output_text but strips completion metadata before
    # resubmitting history. Test that exact shape, not only the server's own objects.
    history_message={'type':'message','id':'msg_synthetic_history','role':'assistant',
                     'content':[{'type':'output_text','text':'I will inspect the tool result.'}]}
    history_call={k:v for k,v in call.items() if k not in {'status'}}
    continuation={**body,'input':inputs+[history_message,history_call,{'type':'function_call_output','call_id':call['call_id'],
                    'output':'The measured dominant color is green.'},
                    {'role':'user','content':'State the color reported by the tool in one sentence. Do not call another tool.'}],
                  'tool_choice':'none'}
    second=response(spec['upstream'],continuation,token)
    output=' '.join(c.get('text','') for x in second.get('output',[]) if x['type']=='message' for c in x.get('content',[]))
    check_answer('tool_text_color','green' in output.rsplit('</think>',1)[-1].lower(),
                 'Student tool-result continuation preflight failed')
    image=Image.new('RGB',(96,96),(0,0,230));buf=io.BytesIO();image.save(buf,format='PNG')
    image_result={**body,'tool_choice':'none','input':inputs+[history_message,history_call,
                  {'type':'function_call_output','call_id':call['call_id'],'output':[
                      {'type':'input_text','text':'The tool returned a NEW image.'},
                      {'type':'input_image','detail':'auto','image_url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}]},
                  {'role':'user','content':'State the dominant color of the NEW image returned by the tool in one sentence. Do not call another tool.'}]}
    third=response(spec['upstream'],image_result,token)
    image_output=' '.join(c.get('text','') for x in third.get('output',[]) if x['type']=='message' for c in x.get('content',[]))
    check_answer('new_tool_image_color','blue' in image_output.rsplit('</think>',1)[-1].lower(),
                 'Student image inside tool-result preflight failed')
    return {'status':'passed' if all(c['passed'] for c in semantic_checks) else 'semantic_failure',
            'transport_status':'passed','semantics_policy':semantics_policy,'semantic_checks':semantic_checks,
            'transport_scope':'HTTP completion and native function-call event roundtrip; not proof of internal image processing correctness.',
            'benchmark_execution_allowed':True,
            'image_color':'green','function_call':{'name':call['name'],'namespace':call['namespace'],'arguments':call['arguments']},
            'continuation':output,'returned_models':[first.get('model'),second.get('model'),third.get('model')],
            'tool_result_image_color':'blue','tool_result_image_continuation':image_output,
            'usage':[first.get('usage'),second.get('usage'),third.get('usage')],'updated_utc':runtime.now()}


def main():
    spec=json.loads(runtime.SPEC.read_text())
    runtime.ARCHIVE=Path(spec['archive_root'])/runtime.WORKFLOW
    runtime.ROOT.mkdir(parents=True,exist_ok=True);runtime.ARCHIVE.mkdir(parents=True,exist_ok=True)
    (runtime.ARCHIVE/'logs').mkdir(exist_ok=True)
    model=Path(spec['student_checkpoint'])
    manifest=json.loads(Path(spec.get('student_manifest', model.parent/'checkpoint-manifest.json')).read_text())
    if manifest['revision']!=spec['student_revision']:raise ValueError('Wrong frozen Qwen revision')
    for name, row in manifest['files'].items():
        if (model/name).stat().st_size!=row['bytes']:raise ValueError('Checkpoint file size mismatch: '+name)
        digest=hashlib.sha256()
        with (model/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=row['sha256']:raise ValueError('Checkpoint checksum mismatch: '+name)
    token=secrets.token_urlsafe(36)
    os.environ['NVIDIA_INFERENCE_API_KEY']=token # local vLLM token, never the provider credential
    server_python=os.environ['EFE_SERVING_PYTHON']
    command=[server_python,'-m','vllm.entrypoints.openai.api_server','--model',str(model),
             '--served-model-name',spec['request_model'],'--host','127.0.0.1','--port','18880',
             '--tensor-parallel-size','4','--max-model-len','131072','--max-num-seqs','4',
             '--max-num-batched-tokens','8192','--gpu-memory-utilization','0.90','--dtype','bfloat16',
             '--enable-auto-tool-choice','--tool-call-parser','hermes','--reasoning-parser','deepseek_r1',
             '--limit-mm-per-prompt',json.dumps({'image':64,'video':0}),
             '--mm-processor-kwargs',json.dumps({'max_pixels':1048576}),
             '--generation-config','auto']
    transport_args,transport_env=serving_transport_options(spec)
    command.extend(transport_args)
    safe_command=[('[LOCAL_TOKEN]' if x==token else x) for x in command]
    runtime.write('student-serving.json',{'command':safe_command,'revision':spec['student_revision'],
                  'model':spec['request_model'],'baseline_evaluation_started':False,'training_started':False,
                  'serving_transport':spec.get('serving_transport',{'mode':'default'}),
                  'transport_environment':transport_env,
                  'vision_processing':'Legacy max_pixels=1048576 request retained identically across the native base/warmstart/LeAct comparison. A later audit found this override does not enforce the HF training image size in this vLLM version; do not interpret it as a verified pixel cap. Raw observations retained.',
                  'reasoning_parser':'deepseek_r1 uses the same think delimiters and accepts the opening tag already supplied by Qwen3-VL Thinking chat template; qwen3 in vLLM 0.16 incorrectly requires the opening tag in generated text.',
                  'updated_utc':runtime.now()})
    runtime.progress('qwen_server_starting','Starting Qwen3-VL-8B-Thinking on four H100 GPUs; native development evaluation follows the image/tool preflight.')
    log=(runtime.ARCHIVE/'logs/student-vllm.log').open('w')
    server=subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3','VLLM_PORT':'29880',
                                         'OMP_NUM_THREADS':'2','VLLM_ENABLE_RESPONSES_API_STORE':'1','NCCL_DEBUG':'INFO',
                                         **transport_env},
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError('Qwen serving exited; inspect logs/student-vllm.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:18880/health',timeout=5) as reply:
                    if reply.status==200:break
            except Exception:time.sleep(5)
        else:raise TimeoutError('Qwen server exceeded its 20-minute startup bound')
        runtime.progress('qwen_image_tool_preflight','Qwen server ready; testing streamed image recognition and tool-result continuation.')
        runtime.write('student-preflight.json',preflight(spec,token))
        module_spec=importlib.util.spec_from_file_location('efe_agent_pilot','/tmp/efe-astra-pilot.py')
        pilot=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(pilot)
        pilot.main()
    except Exception as exc:
        if server.poll() is not None:
            failure=serving_failure(runtime.ARCHIVE/'logs/student-vllm.log',server.returncode,token)
            runtime.write('student-serving-failure.json',failure)
            print(json.dumps({'student_serving_failure':failure}),flush=True)
        runtime.progress('qwen_baseline_needs_review',str(exc))
        runtime.write('student-preflight.json',{'status':'failed','error':str(exc),'updated_utc':runtime.now()})
        # A bounded diagnostic window keeps the loaded server available to inspect the
        # exact failure. A reviewed replacement script can run without another model load.
        for _ in range(int(spec.get('preflight_debug_seconds', 0)) // 5):
            if (runtime.ROOT/'end-preflight-debug').exists():break
            time.sleep(5)
        runtime.write('complete.json',{'completed_utc':runtime.now(),'error':str(exc),'completed_trials':0,
                                       'planned_trials':len(spec['tasks'])})
        for _ in range(60):
            if (runtime.ROOT/'collected').exists():break
            time.sleep(5)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        log.close()


if __name__=='__main__':main()
