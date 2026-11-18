"""Drive the existing native Harbor action agent from PEAR's external policy loop.

The HTTP endpoint carries observations and actions only. It does not perform
inference, choose actions, reshape rewards, or replace the native verifier.
"""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import socket
import subprocess
import threading
import time


class NativeInfrastructureError(RuntimeError):
    """No trustworthy native episode grade; never convert this into reward zero."""


def native_reward(trial, returncode):
    rewards=(trial.get('verifier_result') or {}).get('rewards') or {}
    exc=trial.get('exception_info') or {}
    deadline=(exc.get('exception_type')=='AgentTimeoutError'
        and rewards.get('finalized')==1 and rewards.get('success')==0
        and not any(rewards.get(k) for k in ['no_verdict','missing_log','sim_error','sim_crashed_on_call']))
    if (returncode!=0 or (exc and not deadline) or rewards.get('success') not in (0,1)
        or rewards.get('resource_enforced')!=1 or any(rewards.get(k) for k in
        ['infra_error','grader_error','instance_mismatch','resource_monitor_missing','resource_switch_error'])):
        raise NativeInfrastructureError('Native verifier did not produce a valid resource-enforced outcome')
    return float(rewards['success'])


def observation_payload(body):
    """Accept only the existing action bridge's single public observation turn."""
    messages=body.get('input',[])
    if len(messages)!=1 or messages[0].get('role')!='user':
        raise ValueError('Unexpected native bridge conversation shape')
    content=messages[0]['content']
    if not content or content[0].get('type')!='input_text':
        raise ValueError('Native observation metadata is missing')
    visible=json.loads(content[0]['text'])
    if visible.get('mode')!='task' or not visible.get('action_contract'):
        raise ValueError('PEAR requires native task-mode action contracts')
    if any(part.get('type') not in {'input_text','input_image'} for part in content):
        raise ValueError('Only public text and RGB observations are allowed')
    return visible,content


def image_blocks(content):
    from PIL import Image
    try:from .efe_pear_context import interface_context
    except ImportError:from efe_pear_context import interface_context
    blocks=[]
    for part in content:
        if part['type']=='input_text':blocks.append({'type':'text','text':part['text']})
        else:
            url=part['image_url']
            if not url.startswith('data:image/png;base64,'):
                raise ValueError('Only inline native PNG frames are allowed')
            with Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))) as im:
                blocks.append({'type':'image','image':im.convert('RGB').copy()})
    visible=json.loads(content[0]['text'])
    images=[block['image']for block in blocks if block['type']=='image']
    labels=[part['text'].removeprefix('Current camera: ')for part in content
            if part.get('type')=='input_text'and part['text'].startswith('Current camera: ')]
    if labels!=visible['action_contract'].get('cameras'):
        raise ValueError('RGB sensor labels differ from the action contract camera order')
    guide=interface_context(visible,image_sizes=[image.size for image in images])
    return [{'type':'text','text':guide['text']},*blocks]


class HarborEpisode:
    def __init__(self, row, config, episode_id):
        self.row,self.config,self.episode_id=row,config,episode_id
        self.root=Path(config['episode_root'])/episode_id
        self.root.mkdir(parents=True,exist_ok=False)
        self.requests=queue.Queue(maxsize=1);self.actions=queue.Queue(maxsize=1)
        self.token=secrets.token_urlsafe(32)
        self.server=None;self.process=None;self.log=None
        self.current=None;self.calls=0;self.closed=False
        self.deadline=time.monotonic()+int(row['cutoff_seconds'])+2400

    def start(self):
        bridge=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                if self.path!='/v1/responses' or self.headers.get('Authorization')!='Bearer '+bridge.token:
                    self.send_error(403);return
                try:
                    size=int(self.headers['Content-Length'])
                    if not 0<size<=32*1024*1024:raise ValueError('Invalid request size')
                    body=json.loads(self.rfile.read(size));observation_payload(body)
                    bridge.calls+=1
                    if bridge.config.get('retain_transport_observations',True):
                        (bridge.root/f'observation-{bridge.calls:04d}.json').write_text(json.dumps(body)+'\n')
                    bridge.requests.put(body,timeout=5)
                    action=bridge.actions.get(timeout=900)
                    response={'type':'response.completed','response':{
                        'id':f'{bridge.episode_id}-{bridge.calls}','object':'response','status':'completed',
                        'model':'external-pear-policy','output':[{'type':'message','role':'assistant',
                            'content':[{'type':'output_text','text':json.dumps(action,allow_nan=False)}]}],
                        'usage':{},'metadata':{'token_accounting':'upstream PEAR sampler, not this transport'}}}
                    data=('data: '+json.dumps(response)+'\n\ndata: [DONE]\n\n').encode()
                    self.send_response(200);self.send_header('Content-Type','text/event-stream')
                    self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                except Exception as exc:
                    bridge.requests.put_nowait(exc) if bridge.requests.empty() else None
                    self.send_error(500,'Native bridge transport failed')
        address=self.config.get('advertise_address') or socket.gethostbyname(socket.gethostname())
        self.server=ThreadingHTTPServer((address,0),Handler)
        self.server.daemon_threads=True
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        agent_dir=self.root/'agent_source';agent_dir.mkdir()
        source=Path(self.config['agent_source'])
        for name in ['efe_action_agent.py','efe_action_loop.py']:
            shutil.copyfile(source/name,agent_dir/name)
        (agent_dir/'action-config.json').write_text(json.dumps(self.config['action_config'])+'\n')
        env={**os.environ,**self.config.get('container_environment',{}),
            'OPENAI_API_KEY':self.token,'OPENAI_BASE_URL':f'http://{address}:{self.server.server_port}/v1',
            'RCB_LIMITS_ENFORCE':'1','PYTHONPATH':str(agent_dir)+':'+os.environ.get('PYTHONPATH','')}
        command=[self.config['harbor'],'run','-d','embodied-first-exam@all','-p',self.row['path'],
            '-a','efe_action_agent:FixedActionAgent','-m','external-pear-policy','-e','podman','-y',
            '-k','1','-n','1','--max-retries','0','--job-name',self.episode_id,
            '-o',str(self.root/'jobs'),'--allow-agent-host',address]
        self.log=(self.root/'harbor.log').open('w')
        self.process=subprocess.Popen(command,cwd=self.config['reference'],env=env,
            stdin=subprocess.DEVNULL,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        return self.next()

    def next(self):
        while time.monotonic()<self.deadline:
            try:body=self.requests.get(timeout=.2)
            except queue.Empty:
                if self.process.poll() is not None:return self.finish()
                continue
            if isinstance(body,Exception):raise NativeInfrastructureError(str(body)) from body
            self.current=body
            return {'done':False,'body':body,'reward':0.}
        raise NativeInfrastructureError('Native Harbor episode exceeded its infrastructure deadline')

    def step(self,action):
        if self.current is None:raise RuntimeError('No observation awaiting a policy action')
        self.current=None
        self.actions.put(action,timeout=5)
        return self.next()

    def finish(self):
        matches=list((self.root/'jobs'/self.episode_id).glob('*/result.json'))
        if len(matches)!=1:raise NativeInfrastructureError('Missing or ambiguous native trial result')
        trial=json.loads(matches[0].read_text())
        if trial.get('task_name') not in {self.row['task'],'embodied-first-exam/'+self.row['task']}:
            raise NativeInfrastructureError('Native trial task differs from the requested task')
        if not self.calls:raise NativeInfrastructureError('Native trial never produced a public observation')
        return {'done':True,'body':None,'reward':native_reward(trial,self.process.returncode),
                'native_trial_path':str(matches[0])}

    def close(self):
        if self.closed:return
        self.closed=True
        if self.current is not None and self.actions.empty():
            self.actions.put({'stop':True,'summary':'Policy episode ended.'})
        if self.process is not None and self.process.poll() is None:
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGTERM)
                try:self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if self.server is not None:self.server.shutdown();self.server.server_close()
        if self.log is not None:self.log.close()
