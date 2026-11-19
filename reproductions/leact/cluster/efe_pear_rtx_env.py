"""Constrained policy episodes using pinned native RTX SDKs and live graders.

The native service and collect script are unchanged. A scheduling-only entry
point runs finalization on the simulator owner thread. This training transport
is deliberately separate from Harbor's resource-enforced benchmark scores.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from efe_action_loop import native_contract, parse_action
from efe_pear_context import VERSION
from efe_pear_public_observation import public_blocks


class NativeRtxEpisode:
    def __init__(self, *, task, seed=0, config=None):
        config=config or json.loads(Path(os.environ['EFE_PEAR_RTX_CONFIG']).read_text())
        self.config=config;item=config['training_tasks'][task]
        checking=config.get('qualification_only')is True
        if checking and os.environ.get('PEAR_PRIMARY_TRAINING')=='1':
            raise ValueError('A simulator qualification manifest cannot run primary training')
        if not checking and item['qualification']!='passed':
            raise ValueError('RTX task has not passed native reset/action/camera/verdict checks')
        self.row=item['row'];self.scene_sha=item['scene_sha256']
        if seed!=self.row['seeds']['default']:
            raise ValueError('Native RTX training preserves its frozen task seed')
        self.source=Path(config['source']);self.root=Path(config['episode_root'])/uuid4().hex
        self.root.mkdir(mode=0o700,parents=True);self.srv=self.root/'srv';self.srv.mkdir()
        self.data=self.root/'data';self.data.mkdir();self.process=None;self.closed=False
        self.started=time.monotonic();self.last_action=None;self.turns=0;self.previous_steps=0

    def open(self):
        from rcb_client import Robot
        import yaml
        suite=self.row['suite'];tool_relative=self.config['native_tool_relative']
        tool_name=Path(tool_relative).name
        for name in ['server.py','live_video.py',tool_name,'finalize.sh']:
            original=self.source/suite/('sim/'+name if name!=tool_name else tool_relative)
            shutil.copyfile(original,self.srv/name)
        for name in ['rcb1.py','rcb_limited.py']:
            shutil.copyfile(self.source/'standard'/name,self.srv/name)
        task=self.source/self.row['path'];instance=task/'environment/sim/instance.json'
        native=yaml.safe_load((task/'environment/docker-compose.yaml').read_text())['services']['sim']['environment']
        self.env={**os.environ,**{k:str(v)for k,v in native.items()},
            'PYTHONPATH':str(self.srv)+':'+os.environ.get('PYTHONPATH',''),
            'RCB_INSTANCE':str(instance),'RCB_DATA':str(self.data),
            'RCB_TOOL':str(self.srv/tool_name),'RCB_SRV':str(self.srv),
            'RCB_SERVER':str(self.srv/'server.py'),'SIM_URL':'http://127.0.0.1:8000',
            'NVIDIA_DRIVER_CAPABILITIES':'all','OMNI_KIT_ALLOW_ROOT':'1'}
        if self.config.get('control_interface'):
            from efe_pear_eef import enabled
            enabled(self.config['control_interface'])
            self.env['PEAR_NATIVE_CONTROL_INTERFACE']=self.config['control_interface']
        entry=Path(__file__).with_name('efe_pear_native_entry.py')
        with(self.root/'native.log').open('w')as log:
            self.process=subprocess.Popen([sys.executable,'-u',str(entry),str(self.srv/'server.py')],
                env=self.env,cwd=self.srv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        threading.Thread(target=self.process.wait,daemon=True).start()
        self.robot=Robot(base='http://127.0.0.1:8000',timeout=1000)
        self.spec=self.robot.spec()
        if self.config.get('control_interface'):
            from efe_pear_eef import contract,public_spec
            self.contract=contract(self.spec,suite)
            self.spec=public_spec(self.spec,self.contract)
        else:
            self.contract=native_contract(self.spec,allowed_action_types=['joint_position'])
        self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        self.instruction=(task/'instruction.md').read_text()
        self.observation=self.robot.get_obs(cameras=self.contract['cameras'],include_depth=False)
        if self.observation.done:
            raise RuntimeError('RTX training reset is already terminal')
        return self._result()

    def _result(self, *, done=False, success=False, executed=0):
        from PIL import Image
        blocks=None;images=None;context_sha=None;state=None
        if not done:
            images=[Image.fromarray(self.observation.last[c]).convert('RGB')for c in self.contract['cameras']]
            visible={'mode':'task','objective':self.instruction,'spec':self.spec,
                'action_contract':self.contract,'observation':self.observation.meta,
                'recent_actions_and_feedback':[]if self.last_action is None else[{
                    'action':self.last_action,'execution_completed':True,
                    'control_steps_before':self.previous_steps,
                    'control_steps_after':self.observation.meta.get('control_steps'),
                    **({'eef_execution':self.observation.meta['eef_execution']}
                       if 'eef_execution'in self.observation.meta else{})}]}
            blocks,context_sha=public_blocks(visible,images);state=visible['observation']
        actions=[]if self.last_action is None or self.last_action.get('stop')else[self.last_action]
        return {'images':images,'state':state,'instruction':self.instruction,'actions':actions,
            'rewards':[float(success)],'done':bool(done),'executed':len(actions),
            'outcome':'done'if done else'observed','public_blocks':blocks,
            'action_contract':self.contract,'action_contract_sha256':self.contract_sha,
            'scene_sha256':self.scene_sha,'interface_context_version':VERSION,
            'interface_context_sha256':context_sha,'native_trial_path':str(self.data/'final.json'),
            'requested_control_steps':0 if self.last_action is None else self.last_action.get('repeat',0),
            'executed_control_steps':executed,'grading_scope':'unchanged_native_live_training_verdict',
            'benchmark_resource_enforced':False,
            **({'control_interface':self.contract['control_interface'],
                'eef_execution':self.observation.meta.get('eef_execution')}
               if self.contract.get('control_interface')else{})}

    def _finalize(self, executed=0):
        with(self.root/'finalize.log').open('w')as log:
            process=subprocess.run(['sh',str(self.srv/'finalize.sh')],env=self.env,cwd=self.srv,
                stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        verdict=json.loads((self.data/'final.json').read_text())
        if (process.returncode or verdict.get('finalized')is not True or
            type(verdict.get('success'))is not bool or any(verdict.get(k)for k in
                ['infra_error','instance_mismatch','sim_error','no_verdict','settle_error'])):
            raise RuntimeError('RTX simulator lacks a trustworthy finalized native verdict')
        if verdict.get('graded_by')!='live' or verdict.get('replays')!=0:
            raise ValueError('RTX result was not graded by the actual live episode')
        return self._result(done=True,success=verdict['success'],executed=executed)

    def step(self, action):
        parse_action(json.dumps(action),contract=self.contract,allow_stop=True)
        self.last_action=action;self.turns+=1
        if action.get('stop'):
            return self._finalize()
        self.previous_steps=int(self.observation.meta['control_steps'])
        vector=action['action']
        if self.contract.get('control_interface'):
            from efe_pear_eef import native_pose
            vector=native_pose(vector,self.contract['action_types'][action['action_type']])
        self.observation=self.robot.submit(action['action_type'],[vector]*action['repeat'],
            dt=self.contract['control_dt'],cameras=self.contract['cameras'],video='frames',
            include_depth=False,note='PEAR constrained policy action')
        executed=int(self.observation.meta['control_steps'])-self.previous_steps
        if not 0<=executed<=action['repeat']:
            raise ValueError('Native control-step accounting differs from the policy command')
        if self.contract.get('control_interface'):
            receipt=self.observation.meta.get('eef_execution',{})
            if(receipt.get('requested_control_steps')!=action['repeat']or
                receipt.get('executed_control_steps')!=executed):
                raise ValueError('Native EEF controller did not return its exact motion budget receipt')
        if (self.observation.done or self.turns>=self.config.get('max_task_calls',180)or
                time.monotonic()-self.started>=self.config.get('wall_seconds',3600)):
            return self._finalize(executed)
        return self._result(executed=executed)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.process is not None and self.process.poll()is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if self.process is not None and (self.data/'final.json').is_file():
            from efe_pear_episode_archive import completed_episode
            completed_episode(self.root,self.row['suite']+'_'+self.root.name)
