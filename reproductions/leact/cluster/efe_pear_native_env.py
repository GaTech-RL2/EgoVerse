"""PEAR stateful simulator and tool using the existing EFE constrained interface."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from pear.agent_loop.protocol import InvalidAction
from pear.envs.tool import EnvironmentTool
from pear.sim.env import EpisodeLost, StatefulEnv
from pear.sim.process import SimulatorProcess

from efe_action_loop import SYSTEM, parse_action
from efe_pear_bridge import HarborEpisode, NativeInfrastructureError, image_blocks, observation_payload
from efe_pear_context import VERSION as INTERFACE_CONTEXT_VERSION


class NativeEfeEnv(StatefulEnv):
    SERVER_RESOURCES={'num_cpus':1,'resources':{'efe':1}}
    START_TIMEOUT_S=1200.
    OPEN_TIMEOUT_S=1800.
    STEP_TIMEOUT_S=1800.
    SCHEMA={}
    CAMERAS=()
    ENVIRONMENT_PROMPT=''
    LOG_KEYS=('native_trial_path','action_contract_sha256','requested_control_steps','executed_control_steps',
              'interface_context_version','interface_context_sha256')

    @classmethod
    def setup_env(cls,context):
        process=SimulatorProcess('efe',cls,sys.executable,[],dict(os.environ),
                                 Path(__file__).resolve().parent,context.log_dir)
        return [process],1

    @classmethod
    def task_name(cls,env_args):return env_args['task']

    def __init__(self,task,seed=0):
        self.config=json.loads(Path(os.environ['EFE_PEAR_NATIVE_CONFIG']).read_text())
        membership=json.loads(Path(self.config['membership_file']).read_text())
        item=membership['training_tasks'][task]
        if not self.config.get('qualification_only') and item.get('qualification')!='passed':
            raise ValueError('Task has not passed native environment qualification')
        if self.config.get('qualification_only') and os.environ.get('PEAR_PRIMARY_TRAINING')=='1':
            raise ValueError('An environment smoke-test manifest cannot be used for primary RL')
        self.row=item['row'];self.expected_scene=item['scene_sha256']
        if seed!=self.row['seeds']['default']:
            raise ValueError('This native adapter preserves the frozen task reset; arbitrary seeds are unsupported')
        reference=Path(self.config['reference'])
        metadata=reference/self.row['path']/'task.toml'
        if hashlib.sha256(metadata.read_bytes()).hexdigest()!=self.row['task_toml_sha256']:
            raise ValueError('Task metadata changed')
        from scene_identity import scene_identity
        if scene_identity(reference,self.row)['scene_sha256']!=self.expected_scene:
            raise ValueError('Frozen scene identity changed')
        self.bridge=HarborEpisode(self.row,self.config,uuid4().hex)
        self.contract=None;self.contract_sha=None;self.last_action=None;self.instruction=''

    def _result(self,value):
        images=None;blocks=None;state=None;executed_controls=None;context_sha=None
        if not value['done']:
            visible,content=observation_payload(value['body'])
            self.contract=visible['action_contract'];self.instruction=visible['objective']
            self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
            blocks=image_blocks(content);images=[b['image'] for b in blocks if b['type']=='image']
            context_sha=hashlib.sha256(blocks[0]['text'].encode()).hexdigest()
            state={'public_observation':visible['observation']}
            feedback=visible.get('recent_actions_and_feedback',[])
            if self.last_action and feedback:
                before=feedback[-1].get('control_steps_before');after=feedback[-1].get('control_steps_after')
                if type(before) is int and type(after) is int:executed_controls=after-before
        action=self.last_action
        # A native EFE action is a typed chunk. Do not count requested repeats
        # as executed physics steps: the simulator may stop a chunk early.
        actions=[] if action is None or action.get('stop') else [action]
        return {'images':images,'state':state,'actions':actions,'rewards':[value['reward']],
            'done':value['done'],'executed':len(actions),'outcome':'done' if value['done'] else 'observed',
            'instruction':self.instruction,'public_blocks':blocks,'action_contract':self.contract,
            'action_contract_sha256':self.contract_sha,'native_trial_path':value.get('native_trial_path'),
            'interface_context_version':INTERFACE_CONTEXT_VERSION,'interface_context_sha256':context_sha,
            'requested_control_steps':action.get('repeat',0) if action else 0,
            'executed_control_steps':executed_controls}

    def open(self):
        try:return self._result(self.bridge.start())
        except NativeInfrastructureError as exc:raise EpisodeLost(str(exc)) from exc

    def step(self,action):
        # The exact same parser runs inside the native container before execution.
        parse_action(json.dumps(action),contract=self.contract,allow_stop=True)
        self.last_action=action
        try:return self._result(self.bridge.step(action))
        except NativeInfrastructureError as exc:raise EpisodeLost(str(exc)) from exc

    def close(self):self.bridge.close()


class NativeEfeTool(EnvironmentTool):
    def __init__(self,*args,**kwargs):
        self.contracts={}
        super().__init__(*args,**kwargs)

    async def open(self,episode):
        content,result=await super().open(episode)
        self.contracts[episode.uid]=result['action_contract']
        return content,result

    def system_prompt(self,result):
        # The action schema and controller are unchanged; PEAR stores the model's
        # reasoning separately and its parser expects the final fenced JSON.
        return SYSTEM.replace('Return only one JSON object with action_type',
            'End your response with one json code block containing action_type')+(
            '\nGround your reasoning in the action and observation guide supplied with each observation, '
            'including its coordinate frames, units and named sensor layout.')

    def _observation(self,result):return result['public_blocks']

    async def step(self,episode,action):
        try:parse_action(json.dumps(action),contract=self.contracts[episode.uid],allow_stop=True)
        except (ValueError,TypeError,KeyError) as exc:raise InvalidAction(str(exc)) from exc
        before,start=len(episode.actions),time.monotonic()
        result=await self.client.step(episode.uid,action)
        self._log_call(episode,before,action,result,time.monotonic()-start)
        return self._observation(result),result

    async def close(self,episode):
        try:await super().close(episode)
        finally:self.contracts.pop(episode.uid,None)
