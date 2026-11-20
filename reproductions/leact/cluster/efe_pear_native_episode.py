"""Model-independent native EFE episode; simulator workers need no training imports."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from efe_action_loop import parse_action
from efe_pear_bridge import HarborEpisode,NativeInfrastructureError,image_blocks,observation_payload
from efe_pear_context import VERSION as INTERFACE_CONTEXT_VERSION
from efe_pear_observation_check import validate_frames


class NativeEfeEpisode:
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
        if self.config.get('control_interface'):
            from efe_pear_eef import enabled
            enabled(self.config['control_interface'])
            self.config={**self.config,'action_config':{**self.config['action_config'],
                'control_interface':self.config['control_interface'],'suite':self.row['suite']}}
        self.bridge=HarborEpisode(self.row,self.config,uuid4().hex)
        self.contract=None;self.contract_sha=None;self.last_action=None;self.instruction=''

    def _result(self,value):
        images=None;blocks=None;state=None;executed_controls=None;context_sha=None
        if not value['done']:
            visible,content=observation_payload(value['body'])
            self.contract=visible['action_contract'];self.instruction=visible['objective']
            self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
            blocks=image_blocks(content);images=[b['image'] for b in blocks if b['type']=='image']
            validate_frames(images,self.contract['cameras'])
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
            'scene_sha256':self.expected_scene,
            'interface_context_version':INTERFACE_CONTEXT_VERSION,'interface_context_sha256':context_sha,
            'requested_control_steps':action.get('repeat',0) if action else 0,
            'executed_control_steps':executed_controls}

    def open(self):
        try:return self._result(self.bridge.start())
        except NativeInfrastructureError as exc:raise NativeInfrastructureError(str(exc)) from exc

    def step(self,action):
        # The exact same parser runs inside the native container before execution.
        parse_action(json.dumps(action),contract=self.contract,allow_stop=True)
        self.last_action=action
        try:return self._result(self.bridge.step(action))
        except NativeInfrastructureError as exc:raise NativeInfrastructureError(str(exc)) from exc

    def close(self):
        self.bridge.close()
        from efe_pear_episode_archive import completed_episode
        completed_episode(self.bridge.root,self.row['suite']+'_'+self.bridge.episode_id)
