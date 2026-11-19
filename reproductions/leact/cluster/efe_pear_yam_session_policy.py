"""Persistent native YAM execution behind PEAR's constrained action tool.

Only RGB and robot proprioception cross to the policy. Reset selection uses the
native layout event, and the unchanged native termination manager supplies all
success/failure outcomes. The observation hook preserves its return value.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase,PolicyCfg

from efe_action_loop import parse_action
from efe_pear_public_observation import public_blocks
from efe_pear_file_rpc import atomic_json,claim_episode,publish_result,wait_command
from efe_pear_yam_api_policy import YamApiProbePolicy,YamApiProbePolicyCfg,array
from efe_pear_yam_contract import termination_snapshot,observe_termination_computes,terminal_result
from efe_pear_yam_reset_bank import pose_fingerprint


@dataclass
class YamSessionPolicyCfg(PolicyCfg):
    transport_root:str=''
    task:str=''
    training_reset_manifest:str=''
    training_reset_manifest_sha256:str=''
    output:str=''
    worker_id:str=''


@register_policy
class YamSessionPolicy(PolicyBase[YamSessionPolicyCfg]):
    name='rp_jhh_yam_pear_session'

    def __init__(self,config):
        super().__init__(config)
        raw=Path(config.training_reset_manifest).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=config.training_reset_manifest_sha256:
            raise ValueError('Independent native training reset bank changed')
        self.root=Path(config.transport_root);self.output=Path(config.output)
        self.output.mkdir(parents=True,exist_ok=True);self.schedule=None
        self.terms=None;self.steps=0;self.folder=None;self.bound=False
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',config.worker_id):raise ValueError('Invalid native worker identity')
        workers=self.root/'workers';workers.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.stopped=threading.Event()
        def heartbeat():
            while not self.stopped.is_set():
                atomic_json(workers/(config.worker_id+'.json'),{'updated_unix':time.time(),
                    'task':config.task,'kind':'native_yam','worker_id':config.worker_id})
                self.stopped.wait(10)
        self.heartbeat=threading.Thread(target=heartbeat,daemon=True);self.heartbeat.start()

    def _bind(self,env):
        base=env.unwrapped
        if base.num_envs!=1:raise ValueError('Each native session must own exactly one simulated robot')
        self.schedule=copy.deepcopy(base.cfg.tgen_reset_layouts)
        if self.schedule['source_sha256']!=self.config.training_reset_manifest_sha256:
            raise ValueError('Native layout schedule differs from the pinned independent bank')
        manager=base.termination_manager
        if'success'not in manager.active_terms:raise ValueError('Native task lacks an official success termination')
        observe_termination_computes(manager,lambda values:setattr(self,'terms',values))
        self.bound=True

    def _public_result(self,env,observation,*,done=False,action=None,executed=0):
        if done:
            return terminal_result(self.contract,self.instruction,self.scene_sha256,self.terms,
                action,executed,self.folder/'native-verdict.json')
        from PIL import Image
        turn=self.folder/'observations'/f'{self.steps:08d}';turn.mkdir(parents=True,exist_ok=True)
        view=YamApiProbePolicy(YamApiProbePolicyCfg(report_path=str(turn/'api-check.json'),save_frames=False))
        hold=view.get_action(env,observation);contract=view.initial['action_contract']
        images=[]
        for name in contract['cameras']:
            rgb=array(observation['camera_obs'][name]);rgb=rgb[0]if rgb.ndim==4 else rgb
            if rgb.dtype.name!='uint8':
                import numpy as np
                rgb=(np.clip(rgb,0,1)*255).astype(np.uint8)if rgb.max()<=1 else np.clip(rgb,0,255).astype(np.uint8)
            images.append(Image.fromarray(rgb[...,:3]))
        joint=contract['action_types']['joint_position']
        measured=array(hold)[0].tolist()
        if os.environ.get('PEAR_NATIVE_FLOAT32_JSON')=='1':
            from efe_pear_float32_json import public_number
            measured=[public_number(v)for v in measured]
        visible={'mode':'task','objective':env.unwrapped.get_language_instruction(),
            'action_contract':contract,'spec':{'frames':{
                'world':'Native simulator world frame; +Z is up. Camera image axes do not define robot axes.',
                'joint_position':'Positive angles follow the named native joint axes; use the exact joint order.'},
                'cameras':{name:{'rgb':True,'size':list(image.size),'moving':'wrist'in name}
                    for name,image in zip(contract['cameras'],images,strict=True)}},
            'observation':{'proprio':{'layout':joint['layout'],'now':measured,
                'units':'Arm angles in radians; mirrored prismatic finger targets in meters.'}},
            'recent_actions_and_feedback':[]if action is None else[{'action':action,'execution_completed':True,
                'control_steps_before':self.steps-executed,'control_steps_after':self.steps}]}
        blocks,context_sha=public_blocks(visible,images)
        self.contract=contract;self.instruction=visible['objective']
        reward=float(bool(self.terms and self.terms.get('success')))if done else 0.
        return {'images':None if done else images,'state':None if done else visible['observation'],
            'public_blocks':None if done else blocks,'action_contract':contract,
            'instruction':visible['objective'],'actions':[]if action is None or action.get('stop')else[action],
            'rewards':[reward],'done':done,'executed':int(bool(action and not action.get('stop'))),
            'outcome':'done'if done else'observed','native_trial_path':str(self.folder/'native-verdict.json'),
            'scene_sha256':self.scene_sha256,'native_termination_terms':self.terms,
            'interface_context_version':'pear-public-interface-language-v1',
            'interface_context_sha256':context_sha,
            'requested_control_steps':0 if action is None else action.get('repeat',0),
            'executed_control_steps':executed,'action_contract_sha256':view.initial['action_contract_sha256']}

    def get_action(self,env,observation):
        if not self.bound:self._bind(env)
        while True:
            claimed=claim_episode(self.root,self.config.worker_id,kinds={'native_yam/'+self.config.task})
            if claimed:break
            time.sleep(.1)
        self.folder,request=claimed;self.steps=0;committed=False
        try:
            args=request['env_args']
            if set(args)!={'task','reset_id'}or args['task']!=self.config.task:
                raise ValueError('Native task/reset request is malformed')
            selected=[row for row in self.schedule['images']if row['initialization_id']==args['reset_id']]
            if len(selected)!=1:raise ValueError('Requested reset is absent from the independent training bank')
            base=env.unwrapped
            base.cfg.tgen_reset_layouts={**copy.deepcopy(self.schedule),'images':copy.deepcopy(selected),'cyclic_inventory':True}
            observation,_=env.reset()
            # Use the native camera-refresh helper after every reset, including
            # wrist sensors whose transforms otherwise lag the robot reset.
            from gr00t_sim.embodiments.common.camera_sync import refresh_reset_camera_observation
            observation=refresh_reset_camera_observation(env,observation,[0],wrist_sync_term='sync_wrist_cameras')
            layout=base._tgen_reset_layouts[0]
            self.scene_sha256=pose_fingerprint(layout['object_init_poses'])
            if self.scene_sha256!=pose_fingerprint(selected[0]['object_init_poses']):
                raise ValueError('Native reset assignment differs from the requested training scene')
            # Recompute official termination buffers through one ordinary hold
            # step after reset. A preceding episode's latched buffers must not
            # be mistaken for the new reset's result.
            hold=YamApiProbePolicy(YamApiProbePolicyCfg(report_path=str(self.folder/'reset-hold.json'),save_frames=False)).get_action(env,observation)
            observation,_,terminated,truncated,_=env.step(hold)
            if bool(terminated.any()or truncated.any()):raise ValueError('Native reset terminated during its initial hold')
            self.terms=termination_snapshot(base.termination_manager)
            if any(self.terms.values()):raise ValueError('Native training reset is already terminal')
            publish_result(self.folder,0,self._public_result(env,observation));index=0
            while True:
                index+=1;action=wait_command(self.folder,index)
                if action is None:break
                parse_action(json.dumps(action),contract=self.contract,allow_stop=True)
                if action.get('stop'):
                    value=self._public_result(env,observation,done=True,action=action)
                    atomic_json(self.folder/'native-verdict.json',{'native_termination_terms':self.terms,
                        'control_steps':self.steps,'policy_stop':True,'success':bool(value['rewards'][0])})
                    publish_result(self.folder,index,value);committed=True;break
                executed=0;done=False
                target=torch.tensor([action['action']],dtype=torch.float32,device=base.device)
                for _ in range(action['repeat']):
                    observation,_,terminated,truncated,_=env.step(target);self.steps+=1;executed+=1
                    done=bool(terminated.any()or truncated.any())
                    if done:break
                value=self._public_result(env,observation,done=done,action=action,executed=executed)
                if done:atomic_json(self.folder/'native-verdict.json',{'native_termination_terms':self.terms,
                    'control_steps':self.steps,'policy_stop':False,'success':bool(value['rewards'][0])})
                publish_result(self.folder,index,value)
                if done:committed=True;break
            return YamApiProbePolicy(YamApiProbePolicyCfg(report_path=str(self.folder/'hold.json'),
                save_frames=False,joint_hold_only=True)).get_action(env,observation)
        except Exception as exc:
            if not committed:atomic_json(self.folder/'infrastructure-error.json',{'error':repr(exc)})
            else:print('PEAR_NATIVE_POST_VERDICT_ERROR '+type(exc).__name__,flush=True)
            raise

    def close(self):
        self.stopped.set();self.heartbeat.join(timeout=2)
