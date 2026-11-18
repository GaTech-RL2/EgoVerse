"""Check probe completion against the native runner's ordinary step budget."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from reproductions.leact.cluster import efe_pear_yam_contract,efe_pear_yam_reset_bank


@pytest.mark.parametrize('reset_count',[1,3])
def test_probe_finishes_normally_after_every_reset_has_31_steps(monkeypatch,tmp_path,reset_count):
    def module(name,**values):
        item=ModuleType(name);item.__dict__.update(values);monkeypatch.setitem(sys.modules,name,item)
        return item
    class Base:
        @classmethod
        def __class_getitem__(cls,_):return cls
        def __init__(self,config):self.config=config
    module('torch',tensor=lambda value,**_:np.array(value),float32=np.float32)
    settings={'/renderer/activeGpu':0,'/renderer/multiGpu/enabled':False,'/renderer/multiGpu/autoEnable':False}
    module('carb',settings=SimpleNamespace(get_settings=lambda:SimpleNamespace(get=settings.get)))
    module('isaaclab_arena');module('isaaclab_arena.assets');module('isaaclab_arena.policy')
    module('isaaclab_arena.assets.register',register_policy=lambda cls:cls)
    module('isaaclab_arena.policy.policy_base',PolicyBase=Base,PolicyCfg=type('PolicyCfg',(),{}))
    monkeypatch.setitem(sys.modules,'efe_pear_yam_contract',efe_pear_yam_contract)
    monkeypatch.setitem(sys.modules,'efe_pear_yam_reset_bank',efe_pear_yam_reset_bank)
    monkeypatch.setenv('RP_JHH_EXPECTED_RENDERER_INDEX','0')
    path=Path(__file__).parents[1]/'cluster/efe_pear_yam_api_policy.py'
    spec=importlib.util.spec_from_file_location('probe_test_policy',path)
    policy_module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,policy_module);spec.loader.exec_module(policy_module)
    layouts=[{'initialization_id':str(i),'object_init_poses':{'cube':[float(i),0.,0.,1.,0.,0.,0.]}}
             for i in range(reset_count)]
    fingerprints=[efe_pear_yam_reset_bank.pose_fingerprint(x['object_init_poses'])for x in layouts]
    data=SimpleNamespace(joint_pos=np.array([[0.]]),soft_joint_pos_limits=np.array([[[-1.,1.]]]),
                         root_state_w=np.zeros((1,13)))
    term=SimpleNamespace(_joint_ids=[0],_asset=SimpleNamespace(data=data),action_dim=1,
                         _joint_names=['joint1'],cfg=SimpleNamespace(scale=1.,offset=0.))
    manager=SimpleNamespace(active_terms=['arm'],get_term=lambda _:term)
    terms=SimpleNamespace(active_terms=['success','time_out'],get_term=lambda _:np.array([False]))
    base=SimpleNamespace(action_manager=manager,termination_manager=terms,step_dt=1/30,
        device='cpu',scene=SimpleNamespace(articulations={'robot':SimpleNamespace(data=data)},rigid_objects={}),
        _tgen_reset_layouts=[layouts[0]])
    observation={'camera_obs':{'top':np.arange(12,dtype=np.uint8).reshape(1,2,2,3)*10}}
    visited=[0]
    def reset():
        visited[0]+=1;base._tgen_reset_layouts=[layouts[visited[0]]]
        return observation,{}
    env=SimpleNamespace(unwrapped=base,reset=reset)
    target=tmp_path/'api-result.json'
    cfg=policy_module.YamApiProbePolicyCfg(report_path=str(target),reset_count=reset_count,
        training_reset_manifest='generated.json'if reset_count>1 else'',expected_pose_fingerprints=fingerprints)
    policy=policy_module.YamApiProbePolicy(cfg)
    # The native runner calls get_action then env.step; the last call must
    # return normally so its close hooks run instead of catching SystemExit.
    for _ in range(31*reset_count+1):assert policy.get_action(env,observation).shape==(1,1)
    receipt=json.loads(target.read_text())
    assert receipt['api_reset_action_observation_passed']
    assert receipt['hold_control_steps']==31
    if reset_count>1:
        assert len(receipt['resets'])==reset_count
        assert all(x['hold_control_steps']==31 for x in receipt['resets'].values())
    assert not receipt['training_admitted']
