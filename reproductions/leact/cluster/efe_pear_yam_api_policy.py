"""Two-step native YAM API probe, never policy demonstrations or RL data."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase,PolicyCfg


def array(value):
    value=value.torch if hasattr(value,'torch')else value
    return value.detach().cpu().numpy()if hasattr(value,'detach')else np.asarray(value)


@dataclass
class YamApiProbePolicyCfg(PolicyCfg):
    report_path:str=''


@register_policy
class YamApiProbePolicy(PolicyBase[YamApiProbePolicyCfg]):
    name='rp_jhh_yam_api_probe'

    def __init__(self,config):
        super().__init__(config);self.config=config;self.calls=0;self.initial=None

    def get_action(self,env,observation):
        from PIL import Image
        import carb
        settings=carb.settings.get_settings()
        if (settings.get('/renderer/activeGpu')!=int(os.environ['RP_JHH_EXPECTED_RENDERER_INDEX'])or
            settings.get('/renderer/multiGpu/enabled')or settings.get('/renderer/multiGpu/autoEnable')):
            raise ValueError('Native renderer is not isolated to its assigned GPU')
        base=env.unwrapped;manager=base.action_manager;layout=[];values=[];lower=[];upper=[]
        for name in manager.active_terms:
            term=manager.get_term(name);ids=term._joint_ids;asset=term._asset
            measured=array(asset.data.joint_pos)[0,ids].reshape(-1)
            limits=array(asset.data.soft_joint_pos_limits)[0,ids]
            if int(term.action_dim)==1 and len(measured)>1:
                measured=np.array([measured.mean()]);limits=np.array([[limits[:,0].max(),limits[:,1].min()]])
                names=[name+'_mirrored_finger_target_m']
            else:names=[name+'/'+n for n in term._joint_names]
            scale=np.broadcast_to(array(getattr(term,'_scale',getattr(term.cfg,'scale',1.))).reshape(-1),measured.shape)
            offset=np.broadcast_to(array(getattr(term,'_offset',getattr(term.cfg,'offset',0.))).reshape(-1),measured.shape)
            if np.any(scale==0):raise ValueError('Native action scale is zero')
            raw=(measured-offset)/scale;lo=(limits[:,0]-offset)/scale;hi=(limits[:,1]-offset)/scale
            lo,hi=np.minimum(lo,hi),np.maximum(lo,hi)
            if len(names)!=term.action_dim or not np.isfinite(np.r_[raw,lo,hi]).all():raise ValueError('Native action layout/bounds are invalid')
            if np.any(raw<lo-.005)or np.any(raw>hi+.005):raise ValueError('Native reset lies outside declared joint limits')
            layout.extend(names);values.extend(np.clip(raw,lo,hi).tolist());lower.extend(lo.tolist());upper.extend(hi.tolist())
        cameras={}
        for name,image in (observation.get('camera_obs')or{}).items():
            rgb=array(image)
            if rgb.ndim==4:rgb=rgb[0]
            if rgb.ndim!=3 or rgb.shape[-1]not in (3,4):continue
            if not np.isfinite(rgb).all():raise ValueError('Native RGB contains nonfinite values')
            if rgb.dtype!=np.uint8:rgb=(np.clip(rgb,0,1)*255).astype(np.uint8)if rgb.max()<=1 else np.clip(rgb,0,255).astype(np.uint8)
            rgb=rgb[...,:3];cameras[name]={'shape':list(rgb.shape),'std':float(rgb.std())}
            if self.calls==0:
                output=Path(self.config.report_path).parent/'frames';output.mkdir(exist_ok=True)
                Image.fromarray(rgb).save(output/(name.replace('/','_')+'.png'))
        if not cameras or all(v['std']<1 for v in cameras.values()):raise ValueError('Native reset has no usable rendered camera')
        contract={'action_types':{'joint_position':{'dim':len(values),'layout':layout,'lower':lower,'upper':upper,
            'semantics':'Absolute joint targets in native action-term order; arm radians and mirrored finger translations in meters.'}},
            'cameras':list(cameras),'control_dt':float(base.step_dt),'max_repeat':25,
            'protocol':'efe-constrained-action-v3','output_keys':['action_type','action','repeat','summary']}
        digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        if self.calls==0:self.initial={'action_contract':contract,'action_contract_sha256':digest,'cameras':cameras}
        elif digest!=self.initial['action_contract_sha256']:raise ValueError('Native action contract changed after one control step')
        self.calls+=1
        if self.calls>=32:
            terms={}
            for name,value in getattr(base.termination_manager,'_term_dones',{}).items():terms[name]=bool(array(value).reshape(-1)[0])
            result={**self.initial,'api_reset_action_observation_passed':True,'native_termination_terms':terms,
                'policy_model_inference_calls':0,'primary_rl_updates':0,'training_admitted':False,'hold_control_steps':self.calls-1,
                'scope':'Authored fixture API probe only. Independent training reset generation and reward qualification remain separate.'}
            target=Path(self.config.report_path);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(result,indent=2)+'\n')
            raise SystemExit(0)
        return torch.tensor([values],dtype=torch.float32,device=base.device)

    def reset(self,env_ids=None):pass
    def close(self):pass
