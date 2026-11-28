import copy
import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from .hd2_model import HPTModel
from .objectives import flow_loss,StateReadout,state_loss,fk_loss,masked_mean
from .kinematics import RobotFK

class CampaignPolicy(nn.Module):
    def __init__(self,reference,arm,normalizer,robot):
        super().__init__();self.arm=arm
        cfg=OmegaConf.load(reference).model.robomimic_model
        m=HPTModel(**OmegaConf.to_container(cfg.trunk,resolve=True))
        m.auxiliary_ac_keys={'rby1':['dino_wm_future']} if arm=='B' else {}
        m.diffusion=True
        m.init_domain_stem('shared',instantiate(cfg.shared_stem_specs))
        m.shared_keys=list(cfg.shared_obs_keys)
        m.init_domain_stem('rby1',instantiate(cfg.stem_specs.rby1))
        m.init_domain_head('rby1',instantiate(cfg.head_specs.rby1))
        # Instantiate identically before pruning so paired seeds share policy weights.
        m.init_domain_head('rby1_dino_wm_future',instantiate(cfg.head_specs.rby1_dino_wm_future))
        for name,value in cfg.encoder_specs.items():m.init_encoder(name,instantiate(value))
        m.finalize_modules();m.finalize_modules() # matches historical HPT constructor
        if arm!='B':del m.heads['rby1_dino_wm_future']
        self.policy=m
        self.register_buffer('action_center',torch.tensor(normalizer['action']['center'],dtype=torch.float32))
        self.register_buffer('action_scale',torch.tensor(normalizer['action']['scale'],dtype=torch.float32))
        if arm=='C':
            self.state_readout=StateReadout()
            self.register_buffer('state_mean',torch.tensor(normalizer['current_q']['mean']))
            self.register_buffer('state_std',torch.tensor(normalizer['current_q']['std']))
        if arm=='D':self.fk=RobotFK(robot['urdf'],robot['joint_names'],robot['links'])

    def forward(self,batch):
        if self.training and hasattr(self,'photometric_config'):
            from .photometric import training_forward
            return training_forward(self,batch)
        return self._forward_once(batch)

    def _forward_once(self,batch,return_clean=False):
        self.policy.device=next(self.parameters()).device
        self.policy.train_mode=self.training
        data={k:v for k,v in batch.items() if k.startswith('front_pcd_') or k.startswith('state_')}
        features,_=self.policy.forward_features('rby1',data)
        la,clean,t=flow_loss(self.policy.heads['rby1'],features,batch['action'],batch['action_valid'])
        aux=la*0;metrics={'action':la.detach()}
        if self.arm=='B':
            # Existing DINO FM objective unchanged; only non-padding +10-row samples.
            keep=batch['dino_valid'].bool()
            if keep.any():
                aux=self.policy.heads['rby1_dino_wm_future'].compute_loss(features[keep],{'action':batch['dino'][keep]})
        elif self.arm=='C':
            pred=self.state_readout(self.policy.encoders['front_pcd_1'],batch['clean_cloud'])
            aux=.1*state_loss(pred,batch['current_q'],batch['current_q_valid'],self.state_mean,self.state_std)
        elif self.arm=='D':
            pred=clean.float()*self.action_scale+self.action_center
            target=batch['action'].float()*self.action_scale+self.action_center
            lf,parts=fk_loss(self.fk,pred,target,batch['action_valid'])
            aux=.1*lf;metrics.update(parts)
        metrics['aux']=aux.detach()
        return (la+aux,metrics,clean) if return_clean else (la+aux,metrics)

    def transfer_policy(self,source_state):
        # All target arms use fresh DINO head; retain initial target buffers/statistics.
        own=self.state_dict()
        keys=[k for k in own if k.startswith('policy.') and not k.startswith('policy.heads.rby1_dino_wm_future.')]
        for k in keys:
            if k not in source_state or source_state[k].shape!=own[k].shape:raise ValueError('Transfer mismatch: '+k)
            own[k]=source_state[k]
        self.load_state_dict(own,strict=True)
        return keys
