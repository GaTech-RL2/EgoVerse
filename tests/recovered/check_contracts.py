import json,importlib.util
from pathlib import Path
import torch
import numpy as np
from campaign.policy import CampaignPolicy
from campaign.objectives import flow_loss,fk_loss

root=Path(__file__).resolve().parents[2];torch.set_num_threads(4)
robot=json.loads((root/'assets/robot.json').read_text());robot['urdf']=str(root/'assets/robot.urdf')
norm=dict(action=dict(center=[0.]*49,scale=[1.]*49),current_q=dict(mean=[0.]*46,std=[1.]*46))
reference=None
for arm in 'ABCD':
    torch.manual_seed(42);model=CampaignPolicy(root/'configs/hd2-reference.yaml',arm,norm,robot)
    common={k:v.clone() for k,v in model.state_dict().items() if k.startswith('policy.') and not k.startswith('policy.heads.rby1_dino_wm_future.')}
    if reference is None:reference=common
    else:
        assert common.keys()==reference.keys()
        for k in common:torch.testing.assert_close(common[k],reference[k],rtol=0,atol=0)
    if arm=='D':
        head=model.policy.heads['rby1'];features=torch.randn(2,64,256);action=torch.randn(2,64,49)*.1;mask=torch.ones(2,64,dtype=torch.bool)
        rng=torch.get_rng_state();original=head.compute_loss(features,{'action':action})
        torch.set_rng_state(rng);new,clean,_=flow_loss(head,features,action,mask)
        torch.testing.assert_close(new,original,rtol=1e-6,atol=1e-6)
        lf,_=fk_loss(model.fk,clean,action,mask)
        grads=torch.autograd.grad(lf,tuple(head.parameters()),allow_unused=True)
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    del model
torch.manual_seed(123);target=CampaignPolicy(root/'configs/hd2-reference.yaml','B',norm,robot)
dino={k:v.clone() for k,v in target.state_dict().items() if k.startswith('policy.heads.rby1_dino_wm_future.')}
keys=target.transfer_policy(reference)
for k in keys:torch.testing.assert_close(target.state_dict()[k],reference[k],rtol=0,atol=0)
for k in dino:torch.testing.assert_close(target.state_dict()[k],dino[k],rtol=0,atol=0)
spec=importlib.util.spec_from_file_location('stats',root/'scripts/fit_normalizer.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
x=np.arange(400,dtype=float).reshape(100,4);n=stats.fit(x,0);c=np.array(n['center']);s=np.array(n['scale'])
np.testing.assert_allclose(((x-c)/s)*s+c,x,atol=1e-12)
try:stats.fit(np.zeros((10,4)),0)
except AssertionError:pass
else:raise AssertionError('Constant dimensions must fail without reviewed floors')
print(json.dumps(dict(passed=True,paired_initialization=True,hd2_fm_parity=True,fk_gradient_to_action_head=True,policy_transfer=True,fresh_dino=True,normalizer_roundtrip=True,constant_guard=True)))
