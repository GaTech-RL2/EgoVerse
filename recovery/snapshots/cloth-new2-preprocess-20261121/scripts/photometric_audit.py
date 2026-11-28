"""Real-loader GPU contracts before either smoke; runs on an allocated GPU."""
import json,os,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from campaign.policy import CampaignPolicy
from campaign.data import EpisodeDataset,sha
from campaign.photometric import augment,rng_state,restore_rng
from campaign.objectives import masked_mean
R=Path(os.environ['CAMPAIGN_ROOT'])
cfg=json.loads((R/'configs/cloth-new2-B-msc-scratch-s42.json').read_text())
norm=json.loads((R/'releases/cloth-new2-scratch/normalizer.json').read_text())
robot=json.loads((R/'assets/robot.json').read_text());robot['urdf']=str(R/'assets/robot.urdf')
torch.manual_seed(42);np.random.seed(42);random.seed(42)
model=CampaignPolicy(R/'configs/hd2-reference.yaml','B',norm,robot).cuda().train()
model.photometric_config=cfg['photometric']
for e in model.policy.encoders.values():e.colour_brightness_jitter=0.;e.colour_channel_jitter=0.
ds=EpisodeDataset(R/'releases/cloth-new2-scratch/target.json','train','B',norm)
batch={k:v.cuda() for k,v in default_collate([ds[i] for i in range(32)]).items()}
for k,x in batch.items():
    if k.startswith('front_pcd_'):
        rgb=x.reshape(len(x),-1,6)[...,3:];assert rgb.min()>=0 and rgb.max()<=1
other=augment(batch)
for k,x in batch.items():
    if k.startswith('front_pcd_'):
        assert torch.equal(x.reshape(len(x),-1,6)[...,:3],other[k].reshape(len(x),-1,6)[...,:3])
    else:assert torch.equal(x,other[k])
assert any(not torch.equal(batch[k],other[k]) for k in batch if k.startswith('front_pcd_'))
# Confirm wrapper parity with the unchanged upstream classes on concatenated RGB.
from campaign.pointcept_color import RandomColorJitter,ChromaticJitter
synthetic={k:torch.rand(2,2,1,n*6,device='cuda') for k,n in [('front_pcd_1',8),('front_pcd_2',4)]}
synthetic['front_pcd_1'][0,0,0,:6]=0
s=rng_state();y=augment(synthetic);restore_rng(s)
for i in range(2):
    arrays=[x[i].cpu().numpy().reshape(-1,6) for x in synthetic.values()]
    masks=[np.any(x!=0,axis=-1) for x in arrays]
    colors=np.concatenate([x[m,3:] for x,m in zip(arrays,masks)])*255.
    ref=ChromaticJitter(p=.95,std=.05)(RandomColorJitter(.4,.4,.2,.02,p=.8)({'color':colors}))['color']/255.
    got=np.concatenate([x[i].cpu().numpy().reshape(-1,6)[m,3:] for x,m in zip(y.values(),masks)])
    assert np.array_equal(ref,got)
assert y['front_pcd_1'][0,0,0,:6].eq(0).all()
j=RandomColorJitter()
colors=np.array([[32.,64.,128.],[192.,128.,64.]],dtype=np.float32)
assert np.array_equal(j.adjust_brightness(colors,1),colors)
assert np.allclose(j.adjust_contrast(colors,0),np.mean(.2989*colors[:,0]+.587*colors[:,1]+.114*colors[:,2]))
assert np.allclose(j.adjust_saturation(colors,0)[:,0],j.adjust_saturation(colors,0)[:,2])
# Replay must give identical clean estimates with identical photometry, while
# different photometry must yield finite, nonzero differentiable consistency.
s=rng_state()
with torch.autocast('cuda',dtype=torch.bfloat16):
    l1,m1,c1=model._forward_once(batch,return_clean=True)
    restore_rng(s)
    l2,m2,c2=model._forward_once(batch,return_clean=True)
assert torch.equal(c1,c2) and torch.equal(l1,l2)
del l2,m2,c2
restore_rng(s)
with torch.autocast('cuda',dtype=torch.bfloat16):
    l3,m3,c3=model._forward_once(other,return_clean=True)
cons=masked_mean((c1.float()-c3.float()).square(),batch['action_valid'][...,None])
assert torch.isfinite(cons) and cons.item()>0
cons.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.policy.encoders.parameters())
cons_value=cons.item()
del l1,l3,c1,c3,cons
model.zero_grad(set_to_none=True)
model.eval();s=rng_state()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    v1,_=model(batch);restore_rng(s);v2,_=model._forward_once(batch)
assert torch.equal(v1,v2)
report=dict(passed=True,device=torch.cuda.get_device_name(),batch=32,train_rows=len(ds),
    xyz_labels_proprio_unchanged=True,shared_photometry_padding_checked=True,upstream_wrapper_exact_parity=True,
    replay_identity_exact=True,consistency_nonzero=cons_value,encoder_consistency_gradient_finite_nonzero=True,
    eval_unchanged=True,policy_initialized_from_scratch=True,all_policy_parameters_trainable=all(p.requires_grad for p in model.parameters()))
assert cfg['stage']=='scratch' and not any(k in cfg for k in ['parent','parent_step','parent_checkpoint_sha256'])
(R/'reports/photometric-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
