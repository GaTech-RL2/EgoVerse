"""Real-loader GeoAug contracts and train/eval wiring on allocated GPU."""
import os,json,sys,random,copy
from pathlib import Path
import numpy as np,torch
from torch.utils.data import default_collate
from campaign.data import EpisodeDataset
from campaign.policy import CampaignPolicy
from campaign.photometric import augment,rng_state,restore_rng
from campaign.geoaug import augment as geo
R=Path(os.environ['CAMPAIGN_ROOT']);cfg=json.load(open(R/'configs'/(sys.argv[1]+'.json')))
norm=json.load(open(R/'releases/cart/normalizer.json'))
ds=EpisodeDataset(R/'releases/cart/target.json','train','B',norm)
b={k:v.cuda()for k,v in default_collate([ds[i]for i in range(32)]).items()}
original={k:v.clone()for k,v in b.items()}
random.seed(42);np.random.seed(42);torch.manual_seed(42)
s=rng_state();y=augment(b,cfg['photometric']);restore_rng(s);yy=augment(b,cfg['photometric'])
assert all(torch.equal(y[k],yy[k])for k in y)
assert all(torch.equal(b[k],original[k])for k in b)
for k in b:
 if not k.startswith('front_pcd_'):assert torch.equal(b[k],y[k])
 else:
  assert y[k].shape==b[k].shape and torch.isfinite(y[k]).all()
  assert not torch.equal(b[k],y[k])
  rgb=y[k].reshape(32,-1,6)[...,3:];assert rgb.min()>=0 and rgb.max()<=1
# Edge cases: zero padding survives; valid points cannot disappear.
syn={'front_pcd_1':torch.zeros(2,1,1,48,device='cuda'),'front_pcd_2':torch.ones(2,1,1,24,device='cuda')}
syn['front_pcd_1'][0,0,0,6:12]=1
z=geo(syn,cfg['photometric']['geoaug'])
assert z['front_pcd_1'][1].eq(0).all() and z['front_pcd_1'][0,0,0,:6].eq(0).all()
assert z['front_pcd_1'][0,0,0,6:12].ne(0).any()
# Isolate metric jitter bounds; all three clouds share grayscale decision.
g=copy.deepcopy(cfg['photometric']['geoaug']);g['point_dropout']['p']=0;g['xyz_jitter']['p']=1;g['grayscale']['p']=1
z=geo(b,g)
for k in z:
 if k.startswith('front_pcd_'):
  a=b[k].reshape(32,-1,6);v=z[k].reshape(32,-1,6)
  assert (a[...,:3]-v[...,:3]).abs().max()<=.009001
  assert torch.equal(v[...,3],v[...,4])and torch.equal(v[...,4],v[...,5])
robot=json.load(open(R/'assets/robot.json'));robot['urdf']=str(R/'assets/robot.urdf')
model=CampaignPolicy(R/'configs/hd2-reference.yaml','B',norm,robot).cuda().train();model.photometric_config=cfg['photometric']
for enc in model.policy.encoders.values():enc.colour_brightness_jitter=0.;enc.colour_channel_jitter=0.
s=rng_state()
with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model(b)
assert torch.isfinite(loss);loss.backward()
assert all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None)
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.policy.encoders.parameters())
model.zero_grad(set_to_none=True);restore_rng(s)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 first,second=augment(b,cfg['photometric']),augment(b,cfg['photometric'])
 expected,_=model._forward_once(first)
assert torch.equal(loss.detach(),expected)
model.eval();s=rng_state()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 actual,_=model(b);restore_rng(s);expected,_=model._forward_once(b)
assert torch.equal(actual,expected)
report=dict(passed=True,real_loader_batch=32,labels_proprio_unchanged=True,input_immutable=True,zero_padding_preserved=True,empty_cloud_preserved=True,metric_jitter_bounded=True,shared_grayscale=True,rng_replay_exact=True,train_wiring_exact=True,eval_unchanged=True,finite_nonzero_encoder_gradients=True,geoaug=cfg['photometric']['geoaug'])
(R/'reports'/('photometric-audit-'+cfg['id']+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
