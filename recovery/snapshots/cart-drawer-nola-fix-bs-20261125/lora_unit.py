import os,sys,json,copy
from pathlib import Path
import torch
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');W=R/'training/cart/lora';sys.path[:0]=[str(W),str(W/'source')]
from campaign.policy import CampaignPolicy
from campaign.data import sha
from campaign.dino_lora import install,transfer,adapter_key,PREFIX
torch.set_num_threads(4);torch.manual_seed(42)
cfg=json.load(open(next((W/'configs').glob('*.json'))));norm=json.load(open(W/'releases/cart/normalizer.json'));robot=json.load(open(W/'assets/robot.json'));robot['urdf']=str(W/'assets/robot.urdf')
m=CampaignPolicy(W/'configs/hd2-reference.yaml','B',norm,robot)
parent=W/'runs'/cfg['parent']/'step_180000.pt';assert sha(parent)==cfg['parent_checkpoint_sha256'];ck=torch.load(parent,map_location='cpu',weights_only=False)
sd=m.state_dict();sd.update({k:v for k,v in ck['model'].items()if k.startswith('policy.')});m.load_state_dict(sd);m.eval();head=m.policy.heads['rby1_dino_wm_future'];x=torch.randn(2,16,768);t=torch.rand(2);cond=torch.randn(2,16,256,requires_grad=True)
with torch.no_grad():old=head.model(x,t,cond).clone()
modules=install(m,cfg['dino_head_lora']);keys=transfer(m,ck['model']);m.eval()
with torch.no_grad():new=head.model(x,t,cond)
# Parametrized MHA can select a different CPU matmul path. Effective weights
# must remain bit-identical; allow float32 accumulation error in its forward.
forward_max_abs=float((old-new).abs().max())
torch.testing.assert_close(old,new,atol=1e-5,rtol=1e-5)
initial={k:v.detach().clone()for k,v in m.named_parameters()};opt=torch.optim.AdamW([p for p in m.parameters()if p.requires_grad],lr=1e-4)
for _ in range(3):
 opt.zero_grad();loss=head.model(x,t,cond).square().mean();loss.backward();assert torch.isfinite(loss);assert cond.grad is not None and torch.isfinite(cond.grad).all()and cond.grad.abs().sum()>0;opt.step()
for k,p in m.named_parameters():
 if k.startswith(PREFIX)and not adapter_key(k):assert torch.equal(initial[k],p),k
assert any(not torch.equal(initial[k],p)for k,p in m.named_parameters()if adapter_key(k))
report=dict(passed=True,parent_sha256=sha(parent),zero_adapter_forward_max_abs=forward_max_abs,zero_adapter_forward_tolerance=dict(atol=1e-5,rtol=1e-5),parent_weights_exact=True,frozen_head_unchanged=True,adapters_changed=True,upstream_feature_gradient_nonzero=True,all_nonhead_trainable=all(p.requires_grad for k,p in m.named_parameters()if not k.startswith(PREFIX)),modules=modules,lora_code_sha256=sha(W/'campaign/dino_lora.py'))
(W/'reports/lora-unit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
