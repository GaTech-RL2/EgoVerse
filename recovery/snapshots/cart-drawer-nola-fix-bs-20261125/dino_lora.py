"""LoRA confined to the inherited future-DINO prediction head."""
import math,re
import torch
from torch import nn
from torch.nn.utils import parametrize
PREFIX='policy.heads.rby1_dino_wm_future.'
class LowRank(nn.Module):
 def __init__(self,weight,rank,alpha):
  super().__init__();self.scale=alpha/rank
  self.lora_A=nn.Parameter(torch.empty(rank,weight.shape[1],device=weight.device,dtype=weight.dtype));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
  self.lora_B=nn.Parameter(torch.zeros(weight.shape[0],rank,device=weight.device,dtype=weight.dtype))
 def forward(self,weight):return weight+(self.lora_B@self.lora_A)*self.scale
def adapter_key(k):return '.parametrizations.'in k and(k.endswith('.lora_A')or k.endswith('.lora_B'))
def source_key(k):return re.sub(r'\.parametrizations\.([^.]+)\.original$',r'.\1',k)
def install(model,cfg):
 assert cfg['rank']==8 and cfg['alpha']==16 and cfg['dropout']==0 and cfg['base_frozen'] and cfg['inherit']
 head=model.policy.heads['rby1_dino_wm_future'];head.requires_grad_(False);names=[]
 for name,module in list(head.named_modules()):
  fields=['weight']if isinstance(module,nn.Linear)else(['in_proj_weight']if isinstance(module,nn.MultiheadAttention)and module.in_proj_weight is not None else[])
  for field in fields:
   weight=getattr(module,field);parametrize.register_parametrization(module,field,LowRank(weight,cfg['rank'],cfg['alpha']));names.append(name+'.'+field)
 assert names
 for k,p in model.named_parameters():
  assert p.requires_grad==(adapter_key(k)if k.startswith(PREFIX)else True),(k,p.requires_grad)
 return names
def transfer(model,source):
 own=model.state_dict();keys=[]
 for k,v in own.items():
  if not k.startswith('policy.')or adapter_key(k):continue
  sk=source_key(k);assert sk in source and source[sk].shape==v.shape,(k,sk)
  own[k]=source[sk];keys.append(k)
 model.load_state_dict(own)
 assert any(k.startswith(PREFIX)for k in keys)
 for k in keys:assert torch.equal(model.state_dict()[k].cpu(),source[source_key(k)].cpu())
 assert all(torch.count_nonzero(p)==0 for k,p in model.named_parameters()if k.endswith('.lora_B'))
 return keys
def smoke_checks(model,initial):
 base=[];adapters=[];other=[]
 for k,p in model.named_parameters():
  changed=not torch.equal(initial[k],p.detach().cpu())
  if k.startswith(PREFIX):
   if adapter_key(k):adapters.append(changed)
   else:assert not changed,k;base.append(k)
  else:other.append(changed)
 assert base and any(adapters)and any(other)
 return dict(inherited_head_base_unchanged=True,lora_weights_changed=True,rest_policy_changed=True,frozen_parameters=len(base))
