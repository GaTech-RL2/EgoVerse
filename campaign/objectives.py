"""Masked A–D objectives. No label or FK-parameter gradients."""
import contextlib
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = (slice(0,6),slice(6,8),slice(8,15),slice(15,22),slice(22,34),slice(34,46))

def masked_mean(value, mask):
    mask = torch.broadcast_to(mask.bool(), value.shape)
    if not torch.isfinite(value[mask]).all():
        raise ValueError('Nonfinite supervised values')
    return torch.where(mask, value, 0.).sum()/mask.sum().clamp_min(1)

def flow_loss(head, features, action, valid):
    """hd2 noise/time/hierarchy unchanged; expose differentiable clean estimate."""
    action, features = head.preprocess_compute_loss(features, {'action':action})
    noise = torch.randn_like(action)
    t = head._sample_time(len(action), action.device)
    xt = t[:,None,None]*noise+(1-t[:,None,None])*action
    parents = [None]*head.n_blocks
    clean=[]; losses=[]
    for i,sl in enumerate(head.block_slices):
        v=head.block_models[i](xt[...,sl],t,head._build_cond(features,parents,i))
        pred=xt[...,sl]-t[:,None,None]*v
        clean.append(pred)
        parents[i]=pred.detach()  # preserve historical hierarchy detach
        losses.append(masked_mean((v-(noise[...,sl]-action[...,sl])).square(),valid[...,None]))
    return sum(losses),torch.cat(clean,-1),t

@contextlib.contextmanager
def clean_encoder_view(encoder):
    # Share weights, disable only coordinate/RGB augmentation and segment sampling.
    # This temporary setting is scoped to a sequential auxiliary forward, not threads.
    names=('pose_jitter_deg','pose_jitter_m','point_noise_std_m',
           'colour_brightness_jitter','colour_channel_jitter','segments')
    old={n:getattr(encoder,n) for n in names}
    try:
        for n in names:setattr(encoder,n,None if n=='segments' else 0.)
        yield
    finally:
        for n,v in old.items():setattr(encoder,n,v)

class StateReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,46))
    def forward(self,encoder,clean_cloud):
        with clean_encoder_view(encoder):tokens=encoder(clean_cloud)
        return self.head(tokens.mean(1))

def state_loss(pred,q,valid,mean,std):
    if not torch.isfinite(q[valid.bool()]).all():raise ValueError('Nonfinite valid state target')
    target=(torch.where(valid.bool(),q.detach(),mean)-mean)/std.clamp_min(.05)
    values=F.smooth_l1_loss(pred,target,reduction='none')
    terms=[masked_mean(values[...,g],valid[...,g]) for g in GROUPS if valid[...,g].any()]
    return sum(terms)/len(terms) if terms else pred.sum()*0

def fk_loss(fk,pred,target,valid):
    # All kinematics in FP32 even under BF16 autocast.
    with torch.autocast(device_type=pred.device.type,enabled=False):
        p=fk(pred.float());t=fk(target.detach().float())
        pos=F.smooth_l1_loss(p[...,:3,3]/.05,t[...,:3,3]/.05,reduction='none')
        lp=masked_mean(pos,valid[...,None,None])
        rot=(p[...,:2,:3,:3]-t[...,:2,:3,:3]).square().sum((-1,-2))/8
        lr=masked_mean(rot,valid[...,None])
        q=pred[...,3:].float()
        lim=(F.relu(fk.lower-q)+F.relu(q-fk.upper))/fk.width
        ll=masked_mean(lim.square(),valid[...,None]&fk.limited)
        return lp+lr+.1*ll,dict(pos=lp.detach(),rot=lr.detach(),limit=ll.detach())
