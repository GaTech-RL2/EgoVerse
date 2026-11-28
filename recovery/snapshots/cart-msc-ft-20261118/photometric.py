"""Pinned official MSC color transforms; shared per observation, training only."""
import random
import numpy as np
import torch
from .objectives import masked_mean
from .pointcept_color import RandomColorJitter,ChromaticJitter

def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()

def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])

def augment(batch):
    # Keep the pinned NumPy implementation exactly, including random operation
    # order and independent probability checks. Input arrays remain float32.
    keys=[k for k in batch if k.startswith('front_pcd_')]
    assert keys
    arrays={k:batch[k].detach().cpu().numpy().copy().reshape(len(batch[k]),-1,6) for k in keys}
    jitter=RandomColorJitter(brightness=.4,contrast=.4,saturation=.2,hue=.02,p=.8)
    noise=ChromaticJitter(p=.95,std=.05)
    for i in range(len(batch[keys[0]])):
        masks={k:np.any(arrays[k][i]!=0,axis=-1) for k in keys}
        counts=[int(masks[k].sum()) for k in keys]
        if not sum(counts):continue
        colors=np.concatenate([arrays[k][i,masks[k],3:6] for k in keys],axis=0)
        assert np.isfinite(colors).all() and colors.min()>=0 and colors.max()<=1
        result=noise(jitter({'color':colors*255.}))['color']/255.
        offset=0
        for k,n in zip(keys,counts):
            arrays[k][i,masks[k],3:6]=result[offset:offset+n];offset+=n
    out=dict(batch)
    for k in keys:out[k]=torch.from_numpy(arrays[k]).to(batch[k]).reshape_as(batch[k])
    return out

def training_forward(model,batch):
    # Both experiments draw both views to preserve paired RNG advancement.
    b1,b2=augment(batch),augment(batch)
    start=rng_state()
    l1,m1,c1=model._forward_once(b1,return_clean=True)
    end=rng_state();weight=model.photometric_config['consistency_weight']
    if weight==0:
        m1['consistency']=l1.detach()*0
        return l1,m1
    restore_rng(start)
    try:l2,m2,c2=model._forward_once(b2,return_clean=True)
    finally:restore_rng(end)
    consistency=masked_mean((c1.float()-c2.float()).square(),batch['action_valid'][...,None])
    metrics={k:(m1[k]+m2[k])*.5 for k in m1}
    metrics['consistency']=consistency.detach()
    metrics['consistency_weighted']=weight*consistency.detach()
    return (l1+l2)*.5+weight*consistency,metrics
