"""Pinned Cart MSC color recipe on RGB pixels, before ImageNet normalization."""
import numpy as np
import torch
from .pointcept_color import RandomColorJitter, ChromaticJitter

def augment_rgb(batch, config):
    image=batch['front_rgb']
    assert image.dtype==torch.uint8 and image.ndim==5 and image.shape[1:]==(1,3,224,224)
    # The reference transforms operate on float32 [0,255], not quantized uint8.
    colors=image.detach().cpu().numpy().transpose(0,1,3,4,2).astype(np.float32).copy()
    jitter=RandomColorJitter(brightness=config['brightness'],contrast=config['contrast'],saturation=config['saturation'],hue=config['hue'],p=config['p'])
    noise=ChromaticJitter(p=config['color_noise_p'],std=config['color_noise_std'])
    for i in range(len(colors)):
        colors[i]=noise(jitter({'color':colors[i].reshape(-1,3)}))['color'].reshape(colors[i].shape)
    assert np.isfinite(colors).all() and colors.min()>=0 and colors.max()<=255
    out=dict(batch)
    out['front_rgb']=torch.from_numpy(colors.transpose(0,1,4,2,3).copy()).to(device=image.device)
    return out

def training_views(batch, config):
    assert config['recipe']=='pointcept_msc' and config['consistency_weight']==0 and not config['gamma']
    first=augment_rgb(batch,config)
    # Match the original augmentation-only recipe's two-view RNG advancement.
    second=augment_rgb(batch,config)
    del second
    return first
