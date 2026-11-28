"""RGB-only visual input; shared HPT/action+DINO objective with the Cart B recipe."""
import torch
from torch import nn
import timm
from safetensors.torch import load_file
from .policy import CampaignPolicy
from .objectives import flow_loss

class FrozenRGBEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.net = timm.create_model('vit_base_patch16_dinov3', pretrained=False, num_classes=0, img_size=224)
        self.net.load_state_dict(load_file(checkpoint), strict=True)
        self.net.requires_grad_(False)
        self.net.eval()
        self.proj = nn.Linear(768, 256)
        self.register_buffer('mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, images):
        assert images.dtype in (torch.uint8, torch.float32) and images.shape[1:] == (1,3,224,224)
        x = (images[:,0].float()/255-self.mean)/self.std
        with torch.no_grad():
            tokens = self.net.forward_features(x)[:, self.net.num_prefix_tokens:]
        assert tokens.shape[1:] == (196,768)
        return self.proj(tokens).unsqueeze(1)

class RGBPolicy(CampaignPolicy):
    def __init__(self, reference, arm, normalizer, robot):
        assert arm == 'B'
        super().__init__(reference, arm, normalizer, robot)
        assert self.policy.shared_keys == ['front_rgb']

    def forward(self, batch):
        if self.training:
            from .msc_rgb import training_views
            batch=training_views(batch,self.photometric_config)
            self.msc_training_calls=getattr(self,'msc_training_calls',0)+1
        return self._forward_once(batch)

    def _forward_once(self, batch):
        self.policy.device = next(self.parameters()).device
        self.policy.train_mode = self.training
        assert not any(k.startswith('front_pcd') for k in batch)
        data = {k:v for k,v in batch.items() if k == 'front_rgb' or k.startswith('state_')}
        features,_ = self.policy.forward_features('rby1', data)
        action,_,_ = flow_loss(self.policy.heads['rby1'],features,batch['action'],batch['action_valid'])
        keep = batch['dino_valid'].bool()
        aux = action*0
        if keep.any():
            aux = self.policy.heads['rby1_dino_wm_future'].compute_loss(features[keep], {'action':batch['dino'][keep]})
        return action+aux, {'action':action.detach(),'aux':aux.detach()}
