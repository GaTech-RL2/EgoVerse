import torch
torch.manual_seed(2)
from egomimic.models.hnet_nets.image_encoders import SimpleConv
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.algo.hnet import FlatFusedPolicy
from egomimic.models.hnet_nets.action_heads import action_head_raw
IMG=96; D=128; CK=8
def make_img(t):
    img=torch.zeros(2,3,IMG,IMG); x0=10+t; img[:,:,30:45,x0:x0+15]=1.0; return img
def obs(t): return {"front_img_1": make_img(t), "state_agent_obj": torch.zeros(2,2)+0.5}
sc=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
enc=CondEncoderModule(d_cond=D,
    obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
    img_encoders={"front_img_1":sc}, cond_proj_widths=[128,128])
pol=FlatFusedPolicy(action_dim=2,action_horizon=1024,d_model=D,d_cond=D,
    cond_encoder=enc,arch_layout="T4",num_heads=4,d_intermediate=512,
    action_head_cfg={"mode":"continuous","chunk_k":CK},cond_mode="adaln")
pol.eval()
o=obs(0)
with torch.no_grad():
    new = pol.chunk_forward_history(o)
    # hand-rolled
    c=pol._encode_cond(o,1); ct=pol.cond_in(c); bos=pol.bos.expand(c.shape[0],1,-1)
    x=torch.cat([ct,bos],dim=1)+pol.pos_emb[:,:2]
    xb=pol.backbone(x)
    raw=action_head_raw(pol, xb[:,1:2])  # shape?
    print("raw shape", tuple(raw.shape), "new shape", tuple(new.shape))
    # chunk_forward_history reads x[:, N:N+1] = x[:,1:2] then action_head_raw -> raw[:,0]
    raw0 = raw[:,0]
    print("raw0 shape", tuple(raw0.shape))
    print("equal raw0 vs new:", torch.equal(raw0, new), " maxdiff", float((raw0-new).abs().max()))
    # also compare backbone(x) vs backbone(x, cond=None)
    xb2=pol.backbone(x, cond=None)
    print("backbone(x) vs backbone(x,cond=None) equal:", torch.equal(xb, xb2))
