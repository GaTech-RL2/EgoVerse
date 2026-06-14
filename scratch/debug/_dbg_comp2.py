import torch
from egomimic.models.hnet_nets.image_encoders import SimpleConv
from egomimic.models.hnet_nets.cond_encoders import SpatialCondEncoderModule
from egomimic.algo.hnet import FlatFusedPolicy
IMG=96; D=128; CK=8
def make_obs(t):
    img=torch.zeros(2,3,IMG,IMG); x0=10+t; img[:,:,30:45,x0:x0+15]=1.0
    return {"front_img_1":img,"state_agent_obj":torch.zeros(2,2)+0.5}
def run(compress, seed=1):
    torch.manual_seed(seed)
    sp=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
    st=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,return_tokens=True)
    e=SpatialCondEncoderModule(d_cond=D,
        obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
        img_encoders={"front_img_1":sp}, cond_proj_widths=[128,128],
        spatial_img_encoders={"front_img_1":st}, compress_tokens=compress, compress_heads=4)
    pol=FlatFusedPolicy(action_dim=2,action_horizon=1024,d_model=D,d_cond=D,cond_encoder=e,
        arch_layout="T4",num_heads=4,d_intermediate=512,
        action_head_cfg={"mode":"continuous","chunk_k":CK},cond_mode="cross_attn")
    pol.eval()
    with torch.no_grad():
        a=pol.chunk_forward_history(make_obs(0)); b=pol.chunk_forward_history(make_obs(40))
        # also probe the spatial tokens diff post-encoder (with pos emb)
        ta=e.encode(make_obs(0),3)["spatial_cond_tokens"]; tb=e.encode(make_obs(40),3)["spatial_cond_tokens"]
    print(f"compress={compress}: pred d_img={float((a-b).abs().mean()):.6f}  spatial_tok d={float((ta-tb).abs().mean()):.6f}")
run(None)
run(16)
