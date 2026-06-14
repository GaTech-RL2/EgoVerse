import torch
from egomimic.models.hnet_nets.image_encoders import SimpleConv
from egomimic.models.hnet_nets.cond_encoders import SpatialCondEncoderModule
from egomimic.algo.hnet import FlatFusedPolicy

IMG=96; D=128; CK=8
def make_img(t):
    img=torch.zeros(2,3,IMG,IMG); x0=10+t; img[:,:,30:45,x0:x0+15]=1.0; return img
def obs(t): return {"front_img_1": make_img(t), "state_agent_obj": torch.zeros(2,2)+0.5}

def run(compress):
    torch.manual_seed(1)
    sc_pool=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
    sc_tok=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,return_tokens=True)
    enc=SpatialCondEncoderModule(d_cond=D,
        obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
        img_encoders={"front_img_1":sc_pool}, cond_proj_widths=[128,128],
        spatial_img_encoders={"front_img_1":sc_tok},
        compress_tokens=compress, compress_heads=4)
    pol=FlatFusedPolicy(action_dim=2,action_horizon=1024,d_model=D,d_cond=D,
        cond_encoder=enc,arch_layout="T4",num_heads=4,d_intermediate=512,
        action_head_cfg={"mode":"continuous","chunk_k":CK},cond_mode="cross_attn")
    pol.eval()
    with torch.no_grad():
        a=pol.chunk_forward_history(obs(0)); b=pol.chunk_forward_history(obs(40))
        n=enc.encode(obs(0),1)["spatial_cond_tokens"].shape[2]
        print(f"compress={compress}  n_spatial={n}  mean|pred_a-pred_b|={float((a-b).abs().mean()):.6f} max={float((a-b).abs().max()):.6f}")

run(None)   # raw tokens -> backbone cross-attn
run(16)     # HPT-style compressed (untrained averages)
