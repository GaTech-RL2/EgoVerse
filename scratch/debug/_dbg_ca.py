import torch
from egomimic.models.hnet_nets.image_encoders import SimpleConv
from egomimic.models.hnet_nets.cond_encoders import SpatialCondEncoderModule
from egomimic.algo.hnet import FlatFusedPolicy

IMG=96; D=128; CK=8
def make_img(t):
    img=torch.zeros(2,3,IMG,IMG); x0=10+t; img[:,:,30:45,x0:x0+15]=1.0; return img
def obs(t): return {"front_img_1": make_img(t), "state_agent_obj": torch.zeros(2,2)+0.5}

torch.manual_seed(1)
sc_pool=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
sc_tok=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,return_tokens=True)
enc=SpatialCondEncoderModule(d_cond=D,
    obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
    img_encoders={"front_img_1":sc_pool}, cond_proj_widths=[128,128],
    spatial_img_encoders={"front_img_1":sc_tok}, compress_tokens=None)
pol=FlatFusedPolicy(action_dim=2,action_horizon=1024,d_model=D,d_cond=D,
    cond_encoder=enc,arch_layout="T4",num_heads=4,d_intermediate=512,
    action_head_cfg={"mode":"continuous","chunk_k":CK},cond_mode="cross_attn")
pol.eval()

# Probe a single cross-attn layer directly with the two cond token sets.
ca = None
for m in pol.backbone.modules():
    if m.__class__.__name__ == "CrossMultiHeadAttention":
        ca = m; break
print("found cross-attn:", ca is not None)
print("out_proj weight std:", float(ca.out_proj.weight.std()))

with torch.no_grad():
    spa = enc.encode(obs(0),1)["spatial_cond_tokens"][:,0]   # (B, M, d_cond)
    spb = enc.encode(obs(40),1)["spatial_cond_tokens"][:,0]
    print("cond token diff:", float((spa-spb).abs().mean()))
    xq = torch.randn(2, 2, D)  # dummy query tokens
    oa = ca(xq, spa); ob = ca(xq, spb)
    print("cross-attn OUTPUT diff for differing cond:", float((oa-ob).abs().mean()))

# Now check: does backbone forward see cond at all? Patch a hook to log cond.
seen = {}
def hook(mod, inp, out):
    seen["called"] = True
h = ca.register_forward_hook(hook)
with torch.no_grad():
    pol.chunk_forward_history(obs(0))
print("cross-attn layer CALLED during chunk_forward_history:", seen.get("called", False))
h.remove()
