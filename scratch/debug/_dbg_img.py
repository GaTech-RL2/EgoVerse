import torch
torch.manual_seed(1)
from egomimic.models.hnet_nets.image_encoders import SimpleConv

IMG=96
def make_img(t_shift):
    img = torch.zeros(2,3,IMG,IMG)
    x0=10+t_shift
    img[:,:,30:45,x0:x0+15]=1.0
    return img

# pooled encoder
sc_pool = SimpleConv(in_channels=3, channels=[32,64,128], embed_dim=128, image_size=IMG, spatial=False)
sc_pool.eval()
with torch.no_grad():
    pa = sc_pool(make_img(0))
    pb = sc_pool(make_img(40))
    print("pooled feat diff mean|a-b| =", float((pa-pb).abs().mean()))

# token encoder
sc_tok = SimpleConv(in_channels=3, channels=[32,64,128], embed_dim=128, image_size=IMG, return_tokens=True)
sc_tok.eval()
with torch.no_grad():
    ta = sc_tok(make_img(0))
    tb = sc_tok(make_img(40))
    print("token shape", tuple(ta.shape), "n_tokens", sc_tok.n_tokens)
    print("token feat diff mean|a-b| =", float((ta-tb).abs().mean()))

# Now the full spatial encoder + compressor
from egomimic.models.hnet_nets.cond_encoders import SpatialCondEncoderModule
enc = SpatialCondEncoderModule(
    d_cond=128,
    obs_specs={"state_agent_obj": {"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
    img_encoders={"front_img_1": sc_pool},
    cond_proj_widths=[128,128],
    spatial_img_encoders={"front_img_1": sc_tok},
    compress_tokens=16, compress_heads=4,
)
enc.eval()
def obs(t):
    return {"front_img_1": make_img(t), "state_agent_obj": torch.zeros(2,2)+0.5}
with torch.no_grad():
    da = enc.encode(obs(0),1)
    db = enc.encode(obs(40),1)
    print("fused_cond diff =", float((da["fused_cond"]-db["fused_cond"]).abs().mean()))
    print("spatial_cond_tokens diff =", float((da["spatial_cond_tokens"]-db["spatial_cond_tokens"]).abs().mean()))
