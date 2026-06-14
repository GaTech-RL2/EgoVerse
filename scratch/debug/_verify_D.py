import torch
torch.manual_seed(0)
from egomimic.models.hnet_nets.image_encoders import SimpleConv
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule, SpatialCondEncoderModule
from egomimic.algo.hnet import FlatFusedPolicy
from egomimic.models.hnet_nets.action_heads import action_head_raw

D=128; A=2; CK=8; IMG=96

def make_obs(t_shift, state_add=0.0):
    img=torch.zeros(2,3,IMG,IMG); x0=10+t_shift; img[:,:,30:45,x0:x0+15]=1.0
    return {"front_img_1": img, "state_agent_obj": torch.zeros(2,2)+0.5+state_add}

def spatial_enc():
    sp=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
    st=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,return_tokens=True)
    return SpatialCondEncoderModule(d_cond=D,
        obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
        img_encoders={"front_img_1":sp}, cond_proj_widths=[128,128],
        spatial_img_encoders={"front_img_1":st}, compress_tokens=None)

def pool_enc():
    sp=SimpleConv(3,[32,64,128],embed_dim=D,image_size=IMG,spatial=False)
    return CondEncoderModule(d_cond=D,
        obs_specs={"state_agent_obj":{"input_dim":2,"embed_dim":64,"widths":[128],"input_slice":[0,2]}},
        img_encoders={"front_img_1":sp}, cond_proj_widths=[128,128])

hc={"mode":"continuous","chunk_k":CK}

# ---- (b) cross_attn conditions on the image ----
torch.manual_seed(1)
e=spatial_enc()
pol=FlatFusedPolicy(action_dim=A,action_horizon=1024,d_model=D,d_cond=D,cond_encoder=e,
    arch_layout="T4",num_heads=4,d_intermediate=512,action_head_cfg=hc,cond_mode="cross_attn")
pol.eval()
nM=e.encode(make_obs(0),3)["spatial_cond_tokens"].shape[2]
hasX=any(m.__class__.__name__=="CrossMultiHeadAttention" for m in pol.modules())
print("[cross_attn] spatial tokens/frame =", nM, " backbone has cross-attn =", hasX)
with torch.no_grad():
    oa=pol.chunk_forward_history(make_obs(0))
    ob=pol.chunk_forward_history(make_obs(40))   # ONLY the T location differs
    oc=pol.chunk_forward_history(make_obs(0, state_add=1.0))  # only state differs
    d_img=float((oa-ob).abs().mean()); d_st=float((oa-oc).abs().mean())
print("[cross_attn] mean|pred(T@left)-pred(T@right)| =", round(d_img,6))
print("[cross_attn] mean|pred(state moved)|          =", round(d_st,6))
CROSS_IMG = d_img > 1e-4

# eval(generate) == train(chunk_forward_history)
with torch.no_grad():
    pol._act_chunk_k=CK
    g=pol.generate(make_obs(0), batch_size=2, device=torch.device("cpu"), T=CK)
    tf=pol.chunk_forward_history(make_obs(0))
    GEN_EQ=torch.allclose(g,tf,atol=1e-5)
print("[cross_attn] eval generate == train chunk_forward :", GEN_EQ)

# ---- (a) adaln byte-identical ----
torch.manual_seed(2)
ea=pool_enc()
pa=FlatFusedPolicy(action_dim=A,action_horizon=1024,d_model=D,d_cond=D,cond_encoder=ea,
    arch_layout="T4",num_heads=4,d_intermediate=512,action_head_cfg=hc,cond_mode="adaln")
pa.eval()
hasX_a=any(m.__class__.__name__=="CrossMultiHeadAttention" for m in pa.modules())
print("[adaln] backbone has cross-attn =", hasX_a, "(must be False)")
o=make_obs(0)
with torch.no_grad():
    # The byte-identical claim: adaln chunk_forward_history must equal the SAME
    # computation with the backbone called WITHOUT a cond arg (== pre-D code).
    any_v=next(iter(o.values())); N=any_v.shape[1] if any_v.dim()>=3 else 1
    c=pa._encode_cond(o,N); ct=pa.cond_in(c); bos=pa.bos.expand(c.shape[0],1,-1)
    x=torch.cat([ct,bos],dim=1)+pa.pos_emb[:,:N+1]
    xb_old=pa.backbone(x)                  # pre-D: no cond arg
    raw_old=action_head_raw(pa, xb_old[:,N:N+1])[:,0]
    new=pa.chunk_forward_history(o)
    BYTE=torch.equal(raw_old,new)
    # determinism
    DET=torch.equal(pa.chunk_forward_history(o), pa.chunk_forward_history(o))
print("[adaln] chunk_forward_history == pre-D (no-cond backbone):", BYTE, " det:", DET)

print("="*52)
print("RESULT cross_attn_conditions_on_image =", CROSS_IMG, "(d_img=%.6f)"%d_img)
print("RESULT cross_attn_train_eq_eval       =", GEN_EQ)
print("RESULT adaln_byte_identical           =", BYTE)
print("RESULT adaln_deterministic            =", DET)
print("RESULT adaln_no_crossattn             =", (not hasX_a))
ALL = CROSS_IMG and GEN_EQ and BYTE and DET and (not hasX_a)
print("RESULT ALL_PASS =", ALL)
