import torch
torch.manual_seed(1)
from egomimic.models.hnet_nets.blocks import CrossMultiHeadAttention
ca = CrossMultiHeadAttention(d_model=128, d_cond=128, num_heads=4, causal=False)
ca.eval()
B,M,D=2,16,128
torch.manual_seed(5)
base = torch.randn(B,M,D)
# Simulate "T at position p" as a one-hot-ish hot token; translate it.
def hot(p):
    t = base.clone()*0.1
    t[:,p] += 3.0
    return t
xq = torch.randn(B,2,D)
with torch.no_grad():
    oa = ca(xq, hot(2)); ob = ca(xq, hot(11))
    print("NO pos emb: cross-attn out diff =", float((oa-ob).abs().mean()))
# Add a fixed positional embedding to the cond tokens.
pos = torch.randn(1,M,D)*0.5
with torch.no_grad():
    oa2 = ca(xq, hot(2)+pos); ob2 = ca(xq, hot(11)+pos)
    print("WITH pos emb: cross-attn out diff =", float((oa2-ob2).abs().mean()))
