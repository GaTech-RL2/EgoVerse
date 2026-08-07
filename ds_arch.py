import collections
import sys

import hydra.utils as hu
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
os_ = hu.instantiate(cfg.robomimic_model.outer_stage)
g = collections.OrderedDict()
for n, p in os_.named_parameters():
    g[n.split(".")[0]] = g.get(n.split(".")[0], 0) + p.numel()
total = sum(g.values())
print(f"=== DS200M outer_stage TOTAL = {total/1e6:.2f}M ===")
for seg, n in sorted(g.items(), key=lambda x: -x[1]):
    print(f"  {n/1e6:8.2f}M  {100*n/total:5.1f}%  {seg}")
trunk = g.get("inner_stage", 0)
agn = g.get("agnostic_input", 0)
cond = g.get("cond_encoder", 0)
spec = g.get("input_modules", 0)
head = g.get("action_out", 0)
active = trunk + agn + spec / 2 + head / 2 + cond
print(
    f"--- TOTAL {total/1e6:.1f}M | ACTIVE (trunk+agnostic+1 specific enc+1 head) {active/1e6:.1f}M ---"
)
print(
    f"--- matched part excl encoders = trunk {trunk/1e6:.1f}M + 1 head {head/2/1e6:.2f}M = {(trunk+head/2)/1e6:.1f}M  (baseline core ~190M) ---"
)
