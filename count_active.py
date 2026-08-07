"""Module-tree ACTIVE param count for the chunkchunk_trunk H-Net.

Active = all shared modules + exactly ONE embodiment's branch of every
per-emb container (PerEmbodimentStage.sub_stages / MultiEmbodimentCondEncoder.encoders
/ gmm_heads ModuleDict). Walks the tree by IDENTITY, not names.
"""

import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

CFG = sys.argv[1]
EMB = "pushshapes_sim"  # the branch counted as active

cfg = OmegaConf.load(CFG)
outer = instantiate(cfg.robomimic_model.outer_stage)

M = 1e6


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())


def active_params(mod):
    """Recurse; at any per-emb dict container keep only EMB's branch."""
    kids = dict(mod.named_children())
    # per-emb containers in this codebase
    for attr in ("sub_stages", "encoders"):
        sub = getattr(mod, attr, None)
        if isinstance(sub, torch.nn.ModuleDict) and EMB in sub:
            others = {k: v for k, v in sub.items() if k != EMB}
            total = n_params(mod)
            dead = sum(n_params(v) for v in others.values())
            # recurse into the kept branch for nested per-emb containers
            kept = sub[EMB]
            kept_total = n_params(kept)
            kept_active = active_params(kept)
            return total - dead - (kept_total - kept_active)
    tot = 0
    counted = set()
    for name, k in kids.items():
        tot += active_params(k)
        counted.add(name)
    # direct parameters on this module
    tot += sum(p.numel() for n, p in mod.named_parameters(recurse=False))
    return tot


def report(label, mod):
    if mod is None:
        return 0
    a = active_params(mod)
    t = n_params(mod)
    print(f"{label:42s} active {a/M:8.2f}M   total {t/M:8.2f}M")
    return a


print("=== component breakdown ===")
tot = 0
tot += report("obs encoders (input_modules)", outer.input_modules)
hnet = outer.inner_stage
for i, st in enumerate(hnet.stages):
    # count each stage's OWN params (exclude the chained inner_stage)
    class Own(torch.nn.Module):
        def __init__(self, s):
            super().__init__()
            for n, c in s.named_children():
                if n != "inner_stage":
                    self.add_module(n, c)
            self._direct = [p for _, p in s.named_parameters(recurse=False)]

    own = Own(st)
    tot += report(f"stage[{i}] {type(st).__name__}", own)
# heads
gh = getattr(outer, "gmm_heads", None) or getattr(outer, "action_out", None)
if isinstance(gh, torch.nn.ModuleDict):
    a = n_params(gh[EMB]) if EMB in gh else 0
    print(f"{'gmm head (1 emb)':42s} active {a/M:8.2f}M   total {n_params(gh)/M:8.2f}M")
    tot += a
else:
    tot += report("action head", gh)
print(f"{'TOTAL ACTIVE (1 emb)':42s} {tot/M:8.2f}M")
print(f"{'TOTAL (all params)':42s} {n_params(outer)/M:8.2f}M")

# fine-grained chunker internals for the budget-swap design
print("\n=== chunker internals (per emb, level 0 + 1) ===")
for i, st in enumerate(hnet.stages):
    sub = getattr(st, "sub_stages", None)
    if isinstance(sub, torch.nn.ModuleDict) and EMB in sub:
        ch = sub[EMB]
        for name, child in ch.named_children():
            if name == "inner_stage":
                continue
            print(f"  L{i}.{name:24s} {n_params(child)/M:7.3f}M")
        d = sum(p.numel() for _, p in ch.named_parameters(recurse=False))
        if d:
            print(f"  L{i}.<direct params>          {d/M:7.3f}M")
print("DONE")
