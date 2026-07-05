"""Module-tree ACTIVE param count for per-embodiment H-Net chains.

Active = all shared modules + exactly ONE embodiment's branch of every
per-emb ModuleDict container (PerEmbodimentStage.sub_stages /
MultiEmbodimentCondEncoder.encoders / gmm_heads / any ModuleDict keyed by
embodiment names).

Counting is IDENTITY-DEDUPED: params are collected into an {id: numel} dict,
so the HNet chain wiring (which registers the whole downstream chain as an
``inner_stage`` child of every stage AND of every per-emb sub-stage) cannot
double-count or double-subtract anything. The old subtraction-based version
went negative on exactly these chains.
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


def _is_per_emb(child):
    return isinstance(child, torch.nn.ModuleDict) and EMB in child


def collect(mod, active_only, skip_chain, out=None):
    """{id(param): numel} over the tree.

    active_only: at any per-emb ModuleDict descend only into the EMB branch.
    skip_chain:  don't follow child edges named 'inner_stage' (used ONLY for
                 the per-stage breakdown, where each stage must be counted
                 without the downstream chain it aliases; totals never need
                 it because the id-dict dedupes).
    """
    if out is None:
        out = {}
    for _, p in mod.named_parameters(recurse=False):
        out[id(p)] = p.numel()
    for name, child in mod.named_children():
        if skip_chain and name == "inner_stage":
            continue
        if active_only and _is_per_emb(child):
            collect(child[EMB], active_only, skip_chain, out)
            continue
        collect(child, active_only, skip_chain, out)
    return out


def msum(ids):
    return sum(ids.values()) / M


total_ids = collect(outer, active_only=False, skip_chain=False)
active_ids = collect(outer, active_only=True, skip_chain=False)

print("=== component breakdown (own params, chain links excluded) ===")
hnet = outer.inner_stage
if outer.input_modules is not None:
    ids_a = collect(outer.input_modules, True, True)
    ids_t = collect(outer.input_modules, False, True)
    print(
        f"{'obs encoders (input_modules)':42s} active {msum(ids_a):8.2f}M   total {msum(ids_t):8.2f}M"
    )
for i, st in enumerate(hnet.stages):
    ids_a = collect(st, True, True)
    ids_t = collect(st, False, True)
    print(
        f"stage[{i}] {type(st).__name__:32s} active {msum(ids_a):8.2f}M   total {msum(ids_t):8.2f}M"
    )
gh = getattr(outer, "gmm_heads", None) or getattr(outer, "action_out", None)
if gh is not None:
    ids_a = collect(gh, True, True)
    ids_t = collect(gh, False, True)
    print(
        f"{'action head(s)':42s} active {msum(ids_a):8.2f}M   total {msum(ids_t):8.2f}M"
    )

print(f"\n{'TOTAL ACTIVE (1 emb)':42s} {msum(active_ids):8.2f}M")
print(f"{'TOTAL (all params)':42s} {msum(total_ids):8.2f}M")

# fine-grained chunker internals for the budget-swap design
print("\n=== chunker internals (per emb, per level) ===")
for i, st in enumerate(hnet.stages):
    sub = getattr(st, "sub_stages", None)
    if isinstance(sub, torch.nn.ModuleDict) and EMB in sub:
        ch = sub[EMB]
        for name, child in ch.named_children():
            if name == "inner_stage":
                continue
            print(f"  L{i}.{name:24s} {msum(collect(child, True, True)):7.3f}M")
        d = sum(p.numel() for _, p in ch.named_parameters(recurse=False))
        if d:
            print(f"  L{i}.<direct params>          {d/M:7.3f}M")
print("DONE")
