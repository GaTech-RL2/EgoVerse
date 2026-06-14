"""Static PCA plot of inner-chunker tokens, colored by embodiment.

Iterates N training episodes per emb, captures the inner ComputeStage's
output tokens, fits a single shared PCA, scatters with one color per emb.
Saves a PNG to /coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/
so we can scp it locally.
"""
import os, sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf

CFG_DIR = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs"
CKPT = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/cotrain_200ep_v2_resume_last.ckpt"
NORM_PATH = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/cotrain_200ep_norm_stats.json"
CIRCLE = "/coc/flash7/paphiwetsa3/datasets/pushT/circle_750/circle"
STICK = "/coc/flash7/paphiwetsa3/datasets/pushT/stick_312/stick"
OUT_PNG = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/cross_emb_pca.png"
N_EPS_PER_EMB = 100

with initialize_config_dir(CFG_DIR, version_base=None):
    cfg = compose(
        "train_zarr_cartesian",
        overrides=[
            "mode=eval",
            "data=tsimulation_cotrain_pairs",
            "model=hnet_pushshapes_cotrain",
            "evaluator=eval_hnet_pairs",
            "trainer=debug",
            "logger=csv_wandb",
            "trainer.limit_val_batches=1",
            "trainer.profiler=null",
            f"data.train_datasets.pushshapes_sim.resolver.folder_path={CIRCLE}",
            f"data.train_datasets.pushshapes_sim_stick.resolver.folder_path={STICK}",
            f"data.valid_datasets.pushshapes_sim.resolver.folder_path={CIRCLE}",
            f"data.valid_datasets.pushshapes_sim_stick.resolver.folder_path={STICK}",
            f"norm_stats.precomputed_norm_path={NORM_PATH}",
            f"ckpt_path={CKPT}",
        ],
    )

# Build everything trainHydra does, minus running validate().
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

dm = hydra.utils.instantiate(cfg.data)
dm.setup("fit")

norm_stats = MultiDataset(
    name="train",
    norm_mode=OmegaConf.select(cfg, "norm_stats.norm_mode", default="quantile"),
)
norm_stats.populate_from_datasets(dm.train_datasets)
for d in dm.train_datasets.values():
    norm_stats.infer_shapes_from_batch(d[0])
norm_stats.load_precomputed_norm(NORM_PATH)
for ds in dm.train_datasets.values():
    ds.set_norm_stats_from(norm_stats)
for ds in dm.valid_datasets.values():
    ds.set_norm_stats_from(norm_stats)

OmegaConf.set_struct(cfg.model, False)
cfg.model.robomimic_model.norm_stats = None
model = hydra.utils.instantiate(cfg.model)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
state = sd.get("state_dict", sd)
model.load_state_dict(state, strict=False)
model.norm_stats = norm_stats
algo = model.model
algo.norm_stats = norm_stats
algo.cuda()
algo.eval()

# Find innermost ComputeStage's main_network for the hook.
policy = algo.nets["policy"]
hnet = policy.hnet
inner_module = None
for stage in reversed(hnet.stages):
    mn = getattr(stage, "main_network", None)
    if mn is not None:
        inner_module = mn
        break
assert inner_module is not None, "couldn't find inner main_network"

captured = []


def _hook(_m, _ins, output):
    t = output.detach().float().cpu()
    if t.dim() == 3:
        t = t.reshape(-1, t.shape[-1])
    captured.append(t)


tokens_by_emb = {}
combined = dm.train_dataloader()
loaders = combined.iterables if hasattr(combined, "iterables") else combined
device = next(algo.nets["policy"].parameters()).device

for emb_name, loader in loaders.items():
    print(f"[CROSS_PCA] iterating {emb_name}", flush=True)
    eps_done = 0
    cap_for_emb = []
    for raw_batch in loader:
        processed = algo.process_batch_for_training({emb_name: raw_batch})
        for emb_id, eb in processed.items():
            if not eb.get("_packed", False):
                continue
            captured.clear()
            handle = inner_module.register_forward_hook(_hook)
            try:
                ac_key = algo.resolved_ac_keys[emb_id]
                obs = algo._build_obs(eb, emb_id)
                actions = eb[ac_key]
                cu = eb["cu_seqlens"]
                max_seqlen = int(eb["max_seq_len"])
                with torch.no_grad():
                    policy.forward_packed(
                        actions, obs, cu, max_seqlen, embodiment_id=emb_name
                    )
            finally:
                handle.remove()
            if captured:
                cap_for_emb.append(captured[-1].numpy())
                eps_done += int(eb["seq_lens"].shape[0])
        if eps_done >= N_EPS_PER_EMB:
            break
    if cap_for_emb:
        tokens_by_emb[emb_name] = np.concatenate(cap_for_emb, axis=0)
        print(
            f"[CROSS_PCA]   collected {tokens_by_emb[emb_name].shape[0]} tokens from {eps_done} eps",
            flush=True,
        )

all_tokens = []
labels = []
for emb_name, toks in tokens_by_emb.items():
    all_tokens.append(toks)
    labels.extend([emb_name] * toks.shape[0])
X = np.concatenate(all_tokens, axis=0)
labels = np.array(labels)
print(f"[CROSS_PCA] total tokens for PCA: {X.shape}", flush=True)

# Center and PCA via SVD
X_c = X - X.mean(0, keepdims=True)
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
proj = X_c @ Vt[:2].T  # (N, 2)
print(f"[CROSS_PCA] explained variance ratio first 2: {(S[:2]**2 / (S**2).sum()).round(3)}")

fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
colors = {"pushshapes_sim": "#1f77b4", "pushshapes_sim_stick": "#d62728"}
for emb_name in tokens_by_emb.keys():
    mask = labels == emb_name
    ax.scatter(
        proj[mask, 0],
        proj[mask, 1],
        s=18,
        c=colors.get(emb_name, "grey"),
        alpha=0.55,
        edgecolors="none",
        label=f"{emb_name} (n={mask.sum()})",
    )
ax.legend(loc="best", fontsize=10)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(
    f"Cross-emb PCA of H-Net inner tokens\n"
    f"{N_EPS_PER_EMB} train episodes per emb"
)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"[CROSS_PCA] saved {OUT_PNG}", flush=True)
