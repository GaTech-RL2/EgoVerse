"""Count inner-chunker tokens per episode for the 3 pair episodes per emb."""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, "/coc/flash7/paphiwetsa3/projects/EgoVerse2")

import torch, hydra
from hydra import initialize_config_dir, compose

CFG_DIR = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/egomimic/hydra_configs"
CKPT = "/coc/flash7/paphiwetsa3/projects/EgoVerse2/external_ckpts/cotrain_200ep_v2_resume_last.ckpt"
PAIR_C = "/coc/flash7/paphiwetsa3/datasets/pushT/pairs/circle"
PAIR_S = "/coc/flash7/paphiwetsa3/datasets/pushT/pairs/stick"

with initialize_config_dir(CFG_DIR, version_base=None):
    cfg = compose(
        "train_zarr_cartesian",
        overrides=[
            "data=tsimulation_cotrain_pairs",
            "model=hnet_pushshapes_cotrain",
            f"data.train_datasets.pushshapes_sim.resolver.folder_path={PAIR_C}",
            f"data.train_datasets.pushshapes_sim_stick.resolver.folder_path={PAIR_S}",
            f"data.valid_datasets.pushshapes_sim.resolver.folder_path={PAIR_C}",
            f"data.valid_datasets.pushshapes_sim_stick.resolver.folder_path={PAIR_S}",
        ],
    )

dm = hydra.utils.instantiate(cfg.data)
dm.setup("validate")
model = hydra.utils.instantiate(cfg.model)
model.robomimic_model.set_norm_stats(dm.norm_stats)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
algo = model.robomimic_model
policy = algo.nets["policy"]

for emb_name, loader_key in [("pushshapes_sim", "pushshapes_sim"), ("pushshapes_sim_stick", "pushshapes_sim_stick")]:
    loader = dm.val_dataloader()[loader_key]
    batch = next(iter(loader))
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
    emb_id = int(batch["embodiment"][0]) if "embodiment" in batch else None
    _packed = batch.get("_packed", False)
    cu = batch["cu_seqlens"]
    print(f"=== {emb_name} (emb_id={emb_id}) ===")
    print(f"  episode lengths (cu diffs): {(cu[1:] - cu[:-1]).tolist()}")

    obs = algo._build_obs(batch, emb_id)
    ac_key = algo.resolved_ac_keys[emb_id]
    actions = batch[ac_key]
    max_seqlen = int(batch["max_seq_len"])
    with torch.no_grad():
        _, aux = policy.forward_packed(actions, obs, cu, max_seqlen, embodiment_id=emb_name)

    print(f"  num chunker stages w/ bpred: {sum(1 for e in aux if isinstance(e, dict) and 'bpred' in e)}")
    for i, entry in enumerate(aux):
        if isinstance(entry, dict) and "bpred" in entry:
            bp = entry["bpred"]
            mask = bp.boundary_mask.detach().cpu().to(torch.bool)
            total_tokens = int(mask.sum().item())
            print(f"  stage[{i}] bmask shape={list(mask.shape)} total boundaries={total_tokens}")
            cs = mask.cumsum(0)
            for b in range(len(cu) - 1):
                s, e = int(cu[b]), int(cu[b+1])
                ep_tokens = int(mask[s:e].sum().item())
                print(f"    ep{b} (frames {s}..{e}, len={e-s}): {ep_tokens} chunks")
