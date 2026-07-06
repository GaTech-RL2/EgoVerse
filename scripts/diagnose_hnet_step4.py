"""Use REAL val batch obs (varying across time) and check whether a_0 varies per episode."""
import torch, numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/checkpoints/last.ckpt"
CFG = "logs/hnet_variants/hnet_iterA_obsnoise_pace_l40s_2026-05-19_05-44-22_2026-05-19_05-44-34/.hydra/config.yaml"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(CFG)
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
    norm_stats = MultiDataset.from_state(ckpt["hyper_parameters"]["norm_stats_state"])
    algo = instantiate(cfg.model.robomimic_model, norm_stats=norm_stats)
    new_sd = {k[len("nets."):] if k.startswith("nets.") else k: v for k, v in ckpt["state_dict"].items()}
    algo.nets.load_state_dict(new_sd, strict=False)
    algo.nets = algo.nets.to(device); algo.nets.eval()
    policy = algo.nets["policy"]
    ac_key, emb_id = "actions", 15

    # Pull a real val batch (multi-episode)
    dm = instantiate(cfg.data)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    if isinstance(batch, tuple): batch = batch[0]
    processed = algo.process_batch_for_training(batch)
    emb_first = list(processed.keys())[0]
    b = processed[emb_first]
    print(f"  cu_seqlens: {b['cu_seqlens'].tolist()[:6]}")
    cu = b["cu_seqlens"]
    n_eps = len(cu) - 1
    print(f"  n_eps in val batch: {n_eps}")

    # Run forward_packed on the full batch
    actions = b["actions"]
    obs = {k: b[k] for k in ("state_agent_obj", "front_img_1") if k in b}
    max_seqlen = int(b["max_seq_len"])
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred, _ = policy.forward_packed(actions, obs, cu, max_seqlen)
    # pred: (T_total, action_dim). a_0 of each episode is at cu[i]
    print(f"\n  pred shape: {pred.shape}")
    print(f"\n  a_0 of each episode (forward_packed, REAL obs):")
    for i in range(min(n_eps, 8)):
        start = int(cu[i].item())
        a0 = pred[start].detach().float().cpu().numpy()
        a0_un = norm_stats.unnormalize({ac_key: pred[start]}, emb_id)[ac_key].detach().float().cpu().numpy()
        # what's the obs at this frame?
        s = b["state_agent_obj"][start].detach().float().cpu().numpy()
        print(f"    ep[{i}] state[0]={s.round(3).tolist()}  a0_norm={a0.round(4).tolist()}  a0_unnorm={a0_un.round(2).tolist()}")

if __name__ == "__main__":
    main()
